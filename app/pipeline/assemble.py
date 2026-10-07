"""Assemble synthesized WAV chunks into chaptered M4B via ffmpeg.

  * Chapters are emitted only when the book actually has structure (>1
    chapter). Each chapter's start time is computed from chunk durations read
    straight from the WAV headers — no ffprobe needed.
  * The default produces one M4B for the whole book. An explicit
    `max_part_mib` retains the legacy split behavior for callers that need it.
  * Audio is encoded once per part (parts are disjoint), AAC at the configured
    bitrate, +faststart for smooth streaming/seeking.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..files import replace_and_fsync
from ..log import get_logger
from .synth import Chunk

log = get_logger("assemble")


class AssembleError(Exception):
    """M4B assembly failure (shown to the user, Russian message)."""


class AssembleCancelled(AssembleError):
    """Internal signal: the user cancelled M4B assembly."""


@dataclass
class _ChunkInfo:
    chunk: Chunk
    path: Path
    duration: float  # seconds


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.getnframes()
            if (
                channels != 1
                or sample_width != 2
                or w.getcomptype() != "NONE"
                or rate <= 0
                or frames <= 0
            ):
                raise ValueError("ожидался моно PCM WAV с ненулевой длительностью")
            pcm = w.readframes(frames)
            if len(pcm) != frames * channels * sample_width or w.readframes(1):
                raise ValueError("PCM-данные WAV обрезаны или повреждены")
            return frames / rate
    except Exception as e:
        raise AssembleError(f"Повреждённый аудио-фрагмент {path.name}: {e}") from e


def _escape_meta(value: str) -> str:
    # ffmetadata escaping: = ; # \ and newlines.
    for ch in ("\\", "=", ";", "#"):
        value = value.replace(ch, "\\" + ch)
    # ffmetadata also ends a line at \r; collapse every line break.
    return " ".join(value.splitlines()).strip()


def _has_structure(plan: list[Chunk]) -> bool:
    return len({c.chapter_index for c in plan}) > 1


def _chapter_titles(plan: list[Chunk]) -> dict[int, str]:
    """Resolve titles once from the complete plan, before parts split it."""
    titles: dict[int, str] = {}
    for chunk in plan:
        if chunk.chapter_index not in titles:
            titles[chunk.chapter_index] = (
                chunk.chapter_title or f"Глава {len(titles) + 1}"
            )
    return titles


def _build_ffmeta(
    infos: list[_ChunkInfo],
    title: str | None,
    author: str | None,
    with_chapters: bool,
    part_no: int | None,
    part_total: int | None,
    chapter_titles: dict[int, str] | None = None,
) -> str:
    lines = [";FFMETADATA1"]
    disp_title = title or "Аудиокнига"
    if part_total and part_total > 1:
        disp_title = f"{disp_title} (часть {part_no})"
    lines.append(f"title={_escape_meta(disp_title)}")
    if author:
        lines.append(f"artist={_escape_meta(author)}")
        lines.append(f"album_artist={_escape_meta(author)}")
    lines.append(f"album={_escape_meta(title or 'Аудиокнига')}")
    lines.append("genre=Audiobook")
    lines.append("media_type=2")  # iTunes: audiobook

    if with_chapters:
        # Group consecutive chunks by chapter_index into chapter spans.
        t_ms = 0
        spans: list[tuple[int, int, str]] = []  # (start_ms, end_ms, title)
        cur_chapter = None
        cur_start = 0
        cur_title = ""
        for info in infos:
            ci = info.chunk.chapter_index
            if ci != cur_chapter:
                if cur_chapter is not None:
                    spans.append((cur_start, t_ms, cur_title))
                cur_chapter = ci
                cur_start = t_ms
                cur_title = (chapter_titles or {}).get(
                    ci, info.chunk.chapter_title or f"Глава {len(spans) + 1}"
                )
            t_ms += int(round(info.duration * 1000))
        if cur_chapter is not None:
            spans.append((cur_start, t_ms, cur_title))

        for start_ms, end_ms, ctitle in spans:
            if end_ms <= start_ms:
                end_ms = start_ms + 1
            lines.append("")
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={start_ms}")
            lines.append(f"END={end_ms}")
            lines.append(f"title={_escape_meta(ctitle)}")
    return "\n".join(lines) + "\n"


def _partition(
    infos: list[_ChunkInfo], bitrate_bps: int, max_part_mib: int
) -> list[list[_ChunkInfo]]:
    """Greedily pack chunks into parts that each stay under the byte limit.
    Splits only at chunk boundaries; chapters may span a part edge."""
    max_bytes = max_part_mib * 1024 * 1024 * 0.97  # container headroom
    bytes_per_sec = bitrate_bps / 8
    parts: list[list[_ChunkInfo]] = []
    cur: list[_ChunkInfo] = []
    cur_bytes = 0.0
    for info in infos:
        chunk_bytes = info.duration * bytes_per_sec
        if cur and cur_bytes + chunk_bytes > max_bytes:
            parts.append(cur)
            cur, cur_bytes = [], 0.0
        cur.append(info)
        cur_bytes += chunk_bytes
    if cur:
        parts.append(cur)
    return parts or [[]]


def _run_ffmpeg(
    wav_paths: list[Path],
    ffmeta_path: Path,
    out_path: Path,
    bitrate: str,
    sample_rate: int,
    with_chapters: bool,
    work_dir: Path,
    tempo: float = 1.0,
    cancel_check: Callable[[], bool] | None = None,
    cover_path: Path | None = None,
) -> None:
    list_path = work_dir / (out_path.stem + ".concat.txt")
    # concat demuxer needs single-quoted absolute paths with quotes escaped.
    lines = []
    for p in wav_paths:
        ap = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{ap}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-i", str(ffmeta_path),
    ]
    if cover_path is not None:
        cmd += ["-i", str(cover_path)]
    cmd += ["-map", "0:a", "-map_metadata", "1"]
    if cover_path is not None:
        cmd += ["-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    cmd += ["-map_chapters", "1" if with_chapters else "-1"]
    if abs(tempo - 1.0) > 1e-3:
        # Pitch-preserving speed change baked into the output.
        cmd += ["-filter:a", f"atempo={tempo:.4f}"]
    cmd += [
        "-c:a", "aac", "-b:a", bitrate, "-ar", str(sample_rate), "-ac", "1",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(out_path),
    ]
    log.info("ffmpeg → %s (%d фрагментов)", out_path.name, len(wav_paths))
    proc: subprocess.Popen[str] | None = None
    try:
        if cancel_check and cancel_check():
            raise AssembleCancelled()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:
            raise AssembleError(f"Не удалось запустить ffmpeg: {e}") from e

        while True:
            if cancel_check and cancel_check():
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise AssembleCancelled()
            try:
                _, stderr = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                # communicate() drains stderr/stdout while waiting, unlike a
                # poll loop with PIPEs, so ffmpeg cannot block on a full pipe.
                continue
        if proc.returncode != 0:
            raise AssembleError(
                f"ffmpeg завершился с ошибкой при сборке {out_path.name}: "
                f"{(stderr or '').strip()[-500:]}"
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise AssembleError(f"ffmpeg не создал файл {out_path.name}")
    except Exception:
        out_path.unlink(missing_ok=True)
        raise
    finally:
        list_path.unlink(missing_ok=True)


def _cover_muxes(cover_path: Path, work_dir: Path) -> bool:
    """Mux the cover with a moment of silence: a sub-second check, so a bad
    image never costs a second multi-hour encode of the whole book."""
    probe = work_dir / "cover-probe.m4b"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
        "-i", str(cover_path),
        "-map", "0:a", "-map", "1:v:0", "-c:v", "copy",
        "-disposition:v:0", "attached_pic", "-c:a", "aac",
        "-t", "0.1",  # output option: the silence source is endless
        "-f", "mp4", str(probe),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0 and probe.is_file() and probe.stat().st_size > 0
        if not ok:
            log.warning("Обложка не встраивается в M4B: %s", (result.stderr or "").strip()[-300:])
        return ok
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Не удалось проверить обложку: %s", e)
        return False
    finally:
        probe.unlink(missing_ok=True)


def assemble(
    plan: list[Chunk],
    wav_paths: list[Path],
    output_dir: Path,
    base_name: str,
    *,
    title: str | None,
    author: str | None,
    bitrate: str = "64k",
    sample_rate: int = 48000,
    max_part_mib: int | None = None,
    tempo: float = 1.0,
    work_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    cover_path: Path | None = None,
) -> list[Path]:
    """Produce one or more M4B files. Returns ordered output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    _validate_assembly_args(bitrate, sample_rate, max_part_mib, tempo)
    if cancel_check and cancel_check():
        raise AssembleCancelled()

    if len(plan) != len(wav_paths):
        raise AssembleError(
            f"Рассогласование плана ({len(plan)}) и аудио ({len(wav_paths)})"
        )

    # Durations here describe the OUTPUT: with a tempo change the audio is
    # shorter/longer than the source WAVs, and both the chapter marks and the
    # size-based partitioning must reflect that.
    infos = [
        _ChunkInfo(chunk=c, path=p, duration=_wav_duration(p) / tempo)
        for c, p in zip(plan, wav_paths)
    ]
    total_dur = sum(i.duration for i in infos)
    if total_dur < 0.5:
        raise AssembleError("Суммарная длительность аудио близка к нулю")

    bitrate_bps = _parse_bitrate(bitrate)
    with_chapters = _has_structure(plan)
    chapter_titles = _chapter_titles(plan)
    parts = [infos] if max_part_mib is None else _partition(
        infos, bitrate_bps, max_part_mib
    )
    part_total = len(parts)

    outputs: list[Path] = []
    created: list[Path] = []
    try:
        for pi, part in enumerate(parts, start=1):
            if cancel_check and cancel_check():
                raise AssembleCancelled()
            out_path = _output_path(output_dir, base_name, pi, part_total)
            existed = out_path.exists()
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")

            ffmeta = _build_ffmeta(
                part, title, author, with_chapters,
                part_no=pi, part_total=part_total, chapter_titles=chapter_titles,
            )
            ffmeta_path = work_dir / f"part_{pi:02d}.ffmeta.txt"
            ffmeta_path.write_text(ffmeta, encoding="utf-8")
            try:
                if cover_path is not None and not _cover_muxes(cover_path, work_dir):
                    cover_path = None  # optional artwork never fails a book
                _run_ffmpeg(
                    [i.path for i in part], ffmeta_path, tmp_path,
                    bitrate, sample_rate, with_chapters, work_dir, tempo, cancel_check,
                    cover_path=cover_path,
                )
                if cancel_check and cancel_check():
                    raise AssembleCancelled()
                replace_and_fsync(tmp_path, out_path)
            finally:
                ffmeta_path.unlink(missing_ok=True)
                tmp_path.unlink(missing_ok=True)
            if not existed:
                created.append(out_path)
            outputs.append(out_path)
            log.info("Готово: %s (%.1f МБ)", out_path.name, out_path.stat().st_size / 1e6)
    except Exception:
        # Do not leave a cancelled/failed unique job directory looking ready.
        # Existing finals remain intact because they were never overwritten.
        for path in created:
            path.unlink(missing_ok=True)
        raise

    return outputs


def _parse_bitrate(bitrate: str) -> int:
    try:
        s = bitrate.strip().lower()
        if s.endswith("k"):
            value = int(float(s[:-1]) * 1000)
        elif s.endswith("m"):
            value = int(float(s[:-1]) * 1_000_000)
        else:
            value = int(s)
    except (AttributeError, TypeError, ValueError) as e:
        raise AssembleError("Недопустимый аудиобитрейт") from e
    if value <= 0:
        raise AssembleError("Аудиобитрейт должен быть положительным")
    return value


def _validate_assembly_args(
    bitrate: str, sample_rate: int, max_part_mib: int | None, tempo: float
) -> None:
    _parse_bitrate(bitrate)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AssembleError("Частота дискретизации должна быть положительным целым")
    if max_part_mib is not None and (
        isinstance(max_part_mib, bool)
        or not isinstance(max_part_mib, int)
        or max_part_mib <= 0
    ):
        raise AssembleError("Максимальный размер части должен быть положительным целым")
    if not isinstance(tempo, (int, float)) or not 0.5 <= tempo <= 2.0:
        raise AssembleError("Темп должен быть в диапазоне от 0.5 до 2.0")


def _output_path(output_dir: Path, base_name: str, part_no: int, part_total: int) -> Path:
    suffix = f"_part_{part_no:02d}.m4b" if part_total > 1 else ".m4b"
    # ext4 permits 255 bytes per component. Keep room for suffix and use a
    # fixed fallback because callers may supply an imported book title.
    encoded = str(base_name).encode("utf-8")
    limit = 240 - len(suffix.encode("utf-8"))
    if len(encoded) > limit:
        encoded = encoded[:limit]
        while encoded:
            try:
                name = encoded.decode("utf-8").rstrip(" .")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
        else:
            name = "audiobook"
    else:
        name = str(base_name).rstrip(" .")
    return output_dir / f"{name or 'audiobook'}{suffix}"
