"""Local speech synthesis with Silero TTS (CPU, ARM64).

Engine: Silero `v5_5_ru` package (multi-speaker), default voice `eugene`.
The model is a `torch.package` archive (~tens of MB) downloaded on demand into
the voices volume and loaded once into memory.

Design goals (unchanged from before):
  * Restart-safe: work is split into small numbered chunks; an existing valid
    chunk WAV is never re-synthesized. The chunk plan is persisted so a resumed
    job rebuilds the exact same numbering and chapter boundaries.
  * Conservative: one model resident in memory; torch thread count is capped so
    the Pi stays cool.
  * Robust: per-chunk retries; a model missing on disk is downloaded on demand
    from models.silero.ai.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
import wave
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

import httpx

from ..files import replace_and_fsync
from ..log import get_logger
from .clean import clean_text, split_sentences
from .types import Document

log = get_logger("synth")

_SILERO_BASE = "https://models.silero.ai/models/tts/ru"

# A release is usable only when its SHA-256 is explicitly pinned here.
_MODEL_SHA256_ALLOWLIST: dict[str, str] = {
    "v5_5_ru": "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_HASH_CACHE: dict[str, tuple[tuple[int, int, int, int, int], str]] = {}


class SynthError(Exception):
    """Speech synthesis failure (shown to the user, Russian message)."""


class SynthCancelled(SynthError):
    """Internal signal: the user cancelled synthesis."""


_CACHE_SCHEMA = 1
_CACHE_FILE = "synthesis-cache.json"
_SILENCED_FILE = "silenced.json"
_SILENCED_SCHEMA = 1


# ---------------------------------------------------------------------
# Voice / model management
# ---------------------------------------------------------------------

def ensure_voice(model_id: str, voices_dir: Path) -> Path:
    """Return the local path to the Silero model package, downloading it from
    models.silero.ai on first use. `model_id` is e.g. ``v5_5_ru``."""
    expected_sha256 = _expected_model_sha256(model_id)
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    dest = voices_dir / f"{model_id}.pt"

    if dest.exists():
        try:
            _verify_model_file(dest, model_id=model_id)
        except SynthError:
            # Leave a previously cached model intact until its replacement has
            # passed verification and can be atomically promoted.
            log.warning("Кэш модели Silero %s не прошёл проверку целостности", model_id)
        else:
            return dest

    url = f"{_SILERO_BASE}/{model_id}.pt"
    log.info("Скачиваю модель Silero %s …", model_id)
    try:
        _download(url, dest, expected_sha256)
    except Exception:
        raise SynthError(
            f"Не удалось скачать или проверить модель {model_id}. "
            "Проверьте подключение к интернету для первой загрузки модели."
        ) from None
    log.info("Модель %s готова (%.1f МБ)", model_id, dest.stat().st_size / 1e6)
    return dest


def _expected_model_sha256(model_id: str) -> str:
    """Return the pinned digest for a supported model, or fail closed."""
    if not isinstance(model_id, str) or model_id not in _MODEL_SHA256_ALLOWLIST:
        raise SynthError("Неподдерживаемая модель Silero")
    digest = _MODEL_SHA256_ALLOWLIST[model_id]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise SynthError(
            f"Для модели Silero {model_id} не настроена проверенная контрольная сумма"
        )
    return digest


def _model_file_signature_from_stat(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(file_stat.st_mode):
        raise SynthError("Файл модели Silero не должен быть символической ссылкой")
    if file_stat.st_size <= 0:
        raise SynthError("Файл модели Silero пуст")
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _model_file_signature(path: Path) -> tuple[int, int, int, int, int]:
    return _model_file_signature_from_stat(os.stat(path, follow_symlinks=False))


def _open_model_file(
    path: Path, expected_signature: tuple[int, int, int, int, int],
):
    """Open an already-verified package without following a replacement link."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:  # pragma: no cover - Pi/Linux always provides it
        raise SynthError("Платформа не поддерживает безопасную загрузку модели Silero")
    try:
        fd = os.open(path, os.O_RDONLY | nofollow)
    except OSError as e:
        raise SynthError("Не удалось открыть файл модели Silero") from e
    try:
        source = os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise
    try:
        if _model_file_signature_from_stat(os.fstat(source.fileno())) != expected_signature:
            raise SynthError("Файл модели Silero изменился перед загрузкой")
    except Exception:
        source.close()
        raise
    return source


def _verify_model_file(
    path: Path, *, model_id: str | None = None
) -> tuple[int, int, int, int, int]:
    """Verify an allowlisted package before handing it to torch.package.

    The per-process cache is keyed by a conservative immutable-file signature.
    ctime is included so a same-size replacement is not treated as unchanged.
    """
    path = Path(path)
    if model_id is None:
        if path.suffix != ".pt":
            raise SynthError("Недопустимый путь к модели Silero")
        model_id = path.stem
    expected_sha256 = _expected_model_sha256(model_id)
    expected_name = f"{model_id}.pt"
    if path.name != expected_name:
        raise SynthError("Недопустимый путь к модели Silero")

    try:
        signature = _model_file_signature(path)
    except OSError as e:
        raise SynthError("Не удалось прочитать файл модели Silero") from e
    cache_key = str(path.resolve())
    cached = _MODEL_HASH_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        actual_sha256 = cached[1]
    else:
        digest = sha256()
        try:
            with _open_model_file(path, signature) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            if _model_file_signature(path) != signature:
                raise SynthError("Файл модели Silero изменился во время проверки")
        except OSError as e:
            raise SynthError("Не удалось прочитать файл модели Silero") from e
        actual_sha256 = digest.hexdigest()
        _MODEL_HASH_CACHE[cache_key] = (signature, actual_sha256)

    if actual_sha256 != expected_sha256:
        raise SynthError("Контрольная сумма модели Silero не совпадает")
    return signature


def _validate_model_download_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "models.silero.ai"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise SynthError("Недопустимый адрес загрузки модели Silero")


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Stream a pinned model into a private temporary file and promote it."""
    _validate_model_download_url(url)
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise SynthError("Некорректная контрольная сумма модели Silero")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".part", dir=dest.parent
    )
    tmp = Path(tmp_name)
    try:
        digest = sha256()
        with os.fdopen(fd, "wb") as output:
            with httpx.Client(follow_redirects=False, timeout=300) as client:
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        raise SynthError("Перенаправление при загрузке модели Silero запрещено")
                    response.raise_for_status()
                    for block in response.iter_bytes(chunk_size=1 << 16):
                        output.write(block)
                        digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise SynthError("Контрольная сумма скачанной модели Silero не совпадает")
        tmp.replace(dest)
        _MODEL_HASH_CACHE.pop(str(dest.resolve()), None)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------

@dataclass
class Chunk:
    index: int
    chapter_index: int
    chapter_title: Optional[str]
    text: str


def build_plan(doc: Document, chunk_chars: int) -> list[Chunk]:
    """Split each chapter's text into ~chunk_chars chunks on sentence
    boundaries. Chapter boundaries are preserved for M4B chapter marks.

    Silero degrades on very long inputs, so chunk_chars should stay well under
    ~1000; a single oversized sentence is hard-split as a last resort."""
    chunks: list[Chunk] = []
    idx = 0
    # Hard ceiling per apply_tts call — Silero recommends staying under ~1000.
    hard_max = max(chunk_chars, 200) + 200
    for ch_i, chapter in enumerate(doc.chapters):
        # Prepend the chapter title so it is spoken before the body. The
        # spoken form goes through clean_text — titles bypass the body
        # cleaning, yet digits/roman numerals ("Глава 1", "Глава XIV") are
        # outside Silero's symbol set and would otherwise be dropped from
        # speech. plan.chapter_title keeps the raw title for the M4B TOC.
        body = chapter.text
        if chapter.title:
            spoken_title = clean_text(chapter.title)
            if spoken_title:
                body = f"{spoken_title}.\n{body}"
        sentences = split_sentences(body)
        if not sentences:
            continue
        buf: list[str] = []
        buf_len = 0
        first_in_chapter = True

        def flush(first: bool) -> bool:
            nonlocal idx, buf, buf_len
            if not buf:
                return first
            chunks.append(
                Chunk(idx, ch_i,
                      chapter.title if first else None,
                      " ".join(buf))
            )
            idx += 1
            buf, buf_len = [], 0
            return False

        for sent in sentences:
            for piece in _hard_split(sent, hard_max):
                if buf and buf_len + len(piece) + 1 > chunk_chars:
                    first_in_chapter = flush(first_in_chapter)
                buf.append(piece)
                buf_len += len(piece) + 1
        first_in_chapter = flush(first_in_chapter)
    if not chunks:
        raise SynthError("Нет текста для синтеза после разбиения на чанки")
    return chunks


def _hard_split(sentence: str, hard_max: int) -> list[str]:
    """Break a monstrous single sentence into <=hard_max pieces.

    Prefer word boundaries; if OCR/glue leaves a single huge token, split that
    token too so no single Silero call gets an oversized input.
    """
    if len(sentence) <= hard_max:
        return [sentence]
    pieces: list[str] = []
    words = sentence.split(" ")
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0

    for w in words:
        if len(w) > hard_max:
            flush()
            pieces.extend(w[i:i + hard_max] for i in range(0, len(w), hard_max))
            continue
        if cur and cur_len + len(w) + 1 > hard_max:
            flush()
        cur.append(w)
        cur_len += len(w) + 1
    flush()
    return pieces


def save_plan(plan: list[Chunk], path: Path) -> None:
    _atomic_write_text(
        path,
        json.dumps([asdict(c) for c in plan], ensure_ascii=False, indent=1),
    )


def load_plan(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**c) for c in data]


# ---------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write a small JSON/text record without leaving a partial resume file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text)
        # fsync payload and directory: a power cut must not leave an empty
        # cache manifest, whose mismatch on resume discards every chunk.
        replace_and_fsync(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _wav_is_valid(
    path: Path, expected_rate: int | None = None, *,
    require_speech: bool = False, tail_frames: int = 0,
) -> bool:
    """Check the actual PCM payload, not merely a readable WAV header."""
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.getnframes()
            if (
                channels != 1
                or sample_width != 2
                or rate <= 0
                or (expected_rate is not None and rate != expected_rate)
                or w.getcomptype() != "NONE"
                or frames <= 0
            ):
                return False
            pcm = w.readframes(frames)
            if len(pcm) != frames * channels * sample_width or w.readframes(1):
                return False
            if require_speech:
                speech_frames = frames - tail_frames
                if speech_frames < max(1, rate // 12):
                    return False
                samples = memoryview(pcm)[:speech_frames * 2].cast("h")
                voiced = 0
                for sample in samples:
                    if abs(sample) >= 64:
                        voiced += 1
                        if voiced >= max(1, rate // 50):
                            break
                else:
                    return False
            return True
    except Exception:
        return False


def _plan_digest(plan: list[Chunk]) -> str:
    payload = json.dumps(
        [asdict(chunk) for chunk in plan],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _model_identity(model_path: Path) -> dict[str, object]:
    """Cheap, conservative identity for a local model package.

    A model replacement normally changes its inode metadata or size. Including
    the absolute path also prevents reusing audio from another configured voice
    package without hashing a 100+ MiB model on every resume.
    """
    path = Path(model_path)
    identity: dict[str, object] = {"path": str(path.resolve())}
    try:
        stat = path.stat()
    except OSError:
        identity["missing"] = True
    else:
        identity.update(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            inode=getattr(stat, "st_ino", None),
        )
    return identity


def _cache_manifest(
    plan: list[Chunk],
    model_path: Path,
    *,
    speaker: str,
    sample_rate: int,
    put_accent: bool,
    put_yo: bool,
    sentence_silence: float,
) -> dict[str, object]:
    return {
        "schema": _CACHE_SCHEMA,
        "plan_sha256": _plan_digest(plan),
        "model": _model_identity(model_path),
        "speaker": speaker,
        "sample_rate": sample_rate,
        "put_accent": put_accent,
        "put_yo": put_yo,
        "sentence_silence": sentence_silence,
    }


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _invalidate_audio_cache(audio_dir: Path) -> None:
    # A missing manifest is a legacy cache: its voice/settings are unknowable.
    # Only own chunk files are removed; callers may keep unrelated diagnostics.
    for path in audio_dir.glob("chunk_*.wav"):
        path.unlink(missing_ok=True)
    for path in audio_dir.glob("chunk_*.wav.part"):
        path.unlink(missing_ok=True)
    (audio_dir / _SILENCED_FILE).unlink(missing_ok=True)


def _prepare_cache(audio_dir: Path, manifest: dict[str, object]) -> None:
    marker = audio_dir / _CACHE_FILE
    existing = _load_json(marker)
    if existing != manifest:
        _invalidate_audio_cache(audio_dir)
        _atomic_write_text(
            marker,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1),
        )


def _load_silenced(audio_dir: Path, plan: list[Chunk]) -> set[int]:
    data = _load_json(audio_dir / _SILENCED_FILE)
    if not isinstance(data, dict) or data.get("schema") != _SILENCED_SCHEMA:
        return set()
    raw = data.get("indices")
    allowed = {chunk.index for chunk in plan}
    if not isinstance(raw, list):
        return set()
    return {index for index in raw if isinstance(index, int) and index in allowed}


def _save_silenced(audio_dir: Path, indices: set[int]) -> None:
    _atomic_write_text(
        audio_dir / _SILENCED_FILE,
        json.dumps(
            {"schema": _SILENCED_SCHEMA, "indices": sorted(indices)},
            ensure_ascii=False,
            indent=1,
        ),
    )


def _validate_synthesis_args(
    sample_rate: int,
    retries: int,
    sentence_silence: float,
    threads: int,
    disk_min_free_mib: int,
) -> None:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate not in {8000, 24000, 48000}
    ):
        raise SynthError("Недопустимая частота дискретизации Silero")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise SynthError("Число повторов синтеза должно быть неотрицательным целым")
    if (
        isinstance(sentence_silence, bool)
        or not isinstance(sentence_silence, (int, float))
        or not math.isfinite(sentence_silence)
        or sentence_silence < 0
    ):
        raise SynthError("Пауза между фрагментами должна быть неотрицательным числом")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise SynthError("Число потоков синтеза должно быть положительным целым")
    if (
        isinstance(disk_min_free_mib, bool)
        or not isinstance(disk_min_free_mib, (int, float))
        or not math.isfinite(disk_min_free_mib)
        or disk_min_free_mib < 0
    ):
        raise SynthError("Минимальный свободный объём диска не может быть отрицательным")


# Characters Silero can voice; anything else (after cleaning) is dropped so a
# chunk of pure punctuation/symbols doesn't crash apply_tts.
def _has_speakable(text: str) -> bool:
    return any(c.isalnum() for c in text)


# Silero's tokenizer maps each character to an id and raises KeyError on an
# unknown symbol (e.g. the em-dash U+2014). We keep dialogue dashes in the
# human-readable cleaned.txt but sanitize them out at the TTS boundary.
# Dash/hyphen variants that act as separators -> comma (a natural pause).
# NB: the plain ASCII hyphen U+002D is preserved so hyphenated words
# (по-настоящему) stay intact.
_DASH_RE = re.compile(
    "[‐‑‒–—―⁃−­﹘﹣－−]"
)
_STAR_RE = re.compile(r"[*]+")


def _sanitize_for_silero(text: str) -> str:
    """Prosody-aware pre-pass: turn separators into pauses Silero can voice.
    The exhaustive guard is _SileroEngine.filter_text (the model's own symbol
    table); this step only makes the result read naturally."""
    t = _DASH_RE.sub(", ", text)        # em/en dashes -> comma pause
    t = t.replace("…", ". ")
    t = _STAR_RE.sub(" ", t)            # scene-break asterisks
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(?:,\s*){2,}", ", ", t)        # collapse comma runs
    t = re.sub(r"\s+([,.!?:;])", r"\1", t)        # no space before punctuation
    t = re.sub(r"^[\s,.:;!?]+", "", t)            # strip leading punctuation
    return t.strip()


_MAX_SPLIT_DEPTH = 3  # up to 8 parts per chunk
_SPLIT_PAUSE_SEC = 0.3
_SPLIT_POINTS = (
    re.compile(r"[.!?…]+\s+"),   # sentence end
    re.compile(r"[,;:–]\s+"),     # clause
    re.compile(r"\s+"),           # any word boundary
)


def _split_in_half(text: str) -> tuple[str, str] | None:
    """Split near the middle at the strongest available boundary, keeping
    both halves speakable. None if the text cannot be split."""
    middle = len(text) / 2
    for pattern in _SPLIT_POINTS:
        cuts = [m.end() for m in pattern.finditer(text) if 0 < m.end() < len(text)]
        for cut in sorted(cuts, key=lambda c: abs(c - middle)):
            first, second = text[:cut].strip(), text[cut:].strip()
            if _has_speakable(first) and _has_speakable(second):
                return first, second
    return None


class _SileroEngine:
    """Lazy wrapper so the module imports without torch installed."""

    def __init__(self, model_path: Path, speaker: str, threads: int):
        # torch.package executes package code while unpickling, so integrity is
        # established before the archive is ever passed to its importer. The
        # importer receives a descriptor whose metadata still matches that
        # verified immutable-file signature, avoiding a pathname TOCTOU.
        signature = _verify_model_file(model_path)
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise SynthError("Библиотека torch не установлена в контейнере") from e

        self._torch = torch
        torch.set_grad_enabled(False)

        try:
            with _open_model_file(model_path, signature) as source:
                importer = torch.package.PackageImporter(source)
                self.model = importer.load_pickle("tts_models", "model")
        except Exception as e:
            raise SynthError(
                f"Не удалось загрузить модель Silero: {e}. "
                "Файл модели повреждён — удалите его из папки voices для повторной загрузки."
            ) from e

        try:
            self.model.to(torch.device("cpu"))
        except Exception:
            pass
        # Set threads only now: the Silero package runs torch.set_num_threads(1)
        # when it is unpickled, which silently made synthesis single-threaded.
        try:
            torch.set_num_threads(max(1, int(threads)))
            log.info("Silero загружен: потоков PyTorch %d", torch.get_num_threads())
        except Exception:
            pass

        speakers = list(getattr(self.model, "speakers", []) or [])
        if speakers and speaker not in speakers:
            sample = ", ".join(speakers[:25])
            raise SynthError(
                f"Голос «{speaker}» недоступен в модели. Доступные голоса: {sample}"
            )
        self.speaker = speaker
        self.speakers = speakers

        # The model's own symbol table is the source of truth: any character
        # outside it makes apply_tts raise KeyError (e.g. em-dash, Latin
        # letters, digits, quotes). We filter every input down to this set so
        # synthesis can never crash on an exotic character.
        syms = getattr(self.model, "symbols", None)
        self.symbols: set[str] | None = set(syms) if syms else None

    def filter_text(self, text: str) -> str:
        """Map text onto the model's symbol set. Letters are kept (apply_tts
        lowercases internally); residual dash variants become the in-vocabulary
        en-dash; everything else (Latin, quotes, brackets, emoji, control
        chars) is dropped. Returns '' if nothing speakable remains."""
        if not self.symbols:
            return text
        syms = self.symbols
        dash = "–" if "–" in syms else ("-" if "-" in syms else ",")
        out: list[str] = []
        for c in text:
            if c in syms or c.lower() in syms:
                out.append(c)
            elif c in "—―‒−﹘﹣－":
                out.append(dash)
            else:
                out.append(" ")
        t = "".join(out)
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\s+([,.!?:;])", r"\1", t)
        # Dropping unsupported characters can leave punctuation runs
        # (e.g. "мир,, круто" or "мир!, сказал"); keep just the first mark.
        t = re.sub(r"([,.!?:;])(?:\s*[,.!?:;])+", r"\1", t)
        t = re.sub(r"^[\s,.:;!?–-]+", "", t)
        return t.strip()

    def _tts_array(
        self, text: str, *, sample_rate: int, put_accent: bool, put_yo: bool, depth: int = 0,
    ):
        """apply_tts as a float32 array. Silero refuses inputs that would yield
        more than ~60 s of audio ("probably it's too long") — dense dialogue
        hits that well below the character cap — so such a chunk is split at a
        sentence boundary and voiced in parts instead of becoming silence."""
        import numpy as np

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo,
            )
        except Exception as e:
            halves = _split_in_half(text) if depth < _MAX_SPLIT_DEPTH else None
            if not halves or "too long" not in str(e).lower():
                raise
            log.info("Фрагмент слишком длинный для Silero (%d симв.) — делю пополам", len(text))
            gap = np.zeros(int(_SPLIT_PAUSE_SEC * sample_rate), dtype="float32")
            first, second = (
                self._tts_array(
                    half, sample_rate=sample_rate, put_accent=put_accent,
                    put_yo=put_yo, depth=depth + 1,
                )
                for half in halves
            )
            return np.concatenate([first, gap, second])
        # apply_tts returns a 1-D float tensor in [-1, 1].
        arr = audio.detach().cpu().numpy().astype("float32")
        return arr.reshape(-1) if arr.ndim > 1 else arr

    def synth_to_wav(
        self,
        text: str,
        out: Path,
        *,
        sample_rate: int,
        put_accent: bool,
        put_yo: bool,
        sentence_silence: float,
    ) -> None:
        import numpy as np

        arr = self._tts_array(
            text, sample_rate=sample_rate, put_accent=put_accent, put_yo=put_yo,
        )
        if arr.size < max(1, sample_rate // 12):
            raise SynthError("Silero вернул пустой звук")
        if not np.isfinite(arr).all():
            raise SynthError("Silero вернул некорректные значения звука")
        if np.count_nonzero(np.abs(arr) >= 64 / 32767) < max(1, sample_rate // 50):
            raise SynthError("Silero вернул звук без речи")
        np.clip(arr, -1.0, 1.0, out=arr)
        pcm = (arr * 32767.0).astype("<i2")

        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
            tail = int(max(0.0, sentence_silence) * sample_rate)
            if tail:
                w.writeframes(b"\x00\x00" * tail)


def synthesize_plan(
    plan: list[Chunk],
    model_path: Path,
    audio_dir: Path,
    *,
    speaker: str = "eugene",
    sample_rate: int = 24000,
    put_accent: bool = True,
    put_yo: bool = True,
    sentence_silence: float = 0.4,
    threads: int = 4,
    retries: int = 2,
    disk_min_free_mib: int = 200,
    silenced_out: list[int] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Synthesize every chunk to audio_dir/chunk_NNNNN.wav, skipping any that
    already exist and are valid. Returns ordered list of WAV paths.

    `cancel_check` is polled before every cache hit and between chunks. Partial
    chunks remain on disk for a later resume. The cache is reusable only when
    the complete plan and all voice-affecting settings match its manifest.
    Chunk indices replaced with silence are persisted and appended to
    `silenced_out` on every resume.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    _validate_synthesis_args(
        sample_rate, retries, sentence_silence, threads, disk_min_free_mib
    )
    if not plan:
        raise SynthError("Нет фрагментов для синтеза")
    if cancel_check and cancel_check():
        raise SynthCancelled()

    _prepare_cache(
        audio_dir,
        _cache_manifest(
            plan,
            model_path,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
            sentence_silence=sentence_silence,
        ),
    )

    engine: _SileroEngine | None = None
    total = len(plan)
    paths: list[Path] = []
    silenced = _load_silenced(audio_dir, plan)
    tail_frames = int(sentence_silence * sample_rate)
    if silenced_out is not None:
        silenced_out.extend(index for index in sorted(silenced) if index not in silenced_out)
    consecutive_silenced = 0
    silence_limit = max(3, math.ceil(total * 0.10))

    def record_silence(chunk: Chunk, out: Path, *, seconds: float, reason: str) -> None:
        nonlocal consecutive_silenced
        prospective = len(silenced | {chunk.index})
        prospective_consecutive = consecutive_silenced + 1
        # Do not turn an unusable model/configuration into a valid but silent
        # audiobook. A few isolated fragments remain recoverable on resume.
        if prospective >= total:
            raise SynthError("Все фрагменты оказались без звука — сборка отменена")
        if prospective_consecutive >= 3:
            raise SynthError("Синтез систематически не создаёт звук (3 фрагмента подряд)")
        if prospective > silence_limit:
            raise SynthError(
                f"Синтез систематически не создаёт звук ({prospective} из {total} фрагментов)"
            )
        _write_silence(out, seconds=seconds, rate=sample_rate)
        silenced.add(chunk.index)
        _save_silenced(audio_dir, silenced)
        if silenced_out is not None and chunk.index not in silenced_out:
            silenced_out.append(chunk.index)
        consecutive_silenced = prospective_consecutive
        log.warning("Чанк %d заменён тишиной: %s", chunk.index, reason)

    for n, chunk in enumerate(plan, start=1):
        out = audio_dir / f"chunk_{chunk.index:05d}.wav"
        paths.append(out)

        # Cancellation deliberately precedes cache use, so a cancelled resume
        # cannot report fake progress or proceed to assembly on cached audio.
        if cancel_check and cancel_check():
            raise SynthCancelled()

        if _wav_is_valid(
            out, expected_rate=sample_rate,
            require_speech=chunk.index not in silenced,
            tail_frames=tail_frames,
        ):
            if chunk.index in silenced:
                consecutive_silenced += 1
            else:
                consecutive_silenced = 0
            if progress_cb:
                progress_cb(n, total)
            continue

        # A stale ledger entry without a corresponding WAV must not survive a
        # resumed retry: this chunk is about to receive a fresh result.
        if chunk.index in silenced:
            silenced.remove(chunk.index)
            _save_silenced(audio_dir, silenced)

        # Periodic disk guard: a long synth writes megabytes per chunk; fail
        # with a clear message before the filesystem fills and corrupts a WAV.
        if n % 25 == 1:
            try:
                free_mib = shutil.disk_usage(audio_dir).free / (1024 * 1024)
                if free_mib < disk_min_free_mib:
                    raise SynthError(
                        f"Недостаточно места на диске для синтеза "
                        f"(свободно {free_mib:.0f} МБ)."
                    )
            except OSError:
                pass

        # Defer loading the model until the first chunk actually needs work
        # (keeps a full resume cheap — no model load if every chunk is cached).
        if engine is None:
            engine = _SileroEngine(model_path, speaker, threads)

        # Pre-pass for prosody, then filter down to the model's symbol table so
        # apply_tts can never KeyError on an unsupported character.
        text = engine.filter_text(_sanitize_for_silero(chunk.text.strip()))
        if not text or not _has_speakable(text):
            # Nothing speakable left — a short silence keeps indices aligned.
            record_silence(chunk, out, seconds=0.3, reason="нет произносимого текста")
            if progress_cb:
                progress_cb(n, total)
            continue

        last_err: Exception | None = None
        done_ok = False
        for attempt in range(1, retries + 2):
            try:
                tmp = out.with_suffix(".wav.part")
                engine.synth_to_wav(
                    text, tmp,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo,
                    sentence_silence=sentence_silence,
                )
                if not _wav_is_valid(
                    tmp, expected_rate=sample_rate,
                    require_speech=True, tail_frames=tail_frames,
                ):
                    raise SynthError("Silero вернул пустой звук")
                tmp.replace(out)
                done_ok = True
                break
            except Exception as e:
                last_err = e
                log.warning(
                    "Чанк %d: попытка %d/%d не удалась: %s",
                    chunk.index, attempt, retries + 1, e,
                )
                out.with_suffix(".wav.part").unlink(missing_ok=True)

        if not done_ok:
            # One pathological chunk must never sink a multi-hour book: log
            # loudly and substitute a brief silence so the job completes.
            log.error(
                "Чанк %d не синтезирован после %d попыток (%s) — заменяю тишиной",
                chunk.index, retries + 1, last_err,
            )
            record_silence(
                chunk, out, seconds=0.6, reason="исчерпаны повторы синтеза"
            )
        else:
            consecutive_silenced = 0

        if progress_cb:
            progress_cb(n, total)

    return paths


def _write_silence(path: Path, seconds: float = 0.3, rate: int = 24000) -> None:
    frames = int(seconds * rate)
    if frames <= 0:
        raise SynthError("Длительность тишины должна быть положительной")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * frames)
        if not _wav_is_valid(tmp, expected_rate=rate):
            raise SynthError("Не удалось записать фрагмент тишины")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
