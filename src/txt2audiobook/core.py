import os
import re
import tempfile
import warnings
from threading import local
from typing import List, Dict, Optional

from TTS.api import TTS
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3

# Suppress attention_mask warnings from Transformers
warnings.filterwarnings(
    "ignore",
    message=(
        "The attention mask is not set and cannot be inferred "
        "from input because pad token is same as eos token"
    ),
)


# Defaults (overridable via CLI)
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER = "Viktor Menelaos"
DEFAULT_LANG = "ru"
DEFAULT_MAX_CHUNK_LEN = 180

_thread_local = local()


def _ensure_tts(model: str) -> TTS:
    if not hasattr(_thread_local, "tts"):
        _thread_local.tts = TTS(model_name=model)
    return _thread_local.tts


def split_by_chapters(txt: str) -> List[Dict[str, str]]:
    """Split text by 'Глава N' markers, fallback to ~8000 char chunks.

    Returns a list of dicts: {"title": str, "text": str}
    """
    marks = list(re.finditer(r"(?:^|\n)Глава\s+\d+.*", txt))
    if not marks:
        step, blocks = 8000, []
        for i in range(0, len(txt), step):
            blocks.append({
                "title": f"Part {len(blocks) + 1}",
                "text": txt[i:i + step],
            })
        return blocks

    chapters: List[Dict[str, str]] = []
    for idx, m in enumerate(marks):
        start = m.start()
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(txt)
        chapters.append({
            "title": m.group().strip(),
            "text": txt[start:end].strip(),
        })
    return chapters


def _synthesize_segment(text: str, out_path: str, model: str, speaker: str, lang: str) -> Optional[str]:
    try:
        tts = _ensure_tts(model)
        tts.tts_to_file(text=text, file_path=out_path, speaker=speaker, language=lang)
        return out_path
    except Exception as e:  # pragma: no cover - device / runtime dependent
        print(f"\n⚠️ Ошибка синтеза: {e}")
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None


def _split_into_sayable_chunks(text: str, max_len: int) -> List[str]:
    chunks: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            chunks.append(para)
        else:
            sents = re.split(r"(?<=[.!?])\s+", para)
            cur = ""
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                if len(cur) + len(sent) + 1 <= max_len:
                    cur = f"{cur} {sent}".strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = sent
            if cur:
                chunks.append(cur)
    return chunks


def voice_chapters(
    chapters: List[Dict[str, str]],
    out_dir: str,
    *,
    model: str = MODEL,
    speaker: str = DEFAULT_SPEAKER,
    lang: str = DEFAULT_LANG,
    max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
    tmp_root: str = "tmp",
) -> List[str]:
    """Generate one WAV per chapter by splitting into sub-chunks."""
    try:
        # Test TTS availability early
        _ensure_tts(model)
    except Exception as e:  # pragma: no cover - runtime dependent
        print(f"❌ Ошибка инициализации TTS модели: {e}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    paths: List[str] = []
    for n, ch in enumerate(chapters, 1):
        final_wav = os.path.join(out_dir, f"chapter_{n:02d}.wav")
        if os.path.exists(final_wav):
            print(f"⏭️ Глава {n:02d} уже озвучена, пропускаем.")
            paths.append(final_wav)
            continue

        print(f"▶️  {ch['title']}")
        sayable = _split_into_sayable_chunks(ch["text"], max_chunk_len)

        with tempfile.TemporaryDirectory(dir=tmp_root) as temp_dir:
            chunk_paths: List[str] = []
            total = len(sayable)
            print(f"  🔄 Обработка {total} частей...")
            for idx, chunk in enumerate(sayable, 1):
                seg_path = os.path.join(temp_dir, f"seg_{idx:03d}.wav")
                result = _synthesize_segment(chunk, seg_path, model, speaker, lang)
                if result:
                    chunk_paths.append(result)
                print(f"  📝 Готово: {idx}/{total}", end="\r", flush=True)
            print()

            if not chunk_paths:
                continue

            # Merge segments in small batches to limit memory usage
            try:
                combined = AudioSegment.empty()
                batch_n = 5
                for i in range(0, len(chunk_paths), batch_n):
                    batch = chunk_paths[i:i + batch_n]
                    batch_audio = AudioSegment.empty()
                    for p in batch:
                        try:
                            batch_audio += AudioSegment.from_wav(p)
                        except (CouldntDecodeError, Exception) as e:
                            print(f"⚠️ Ошибка сегмента {p}: {e}")
                    combined += batch_audio
                combined.export(final_wav, format="wav")
                paths.append(final_wav)
            except Exception as e:  # pragma: no cover - IO dependent
                print(f"❌ Ошибка сохранения главы {final_wav}: {e}")
                if os.path.exists(final_wav):
                    try:
                        os.remove(final_wav)
                    except OSError:
                        pass
                continue

    return paths


def merge_to_mp3(wavs: List[str], mp3_path: str) -> bool:
    """Merge WAV files into a single MP3 with basic ID3 tags."""
    try:
        combo = AudioSegment.empty()
        batch_size = 5
        total_wavs = len(wavs)
        for i in range(0, total_wavs, batch_size):
            chunk_end = min(i + batch_size, total_wavs)
            print(f"  🔄 Сборка MP3: {chunk_end}/{total_wavs}", end="\r", flush=True)
            batch = wavs[i:chunk_end]
            batch_audio = AudioSegment.empty()
            for w in batch:
                try:
                    audio = AudioSegment.from_wav(w)
                    batch_audio += audio + AudioSegment.silent(1000)
                except Exception as e:
                    print(f"\n⚠️ Ошибка добавления WAV {w}: {e}")
                    continue
            combo += batch_audio
        print()

        combo.export(mp3_path, format="mp3", bitrate="192k")
        try:
            audio = MP3(mp3_path, ID3=EasyID3)
            audio["title"] = os.path.splitext(os.path.basename(mp3_path))[0]
            audio["artist"] = "XTTS v2 – built-in voice"
            audio["album"] = "Аудиокнига"
            audio.save()
        except Exception as e:
            print(f"⚠️ Не удалось записать ID3 теги: {e}")
        return True
    except Exception as e:  # pragma: no cover - IO dependent
        print(f"❌ Ошибка сохранения MP3 {mp3_path}: {e}")
        if os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
            except OSError:
                pass
        return False


def process_txt_file(
    txt_path: str,
    *,
    out_dir: str,
    chapters_dir: str,
    tmp_root: str,
    model: str = MODEL,
    speaker: str = DEFAULT_SPEAKER,
    lang: str = DEFAULT_LANG,
    max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
) -> Optional[str]:
    """Process a single .txt file into an audiobook MP3. Returns MP3 path or None."""
    txt_name = os.path.basename(txt_path)
    print(f"📖 Обработка книги: {txt_name}")
    try:
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:  # pragma: no cover - IO dependent
        print(f"❌ Ошибка чтения файла {txt_name}: {e}")
        return None

    chapters = split_by_chapters(text)
    if not chapters:
        print(f"❌ Главы не найдены в {txt_name} — пропускаю.")
        return None

    base = os.path.splitext(txt_name)[0]
    mp3_path = os.path.join(out_dir, f"{base}_audiobook.mp3")
    if os.path.exists(mp3_path):
        print(f"⏭️ Аудиокнига {os.path.basename(mp3_path)} уже существует, пропускаем книгу.")
        return mp3_path

    wavs = voice_chapters(
        chapters,
        chapters_dir,
        model=model,
        speaker=speaker,
        lang=lang,
        max_chunk_len=max_chunk_len,
        tmp_root=tmp_root,
    )
    if not wavs:
        print(f"❌ Не удалось создать WAV файлы для {txt_name}")
        return None

    if merge_to_mp3(wavs, mp3_path):
        print(f"✅ Готово: {mp3_path}")
        return mp3_path
    return None

