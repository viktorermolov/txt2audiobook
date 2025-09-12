import os
import argparse
from signal import SIGINT, SIGTERM, signal

from .core import (
    process_txt_file,
)


def _graceful_exit(_sig, _frame):  # pragma: no cover - signal handling
    print("\n⏹️  Прерывание пользователем – завершаю…")
    raise SystemExit(130)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="txt2audiobook",
        description="Преобразование TXT в русскую аудиокнигу (XTTS v2)",
    )
    p.add_argument(
        "input",
        nargs="?",
        default="txtbook",
        help="Папка с .txt файлами",
    )
    p.add_argument("--out", default="audiobook", help="Папка для MP3")
    p.add_argument(
        "--chapters",
        default="audio_chapters",
        help="Папка для WAV глав",
    )
    p.add_argument("--tmp", default="tmp", help="Временная папка для сегментов")
    p.add_argument(
        "--model",
        default=os.getenv(
            "T2A_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
        ),
    )
    p.add_argument(
        "--speaker",
        default=os.getenv("T2A_SPEAKER", "Viktor Menelaos"),
    )
    p.add_argument("--lang", default=os.getenv("T2A_LANG", "ru"))
    p.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("T2A_MAX_CHARS", 180)),
        help="Макс. символов в сегменте",
    )
    p.add_argument(
        "--delete-txt",
        action="store_true",
        help="Удалять исходные .txt после успешной сборки",
    )
    p.add_argument(
        "--keep-wavs",
        action="store_true",
        help="Не удалять WAV главы после сборки MP3",
    )
    return p


def main(argv=None) -> int:
    signal(SIGINT, _graceful_exit)
    signal(SIGTERM, _graceful_exit)

    args = build_parser().parse_args(argv)

    # Ensure directories
    for d in [args.input, args.out, args.chapters, args.tmp]:
        os.makedirs(d, exist_ok=True)

    try:
        txt_files = sorted([f for f in os.listdir(args.input) if f.endswith(".txt")])
    except OSError as e:  # pragma: no cover - IO dependent
        print(f"❌ Ошибка чтения директории {args.input}: {e}")
        return 1

    if not txt_files:
        print("❌ Книг для озвучивания нет.")
        return 1

    exit_code = 0
    for name in txt_files:
        txt_path = os.path.join(args.input, name)
        mp3 = process_txt_file(
            txt_path,
            out_dir=args.out,
            chapters_dir=args.chapters,
            tmp_root=args.tmp,
            model=args.model,
            speaker=args.speaker,
            lang=args.lang,
            max_chunk_len=args.max_chars,
        )

        if mp3:
            if args.delete_txt:
                try:
                    os.remove(txt_path)
                except OSError as e:  # pragma: no cover - IO dependent
                    print(f"⚠️ Не удалось удалить исходный файл {txt_path}: {e}")

            if not args.keep_wavs:
                # Remove WAVs for this book only
                try:
                    for fn in os.listdir(args.chapters):
                        if fn.startswith("chapter_") and fn.endswith(".wav"):
                            os.remove(os.path.join(args.chapters, fn))
                except OSError as e:  # pragma: no cover - IO dependent
                    print(f"⚠️ Ошибка очистки WAV: {e}")
        else:
            exit_code = 2

    print("🎉 Все книги озвучены. Книги закончились.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
