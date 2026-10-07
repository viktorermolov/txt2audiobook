"""Real Silero/ffmpeg smoke test with temporary state and no external services."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from ebooklib import epub

from app.config import AppConfig
from app.state import Stage, Store
from app.worker import CancelRegistry, Worker


class RecordingNotifier:
    def notify(self, _text, _chat=None):
        pass

    def send_message(self, _text, _chat=None):
        pass


class LocalPublisher:
    def publish(self, job, _meta, **_kwargs):
        if len(job.outputs) != 1 or not Path(job.outputs[0]).is_file():
            raise AssertionError("Ожидался один готовый M4B")
        return {"url": "https://example.invalid/offline-test", "item_id": "offline-test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="audiobook-smoke-") as temporary:
        root = Path(temporary)
        cfg = AppConfig.model_validate({
            "telegram": {"bot_token": "offline-test", "allowed_user_id": 1},
            "tts": {"threads": 2},
            "paths": {**{key: root / key for key in ("books", "audiobook", "state", "work")},
                      "voices": args.voices},
            "audiobookshelf": {"library_path": root / "library"},
        })
        cfg.ensure_dirs()
        book = epub.EpubBook()
        book.set_identifier("offline-quality-smoke")
        book.set_title("Проверка качества")
        book.set_language("ru")
        book.add_author("Тест сервиса")
        short = epub.EpubHtml(title="Короткий раздел", file_name="short.xhtml", lang="ru")
        short.content = "<p>Не уходи.</p>"
        main_chapter = epub.EpubHtml(title="Основной раздел", file_name="main.xhtml", lang="ru")
        main_chapter.content = (
            "<p>Это <em>не</em>правда. Длина пять сантиметров.</p>"
            "<p>Дата: 12.05.2024. Он пришёл к 5-му дому. Длина 5 см.</p>"
        )
        for item in (short, main_chapter, epub.EpubNcx(), epub.EpubNav()):
            book.add_item(item)
        book.spine = [short, main_chapter]
        source = root / "sample.epub"
        epub.write_epub(str(source), book)
        store = Store(cfg.paths.state / "jobs.db")
        job = store.create_job(source.name, str(source), hashlib.sha256(source.read_bytes()).hexdigest(), 1)
        loop = asyncio.new_event_loop()
        try:
            worker = Worker(cfg, store, RecordingNotifier(), loop, CancelRegistry())
            worker.publisher = LocalPublisher()
            worker._process(job.id)
        finally:
            loop.close()
        job = store.get(job.id)
        if job.stage != Stage.DONE:
            raise AssertionError(job.error or job.delivery_error or job.stage)
        work = cfg.paths.work / str(job.id)
        # The plan includes spoken headings, which cleaned.txt stores separately.
        spoken = " ".join(chunk["text"] for chunk in json.loads((work / "plan.json").read_text()))
        if "Не уходи." not in spoken or "неправда" not in spoken:
            raise AssertionError("Текст EPUB потерян или изменён")
        quality = json.loads((work / "quality.json").read_text())
        if quality["silenced"]:
            raise AssertionError("Тестовая речь содержит пропуски")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_chapters", "-show_streams",
             "-of", "json", job.outputs[0]], check=True, capture_output=True, text=True,
        )
        info = json.loads(probe.stdout)
        audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
        if len(info["chapters"]) != 2 or float(info["format"]["duration"]) <= 5:
            raise AssertionError("Главы или длительность M4B неверны")
        if audio["codec_name"] != "aac" or audio["channels"] != 1 or int(audio["sample_rate"]) != 48000:
            raise AssertionError("Параметры аудио изменились")
        print(json.dumps({"result": "ok", "files": len(job.outputs), "chapters": 2,
                          "duration": float(info["format"]["duration"]),
                          "chunks": quality["chunks"], "silenced": quality["silenced"]}))


if __name__ == "__main__":
    main()
