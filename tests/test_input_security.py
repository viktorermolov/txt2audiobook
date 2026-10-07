from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import AppConfig
from app.ingest import ingest_file
from app.pipeline.types import Chapter, Document, ExtractError
from app.state import Stage, Store
from app.watcher import FolderWatcher

extract_module = importlib.import_module("app.pipeline.extract")
ingest_module = importlib.import_module("app.ingest")


class _RecordingNotifier:
    def __init__(self) -> None:
        self.notifications: list[str] = []

    def notify(self, text: str, chat_id: int | None = None) -> None:
        self.notifications.append(text)


class _ReadRecorder(BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _write_epub(path: Path, chapter: str) -> None:
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Большая книга</dc:title><dc:creator>Иван Петров</dc:creator>
    <meta name="calibre:series" content="Длинная серия"/>
    <meta name="calibre:series_index" content="12"/>
  </metadata>
  <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="chapter"/></spine>
</package>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/content.opf", opf)
        archive.writestr("OPS/chapter.xhtml", chapter)


class InputSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = AppConfig.model_validate({
            "telegram": {"bot_token": "test-token", "allowed_user_id": 7},
            "paths": {
                "books": str(self.root / "books"),
                "audiobook": str(self.root / "audiobook"),
                "state": str(self.root / "state"),
                "work": str(self.root / "work"),
                "voices": str(self.root / "voices"),
            },
            "audiobookshelf": {
                "url": "http://abs.test",
                "public_url": "https://abs.example",
                "library_path": str(self.root / "library"),
            },
        })
        self.cfg.ensure_dirs()
        self.store = Store(self.cfg.paths.state / "jobs.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is unavailable")
    def test_ingest_rejects_symlink_without_staging_or_database_side_effects(self) -> None:
        target = self.root / "target.txt"
        target.write_text("Секретный текст не должен быть принят.", encoding="utf-8")
        source = self.root / "book.txt"
        source.symlink_to(target)

        job, reason = ingest_file(self.cfg, self.store, source, chat_id=None)

        self.assertIsNone(job)
        self.assertIn("Символические ссылки", reason)
        self.assertEqual([], self.store.list_jobs())
        self.assertFalse((self.cfg.paths.work / "_staging").exists())
        self.assertEqual("Секретный текст не должен быть принят.", target.read_text(encoding="utf-8"))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is unavailable")
    def test_path_swap_after_open_copies_original_inode_not_symlink_target(self) -> None:
        original = "Исходный текст, открытый до подмены пути."
        secret = "Этот файл не должен попасть в задачу."
        source = self.root / "race.txt"
        source.write_text(original, encoding="utf-8")
        target = self.root / "secret.txt"
        target.write_text(secret, encoding="utf-8")
        real_open = ingest_module._open_source

        def open_then_swap(path, flags):
            fd = real_open(path, flags)
            source.unlink()
            source.symlink_to(target)
            return fd

        with patch.object(ingest_module, "_open_source", side_effect=open_then_swap):
            job, reason = ingest_file(self.cfg, self.store, source, chat_id=None)

        self.assertIsNone(reason)
        self.assertIsNotNone(job)
        self.assertEqual(original, Path(job.source_path).read_text(encoding="utf-8"))
        self.assertNotIn(secret, Path(job.source_path).read_text(encoding="utf-8"))
        self.assertEqual(secret, target.read_text(encoding="utf-8"))

    def test_ingest_durability_order_is_source_fsync_db_rename_then_queue(self) -> None:
        source = self.root / "durable.txt"
        source.write_text("Полностью записанный исходный текст.", encoding="utf-8")
        events: list[str] = []
        real_fsync = ingest_module.fsync_file
        real_create = self.store.create_job
        real_rename = ingest_module.rename_directory_and_fsync
        real_update = self.store._update

        def fsync_source(path: Path) -> None:
            real_fsync(path)
            events.append("source-fsync")

        def create_job(*args, **kwargs):
            self.assertEqual(Stage.INGESTING, kwargs["stage"])
            job = real_create(*args, **kwargs)
            events.append("db-ingesting")
            return job

        def promote(source_dir: Path, job_dir: Path) -> None:
            real_rename(source_dir, job_dir)
            events.append("durable-rename")

        def update_job(job_id: int, **fields) -> None:
            real_update(job_id, **fields)
            if fields.get("stage") == Stage.QUEUED.value:
                events.append("db-queued")

        with (
            patch.object(ingest_module, "fsync_file", side_effect=fsync_source),
            patch.object(self.store, "create_job", side_effect=create_job),
            patch.object(
                ingest_module,
                "rename_directory_and_fsync",
                side_effect=promote,
            ),
            patch.object(self.store, "_update", side_effect=update_job),
        ):
            job, reason = ingest_file(self.cfg, self.store, source, chat_id=None)

        self.assertIsNone(reason)
        self.assertIsNotNone(job)
        self.assertEqual(
            ["source-fsync", "db-ingesting", "durable-rename", "db-queued"],
            events,
        )
        self.assertEqual(source.read_bytes(), Path(job.source_path).read_bytes())

    def test_watcher_ignores_symlinks_without_notification(self) -> None:
        target = self.root / "outside.txt"
        target.write_text("Текст вне папки.", encoding="utf-8")
        source = self.cfg.paths.books / "linked.txt"
        source.symlink_to(target)
        notifier = _RecordingNotifier()
        wakes: list[bool] = []
        watcher = FolderWatcher(self.cfg, self.store, notifier, lambda: wakes.append(True))

        watcher._scan()
        watcher._scan()

        self.assertEqual([], self.store.list_jobs())
        self.assertEqual([], notifier.notifications)
        self.assertEqual([], wakes)

    def test_watcher_does_not_reingest_terminal_job(self) -> None:
        source = self.cfg.paths.books / "known.txt"
        payload = "Уже обработанный текст книги.".encode()
        source.write_bytes(payload)
        self.store.create_job(
            source.name,
            str(source),
            hashlib.sha256(payload).hexdigest(),
            None,
            Stage.DONE,
        )
        notifier = _RecordingNotifier()
        wakes: list[bool] = []
        watcher = FolderWatcher(self.cfg, self.store, notifier, lambda: wakes.append(True))

        watcher._scan()
        watcher._scan()

        self.assertEqual(1, len(self.store.list_jobs()))
        self.assertEqual([], notifier.notifications)
        self.assertEqual([], wakes)

    def test_watcher_retries_a_transient_ingest_failure_but_reports_it_once(self) -> None:
        (self.cfg.paths.books / "book.txt").write_text("Текст книги.", encoding="utf-8")
        notifier = _RecordingNotifier()
        watcher = FolderWatcher(self.cfg, self.store, notifier, lambda: None)
        failure = (None, "Не удалось принять файл: [Errno 28] No space left on device")

        with patch("app.watcher.ingest_file", return_value=failure) as ingest:
            for _ in range(4):
                watcher._scan()

        self.assertEqual(3, ingest.call_count)  # still retried on every stable poll
        self.assertEqual(1, len(notifier.notifications))

    def test_unknown_extension_sniff_reads_only_prefix(self) -> None:
        path = self.root / "book.custom"
        path.write_bytes("Обычный текст книги. ".encode() + b"x" * (2 * 1024 * 1024))
        reader = _ReadRecorder(path.read_bytes())
        document = Document(None, None, [Chapter(None, "Достаточно длинный текст книги.")])

        with (
            patch.object(Path, "open", return_value=reader),
            patch.object(extract_module, "_extract_txt", return_value=document),
        ):
            result = extract_module.extract(path)

        self.assertIs(result, document)
        self.assertEqual([8192], reader.read_sizes)

    def test_epub_pathological_expansion_is_rejected_before_parser_import(self) -> None:
        path = self.root / "expanded.epub"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("large.xhtml", b"A" * (2 * 1024 * 1024))

        with (
            patch.object(extract_module, "_EPUB_RATIO_CHECK_BYTES", 1024),
            patch.object(extract_module, "_EPUB_MAX_EXPANSION_RATIO", 10),
        ):
            with self.assertRaisesRegex(ExtractError, "коэффициент сжатия"):
                extract_module.extract(path)

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "lxml is not installed")
    def test_nested_fb2_blocks_are_linear_and_keep_inline_prose_order(self) -> None:
        depth = 500
        nested = "Начало " + "".join(f"<p>Уровень {i} " for i in range(depth))
        nested += "центр" + " конец</p>" * depth + " завершение"
        content = f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body><section>
<p>До <emphasis>очень</emphasis> важного.</p><p>{nested}</p>
</section></body></FictionBook>"""
        path = self.root / "nested.fb2"
        path.write_text(content, encoding="utf-8")

        document = extract_module.extract(path)

        self.assertIn("До очень важного.", document.full_text)
        self.assertEqual(depth, document.full_text.count("Уровень "))
        self.assertLess(len(document.full_text), 30_000)
        self.assertLess(document.full_text.index("Начало"), document.full_text.index("центр"))
        self.assertLess(document.full_text.index("центр"), document.full_text.rindex("завершение"))

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "lxml is not installed")
    def test_large_valid_fb2_and_metadata_are_preserved(self) -> None:
        paragraph = "Большой, но обычный абзац художественной прозы. " * 5
        paragraphs = "".join(f"<p>{paragraph}{index}</p>" for index in range(16_000))
        content = f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>Большая книга</book-title>
<author><first-name>Иван</first-name><last-name>Петров</last-name></author>
<sequence name="Большая серия" number="7"/></title-info></description>
<body><section><title><p>Глава первая</p></title>{paragraphs}</section></body>
</FictionBook>"""
        path = self.root / "large.fb2"
        path.write_text(content, encoding="utf-8")

        document = extract_module.extract(path)

        self.assertGreater(document.char_count, 3_000_000)
        self.assertEqual(("Большая книга", "Иван Петров"), (document.title, document.author))
        self.assertEqual(("Большая серия", "7"), (document.series, document.series_index))
        self.assertIn("15999", document.full_text)

    @unittest.skipUnless(
        importlib.util.find_spec("ebooklib")
        and importlib.util.find_spec("lxml")
        and importlib.util.find_spec("bs4"),
        "EPUB parser dependencies are not installed",
    )
    def test_large_highly_compressible_epub_and_metadata_are_accepted(self) -> None:
        prose = "Длинный, но допустимый текст главы с обычными словами. "
        body = (prose * ((6 * 1024 * 1024 // len(prose.encode("utf-8"))) + 1))
        chapter = f"<html><body><h1>Глава первая</h1><p>{body}</p></body></html>"
        path = self.root / "large.epub"
        _write_epub(path, chapter)

        document = extract_module.extract(path)

        self.assertGreater(document.char_count, 3_000_000)
        self.assertEqual(("Большая книга", "Иван Петров"), (document.title, document.author))
        self.assertEqual(("Длинная серия", "12"), (document.series, document.series_index))
        self.assertIn("Длинный, но допустимый текст", document.full_text)

    def test_unauthorized_flood_is_silent_and_log_rate_limited(self) -> None:
        try:
            bot_module = importlib.import_module("app.bot")
        except ImportError as e:
            self.skipTest(f"bot dependencies are not installed: {e}")

        middleware = bot_module.AuthMiddleware(allowed_user_id=7)
        handler = AsyncMock()
        event = SimpleNamespace(answer=AsyncMock())

        async def exercise() -> None:
            for user_id in range(100, 200):
                await middleware(
                    handler,
                    event,
                    {"event_from_user": SimpleNamespace(id=user_id)},
                )
            await middleware(
                handler,
                event,
                {"event_from_user": SimpleNamespace(id=200)},
            )

        with (
            patch.object(
                bot_module,
                "_monotonic",
                side_effect=[100.0] * 100 + [161.0],
            ),
            patch.object(bot_module.log, "warning") as warning,
        ):
            asyncio.run(exercise())

        handler.assert_not_awaited()
        event.answer.assert_not_awaited()
        self.assertEqual(2, warning.call_count)
        self.assertEqual(99, warning.call_args.args[-1])



class ConfigSecretTests(unittest.TestCase):
    TOKEN = "123456789:SECRETSECRETSECRETSECRETSECRETSECRE"

    def test_bot_token_is_secret_in_repr_and_validation_errors(self) -> None:
        from pydantic import ValidationError

        cfg = AppConfig.model_validate(
            {"telegram": {"bot_token": self.TOKEN, "allowed_user_id": 7}}
        )
        self.assertEqual(self.TOKEN, cfg.telegram.bot_token.get_secret_value())
        self.assertNotIn("SECRET", repr(cfg))
        broken = [
            {"telegram": {"bot_token": self.TOKEN}},
            {"telegram": {"bot_token": self.TOKEN, "allowed_user_id": "x"}},
            {"telegram": {"bot_token": self.TOKEN, "allowed_user_id": 7},
             "audiobookshelf": {"public_url": "ftp://x", "token": "SECRET-abs"}},
        ]
        for raw in broken:
            with self.subTest(raw=list(raw)), self.assertRaises(ValidationError) as ctx:
                AppConfig.model_validate(raw)
            self.assertNotIn("SECRET", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
