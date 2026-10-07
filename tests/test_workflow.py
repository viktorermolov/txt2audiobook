from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.bot import parse_book_caption
from app.config import AppConfig
from app.ingest import ingest_file
from app.library import LibraryPublisher, PublicationError, book_relative_path
from app.pipeline.types import Chapter, Document
from app.state import Stage, Store
from app.worker import CancelRegistry, Worker, _Cancelled


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int | None]] = []
        self.notifications: list[tuple[str, int | None]] = []

    def send_message(self, text: str, chat_id: int | None = None) -> None:
        self.messages.append((text, chat_id))

    def notify(self, text: str, chat_id: int | None = None) -> None:
        self.notifications.append((text, chat_id))


class IndexedResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


class IndexedClient:
    def __init__(self, *, destination: Path, **_kwargs) -> None:
        self.destination = destination
        self.calls: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def post(self, path: str) -> IndexedResponse:
        self.calls.append(("POST", path))
        return IndexedResponse()

    def get(self, path: str, params: dict) -> IndexedResponse:
        self.calls.append(("GET", path))
        if path.endswith("/items"):
            return IndexedResponse({"results": [{"id": "book-1", "path": str(self.destination)}], "total": 1})
        return IndexedResponse({"media": {"numAudioFiles": 1, "duration": 42}})


class WorkflowTests(unittest.TestCase):
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
                "token": "abs-token",
                "library_id": "library-1",
                "library_path": str(self.root / "library"),
                "scan_timeout_sec": 5,
            },
        })
        self.cfg.ensure_dirs()
        self.store = Store(self.cfg.paths.state / "jobs.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def create_job(self, *, stage: Stage = Stage.QUEUED, content_hash: str = "a" * 64) -> tuple[object, Path]:
        source = self.root / f"source-{content_hash[:4]}.txt"
        source.write_text("Текст книги", encoding="utf-8")
        job = self.store.create_job(source.name, str(source), content_hash, 7, stage)
        return job, source

    def make_worker(self) -> tuple[Worker, RecordingNotifier]:
        notifier = RecordingNotifier()
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        return Worker(self.cfg, self.store, notifier, loop, CancelRegistry()), notifier

    def test_ingest_marks_prepared_source_ingesting_before_queueing(self) -> None:
        incoming = self.root / "incoming.txt"
        incoming.write_text("Полностью подготовленный текст", encoding="utf-8")
        captured: dict[str, object] = {}
        original = self.store.create_job

        def create_job(*args, **kwargs):
            captured["stage"] = kwargs["stage"]
            captured["source"] = Path(kwargs["source_path"])
            captured["bytes"] = Path(kwargs["source_path"]).read_bytes()
            return original(*args, **kwargs)

        with patch.object(self.store, "create_job", side_effect=create_job):
            job, error = ingest_file(self.cfg, self.store, incoming, 7)

        self.assertIsNone(error)
        self.assertIsNotNone(job)
        self.assertEqual(Stage.INGESTING, captured["stage"])
        self.assertEqual(incoming.read_bytes(), captured["bytes"])
        self.assertEqual(Stage.QUEUED, job.stage)
        self.assertEqual(incoming.read_bytes(), Path(job.source_path).read_bytes())
        self.assertFalse(Path(captured["source"]).exists())

    def test_caption_metadata_is_plain_russian_and_persisted_before_queueing(self) -> None:
        incoming = self.root / "caption.txt"
        incoming.write_text("Текст", encoding="utf-8")
        metadata = parse_book_caption(
            "Автор: Иван Петров\nНазвание: Книга\nСерия: Цикл\nНомер: 03"
        )
        captured: dict[str, object] = {}
        original = self.store.create_job

        def create_job(*args, **kwargs):
            staging = Path(kwargs["source_path"]).parent
            captured["stage"] = kwargs["stage"]
            captured["metadata"] = json.loads((staging / "input_meta.json").read_text())
            return original(*args, **kwargs)

        with patch.object(self.store, "create_job", side_effect=create_job):
            job, error = ingest_file(self.cfg, self.store, incoming, 7, metadata)

        self.assertIsNone(error)
        self.assertEqual(
            {"author": "Иван Петров", "title": "Книга", "series": "Цикл", "series_index": "03"},
            metadata,
        )
        self.assertEqual(Stage.INGESTING, captured["stage"])
        self.assertEqual(metadata, captured["metadata"])
        self.assertEqual(metadata, json.loads((self.cfg.paths.work / str(job.id) / "input_meta.json").read_text()))

    def test_claim_and_cancel_are_conditional_race_safe(self) -> None:
        job, _ = self.create_job()
        barrier = threading.Barrier(3)
        results: list[bool] = []

        def attempt(operation) -> None:
            barrier.wait()
            results.append(operation(job.id))

        claim = threading.Thread(target=attempt, args=(self.store.claim_queued,))
        cancel = threading.Thread(target=attempt, args=(self.store.cancel_queued,))
        claim.start()
        cancel.start()
        barrier.wait()
        claim.join()
        cancel.join()

        self.assertEqual([True, False], sorted(results, reverse=True))
        self.assertIn(self.store.get(job.id).stage, {Stage.EXTRACTING, Stage.CANCELLED})

    def test_requeue_has_friendly_missing_source_and_duplicate_errors(self) -> None:
        missing = self.store.create_job("lost.txt", str(self.root / "lost.txt"), "b" * 64, 7, Stage.FAILED)
        with self.assertRaisesRegex(ValueError, "Исходный файл отсутствует"):
            self.store.requeue(missing.id)

        retry, source = self.create_job(stage=Stage.FAILED, content_hash="c" * 64)
        self.store.create_job(source.name, str(source), "c" * 64, 7, Stage.EXTRACTING)
        with self.assertRaisesRegex(ValueError, "Эта книга уже обрабатывается"):
            self.store.requeue(retry.id)
        self.assertEqual(Stage.FAILED, self.store.get(retry.id).stage)

    def test_delivery_flag_false_preserves_error_and_ready_is_not_active(self) -> None:
        ready, _ = self.create_job(stage=Stage.READY, content_hash="d" * 64)
        self.store.set_delivery_error(ready.id, "ABS недоступен")
        self.store.set_delivered(ready.id, False)
        self.assertEqual("ABS недоступен", self.store.get(ready.id).delivery_error)
        self.assertIsNone(self.store.active_job())

    def test_history_is_paginated_after_filtering(self) -> None:
        first, _ = self.create_job(stage=Stage.FAILED, content_hash="11" * 32)
        self.create_job(stage=Stage.DONE, content_hash="22" * 32)
        last, _ = self.create_job(stage=Stage.FAILED, content_hash="33" * 32)
        self.assertEqual(3, self.store.count_jobs())
        self.assertEqual(2, self.store.count_jobs(Stage.FAILED))
        self.assertEqual([last.id], [j.id for j in self.store.list_jobs(limit=1, stage=Stage.FAILED)])
        self.assertEqual([first.id], [j.id for j in self.store.list_jobs(limit=1, offset=1, stage=Stage.FAILED)])

    def test_library_publisher_is_atomic_idempotent_and_waits_for_index(self) -> None:
        job, source = self.create_job(content_hash="e" * 64)
        audio = self.root / "book.m4b"
        audio.write_bytes(b"m4b bytes")
        self.store.set_outputs(job.id, [str(audio)])
        job = self.store.get(job.id)
        meta = {"source_name": source.name, "title": "Книга", "author": "Автор", "series": "Цикл", "series_index": "02"}
        destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, job.content_hash, job.id)
        clients: list[IndexedClient] = []

        def client_factory(**kwargs):
            client = IndexedClient(destination=destination, **kwargs)
            clients.append(client)
            return client

        publisher = LibraryPublisher(self.cfg)
        with patch("app.library.httpx.Client", side_effect=client_factory):
            first = publisher.publish(job, meta)
            before = sorted(path.relative_to(destination) for path in destination.rglob("*"))
            second = publisher.publish(job, meta)

        self.assertEqual(destination, self.cfg.audiobookshelf.library_path / Path("Автор", "Цикл", f"02 - Книга [eeeeeeeeeeee-{job.id}]"))
        self.assertEqual(first, second)
        self.assertEqual("book-1", first["item_id"])
        self.assertEqual(before, sorted(path.relative_to(destination) for path in destination.rglob("*")))
        self.assertTrue((destination / "01.m4b").is_file())
        self.assertTrue((destination / "metadata.opf").is_file())
        self.assertTrue(source.exists())
        self.assertEqual(2, len(clients))

    def test_cover_is_published_atomically_and_checked_on_retry(self) -> None:
        job, source = self.create_job(content_hash="7" * 64)
        audio = self.root / "book.m4b"
        audio.write_bytes(b"audio")
        self.store.set_outputs(job.id, [str(audio)])
        job = self.store.get(job.id)
        directory = self.cfg.paths.work / str(job.id)
        directory.mkdir()
        (directory / "cover.jpg").write_bytes(b"normalized JPEG")
        meta = {"source_name": source.name, "title": "Book"}
        destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, job.content_hash, job.id)
        with patch("app.library.httpx.Client", side_effect=lambda **kwargs: IndexedClient(destination=destination, **kwargs)):
            publisher = LibraryPublisher(self.cfg)
            publisher.publish(job, meta)
            publisher.publish(job, meta)
            receipt = json.loads((destination / ".txt2audiobook.json").read_text())
            self.assertEqual("cover.jpg", receipt["cover"]["name"])
            self.assertEqual(b"normalized JPEG", (destination / "cover.jpg").read_bytes())
            (destination / "cover.jpg").write_bytes(b"changed artwork")
            with self.assertRaisesRegex(PublicationError, "не прошёл проверку"):
                publisher.publish(job, meta)
        self.assertEqual(b"changed artwork", (destination / "cover.jpg").read_bytes())
        self.assertEqual(b"normalized JPEG", (directory / "cover.jpg").read_bytes())

    def test_unreadable_optional_cover_does_not_block_publication(self) -> None:
        job, source = self.create_job(content_hash="6" * 64)
        audio = self.root / "book.m4b"
        audio.write_bytes(b"audio")
        self.store.set_outputs(job.id, [str(audio)])
        job = self.store.get(job.id)
        meta = {"source_name": source.name, "title": "Book"}
        destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, job.content_hash, job.id)
        with (
            patch("app.library.httpx.Client", side_effect=lambda **kwargs: IndexedClient(destination=destination, **kwargs)),
            patch.object(Path, "read_bytes", side_effect=PermissionError("unreadable artwork")),
        ):
            result = LibraryPublisher(self.cfg).publish(job, meta)
        self.assertEqual("book-1", result["item_id"])
        self.assertFalse((destination / "cover.jpg").exists())
        self.assertNotIn("cover", json.loads((destination / ".txt2audiobook.json").read_text()))

    def test_same_source_republished_after_done_uses_a_new_folder_and_audio(self) -> None:
        first, source = self.create_job(content_hash="9" * 64)
        old_audio = self.root / "old.m4b"
        old_audio.write_bytes(b"first render")
        self.store.set_outputs(first.id, [str(old_audio)])
        first = self.store.get(first.id)
        self.store.mark_done(first.id, delivered=True)

        second = self.store.create_job(source.name, str(source), first.content_hash, 7)
        new_audio = self.root / "new.m4b"
        new_audio.write_bytes(b"second render")
        self.store.set_outputs(second.id, [str(new_audio)])
        second = self.store.get(second.id)
        meta = {"source_name": source.name, "title": "Книга", "author": "Автор"}
        first_destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, first.content_hash, first.id)
        second_destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, second.content_hash, second.id)
        destinations = iter([first_destination, second_destination])

        def client_factory(**kwargs):
            return IndexedClient(destination=next(destinations), **kwargs)

        with patch("app.library.httpx.Client", side_effect=client_factory):
            LibraryPublisher(self.cfg).publish(first, meta)
            LibraryPublisher(self.cfg).publish(second, meta)

        self.assertNotEqual(first_destination, second_destination)
        self.assertEqual(b"first render", (first_destination / "01.m4b").read_bytes())
        self.assertEqual(b"second render", (second_destination / "01.m4b").read_bytes())
        self.assertEqual(first.id, json.loads((first_destination / ".txt2audiobook.json").read_text())["job_id"])
        self.assertEqual(second.id, json.loads((second_destination / ".txt2audiobook.json").read_text())["job_id"])

    def test_existing_publication_rejects_changed_rendered_audio(self) -> None:
        job, source = self.create_job(content_hash="8" * 64)
        audio = self.root / "render.m4b"
        audio.write_bytes(b"original render")
        self.store.set_outputs(job.id, [str(audio)])
        job = self.store.get(job.id)
        meta = {"source_name": source.name, "title": "Книга", "author": "Автор"}
        destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, job.content_hash, job.id)

        with patch("app.library.httpx.Client", side_effect=lambda **kwargs: IndexedClient(destination=destination, **kwargs)):
            publisher = LibraryPublisher(self.cfg)
            publisher.publish(job, meta)
            replacement = self.root / "changed.m4b"
            replacement.write_bytes(b"changed render")
            replacement.replace(audio)
            with self.assertRaisesRegex(PublicationError, "не прошёл проверку"):
                publisher.publish(job, meta)

        self.assertEqual(b"original render", (destination / "01.m4b").read_bytes())

    def test_delivery_failure_keeps_audio_and_ready_state(self) -> None:
        job, source = self.create_job(stage=Stage.READY, content_hash="f" * 64)
        audio = self.root / "ready.m4b"
        audio.write_bytes(b"complete")
        self.store.set_outputs(job.id, [str(audio)])
        job_dir = self.cfg.paths.work / str(job.id)
        job_dir.mkdir()
        (job_dir / "meta.json").write_text(json.dumps({"source_name": source.name}), encoding="utf-8")
        worker, notifier = self.make_worker()

        class FailingPublisher:
            def publish(self, *_args, **_kwargs):
                raise PublicationError("индексация не завершена")

        worker.publisher = FailingPublisher()
        worker._deliver(job.id, 7)

        updated = self.store.get(job.id)
        self.assertEqual(Stage.READY, updated.stage)
        self.assertEqual("индексация не завершена", updated.delivery_error)
        self.assertTrue(audio.exists())
        self.assertEqual(1, len(notifier.notifications))

    def test_delivery_lock_and_notification_ledger_prevent_duplicate_publish(self) -> None:
        job, source = self.create_job(stage=Stage.READY, content_hash="1" * 64)
        audio = self.root / "complete.m4b"
        audio.write_bytes(b"complete")
        self.store.set_outputs(job.id, [str(audio)])
        job_dir = self.cfg.paths.work / str(job.id)
        job_dir.mkdir()
        (job_dir / "meta.json").write_text(json.dumps({"source_name": source.name}), encoding="utf-8")
        worker, notifier = self.make_worker()
        entered = threading.Event()
        release = threading.Event()

        class BlockingPublisher:
            calls = 0

            def publish(self, *_args, **_kwargs):
                self.calls += 1
                entered.set()
                release.wait(timeout=2)
                return {"item_id": "book-1", "path": "library/book", "url": "https://abs.example/item/book-1"}

        publisher = BlockingPublisher()
        worker.publisher = publisher
        first = threading.Thread(target=worker._deliver, args=(job.id, 7))
        first.start()
        self.assertTrue(entered.wait(timeout=2))
        worker._deliver(job.id, 7)
        release.set()
        first.join(timeout=2)
        worker._deliver(job.id, 7)

        self.assertEqual(1, publisher.calls)
        self.assertEqual(1, len(notifier.messages))
        self.assertTrue(json.loads((job_dir / "publication.json").read_text())["notified"])
        self.assertEqual(Stage.DONE, self.store.get(job.id).stage)

    def test_cancel_after_assemble_prevents_ready_and_publication(self) -> None:
        job, _ = self.create_job(content_hash="2" * 64)
        output = self.root / "assembled.m4b"
        output.write_bytes(b"complete")
        worker, _ = self.make_worker()
        worker._check_disk = lambda _job_dir: None
        worker._extract_clean = lambda *_args: (Document("Книга", "Автор", [Chapter(None, "Текст")]), "book")
        worker._plan = lambda *_args: []
        worker._synth = lambda *_args: []

        def assemble_then_cancel(*_args):
            worker.cancels.request(job.id)
            return [output]

        worker._assemble = assemble_then_cancel
        with patch.object(worker, "_deliver") as deliver:
            worker._process(job.id)

        self.assertEqual(Stage.CANCELLED, self.store.get(job.id).stage)
        self.assertEqual([], self.store.get(job.id).outputs)
        deliver.assert_not_called()

    def prepare_ready(self, content_hash: str):
        job, source = self.create_job(stage=Stage.READY, content_hash=content_hash)
        audio = self.root / f"ready-{job.id}.m4b"
        audio.write_bytes(b"complete")
        self.store.set_outputs(job.id, [str(audio)])
        directory = self.cfg.paths.work / str(job.id)
        directory.mkdir()
        (directory / "meta.json").write_text(json.dumps({"source_name": source.name}))
        return self.store.get(job.id), directory

    def test_slow_failed_publication_waits_from_finish_and_does_not_starve_next(self) -> None:
        first, _ = self.prepare_ready("5" * 64)
        second, _ = self.prepare_ready("6" * 64)
        worker, _ = self.make_worker()
        now = [1000.0]

        def fail(*_args, **_kwargs):
            now[0] += 181
            raise PublicationError("не завершена индексация")

        worker.publisher = Mock()
        worker.publisher.publish.side_effect = fail
        with patch("app.worker.time.monotonic", side_effect=lambda: now[0]):
            self.assertEqual(first.id, worker._due_ready_job().id)
            self.assertEqual("failed", worker.resend(first.id).status)
            self.assertEqual(now[0], worker._last_publish[first.id])
            self.assertEqual(second.id, worker._due_ready_job().id)
            worker.resend(second.id)
            self.assertEqual(first.id, worker._due_ready_job().id)

    def test_done_never_republishes_without_explicit_restore_even_with_lost_marker(self) -> None:
        job, directory = self.prepare_ready("7" * 64)
        self.store.mark_done(job.id, delivered=True)
        worker, _ = self.make_worker()
        worker.publisher = Mock()
        for content in (None, '{"notified":true}', '{"notified":false}', 'broken'):
            marker = directory / "publication.json"
            if content is not None:
                marker.write_text(content)
            self.assertEqual("already_done", worker.resend(job.id).status)
            self.assertEqual(Stage.DONE, self.store.get(job.id).stage)
        worker.publisher.publish.assert_not_called()
        self.assertIsNone(worker._due_ready_job())

    def test_explicit_restore_republishes_and_sends_fresh_url(self) -> None:
        job, directory = self.prepare_ready("a1" * 32)
        self.store.mark_done(job.id, delivered=True)
        marker = directory / "publication.json"
        marker.write_text('{"notified":true,"url":"https://abs.example/item/old"}')
        worker, notifier = self.make_worker()
        worker.publisher = Mock()

        def publish(*_args, **_kwargs):
            self.assertFalse(json.loads(marker.read_text())["notified"])
            self.assertFalse(self.store.get(job.id).delivered)
            return {"url": "https://abs.example/item/new", "item_id": "new"}

        worker.publisher.publish.side_effect = publish
        result = worker.resend(job.id, restore=True)
        self.assertEqual("delivered", result.status)
        self.assertEqual("https://abs.example/item/new", result.url)
        self.assertEqual(Stage.DONE, self.store.get(job.id).stage)
        self.assertEqual(1, len(notifier.messages))
        self.assertIn("/item/new", notifier.messages[0][0])

    def test_failed_explicit_restore_is_retryable_and_notified_flag_is_reset(self) -> None:
        job, directory = self.prepare_ready("a2" * 32)
        self.store.mark_done(job.id, delivered=True)
        (directory / "publication.json").write_text('{"notified":true}')
        worker, _ = self.make_worker()
        worker.publisher = Mock()
        worker.publisher.publish.side_effect = PublicationError("timeout")
        self.assertEqual("failed", worker.resend(job.id, restore=True).status)
        self.assertEqual(Stage.READY, self.store.get(job.id).stage)
        self.assertFalse(self.store.get(job.id).delivered)
        self.assertFalse(json.loads((directory / "publication.json").read_text())["notified"])

    def test_publish_results_cover_busy_missing_and_stale_restore(self) -> None:
        job, _ = self.prepare_ready("a3" * 32)
        worker, _ = self.make_worker()
        self.assertEqual("not_found", worker.resend(99999).status)
        self.assertEqual("not_ready", worker.resend(job.id, restore=True).status)
        with worker._publish_lock:
            self.assertEqual("busy", worker.resend(job.id).status)
        self.assertNotIn(job.id, worker._last_publish)

    def test_final_notification_keeps_quality_warning_visible(self) -> None:
        job, directory = self.prepare_ready("a4" * 32)
        (directory / "quality.json").write_text('{"chunks":100,"silenced":[2,9]}')
        worker, notifier = self.make_worker()
        worker.publisher = Mock()
        worker.publisher.publish.return_value = {"url": "https://abs.example/item/quality"}
        self.assertEqual("delivered", worker.resend(job.id).status)
        self.assertIn("с пропусками (2 фрагм.)", notifier.messages[0][0])

    def test_untrusted_long_title_cannot_exceed_telegram_message_limit(self) -> None:
        from app.notify import TELEGRAM_TEXT_LIMIT, clip_text

        job, _directory = self.prepare_ready("a5" * 32)
        self.store.set_meta(job.id, "Т" * 5000, None)
        worker, notifier = self.make_worker()
        worker.publisher = Mock()
        worker.publisher.publish.return_value = {"url": "https://abs.example/item/long"}
        self.assertEqual("delivered", worker.resend(job.id).status)
        notice = notifier.messages[0][0]
        self.assertLess(len(notice), 600)
        self.assertIn("https://abs.example/item/long", notice)
        self.assertEqual(TELEGRAM_TEXT_LIMIT, len(clip_text("x" * 10000)))
        self.assertEqual("short", clip_text("short"))

    def test_chapter_metadata_collapses_every_line_break(self) -> None:
        from app.pipeline.assemble import _escape_meta

        self.assertEqual("Глава 1 Начало \\= конец", _escape_meta("Глава 1\rНачало\r\n= конец"))
        self.assertEqual("a b c", _escape_meta("a\rb\u2028c"))

    def test_deleted_library_files_only_return_after_explicit_restore(self) -> None:
        job, directory = self.prepare_ready("a5" * 32)
        meta = json.loads((directory / "meta.json").read_text())
        destination = self.cfg.audiobookshelf.library_path / book_relative_path(meta, job.content_hash, job.id)
        worker, notifier = self.make_worker()
        with patch("app.library.httpx.Client", side_effect=lambda **kw: IndexedClient(destination=destination, **kw)):
            self.assertEqual("delivered", worker.resend(job.id).status)
            shutil.rmtree(destination)
            self.assertIsNone(worker._due_ready_job())
            self.assertEqual("already_done", worker.resend(job.id).status)
            self.assertFalse(destination.exists())
            self.assertEqual("delivered", worker.resend(job.id, restore=True).status)
        self.assertTrue((destination / "01.m4b").is_file())
        self.assertEqual(2, len(notifier.messages))

    def test_worker_assembly_keeps_single_file_without_size_cap(self) -> None:
        job, _ = self.create_job(content_hash="a6" * 32)
        worker, _ = self.make_worker()
        doc = Document("Книга", "Автор", [Chapter(None, "Текст")])
        with patch("app.worker.assemble", return_value=[]) as assemble:
            worker._assemble(job, self.cfg.paths.work / str(job.id), [], [], doc, "book")
        self.assertIsNone(assemble.call_args.kwargs["max_part_mib"])

    def test_successful_pipeline_only_sends_completion(self) -> None:
        job, source = self.create_job()
        source.write_text("Это достаточно длинный текст книги для проверки всех этапов обработки. " * 5)
        worker, notifier = self.make_worker()
        audio = self.root / "book.m4b"
        audio.write_bytes(b"audio")
        with (
            patch("app.worker.ensure_voice", return_value=self.root / "model.pt"),
            patch("app.worker.synthesize_plan", return_value=[]),
            patch("app.worker.assemble", return_value=[audio]),
            patch.object(worker.publisher, "publish", return_value={"item_id": "book", "url": "https://abs.example/item/book"}),
        ):
            worker._process(job.id)
        self.assertEqual(Stage.DONE, self.store.get(job.id).stage)
        self.assertEqual([], notifier.notifications)
        self.assertEqual(1, len(notifier.messages))
        self.assertIn("завершена", notifier.messages[0][0])

    def test_publication_retries_do_not_repeat_error_notifications(self) -> None:
        job, source = self.create_job(stage=Stage.READY)
        audio = self.root / "book.m4b"
        audio.write_bytes(b"audio")
        self.store.set_outputs(job.id, [str(audio)])
        directory = self.cfg.paths.work / str(job.id)
        directory.mkdir()
        (directory / "meta.json").write_text(json.dumps({"source_name": source.name}))
        worker, notifier = self.make_worker()
        with patch.object(worker.publisher, "publish", side_effect=[PublicationError("timeout"), PublicationError("HTTP 503")]):
            worker.resend(job.id)
            worker.resend(job.id)
        self.assertEqual(1, len(notifier.notifications))
        self.assertEqual("HTTP 503", self.store.get(job.id).delivery_error)

    def test_ready_publication_retries_while_a_conversion_is_running(self) -> None:
        ready, _ = self.create_job(stage=Stage.READY, content_hash="3" * 64)
        output = self.root / "ready.m4b"
        output.write_bytes(b"ready")
        self.store.set_outputs(ready.id, [str(output)])
        queued, _ = self.create_job(content_hash="4" * 64)
        worker, _ = self.make_worker()
        published = threading.Event()
        events: list[tuple[str, int]] = []

        def process(job_id: int) -> None:
            events.append(("process-start", job_id))
            # A multi-hour conversion: it only ends after the READY book was
            # published, so the retry cannot have waited for it.
            published.wait(5)
            events.append(("process-end", job_id))
            worker._stop = True

        def resend(job_id: int) -> None:
            events.append(("resend", job_id))
            published.set()

        worker._process = process
        worker.resend = resend

        async def run_worker() -> None:
            worker.loop = asyncio.get_running_loop()
            await asyncio.wait_for(worker.run(), timeout=10)

        asyncio.run(run_worker())

        self.assertTrue(published.is_set())
        self.assertLess(
            events.index(("resend", ready.id)), events.index(("process-end", queued.id))
        )

    def test_shutdown_waits_for_an_in_flight_publication_retry(self) -> None:
        ready, _ = self.create_job(stage=Stage.READY, content_hash="3b" * 32)
        output = self.root / "ready-shutdown.m4b"
        output.write_bytes(b"ready")
        self.store.set_outputs(ready.id, [str(output)])
        worker, _ = self.make_worker()
        started = threading.Event()
        events: list[str] = []

        def resend(_job_id: int) -> None:
            started.set()
            time.sleep(0.5)
            events.append("resend-finished")

        worker.resend = resend

        async def run_worker() -> None:
            worker.loop = asyncio.get_running_loop()
            task = asyncio.create_task(worker.run())
            await worker.loop.run_in_executor(None, started.wait, 5)
            worker._stop = True
            worker.request_wake()
            await asyncio.wait_for(task, timeout=10)
            events.append("run-returned")

        asyncio.run(run_worker())
        self.assertEqual(["resend-finished", "run-returned"], events)

    def test_worker_loop_survives_transient_store_errors(self) -> None:
        queued, _ = self.create_job(content_hash="5a" * 32)
        worker, _ = self.make_worker()
        calls = {"n": 0}
        processed: list[int] = []

        def next_queued():
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            worker._stop = True
            return queued

        worker._process = processed.append

        async def run_worker() -> None:
            worker.loop = asyncio.get_running_loop()
            await asyncio.wait_for(worker.run(), timeout=10)

        with (
            patch.object(self.store, "next_queued", side_effect=next_queued),
            patch("app.worker._LOOP_ERROR_BACKOFF_SEC", 0),
        ):
            asyncio.run(run_worker())
        self.assertEqual([queued.id], processed)

    def test_confirmed_cancel_is_recorded_even_during_shutdown(self) -> None:
        cancelled, _ = self.create_job(content_hash="c1" * 32)
        resumable, _ = self.create_job(content_hash="c2" * 32)

        worker, _ = self.make_worker()

        def stop_after_confirmed_cancel(_job_dir):
            worker.cancels.request(cancelled.id)
            worker._stop = True
            raise _Cancelled()

        with patch.object(worker, "_check_disk", side_effect=stop_after_confirmed_cancel):
            worker._process(cancelled.id)
        self.assertEqual(Stage.CANCELLED, self.store.get(cancelled.id).stage)

        worker, _ = self.make_worker()

        def plain_shutdown(_job_dir):
            worker._stop = True
            raise _Cancelled()

        with patch.object(worker, "_check_disk", side_effect=plain_shutdown):
            worker._process(resumable.id)
        # Left for startup reconciliation to resume, not cancelled.
        self.assertNotIn(
            self.store.get(resumable.id).stage, {Stage.CANCELLED, Stage.FAILED}
        )

    def test_notice_retry_after_indexed_publication_does_not_republish(self) -> None:
        job, _ = self.prepare_ready("c3" * 32)
        worker, notifier = self.make_worker()
        worker.publisher = Mock()
        worker.publisher.publish.return_value = {
            "item_id": "book-9", "path": "/library/book", "url": "https://abs.example/item/book-9",
        }
        sent: list[str] = []

        def flaky_send(text: str, chat_id: int | None = None) -> None:
            sent.append(text)
            if len(sent) == 1:
                raise RuntimeError("flood control")

        notifier.send_message = flaky_send
        self.assertEqual("failed", worker.resend(job.id).status)
        self.assertEqual(Stage.READY, self.store.get(job.id).stage)
        # The owner may delete the book in ABS meanwhile; the retry only
        # re-sends the notice and must not stage the files again.
        self.assertEqual("delivered", worker.resend(job.id).status)
        self.assertEqual(1, worker.publisher.publish.call_count)
        self.assertIn("https://abs.example/item/book-9", sent[-1])
        self.assertEqual(Stage.DONE, self.store.get(job.id).stage)

    def test_publication_error_after_assembly_keeps_job_ready(self) -> None:
        job, source = self.create_job(content_hash="c4" * 32)
        source.write_text("Это достаточно длинный текст книги для проверки всех этапов. " * 5)
        worker, _ = self.make_worker()
        audio = self.root / "book-ready.m4b"
        audio.write_bytes(b"audio")
        with (
            patch("app.worker.ensure_voice", return_value=self.root / "model.pt"),
            patch("app.worker.synthesize_plan", return_value=[]),
            patch("app.worker.assemble", return_value=[audio]),
            patch.object(worker, "_deliver", side_effect=sqlite3.OperationalError("disk I/O error")),
        ):
            worker._process(job.id)
        self.assertEqual(Stage.READY, self.store.get(job.id).stage)


if __name__ == "__main__":
    unittest.main()
