from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig
from app.files import rename_directory_and_fsync, replace_and_fsync, write_json
from app.locking import ServiceAlreadyRunningError, ServiceLock
from app.state import Stage, Store
from app.worker import CancelRegistry, Worker


class RecordingNotifier:
    def send_message(self, *_args, **_kwargs) -> None:
        pass

    def notify(self, *_args, **_kwargs) -> None:
        pass


class DurabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_second_process_cannot_acquire_service_lock(self) -> None:
        path = self.root / "state" / "service.lock"
        code = """
import sys
from pathlib import Path
from app.locking import ServiceLock

lock = ServiceLock(Path(sys.argv[1]))
lock.acquire()
print('ready', flush=True)
try:
    sys.stdin.read()
finally:
    lock.close()
"""
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        holder = subprocess.Popen(
            [sys.executable, "-c", code, str(path)],
            cwd=project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready, _, _ = select.select([holder.stdout], [], [], 15)
            self.assertTrue(ready, "дочерний процесс не сообщил о готовности за 15 секунд")
            self.assertEqual("ready\n", holder.stdout.readline())
            with self.assertRaisesRegex(ServiceAlreadyRunningError, "Сервис уже запущен"):
                ServiceLock(path).acquire()
        finally:
            try:
                _, stderr = holder.communicate(input="", timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                _, stderr = holder.communicate()
                self.fail(f"дочерний процесс блокировки не завершился: {stderr}")
        self.assertEqual(0, holder.returncode, stderr)

    def test_service_lock_rejects_symlinks_and_closes_on_flock_error(self) -> None:
        target = self.root / "target"
        target.write_text("not a lock", encoding="utf-8")
        link = self.root / "state" / "service.lock"
        link.parent.mkdir()
        link.symlink_to(target)

        with self.assertRaisesRegex(ServiceAlreadyRunningError, "безопасно открыть"):
            ServiceLock(link).acquire()
        self.assertTrue(link.is_symlink())
        self.assertEqual("not a lock", target.read_text(encoding="utf-8"))

        fifo = self.root / "state" / "service.fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ServiceAlreadyRunningError, "обычный файл"):
            ServiceLock(fifo).acquire()

        lock = ServiceLock(self.root / "state" / "flock-error.lock")
        with (
            patch("app.locking.fcntl.flock", side_effect=OSError("flock failed")),
            patch("app.locking.os.close", wraps=os.close) as close,
        ):
            with self.assertRaisesRegex(ServiceAlreadyRunningError, "Не удалось захватить"):
                lock.acquire()
        close.assert_called_once()
        self.assertIsNone(lock._fd)

    def test_json_replace_fsyncs_payload_before_directory_entry(self) -> None:
        path = self.root / "state" / "record.json"
        events: list[str] = []
        real_replace = os.replace

        def tracked_replace(source, destination) -> None:
            events.append("replace")
            real_replace(source, destination)

        with (
            patch("app.files.os.fsync", side_effect=lambda _fd: events.append("payload")),
            patch("app.files.os.replace", side_effect=tracked_replace),
            patch("app.files.fsync_directory", side_effect=lambda _path: events.append("directory")),
        ):
            write_json(path, {"ready": True})

        self.assertEqual(["payload", "replace", "directory"], events)
        self.assertEqual({"ready": True}, json.loads(path.read_text(encoding="utf-8")))

    def test_synthesis_records_are_fsynced_before_replacement(self) -> None:
        from app.pipeline import synth

        path = self.root / "work" / "synthesis-cache.json"
        with patch("app.pipeline.synth.replace_and_fsync", wraps=synth.replace_and_fsync) as durable:
            synth._atomic_write_text(path, '{"ok": true}')

        durable.assert_called_once()
        self.assertEqual(path, durable.call_args.args[1])
        self.assertEqual('{"ok": true}', path.read_text(encoding="utf-8"))
        self.assertEqual([path], list(path.parent.iterdir()))

    def test_rename_helpers_persist_payload_before_both_directory_entries(self) -> None:
        source = self.root / "output.part"
        destination = self.root / "output.m4b"
        source.write_bytes(b"complete m4b")
        events: list[str] = []
        real_replace = os.replace

        def tracked_replace(src, dst) -> None:
            events.append("replace")
            real_replace(src, dst)

        with (
            patch("app.files.fsync_file", side_effect=lambda _path: events.append("payload")),
            patch("app.files.os.replace", side_effect=tracked_replace),
            patch("app.files.fsync_directory", side_effect=lambda _path: events.append("directory")),
        ):
            replace_and_fsync(source, destination)

        self.assertEqual(["payload", "replace", "directory"], events)
        self.assertEqual(b"complete m4b", destination.read_bytes())

        staging_parent = self.root / ".staging"
        staging = staging_parent / "1"
        final_parent = self.root / "library" / "Author"
        final = final_parent / "Book"
        staging.mkdir(parents=True)
        final_parent.mkdir(parents=True)
        events.clear()
        real_rename = os.rename

        def tracked_rename(src, dst) -> None:
            events.append("rename")
            real_rename(src, dst)

        with (
            patch("app.files.os.rename", side_effect=tracked_rename),
            patch("app.files.fsync_directory", side_effect=lambda _path: events.append("directory")),
        ):
            rename_directory_and_fsync(staging, final)

        self.assertEqual(["rename", "directory", "directory"], events)
        self.assertTrue(final.is_dir())

    def test_busy_manual_publish_keeps_next_retry_eligible_and_wakes_worker(self) -> None:
        cfg = AppConfig.model_validate({
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
            },
        })
        cfg.ensure_dirs()
        store = Store(cfg.paths.state / "jobs.db")
        source = self.root / "book.txt"
        source.write_text("text", encoding="utf-8")
        first = store.create_job(source.name, str(source), "1" * 64, 7, Stage.READY)
        second = store.create_job(source.name, str(source), "2" * 64, 7, Stage.READY)
        for job in (first, second):
            audio = self.root / f"{job.id}.m4b"
            audio.write_bytes(b"complete")
            store.set_outputs(job.id, [str(audio)])
            job_dir = cfg.paths.work / str(job.id)
            job_dir.mkdir()
            write_json(job_dir / "meta.json", {"source_name": source.name})

        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        worker = Worker(cfg, store, RecordingNotifier(), loop, CancelRegistry())
        entered = threading.Event()
        release = threading.Event()

        class BlockingPublisher:
            calls: list[int] = []

            def publish(self, job, *_args, **_kwargs) -> dict:
                self.calls.append(job.id)
                if job.id == first.id:
                    entered.set()
                    release.wait(5)
                return {"item_id": str(job.id), "path": "library/book", "url": "https://abs.example/item/book"}

        publisher = BlockingPublisher()
        worker.publisher = publisher
        first_thread = threading.Thread(target=worker._deliver, args=(first.id, 7))
        with patch.object(worker, "request_wake_threadsafe") as wake:
            first_thread.start()
            self.assertTrue(entered.wait(5))
            worker._deliver(second.id, 7)
            self.assertNotIn(second.id, worker._last_publish)
            self.assertEqual([first.id], publisher.calls)
            release.set()
            first_thread.join(5)
            self.assertFalse(first_thread.is_alive())
            self.assertTrue(wake.called)

        worker._deliver(second.id, 7)
        self.assertEqual([first.id, second.id], publisher.calls)
        self.assertEqual(Stage.DONE, store.get(second.id).stage)


if __name__ == "__main__":
    unittest.main()
