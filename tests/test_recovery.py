from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.state import Stage, Store


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StartupRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.store = Store(self.root / "state" / "jobs.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _ingesting_job(self, source_path: Path, data: bytes):
        return self.store.create_job(
            source_name="book.fb2",
            source_path=str(source_path),
            content_hash=_sha256(data),
            chat_id=7,
            stage=Stage.INGESTING,
        )

    def test_recovers_source_already_promoted_to_job_directory(self) -> None:
        data = b"complete source"
        missing_staged = self.work / "_staging" / "gone" / "source.fb2"
        job = self._ingesting_job(missing_staged, data)
        final_source = self.work / str(job.id) / "source.fb2"
        final_source.parent.mkdir()
        final_source.write_bytes(data)

        recovered = self.store.reconcile_on_startup(self.work)

        updated = self.store.get(job.id)
        self.assertEqual(1, recovered)
        self.assertEqual(Stage.QUEUED, updated.stage)
        self.assertEqual(final_source, Path(updated.source_path))
        self.assertIsNone(updated.error)

    def test_atomically_promotes_valid_staging_directory(self) -> None:
        data = b"prepared source"
        staging = self.work / "_staging" / "upload-1"
        staging.mkdir(parents=True)
        staged_source = staging / "source.epub"
        staged_source.write_bytes(data)
        (staging / "input_meta.json").write_text('{"title":"Book"}', encoding="utf-8")
        job = self._ingesting_job(staged_source, data)

        recovered = self.store.reconcile_on_startup(self.work)

        updated = self.store.get(job.id)
        final_dir = self.work / str(job.id)
        self.assertEqual(1, recovered)
        self.assertEqual(Stage.QUEUED, updated.stage)
        self.assertEqual(final_dir / "source.epub", Path(updated.source_path))
        self.assertFalse(staging.exists())
        self.assertTrue((final_dir / "input_meta.json").is_file())

    def test_rejects_hash_mismatch_without_promoting_source(self) -> None:
        expected = b"expected source"
        staging = self.work / "_staging" / "upload-2"
        staging.mkdir(parents=True)
        staged_source = staging / "source.txt"
        staged_source.write_bytes(b"corrupt source")
        job = self._ingesting_job(staged_source, expected)

        recovered = self.store.reconcile_on_startup(self.work)

        updated = self.store.get(job.id)
        self.assertEqual(0, recovered)
        self.assertEqual(Stage.FAILED, updated.stage)
        self.assertIn("повреждён", updated.error)
        self.assertTrue(staged_source.is_file())
        self.assertFalse((self.work / str(job.id)).exists())

    def test_rejects_staging_file_outside_source_suffix_shape(self) -> None:
        data = b"prepared source"
        staging = self.work / "_staging" / "upload-unsafe"
        staging.mkdir(parents=True)
        staged_source = staging / "source.fb2.part"
        staged_source.write_bytes(data)
        job = self._ingesting_job(staged_source, data)

        recovered = self.store.reconcile_on_startup(self.work)

        self.assertEqual(0, recovered)
        self.assertEqual(Stage.FAILED, self.store.get(job.id).stage)
        self.assertTrue(staged_source.is_file())
        self.assertFalse((self.work / str(job.id)).exists())

    def test_default_call_preserves_failed_ingest_behavior(self) -> None:
        data = b"prepared source"
        staging = self.work / "_staging" / "upload-3"
        staging.mkdir(parents=True)
        staged_source = staging / "source.txt"
        staged_source.write_bytes(data)
        job = self._ingesting_job(staged_source, data)

        recovered = self.store.reconcile_on_startup()

        self.assertEqual(0, recovered)
        self.assertEqual(Stage.FAILED, self.store.get(job.id).stage)
        self.assertTrue(staged_source.is_file())


    def test_startup_removes_only_unreferenced_upload_leftovers(self) -> None:
        data = b"failed ingest keeps its copy"
        kept = self.work / "_staging" / "upload-kept"
        kept.mkdir(parents=True)
        (kept / "source.txt").write_bytes(b"corrupt")
        job = self._ingesting_job(kept / "source.txt", data)  # hash mismatch -> FAILED
        orphan = self.work / "_staging" / "tmp-orphan"  # crashed before create_job
        orphan.mkdir()
        (orphan / "source.epub").write_bytes(b"x" * 1024)
        download = self.work / "_incoming" / "tmp-download"
        download.mkdir(parents=True)
        (download / "book.fb2").write_bytes(b"partial")

        self.store.reconcile_on_startup(self.work)

        self.assertEqual(Stage.FAILED, self.store.get(job.id).stage)
        self.assertTrue((kept / "source.txt").is_file())
        self.assertFalse(orphan.exists())
        self.assertFalse(download.exists())

    def test_interrupted_publication_clears_legacy_delivered_flag(self) -> None:
        job = self.store.create_job("book.txt", "source", "f" * 64, 7, Stage.DELIVERING)
        self.store.set_outputs(job.id, ["book.m4b"])
        self.store.set_delivered(job.id, True)
        self.store.reconcile_on_startup(self.work)
        recovered = self.store.get(job.id)
        self.assertEqual(Stage.READY, recovered.stage)
        self.assertFalse(recovered.delivered)
        self.assertEqual([job.id], [item.id for item in self.store.ready_undelivered()])


if __name__ == "__main__":
    unittest.main()
