"""Shared ingestion: turn an incoming file into a queued Job (dedup-aware).

Used by both the Telegram upload handler and the folder watcher. The source
file is copied into a per-job work directory so the original (in books/) stays
untouched and the job is self-contained for restart-safe resume.
"""

from __future__ import annotations

import hashlib
import errno
import os
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .files import fsync_file, rename_directory_and_fsync, write_json
from .log import get_logger
from .pipeline.extract import ALL_EXTS
from .state import Job, Stage, Store

log = get_logger("ingest")

# Accept these even if not in ALL_EXTS — extractor will sniff text-likeness.
_KNOWN_EXTS = ALL_EXTS | {".text", ""}
_MAX_SOURCE_BYTES = 200 * 1024 * 1024
_COPY_BLOCK_BYTES = 1 << 20

# Kept as a binding so the descriptor-open race can be exercised without
# replacing os.open process-wide in tests.
_open_source = os.open


class IngestRejected(Exception):
    """The file was rejected before a job was created (Russian message)."""


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a final-component symlink."""
    h = hashlib.sha256()
    fd, _ = _open_regular_source(Path(path))
    with os.fdopen(fd, "rb") as f:
        for block in iter(lambda: f.read(_COPY_BLOCK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _open_regular_source(path: Path) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = _open_source(path, flags)
    except FileNotFoundError as e:
        raise IngestRejected("Файл не найден") from e
    except OSError as e:
        if e.errno in {errno.ELOOP, errno.EMLINK}:
            raise IngestRejected("Символические ссылки не принимаются") from e
        raise IngestRejected(f"Файл не удалось открыть: {e.strerror or e}") from e

    try:
        source_stat = os.fstat(fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise IngestRejected("Источник должен быть обычным файлом")
        if source_stat.st_size == 0:
            raise IngestRejected("Файл пустой")
        if source_stat.st_size > _MAX_SOURCE_BYTES:
            raise IngestRejected("Файл слишком большой (более 200 МБ) — вряд ли это книга")
        return fd, source_stat
    except Exception:
        os.close(fd)
        raise


def _copy_and_hash(fd: int, source_stat: os.stat_result, destination: Path) -> str:
    """Copy and hash the already-open inode, rejecting growth during the copy."""
    digest = hashlib.sha256()
    copied = 0
    with os.fdopen(fd, "rb") as source, destination.open("xb") as output:
        for block in iter(lambda: source.read(_COPY_BLOCK_BYTES), b""):
            copied += len(block)
            if copied > _MAX_SOURCE_BYTES:
                raise IngestRejected("Файл слишком большой (более 200 МБ) — вряд ли это книга")
            output.write(block)
            digest.update(block)
        final_stat = os.fstat(source.fileno())

    if copied == 0:
        raise IngestRejected("Файл пустой")
    if (final_stat.st_size, final_stat.st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise IngestRejected("Файл изменился во время приёма — повторите попытку")
    return digest.hexdigest()


def ingest_file(
    cfg: AppConfig,
    store: Store,
    src: Path,
    chat_id: Optional[int],
    metadata: dict | None = None,
    *,
    include_terminal_duplicates: bool = False,
) -> tuple[Optional[Job], Optional[str]]:
    """Create a queued job for `src`.

    Returns (job, None) on success, or (None, reason) if rejected/duplicate.
    Never raises for ordinary bad input — a bad book must not crash the service.
    """
    src = Path(src)
    job: Optional[Job] = None
    staging: Path | None = None
    source_fd: int | None = None
    try:
        ext = src.suffix.lower()
        if ext and ext not in _KNOWN_EXTS:
            # Unknown extension is allowed only if it sniffs as text; let the
            # extractor decide later, but warn early for obvious binaries.
            if ext in {".pdf", ".doc", ".docx", ".zip", ".mobi", ".djvu",
                       ".jpg", ".png", ".mp3", ".m4b", ".exe"}:
                return None, f"Формат «{ext}» не поддерживается"

        # Open before creating staging state. O_NOFOLLOW and fstat bind all
        # validation/copying to this inode even if the path is swapped later.
        source_fd, source_stat = _open_regular_source(src)
        staging_root = cfg.paths.work / "_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=staging_root))
        prepared = staging / f"source{ext or '.txt'}"
        copy_fd = source_fd
        source_fd = None
        content_hash = _copy_and_hash(copy_fd, source_stat, prepared)
        fsync_file(prepared)
        if metadata:
            write_json(staging / "input_meta.json", metadata)

        # Dedupe: skip if an active or finished-successfully job exists.
        find_duplicate = (
            store.any_job_by_hash if include_terminal_duplicates
            else store.find_active_by_hash
        )
        existing = find_duplicate(content_hash)
        if existing:
            return None, (
                f"Эта книга уже в работе (#{existing.id}, "
                f"{existing.stage.value}). Дубликат пропущен."
            )

        try:
            job = store.create_job(
                source_name=src.name, source_path=str(prepared),
                content_hash=content_hash, chat_id=chat_id, stage=Stage.INGESTING,
            )
        except sqlite3.IntegrityError:
            return None, "Эта книга уже принимается или находится в работе"

        # Copy the source into the per-job work dir.
        job_dir = cfg.paths.work / str(job.id)
        rename_directory_and_fsync(staging, job_dir)
        staging = None
        dest = job_dir / f"source{ext or '.txt'}"
        store._update(job.id, source_path=str(dest), stage=Stage.QUEUED.value)
        job = store.get(job.id)
        log.info("Принята книга «%s» как задача #%d", src.name, job.id)
        return job, None
    except IngestRejected as e:
        return None, str(e)
    except Exception as e:  # pragma: no cover - defensive
        if job is not None:
            try:
                store.mark_failed(job.id, f"Не удалось принять файл: {e}")
            except Exception:
                log.exception("Не удалось пометить задачу #%d как ошибочную", job.id)
        log.exception("Ошибка приёма файла %s", src)
        return None, f"Не удалось принять файл: {e}"
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
