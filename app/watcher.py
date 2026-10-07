"""Folder watcher: pick up books dropped into books/ and queue them.

Polling rather than inotify — robust across Docker bind mounts and SD-card
filesystems, and trivially correct on restart. A file is ingested only once it
has been size-stable for one poll interval (so half-copied files are ignored),
and only if no job already exists for its content (so completed books sitting
in books/ are not re-converted after a restart).
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .ingest import ingest_file
from .log import get_logger
from .notify import Notifier
from .state import Store

log = get_logger("watcher")

POLL_INTERVAL = 15  # seconds


class FolderWatcher:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        notifier: Notifier,
        on_new_job: Callable[[], None],
    ):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.on_new_job = on_new_job
        # path -> (size, mtime) from the previous scan, for stability detection.
        self._prev: dict[str, tuple[int, float]] = {}
        # signatures we've already acted on (ingested or rejected) — don't repeat.
        self._handled: set[tuple[str, int, float]] = set()
        # Transient failures are retried every poll, but reported only once.
        self._reported_failures: set[tuple[str, int, float]] = set()

    async def run(self) -> None:
        log.info("Наблюдатель за папкой %s запущен", self.cfg.paths.books)
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._scan)
            except Exception:
                log.exception("Ошибка сканирования папки books")
            await asyncio.sleep(POLL_INTERVAL)

    def _scan(self) -> None:
        books = self.cfg.paths.books
        if not books.exists():
            return
        current: dict[str, tuple[int, float]] = {}
        for entry in sorted(books.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.suffix.lower() == ".part":
                continue
            try:
                st = entry.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            sig = (st.st_size, st.st_mtime)
            key = str(entry)
            current[key] = sig

            handled_key = (key, sig[0], sig[1])
            if handled_key in self._handled:
                continue
            # Require one stable interval before touching the file.
            if self._prev.get(key) != sig:
                continue
            self._handle(entry, handled_key)

        self._prev = current

    def _handle(self, path: Path, handled_key) -> None:
        job, reason = ingest_file(
            self.cfg,
            self.store,
            path,
            chat_id=None,
            include_terminal_duplicates=True,
        )
        if job is not None:
            self._handled.add(handled_key)
            self.notifier.notify(
                f"📥 Из папки принята книга «{path.name}» → задача #{job.id}."
            )
            self.on_new_job()
        elif reason:
            if reason.startswith("Эта книга уже"):
                self._handled.add(handled_key)
                log.debug("Файл %s уже известен", path.name)
                return
            if not reason.startswith("Не удалось принять файл"):
                self._handled.add(handled_key)
            elif handled_key in self._reported_failures:
                return  # Still failing (e.g. disk full); keep retrying quietly.
            else:
                self._reported_failures.add(handled_key)
            self.notifier.notify(f"⚠️ Файл «{path.name}» отклонён: {reason}")
            log.info("Файл %s отклонён: %s", path.name, reason)
