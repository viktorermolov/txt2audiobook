"""SQLite-backed job state. Single-writer, restart-safe.

A job moves through stages:
    queued -> extracting -> cleaning -> synthesizing -> assembling
          -> ready -> delivering -> done
Any stage may transition to `failed`. Delivery is tracked separately from
conversion: a delivery failure leaves the job in `ready`/`delivered_failed`
state with the M4B intact on disk, never marking conversion as failed.

The DB lives on a mounted volume so progress survives container/Pi restarts.
On startup we reconcile: any job left mid-flight (extracting..assembling) is
reset to `queued` so the worker resumes it from saved intermediate files.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .files import rename_directory_and_fsync

from .log import get_logger

log = get_logger("state")

_MAX_SOURCE_BYTES = 200 * 1024 * 1024


class Stage(str, Enum):
    INGESTING = "ingesting"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    SYNTHESIZING = "synthesizing"
    ASSEMBLING = "assembling"
    READY = "ready"          # M4B built, not yet delivered
    DELIVERING = "delivering"
    DONE = "done"            # delivered (or path reported for oversized)
    FAILED = "failed"
    CANCELLED = "cancelled"


# Stages that represent in-flight conversion work the worker owns.
ACTIVE_STAGES = {
    Stage.EXTRACTING,
    Stage.CLEANING,
    Stage.SYNTHESIZING,
    Stage.ASSEMBLING,
    Stage.DELIVERING,
}

# Human-readable Russian labels for the bot.
STAGE_LABEL_RU = {
    Stage.INGESTING: "приём файла",
    Stage.QUEUED: "🕓 принят, в очереди",
    Stage.EXTRACTING: "📖 извлечение текста",
    Stage.CLEANING: "🧹 очистка текста",
    Stage.SYNTHESIZING: "🗣 синтез речи",
    Stage.ASSEMBLING: "🎚 сборка M4B",
    Stage.READY: "📦 ожидает публикации",
    Stage.DELIVERING: "📚 публикация в библиотеку",
    Stage.DONE: "✅ завершено",
    Stage.FAILED: "❌ ошибка",
    Stage.CANCELLED: "🚫 отменено",
}


@dataclass
class Job:
    id: int
    source_name: str          # original file name as received
    source_path: str          # absolute path to the source file (in work dir)
    content_hash: str         # sha256 of source bytes — dedupe key
    stage: Stage
    progress: float           # 0..1 within the current stage
    title: Optional[str]
    author: Optional[str]
    error: Optional[str]
    output_paths: str         # newline-separated absolute M4B paths
    delivered: int            # 0/1 — whether result reached Telegram
    delivery_error: Optional[str]
    chat_id: Optional[int]    # where to report; null for folder-originated
    created_at: float
    updated_at: float

    @property
    def outputs(self) -> list[str]:
        return [p for p in (self.output_paths or "").splitlines() if p.strip()]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name     TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    stage           TEXT NOT NULL,
    progress        REAL NOT NULL DEFAULT 0,
    title           TEXT,
    author          TEXT,
    error           TEXT,
    output_paths    TEXT NOT NULL DEFAULT '',
    delivered       INTEGER NOT NULL DEFAULT 0,
    delivery_error  TEXT,
    chat_id         INTEGER,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_stage ON jobs(stage);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_hash_active
    ON jobs(content_hash)
    WHERE stage NOT IN ('done', 'failed', 'cancelled');
"""

_TERMINAL = {Stage.DONE, Stage.FAILED, Stage.CANCELLED}


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        source_name=row["source_name"],
        source_path=row["source_path"],
        content_hash=row["content_hash"],
        stage=Stage(row["stage"]),
        progress=row["progress"],
        title=row["title"],
        author=row["author"],
        error=row["error"],
        output_paths=row["output_paths"],
        delivered=row["delivered"],
        delivery_error=row["delivery_error"],
        chat_id=row["chat_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _safe_source_name(path: Path) -> bool:
    """Accept only the source.<suffix> shape created by ingest_file."""
    return bool(path.suffix) and path.name == f"source{path.suffix}"


def _source_matches(path: Path, expected_hash: str) -> bool:
    """Validate a prepared source without loading the whole book into RAM."""
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if path.stat().st_size > _MAX_SOURCE_BYTES:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest() == expected_hash
    except OSError:
        return False


class Store:
    """Synchronous SQLite store. All access is funnelled through the single
    worker/bot process; we use a short-lived connection per call with WAL so
    the bot and worker coroutines don't trip over locks."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ---- duplicate detection ------------------------------------------
    def find_active_by_hash(self, content_hash: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE content_hash=? "
                "AND stage NOT IN ('done','failed','cancelled') "
                "ORDER BY id DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
            return _row_to_job(row) if row else None

    def any_job_by_hash(self, content_hash: str) -> Optional[Job]:
        """Most recent job with this hash in ANY stage — used by the folder
        watcher to avoid re-converting a book that is already done/failed and
        still sitting in books/."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE content_hash=? ORDER BY id DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
            return _row_to_job(row) if row else None

    # ---- creation -----------------------------------------------------
    def create_job(
        self,
        source_name: str,
        source_path: str,
        content_hash: str,
        chat_id: Optional[int],
        stage: Stage = Stage.QUEUED,
    ) -> Job:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source_name, source_path, content_hash, stage, "
                "progress, chat_id, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    source_name,
                    source_path,
                    content_hash,
                    stage.value,
                    0.0,
                    chat_id,
                    now,
                    now,
                ),
            )
            job_id = cur.lastrowid
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return _row_to_job(row)

    # ---- reads --------------------------------------------------------
    def get(self, job_id: int) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return _row_to_job(row) if row else None

    def next_queued(self) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE stage='queued' ORDER BY id ASC LIMIT 1"
            ).fetchone()
            return _row_to_job(row) if row else None

    def active_job(self) -> Optional[Job]:
        """The job currently being worked (non-terminal, non-queued)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE stage IN "
                "('extracting','cleaning','synthesizing','assembling','delivering') "
                "ORDER BY id ASC LIMIT 1"
            ).fetchone()
            return _row_to_job(row) if row else None

    def list_jobs(self, limit: int = 20, *, offset: int = 0, stage: Stage | None = None) -> list[Job]:
        where = " WHERE stage=?" if stage is not None else ""
        params = (stage.value,) if stage is not None else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, max(1, limit), max(0, offset)),
            ).fetchall()
            return [_row_to_job(r) for r in rows]

    def count_jobs(self, stage: Stage | None = None) -> int:
        where = " WHERE stage=?" if stage is not None else ""
        params = (stage.value,) if stage is not None else ()
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM jobs{where}", params).fetchone()[0]

    def queue_depth(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE stage='queued'"
            ).fetchone()
            return row["c"]

    def list_by_stage(self, *stages: Stage) -> list[Job]:
        placeholders = ",".join("?" for _ in stages)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE stage IN ({placeholders}) ORDER BY id ASC",
                tuple(s.value for s in stages),
            ).fetchall()
            return [_row_to_job(r) for r in rows]

    # ---- mutations ----------------------------------------------------
    def _update(self, job_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {cols} WHERE id=?",
                (*fields.values(), job_id),
            )

    def set_stage(
        self, job_id: int, stage: Stage, progress: float = 0.0
    ) -> None:
        self._update(job_id, stage=stage.value, progress=progress)

    def set_progress(self, job_id: int, progress: float) -> None:
        self._update(job_id, progress=max(0.0, min(1.0, progress)))

    def set_meta(
        self, job_id: int, title: Optional[str], author: Optional[str]
    ) -> None:
        self._update(job_id, title=title, author=author)

    def set_outputs(self, job_id: int, paths: list[str]) -> None:
        self._update(job_id, output_paths="\n".join(paths))

    def mark_ready(self, job_id: int, paths: list[str]) -> None:
        self._update(
            job_id,
            stage=Stage.READY.value,
            progress=1.0,
            output_paths="\n".join(paths),
            error=None,
        )

    def mark_failed(self, job_id: int, error: str) -> None:
        self._update(job_id, stage=Stage.FAILED.value, error=error)

    def mark_done(self, job_id: int, delivered: bool) -> None:
        self._update(
            job_id,
            stage=Stage.DONE.value,
            progress=1.0,
            delivered=1 if delivered else 0,
        )

    def mark_cancelled(self, job_id: int) -> None:
        self._update(job_id, stage=Stage.CANCELLED.value)

    def set_delivery_error(self, job_id: int, error: Optional[str]) -> None:
        self._update(job_id, delivery_error=error)

    def set_delivered(self, job_id: int, delivered: bool) -> None:
        fields = {"delivered": int(delivered)}
        if delivered:
            fields["delivery_error"] = None
        self._update(job_id, **fields)

    def claim_queued(self, job_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "UPDATE jobs SET stage='extracting', updated_at=? "
                "WHERE id=? AND stage='queued'", (time.time(), job_id),
            ).rowcount == 1

    def cancel_queued(self, job_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "UPDATE jobs SET stage='cancelled', updated_at=? "
                "WHERE id=? AND stage='queued'", (time.time(), job_id),
            ).rowcount == 1

    # ---- retry / resend ----------------------------------------------
    def requeue(self, job_id: int) -> None:
        """Reset a failed job back to the queue, keeping intermediate files
        so the pipeline resumes rather than restarts."""
        job = self.get(job_id)
        if not job or not Path(job.source_path).is_file():
            raise ValueError("Исходный файл отсутствует. Пришлите книгу заново.")
        try:
            with self._connect() as conn:
                changed = conn.execute(
                    "UPDATE jobs SET stage='queued',error=NULL,progress=0,updated_at=? "
                    "WHERE id=? AND stage IN ('failed','cancelled')", (time.time(), job_id),
                ).rowcount
            if not changed:
                raise ValueError("Задача уже запущена или не требует повтора")
        except sqlite3.IntegrityError as e:
            raise ValueError("Эта книга уже обрабатывается другой задачей") from e

    # ---- startup reconciliation --------------------------------------
    def reconcile_on_startup(self, work_dir: Path | None = None) -> int:
        """Reconcile jobs interrupted by a restart.

        With ``work_dir``, an interrupted ingest is recovered from either its
        staging directory or its final per-job directory after validating the
        source hash. Without it, preserve the legacy behavior and fail such
        jobs so existing callers remain compatible.

        Conversion stages (extracting..assembling) go back to `queued` so the
        worker resumes them from saved intermediate files. A job interrupted
        during DELIVERING had its M4B built already, so it drops to READY — the
        entrypoint auto-resends those instead of re-encoding. Returns the total
        number of ingestion and conversion jobs requeued."""
        now = time.time()
        recovered = 0

        ingesting = self.list_by_stage(Stage.INGESTING)
        if work_dir is None:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET stage='failed',error=?,updated_at=? "
                    "WHERE stage='ingesting'",
                    ("Приём файла прерван. Пришлите книгу заново.", now),
                )
        else:
            for job in ingesting:
                source = self._recover_ingesting_source(job, Path(work_dir))
                if source is None:
                    self._update(
                        job.id,
                        stage=Stage.FAILED.value,
                        error=("Приём файла прерван: сохранённый источник отсутствует "
                               "или повреждён. Пришлите книгу заново."),
                    )
                    continue
                self._update(
                    job.id,
                    source_path=str(source),
                    stage=Stage.QUEUED.value,
                    progress=0.0,
                    error=None,
                )
                recovered += 1
            self._sweep_orphan_uploads(Path(work_dir))

        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET stage='ready', progress=1, delivered=0, updated_at=? "
                "WHERE stage='delivering'",
                (now,),
            )
            cur = conn.execute(
                "UPDATE jobs SET stage='queued', progress=0, updated_at=? "
                "WHERE stage IN "
                "('extracting','cleaning','synthesizing','assembling')",
                (now,),
            )
            recovered += cur.rowcount
        if recovered:
            log.info(
                "Восстановление: %d незавершённых задач возвращено в очередь",
                recovered,
            )
        return recovered

    def _sweep_orphan_uploads(self, work_root: Path) -> None:
        """Remove upload leftovers that no job can ever use.

        Runs at startup under the service lock, before the bot and watcher
        start, so nothing is being written there. A crash between staging and
        job creation leaves a full source copy in ``_staging`` with no job row;
        an interrupted Telegram download leaves a directory in ``_incoming``.
        Staging directories referenced by any job — including failed ingests,
        whose copy is deliberately kept — are never touched."""
        with self._connect() as conn:
            referenced = {
                Path(row[0]).parent.name
                for row in conn.execute("SELECT source_path FROM jobs")
                if row[0] and Path(row[0]).parent.parent.name == "_staging"
            }
        removed = 0
        for area, keep in (("_staging", referenced), ("_incoming", set())):
            root = work_root / area
            if root.is_symlink() or not root.is_dir():
                continue
            for entry in root.iterdir():
                if entry.name in keep:
                    continue
                try:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    removed += 1
                except OSError as error:
                    log.warning("Не удалось удалить остаток загрузки %s: %s", entry, error)
        if removed:
            log.info("Удалено незавершённых загрузок: %d", removed)

    def _recover_ingesting_source(self, job: Job, work_dir: Path) -> Path | None:
        """Return a validated final source, promoting staging atomically."""
        try:
            work_root = Path(work_dir)
            job_dir = work_root / str(job.id)
            recorded = Path(job.source_path)

            expected_name = recorded.name if _safe_source_name(recorded) else None
            if job_dir.is_dir():
                if job_dir.is_symlink():
                    return None
                if expected_name:
                    candidates = [job_dir / expected_name]
                else:
                    candidates = [path for path in job_dir.iterdir()
                                  if _safe_source_name(path)]
                matching = [path for path in candidates
                            if _source_matches(path, job.content_hash)]
                if len(matching) == 1:
                    return matching[0]
                return None

            staging_root = (work_root / "_staging").resolve()
            if (not expected_name or recorded.is_symlink() or
                    recorded.parent.is_symlink()):
                return None
            if recorded.parent.parent.resolve() != staging_root:
                return None
            if not _source_matches(recorded, job.content_hash):
                return None

            job_dir.parent.mkdir(parents=True, exist_ok=True)
            rename_directory_and_fsync(recorded.parent, job_dir)
            return job_dir / expected_name
        except OSError as error:
            log.warning("Не удалось восстановить приём задачи #%d: %s", job.id, error)
            return None

    def ready_undelivered(self) -> list["Job"]:
        """READY jobs with outputs that never reached Telegram — candidates for
        auto-resend after a restart."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE stage='ready' AND delivered=0 "
                "AND output_paths != '' ORDER BY id ASC"
            ).fetchall()
            return [_row_to_job(r) for r in rows]
