"""The conversion worker: one book at a time, restart-safe, resumable.

Runs the blocking pipeline (extract → clean → synth → assemble → deliver) in a
single background thread so the asyncio bot stays responsive. Progress and
stage transitions are persisted to the Store after every step and pushed to
Telegram via the Notifier. Intermediate artifacts live under work/<job_id>/ so
an interrupted job resumes from where it stopped instead of restarting.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .files import write_json
from .library import LibraryPublisher
from .pipeline.cover import extract_cover
from .log import get_logger
from .notify import Notifier
from .pipeline.assemble import AssembleCancelled, AssembleError, assemble
from .pipeline.clean import clean_document
from .pipeline.extract import extract
from .pipeline.synth import (
    SynthCancelled,
    SynthError,
    build_plan,
    ensure_voice,
    load_plan,
    save_plan,
    synthesize_plan,
)
from .pipeline.types import Document, ExtractError, safe_filename
from .state import Stage, Store

log = get_logger("worker")

MIB = 1024 * 1024
# How often the publication scheduler looks for a READY book whose retry is due.
_RETRY_POLL_SEC = 10
# Pause after an unexpected worker-loop error (e.g. SQLite locked) before retrying.
_LOOP_ERROR_BACKOFF_SEC = 10


@dataclass(frozen=True)
class PublishResult:
    status: str
    message: str = ""
    url: str | None = None


class CancelRegistry:
    """Thread-safe set of job ids the user asked to cancel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: set[int] = set()

    def request(self, job_id: int) -> None:
        with self._lock:
            self._ids.add(job_id)

    def is_requested(self, job_id: int) -> bool:
        with self._lock:
            return job_id in self._ids

    def clear(self, job_id: int) -> None:
        with self._lock:
            self._ids.discard(job_id)


class Worker:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        notifier: Notifier,
        loop: asyncio.AbstractEventLoop,
        cancels: CancelRegistry,
    ):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.loop = loop
        self.cancels = cancels
        self.wake = asyncio.Event()
        self._stop = False
        self.publisher = LibraryPublisher(cfg)
        self._publish_lock = threading.Lock()
        self._last_publish: dict[int, float] = {}
        self._retry_inflight: asyncio.Future | None = None

    def request_wake(self) -> None:
        """Call from the loop thread to signal new work."""
        self.wake.set()

    def request_wake_threadsafe(self) -> None:
        self.loop.call_soon_threadsafe(self.wake.set)

    async def run(self) -> None:
        log.info("Воркер запущен")
        # Publication retries run beside conversion: a multi-hour synthesis
        # must not postpone a READY book's retry, which is due 120 s after the
        # previous attempt. _publish_lock already serialises publishers.
        retries = asyncio.create_task(self._publication_retries(), name="publication-retries")
        try:
            while not self._stop:
                try:
                    job = self.store.next_queued()
                    if job is None:
                        self.wake.clear()
                        # Re-check after clearing to avoid a lost-wakeup race.
                        if self.store.next_queued() is None:
                            try:
                                await asyncio.wait_for(self.wake.wait(), timeout=10)
                            except asyncio.TimeoutError:
                                pass
                        continue
                    await self.loop.run_in_executor(None, self._process, job.id)
                except Exception:
                    # A transient store error (locked/full disk) must not end
                    # the worker task and with it the whole service.
                    log.exception("Непредвиденная ошибка цикла воркера")
                    await asyncio.sleep(_LOOP_ERROR_BACKOFF_SEC)
        finally:
            inflight = self._retry_inflight
            if inflight is not None and not inflight.done():
                # Don't let shutdown close Telegram under a running retry. It
                # honours cancel_check=_stop, and amain bounds the whole wait.
                await asyncio.wait({inflight})
            retries.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retries

    async def _publication_retries(self) -> None:
        while not self._stop:
            try:
                pending = None if self._publish_lock.locked() else self._due_ready_job()
                if pending:
                    # Always pause afterwards: an attempt that returns without
                    # publishing (e.g. no outputs) records no timestamp and would
                    # otherwise be picked again in a tight loop.
                    self._retry_inflight = self.loop.run_in_executor(
                        None, self.resend, pending.id
                    )
                    await self._retry_inflight
            except Exception:
                log.exception("Сбой планировщика повторной публикации")
            await asyncio.sleep(_RETRY_POLL_SEC)

    def _due_ready_job(self):
        now = time.monotonic()
        return min(
            (job for job in self.store.ready_undelivered()
             if now - self._last_publish.get(job.id, -1e9)
             >= self.cfg.audiobookshelf.retry_interval_sec),
            key=lambda job: (self._last_publish.get(job.id, -1e9), job.id),
            default=None,
        )

    # ------------------------------------------------------------------
    # Pipeline (runs in executor thread)
    # ------------------------------------------------------------------
    def _process(self, job_id: int) -> None:
        job = self.store.get(job_id)
        if job is None or self._stop or not self.store.claim_queued(job_id):
            return
        chat = job.chat_id or self.cfg.telegram.allowed_user_id
        job_dir = self.cfg.paths.work / str(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)

        ready = False
        try:
            self._check_disk(job_dir)
            doc, base_name = self._extract_clean(job, job_dir, chat)
            plan = self._plan(job, job_dir, doc)
            wavs = self._synth(job, job_dir, plan, chat)
            outputs = self._assemble(job, job_dir, plan, wavs, doc, base_name)
            self._guard_cancel(job.id)
            self.store.mark_ready(job.id, [str(p) for p in outputs])
            ready = True
        except (_Cancelled, SynthCancelled, AssembleCancelled):
            if self._stop and not self.cancels.is_requested(job.id):
                # Startup reconciliation resumes conversion after shutdown. An
                # owner-confirmed cancel is in-memory only, so it is recorded
                # now instead of being lost across the restart.
                return
            self.store.mark_cancelled(job.id)
            self.notifier.notify(f"🚫 Задача #{job.id} отменена.", chat)
            log.info("Задача #%d отменена", job.id)
        except (ExtractError, SynthError, AssembleError, _DiskError) as e:
            self.store.mark_failed(job.id, str(e))
            self.notifier.notify(
                f"❌ Задача #{job.id} («{job.source_name}») не выполнена:\n{e}\n\n"
                f"Повторить: /retry {job.id}", chat,
            )
            log.warning("Задача #%d провалена: %s", job.id, e)
        except Exception as e:  # pragma: no cover - last-resort guard
            self.store.mark_failed(job.id, f"Внутренняя ошибка: {e}")
            self.notifier.notify(
                f"❌ Задача #{job.id}: внутренняя ошибка ({e}). /retry {job.id}",
                chat,
            )
            log.exception("Задача #%d: внутренняя ошибка", job.id)
        if not ready:
            return
        # Conversion succeeded. Publication problems must never mark the job
        # FAILED: its WAVs are about to be removed, so /retry would re-synthesise
        # the whole book. The job stays READY and the scheduler retries it.
        self._cleanup_audio(job.id)
        try:
            self._deliver(job.id, chat)
        except Exception:
            log.exception("Задача #%d: сбой публикации после сборки", job.id)

    # ---- cancellation helpers ----------------------------------------
    def _cancel_check(self, job_id: int) -> bool:
        return self._stop or self.cancels.is_requested(job_id)

    def _guard_cancel(self, job_id: int) -> None:
        if self._cancel_check(job_id):
            raise _Cancelled()

    # ---- stages ------------------------------------------------------
    def _check_disk(self, job_dir: Path) -> None:
        try:
            usage = shutil.disk_usage(job_dir)
        except Exception:
            return
        free_mib = usage.free / MIB
        if free_mib < 300:
            raise _DiskError(
                f"Недостаточно места на диске: свободно {free_mib:.0f} МБ "
                "(нужно минимум ~300 МБ)."
            )

    def _extract_clean(
        self, job, job_dir: Path, chat: int
    ) -> tuple[Document, str]:
        meta_path = job_dir / "meta.json"
        cleaned_path = job_dir / "cleaned.txt"
        plan_path = job_dir / "plan.json"

        # Fast path on resume: cleaned text + plan + meta already exist.
        # All three must parse — a corrupt plan must NOT be silently rebuilt
        # from the single-chapter resume shell, or chunk numbering/chapter
        # marks would no longer match the WAVs already on disk. Falling back
        # to a full re-extract is safe: cleaning is deterministic, so the
        # rebuilt plan matches the existing chunks.
        if cleaned_path.exists() and plan_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                load_plan(plan_path)  # validate, used later by _plan
                doc = _doc_from_cleaned(cleaned_path, plan_path, meta)
                return doc, meta["base_name"]
            except Exception as e:
                log.warning(
                    "Кэш задачи #%d повреждён (%s) — повторяю извлечение",
                    job.id, e,
                )
                plan_path.unlink(missing_ok=True)

        self.store.set_stage(job.id, Stage.EXTRACTING)
        self._guard_cancel(job.id)
        doc = extract(Path(job.source_path))

        overrides = job_dir / "input_meta.json"
        if overrides.exists():
            for key, value in json.loads(overrides.read_text()).items():
                if key in {"title", "author", "series", "series_index"} and isinstance(value, str):
                    setattr(doc, key, value)

        self.store.set_stage(job.id, Stage.CLEANING)
        self._guard_cancel(job.id)
        doc = clean_document(doc)

        title = (doc.title or "").strip() or None
        author = (doc.author or "").strip() or None
        self.store.set_meta(job.id, title, author)
        base_name = _make_base_name(title, author, job.source_name)

        temporary = cleaned_path.with_suffix(".txt.part")
        temporary.write_text(doc.full_text, encoding="utf-8")
        temporary.replace(cleaned_path)
        write_json(meta_path, {"title": title, "author": author, "base_name": base_name,
                              "source_name": job.source_name, "series": doc.series,
                              "series_index": doc.series_index})
        return doc, base_name

    def _plan(self, job, job_dir: Path, doc: Document):
        plan_path = job_dir / "plan.json"
        if plan_path.exists():
            try:
                return load_plan(plan_path)
            except Exception:
                log.warning("Повреждённый plan.json #%d — пересоздаю", job.id)
        plan = build_plan(doc, self.cfg.processing.chunk_chars)
        save_plan(plan, plan_path)
        return plan

    def _synth(self, job, job_dir: Path, plan, chat: int):
        self.store.set_stage(job.id, Stage.SYNTHESIZING)
        self._guard_cancel(job.id)
        model_path = ensure_voice(self.cfg.tts.model, self.cfg.paths.voices)
        audio_dir = job_dir / "audio"

        total = len(plan)

        # Progress is persisted for /status only — no recurring push messages.
        def progress(done: int, tot: int) -> None:
            self.store.set_progress(job.id, done / max(tot, 1))

        silenced: list[int] = []
        wavs = synthesize_plan(
            plan, model_path, audio_dir,
            speaker=self.cfg.tts.speaker,
            sample_rate=self.cfg.tts.sample_rate,
            put_accent=self.cfg.tts.put_accent,
            put_yo=self.cfg.tts.put_yo,
            sentence_silence=self.cfg.tts.sentence_silence,
            threads=self.cfg.tts.threads,
            retries=self.cfg.processing.synth_retries,
            silenced_out=silenced,
            cancel_check=lambda: self._cancel_check(job.id),
            progress_cb=progress,
        )
        if silenced:
            # One-off heads-up (not a recurring update): some fragments could
            # not be voiced and were replaced with short silence.
            self.notifier.notify(
                f"⚠️ #{job.id}: {len(silenced)} из {total} фрагментов не удалось "
                "озвучить — заменены короткой тишиной.", chat,
            )
        write_json(job_dir / "quality.json", {"chunks": total, "silenced": silenced})
        return wavs

    def _assemble(self, job, job_dir: Path, plan, wavs, doc, base_name):
        self.store.set_stage(job.id, Stage.ASSEMBLING)
        self._guard_cancel(job.id)
        cover = extract_cover(Path(job.source_path), job_dir)
        outputs = assemble(
            plan, wavs, self.cfg.paths.audiobook / str(job.id), base_name,
            title=doc.title, author=doc.author,
            bitrate=self.cfg.processing.audio_bitrate,
            sample_rate=self.cfg.processing.audio_sample_rate,
            max_part_mib=None,
            tempo=self.cfg.processing.tempo,
            work_dir=job_dir,
            cover_path=cover,
            cancel_check=lambda: self._cancel_check(job.id),
        )
        return outputs

    # ---- delivery (separate from conversion status) ------------------
    def _deliver(self, job_id: int, chat: int, *, restore: bool = False) -> PublishResult:
        if not self._publish_lock.acquire(blocking=False):
            # A manual /publish can race the scheduled retry. Do not consume
            # the retry interval; the successful publisher wakes us on release.
            return PublishResult("busy", "Публикация уже выполняется. Повторите позже.")
        attempted = False
        try:
            job = self.store.get(job_id)
            if job is None:
                return PublishResult("not_found", "Задача не найдена.")
            if job.stage == Stage.DONE and not restore:
                return PublishResult(
                    "already_done",
                    "Задача уже завершена. Удалённая книга автоматически не возвращается. "
                    f"Для явного восстановления: /restore {job_id}",
                )
            if restore and job.stage != Stage.DONE:
                return PublishResult("not_ready", "Состояние задачи изменилось. Обновите список задач.")
            if not job.outputs or job.stage not in {Stage.READY, Stage.DELIVERING, Stage.DONE}:
                return PublishResult("not_ready", "У задачи нет готового аудио для публикации.")
            attempted = True
            job_dir = self.cfg.paths.work / str(job_id)
            marker = job_dir / "publication.json"
            published = {} if restore or not marker.exists() else json.loads(marker.read_text())
            if restore:
                # Persist notification intent before DELIVERING: recovery may
                # retry this explicit restoration, but never a plain DONE job.
                published["notified"] = False
                write_json(marker, published)
            self.store.set_delivered(job_id, False)
            self.store.set_stage(job_id, Stage.DELIVERING)
            if published.get("item_id") and published.get("url"):
                # A previous attempt already published and ABS indexed the
                # book; only the Telegram notice failed. Re-staging here would
                # resurrect a book the owner has since deleted in the web UI —
                # an explicit /restore clears this marker instead.
                result = published
            else:
                meta = json.loads((job_dir / "meta.json").read_text())
                result = self.publisher.publish(job, meta, cancel_check=lambda: self._stop)
                result["notified"] = published.get("notified", False)
                write_json(marker, result)
            if not result["notified"]:
                quality_note = self._quality_note(job_dir)
                self.notifier.send_message(
                    f"Озвучка #{job_id} завершена{quality_note}: {_short_title(job)}\n"
                    f"Книга добавлена в Audiobookshelf:\n{result['url']}", chat,
                )
                result["notified"] = True
                write_json(marker, result)
            self.store.mark_done(job_id, delivered=True)
            self._cleanup_audio(job_id)
            return PublishResult("delivered", "Книга опубликована в Audiobookshelf.", result["url"])
        except Exception as e:
            if not attempted:
                log.exception("Не удалось проверить публикацию #%d", job_id)
                return PublishResult("failed", "Не удалось проверить состояние публикации.")
            log.warning("Публикация #%d не завершена: %s", job_id, e)
            job = self.store.get(job_id)
            old_error = job.delivery_error if job else None
            self.store.set_stage(job_id, Stage.READY, progress=1.0)
            self.store.set_delivered(job_id, False)
            self.store.set_delivery_error(job_id, str(e))
            if not self._stop and not old_error:
                self.notifier.notify(
                    f"#{job_id}: аудио готово, публикация пока не завершена: {e}. "
                    "Повторю автоматически. /publish для ручного повтора.", chat,
                )
            return PublishResult("failed", f"Аудио сохранено. Публикация будет повторена: {e}")
        finally:
            if attempted:
                self._last_publish[job_id] = time.monotonic()
            self._publish_lock.release()
            self.request_wake_threadsafe()

    @staticmethod
    def _quality_note(job_dir: Path) -> str:
        try:
            quality = json.loads((job_dir / "quality.json").read_text())
            if not isinstance(quality, dict) or not isinstance(quality.get("silenced", []), list):
                return ""
            missing = len(quality.get("silenced", []))
            if missing:
                return f" с пропусками ({missing} фрагм.)"
        except (OSError, ValueError, TypeError):
            pass
        return ""

    def _cleanup_audio(self, job_id: int) -> None:
        """Reclaim disk after delivery: the per-chunk WAVs (often >1 GB for a
        full novel) are no longer needed once the M4B is built and delivered.
        cleaned.txt / meta.json / plan.json are kept for reference and a
        possible future re-assembly."""
        audio_dir = self.cfg.paths.work / str(job_id) / "audio"
        try:
            if audio_dir.exists():
                shutil.rmtree(audio_dir, ignore_errors=True)
                log.info("Очищены промежуточные аудио-чанки задачи #%d", job_id)
        except Exception as e:
            log.warning("Не удалось очистить аудио задачи #%d: %s", job_id, e)

    # ---- on-demand resend (called from bot via loop) -----------------
    def resend(self, job_id: int, *, restore: bool = False) -> PublishResult:
        job = self.store.get(job_id)
        if job is None:
            return PublishResult("not_found", "Задача не найдена.")
        chat = job.chat_id or self.cfg.telegram.allowed_user_id
        return self._deliver(job_id, chat, restore=restore)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

class _Cancelled(Exception):
    pass


class _DiskError(Exception):
    pass


def _short_title(job, limit: int = 300) -> str:
    title = job.title or job.source_name
    return title if len(title) <= limit else title[: limit - 1] + "…"


def _make_base_name(title: Optional[str], author: Optional[str], source_name: str) -> str:
    if title and author:
        raw = f"{author} - {title}"
    elif title:
        raw = title
    else:
        raw = Path(source_name).stem
    return safe_filename(raw, fallback="audiobook")


def _doc_from_cleaned(cleaned_path: Path, plan_path: Path, meta: dict) -> Document:
    """Reconstruct a Document for resume. We only need title/author for the
    assembler; the chapter structure for chapters comes from the saved plan,
    so a single-chapter shell carrying the full text is enough here."""
    from .pipeline.types import Chapter

    text = cleaned_path.read_text(encoding="utf-8")
    return Document(
        title=meta.get("title"),
        author=meta.get("author"),
        chapters=[Chapter(title=None, text=text)],
        series=meta.get("series"),
        series_index=meta.get("series_index"),
    )
