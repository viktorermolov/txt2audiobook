"""Telegram interface for the single service owner."""

from __future__ import annotations

import asyncio
import html
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from aiogram import BaseMiddleware, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
)

from .config import AppConfig
from .ingest import ingest_file
from .log import get_logger
from .notify import Notifier
from .state import STAGE_LABEL_RU, Job, Stage, Store
from .worker import Worker

log = get_logger("bot")

MIB = 1024 * 1024
_UNAUTHORIZED_LOG_INTERVAL_SEC = 60.0
_MAX_SUPPRESSED_REJECTIONS = 1_000_000
_CONFIRM_TTL_SEC = 120.0
_QUEUE_PAGE_SIZE = 7
_monotonic = time.monotonic

_CANCELLABLE = {
    Stage.QUEUED,
    Stage.EXTRACTING,
    Stage.CLEANING,
    Stage.SYNTHESIZING,
    Stage.ASSEMBLING,
}
_QUEUE_FILTERS = {stage.value: stage for stage in Stage}


@dataclass
class BotContext:
    cfg: AppConfig
    store: Store
    notifier: Notifier
    worker: Worker
    loop: asyncio.AbstractEventLoop


@dataclass
class PendingAction:
    user_id: int
    chat_id: int
    job_id: int
    action: str
    expires_at: float
    expected_updated_at: float | None


class AuthMiddleware(BaseMiddleware):
    """Allow only the configured user in their own private chat."""

    def __init__(self, allowed_user_id: int):
        self.allowed = allowed_user_id
        self._last_rejection_log: float | None = None
        self._suppressed_rejections = 0

    async def __call__(self, handler, event: TelegramObject, data: dict):
        user = data.get("event_from_user") or getattr(event, "from_user", None)
        message = getattr(event, "message", None)
        chat = getattr(event, "chat", None) or getattr(message, "chat", None)
        allowed = (
            user is not None
            and user.id == self.allowed
            and chat is not None
            and getattr(chat, "type", None) == "private"
            and chat.id == self.allowed
        )
        if allowed:
            return await handler(event, data)

        uid = getattr(user, "id", "?")
        self._log_rejection(uid)
        if isinstance(event, CallbackQuery):
            try:
                await event.answer("Доступ только владельцу в личном чате.", show_alert=True)
            except Exception:
                log.warning("Не удалось подтвердить отклонённый callback", exc_info=True)
        return None

    def _log_rejection(self, uid: int | str) -> None:
        now = _monotonic()
        if (
            self._last_rejection_log is None
            or now - self._last_rejection_log >= _UNAUTHORIZED_LOG_INTERVAL_SEC
        ):
            log.warning(
                "Отклонён неавторизованный пользователь %s (подавлено: %d)",
                uid,
                self._suppressed_rejections,
            )
            self._last_rejection_log = now
            self._suppressed_rejections = 0
        else:
            self._suppressed_rejections = min(
                self._suppressed_rejections + 1,
                _MAX_SUPPRESSED_REJECTIONS,
            )


HELP = (
    "<b>Генератор аудиокниг</b>\n\n"
    "Пришлите текстовый файл (TXT, EPUB, FB2, RTF, HTML, MD) — "
    "и я озвучу его и добавлю в Audiobookshelf.\n"
    "Можно также класть книги в папку <code>books/</code>.\n\n"
    "<b>Команды</b>\n"
    "/status — что происходит сейчас\n"
    "/queue [страница] [статус] — задачи и действия\n"
    "/cancel [id] — отменить активную или ожидающую задачу\n"
    "/retry [id] — повторить упавшую задачу\n"
    "/publish [id] — повторить публикацию готовой книги\n"
    "/restore id — заново опубликовать завершённую книгу\n"
    "/voice — текущие настройки синтеза\n"
    "/library — открыть библиотеку\n"
    "\nВ подписи к файлу можно задать строки: Автор:, Название:, Серия:, Номер:.\n"
)


def parse_book_caption(caption: str | None) -> dict[str, str]:
    keys = {"автор": "author", "название": "title", "серия": "series", "номер": "series_index"}
    values = {}
    for line in (caption or "").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in keys and value.strip():
            values[keys[key.strip().lower()]] = value.strip()[:200]
    return values


def _fit_filename(name: str, limit: int = 200) -> str:
    """Keep a Telegram file name within filesystem limits (255 bytes).

    Long Cyrillic names exceed it quickly (2 bytes per letter) and the download
    would fail with ENAMETOOLONG on every retry. The extension is preserved
    because ingestion picks the extractor by suffix."""
    if len(name.encode("utf-8")) <= limit:
        return name
    path = Path(name)
    suffix = path.suffix if len(path.suffix.encode("utf-8")) <= 16 else ""
    stem = name[: -len(suffix)] if suffix else name
    budget = limit - len(suffix.encode("utf-8"))
    stem = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(" .")
    return (stem or "upload") + suffix


def _fmt_job_line(job: Job) -> str:
    label = STAGE_LABEL_RU.get(job.stage, job.stage.value)
    extra = f" {int(job.progress * 100)}%" if job.stage == Stage.SYNTHESIZING else ""
    name = _escaped_excerpt(job.title or job.source_name, 180)
    return f"#{job.id} {label}{extra} — {name}"


def _escaped_excerpt(value: str, limit: int) -> str:
    """Truncate after escaping without cutting an HTML entity in half."""
    result: list[str] = []
    used = 0
    for char in value:
        escaped = html.escape(char)
        if used + len(escaped) > limit:
            return "".join(result) + "..."
        result.append(escaped)
        used += len(escaped)
    return "".join(result)


def _job_action(job: Job) -> tuple[str, str] | None:
    if job.stage in _CANCELLABLE:
        return "cancel", f"Отменить #{job.id}"
    if job.stage == Stage.FAILED:
        return "retry", f"Повторить #{job.id}"
    if job.stage == Stage.READY and job.outputs:
        return "publish", f"Опубликовать #{job.id}"
    if job.stage == Stage.DONE and job.outputs:
        return "restore", f"Восстановить #{job.id}"
    return None


def _job_keyboard(
    jobs: list[Job], *, page: int | None = None, total_pages: int | None = None,
    stage: Stage | None = None,
) -> InlineKeyboardMarkup | None:
    rows = []
    for job in jobs:
        action = _job_action(job)
        if action:
            name, label = action
            rows.append([InlineKeyboardButton(text=label, callback_data=f"job:{job.id}:{name}")])
    if page is not None and total_pages is not None:
        scope = stage.value if stage else "all"
        navigation = []
        if page > 1:
            navigation.append(InlineKeyboardButton(text="Назад", callback_data=f"queue:{page - 1}:{scope}"))
        if page < total_pages:
            navigation.append(InlineKeyboardButton(text="Далее", callback_data=f"queue:{page + 1}:{scope}"))
        if navigation:
            rows.append(navigation)
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def _voice_text(ctx: BotContext) -> str:
    tts = ctx.cfg.tts
    processing = ctx.cfg.processing
    return (
        "<b>Текущие настройки озвучки</b>\n"
        f"Модель: <code>{html.escape(tts.model)}</code>\n"
        f"Голос: <code>{html.escape(tts.speaker)}</code>\n"
        f"Частота синтеза: {tts.sample_rate} Гц\n"
        f"Автоударения: {'включены' if tts.put_accent else 'выключены'}\n"
        f"Ё: {'включена' if tts.put_yo else 'выключена'}\n"
        f"Пауза после фрагмента: {tts.sentence_silence:g} с\n"
        f"Темп: {processing.tempo:g}x\n"
        f"Размер фрагмента: {processing.chunk_chars} символов\n"
        f"Частота итогового аудио: {processing.audio_sample_rate} Гц\n"
        f"Битрейт: {html.escape(processing.audio_bitrate)}"
    )


def _parse_job_id(command: CommandObject) -> int | None:
    if not command.args:
        return None
    try:
        value = int(command.args.strip().split()[0])
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_target(ctx: BotContext, command: CommandObject, prefer: str) -> Job | None:
    """Resolve an explicit id, otherwise select the most relevant job."""
    job_id = _parse_job_id(command)
    if job_id is not None:
        return ctx.store.get(job_id)
    if command.args:
        return None
    if prefer == "cancel":
        active = ctx.store.active_job()
        if active:
            return active
        queued = ctx.store.list_by_stage(Stage.QUEUED)
        return queued[0] if queued else None
    if prefer == "failed":
        return next(iter(ctx.store.list_jobs(limit=1, stage=Stage.FAILED)), None)
    if prefer == "ready":
        return next((job for job in ctx.store.list_by_stage(Stage.READY) if job.outputs), None)
    return None


def _action_problem(job: Job | None, action: str) -> str | None:
    if job is None:
        return "Задача не найдена."
    if action == "cancel":
        if job.stage == Stage.DELIVERING:
            return f"Публикация задачи #{job.id} уже выполняется. Отменить её нельзя."
        if job.stage not in _CANCELLABLE:
            return f"Задачу #{job.id} ({STAGE_LABEL_RU.get(job.stage, job.stage.value)}) нельзя отменить."
    elif action == "retry":
        if job.stage not in {Stage.FAILED, Stage.CANCELLED}:
            return f"Повтор доступен только для упавшей или отменённой задачи #{job.id}."
    elif action == "publish":
        if job.stage != Stage.READY or not job.outputs:
            return f"Задача #{job.id} не ожидает публикации."
    elif action == "restore":
        if job.stage != Stage.DONE or not job.outputs:
            return "Восстановить можно только завершённую задачу с готовым M4B."
    else:
        return "Неизвестное действие."
    return None


def _queue_options(args: str | None) -> tuple[int, Stage | None] | None:
    page = 1
    stage = None
    for token in (args or "").lower().split():
        if token.isdecimal() and page == 1:
            page = int(token)
        elif token in _QUEUE_FILTERS and stage is None:
            stage = _QUEUE_FILTERS[token]
        elif token != "all":
            return None
    return (page, stage) if 1 <= page <= 1000 else None


def _queue_page(ctx: BotContext, page: int, stage: Stage | None) -> tuple[list[Job], int]:
    total = ctx.store.count_jobs(stage=stage)
    total_pages = max(1, (total + _QUEUE_PAGE_SIZE - 1) // _QUEUE_PAGE_SIZE)
    return (
        ctx.store.list_jobs(
            _QUEUE_PAGE_SIZE, offset=(page - 1) * _QUEUE_PAGE_SIZE, stage=stage,
        ),
        total_pages,
    )


def _publish_outcome(job_id: int, result) -> str:
    status = getattr(result, "status", "failed")
    message = getattr(result, "message", "")
    url = getattr(result, "url", None)
    if status == "delivered":
        return f"Публикация задачи #{job_id} завершена." + (f"\n{url}" if url else "")
    if status == "busy":
        return message or "В данный момент уже выполняется другая публикация."
    if status == "already_done":
        return f"Задача #{job_id} уже завершена. Для явного восстановления используйте /restore {job_id}."
    if status == "not_found":
        return f"Задача #{job_id} не найдена."
    if status == "not_ready":
        return f"Задача #{job_id} ещё не готова к публикации."
    detail = f": {message}" if message else ""
    return f"Публикация задачи #{job_id} не завершена{detail}"


def build_dispatcher(ctx: BotContext) -> Dispatcher:
    dp = Dispatcher()
    auth = AuthMiddleware(ctx.cfg.telegram.allowed_user_id)
    dp.message.middleware(auth)
    dp.callback_query.middleware(auth)
    pending: dict[str, PendingAction] = {}

    async def request_confirmation(
        message: Message, job: Job | None, action: str, *, actor_id: int,
    ) -> None:
        problem = _action_problem(job, action)
        if problem:
            await message.answer(problem)
            return
        now = _monotonic()
        for token, value in list(pending.items()):
            if value.expires_at <= now:
                pending.pop(token, None)
        token = secrets.token_urlsafe(9)
        pending[token] = PendingAction(
            user_id=actor_id,
            chat_id=message.chat.id,
            job_id=job.id,
            action=action,
            expires_at=now + _CONFIRM_TTL_SEC,
            expected_updated_at=job.updated_at if action in {"retry", "restore"} else None,
        )
        label = {"cancel": "отменить", "retry": "повторить", "restore": "восстановить"}[action]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Подтвердить", callback_data=f"confirm:{token}"),
            InlineKeyboardButton(text="Назад", callback_data=f"dismiss:{token}"),
        ]])
        await message.answer(
            f"Подтвердите: {label} задачу #{job.id}. Запрос действует 2 минуты.",
            reply_markup=keyboard,
        )

    async def run_publish(message: Message, job: Job, *, restore: bool) -> None:
        problem = _action_problem(job, "restore" if restore else "publish")
        if problem:
            await message.answer(problem)
            return
        await message.answer(f"Проверяю публикацию задачи #{job.id} в Audiobookshelf…")
        try:
            result = await ctx.loop.run_in_executor(
                None, lambda: ctx.worker.resend(job.id, restore=restore)
            )
        except Exception:
            log.exception("Ручная публикация задачи #%d завершилась исключением", job.id)
            await message.answer(f"Публикация задачи #{job.id} не завершена из-за внутренней ошибки.")
            return
        await message.answer(_publish_outcome(job.id, result))

    async def render_queue(
        message: Message, page: int, stage: Stage | None, *, edit: bool = False,
    ) -> None:
        jobs, total_pages = _queue_page(ctx, page, stage)
        if not jobs:
            text = "На этой странице задач нет."
            if edit:
                await message.edit_text(text)
            else:
                await message.answer(text)
            return
        filter_label = f", {STAGE_LABEL_RU[stage]}" if stage else ""
        lines = [f"<b>Задачи: страница {page}/{total_pages}{filter_label}</b>"]
        for job in jobs:
            lines.append(_fmt_job_line(job))
            if job.stage == Stage.FAILED and job.error:
                lines.append(f"    ↳ {_escaped_excerpt(job.error, 140)}")
            if job.delivery_error:
                lines.append(f"    ↳ публикация: {_escaped_excerpt(job.delivery_error, 140)}")
        keyboard = _job_keyboard(jobs, page=page, total_pages=total_pages, stage=stage)
        if edit:
            await message.edit_text("\n".join(lines), parse_mode="HTML", reply_markup=keyboard)
        else:
            await message.answer("\n".join(lines), parse_mode="HTML", reply_markup=keyboard)

    @dp.message(Command("library"))
    async def cmd_library(message: Message):
        await message.answer(ctx.cfg.audiobookshelf.public_url)

    @dp.message(Command("voice"))
    async def cmd_voice(message: Message):
        await message.answer(_voice_text(ctx), parse_mode="HTML")

    @dp.message(Command("start", "help"))
    async def cmd_help(message: Message):
        await message.answer(HELP, parse_mode="HTML")

    @dp.message(Command("status"))
    async def cmd_status(message: Message):
        active = ctx.store.active_job()
        queued = ctx.store.list_by_stage(Stage.QUEUED)
        ready = ctx.store.list_by_stage(Stage.READY)
        lines = ["<b>Сейчас:</b> " + _fmt_job_line(active) if active else "Активных задач нет."]
        lines.append(f"<b>В очереди:</b> {len(queued)}")
        lines.extend("  • " + _fmt_job_line(job) for job in queued[:5])
        lines.append(f"<b>Готовы к публикации:</b> {len(ready)}")
        await message.answer(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=_job_keyboard(([active] if active else []) + queued[:5]),
        )
        if ready:
            chunk = ctx.store.list_jobs(_QUEUE_PAGE_SIZE, stage=Stage.READY)
            ready_lines = ["<b>Готовы к публикации</b>"]
            for job in chunk:
                ready_lines.append(_fmt_job_line(job))
                if job.delivery_error:
                    ready_lines.append(f"    ↳ публикация: {_escaped_excerpt(job.delivery_error, 160)}")
            await message.answer(
                "\n".join(ready_lines), parse_mode="HTML",
                reply_markup=_job_keyboard(
                    chunk, page=1,
                    total_pages=(len(ready) + _QUEUE_PAGE_SIZE - 1) // _QUEUE_PAGE_SIZE,
                    stage=Stage.READY,
                ),
            )

    @dp.message(Command("queue"))
    async def cmd_queue(message: Message, command: CommandObject):
        options = _queue_options(command.args)
        if options is None:
            await message.answer("Формат: /queue [страница] [статус]. Например: /queue 2 failed")
            return
        page, stage = options
        await render_queue(message, page, stage)

    @dp.message(Command("cancel"))
    async def cmd_cancel(message: Message, command: CommandObject):
        job = _resolve_target(ctx, command, prefer="cancel")
        if job is None:
            await message.answer("Нечего отменять.")
            return
        await request_confirmation(message, job, "cancel", actor_id=message.from_user.id)

    @dp.message(Command("retry"))
    async def cmd_retry(message: Message, command: CommandObject):
        job = _resolve_target(ctx, command, prefer="failed")
        if job is None:
            await message.answer("Нет упавших задач для повтора.")
            return
        await request_confirmation(message, job, "retry", actor_id=message.from_user.id)

    @dp.message(Command("publish", "resend"))
    async def cmd_publish(message: Message, command: CommandObject):
        job = _resolve_target(ctx, command, prefer="ready")
        if job is None:
            await message.answer("Нет готовых задач, ожидающих публикации.")
            return
        if job.stage == Stage.DONE:
            await message.answer(f"Задача #{job.id} уже завершена. Для восстановления используйте /restore {job.id}.")
            return
        await run_publish(message, job, restore=False)

    @dp.message(Command("restore"))
    async def cmd_restore(message: Message, command: CommandObject):
        job_id = _parse_job_id(command)
        if job_id is None:
            await message.answer("Укажите id завершённой задачи: /restore 123")
            return
        await request_confirmation(
            message, ctx.store.get(job_id), "restore", actor_id=message.from_user.id,
        )

    @dp.callback_query(F.data.startswith("job:"))
    async def on_job_action(callback: CallbackQuery):
        await callback.answer()
        message = callback.message if isinstance(callback.message, Message) else None
        if message is None:
            return
        parts = (callback.data or "").split(":")
        if len(parts) != 3:
            await message.answer("Некорректное действие.")
            return
        _, raw_job_id, action = parts
        try:
            job_id = int(raw_job_id)
        except ValueError:
            await message.answer("Некорректный id задачи.")
            return
        job = ctx.store.get(job_id)
        if action in {"cancel", "retry", "restore"}:
            await request_confirmation(message, job, action, actor_id=callback.from_user.id)
        elif action == "publish":
            if job is None:
                await message.answer("Задача не найдена.")
            else:
                await run_publish(message, job, restore=False)
        else:
            await message.answer("Некорректное действие.")

    @dp.callback_query(F.data.startswith("queue:"))
    async def on_queue_page(callback: CallbackQuery):
        await callback.answer()
        message = callback.message if isinstance(callback.message, Message) else None
        if message is None:
            return
        parts = (callback.data or "").split(":")
        if len(parts) != 3 or not parts[1].isdecimal():
            await message.answer("Некорректная страница очереди.")
            return
        page = int(parts[1])
        stage = None if parts[2] == "all" else _QUEUE_FILTERS.get(parts[2])
        if not 1 <= page <= 1000 or (parts[2] != "all" and stage is None):
            await message.answer("Некорректная страница очереди.")
            return
        await render_queue(message, page, stage, edit=True)

    @dp.callback_query(F.data.startswith("dismiss:"))
    async def on_dismiss(callback: CallbackQuery):
        await callback.answer()
        message = callback.message if isinstance(callback.message, Message) else None
        if message is None:
            return
        token = (callback.data or "").removeprefix("dismiss:")
        action = pending.get(token)
        if (
            action is None
            or action.expires_at <= _monotonic()
            or action.user_id != callback.from_user.id
            or action.chat_id != message.chat.id
        ):
            pending.pop(token, None)
            await message.answer("Подтверждение уже недействительно.")
            return
        pending.pop(token, None)
        await message.answer("Действие отменено.")

    @dp.callback_query(F.data.startswith("confirm:"))
    async def on_confirm(callback: CallbackQuery):
        await callback.answer()
        message = callback.message if isinstance(callback.message, Message) else None
        if message is None:
            return
        token = (callback.data or "").removeprefix("confirm:")
        action = pending.get(token)
        if (
            action is None
            or action.expires_at <= _monotonic()
            or action.user_id != callback.from_user.id
            or action.chat_id != message.chat.id
        ):
            pending.pop(token, None)
            await message.answer("Подтверждение истекло или уже использовано.")
            return
        pending.pop(token, None)  # One use even if the operation subsequently fails.
        job = ctx.store.get(action.job_id)
        if action.expected_updated_at is not None and (
            job is None or job.updated_at != action.expected_updated_at
        ):
            await message.answer("Состояние задачи уже изменилось. Запросите действие заново.")
            return
        problem = _action_problem(job, action.action)
        if problem:
            await message.answer(problem)
            return
        if action.action == "cancel":
            if job.stage == Stage.QUEUED and ctx.store.cancel_queued(job.id):
                await message.answer(f"Задача #{job.id} убрана из очереди.")
            else:
                current = ctx.store.get(job.id)
                problem = _action_problem(current, "cancel")
                if problem:
                    await message.answer(problem)
                    return
                ctx.worker.cancels.request(job.id)
                await message.answer(f"Запрошена отмена задачи #{job.id}. Остановлю на ближайшем шаге.")
        elif action.action == "retry":
            try:
                ctx.worker.cancels.clear(job.id)
                ctx.store.requeue(job.id)
            except ValueError as error:
                await message.answer(str(error))
                return
            ctx.worker.request_wake()
            await message.answer(f"Задача #{job.id} возвращена в очередь.")
        else:
            await run_publish(message, job, restore=True)

    @dp.callback_query()
    async def on_unknown_callback(callback: CallbackQuery):
        await callback.answer("Неизвестное действие.", show_alert=True)

    @dp.message(F.document)
    async def on_document(message: Message):
        doc = message.document
        fname = Path((doc.file_name or f"upload_{doc.file_unique_id}.txt").replace("\\", "/")).name
        if not fname or fname in {".", ".."}:
            await message.answer("Некорректное имя файла")
            return
        if doc.file_size and doc.file_size > 20 * MIB:
            await message.answer(
                "Файл больше 20 МБ — Telegram не отдаёт такие боту. "
                "Положите книгу в папку books/ на устройстве."
            )
            return
        dest_dir = ctx.cfg.paths.work / "_incoming"
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest_dir) as incoming:
            dest = Path(incoming) / _fit_filename(fname)
            try:
                await message.bot.download(doc, destination=dest, timeout=ctx.cfg.telegram.upload_timeout_sec)
            except Exception as error:
                # Only the type: a network error's text can embed the file URL,
                # which contains the bot token.
                log.warning("Не удалось скачать «%s»: %s", fname, type(error).__name__)
                await message.answer("Не удалось скачать файл. Попробуйте отправить его ещё раз.")
                return
            job, reason = await ctx.loop.run_in_executor(
                None, ingest_file, ctx.cfg, ctx.store, dest, message.chat.id,
                parse_book_caption(message.caption),
            )
        if job is not None:
            ctx.worker.request_wake()
            await message.answer(f"Принято: «{fname}» → задача #{job.id} добавлена в очередь.")
        else:
            await message.answer(f"Не принято: {reason}")

    @dp.message(F.text)
    async def on_text(message: Message):
        await message.answer("Пришлите файл книги или используйте команды. /help — справка.")

    return dp
