from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import TelegramMethod
from aiogram.types import Update

from app.bot import BotContext, _fit_filename, _publish_outcome, build_dispatcher
from app.config import AppConfig
from app.menu import register_owner_menu
from app.state import Stage, Store
from app.worker import PublishResult


OWNER_ID = 7
BOT_TOKEN = "42:" + "x" * 35


class RecordingSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod] = []

    async def close(self) -> None:
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        if False:  # pragma: no cover - keeps this an async generator.
            yield b""


class FakeCancels:
    def __init__(self) -> None:
        self.requested: list[int] = []
        self.cleared: list[int] = []

    def request(self, job_id: int) -> None:
        self.requested.append(job_id)

    def clear(self, job_id: int) -> None:
        self.cleared.append(job_id)


class FakeWorker:
    def __init__(self) -> None:
        self.cancels = FakeCancels()
        self.woken = 0
        self.resends: list[tuple[int, bool]] = []

    def request_wake(self) -> None:
        self.woken += 1

    def resend(self, job_id: int, *, restore: bool = False) -> PublishResult:
        self.resends.append((job_id, restore))
        return PublishResult("delivered", url="https://ab.example/item")


def message_update(
    *, text: str, chat_type: str = "private", user_id: int = OWNER_ID,
    chat_id: int | None = None, update_id: int = 1,
) -> Update:
    chat = {"id": chat_id if chat_id is not None else user_id, "type": chat_type}
    if chat_type != "private":
        chat["title"] = "Group"
    return Update.model_validate({
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1,
            "chat": chat,
            "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
            "text": text,
        },
    })


def callback_update(
    *, data: str, user_id: int = OWNER_ID, chat_type: str = "private",
    chat_id: int | None = None, inaccessible: bool = False, update_id: int = 50,
) -> Update:
    chat = {"id": chat_id if chat_id is not None else user_id, "type": chat_type}
    if chat_type != "private":
        chat["title"] = "Group"
    return Update.model_validate({
        "update_id": update_id,
        "callback_query": {
            "id": f"callback-{update_id}",
            "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
            "chat_instance": "test-chat",
            "data": data,
            "message": {
                "message_id": 99,
                "date": 0 if inaccessible else 1,
                "chat": chat,
                **({} if inaccessible else {
                    "from": {"id": 42, "is_bot": True, "first_name": "Bot"},
                    "text": "Подтверждение",
                }),
            },
        },
    })


class BotDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = AppConfig.model_validate({
            "telegram": {"bot_token": "test-token", "allowed_user_id": OWNER_ID},
            "paths": {
                "books": str(root / "books"),
                "audiobook": str(root / "audiobook"),
                "state": str(root / "state"),
                "work": str(root / "work"),
                "voices": str(root / "voices"),
            },
            "audiobookshelf": {"library_path": str(root / "library")},
        })
        self.cfg.ensure_dirs()
        self.store = Store(self.cfg.paths.state / "jobs.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def create_job(self, stage: Stage):
        number = self.store.count_jobs() + 1
        source = self.cfg.paths.work / f"source-{number}.txt"
        source.write_text(f"Текст {number}", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        job = self.store.create_job(source.name, str(source), digest, OWNER_ID, stage)
        if stage in {Stage.READY, Stage.DONE}:
            output = self.cfg.paths.audiobook / f"{job.id}.m4b"
            output.write_bytes(b"m4b")
            self.store.set_outputs(job.id, [str(output)])
        return self.store.get(job.id)

    async def dispatch(self, updates: list[Update]) -> tuple[RecordingSession, FakeWorker]:
        session = RecordingSession()
        bot = Bot(BOT_TOKEN, session=session)
        worker = FakeWorker()
        ctx = BotContext(
            cfg=self.cfg,
            store=self.store,
            notifier=None,
            worker=worker,
            loop=asyncio.get_running_loop(),
        )
        dp = build_dispatcher(ctx)
        try:
            for update in updates:
                await dp.feed_update(bot, update)
        finally:
            await dp.storage.close()
            await bot.session.close()
        return session, worker

    def test_retry_without_id_selects_failed_not_cancelled(self) -> None:
        older = self.create_job(Stage.FAILED)
        failed = self.create_job(Stage.FAILED)
        cancelled = self.create_job(Stage.CANCELLED)

        session, _ = asyncio.run(self.dispatch([message_update(text="/retry")]))

        messages = [call for call in session.calls if type(call).__name__ == "SendMessage"]
        self.assertEqual(1, len(messages))
        self.assertIn(f"#{failed.id}", messages[0].text)
        self.assertNotIn(f"#{cancelled.id}", messages[0].text)
        self.assertNotIn(f"#{older.id}", messages[0].text)
        self.assertIsNotNone(messages[0].reply_markup)

    def test_retry_with_cancelled_id_offers_confirmation(self) -> None:
        cancelled = self.create_job(Stage.CANCELLED)

        session, _ = asyncio.run(self.dispatch([message_update(text=f"/retry {cancelled.id}")]))

        message = next(call for call in session.calls if type(call).__name__ == "SendMessage")
        self.assertIn(f"#{cancelled.id}", message.text)
        self.assertIsNotNone(message.reply_markup)

    def test_non_owner_group_and_private_mismatch_messages_are_ignored(self) -> None:
        session, worker = asyncio.run(self.dispatch([
            message_update(text="/status", user_id=8, update_id=1),
            message_update(text="/status", chat_type="group", update_id=2),
            message_update(text="/status", chat_id=8, update_id=3),
        ]))

        self.assertEqual([], session.calls)
        self.assertEqual([], worker.resends)

    def test_inline_restore_binds_owner_not_bot_message_author(self) -> None:
        job = self.create_job(Stage.DONE)

        async def scenario():
            session = RecordingSession()
            bot = Bot(BOT_TOKEN, session=session)
            worker = FakeWorker()
            ctx = BotContext(
                cfg=self.cfg,
                store=self.store,
                notifier=None,
                worker=worker,
                loop=asyncio.get_running_loop(),
            )
            dp = build_dispatcher(ctx)
            try:
                await dp.feed_update(bot, callback_update(data=f"job:{job.id}:restore"))
                confirmation = next(call for call in session.calls if type(call).__name__ == "SendMessage")
                token = confirmation.reply_markup.inline_keyboard[0][0].callback_data
                session.calls.clear()
                await dp.feed_update(bot, callback_update(data=token))
            finally:
                await dp.storage.close()
                await bot.session.close()
            return session, worker

        session, worker = asyncio.run(scenario())

        self.assertEqual([(job.id, True)], worker.resends)
        self.assertEqual("AnswerCallbackQuery", type(session.calls[0]).__name__)

    def test_group_callback_is_answered_without_running_action(self) -> None:
        job = self.create_job(Stage.DONE)
        session, worker = asyncio.run(self.dispatch([
            callback_update(data=f"job:{job.id}:restore", chat_type="group")
        ]))

        self.assertEqual([], worker.resends)
        self.assertEqual(["AnswerCallbackQuery"], [type(call).__name__ for call in session.calls])

    def test_status_bounds_ready_messages_and_paginates_in_consistent_order(self) -> None:
        jobs = [self.create_job(Stage.READY) for _ in range(8)]
        session, _ = asyncio.run(self.dispatch([
            message_update(text="/status"),
            callback_update(data="queue:2:ready"),
        ]))
        messages = [call for call in session.calls if type(call).__name__ == "SendMessage"]
        edits = [call for call in session.calls if type(call).__name__ == "EditMessageText"]
        self.assertEqual(2, len(messages))
        self.assertIn(f"#{jobs[-1].id} ", messages[1].text)
        self.assertNotIn(f"#{jobs[0].id} ", messages[1].text)
        self.assertIn(f"#{jobs[0].id} ", edits[0].text)
        self.assertNotIn(f"#{jobs[-1].id} ", edits[0].text)

    def test_inaccessible_callback_message_is_answered_without_action(self) -> None:
        job = self.create_job(Stage.DONE)
        session, worker = asyncio.run(self.dispatch([
            callback_update(data=f"job:{job.id}:restore", inaccessible=True)
        ]))

        self.assertEqual([], worker.resends)
        self.assertEqual(["AnswerCallbackQuery"], [type(call).__name__ for call in session.calls])

    def test_expired_replayed_and_stale_restore_confirmations_do_not_publish(self) -> None:
        job = self.create_job(Stage.DONE)

        async def scenario():
            session = RecordingSession()
            bot = Bot(BOT_TOKEN, session=session)
            worker = FakeWorker()
            ctx = BotContext(self.cfg, self.store, None, worker, asyncio.get_running_loop())
            dp = build_dispatcher(ctx)
            try:
                await dp.feed_update(bot, callback_update(data=f"job:{job.id}:restore"))
                token = next(call for call in session.calls if type(call).__name__ == "SendMessage").reply_markup.inline_keyboard[0][0].callback_data
                session.calls.clear()
                with patch("app.bot._monotonic", return_value=10**12):
                    await dp.feed_update(bot, callback_update(data=token, update_id=51))
                await dp.feed_update(bot, callback_update(data=token, update_id=52))

                session.calls.clear()
                await dp.feed_update(bot, callback_update(data=f"job:{job.id}:restore", update_id=53))
                stale_token = next(call for call in session.calls if type(call).__name__ == "SendMessage").reply_markup.inline_keyboard[0][0].callback_data
                self.store.set_stage(job.id, Stage.READY, progress=1.0)
                await dp.feed_update(bot, callback_update(data=stale_token, update_id=54))
            finally:
                await dp.storage.close()
                await bot.session.close()
            return session, worker

        session, worker = asyncio.run(scenario())

        self.assertEqual([], worker.resends)
        messages = [call.text for call in session.calls if type(call).__name__ == "SendMessage"]
        self.assertTrue(any("Состояние задачи уже изменилось" in text for text in messages))

    def test_status_lists_ready_job_without_publication_error(self) -> None:
        ready = self.create_job(Stage.READY)
        session, _ = asyncio.run(self.dispatch([message_update(text="/status")]))

        texts = [call.text for call in session.calls if type(call).__name__ == "SendMessage"]
        self.assertTrue(any(f"#{ready.id}" in text for text in texts))

    def test_queue_navigation_handles_long_escaped_values_within_telegram_limit(self) -> None:
        for _ in range(8):
            job = self.create_job(Stage.FAILED)
            self.store.set_meta(job.id, "<" * 300, None)
            self.store.mark_failed(job.id, "&" * 300)

        async def scenario():
            session = RecordingSession()
            bot = Bot(BOT_TOKEN, session=session)
            worker = FakeWorker()
            ctx = BotContext(self.cfg, self.store, None, worker, asyncio.get_running_loop())
            dp = build_dispatcher(ctx)
            try:
                await dp.feed_update(bot, message_update(text="/queue"))
                queue_message = next(call for call in session.calls if type(call).__name__ == "SendMessage")
                next_page = queue_message.reply_markup.inline_keyboard[-1][0].callback_data
                session.calls.clear()
                await dp.feed_update(bot, callback_update(data=next_page))
            finally:
                await dp.storage.close()
                await bot.session.close()
            return session

        session = asyncio.run(scenario())
        methods = [type(call).__name__ for call in session.calls]
        self.assertEqual("AnswerCallbackQuery", methods[0])
        self.assertIn("EditMessageText", methods)

        # The initial page is emitted before the callback and is bounded even
        # when HTML characters expand during escaping.
        initial_session, _ = asyncio.run(self.dispatch([message_update(text="/queue", update_id=60)]))
        initial = next(call for call in initial_session.calls if type(call).__name__ == "SendMessage")
        self.assertLessEqual(len(initial.text), 4096)

    def test_publish_outcomes_do_not_offer_stale_done_url(self) -> None:
        busy = _publish_outcome(12, PublishResult("busy", "Публикация другой задачи уже идёт."))
        done = _publish_outcome(12, PublishResult("already_done", url="https://stale.example"))

        self.assertIn("другой задачи", busy)
        self.assertIn("/restore 12", done)
        self.assertNotIn("stale.example", done)

    def test_menu_is_scoped_to_owner_private_chat(self) -> None:
        class MenuBot:
            async def set_my_commands(self, commands, *, scope):
                self.commands = commands
                self.scope = scope

        bot = MenuBot()
        asyncio.run(register_owner_menu(bot, OWNER_ID))

        self.assertEqual(OWNER_ID, bot.scope.chat_id)
        self.assertEqual(["status", "queue", "library", "voice", "help"], [item.command for item in bot.commands])

    def test_long_upload_names_fit_the_filesystem_and_keep_the_extension(self) -> None:
        name = "Очень длинное название книги " * 12 + ".fb2"
        fitted = _fit_filename(name)
        self.assertLessEqual(len(fitted.encode("utf-8")), 200)
        self.assertTrue(fitted.endswith(".fb2"))
        self.assertEqual("книга.epub", _fit_filename("книга.epub"))
