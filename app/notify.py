"""Bridge for sending Telegram messages/documents from the (threaded) worker.

The bot runs on the asyncio loop; the conversion pipeline runs in a worker
thread. This class schedules coroutines onto the loop from that thread, either
fire-and-forget (progress updates) or blocking-with-result (document delivery).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from aiogram import Bot
from aiogram.types import FSInputFile

from .log import get_logger

log = get_logger("notify")

# Telegram rejects text messages longer than this; a book title taken from
# untrusted FB2/EPUB metadata must not turn a notice into a permanent failure.
TELEGRAM_TEXT_LIMIT = 4096


def clip_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Notifier:
    def __init__(
        self,
        bot: Bot,
        loop: asyncio.AbstractEventLoop,
        default_chat_id: int,
        send_timeout: float = 660,
    ):
        self.bot = bot
        self.loop = loop
        self.default_chat_id = default_chat_id
        # Upper bound for a blocking document upload; should exceed the HTTP
        # client timeout so aiohttp reports the real error first.
        self.send_timeout = send_timeout

    # ---- fire-and-forget (safe from any thread) -----------------------
    def notify(self, text: str, chat_id: Optional[int] = None) -> None:
        cid = chat_id or self.default_chat_id
        coro = self._safe_send(cid, text)
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _safe_send(self, chat_id: int, text: str) -> None:
        try:
            await self.bot.send_message(chat_id, clip_text(text))
        except Exception as e:
            log.warning("Не удалось отправить сообщение: %s", e)

    def send_message(self, text: str, chat_id: Optional[int] = None) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.bot.send_message(chat_id or self.default_chat_id, clip_text(text)), self.loop
        )
        try:
            future.result(timeout=self.send_timeout)
        except BaseException:
            future.cancel()
            raise

    # ---- blocking document delivery (from worker thread) --------------
    def send_document(
        self,
        path: Path,
        filename: str,
        caption: Optional[str] = None,
        chat_id: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Raises on failure so the worker can record a delivery error."""
        cid = chat_id or self.default_chat_id
        fut = asyncio.run_coroutine_threadsafe(
            self._send_document(cid, path, filename, caption), self.loop
        )
        try:
            fut.result(timeout=timeout or self.send_timeout)
        except BaseException:
            fut.cancel()
            raise

    async def _send_document(
        self, chat_id: int, path: Path, filename: str, caption: Optional[str]
    ) -> None:
        doc = FSInputFile(str(path), filename=filename)
        await self.bot.send_document(chat_id, doc, caption=caption)
