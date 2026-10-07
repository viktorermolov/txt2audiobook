"""Telegram command menu for the service owner.

The menu is scoped to the owner's private chat so it is not advertised in any
other Telegram conversation.
"""

from __future__ import annotations

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeChat


async def register_owner_menu(bot: Bot, chat_id: int) -> None:
    """Register the small native command menu used in the owner's chat."""
    await bot.set_my_commands(
        [
            BotCommand(command="status", description="Текущая задача и очередь"),
            BotCommand(command="queue", description="Задачи и действия"),
            BotCommand(command="library", description="Открыть библиотеку"),
            BotCommand(command="voice", description="Настройки озвучки"),
            BotCommand(command="help", description="Справка"),
        ],
        scope=BotCommandScopeChat(chat_id=chat_id),
    )
