"""Entrypoint: wire config, store, bot, watcher and worker on one asyncio loop.

Everything lives in a single process/container. The bot polls Telegram; the
folder watcher polls books/; the worker drains the queue one job at a time in a
background thread. On startup we reconcile interrupted jobs back to the queue so
work resumes after a restart.
"""

from __future__ import annotations

import asyncio
import os
import signal

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession

from .bot import BotContext, build_dispatcher
from .config import load_config
from .locking import ServiceAlreadyRunningError, ServiceLock
from .log import get_logger, setup_logging
from .menu import register_owner_menu
from .notify import Notifier
from .state import Store
from .watcher import FolderWatcher
from .worker import CancelRegistry, Worker

log = get_logger("main")


async def amain(cfg) -> None:
    setup_logging()
    log.info(
        "Конфигурация загружена. Движок: Silero %s, голос: %s",
        cfg.tts.model, cfg.tts.speaker,
    )

    store = Store(cfg.paths.state / "jobs.db")
    reset = store.reconcile_on_startup(cfg.paths.work)
    if reset:
        log.info("Возобновляю %d незавершённых задач", reset)

    loop = asyncio.get_running_loop()
    # A generous HTTP timeout so multi-MB document uploads don't abort midway
    # on a slow uplink (the library default is 60 s).
    session = AiohttpSession()
    session.timeout = float(cfg.telegram.upload_timeout_sec)
    bot = Bot(token=cfg.telegram.bot_token.get_secret_value(), session=session)
    notifier = Notifier(
        bot, loop, cfg.telegram.allowed_user_id,
        send_timeout=cfg.telegram.upload_timeout_sec + 60,
    )
    cancels = CancelRegistry()
    worker = Worker(cfg, store, notifier, loop, cancels)

    watcher = FolderWatcher(
        cfg, store, notifier, on_new_job=worker.request_wake_threadsafe
    )
    ctx = BotContext(cfg=cfg, store=store, notifier=notifier, worker=worker, loop=loop)
    dp = build_dispatcher(ctx)
    try:
        await asyncio.wait_for(
            register_owner_menu(bot, cfg.telegram.allowed_user_id), timeout=15,
        )
    except Exception as error:
        # Telegram can be temporarily unavailable during startup. Polling and
        # conversion must remain available; the next restart retries setup.
        log.warning("Не удалось зарегистрировать меню команд: %s", error)

    stop = asyncio.Event()

    def _request_stop(*_):
        log.info("Получен сигнал остановки")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover
            pass

    tasks = [
        asyncio.create_task(worker.run(), name="worker"),
        asyncio.create_task(watcher.run(), name="watcher"),
        asyncio.create_task(
            dp.start_polling(bot, handle_signals=False), name="bot"
        ),
    ]
    # Wake the worker once in case there are jobs already queued (resumed).
    worker.request_wake()

    stop_task = asyncio.create_task(stop.wait(), name="shutdown")
    try:
        finished, _ = await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            if task is not stop_task:
                error = task.exception()
                raise RuntimeError(f"Остановлена фоновая задача {task.get_name()}") from error
    finally:
        log.info("Останавливаюсь…")
        worker._stop = True
        worker.request_wake()
        # Let the executor finish its current chunk and persist state before closing Telegram.
        try:
            await asyncio.wait_for(asyncio.shield(tasks[0]), timeout=45)
        except Exception:
            log.warning("Воркер не завершился за время штатной остановки")
        for task in [*tasks, stop_task]:
            task.cancel()
        await asyncio.gather(*tasks, stop_task, return_exceptions=True)
        await bot.session.close()


def main() -> None:
    os.umask(0o077)
    try:
        cfg = load_config()
        # Keep the lock through asyncio.run(), including its default-executor
        # shutdown, so no second process can recover or publish concurrently.
        with ServiceLock(cfg.paths.state / "service.lock"):
            asyncio.run(amain(cfg))
    except ServiceAlreadyRunningError as error:
        raise SystemExit(f"Ошибка запуска: {error}") from None
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
