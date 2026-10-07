"""Explicit live smoke: Telegram file transport + dispatcher + real TTS + ABS.

Run with the normal bot stopped, in `docker compose run --rm -T audiobook`.
Sends a labelled test document and progress/completion messages to the owner.
The incoming Update is synthetic; file upload/download and all downstream work are real.
"""

import asyncio
import json
import os
import time

import httpx
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import BufferedInputFile, Message, Update

from app.bot import BotContext, build_dispatcher
from app.config import load_config
from app.notify import Notifier
from app.locking import ServiceLock
from app.state import Stage, Store
from app.worker import CancelRegistry, Worker


BOOK = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><genre>science</genre>
<author><first-name>Проверка</first-name><last-name>Сервиса</last-name></author>
<book-title>Новая библиотека</book-title><lang>ru</lang>
<sequence name="Проверка Audiobookshelf" number="1"/></title-info></description>
<body><section><title><p>Глава первая</p></title>
<p>Эта короткая книга проверяет новый конвейер. Текст получен через Телеграм и озвучен на домашнем сервере.</p>
</section><section><title><p>Глава вторая</p></title>
<p>Готовая аудиокнига хранится одним файлом. Теперь её можно слушать в библиотеке через защищённое соединение.</p>
</section></body></FictionBook>"""


async def main():
    cfg = load_config()
    store = Store(cfg.paths.state / "jobs.db")
    if store.active_job() or store.next_queued():
        raise RuntimeError("Активные задачи: проверка не запущена")
    loop = asyncio.get_running_loop()
    session = AiohttpSession(timeout=cfg.telegram.upload_timeout_sec)
    bot = Bot(cfg.telegram.bot_token, session=session)
    notifier = Notifier(bot, loop, cfg.telegram.allowed_user_id)
    worker = Worker(cfg, store, notifier, loop, CancelRegistry())
    dispatcher = build_dispatcher(BotContext(cfg, store, notifier, worker, loop))
    worker_task = asyncio.create_task(worker.run())
    before = max((job.id for job in store.list_jobs()), default=0)
    try:
        sent = await bot.send_document(
            cfg.telegram.allowed_user_id,
            BufferedInputFile((BOOK + f"\n<!-- smoke {time.time_ns()} -->").encode(), filename="Проверка_Audiobookshelf.fb2"),
            caption="Техническая проверка нового конвейера. Следом придёт ссылка на озвученную книгу.",
        )
        incoming = Message.model_validate({
            "message_id": sent.message_id, "date": int(time.time()),
            "chat": sent.chat.model_dump(), "document": sent.document.model_dump(),
            "from": {"id": cfg.telegram.allowed_user_id, "is_bot": False, "first_name": "Smoke"},
            "caption": "Название: Проверка новой библиотеки\nСерия: Проверка Audiobookshelf\nНомер: 1",
        })
        await dispatcher.feed_update(bot, Update(update_id=int(time.time()), message=incoming))
        job = next((j for j in store.list_jobs() if j.id > before), None)
        if job is None:
            raise RuntimeError("Обработчик Telegram не создал задачу")
        print("SMOKE_JOB", job.id, flush=True)
        deadline = time.monotonic() + 1200
        previous = None
        while time.monotonic() < deadline:
            job = store.get(job.id)
            if job.stage != previous:
                print("STAGE", job.stage.value, flush=True)
                previous = job.stage
            if job.stage == Stage.FAILED:
                raise RuntimeError(job.error)
            if job.stage == Stage.DONE:
                break
            await asyncio.sleep(2)
        else:
            raise RuntimeError("Проверка не завершена в срок")
        assert len(job.outputs) == 1, "Книга должна состоять из одного M4B"
        receipt = json.loads((cfg.paths.work / str(job.id) / "publication.json").read_text())
        assert receipt["notified"]
        async with httpx.AsyncClient(base_url=cfg.audiobookshelf.url, headers={
            "Authorization": "Bearer " + cfg.audiobookshelf.token.get_secret_value(),
        }) as client:
            response = await client.get(f"/api/items/{receipt['item_id']}", params={"expanded": 1})
            response.raise_for_status()
            item = response.json()
            media = item["media"]
            assert len(media["audioFiles"]) == 1
            assert media["duration"] > 5
            assert len(media["chapters"]) == 2
            assert media["metadata"]["title"] == "Проверка новой библиотеки"
            assert media["metadata"]["series"][0]["sequence"] == "1"
            assert (await client.get("/api/users")).status_code == 403
            print("VERIFIED", json.dumps({"job": job.id, "item": receipt["item_id"],
                  "duration": media["duration"], "chapters": len(media["chapters"]),
                  "files": len(media["audioFiles"]), "url": receipt["url"]}, ensure_ascii=False), flush=True)
    finally:
        worker._stop = True
        worker.request_wake()
        await worker_task
        await bot.session.close()


if __name__ == "__main__":
    os.umask(0o077)
    with ServiceLock(load_config().paths.state / "service.lock"):
        asyncio.run(main())
