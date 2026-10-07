"""Verify external login, private media, byte ranges and WebSockets on the Pi."""

import asyncio
import json
from pathlib import Path

import aiohttp
import httpx

from app.config import load_config
from app.state import Stage, Store


async def main():
    cfg = load_config()
    jobs = Store(cfg.paths.state / "jobs.db").list_by_stage(Stage.DONE)
    job = next(job for job in reversed(jobs)
               if (cfg.paths.work / str(job.id) / "publication.json").is_file())
    publication = json.loads((cfg.paths.work / str(job.id) / "publication.json").read_text())
    credentials = json.loads(Path("/admin/credentials.json").read_text())
    async with httpx.AsyncClient(base_url=cfg.audiobookshelf.public_url, timeout=30) as client:
        assert (await client.get("/api/libraries")).status_code == 401
        response = await client.post("/login", json=credentials, headers={"x-return-tokens": "true"})
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["user"]["accessToken"]
        response = await client.get(f"/api/items/{publication['item_id']}", params={"expanded": 1})
        response.raise_for_status()
        media = response.json()["media"]
        inode = media["audioFiles"][0]["ino"]
        file_url = f"/api/items/{publication['item_id']}/file/{inode}"
        response = await client.get(file_url, headers={"Range": "bytes=0-1023"})
        assert response.status_code == 206, response.status_code
        assert len(response.content) == 1024
        assert response.headers["content-range"].startswith("bytes 0-1023/")
        assert b"ftyp" in response.content[:32]
        print("HTTPS_LOGIN_OK RANGE_206_OK M4B_HEADER_OK", flush=True)
        del client.headers["Authorization"]
        assert (await client.get(file_url)).status_code == 401
        print("UNAUTHORIZED_AUDIO_BLOCKED", flush=True)
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            cfg.audiobookshelf.public_url + "/socket.io/?EIO=4&transport=websocket", timeout=30,
        ) as socket:
            message = await socket.receive(timeout=15)
            assert message.type == aiohttp.WSMsgType.TEXT and message.data.startswith("0{"), message.type
            print("WEBSOCKET_UPGRADE_OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
