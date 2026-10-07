"""Publish complete books atomically, then verify Audiobookshelf indexed them."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from .files import fsync_directory, fsync_file, rename_directory_and_fsync, write_json
from .log import get_logger
from .pipeline.types import safe_filename

log = get_logger("library")

class PublicationError(Exception):
    pass


def book_relative_path(meta: dict, content_hash: str, job_id: int | None = None) -> Path:
    author = safe_filename(meta.get("author") or "Неизвестный автор")
    title = safe_filename(meta.get("title") or Path(meta["source_name"]).stem)
    series = meta.get("series")
    index = meta.get("series_index")
    prefix = f"{safe_filename(str(index), max_len=16)} - " if series and index else ""
    # The hash distinguishes editions and makes retry paths deterministic.
    identity = content_hash[:12] + (f"-{job_id}" if job_id is not None else "")
    folder = f"{prefix}{title} [{identity}]"
    return Path(author, safe_filename(series), folder) if series else Path(author, folder)


def write_opf(path: Path, meta: dict, speaker: str) -> None:
    opf = "http://www.idpf.org/2007/opf"
    dc = "http://purl.org/dc/elements/1.1/"
    ET.register_namespace("", opf)
    ET.register_namespace("dc", dc)
    root = ET.Element(f"{{{opf}}}package", version="2.0")
    metadata = ET.SubElement(root, f"{{{opf}}}metadata")
    for key, value in [("title", meta.get("title") or Path(meta["source_name"]).stem),
                       ("creator", meta.get("author") or "Неизвестный автор"),
                       ("language", "ru")]:
        ET.SubElement(metadata, f"{{{dc}}}{key}").text = value
    ET.SubElement(metadata, f"{{{dc}}}creator", {f"{{{opf}}}role": "nrt"}).text = f"Silero {speaker}"
    if meta.get("series"):
        ET.SubElement(metadata, f"{{{opf}}}meta", name="calibre:series", content=meta["series"])
        if meta.get("series_index"):
            ET.SubElement(metadata, f"{{{opf}}}meta", name="calibre:series_index", content=str(meta["series_index"]))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    fsync_file(path)


class LibraryPublisher:
    def __init__(self, cfg):
        self.cfg = cfg

    def publish(self, job, meta: dict, *, cancel_check=lambda: False) -> dict:
        cfg = self.cfg.audiobookshelf
        if not cfg.library_id or not cfg.token.get_secret_value():
            raise PublicationError("Audiobookshelf ещё не настроен; готовое аудио сохранено")
        root = cfg.library_path
        relative = book_relative_path(meta, job.content_hash, job.id)
        destination = root / relative
        receipt = destination / ".txt2audiobook.json"
        expected = {"content_hash": job.content_hash, "job_id": job.id}
        if destination.exists():
            try:
                saved = json.loads(receipt.read_text())
                if saved.get("content_hash") != job.content_hash or saved.get("job_id") != job.id:
                    raise ValueError("different source")
                files = saved["files"]
                if not files or any(not (destination / f["name"]).is_file() or
                                    (destination / f["name"]).stat().st_size != f["size"] or
                                    _file_hash(destination / f["name"]) != f["sha256"] for f in files):
                    raise ValueError("incomplete files")
                if len(job.outputs) != len(files) or any(_file_hash(Path(source)) != info["sha256"]
                                                        for source, info in zip(job.outputs, files)):
                    raise ValueError("different rendered audio")
                cover = saved.get("cover")
                if cover is not None:
                    image = destination / "cover.jpg"
                    if (cover["name"] != "cover.jpg" or image.stat().st_size != cover["size"]
                            or _file_hash(image) != cover["sha256"]):
                        raise ValueError("incomplete cover")
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise PublicationError("Каталог книги уже существует, но не прошёл проверку; он не перезаписан") from e
        else:
            # Hidden staging is on the same filesystem as the final library.
            staging = root / ".staging" / str(job.id)
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            files = []
            try:
                for number, source in enumerate(job.outputs, 1):
                    if cancel_check():
                        raise PublicationError("Публикация прервана; повторю после запуска")
                    source = Path(source)
                    if not source.is_file() or source.stat().st_size == 0:
                        raise PublicationError(f"Готовый файл отсутствует: {source.name}")
                    name = f"{number:02d}.m4b"
                    target = staging / name
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                    fsync_file(target)
                    files.append({"name": name, "size": target.stat().st_size, "sha256": _file_hash(target)})
                if not files:
                    raise PublicationError("Нет готовых файлов для публикации")
                write_opf(staging / "metadata.opf", meta, self.cfg.tts.speaker)
                artwork = {}
                cover = self.cfg.paths.work / str(job.id) / "cover.jpg"
                try:
                    cover_data = cover.read_bytes()
                except FileNotFoundError:
                    cover_data = None
                except OSError:
                    log.warning("Обложка недоступна; публикую аудиокнигу без отдельного изображения")
                    cover_data = None
                if cover_data:
                    image = staging / "cover.jpg"
                    image.write_bytes(cover_data)
                    fsync_file(image)
                    artwork["cover"] = {"name": "cover.jpg", "size": image.stat().st_size,
                                        "sha256": _file_hash(image)}
                write_json(staging / ".txt2audiobook.json", {**expected, "files": files, **artwork})
                destination.parent.mkdir(parents=True, exist_ok=True)
                fsync_directory(staging)
                rename_directory_and_fsync(staging, destination)
                # A new author/series path has parent entries of its own.
                # Persist them up to the existing library root as well.
                for parent in destination.parents:
                    fsync_directory(parent)
                    if parent == root:
                        break
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

        token = cfg.token.get_secret_value()
        try:
            with httpx.Client(base_url=cfg.url, headers={"Authorization": f"Bearer {token}"}, timeout=30) as client:
                # ABS watches the shared directory, with a scheduled scan as fallback.
                # The converter deliberately holds a read-only, non-admin API key.
                deadline = time.monotonic() + cfg.scan_timeout_sec
                while time.monotonic() < deadline and not cancel_check():
                    page = 0
                    while True:
                        response = client.get(f"/api/libraries/{cfg.library_id}/items", params={"limit": 100, "page": page})
                        response.raise_for_status()
                        payload = response.json()
                        items = payload.get("results", [])
                        for item in items:
                            if Path(item.get("path", "")) == destination:
                                detail = client.get(f"/api/items/{item['id']}", params={"expanded": 1})
                                detail.raise_for_status()
                                media = detail.json().get("media", {})
                                if media.get("numAudioFiles", len(media.get("audioFiles", []))) >= len(files) and media.get("duration", 0) > 0:
                                    return {"item_id": item["id"], "path": str(destination),
                                            "url": f"{cfg.public_url}/item/{item['id']}"}
                        if (page + 1) * 100 >= payload.get("total", len(items)):
                            break
                        page += 1
                    time.sleep(2)
        except httpx.HTTPStatusError as e:
            raise PublicationError(f"Audiobookshelf ответил HTTP {e.response.status_code}; аудио сохранено") from e
        except (httpx.RequestError, ValueError) as e:
            raise PublicationError("Audiobookshelf недоступен или вернул неверный ответ; аудио сохранено") from e
        raise PublicationError("Audiobookshelf ещё не подтвердил индексацию книги; публикация будет повторена")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
