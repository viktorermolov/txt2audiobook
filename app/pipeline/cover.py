"""Best-effort, local-only book covers, normalized to a metadata-free JPEG."""

from __future__ import annotations

import base64
import io
import posixpath
import warnings
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from lxml import etree
from PIL import Image, ImageOps

from ..files import replace_and_fsync
from ..log import get_logger
from .extract import _validate_epub_archive

log = get_logger("cover")

# Limits apply only to optional artwork, never to book text or output audio.
MAX_COVER_BYTES = 16 * 1024 * 1024
MAX_COVER_PIXELS = 24_000_000
MAX_XML_BYTES = 2 * 1024 * 1024


def _member_path(base: str, reference: str) -> str:
    url = urlsplit(reference)
    path = unquote(url.path)
    if url.scheme or url.netloc or not path or path.startswith("/") or "\\" in path:
        raise ValueError("non-local cover reference")
    result = posixpath.normpath(posixpath.join(posixpath.dirname(base), path))
    if result == ".." or result.startswith("../"):
        raise ValueError("cover reference escapes archive")
    return result


def _read_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    if archive.getinfo(name).file_size > limit:
        raise ValueError("oversized cover member")
    with archive.open(name) as member:
        data = member.read(limit + 1)
    if len(data) > limit:
        raise ValueError("oversized cover member")
    return data


def _xml(data: bytes):
    return etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))


def _epub_covers(path: Path):
    _validate_epub_archive(path)
    with zipfile.ZipFile(path) as archive:
        container = _xml(_read_member(archive, "META-INF/container.xml", MAX_XML_BYTES))
        roots = container.xpath("//*[local-name()='rootfile']/@full-path")
        if not roots:
            return None
        package_path = _member_path("", roots[0])
        package = _xml(_read_member(archive, package_path, MAX_XML_BYTES))
        items = package.xpath("//*[local-name()='manifest']/*[local-name()='item']")
        cover_ids = package.xpath("//*[local-name()='metadata']/*[local-name()='meta'][@name='cover']/@content")
        candidates = [item.get("href") for item in items
                      if "cover-image" in item.get("properties", "").split()]
        candidates += [item.get("href") for item in items if item.get("id") in cover_ids]
        candidates += package.xpath("//*[local-name()='guide']/*[local-name()='reference'][@type='cover']/@href")
        for reference in dict.fromkeys(candidates):
            if not reference:
                continue
            try:
                name = _member_path(package_path, reference)
                # EPUB 2 often points to a cover XHTML/SVG page, not the bitmap.
                if posixpath.splitext(name)[1].lower() in {".xhtml", ".html", ".htm", ".svg"}:
                    page = _xml(_read_member(archive, name, MAX_XML_BYTES))
                    refs = page.xpath("//*[local-name()='img']/@src | "
                                      "//*[local-name()='image']/@*[local-name()='href']")
                    if not refs:
                        continue
                    name = _member_path(name, refs[0])
                yield _read_member(archive, name, MAX_COVER_BYTES)
            except (KeyError, ValueError, etree.XMLSyntaxError):
                continue
    return None


def _fb2_cover(path: Path) -> bytes | None:
    with path.open("rb") as source:
        return _fb2_cover_stream(source)


def _fb2_cover_stream(source) -> bytes | None:
    cover_id = None
    # Stream past body text and unrelated binaries instead of retaining the book.
    for _, element in etree.iterparse(source, events=("end",), resolve_entities=False,
                                     no_network=True, huge_tree=True):
        if not isinstance(element.tag, str):
            continue
        name = etree.QName(element).localname
        parent = element.getparent()
        if name == "image" and parent is not None and etree.QName(parent).localname == "coverpage":
            if any(etree.QName(p).localname == "title-info" for p in element.iterancestors()):
                href = element.get("{http://www.w3.org/1999/xlink}href") or element.get("href", "")
                if href.startswith("#") and cover_id is None:
                    cover_id = unquote(href[1:])
        elif name == "binary" and cover_id and element.get("id") == cover_id:
            encoded = element.text or ""
            if len(encoded) > MAX_COVER_BYTES * 2:
                raise ValueError("oversized cover binary")
            data = base64.b64decode("".join(encoded.split()), validate=True)
            if len(data) > MAX_COVER_BYTES:
                raise ValueError("oversized cover binary")
            return data
        element.clear()
        if parent is not None:
            while element.getprevious() is not None:
                del parent[0]
    return None


def extract_cover(source: Path, work_dir: Path) -> Path | None:
    """Overwrite the derived cover on assembly/resume; bad artwork is optional."""
    target = work_dir / "cover.jpg"
    temporary = work_dir / "cover.jpg.part"
    try:
        target.unlink(missing_ok=True)
        if source.suffix.lower() == ".epub":
            candidates = _epub_covers(source)
        elif source.suffix.lower() == ".fb2":
            candidates = [_fb2_cover(source)]
        else:
            return None
        for data in candidates:
            if not data:
                continue
            try:
                _normalize(data, temporary)
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                log.warning("Повреждённое или неподдерживаемое изображение обложки пропущено")
                continue
            replace_and_fsync(temporary, target)
            return target
        return None
    except Exception as exc:
        # Never log embedded content or turn optional artwork into a failed book.
        log.warning("Обложка пропущена: %s", type(exc).__name__)
        return None
    finally:
        temporary.unlink(missing_ok=True)


def _normalize(data: bytes, target: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data), formats=["JPEG", "PNG", "WEBP", "GIF"]) as original:
            if original.width * original.height > MAX_COVER_PIXELS:
                raise ValueError("oversized cover dimensions")
            original.load()
            picture = ImageOps.exif_transpose(original).convert("RGBA")
            picture.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            flattened = Image.new("RGB", picture.size, "white")
            flattened.paste(picture, mask=picture.getchannel("A"))
            flattened.save(target, format="JPEG", quality=90)
