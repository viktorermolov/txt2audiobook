"""Extract title/author/chapters from a source file.

Supported: TXT, EPUB, FB2, RTF, HTML/HTM, MD/Markdown. Unknown but
text-like files fall back to plain-text decoding. Anything that yields no
readable text raises ExtractError with a Russian message.

Russian legacy encodings (CP1251, KOI8-R, etc.) are detected with
charset-normalizer plus a Cyrillic-aware sanity check.
"""

from __future__ import annotations

import html as _html
import re
import zipfile
from pathlib import Path

from charset_normalizer import from_bytes

from ..log import get_logger
from .types import Chapter, Document, ExtractError

log = get_logger("extract")

TEXT_EXTS = {".txt", ".text", ".md", ".markdown", ".html", ".htm", ".xhtml"}
ALL_EXTS = TEXT_EXTS | {".epub", ".fb2", ".rtf"}

# EPUB is a zip container. These are expansion-abuse limits, not book-length
# limits: ordinary content is allowed to be large and highly compressible. The
# ratio rule only activates above 256 MiB expanded, and the absolute ceilings
# remain deliberately generous for illustrated/technical books.
_EPUB_MAX_FILES = 100_000
_EPUB_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_EPUB_MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
_EPUB_RATIO_CHECK_BYTES = 256 * 1024 * 1024
_EPUB_MAX_EXPANSION_RATIO = 200
# Per member, the ratio rule starts much earlier: every XHTML document is
# parsed whole into a BeautifulSoup tree (many times its size in RAM), so a
# 250 MiB repetitive member from a ~1 MiB file would OOM the Pi. Real book
# text compresses roughly 3–8:1, far below the 200:1 threshold.
_EPUB_MEMBER_RATIO_CHECK_BYTES = 16 * 1024 * 1024
_FB2_MAX_SECTION_DEPTH = 32
# Encoding detection scores a bounded sample: scoring builds per-character
# lists, which for a multi-megabyte legacy TXT costs hundreds of MB per candidate.
_DECODE_SCORE_SAMPLE_CHARS = 256 * 1024

# Encodings we try first for Russian content, in order.
_RU_FALLBACKS = ["utf-8", "windows-1251", "koi8-r", "ibm866", "iso-8859-5", "mac-cyrillic"]


# Letters that dominate normal Russian text. Mojibake from a wrong single-byte
# codec leans on rare letters (ъ, ё, э, ф, …), so the share of these common
# letters separates real Russian from garbage.
_COMMON_RU = set("оеаинтсрвлкмдпуяыьгзбчйхжшюцщэфё ОЕАИНТСРВЛКМДПУЯЫЬГЗБЧЙХ")
_FREQUENT_RU = set("оеаинтсрвлкмд")


def _cyrillic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ" or c in "Ёё")
    return cyr / len(letters)


def _text_score(text: str) -> float:
    """Plausibility of `text` as Russian/printable prose. Penalizes
    replacement chars and rewards a natural share of frequent letters."""
    if not text:
        return -9.0
    n = len(text)
    repl = text.count("�") / n
    cyr_letters = [c for c in text if "Ѐ" <= c <= "ӿ" or c in "Ёё"]
    if cyr_letters:
        freq = sum(1 for c in cyr_letters if c.lower() in _FREQUENT_RU) / len(cyr_letters)
    else:
        freq = 0.0
    # Share of bytes that are sensible text characters.
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t") / n
    return _cyrillic_ratio(text) * 0.5 + freq * 1.0 + printable * 0.3 - repl * 3.0


def decode_bytes(data: bytes) -> str:
    """Decode bytes to text. UTF-8 is tried strictly first (it is
    self-validating, so a clean decode is almost never a false positive);
    only genuine legacy single-byte content falls through to detection.
    Never raises — worst case returns lossy UTF-8."""
    if not data:
        return ""
    if data[:3] == b"\xef\xbb\xbf":
        return data.decode("utf-8-sig", errors="replace")
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass

    # 1) Strict UTF-8 first. Russian text in a single-byte codec is almost
    #    never coincidentally valid UTF-8, so success here is decisive.
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 2) Legacy single-byte: gather candidates, pick the most plausible.
    candidates: list[str] = []
    try:
        for match in list(from_bytes(data))[:5]:
            candidates.append(str(match))
    except Exception as e:  # pragma: no cover - defensive
        log.debug("charset-normalizer failed: %s", e)
    for enc in _RU_FALLBACKS:
        try:
            candidates.append(data.decode(enc))
        except (UnicodeDecodeError, LookupError):
            continue

    best_text = None
    best_score = -1e9
    for text in candidates:
        score = _text_score(text[:_DECODE_SCORE_SAMPLE_CHARS])
        if score > best_score:
            best_score, best_text = score, text
    if best_text is None:
        best_text = data.decode("utf-8", errors="replace")
    return best_text


def _read_text_file(path: Path) -> str:
    return decode_bytes(path.read_bytes())


# --- HTML / XHTML -----------------------------------------------------

def _strip_html(markup: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(["sup", "sub", "a"]):
        if tag.attrs is None:  # An ancestor was already decomposed.
            continue
        attrs = f"{tag.get('epub:type', '')} {tag.get('role', '')} {' '.join(tag.get('class', []))}".lower()
        footnote_link = (
            tag.name in {"sup", "sub"}
            and any(
                re.match(r"#(?:fn|note|footnote|endnote)(?:\d|[-_])",
                         str(link.get("href", "")).lower())
                for link in tag.find_all("a")
            )
        )
        if (
            tag.name in {"sup", "sub", "a"}
            and ("noteref" in attrs or "doc-noteref" in attrs
                 or (tag.name in {"sup", "sub"} and re.search(r"(?:footnote|endnote|fnref)", attrs))
                 or (tag.name in {"sup", "sub"} and footnote_link))
        ):
            tag.decompose()
        elif tag.name in {"sup", "sub"}:
            tag.insert_before(" ")
            tag.insert_after(" ")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # Insert separators at block boundaries only. get_text("\n") splits inline
    # spans too, turning e.g. "сло<b>во</b>" into two spoken words.
    blocks = {"address", "article", "blockquote", "dd", "div", "dl", "dt",
              "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main",
              "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul"}
    for tag in soup.find_all(blocks):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    return _html.unescape(text)


# --- chapter splitting heuristics ------------------------------------

# A word ordinal after "Глава"/"Часть" ("Глава двадцать третья"). A bare
# "[а-яё]+" would also accept prose such as "Глава семьи вошёл в комнату".
_ORDINAL_WORD = (
    r"(?:(?:двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|"
    r"девяносто|сто)[ \t]+)?"
    r"(?:перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|одиннадцат|"
    r"двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат|семнадцат|"
    r"восемнадцат|девятнадцат|двадцат|тридцат|сороков|пятидесят|шестидесят|"
    r"семидесят|восьмидесят|девяност|сот|последн|заключительн)[а-яё]*"
)
# A heading is a whole short line. After the number only a separator or a
# capitalised title may follow — never lowercase prose continuing a sentence.
_HEADING_TAIL = r"(?:[ \t]*[.:—–-]?[ \t]*[А-ЯЁA-Z«\"(\d][^\n]{0,80})?[ \t]*[.:]?"
_CHAPTER_RE = re.compile(
    r"^[ \t]*("
    r"(?i:глава|часть)[ \t]+(?:[\dIVXLCDMМ]+(?:-[а-яё]{1,2})?\b|(?i:" + _ORDINAL_WORD + r"))" + _HEADING_TAIL
    + r"|(?i:пролог|эпилог|вступление|предисловие|послесловие)"
    r"(?:[ \t]*[.:—–-][ \t]*[^\n]{0,80})?"
    r"|(?i:chapter)[ \t]+\d+\b" + _HEADING_TAIL
    + r")[ \t]*$",
    re.MULTILINE,
)
_PART_HEADING_RE = re.compile(r"(?i:часть)\b")


def _split_plain_into_chapters(text: str) -> list[Chapter]:
    """Detect chapter headings in plain text. If none found, return a single
    untitled chapter; the assembler treats that as 'no chapter structure'."""
    # CP1251 TXT is usually CRLF; the heading regex anchors on bare "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_CHAPTER_RE.finditer(text))
    if len(matches) < 2:
        return [Chapter(title=None, text=text.strip())]

    chapters: list[Chapter] = []
    # Text before the first heading (foreword etc.) becomes its own chapter.
    head = text[: matches[0].start()].strip()
    if head:
        chapters.append(Chapter(title=None, text=head))

    pending_part: str | None = None
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(1).strip()
        # The heading line is excluded from the body so it isn't read twice.
        body = text[m.end():end].strip()
        if not body:
            # A table-of-contents line or a part heading directly followed by
            # its first chapter. Voicing the title as the body would read it
            # twice; a part title is carried into the next chapter instead.
            if _PART_HEADING_RE.match(title):
                pending_part = title
            continue
        if pending_part:
            title = f"{pending_part.rstrip('.:')}. {title}"
            pending_part = None
        chapters.append(Chapter(title=title, text=body))
    if not chapters:
        return [Chapter(title=None, text=text.strip())]
    return chapters


# --- format handlers --------------------------------------------------

def _extract_txt(path: Path) -> Document:
    text = _read_text_file(path)
    return Document(title=None, author=None, chapters=_split_plain_into_chapters(text))


def _extract_markdown(path: Path) -> Document:
    raw = _read_text_file(path)
    # Render to HTML then strip, so we drop markup but keep headings as text.
    try:
        from markdown_it import MarkdownIt

        html_text = MarkdownIt("commonmark").render(raw)
        text = _strip_html(html_text)
    except Exception:
        text = raw
    return Document(title=None, author=None, chapters=_split_plain_into_chapters(text))


def _extract_html(path: Path) -> Document:
    raw = _read_text_file(path)
    text = _strip_html(raw)
    return Document(title=None, author=None, chapters=_split_plain_into_chapters(text))


_RTF_CPG_RE = re.compile(rb"\\ansicpg(\d+)")


def _extract_rtf(path: Path) -> Document:
    from striprtf.striprtf import rtf_to_text

    raw = path.read_bytes()
    # RTF stores Cyrillic as \'xx byte escapes interpreted in the code page
    # declared by \ansicpgNNNN (1251 for Russian). striprtf uses its `encoding`
    # arg for those bytes, so we must pass the right one or get mojibake.
    encoding = "cp1251"
    m = _RTF_CPG_RE.search(raw[:4096])
    if m:
        encoding = f"cp{m.group(1).decode()}"
    # The control words themselves are ASCII; latin-1 keeps the bytes 1:1.
    try:
        text = rtf_to_text(
            raw.decode("latin-1", errors="ignore"),
            encoding=encoding,
            errors="ignore",
        )
    except LookupError:
        text = rtf_to_text(raw.decode("latin-1", errors="ignore"), errors="ignore")
    except Exception as e:
        raise ExtractError(f"Не удалось разобрать RTF: {e}") from e
    if not text.strip():
        raise ExtractError("RTF не содержит читаемого текста")
    return Document(title=None, author=None, chapters=_split_plain_into_chapters(text))


def _extract_fb2(path: Path) -> Document:
    from lxml import etree

    data = path.read_bytes()
    # FB2 declares its own encoding; let lxml honour it, recover from glitches.
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ExtractError(f"Повреждённый FB2: {e}") from e
    if root is None:
        raise ExtractError("FB2 не удалось разобрать")

    ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}

    def _find(xpath: str):
        res = root.xpath(xpath, namespaces=ns)
        return res

    # Metadata.
    title = None
    author = None
    series = None
    series_index = None
    t = _find("//fb:description/fb:title-info/fb:book-title/text()")
    if t:
        title = str(t[0]).strip()
    fn = _find("//fb:description/fb:title-info/fb:author/fb:first-name/text()")
    ln = _find("//fb:description/fb:title-info/fb:author/fb:last-name/text()")
    name_parts = [str(x).strip() for x in (fn[:1] + ln[:1])]
    if name_parts:
        author = " ".join(p for p in name_parts if p)
    sequences = _find("//fb:description/fb:title-info/fb:sequence")
    if sequences:
        sequence = sequences[0]
        series = _metadata_text(sequence.get("name"))
        series_index = _metadata_text(sequence.get("number"))

    bodies = _find("//fb:body")
    if not bodies:
        # Some FB2 omit the namespace; retry namespace-agnostically.
        bodies = root.xpath("//*[local-name()='body']")

    chapters: list[Chapter] = []

    def _text_blocks(container, excluded: set | None = None) -> list[str]:
        """Collect outer FB2 prose blocks once, retaining inline text order."""
        excluded = excluded or set()
        blocks: list[str] = []
        stack = list(reversed(list(container)))
        while stack:
            child = stack.pop()
            if child in excluded or not isinstance(child.tag, str):
                continue
            localname = etree.QName(child).localname
            if localname in {"p", "v", "subtitle", "text-author"}:
                # Do not descend into malformed nested block elements after
                # collecting the outer block: itertext already includes each
                # descendant exactly once, in document order.
                text = "".join(child.itertext()).strip()
                if text:
                    blocks.append(text)
            else:
                stack.extend(reversed(list(child)))
        return blocks

    def _join_titles(first: str | None, second: str | None) -> str | None:
        if first and second:
            return f"{first.rstrip('.:')}. {second}"
        return first or second

    def _walk_section(section, lead_title: str | None, depth: int = 0) -> None:
        """Leaf sections are chapters; "Часть → Глава" nesting keeps marks.

        A parent title with no prose of its own is carried into its first
        child's title so it is neither lost nor voiced as a separate body."""
        title_els = section.xpath("./*[local-name()='title']")
        title_nodes = []
        for tel in title_els:
            title_nodes.extend(tel.xpath(".//text()"))
        sec_title = " ".join(t.strip() for t in title_nodes if t.strip()) or None
        title = _join_titles(lead_title, sec_title)
        # Paragraphs living inside the section's own <title> are the title —
        # keep them out of the body, or the title gets voiced twice
        # (build_plan prepends the title before the body).
        excluded = {el for tel in title_els for el in tel.iter()}
        children = section.xpath("./*[local-name()='section']")
        if depth >= _FB2_MAX_SECTION_DEPTH:
            children = []  # Pathological nesting: treat the rest as one chapter.
        excluded |= {el for child in children for el in child.iter()}
        # FB2 poetry stores lines in <v>, not <p>. Keep all common prose and
        # verse blocks, including those in notes bodies; neither is metadata.
        own = "\n\n".join(_text_blocks(section, excluded))
        if own.strip():
            chapters.append(Chapter(title=title, text=own))
            title = None
        for i, child in enumerate(children):
            _walk_section(child, title if i == 0 else None, depth + 1)

    for body in bodies:
        sections = body.xpath("./*[local-name()='section']")
        if sections:
            for sec in sections:
                _walk_section(sec, None)
        else:
            # Flat bodies occur in older FB2 files and may contain poetry.
            blocks = _text_blocks(body)
            text = "\n\n".join(blocks)
            if text.strip():
                chapters.append(Chapter(title=None, text=text))

    if not chapters:
        raise ExtractError("FB2 не содержит текста")
    return Document(
        title=_metadata_text(title),
        author=_metadata_text(author),
        chapters=chapters,
        series=series,
        series_index=series_index,
    )


def _metadata_text(value: object | None) -> str | None:
    """Normalize optional bibliographic fields without changing their meaning."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _epub_series_metadata(path: Path) -> tuple[str | None, str | None]:
    """Read Calibre and EPUB 3 series fields from the package OPF, best effort."""
    from lxml import etree

    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        with zipfile.ZipFile(path) as archive:
            container = etree.fromstring(
                archive.read("META-INF/container.xml"), parser=parser
            )
            rootfiles = container.xpath("//*[local-name()='rootfile']")
            if not rootfiles:
                return None, None
            opf = etree.fromstring(
                archive.read(rootfiles[0].get("full-path")), parser=parser
            )
    except Exception:
        return None, None

    series = None
    series_index = None
    collection_ids: dict[str, str] = {}
    positions: dict[str, str] = {}
    for meta in opf.xpath("//*[local-name()='metadata']/*[local-name()='meta']"):
        name = meta.get("name")
        prop = meta.get("property")
        value = _metadata_text(meta.get("content") or "".join(meta.itertext()))
        if name == "calibre:series":
            series = value
        elif name == "calibre:series_index":
            series_index = value
        elif prop == "belongs-to-collection" and value:
            collection_ids[meta.get("id", "")] = value
            series = series or value
        elif prop == "group-position" and value:
            positions[(meta.get("refines") or "").lstrip("#")] = value

    if series_index is None and collection_ids:
        for collection_id in collection_ids:
            if collection_id in positions:
                series_index = positions[collection_id]
                break
    return _metadata_text(series), _metadata_text(series_index)


def _validate_epub_archive(path: Path) -> None:
    """Reject pathological zip expansion before ebooklib decompresses members."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as e:
        raise ExtractError("EPUB не является корректным zip-архивом") from e

    if len(members) > _EPUB_MAX_FILES:
        raise ExtractError("EPUB содержит патологически много файлов")

    expanded = 0
    compressed = 0
    for member in members:
        if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ExtractError("EPUB использует неподдерживаемый метод сжатия")
        if member.file_size > _EPUB_MAX_MEMBER_BYTES:
            raise ExtractError("EPUB содержит патологически большой вложенный файл")
        if (
            member.file_size > _EPUB_MEMBER_RATIO_CHECK_BYTES
            and member.file_size
            > max(member.compress_size, 1) * _EPUB_MAX_EXPANSION_RATIO
        ):
            raise ExtractError("EPUB имеет подозрительно высокий коэффициент сжатия")
        expanded += member.file_size
        compressed += member.compress_size
        if expanded > _EPUB_MAX_EXPANDED_BYTES:
            raise ExtractError("EPUB слишком велик после распаковки")

    if (
        expanded > _EPUB_RATIO_CHECK_BYTES
        and expanded > max(compressed, 1) * _EPUB_MAX_EXPANSION_RATIO
    ):
        raise ExtractError("EPUB имеет подозрительно высокий коэффициент сжатия")


def _extract_epub(path: Path) -> Document:
    _validate_epub_archive(path)

    import ebooklib
    from ebooklib import epub

    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": False})
    except Exception as e:
        # ebooklib can choke on slightly broken zips; verify it's a zip first.
        if not zipfile.is_zipfile(path):
            raise ExtractError("EPUB не является корректным zip-архивом") from e
        # A broken/missing NCX TOC is common in the wild and irrelevant for
        # us (we read the spine) — retry without it before giving up.
        try:
            book = epub.read_epub(str(path), options={"ignore_ncx": True})
        except Exception:
            raise ExtractError(f"Не удалось открыть EPUB: {e}") from e

    title = None
    author = None
    series = None
    series_index = None
    try:
        md_title = book.get_metadata("DC", "title")
        if md_title:
            title = str(md_title[0][0]).strip()
        md_author = book.get_metadata("DC", "creator")
        if md_author:
            author = str(md_author[0][0]).strip()
    except Exception:
        pass
    series, series_index = _epub_series_metadata(path)

    # Build a spine-ordered list of document items.
    chapters: list[Chapter] = []
    items_by_id = {it.get_id(): it for it in book.get_items()}
    spine_ids = [sid for sid, linear in book.spine if linear != "no"] if book.spine else []

    ordered = []
    for sid in spine_ids:
        it = items_by_id.get(sid)
        if (it is not None and it.get_type() == ebooklib.ITEM_DOCUMENT
                and "nav" not in getattr(it, "properties", [])):
            ordered.append(it)
    if not book.spine:
        ordered = [
            it for it in book.get_items()
            if it.get_type() == ebooklib.ITEM_DOCUMENT
            and "nav" not in getattr(it, "properties", [])
        ]

    for it in ordered:
        try:
            content = it.get_content()
        except Exception:
            continue
        text = _strip_html(content.decode("utf-8", errors="replace"))
        text = text.strip()
        if not text:
            continue
        # Use the first non-empty line as a candidate chapter title, and drop
        # it from the body — build_plan prepends the title before the body,
        # so leaving it in would voice the heading twice.
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        ch_title = first_line if 0 < len(first_line) <= 80 else None
        body = text
        if ch_title and text.startswith(first_line):
            rest = text[len(first_line):].strip()
            if rest:
                body = rest
            else:
                ch_title = None  # the whole chapter is just this line
        chapters.append(Chapter(title=ch_title, text=body))

    if not chapters:
        raise ExtractError("EPUB не содержит читаемых глав")
    return Document(
        title=_metadata_text(title),
        author=_metadata_text(author),
        chapters=chapters,
        series=series,
        series_index=series_index,
    )


_HANDLERS = {
    ".txt": _extract_txt,
    ".text": _extract_txt,
    ".md": _extract_markdown,
    ".markdown": _extract_markdown,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".xhtml": _extract_html,
    ".rtf": _extract_rtf,
    ".fb2": _extract_fb2,
    ".epub": _extract_epub,
}


def extract(path: Path) -> Document:
    """Dispatch on extension. Raises ExtractError (Russian message) on any
    unsupported / corrupt / empty input."""
    path = Path(path)
    if not path.exists():
        raise ExtractError("Файл не найден")
    if path.stat().st_size == 0:
        raise ExtractError("Файл пустой")

    ext = path.suffix.lower()
    handler = _HANDLERS.get(ext)
    if handler is None:
        # Unknown extension: only accept if it decodes to mostly-text.
        if ext not in ALL_EXTS:
            with path.open("rb") as stream:
                sample = stream.read(8192)
            text = decode_bytes(sample)
            printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
            if not text or printable / max(len(text), 1) < 0.85:
                raise ExtractError(
                    f"Формат «{ext or 'без расширения'}» не поддерживается"
                )
            handler = _extract_txt

    doc = handler(path)

    if doc.char_count < 1:
        raise ExtractError("Не удалось извлечь текст из файла")
    if doc.char_count < 30:
        raise ExtractError("В файле слишком мало текста для озвучивания")
    return doc
