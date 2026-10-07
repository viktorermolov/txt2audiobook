from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


class ExtractError(Exception):
    """Raised when a source file cannot be turned into usable text.
    The message is shown to the user, so keep it human and in Russian."""


@dataclass
class Chapter:
    title: str | None
    text: str


@dataclass
class Document:
    title: str | None
    author: str | None
    chapters: list[Chapter] = field(default_factory=list)
    # Kept after chapters so existing positional constructors remain valid.
    series: str | None = None
    series_index: str | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chapters if c.text.strip())

    @property
    def char_count(self) -> int:
        return sum(len(c.text) for c in self.chapters)


# --- file-name safety -------------------------------------------------

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_DOTS = re.compile(r"\.+")
_SPACES = re.compile(r"\s+")


def safe_filename(name: str, fallback: str = "audiobook", max_len: int = 120) -> str:
    """Produce a name safe for Linux filesystems and Telegram document names.
    Keeps Cyrillic; strips path separators, control chars and trailing dots."""
    name = unicodedata.normalize("NFC", name or "").strip()
    name = _UNSAFE.sub(" ", name)
    name = name.replace("/", " ").replace("\\", " ")
    name = _SPACES.sub(" ", name).strip()
    name = _DOTS.sub(".", name).strip(". ")
    if not name:
        name = fallback
    # Linux limits filename components by encoded bytes, not Python characters.
    # Leave callers' existing byte budget intact for extensions/part suffixes.
    encoded = name.encode("utf-8")
    if len(encoded) > max_len:
        name = encoded[:max_len].decode("utf-8", errors="ignore").rstrip(". ")
    # Avoid reserved bare names.
    if name.upper() in {"CON", "PRN", "AUX", "NUL"}:
        name = f"_{name}"
    return name or fallback
