"""Russian text normalization for TTS.

Responsibilities (quality-critical):
  * Whitespace / blank-line / page-number / footnote cleanup, de-hyphenation,
    soft-wrap re-flow that preserves dialogue lines.
  * Quote handling: «ёлочки», "умные", "прямые", nested — normalized so they
    never break sentence boundaries and read naturally (silently).
  * Abbreviation handling: expand a high-confidence set into spoken words
    (и т. д. → и так далее) and protect the rest (initials, ул., д.) so
    sentence splitting does not fire on their periods.
  * Number expansion to words via num2words (ru), with thousands separators,
    simple decimals, and common ordinal suffixes.
  * Sentence splitting that respects abbreviations, initials, and closing
    quotes/brackets after terminal punctuation.

The goal is natural-enough speech without over-cleaning ("не переочищать").
"""

from __future__ import annotations

import re
from datetime import date

from num2words import num2words

from ..log import get_logger
from .types import Chapter, Document

log = get_logger("clean")

_SENTINEL = ""  # stands in for a protected period during splitting

# ---------------------------------------------------------------------
# Whitespace & artifacts
# ---------------------------------------------------------------------

_NBSP = "   "
_SOFT_HYPHEN = "­"
# Ligature punctuation is outside Silero's symbols and was silently dropped,
# losing both the intonation and the sentence break ("Что⁈ Мы уходим").
_PUNCT_LIGATURES = str.maketrans({"⁈": "?!", "⁉": "!?", "‼": "!!", "⁇": "??"})

# Only decorated folios are unambiguous enough to discard. A bare number may
# be a chapter heading, list item, verse, or mathematical content.
_PAGE_NUM_RE = re.compile(r"^\s*[-—–]\s*\d{1,4}\s*[-—–]\s*$")
# Bracketed footnote/reference markers: [1] {12} [12] — drop inline.
_FOOTNOTE_RE = re.compile(r"[\[\{]\s*\d{1,3}\s*[\]\}]")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# De-hyphenation across a line break: "сло-\nво" -> "слово".
# The \b anchor keeps a long unspaced \w run linear instead of rescanning it
# from every position.
_HYPHEN_BREAK_RE = re.compile(r"(\b\w*[а-яёА-ЯЁ])-[ \t]*\n\s*([а-яёa-z]\w*)")
# A line break inside a real hyphenated compound ("кто-\nнибудь") must keep
# the hyphen. Only unambiguous particles are recognised: "то"/"ка"/"таки" can
# also be the tail of a split word ("мес-\nто"), so they need a known head.
_ALWAYS_HYPHENATED_TAILS = {"нибудь", "либо"}
_PARTICLE_TAILS = {"то", "ка", "таки"}
_PARTICLE_HEADS = {
    "что", "кто", "где", "куда", "когда", "как", "какой", "какая", "какое",
    "какие", "чей", "чья", "чьё", "почему", "зачем", "откуда", "отчего",
    "сколько", "всё", "все", "давай", "дай", "ну", "скажи", "смотри",
}
# 3+ blank lines collapse to a paragraph break.
_MULTI_BLANK_RE = re.compile(r"\n\s*\n(\s*\n)+")


def _join_line_hyphen(match: re.Match) -> str:
    head, tail = match.group(1), match.group(2)
    lower_tail = tail.lower()
    if (
        lower_tail in _ALWAYS_HYPHENATED_TAILS
        or head.lower() == "кое"
        or (lower_tail in _PARTICLE_TAILS and head.lower() in _PARTICLE_HEADS)
    ):
        return f"{head}-{tail}"
    return head + tail


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_SOFT_HYPHEN, "")
    for ch in _NBSP:
        text = text.replace(ch, " ")
    text = _HYPHEN_BREAK_RE.sub(_join_line_hyphen, text)
    # Trim trailing spaces on each line.
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def _strip_artifacts(text: str) -> str:
    out_lines: list[str] = []
    for line in text.split("\n"):
        if _PAGE_NUM_RE.match(line):
            continue
        line = _FOOTNOTE_RE.sub("", line)
        line = _SUPERSCRIPT_RE.sub(lambda m: " " + m.group().translate(_SUPERSCRIPT_DIGITS), line)
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text


# ---------------------------------------------------------------------
# Soft-wrap reflow
# ---------------------------------------------------------------------

_TERMINALS = ".!?…»\"”)"
_DIALOGUE_START_RE = re.compile(r"^\s*[—–-]\s")


def _reflow(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph into flowing text, while
    keeping paragraph breaks (blank lines) and dialogue lines (starting with
    a dash) intact."""
    paragraphs = re.split(r"\n\s*\n", text)
    result: list[str] = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        merged: list[str] = []
        for ln in lines:
            if not merged:
                merged.append(ln)
                continue
            prev = merged[-1]
            # Keep dialogue and heading-like lines separate.
            if _DIALOGUE_START_RE.match(ln):
                merged.append(ln)
                continue
            # If previous line ended a sentence, start a new logical line.
            if prev and prev[-1] in _TERMINALS:
                merged.append(ln)
            else:
                merged[-1] = prev + " " + ln
        result.append("\n".join(merged))
    return "\n\n".join(result)


# ---------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------

def _normalize_quotes(text: str) -> str:
    # Unify smart double quotes to «ёлочки» feel is unnecessary for speech;
    # we only ensure straight/curly quotes are consistent and don't glue to
    # punctuation. Map curly to straight equivalents.
    replacements = {
        "“": "«", "”": "»", "„": "«", "‟": "»",
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "''": '"',
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


# ---------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------

# Multi-word / dotted abbreviations expanded into spoken words.
# Order matters: longer patterns first. \. allows optional space between parts.
_CLOSERS = "»\"”)]"
# What may follow an abbreviation's period: space, end, comma/semicolon, or a
# closing quote/bracket ("в 1945 г.»").
_ABBR_END = r"(?=[\s»\"”)\]]|$|[,;])"


def _sentence_period(match: re.Match) -> str:
    """"." when the abbreviation's period also ends the sentence, else "".

    An expanded abbreviation consumes its period; re-adding it mid-sentence
    ("в 1905 г. он родился") would make Silero pause with falling intonation.
    """
    rest = match.string[match.end():].lstrip(" \t")
    # «Он умер в 1945 г.» Все молчали. — the sentence ends inside the quote.
    if rest[:1] in _CLOSERS:
        rest = rest.lstrip(_CLOSERS + " \t")
    if not rest or rest[0] == "\n" or rest[0].isupper() or rest[0] in "«\"„“":
        return "."
    return ""


# Multi-word / dotted abbreviations expanded into spoken words.
# Order matters: longer patterns first. \. allows optional space between parts.
# Ambiguous short forms stay unexpanded: "им." is far more often the pronoun
# ("горжусь им.") and "рис." the grain than the abbreviations.
_ABBR_PATTERNS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"\bи\s*т\.?\s*д\.", re.I), lambda m: "и так далее" + _sentence_period(m)),
    (re.compile(r"\bи\s*т\.?\s*п\.", re.I), lambda m: "и тому подобное" + _sentence_period(m)),
    (re.compile(r"\bи\s*т\.?\s*д\b", re.I), "и так далее"),
    # The first period is mandatory: "т\.?е\." would also eat the word "те.".
    (re.compile(r"\bт\.\s*е\.", re.I), "то есть"),
    (re.compile(r"\bт\.\s*к\.", re.I), "так как"),
    (re.compile(r"\bт\.\s*н\.", re.I), "так называемый"),
    (re.compile(r"\bи\s*др\.", re.I), "и другие"),
    (re.compile(r"\bи\s*пр\.", re.I), "и прочее"),
    (re.compile(r"\bнапр\.", re.I), "например"),
    (re.compile(r"\bсм\.", re.I), "смотри"),
    (re.compile(r"\bср\.", re.I), "сравни"),
    (re.compile(r"\bстр\.", re.I), "страница"),
    (re.compile(r"\bс\.\s*(?=\d)", re.I), "страница "),
    (re.compile(r"\bгл\.", re.I), "глава"),
    (re.compile(r"\bрис\.\s*(?=\d)", re.I), "рисунок "),
    (re.compile(r"\bтабл\.", re.I), "таблица"),
    # "театр им. Пушкина" — only after an institution noun, never the pronoun.
    (re.compile(
        r"\b((?i:театр|музе|библиотек|школ|университет|институт|академи|гимнази|"
        r"лице|училищ|консерватори|парк|площад|сквер|сад|завод|фабрик|комбинат|"
        r"больниц|стадион|двор|преми|орден|медал|стипенди|колледж|клуб|"
        r"проспект|улиц)\w*)\s+им\.\s*(?=[А-ЯЁ])"
    ), r"\1 имени "),
    (re.compile(r"\bобл\.", re.I), "область"),
]

# Context-dependent address/time abbreviations.
_ABBR_CONTEXT: list[tuple[re.Pattern, str]] = [
    # Only a four-digit number before г. is a year ("1905г." without the space
    # the year rules need); "500 г." is grams, handled by _GRAM_RE.
    (re.compile(r"(?<=\d{4})(?<!\d{5})\s*г\." + _ABBR_END), " год"),
    (re.compile(r"(?<=\d)\s*гг\." + _ABBR_END), " годы"),
    # "г. Москва" -> "город Москва" (followed by a capitalized word)
    (re.compile(r"\bг\.\s*(?=[А-ЯЁ][а-яё])"), "город "),
    (re.compile(r"\bул\.\s*"), "улица "),
    (re.compile(r"\bпер\.\s*(?=[А-ЯЁ])"), "переулок "),
    (re.compile(r"\bпросп\.\s*"), "проспект "),
    (re.compile(r"\bд\.\s*(?=\d)"), "дом "),
    (re.compile(r"\bкв\.\s*(?=\d)"), "квартира "),
]

# Single-letter initials: protect their periods from sentence splitting.
_INITIAL_RE = re.compile(r"(?<![А-ЯЁA-Z])([А-ЯЁA-Z])\.")
# Residual lone abbreviations whose period we protect but don't expand.
_PROTECT_ABBR = ["др", "пр", "т", "е", "к", "н", "г", "д", "стр", "с", "в", "р", "руб", "коп"]


def _expand_abbreviations(text: str) -> str:
    text = _CENTIMETER_RE.sub(_expand_centimeters, text)
    text = _YEAR_RANGE_RE.sub(_expand_year_range, text)
    text = _YEAR_PREP_RE.sub(_expand_prep_year, text)
    text = _ERA_RE.sub(
        lambda m: ("до нашей эры" if m.group(1) else "нашей эры") + _sentence_period(m), text
    )
    text = _YEAR_RE.sub(
        lambda m: f"{_ordinal_to_words(int(m.group(1)), 'й')} год" + _sentence_period(m),
        text,
    )
    text = _YEAR_WORD_RE.sub(_expand_year_word, text)
    text = _GRAM_RE.sub(
        lambda m: f"{m.group(1)} {_unit_form(m.group(1), 'грамм', 'грамма', 'граммов')}"
        + _sentence_period(m), text,
    )
    text = _PERCENT_RE.sub(
        lambda m: f"{m.group(1)} {_unit_form(m.group(1), 'процент', 'процента', 'процентов')}",
        text,
    )
    text = _NUMBER_SIGN_RE.sub(lambda m: "номера " if len(m.group(1)) > 1 else "номер ", text)
    for pat, repl in _ABBR_PATTERNS:
        text = pat.sub(repl, text)
    for pat, repl in _ABBR_CONTEXT:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------
# Roman numerals & Latin transliteration
# ---------------------------------------------------------------------

# Strict roman numeral (1..3999). Single letters are converted only for
# I/V/X — a lone C, D, L or M is more likely an initial or unit than a numeral.
_ROMAN_TOKEN_RE = re.compile(r"\b[IVXLCDM]{1,15}\b")
_ROMAN_VALID_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def _expand_roman(text: str) -> str:
    """Replace roman numerals with arabic digits (later expanded to words).
    Russian books use them for chapters and monarchs (Глава XIV, Пётр I);
    Silero cannot voice Latin letters, so without this they vanish."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if not _ROMAN_VALID_RE.match(tok):
            return tok
        if len(tok) == 1 and tok not in ("I", "V", "X"):
            return tok
        value = _roman_to_int(tok)
        # Centuries are ordinal: "XIX века" -> "19-го века" -> "девятнадцатого".
        century = _CENTURY_AFTER_RE.match(m.string, m.end())
        if century:
            return f"{value}-{_CENTURY_SUFFIX[century.group(1).lower()]}"
        return str(value)

    return _ROMAN_TOKEN_RE.sub(repl, text)


_CENTURY_AFTER_RE = re.compile(r"\s+(век|века|веке|веку|веком|веков|веках)\b", re.I)
_CENTURY_SUFFIX = {
    "век": "й", "века": "го", "веке": "м", "веку": "му", "веком": "ым",
    "веков": "х", "веках": "х",
}


# Digraph-aware Latin→Cyrillic transliteration. Silero's symbol table has no
# Latin letters, so untransliterated words are silently dropped from speech —
# a rough phonetic rendering is far better than a hole in the sentence.
_TRANSLIT_DIGRAPHS = [
    ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"), ("th", "т"),
    ("zh", "ж"), ("kh", "х"), ("qu", "кв"), ("ck", "к"),
    ("ya", "я"), ("yu", "ю"), ("yo", "йо"), ("ja", "я"), ("ju", "ю"),
]
_TRANSLIT_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")


def _transliterate_word(word: str) -> str:
    s = word.lower()
    out: list[str] = []
    i = 0
    while i < len(s):
        for di, (lat, cyr) in enumerate(_TRANSLIT_DIGRAPHS):
            if s.startswith(lat, i):
                out.append(cyr)
                i += len(lat)
                break
        else:
            out.append(_TRANSLIT_SINGLE.get(s[i], s[i]))
            i += 1
    return "".join(out)


def _transliterate_latin(text: str) -> str:
    return _LATIN_RUN_RE.sub(lambda m: _transliterate_word(m.group(0)), text)


# OCR and pirated TXT often mix look-alike Latin letters into Russian words
# ("pека", "cад"). Transliterating them phonetically gives "пека"/"кад", so
# inside words that contain both scripts they are mapped back by shape first.
_HOMOGLYPHS = str.maketrans("aceopxykACEHKMOPTXBY", "асеорхукАСЕНКМОРТХВУ")
_MIXED_SCRIPT_RE = re.compile(r"\b(?=\w*[А-Яа-яЁё])(?=\w*[A-Za-z])\w+\b")


def _fix_homoglyphs(text: str) -> str:
    return _MIXED_SCRIPT_RE.sub(lambda m: m.group(0).translate(_HOMOGLYPHS), text)


# ---------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------

# "10 000" / "1 000 000": the leading group is 1–3 digits and every following
# group exactly 3, so unrelated neighbours ("в 1943 150 танков") stay apart.
_THOUSANDS_RE = re.compile(r"(?<![\d.,])(\d{1,3})((?:[   ]\d{3})+)(?!\d)")
# decimal like 3,14 or 3.14
_DECIMAL_RE = re.compile(r"(?<![\w.])(\d+)[.,](\d+)(?!\w|\.[\w])")
# ordinal like 5-й, 2-го, 21-я, 3-е, 10-м, 7-х, 1-ый, 90-ые
_ORDINAL_RE = re.compile(
    r"\b(\d+)-(ого|ому|ую|ым|ых|ые|ый|ий|ая|ое|ее|го|му|ой|й|м|х|я|е|ю)\b"
)
# "90-е годы" is plural ("девяностые"), not the neuter singular "девяностое".
_DECADE_RE = re.compile(r"\b(\d+)-е(?=\s+год)")
# Grams: 1–3 digits (four digits are years) followed by a lowercase word.
_GRAM_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d+)?)\s*г\.?(?=\s+[а-яё])")
_PERCENT_RE = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*%")
_NUMBER_SIGN_RE = re.compile(r"(№+)\s*(?=\d)")
# "в 1812 году" is ordinal. "году"/"годе" always are; "год"/"года"/"годом"
# only for four-digit years, because "3 года" is a count, not a date.
_YEAR_WORD_RE = re.compile(r"(?<![\w.,-])(\d{1,4})\s+(году|годе|годом|года|год)\b")
_DATE_RE = re.compile(r"(?<![\w.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\w|\.[\w])")
_CENTIMETER_RE = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*см\.(?=\s|$|[,;])", re.I)
# "в 988 г.", "к 1812 г.", "300 г. до н. э.": a number before "г." after a
# preposition is a year when an era follows, when it has four digits, or when
# it has three digits and is not an amount. An amount ("по 300 г. муки",
# "около 200 г. масла") needs a following lowercase content word; "в"/"к"
# always mean a year ("в 100 г. муки" is the accepted trade-off).
_YEAR_PREP_RE = re.compile(
    r"(?<![\w.,])(?:(?P<prep>[Вв]|[Кк]о?|[Сс]о?|[Дд]о|[Пп]осле|[Оо]коло|[Пп]о)\s+)?"
    r"(?P<n>\d{1,4})\s*г\." + _ABBR_END
)
# "с 1941 по 1945 г.", "с 1914 по 1918 гг.": the range start has no "г." of
# its own and would otherwise be read as a cardinal number.
_YEAR_RANGE_RE = re.compile(
    r"(?<![\w.,])(?P<p1>[Сс]о?)\s+(?P<a>\d{3,4})\s+по\s+(?P<b>\d{3,4})\s*гг?\." + _ABBR_END
)
# "с 879 по 912 г." / "с 1941 г. по 1945 г.": "по" ends a range (accusative);
# elsewhere ("данные по 2020 г.") it means "concerning" (dative).
_RANGE_START_BEHIND_RE = re.compile(
    r"\b[Сс]о?\s+\d{1,4}(?:\s*г\.)?(?:\s*(?:до\s+)?н\.\s*э\.)?\s+$"
)
_ERA_AHEAD_RE = re.compile(r"\s*(?:до\s+)?н\.\s*э\.")
_WORD_AHEAD_RE = re.compile(r"\s+([а-яё]+)")
# Lowercase words that continue a sentence after a year, never a measured noun.
_YEAR_FOLLOWERS = frozenset(
    "и а но или на в во до после к ко по при с со от из за у о об про через же ли "
    "он она оно они мы я ты вы это этот эта эти тот та те был была было были "
    "уже ещё еще когда тогда здесь там его её ее их".split()
)
_YEAR_PREP_FORMS = {
    "в": ("м", "году"), "к": ("му", "году"), "ко": ("му", "году"), "по": ("му", "году"),
    "с": ("го", "года"), "со": ("го", "года"), "до": ("го", "года"),
    "после": ("го", "года"), "около": ("го", "года"),
}
_ERA_RE = re.compile(r"\b(до\s+)?н\.\s*э\.")
_YEAR_RE = re.compile(r"(?<!\w)(\d{4})\s+г\." + _ABBR_END)
_MONTHS_GENITIVE = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
# bare integer (after thousands separators removed)
_INTEGER_RE = re.compile(r"\b\d+\b")

_FRAC_NAMES = {
    1: "десятых",
    2: "сотых",
    3: "тысячных",
    4: "десятитысячных",
}

def _int_to_words(n: int) -> str:
    try:
        return num2words(n, lang="ru")
    except Exception:
        return str(n)


def _unit_form(value: str, one: str, few: str, many: str) -> str:
    """Russian noun form after a number: 1 грамм, 2 грамма, 5 граммов;
    fractional values take the "few" form (2,5 грамма)."""
    if "." in value or "," in value:
        return few
    n = int(value)
    last_two = n % 100
    last = n % 10
    if last == 1 and last_two != 11:
        return one
    if 2 <= last <= 4 and not 12 <= last_two <= 14:
        return few
    return many


def _expand_centimeters(match: re.Match) -> str:
    value = match.group(1)
    return f"{value} {_unit_form(value, 'сантиметр', 'сантиметра', 'сантиметров')}."


def _expand_year_range(match: re.Match) -> str:
    start = _ordinal_to_words(int(match.group("a")), "го")
    end = _ordinal_to_words(int(match.group("b")), "й")
    return f"{match.group('p1')} {start} по {end} год" + _sentence_period(match)


def _expand_prep_year(match: re.Match) -> str:
    prep, digits = match.group("prep"), match.group("n")
    n = int(digits)
    p = prep.lower() if prep else None
    era = _ERA_AHEAD_RE.match(match.string, match.end())
    range_end = p == "по" and _RANGE_START_BEHIND_RE.search(
        match.string, max(0, match.start() - 40), match.start()
    )
    word = _WORD_AHEAD_RE.match(match.string, match.end())
    amount = word is not None and word.group(1) not in _YEAR_FOLLOWERS
    short_year = p is not None and n >= 100 and (p in {"в", "к", "ко"} or range_end or not amount)
    if not (era or (p and len(digits) == 4) or short_year):
        return match.group(0)  # an amount (grams) or a bare 4-digit _YEAR_RE
    if not prep:
        return f"{_ordinal_to_words(n, 'й')} год" + _sentence_period(match)
    suffix, noun = ("й", "год") if range_end else _YEAR_PREP_FORMS[p]
    return f"{prep} {_ordinal_to_words(n, suffix)} {noun}" + _sentence_period(match)


def _expand_year_word(match: re.Match) -> str:
    n = int(match.group(1))
    word = match.group(2)
    if word in ("год", "года", "годом") and not 1000 <= n <= 2999:
        return match.group(0)
    if word == "году":
        before = match.string[:match.start()].rstrip()
        prev = before.rsplit(None, 1)[-1].lower() if before else ""
        suffix = "му" if prev in ("к", "ко", "по") else "м"
    else:
        suffix = {"годе": "м", "год": "й", "года": "го", "годом": "ым"}[word]
    return f"{_ordinal_to_words(n, suffix)} {word}"


def _ordinal_to_words(n: int, suffix: str) -> str:
    try:
        base = num2words(n, lang="ru", to="ordinal")
    except Exception:
        return f"{n}-{suffix}"
    # Full endings written in books ("1-ый", "2-ая") mean the short forms.
    suffix = {"ый": "й", "ий": "й", "ая": "я", "ое": "е", "ее": "е"}.get(suffix, suffix)
    if suffix == "й":
        return base
    endings = {
        "го": "ого", "ого": "ого", "му": "ому", "ому": "ому",
        "м": "ом", "х": "ых", "я": "ая", "е": "ое",
        "ю": "ую", "ой": "ой", "ую": "ую", "ым": "ым", "ых": "ых",
        "ые": "ые",
    }
    if base.endswith("третий"):
        soft = {
            "го": "третьего", "ого": "третьего", "му": "третьему",
            "ому": "третьему", "м": "третьем", "х": "третьих",
            "я": "третья", "е": "третье", "ю": "третью",
            "ой": "третьей", "ую": "третью", "ым": "третьим", "ых": "третьих",
            "ые": "третьи",
        }
        return base[:-6] + soft[suffix]
    if base.endswith(("ый", "ой")):
        return base[:-2] + endings[suffix]
    return base


def _date_to_words(match: re.Match) -> str:
    day, month, year = map(int, match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return match.group()
    return f"{_ordinal_to_words(day, 'го')} {_MONTHS_GENITIVE[month - 1]} {_ordinal_to_words(year, 'го')} года"


def _decimal_to_words(int_part: str, frac_part: str) -> str:
    whole = _int_to_words(int(int_part))
    frac = _int_to_words(int(frac_part))
    denom = _FRAC_NAMES.get(len(frac_part), "")
    # "целых" used uniformly; minor grammar imperfection, acceptable for TTS.
    if denom:
        return f"{whole} целых {frac} {denom}"
    return f"{whole} запятая {frac}"


def _expand_numbers(text: str) -> str:
    text = _THOUSANDS_RE.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), text)
    text = _DATE_RE.sub(_date_to_words, text)
    text = _DECIMAL_RE.sub(lambda m: _decimal_to_words(m.group(1), m.group(2)), text)
    text = _DECADE_RE.sub(lambda m: _ordinal_to_words(int(m.group(1)), "ые"), text)
    text = _ORDINAL_RE.sub(
        lambda m: _ordinal_to_words(int(m.group(1)), m.group(2)), text
    )
    text = _INTEGER_RE.sub(lambda m: _int_to_words(int(m.group(0))), text)
    return text


# ---------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------

_BOUNDARY_RE = re.compile(
    r"([.!?…]+[»\"'”)\]]*)\s+(?=[«\"'“(\[—–-]?[А-ЯЁA-Z0-9])"
)


def _protect_periods(text: str) -> str:
    """Replace periods that belong to initials / protected abbreviations with
    a sentinel so the boundary regex won't split on them.

    Initials (single capital letter) are protected unconditionally — they are
    routinely followed by another capital (А. С. Пушкин). Residual lone
    abbreviations are protected only when NOT followed by an uppercase word,
    so a real sentence end like "5 руб. Это дёшево." still splits."""
    text = _INITIAL_RE.sub(lambda m: m.group(1) + _SENTINEL, text)
    for abbr in _PROTECT_ABBR:
        text = re.sub(
            rf"\b{abbr}\.(?!\s+[А-ЯЁA-Z])",
            abbr + _SENTINEL,
            text,
            flags=re.IGNORECASE,
        )
    return text


def split_sentences(text: str) -> list[str]:
    protected = _protect_periods(text)
    # Insert a split marker after each boundary.
    marked = _BOUNDARY_RE.sub(r"\1\n", protected)
    sentences = []
    for piece in marked.split("\n"):
        piece = piece.replace(_SENTINEL, ".").strip()
        if piece:
            # Lines without final punctuation (headings, verse, epigraphs) are
            # later joined with a space; without a mark Silero runs them into
            # the next sentence with no pause ("Часть вторая Было утро").
            if piece[-1].isalnum():
                piece += "."
            sentences.append(piece)
    return sentences


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Full normalization producing speakable prose (paragraphs preserved)."""
    text = _normalize_whitespace(text).translate(_PUNCT_LIGATURES)
    text = _strip_artifacts(text)
    text = _normalize_quotes(text)
    text = _reflow(text)
    text = _fix_homoglyphs(text)
    # "+" is Silero's stress marker, so a literal plus ("2+2") must be spoken.
    text = re.sub(r"[ \t]*\+[ \t]*", " плюс ", text)
    # Roman → digits must precede transliteration (else XIV becomes "ксив")
    # and number expansion (which turns the digits into words).
    text = _expand_roman(text)
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    text = _transliterate_latin(text)
    # Final tidy: collapse runs of spaces, trim.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(doc: Document) -> Document:
    cleaned: list[Chapter] = []
    for ch in doc.chapters:
        body = clean_text(ch.text)
        if not body.strip():
            continue
        title = ch.title.strip() if ch.title else None
        cleaned.append(Chapter(title=title, text=body))
    if not cleaned:
        # Nothing survived cleaning — treat as empty so the worker fails cleanly.
        from .types import ExtractError

        raise ExtractError("После очистки не осталось текста для озвучивания")
    return Document(
        title=doc.title,
        author=doc.author,
        chapters=cleaned,
        series=doc.series,
        series_index=doc.series_index,
    )
