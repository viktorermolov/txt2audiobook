from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import zipfile
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from unittest.mock import patch

if "num2words" not in sys.modules and importlib.util.find_spec("num2words") is None:
    stub = types.ModuleType("num2words")
    stub.__spec__ = ModuleSpec("num2words", loader=None)
    stub.num2words = lambda value, **_: str(value)
    sys.modules.setdefault("num2words", stub)

from app.pipeline import clean, extract


def fake_num2words(value: int, *, to: str | None = None, **_) -> str:
    cardinal = {
        1: "один", 2: "два", 5: "пять", 12: "двенадцать",
        2020: "две тысячи двадцать",
    }
    ordinal = {
        1: "первый", 3: "третий", 5: "пятый", 12: "двенадцатый",
        1905: "тысяча девятьсот пятый", 2020: "две тысячи двадцатый",
        2024: "две тысячи двадцать четвёртый",
    }
    return (ordinal if to == "ordinal" else cardinal).get(value, str(value))


class TextQualityTests(unittest.TestCase):
    def test_inline_markup_preserves_words_and_math_but_drops_marked_notes(self) -> None:
        if not importlib.util.find_spec("bs4"):
            self.skipTest("beautifulsoup4 is not installed")
        markup = (
            '<html><body><p>сло<b>во</b> и x<sup>2</sup>, H<sub>2</sub>O '
            '<sup><a href="#fn1">1</a></sup>.</p><p>Далее.</p></body></html>'
        )
        text = extract._strip_html(markup)
        self.assertIn("слово", text)
        self.assertIn("x 2", text)
        self.assertIn("H 2 O", text)
        self.assertNotIn("1.", text)
        self.assertIn(".\n", text)

    def test_epub_keeps_short_linear_spine_text_not_nav_or_nonlinear(self) -> None:
        class Item:
            def __init__(self, id_: str, text: str, properties=()):
                self.id_ = id_
                self.text = text
                self.properties = properties

            def get_id(self):
                return self.id_

            def get_type(self):
                return 9

            def get_content(self):
                return self.text.encode("utf-8")

        items = [
            Item("nav", "Навигация", ["nav"]),
            Item("preface", "Пролог."),
            Item("hidden", "Скрытый текст."),
            Item("main", "Это основная глава с достаточно длинным текстом."),
        ]
        book = types.SimpleNamespace(
            get_items=lambda: items,
            get_metadata=lambda *_: [],
            spine=[("nav", "yes"), ("preface", "yes"),
                   ("hidden", "no"), ("main", "yes")],
        )
        ebooklib = types.ModuleType("ebooklib")
        ebooklib.ITEM_DOCUMENT = 9
        ebooklib.epub = types.SimpleNamespace(read_epub=lambda *_, **__: book)
        with (
            patch.dict(sys.modules, {"ebooklib": ebooklib}),
            patch.object(extract, "_validate_epub_archive"),
            patch.object(extract, "_epub_series_metadata", return_value=(None, None)),
            patch.object(extract, "_strip_html", side_effect=lambda value: value),
        ):
            doc = extract._extract_epub(Path("dummy.epub"))
        self.assertEqual(["Пролог.", items[-1].text], [c.text for c in doc.chapters])

    def test_units_city_year_dates_and_inflected_ordinals(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            self.assertEqual(
                "один сантиметр. два сантиметра. пять сантиметров.",
                clean.clean_text("1 см. 2 см. 5 см."),
            )
            self.assertEqual("пять сантиметров. Далее.", clean.clean_text("5 см. Далее."))
            self.assertIn("11 сантиметров.", clean.clean_text("11 см."))
            self.assertIn("21 сантиметр.", clean.clean_text("21 см."))
            self.assertEqual("смотри текст", clean.clean_text("см. текст"))
            self.assertEqual("город Москва", clean.clean_text("г. Москва"))
            self.assertNotIn("город", clean.clean_text("2020 г. Москва"))
            self.assertEqual(
                "В тысяча девятьсот пятом году. Москва.",
                clean.clean_text("В 1905 г. Москва."),
            )
            self.assertEqual(
                "тысяча девятьсот пятый год.", clean.clean_text("1905 г."),
            )
            self.assertEqual("к пятому дому", clean.clean_text("к 5-му дому"))
            self.assertEqual("с третьим", clean.clean_text("с 3-ым"))
            self.assertEqual(
                "двенадцатого марта две тысячи двадцать четвёртого года",
                clean.clean_text("12.03.2024"),
            )

    def test_invalid_dates_and_dotted_identifiers_are_not_decimals(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            invalid = clean.clean_text("31.02.2024")
            leap_invalid = clean.clean_text("29.02.2023")
            leap_valid = clean.clean_text("29.02.2024")
            identifier = clean.clean_text("v1.02.2024")
            decimal = clean.clean_text("3.14.")
        self.assertEqual(2, invalid.count("."))
        self.assertNotIn("целых", invalid)
        self.assertNotIn("февраля", invalid)
        self.assertNotIn("февраля", leap_invalid)
        self.assertIn("февраля", leap_valid)
        self.assertNotIn("целых", identifier)
        self.assertNotIn("февраля", identifier)
        self.assertIn("целых", decimal)

    @unittest.skipUnless(PathFinder.find_spec("num2words"), "num2words is not installed")
    def test_real_num2words_units_year_and_date(self) -> None:
        self.assertIn("два сантиметра.", clean.clean_text("2 см."))
        self.assertIn("пятом году.", clean.clean_text("В 1905 г."))
        self.assertIn("марта", clean.clean_text("12.03.2024"))

    @unittest.skipUnless(
        all(PathFinder.find_spec(name) for name in ("ebooklib", "lxml", "bs4", "num2words")),
        "real EPUB and Russian number dependencies are not installed",
    )
    def test_real_epub_extraction_and_cleaning_preserves_semantic_scripts(self) -> None:
        container = (
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            'version="1.0"><rootfiles><rootfile full-path="OPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>'
        )
        opf = (
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Книга</dc:title>'
            '<dc:creator>Автор</dc:creator></metadata><manifest><item id="chapter" '
            'href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="chapter"/></spine></package>'
        )
        chapter = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Книга</title></head>'
            '<body><p>Это достаточно длинный текст. Сохраняем сло<b>во</b>, '
            'x<sup>2</sup> и H<sub>2</sub>O<sup><a href="#fn1">1</a></sup>.</p>'
            '</body></html>'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
                archive.writestr("META-INF/container.xml", container)
                archive.writestr("OPS/content.opf", opf)
                archive.writestr("OPS/chapter.xhtml", chapter)
            spoken = clean.clean_document(extract.extract(path)).full_text
        self.assertIn("слово", spoken)
        self.assertIn("кс два", spoken)
        self.assertIn("х два о", spoken)
        self.assertNotIn("один", spoken)

    def test_bare_numeric_lines_and_superscript_digits_survive(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            text = clean.clean_text("Глава\n5\n\nСтепень x².\n\n— 12 —")
        self.assertIn("пять", text)
        self.assertIn("кс два", text)
        self.assertNotIn("двенадцать", text)

    def test_ambiguous_short_abbreviations_keep_ordinary_words(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            self.assertEqual("Я горжусь им. Потом ушёл.", clean.clean_text("Я горжусь им. Потом ушёл."))
            self.assertEqual("Он ел рис. Вкусно.", clean.clean_text("Он ел рис. Вкусно."))
            self.assertEqual("— Не те. Другие.", clean.clean_text("— Не те. Другие."))
            self.assertEqual("Это то есть пример.", clean.clean_text("Это т. е. пример."))
            self.assertEqual("Театр имени Пушкина.", clean.clean_text("Театр им. Пушкина."))
            self.assertEqual("смотри рисунок пять.", clean.clean_text("см. рис. 5."))

    def test_expanded_abbreviations_end_sentences_only_when_they_should(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            self.assertEqual(
                "В тысяча девятьсот пятом году он родился.",
                clean.clean_text("В 1905 г. он родился."),
            )
            self.assertEqual("и так далее и так", clean.clean_text("и т. д. и так"))
            self.assertEqual("и так далее. Потом", clean.clean_text("и т. д. Потом"))
            self.assertEqual("500 граммов муки", clean.clean_text("500 г. муки"))

    def test_numbers_symbols_and_mixed_script_words(self) -> None:
        with patch.object(clean, "num2words", side_effect=fake_num2words):
            self.assertEqual("в 1943 150 танков", clean.clean_text("в 1943 150 танков"))
            self.assertEqual("Было 10000 рублей.", clean.clean_text("Было 10 000 рублей."))
            self.assertEqual("два плюс два", clean.clean_text("2+2"))
            self.assertEqual(
                "Рост пять процентов и один процент.", clean.clean_text("Рост 5% и 1%.")
            )
            self.assertEqual("Дом номер пять.", clean.clean_text("Дом №5."))
            self.assertEqual("Это река и сад.", clean.clean_text("Это pека и cад."))

    def test_line_break_hyphen_keeps_real_compounds(self) -> None:
        self.assertEqual("кто-нибудь пришёл", clean.clean_text("кто-\nнибудь пришёл"))
        self.assertEqual("что-то", clean.clean_text("что-\nто"))
        self.assertEqual("место", clean.clean_text("мес-\nто"))
        self.assertEqual("слово", clean.clean_text("сло-\nво"))

    def test_unpunctuated_lines_get_a_pause_before_the_next_sentence(self) -> None:
        self.assertEqual(
            ["Часть вторая.", "Было утро."], clean.split_sentences("Часть вторая\nБыло утро.")
        )

    @unittest.skipUnless(PathFinder.find_spec("num2words"), "num2words is not installed")
    def test_real_num2words_years_centuries_decades_units(self) -> None:
        self.assertIn("двенадцатом году", clean.clean_text("В 1812 году война."))
        self.assertIn("двенадцатому году", clean.clean_text("К 1812 году всё кончилось."))
        self.assertIn("двенадцатого года", clean.clean_text("События 1812 года."))
        self.assertIn("три года", clean.clean_text("Прошло 3 года."))
        self.assertIn("девятнадцатого века", clean.clean_text("Начало XIX века."))
        self.assertIn("двадцатом веке", clean.clean_text("В XX веке."))
        self.assertIn("девяностые годы", clean.clean_text("В 90-е годы."))
        self.assertIn("первый раз", clean.clean_text("1-ый раз."))
        self.assertIn("пятьсот граммов", clean.clean_text("500 г. муки"))
        self.assertIn("пятьдесят процентов", clean.clean_text("50% людей"))


    @unittest.skipUnless(PathFinder.find_spec("num2words"), "num2words is not installed")
    def test_real_num2words_short_years_with_preposition_or_era(self) -> None:
        self.assertIn("восемьдесят восьмом году князь", clean.clean_text("В 988 г. князь пришёл."))
        self.assertIn("трёхсотого года до нашей эры", clean.clean_text("Около 300 г. до н. э. жили."))
        self.assertIn("пятидесятом году до нашей эры", clean.clean_text("Это было в 50 г. до н. э."))
        self.assertIn("двенадцатому году", clean.clean_text("К 1812 г. всё кончилось."))
        self.assertIn("пятьсот граммов муки", clean.clean_text("500 г. муки"))
        self.assertIn("пятьдесят граммов муки", clean.clean_text("В 50 г. муки"))

    def test_hyphen_join_is_linear_on_long_unspaced_runs(self) -> None:
        import time

        started = time.monotonic()
        clean._HYPHEN_BREAK_RE.sub(clean._join_line_hyphen, "а" * 200_000)
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
