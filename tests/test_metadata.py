from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

# The Pi image supplies num2words. This local fallback only permits importing
# clean_document for its metadata-preservation test on a dependency-free host.
if importlib.util.find_spec("num2words") is None:
    stub = types.ModuleType("num2words")
    stub.num2words = lambda value, **_kwargs: str(value)
    sys.modules["num2words"] = stub

from app.pipeline.clean import clean_document
from app.pipeline.extract import extract
from app.pipeline.types import Chapter, Document, safe_filename


class MetadataTests(unittest.TestCase):
    def test_plain_text_does_not_invent_series_and_keeps_short_preface(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.txt"
            path.write_text(
                "Короткое предисловие.\n\nГлава 1\nТекст первой главы.\n\nГлава 2\nТекст второй главы.",
                encoding="utf-8",
            )
            document = extract(path)

        self.assertIsNone(document.series)
        self.assertIsNone(document.series_index)
        self.assertEqual(document.chapters[0].text, "Короткое предисловие.")

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "lxml is not installed")
    def test_fb2_sequence_and_verse_are_extracted(self):
        content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><description><title-info>
<book-title>  Книга   </book-title><author><first-name>Иван</first-name><last-name>Петров</last-name></author>
<sequence name=" Цикл  " number=" 02 "/></title-info></description><body><section><title><p>Глава 1</p></title>
<p>Проза.</p><poem><stanza><v>Первая строка.</v><v>Вторая строка.</v></stanza></poem></section></body></FictionBook>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.fb2"
            path.write_text(content, encoding="utf-8")
            document = extract(path)

        self.assertEqual((document.title, document.author), ("Книга", "Иван Петров"))
        self.assertEqual((document.series, document.series_index), ("Цикл", "02"))
        self.assertIn("Первая строка.", document.chapters[0].text)
        self.assertIn("Вторая строка.", document.chapters[0].text)

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "lxml is not installed")
    def test_fb2_comment_and_external_entity_do_not_hide_or_expose_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "secret.txt"
            secret.write_text("НЕ ДОЛЖНО БЫТЬ ПРОЧИТАНО", encoding="utf-8")
            path = Path(directory) / "book.fb2"
            path.write_text(
                f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE FictionBook [<!ENTITY secret SYSTEM "file://{secret}">]>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body><section>
<!-- комментарий не является текстовым блоком -->
<p>Настоящая проза сохранена. &secret;</p>
</section></body></FictionBook>''',
                encoding="utf-8",
            )
            document = extract(path)

        text = document.full_text
        self.assertIn("Настоящая проза сохранена.", text)
        self.assertNotIn("НЕ ДОЛЖНО БЫТЬ ПРОЧИТАНО", text)

    @unittest.skipUnless(
        importlib.util.find_spec("ebooklib") and importlib.util.find_spec("lxml"),
        "ebooklib or lxml is not installed",
    )
    def test_epub_calibre_series_metadata(self):
        container = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
        opf = """<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Книга</dc:title><dc:creator>Автор</dc:creator><meta name="calibre:series" content="Серия"/><meta name="calibre:series_index" content="3.5"/></metadata><manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="chapter"/></spine></package>"""
        chapter = "<html><body><h1>Глава</h1><p>Достаточно длинный текст главы для извлечения.</p></body></html>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("META-INF/container.xml", container)
                archive.writestr("OPS/content.opf", opf)
                archive.writestr("OPS/chapter.xhtml", chapter)
            document = extract(path)

        self.assertEqual((document.series, document.series_index), ("Серия", "3.5"))

    def test_clean_document_preserves_series_metadata(self):
        document = Document("Книга", "Автор", [Chapter(None, "Текст.")], "Серия", "1")
        cleaned = clean_document(document)
        self.assertEqual((cleaned.series, cleaned.series_index), ("Серия", "1"))

    def _txt_chapters(self, text: str) -> list[Chapter]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.txt"
            path.write_text(text, encoding="utf-8")
            return extract(path).chapters

    def test_txt_prose_starting_with_heading_words_is_not_a_chapter(self):
        chapters = self._txt_chapters(
            "Вступление в должность было трудным.\n\nГлава семьи вошёл в комнату.\n\n"
            "Глава 1\nТекст первой главы.\n\nГлава 5 закончилась ничем, сказал он.\n\n"
            "Глава вторая. Встреча\nТекст второй главы."
        )
        self.assertEqual([None, "Глава 1", "Глава вторая. Встреча"], [c.title for c in chapters])
        self.assertIn("Глава 5 закончилась", chapters[1].text)

    def test_txt_table_of_contents_and_part_headings_are_not_voiced_twice(self):
        chapters = self._txt_chapters(
            "Глава 1\nГлава 2\n\nЧасть первая\n\nГлава 1\nТекст первой главы.\n\n"
            "Глава 2\nТекст второй главы."
        )
        self.assertEqual(["Часть первая. Глава 1", "Глава 2"], [c.title for c in chapters])
        for chapter in chapters:
            self.assertNotEqual(chapter.title, chapter.text)

    def test_txt_crlf_line_endings_keep_chapter_headings(self):
        chapters = self._txt_chapters(
            "Глава 1\r\nТекст первой главы.\r\n\r\nГлава 2-я\r\nТекст второй главы.\r\n"
        )
        self.assertEqual(["Глава 1", "Глава 2-я"], [c.title for c in chapters])
        self.assertNotIn("\r", chapters[0].text)

    @unittest.skipUnless(importlib.util.find_spec("lxml"), "lxml is not installed")
    def test_fb2_nested_sections_become_chapters(self):
        content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body><section>
<title><p>Часть первая</p></title>
<section><title><p>Глава 1</p></title><p>Проза первой главы.</p></section>
<section><title><p>Глава 2</p></title><p>Проза второй главы.</p></section>
</section></body></FictionBook>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.fb2"
            path.write_text(content, encoding="utf-8")
            chapters = extract(path).chapters

        self.assertEqual(["Часть первая. Глава 1", "Глава 2"], [c.title for c in chapters])
        self.assertEqual("Проза первой главы.", chapters[0].text)
        self.assertNotIn("Глава 2", chapters[0].text)

    def test_epub_highly_compressed_member_is_rejected_below_global_threshold(self):
        from app.pipeline import extract as extract_module
        from app.pipeline.types import ExtractError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bomb.epub"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("OPS/chapter.xhtml", b"<p>a</p>" * (3 * 1024 * 1024))
            with self.assertRaises(ExtractError):
                extract_module._validate_epub_archive(path)

    def test_safe_filename_limits_utf8_bytes(self):
        name = safe_filename("Ж" * 100, max_len=120)
        self.assertLessEqual(len(name.encode("utf-8")), 120)
        self.assertTrue(name)


if __name__ == "__main__":
    unittest.main()
