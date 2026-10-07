import base64
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.pipeline import cover


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (40, 60), (200, 30, 40, 128)).save(output, format="PNG")
    return output.getvalue()


class CoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()

    def fb2(self, image=None, href="#cover", extra=""):
        data = image_bytes() if image is None else image
        source = self.root / "book.fb2"
        source.write_text(
            '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" '
            'xmlns:l="http://www.w3.org/1999/xlink"><description><title-info>'
            f'<coverpage><image l:href="{href}"/></coverpage></title-info></description>'
            '<body><section><p>Book text.</p></section></body>'
            f'{extra}<binary id="cover" content-type="image/png">\n'
            + base64.b64encode(data).decode() + '\n</binary></FictionBook>'
        )
        return source

    def epub(self, metadata='', manifest='', guide='', members=None):
        source = self.root / "book.epub"
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('META-INF/container.xml',
                             '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
            archive.writestr('OPS/book.opf', '<package><metadata>' + metadata
                             + '</metadata><manifest>' + manifest + '</manifest>'
                             + guide + '</package>')
            for name, data in (members or {'OPS/images/cover.png': image_bytes()}).items():
                archive.writestr(name, data)
        return source

    def assert_cover(self, source):
        result = cover.extract_cover(source, self.work)
        self.assertEqual(self.work / 'cover.jpg', result)
        with Image.open(result) as image:
            self.assertEqual('JPEG', image.format)
            self.assertEqual('RGB', image.mode)
            self.assertEqual((40, 60), image.size)
            self.assertNotIn('exif', image.info)
        self.assertFalse((self.work / 'cover.jpg.part').exists())
        return result

    def test_fb2_uses_referenced_binary_not_first_image(self):
        source = self.fb2(extra='<binary id="illustration">bm90IGFuIGltYWdl</binary>')
        self.assert_cover(source)

    def test_fb2_external_reference_is_not_fetched(self):
        self.assertIsNone(cover.extract_cover(self.fb2(href='https://example.org/cover'), self.work))

    def test_epub3_cover_image_property(self):
        self.assert_cover(self.epub(manifest='<item id="art" href="images/cover.png" properties="foo cover-image"/>'))

    def test_epub2_cover_metadata(self):
        self.assert_cover(self.epub(metadata='<meta name="cover" content="art"/>',
                                    manifest='<item id="art" href="images/cover.png"/>'))

    def test_bad_epub_candidate_does_not_mask_valid_explicit_cover(self):
        self.assert_cover(self.epub(
            metadata='<meta name="cover" content="good"/>',
            manifest='<item id="bad" href="bad.png" properties="cover-image"/>'
                     '<item id="good" href="images/cover.png"/>',
            members={'OPS/bad.png': b'bad image', 'OPS/images/cover.png': image_bytes()},
        ))

    def test_epub2_guide_xhtml_relative_encoded_path(self):
        self.assert_cover(self.epub(
            guide='<guide><reference type="cover" href="pages/cover.xhtml#top"/></guide>',
            members={'OPS/pages/cover.xhtml': '<html><body><img src="../images/my%20cover.png"/></body></html>',
                     'OPS/images/my cover.png': image_bytes()},
        ))

    def test_epub_svg_wrapper_uses_local_bitmap_not_svg_rendering(self):
        self.assert_cover(self.epub(
            guide='<guide><reference type="cover" href="cover.svg"/></guide>',
            members={'OPS/cover.svg': '<svg xmlns:xlink="http://www.w3.org/1999/xlink"><image xlink:href="images/cover.png"/></svg>',
                     'OPS/images/cover.png': image_bytes()},
        ))

    def test_no_designated_cover_does_not_select_random_illustration(self):
        self.assertIsNone(cover.extract_cover(self.epub(manifest='<item id="art" href="images/cover.png"/>'), self.work))

    def test_remote_and_escaping_epub_references_are_ignored(self):
        for href in ['https://example.org/cover.png', '//example.org/cover.png', '../../cover.png',
                     '%2Fetc/passwd', '..%2F..%2Fcover.png', 'file:///etc/passwd']:
            with self.subTest(href=href):
                source = self.epub(manifest=f'<item id="art" href="{href}" properties="cover-image"/>')
                self.assertIsNone(cover.extract_cover(source, self.work))

    def test_bad_bitmap_and_base64_are_optional(self):
        source = self.fb2(image=b'not an image')
        self.assertIsNone(cover.extract_cover(source, self.work))
        source.write_text(source.read_text().replace('bm90IGFuIGltYWdl', '!not-base64!'))
        self.assertIsNone(cover.extract_cover(source, self.work))

    def test_image_byte_and_pixel_limits_do_not_fail_book(self):
        source = self.fb2()
        with patch.object(cover, 'MAX_COVER_BYTES', 10):
            self.assertIsNone(cover.extract_cover(source, self.work))
        with patch.object(cover, 'MAX_COVER_PIXELS', 100):
            self.assertIsNone(cover.extract_cover(source, self.work))
        source = self.epub(manifest='<item id="art" href="images/cover.png" properties="cover-image"/>')
        with patch.object(cover, 'MAX_COVER_BYTES', 10):
            self.assertIsNone(cover.extract_cover(source, self.work))

    def test_xml_entities_do_not_load_external_cover(self):
        source = self.fb2()
        text = source.read_text().replace('<FictionBook ', '<!DOCTYPE FictionBook [<!ENTITY secret SYSTEM "file:///etc/passwd">]><FictionBook ', 1)
        encoded = base64.b64encode(image_bytes()).decode()
        source.write_text(text.replace(encoded, '&secret;'))
        self.assertIsNone(cover.extract_cover(source, self.work))

    def test_resume_replaces_derived_cover_and_removes_stale_cover(self):
        source = self.fb2()
        target = self.assert_cover(source)
        target.write_bytes(b'corrupt cache')
        self.assert_cover(source)
        source.write_text('<FictionBook><body><section><p>Text</p></section></body></FictionBook>')
        self.assertIsNone(cover.extract_cover(source, self.work))
        self.assertFalse(target.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe required')
    def test_real_m4b_embeds_cover_and_preserves_audio_and_chapters(self):
        from app.pipeline.assemble import assemble
        from app.pipeline.synth import Chunk

        artwork = self.assert_cover(self.fb2())
        wav = self.root / 'sample.wav'
        with wave.open(str(wav), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b'\x00\x10' * 8000)
        plan = [Chunk(0, 0, 'First', 'Text'), Chunk(1, 1, 'Second', 'Text')]
        for image in [artwork, None]:
            outputs = assemble(plan, [wav, wav], self.root / ('with' if image else 'without'), 'book',
                               title='Book', author='Author', work_dir=self.work, cover_path=image)
            self.assertEqual(1, len(outputs))
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_chapters',
                                     '-show_format', '-of', 'json', str(outputs[0])],
                                    check=True, capture_output=True, text=True)
            info = json.loads(result.stdout)
            self.assertEqual(2, len(info['chapters']))
            self.assertAlmostEqual(2, float(info['format']['duration']), delta=0.1)
            self.assertEqual(1, sum(s['codec_type'] == 'audio' for s in info['streams']))
            covers = [s for s in info['streams'] if s.get('disposition', {}).get('attached_pic')]
            self.assertEqual(1 if image else 0, len(covers))
            if image:
                self.assertEqual('mjpeg', covers[0]['codec_name'])
        from app.pipeline.assemble import _cover_muxes

        broken = self.work / 'broken.jpg'
        broken.write_bytes(b'not an image')
        self.assertTrue(_cover_muxes(artwork, self.work))
        self.assertFalse(_cover_muxes(broken, self.work))
        self.assertFalse((self.work / 'cover-probe.m4b').exists())

    def test_unmuxable_cover_is_dropped_before_the_single_encode(self):
        from app.pipeline.assemble import AssembleCancelled, AssembleError, assemble
        from app.pipeline.synth import Chunk

        wav = self.root / 'audio.wav'
        with wave.open(str(wav), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b'\x00\x10' * 8000)
        calls = []

        def mux(_paths, _meta, target, *args, cover_path=None):
            calls.append(cover_path)
            target.write_bytes(b'audio')

        kwargs = dict(title='Book', author=None, work_dir=self.work, cover_path=self.work / 'cover.jpg')
        plan = [Chunk(0, 0, None, 'Text')]
        with (
            patch('app.pipeline.assemble._cover_muxes', return_value=False),
            patch('app.pipeline.assemble._run_ffmpeg', side_effect=mux),
        ):
            result = assemble(plan, [wav], self.root / 'out', 'book', **kwargs)
        self.assertEqual([None], calls)
        self.assertEqual(b'audio', result[0].read_bytes())
        # An audio error is not retried: a 20-hour book is never encoded twice.
        with (
            patch('app.pipeline.assemble._cover_muxes', return_value=True),
            patch('app.pipeline.assemble._run_ffmpeg', side_effect=AssembleError('bad audio')) as mock,
        ):
            with self.assertRaises(AssembleError):
                assemble(plan, [wav], self.root / 'out', 'book', **kwargs)
            self.assertEqual(1, mock.call_count)
        with (
            patch('app.pipeline.assemble._cover_muxes', return_value=True),
            patch('app.pipeline.assemble._run_ffmpeg', side_effect=AssembleCancelled) as mock,
        ):
            with self.assertRaises(AssembleCancelled):
                assemble(plan, [wav], self.root / 'out', 'book', **kwargs)
            self.assertEqual(1, mock.call_count)


if __name__ == '__main__':
    unittest.main()
