import importlib.util
import sys
import tempfile
import types
import unittest
import wave
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch

# synth imports clean, which imports num2words. Audio unit tests do not need
# normalization and must run on a host without the container dependencies.
_STUBBED_NUM2WORDS = (
    "num2words" not in sys.modules and importlib.util.find_spec("num2words") is None
)
if _STUBBED_NUM2WORDS:
    module = types.ModuleType("num2words")
    module.__spec__ = ModuleSpec("num2words", loader=None)
    module.num2words = lambda value, **_: str(value)
    sys.modules["num2words"] = module

from app.pipeline import assemble, synth  # noqa: E402 - temporary dependency stub above
RealSileroEngine = synth._SileroEngine

if _STUBBED_NUM2WORDS:
    del sys.modules["num2words"]


def write_wav(path: Path, *, rate: int = 8000, frames: int = 8000,
              amplitude: int = 4096) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(amplitude.to_bytes(2, "little", signed=True) * frames)


class FakeEngine:
    calls: list[str] = []
    fail_texts: set[str] = set()
    silent_once: set[str] = set()

    def __init__(self, *_):
        pass

    def filter_text(self, text: str) -> str:
        return text

    def synth_to_wav(self, text: str, out: Path, *, sample_rate: int, **_) -> None:
        self.calls.append(text)
        if text in self.fail_texts:
            raise RuntimeError("fake TTS failure")
        if text in self.silent_once:
            self.silent_once.remove(text)
            write_wav(out, rate=sample_rate, amplitude=0)
            return
        write_wav(out, rate=sample_rate)


class AudioTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeEngine.calls = []
        FakeEngine.fail_texts = set()
        FakeEngine.silent_once = set()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model.pt"
        self.model.write_bytes(b"fake model")
        self.audio = self.root / "audio"
        self.plan = [
            synth.Chunk(0, 0, "Первая", "первый фрагмент"),
            synth.Chunk(1, 1, "Вторая", "второй фрагмент"),
            synth.Chunk(2, 1, None, "третий фрагмент"),
        ]
        self.engine = patch.object(synth, "_SileroEngine", FakeEngine)
        self.engine.start()

    def tearDown(self) -> None:
        self.engine.stop()
        self.tmp.cleanup()

    def synthesize(self, plan=None, **kwargs):
        return synth.synthesize_plan(
            plan or self.plan,
            self.model,
            self.audio,
            sample_rate=8000,
            retries=0,
            **kwargs,
        )

    def test_cache_is_tied_to_settings_and_legacy_wavs_are_invalidated(self) -> None:
        self.audio.mkdir()
        write_wav(self.audio / "chunk_00000.wav")  # no manifest: legacy cache
        self.synthesize()
        self.assertEqual(3, len(FakeEngine.calls))

        FakeEngine.calls = []
        self.synthesize()
        self.assertEqual([], FakeEngine.calls)

        self.synthesize(sentence_silence=0.7)
        self.assertEqual(3, len(FakeEngine.calls))

    def test_silenced_ledger_is_reported_on_resume(self) -> None:
        FakeEngine.fail_texts = {"второй фрагмент"}
        silenced: list[int] = []
        self.synthesize(silenced_out=silenced)
        self.assertEqual([1], silenced)
        self.assertTrue((self.audio / "silenced.json").exists())

        FakeEngine.calls = []
        resumed: list[int] = []
        self.synthesize(silenced_out=resumed)
        self.assertEqual([1], resumed)
        self.assertEqual([], FakeEngine.calls)

    def test_cancellation_precedes_cache_hit(self) -> None:
        self.synthesize()
        with self.assertRaises(synth.SynthCancelled):
            self.synthesize(cancel_check=lambda: True)

    def test_systematic_failure_does_not_create_silent_book(self) -> None:
        FakeEngine.fail_texts = {"единственный"}
        one = [synth.Chunk(0, 0, None, "единственный")]
        with self.assertRaises(synth.SynthError):
            self.synthesize(plan=one)
        self.assertFalse((self.audio / "chunk_00000.wav").exists())

    def test_three_consecutive_failures_abort_before_ratio_limit(self) -> None:
        plan = [synth.Chunk(i, 0, None, f"сбой{i}") for i in range(100)]
        FakeEngine.fail_texts = {f"сбой{i}" for i in range(3)}

        with self.assertRaisesRegex(synth.SynthError, "3 фрагмента подряд"):
            self.synthesize(plan=plan)

        self.assertTrue((self.audio / "chunk_00000.wav").exists())
        self.assertTrue((self.audio / "chunk_00001.wav").exists())
        self.assertFalse((self.audio / "chunk_00002.wav").exists())

    def test_wav_validation_requires_expected_pcm_payload(self) -> None:
        path = self.root / "bad.wav"
        write_wav(path)
        self.assertTrue(synth._wav_is_valid(path, 8000))
        self.assertFalse(synth._wav_is_valid(path, 48000))
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\x00\x10" + b"\x00\x00" * 7999)
        self.assertFalse(synth._wav_is_valid(path, 8000, require_speech=True))
        write_wav(path)
        raw = path.read_bytes()
        path.write_bytes(raw[:-16])
        self.assertFalse(synth._wav_is_valid(path, 8000))

    def test_silent_cached_wav_is_repaired_without_invalidating_manifest(self) -> None:
        self.synthesize()
        silent = self.audio / "chunk_00001.wav"
        write_wav(silent, amplitude=0)
        FakeEngine.calls = []

        self.synthesize()

        self.assertEqual(["второй фрагмент"], FakeEngine.calls)
        self.assertTrue(synth._wav_is_valid(
            silent, 8000, require_speech=True, tail_frames=3200,
        ))

    def test_empty_speech_wav_is_retried(self) -> None:
        FakeEngine.silent_once = {"второй фрагмент"}
        synth.synthesize_plan(
            self.plan, self.model, self.audio, sample_rate=8000, retries=1,
        )
        self.assertEqual(2, FakeEngine.calls.count("второй фрагмент"))
        self.assertFalse((self.audio / "silenced.json").exists())

    def test_silence_ledger_allows_intentional_silence_on_resume(self) -> None:
        FakeEngine.fail_texts = {"второй фрагмент"}
        self.synthesize()
        FakeEngine.calls = []
        self.synthesize()
        self.assertEqual([], FakeEngine.calls)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
    def test_model_nonfinite_samples_are_rejected_before_pcm_conversion(self) -> None:
        import numpy as np

        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.full(8000, np.nan, dtype=np.float32)

        engine = object.__new__(RealSileroEngine)
        engine.model = types.SimpleNamespace(apply_tts=lambda **_: Tensor())
        engine.speaker = "eugene"
        out = self.root / "nonfinite.wav"
        with self.assertRaisesRegex(synth.SynthError, "некорректные значения"):
            engine.synth_to_wav(
                "текст", out, sample_rate=8000, put_accent=True,
                put_yo=True, sentence_silence=0.4,
            )
        self.assertFalse(out.exists())

    def test_too_long_chunk_is_split_at_sentences_instead_of_silenced(self) -> None:
        import numpy as np

        calls: list[str] = []

        class Tensor:
            def __init__(self, n: int) -> None:
                self.n = n

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.full(self.n, 0.25, dtype=np.float32)

        def apply_tts(*, text, **_):
            calls.append(text)
            if len(text) > 40:
                raise ValueError("Model couldn't generate your text, probably it's too long")
            return Tensor(8000)

        engine = object.__new__(RealSileroEngine)
        engine.model = types.SimpleNamespace(apply_tts=apply_tts)
        engine.speaker = "eugene"
        text = "Первое предложение тут. Второе предложение здесь. Третье и последнее."
        out = self.root / "split.wav"
        engine.synth_to_wav(
            text, out, sample_rate=8000, put_accent=True, put_yo=True, sentence_silence=0,
        )
        spoken = [part for part in calls if len(part) <= 40]
        self.assertEqual(text, " ".join(spoken))
        self.assertTrue(all(part.rstrip()[-1] in ".!?" for part in spoken[:-1]))
        with wave.open(str(out)) as audio:
            pause = int(synth._SPLIT_PAUSE_SEC * 8000)
            self.assertEqual(len(spoken) * 8000 + (len(spoken) - 1) * pause, audio.getnframes())

    def test_other_model_errors_are_not_split(self) -> None:
        calls: list[str] = []

        def apply_tts(*, text, **_):
            calls.append(text)
            raise KeyError("ʃ")

        engine = object.__new__(RealSileroEngine)
        engine.model = types.SimpleNamespace(apply_tts=apply_tts)
        engine.speaker = "eugene"
        with self.assertRaises(KeyError):
            engine.synth_to_wav(
                "Одно. Два. Три.", self.root / "x.wav", sample_rate=8000,
                put_accent=True, put_yo=True, sentence_silence=0,
            )
        self.assertEqual(1, len(calls))

    def test_split_in_half_prefers_sentence_then_clause_boundaries(self) -> None:
        self.assertEqual(("Раз два.", "Три четыре."), synth._split_in_half("Раз два. Три четыре."))
        self.assertEqual(("Раз два,", "три четыре"), synth._split_in_half("Раз два, три четыре"))
        self.assertIsNone(synth._split_in_half("Слово"))

    def test_part_metadata_uses_title_from_full_plan(self) -> None:
        info = assemble._ChunkInfo(self.plan[2], self.root / "x.wav", 1.0)
        meta = assemble._build_ffmeta(
            [info], "Книга", None, True, 2, 2,
            chapter_titles=assemble._chapter_titles(self.plan),
        )
        self.assertIn("title=Вторая", meta)
        self.assertNotIn("title=Глава 1", meta)

    def test_run_ffmpeg_cancellation_terminates_and_removes_temporary_output(self) -> None:
        class RunningProcess:
            def __init__(self, *_args, **_kwargs):
                self.returncode = None
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def communicate(self, **_):
                return "", ""

        wav = self.root / "source.wav"
        write_wav(wav)
        meta = self.root / "meta.txt"
        meta.write_text(";FFMETADATA1\n", encoding="utf-8")
        out = self.root / "out.m4b.part"
        out.write_bytes(b"partial")
        checks = iter([False, True])
        with patch.object(assemble.subprocess, "Popen", RunningProcess):
            with self.assertRaises(assemble.AssembleCancelled):
                assemble._run_ffmpeg(
                    [wav], meta, out, "64k", 8000, False, self.root,
                    cancel_check=lambda: next(checks),
                )
        self.assertFalse(out.exists())

    def test_assemble_replaces_final_only_after_success(self) -> None:
        wav = self.root / "source.wav"
        write_wav(wav, frames=8000)
        final = self.root / "output" / "book.m4b"
        final.parent.mkdir()
        final.write_bytes(b"previous")

        def fake_ffmpeg(_wavs, _meta, output, *_args, **_kwargs):
            output.write_bytes(b"new")

        with patch.object(assemble, "_run_ffmpeg", fake_ffmpeg):
            outputs = assemble.assemble(
                [self.plan[0]], [wav], final.parent, "book",
                title="Книга", author=None, sample_rate=8000, max_part_mib=1,
                work_dir=self.root / "work",
            )
        self.assertEqual([final], outputs)
        self.assertEqual(b"new", final.read_bytes())

    def test_default_assembly_never_splits_a_huge_book(self) -> None:
        wavs = [self.root / "first.wav", self.root / "second.wav"]
        calls: list[list[Path]] = []

        def fake_ffmpeg(paths, _meta, output, *_args, **_kwargs):
            calls.append(paths)
            output.write_bytes(b"single m4b")

        with (
            patch.object(assemble, "_wav_duration", return_value=1_000_000_000.0),
            patch.object(assemble, "_run_ffmpeg", fake_ffmpeg),
        ):
            outputs = assemble.assemble(
                self.plan[:2], wavs, self.root / "output", "book",
                title="Книга", author=None, sample_rate=8000,
                work_dir=self.root / "work",
            )

        self.assertEqual([self.root / "output" / "book.m4b"], outputs)
        self.assertEqual([wavs], calls)


if __name__ == "__main__":
    unittest.main()
