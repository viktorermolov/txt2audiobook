import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

# synth imports clean, which imports num2words. These tests exercise file and
# HTTP integrity only and must also run outside the container image.
_STUBBED_NUM2WORDS = "num2words" not in sys.modules
if _STUBBED_NUM2WORDS:
    module = types.ModuleType("num2words")
    module.num2words = lambda value, **_: str(value)
    sys.modules["num2words"] = module

from app.pipeline import synth  # noqa: E402 - temporary dependency stub above

if _STUBBED_NUM2WORDS:
    del sys.modules["num2words"]


MODEL_ID = "v5_5_ru"


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = Path(self.tmp.name) / "voices"
        self.good_model = b"verified fake Silero package"
        self.good_digest = hashlib.sha256(self.good_model).hexdigest()
        self.allowlist = patch.dict(
            synth._MODEL_SHA256_ALLOWLIST,
            {MODEL_ID: self.good_digest},
            clear=True,
        )
        self.allowlist.start()
        synth._MODEL_HASH_CACHE.clear()

    def tearDown(self) -> None:
        self.allowlist.stop()
        synth._MODEL_HASH_CACHE.clear()
        self.tmp.cleanup()

    def _client_for(self, handler):
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_checksum_failure_preserves_existing_model_and_cleans_temporary_file(self) -> None:
        self.voices.mkdir(parents=True)
        existing = self.voices / f"{MODEL_ID}.pt"
        existing.write_bytes(b"old corrupt cache")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual("https", request.url.scheme)
            return httpx.Response(200, content=b"bad replacement")

        client = self._client_for(handler)
        with patch.object(synth.httpx, "Client", return_value=client):
            with self.assertRaisesRegex(synth.SynthError, "скачать или проверить"):
                synth.ensure_voice(MODEL_ID, self.voices)

        self.assertEqual(b"old corrupt cache", existing.read_bytes())
        self.assertEqual([], list(self.voices.glob("*.part")))

    def test_unknown_model_id_is_rejected_before_path_or_network_use(self) -> None:
        with patch.object(synth, "_download") as download:
            with self.assertRaisesRegex(synth.SynthError, "Неподдерживаемая"):
                synth.ensure_voice("../../outside", self.voices)
        download.assert_not_called()
        self.assertFalse(self.voices.exists())

    def test_download_rejects_redirects_without_following_them(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                302,
                headers={"Location": "https://example.invalid/model.pt"},
                request=request,
            )

        client = self._client_for(handler)
        dest = self.voices / f"{MODEL_ID}.pt"
        with patch.object(synth.httpx, "Client", return_value=client) as factory:
            with self.assertRaisesRegex(synth.SynthError, "Перенаправление"):
                synth._download(
                    "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
                    dest,
                    self.good_digest,
                )
        self.assertEqual(1, len(requests))
        self.assertFalse(dest.exists())
        self.assertEqual([], list(self.voices.glob("*.part")))
        self.assertFalse(factory.call_args.kwargs["follow_redirects"])

    def test_download_rejects_non_https_source_before_network_use(self) -> None:
        with patch.object(synth.httpx, "Client") as client:
            with self.assertRaisesRegex(synth.SynthError, "Недопустимый адрес"):
                synth._download(
                    "http://models.silero.ai/models/tts/ru/v5_5_ru.pt",
                    self.voices / f"{MODEL_ID}.pt",
                    self.good_digest,
                )
        client.assert_not_called()

    def test_unverified_cached_model_is_rejected_before_torch_package_load(self) -> None:
        model = self.voices / f"{MODEL_ID}.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"corrupt cache")
        importer = unittest.mock.Mock()
        fake_torch = types.SimpleNamespace(
            package=types.SimpleNamespace(PackageImporter=importer),
            set_num_threads=lambda _: None,
            set_grad_enabled=lambda _: None,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(synth.SynthError, "Контрольная сумма"):
                synth._SileroEngine(model, "eugene", 1)
        importer.assert_not_called()

    def test_same_size_cached_replacement_is_hashed_again(self) -> None:
        model = self.voices / f"{MODEL_ID}.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(self.good_model)
        synth._verify_model_file(model)

        # Same-size content must not inherit a previous verification result.
        model.write_bytes(b"x" * len(self.good_model))
        with self.assertRaisesRegex(synth.SynthError, "Контрольная сумма"):
            synth._verify_model_file(model)

    def test_symlinked_model_is_rejected_before_torch_package_load(self) -> None:
        self.voices.mkdir(parents=True)
        target = self.voices / "model-target.pt"
        target.write_bytes(self.good_model)
        model = self.voices / f"{MODEL_ID}.pt"
        model.symlink_to(target)
        importer = unittest.mock.Mock()
        fake_torch = types.SimpleNamespace(
            package=types.SimpleNamespace(PackageImporter=importer),
            set_num_threads=lambda _: None,
            set_grad_enabled=lambda _: None,
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(synth.SynthError, "символической ссылкой"):
                synth._SileroEngine(model, "eugene", 1)
        importer.assert_not_called()

    def test_reopened_model_must_match_verified_metadata(self) -> None:
        model = self.voices / f"{MODEL_ID}.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(self.good_model)
        signature = synth._verify_model_file(model)

        model.write_bytes(b"x" * len(self.good_model))
        with self.assertRaisesRegex(synth.SynthError, "изменился перед загрузкой"):
            synth._open_model_file(model, signature)

    def test_importer_receives_rechecked_file_descriptor(self) -> None:
        model = self.voices / f"{MODEL_ID}.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(self.good_model)
        test_case = self

        class FakeImporter:
            def __init__(self, source) -> None:
                test_case.assertTrue(hasattr(source, "fileno"))
                test_case.assertEqual(test_case.good_model, source.read())
                source.seek(0)

            def load_pickle(self, *_):
                return types.SimpleNamespace(speakers=[], symbols=[], to=lambda _: None)

        fake_torch = types.SimpleNamespace(
            package=types.SimpleNamespace(PackageImporter=FakeImporter),
            device=lambda _: "cpu",
            set_num_threads=lambda _: None,
            set_grad_enabled=lambda _: None,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            synth._SileroEngine(model, "eugene", 1)

    def test_logs_do_not_contain_download_error_details(self) -> None:
        self.voices.mkdir(parents=True)
        existing = self.voices / f"{MODEL_ID}.pt"
        existing.write_bytes(b"bad cache")

        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("credential=secret-value")

        client = self._client_for(handler)
        with patch.object(synth.httpx, "Client", return_value=client):
            with self.assertLogs(synth.log.name, level="INFO") as logs:
                with self.assertRaises(synth.SynthError) as raised:
                    synth.ensure_voice(MODEL_ID, self.voices)
        self.assertNotIn("secret-value", "\n".join(logs.output))
        self.assertNotIn("secret-value", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
