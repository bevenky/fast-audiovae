"""Backend selection tests do not require native binaries or model weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fast_audiovae import runtime


class FakeOptions:
    def __init__(self):
        self.settings = {}
        self.libraries = []

    def add_session_config_entry(self, key, value):
        self.settings[key] = value

    def register_custom_ops_library(self, path):
        self.libraries.append(path)


class FakeSession:
    def __init__(self, path, sess_options, providers):
        self.path, self.options, self.providers = path, sess_options, providers
        self.fallback_disabled = False

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_providers(self):
        return self.providers


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {"onnxruntime": "1.29.0", "fallback": "decoder.onnx", "native": {
            "Linux/x86_64": {"library": "native.so", "model": "native.onnx", "math": "sleef_u10",
                             "experiment": "phase_fused", "tested_cpu": "AMD EPYC 9654"}}}
        self.write_manifest()
        self.patches = [patch.object(runtime.ort, "SessionOptions", FakeOptions),
                        patch.object(runtime.ort, "InferenceSession", FakeSession),
                        patch.object(runtime.ort, "__version__", "1.29.0")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def write_manifest(self):
        (self.root / "bundle.json").write_text(json.dumps(self.manifest))

    def test_unsupported_platform_uses_only_cpu(self):
        with patch.object(runtime.platform, "system", return_value="Windows"), \
             patch.object(runtime.platform, "machine", return_value="AMD64"):
            session, info = runtime.load_decoder(self.root)
        self.assertEqual(info["selected"], "portable_onnx")
        self.assertEqual(session.providers, ["CPUExecutionProvider"])
        self.assertTrue(session.fallback_disabled)
        self.assertEqual(session.options.settings["session.intra_op.allow_spinning"], "0")

    def test_missing_native_library_falls_back(self):
        with patch.object(runtime.platform, "system", return_value="Linux"), \
             patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime.ctypes, "CDLL", side_effect=OSError("missing")):
            session, info = runtime.load_decoder(self.root)
        self.assertEqual(info["selected"], "portable_onnx")
        self.assertTrue(session.fallback_disabled)

    def test_wrong_runtime_does_not_load_native_binary(self):
        with patch.object(runtime.platform, "system", return_value="Linux"), \
             patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime.ort, "__version__", "1.28.0"), \
             patch.object(runtime.ctypes, "CDLL") as load:
            _, info = runtime.load_decoder(self.root)
        load.assert_not_called()
        self.assertEqual(info["selected"], "portable_onnx")

    def test_invalid_thread_counts_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.load_decoder(self.root, threads=value)


if __name__ == "__main__":
    unittest.main()
