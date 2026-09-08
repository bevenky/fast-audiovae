"""Automatic installation, cache and public-mode contract without model inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fast_audiovae import automatic
from fast_audiovae.recipes.common import completed_bundle


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cpu = {"system": "Linux", "machine": "aarch64", "platform": "Linux/aarch64",
                    "vendor": "unknown", "max_threads": 4, "usable": {},
                    "probe": {"status": "unavailable"}}
        self.calls = []
        self.patches = [
            patch("fast_audiovae.platforms.detect_cpu", return_value=self.cpu),
            patch("fast_audiovae.native_payload.inspect_payload", return_value=None),
            patch("fast_audiovae.build_resources.resource_fingerprint", return_value="sources"),
            patch("fast_audiovae.build_resources.materialize_resources", return_value="sources"),
            patch.object(automatic, "verify_model"),
            patch.object(automatic, "_build", side_effect=self.build),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def build(self, work, source, cpu, plan):
        self.calls.append(plan.copy())
        bundle = work / "bundles" / plan["mode"]
        bundle.mkdir(parents=True)
        (bundle / "model.onnx").write_bytes(b"frozen graph")
        (bundle / "bundle.json").write_text(json.dumps({"fallback": "model.onnx", "native": {}}))
        return bundle

    def setup(self, **options):
        return automatic.setup(cache_dir=self.root, source=self.root / "model.onnx", **options)

    def test_default_streaming_uses_cache_and_never_requests_compilation(self):
        import fast_audiovae.platforms as platforms
        first = self.setup()
        second = self.setup()
        self.assertEqual(first["mode"], "streaming")
        self.assertEqual(first["threads"], 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(len(self.calls), 1)
        platforms.detect_cpu.assert_called_with(allow_compile=False)

    def test_mode_caches_are_independent(self):
        streaming = self.setup()
        batch = self.setup(mode="batch")
        self.assertNotEqual(streaming["cache_key"], batch["cache_key"])
        self.assertNotEqual(streaming["model_dir"], batch["model_dir"])

    def test_changed_cached_graph_is_rejected(self):
        first = self.setup()
        (Path(first["model_dir"]) / "model.onnx").write_bytes(b"different graph")
        with self.assertRaisesRegex(RuntimeError, "Cached decoder files changed"):
            self.setup()
        self.assertEqual(len(self.calls), 1)

    def test_native_without_wheel_falls_back_without_building_native(self):
        self.cpu.update(system="Darwin", machine="arm64", platform="Darwin/arm64", vendor="apple",
                        usable={"neon": True}, probe={"status": "ok"})
        result = self.setup()
        self.assertEqual(result["recipe"], "portable")
        self.assertIn("platform wheel", result["reason"])
        self.assertEqual(self.calls[0]["recipe"], "portable")

    def test_public_load_defaults_to_streaming(self):
        import fast_audiovae
        with patch.object(automatic, "load", return_value="loaded") as load:
            self.assertEqual(fast_audiovae.load(), "loaded")
        load.assert_called_once_with(mode="streaming", threads=1)

    def test_invalid_options_fail_before_build(self):
        for arguments in ({"mode": "auto"}, {"threads": 0}, {"threads": True}, {"build_native": "yes"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.setup(**arguments)
        self.assertEqual(self.calls, [])

    def test_payload_os_and_architecture_gate(self):
        payload = {"manifest": {"wheel_platform": "macosx_26_0_arm64", "minimum_macos": "26.2"}}
        cpu = {"platform": "Darwin/arm64", "system": "Darwin"}
        with patch.object(automatic.platform, "mac_ver", return_value=("26.1", (), "")):
            self.assertFalse(automatic._payload_supported(payload, cpu))
        with patch.object(automatic.platform, "mac_ver", return_value=("26.5", (), "")):
            self.assertTrue(automatic._payload_supported(payload, cpu))
        self.assertFalse(automatic._payload_supported(payload, {"platform": "Linux/x86_64", "system": "Linux"}))

    def test_linux_glibc_gate_precedes_native_loading(self):
        payload = {"manifest": {"wheel_platform": "linux_x86_64", "minimum_glibc": "2.38"}}
        cpu = {"platform": "Linux/x86_64", "system": "Linux"}
        for version, expected in (("2.35", False), ("2.38", True), ("2.39", True)):
            with self.subTest(version=version), patch.object(automatic.platform, "libc_ver", return_value=("glibc", version)):
                self.assertEqual(automatic._payload_supported(payload, cpu), expected)
        with patch.object(automatic.platform, "libc_ver", return_value=("musl", "1.2.5")):
            self.assertFalse(automatic._payload_supported(payload, cpu))


class CompletionTests(unittest.TestCase):
    def test_interrupted_recipe_is_never_published_or_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result"
            def broken(stage):
                stage.mkdir()
                (stage / "bundle.json").write_text('{"fallback":"missing.onnx"}')
                raise RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                completed_bundle(destination, broken)
            self.assertFalse(destination.exists())
            def good(stage):
                stage.mkdir()
                (stage / "model.onnx").write_bytes(b"graph")
                (stage / "bundle.json").write_text('{"fallback":"model.onnx"}')
            completed_bundle(destination, good)
            completed_bundle(destination, Mock(side_effect=AssertionError("must not rebuild")))
            (destination / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                completed_bundle(destination, good)


class FacadeTests(unittest.TestCase):
    def test_stream_history_lifecycle_delegates_to_existing_runtime(self):
        existing = Mock()
        wrapper = automatic.AudioVAEDecoder(existing, {}, "streaming")
        self.assertIs(wrapper.stream(), existing.streaming_decode.return_value)
        with self.assertRaisesRegex(RuntimeError, "stream"):
            wrapper.decode(np.zeros((1, 64, 1), np.float32))

    def test_batch_returns_session_audio_without_processing(self):
        session = Mock()
        session.get_inputs.return_value = [Mock(name="latent")]
        session.get_inputs.return_value[0].name = "z"
        expected = np.ones((1, 1, 1920), np.float32)
        session.run.return_value = [expected]
        wrapper = automatic.AudioVAEDecoder(session, {}, "batch")
        self.assertIs(wrapper.decode(np.zeros((1, 64, 1), np.float32)), expected)
        with self.assertRaisesRegex(RuntimeError, "streaming"):
            wrapper.stream()
        with self.assertRaisesRegex(ValueError, "finite"):
            wrapper.decode(np.full((1, 64, 1), np.nan, np.float32))
