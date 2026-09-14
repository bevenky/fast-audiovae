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
        with patch("onnxruntime.__version__", "1.30.0"):
            result = self.setup()
        self.assertEqual(result["recipe"], "portable")
        self.assertIn("platform wheel", result["reason"])
        self.assertEqual(self.calls[0]["recipe"], "portable")

    def test_selected_apple_is_automatic_and_four_thread_request_is_retained(self):
        self.cpu.update(system="Darwin", machine="arm64", platform="Darwin/arm64", vendor="apple",
                        usable={"neon": True, "sme": True, "sme2": True}, probe={"status": "ok"})
        payload = {"identity": "selected-wheel", "manifest": {"wheel_platform": "macosx_26_0_arm64"}}
        with patch("onnxruntime.__version__", "1.30.0"), \
             patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
             patch("fast_audiovae.native_payload.materialize_payload", return_value={"base_build": "verified"}), \
             patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")):
            result = self.setup(threads=4)
        self.assertEqual(result["recipe"], "apple_stream_selected")
        self.assertEqual(result["mode"], "streaming")
        self.assertEqual(result["threads"], 4)
        self.assertFalse(result["fallback"])

    def test_apple_runtime_contract_does_not_silently_load_legacy_runtime(self):
        self.cpu.update(platform="Darwin/arm64", system="Darwin", machine="arm64", vendor="apple",
                        usable={"neon": True, "sme": True, "sme2": True}, probe={"status": "ok"})
        with patch("onnxruntime.__version__", "1.29.0"):
            result = self.setup()
        self.assertEqual(result["recipe"], "portable")
        self.assertIn("1.30.0", result["reason"])

    def test_selected_amd_supports_declared_runtimes_without_changing_intel(self):
        self.cpu.update(platform="Linux/x86_64", system="Linux", machine="x86_64", vendor="amd",
                        usable={"avx2": True, "avx512": True, "avx512_vnni": True},
                        probe={"status": "ok"})
        payload = {"identity": "selected-amd", "manifest": {"wheel_platform": "linux_x86_64",
                   "recipes": {"amd_stream_selected": {}}}}
        for version in ("1.29.0", "1.30.0"):
            with self.subTest(version=version), patch("onnxruntime.__version__", version), \
                 patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
                 patch("fast_audiovae.native_payload.materialize_payload", return_value={"manifest": "verified"}), \
                 patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")), \
                 tempfile.TemporaryDirectory() as cache:
                result = automatic.setup(cache_dir=cache, source=self.root / "model.onnx")
            self.assertEqual(result["recipe"], "amd_stream_selected")
            self.assertEqual(result["threads"], 1)
            self.assertFalse(result["fallback"])
        self.assertEqual(automatic._recipe_runtimes("intel_stream_projection", self.cpu), ("1.29.0",))
        self.assertEqual(automatic._recipe_runtimes("amd_precision", self.cpu), ("1.29.0",))

    def test_legacy_amd_wheel_retains_its_working_recipe(self):
        self.cpu.update(platform="Linux/x86_64", system="Linux", machine="x86_64", vendor="amd",
                        usable={"avx2": True, "avx512": True, "avx512_vnni": True},
                        probe={"status": "ok"})
        payload = {"identity": "legacy-amd", "manifest": {"wheel_platform": "linux_x86_64",
                   "recipes": {"amd_precision": {}}}}
        with patch("onnxruntime.__version__", "1.29.0"), \
             patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
             patch("fast_audiovae.native_payload.materialize_payload", return_value={"manifest": "verified"}), \
             patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")):
            result = self.setup()
        self.assertEqual(result["recipe"], "amd_precision")
        self.assertFalse(result["fallback"])
        self.assertIn("predates", result["reason"])

    def test_selected_intel_is_default_with_qualified_runtime_and_payload(self):
        self.cpu.update(platform="Linux/x86_64", system="Linux", machine="x86_64", vendor="intel",
                        usable={"avx2": True, "avx512": True, "avx512_vnni": True}, probe={"status": "ok"})
        payload = {"identity": "selected-intel", "manifest": {"wheel_platform": "linux_x86_64",
                   "recipes": {"intel_stream_selected": {}, "intel_stream_projection": {}, "intel_precision": {}}}}
        with patch("onnxruntime.__version__", "1.29.0"), \
             patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
             patch("fast_audiovae.native_payload.materialize_payload", return_value={"manifest": "verified"}), \
             patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")):
            streaming = self.setup()
            batch = self.setup(mode="batch", threads=2)
        self.assertEqual((streaming["recipe"], streaming["mode"], streaming["threads"]), ("intel_stream_selected", "streaming", 1))
        self.assertFalse(streaming["fallback"])
        self.assertEqual((batch["recipe"], batch["threads"]), ("intel_precision", 2))
        self.assertNotEqual(streaming["cache_key"], batch["cache_key"])
        self.assertEqual(automatic._recipe_runtimes("intel_stream_selected", self.cpu), ("1.29.0",))

    def test_legacy_intel_wheel_retains_its_paired_streaming_recipe(self):
        self.cpu.update(platform="Linux/x86_64", system="Linux", machine="x86_64", vendor="intel",
                        usable={"avx2": True, "avx512": True, "avx512_vnni": True}, probe={"status": "ok"})
        payload = {"identity": "legacy-intel", "manifest": {"wheel_platform": "linux_x86_64",
                   "recipes": {"intel_stream_projection": {}, "intel_precision": {}}}}
        with patch("onnxruntime.__version__", "1.29.0"), \
             patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
             patch("fast_audiovae.native_payload.materialize_payload", return_value={"manifest": "verified"}), \
             patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")):
            result = self.setup()
        self.assertEqual(result["recipe"], "intel_stream_projection")
        self.assertFalse(result["fallback"])
        self.assertIn("predates", result["reason"])

    def test_selected_intel_does_not_load_unqualified_runtime(self):
        self.cpu.update(platform="Linux/x86_64", system="Linux", machine="x86_64", vendor="intel",
                        usable={"avx2": True, "avx512": True, "avx512_vnni": True}, probe={"status": "ok"})
        with patch("onnxruntime.__version__", "1.30.0"):
            result = self.setup()
        self.assertEqual(result["recipe"], "portable")
        self.assertIn("1.29.0", result["reason"])
        self.assertFalse(any(row["recipe"] == "intel_stream_selected" for row in self.calls))

    def test_selected_apple_batch_does_not_change_streaming_default(self):
        self.cpu.update(system="Darwin", machine="arm64", platform="Darwin/arm64", vendor="apple",
                        usable={"neon": True, "sme": True, "sme2": True}, probe={"status": "ok"})
        payload = {"identity": "selected-wheel", "manifest": {"wheel_platform": "macosx_26_0_arm64"}}
        with patch("onnxruntime.__version__", "1.30.0"), \
             patch("fast_audiovae.native_payload.inspect_payload", return_value=payload), \
             patch("fast_audiovae.native_payload.materialize_payload", return_value={"base_build": "verified"}), \
             patch("fast_audiovae.native_payload.probe_payload", return_value=(True, "")):
            batch = self.setup(mode="batch", threads=4)
            default = self.setup()
        self.assertEqual((batch["recipe"], batch["mode"], batch["threads"]), ("apple_batch_selected", "batch", 4))
        self.assertEqual((default["recipe"], default["mode"], default["threads"]), ("apple_stream_selected", "streaming", 1))
        self.assertNotEqual(batch["cache_key"], default["cache_key"])

    def test_public_load_defaults_to_streaming(self):
        import fast_audiovae
        with patch.object(automatic, "load", return_value="loaded") as load:
            self.assertEqual(fast_audiovae.load(), "loaded")
        load.assert_called_once_with(mode="streaming", threads=1, device="cpu")

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
