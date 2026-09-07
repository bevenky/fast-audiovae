"""Backend selection tests do not require native binaries or model weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

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

    def test_automatic_workers_respect_process_affinity(self):
        with patch.object(runtime.os, 'sched_getaffinity', return_value={6, 7}, create=True), \
             patch.object(runtime.os, 'cpu_count', return_value=192), \
             patch.object(runtime.platform, 'system', return_value='Windows'):
            session, info = runtime.load_decoder(self.root)
        self.assertEqual(session.options.intra_op_num_threads, 2)
        self.assertEqual(info['threads'], 2)
        self.assertEqual(info['thread_policy'], 'visible_cpus_capped_at_four')

    def test_large_affinity_retains_four_worker_default(self):
        with patch.object(runtime.os, 'sched_getaffinity', return_value=set(range(18)), create=True):
            self.assertEqual(runtime._default_threads(), 4)

    def test_unavailable_affinity_uses_cpu_count(self):
        for error in (OSError, AttributeError):
            for count, expected in ((2, 2), (18, 4), (None, 1)):
                with self.subTest(error=error.__name__, count=count), \
                     patch.object(runtime.os, 'sched_getaffinity', side_effect=error('unavailable'), create=True), \
                     patch.object(runtime.os, 'cpu_count', return_value=count):
                    self.assertEqual(runtime._default_threads(), expected)

    def test_explicit_worker_count_is_preserved(self):
        with patch.object(runtime, '_default_threads') as detect, \
             patch.object(runtime.platform, 'system', return_value='Windows'):
            session, info = runtime.load_decoder(self.root, threads=4)
        detect.assert_not_called()
        self.assertEqual(session.options.intra_op_num_threads, 4)
        self.assertEqual(info['thread_policy'], 'explicit')

    def native_library(self, vector_sine=True):
        return SimpleNamespace(ncc_abi_version=Mock(return_value=1),
            ncc_selected_backend=Mock(return_value=3), ncc_capabilities=Mock(return_value=64),
            ncc_snake_math_name=Mock(return_value=b'sleef_u10'),
            ncc_vector_sine_available=Mock(return_value=int(vector_sine)))

    def test_x86_selects_native_only_with_accurate_vector_sine(self):
        for eligible in (False, True):
            with self.subTest(eligible=eligible), \
                 patch.object(runtime.platform, 'system', return_value='Linux'), \
                 patch.object(runtime.platform, 'machine', return_value='AMD64'), \
                 patch.object(runtime.ctypes, 'CDLL', return_value=self.native_library(eligible)):
                session, info = runtime.load_decoder(self.root)
            self.assertEqual(info['selected'], 'native' if eligible else 'portable_onnx')
            self.assertEqual(len(session.options.libraries), int(eligible))
            self.assertEqual(session.providers, ['CPUExecutionProvider'])

    def test_amd_requires_opt_in_supported_cpu_and_validated_threads(self):
        self.manifest['native']['Linux/x86_64']['packed'] = {
            'library': 'packed.so', 'model': 'packed.onnx', 'validated_threads': [1, 4],
            'experiment': 'phase_aocl_rows', 'tested_cpu': 'AMD EPYC 9654'}
        self.write_manifest()
        for requested, supported, threads in ((False, True, 4), (True, False, 4),
                                               (True, True, 2), (True, True, 4)):
            packed = SimpleNamespace(ncc_aocl_cpu_supported=Mock(return_value=int(supported)),
                                     ncc_aocl_adapter_abi=Mock(return_value=1))
            with self.subTest(requested=requested, supported=supported, threads=threads), \
                 patch.object(runtime.platform, 'system', return_value='Linux'), \
                 patch.object(runtime.platform, 'machine', return_value='x86_64'), \
                 patch.object(runtime.ctypes, 'CDLL', side_effect=[self.native_library(), packed]) as load:
                session, info = runtime.load_decoder(self.root, threads=threads, prefer_packed=requested)
            expected = requested and supported and threads in (1, 4)
            self.assertEqual(info.get('packed_weights', False), expected)
            self.assertEqual(Path(session.path).name, 'packed.onnx' if expected else 'native.onnx')
            self.assertEqual(load.call_count, 2 if requested and threads in (1, 4) else 1)

    def test_native_abi_mismatch_fails_before_session_creation(self):
        library = self.native_library()
        library.ncc_abi_version.return_value = 2
        with patch.object(runtime.platform, 'system', return_value='Linux'), \
             patch.object(runtime.platform, 'machine', return_value='x86_64'), \
             patch.object(runtime.ctypes, 'CDLL', return_value=library), \
             patch.object(runtime.ort, 'InferenceSession') as create:
            with self.assertRaisesRegex(RuntimeError, 'ABI mismatch'):
                runtime.load_decoder(self.root)
        create.assert_not_called()

    def test_non_cpu_provider_is_rejected(self):
        with patch.object(runtime.platform, 'system', return_value='Windows'), \
             patch.object(runtime.platform, 'machine', return_value='AMD64'), \
             patch.object(FakeSession, 'get_providers', return_value=['CUDAExecutionProvider']):
            with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
                runtime.load_decoder(self.root)


if __name__ == "__main__":
    unittest.main()
