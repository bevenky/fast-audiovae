"""Apple experiment source and build-contract checks. No model execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('apple_stage_build', ROOT/'build.py')
build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)


class ApplePackagingTests(unittest.TestCase):
    def test_frozen_native_sources_and_shared_matrix_match(self):
        record = json.loads((ROOT/'source-hashes.json').read_text())
        self.assertFalse(record['default_runtime_integration'])
        self.assertEqual(set(record['source_sha256']), {'custom_op.cpp', 'stage_dw.h',
                         '../matrix/fused_pointwise.c', '../matrix/fused_pointwise.h'})
        for path, expected in record['source_sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), expected)

    def test_python_sources_parse(self):
        for path in ROOT.rglob('*.py'):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(), filename=str(path))

    def test_evidence_contains_correctness_without_timing_claim(self):
        record = json.loads((ROOT/'test-summary.json').read_text())
        self.assertEqual(record['focused']['comparison_records'], 270)
        self.assertEqual(record['focused']['reference_bitwise_equal'], 270)
        self.assertEqual(record['focused']['segmentation_bitwise_equal'], 270)
        self.assertEqual(record['captured']['comparison_records'], 12)
        self.assertEqual(record['captured']['bitwise_equal'], 12)
        self.assertFalse(record['timing_benchmark'])
        self.assertFalse(record['full_decoder_waveform_gate'])
        self.assertFalse(record['default_runtime_integration'])

    def test_native_build_validation_and_deployment_target(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp)/'native.dylib'; library.write_bytes(b'fixture-library')
            header = Path(temp)/'native_kernels.h'; header.write_bytes(b'fixture-header')
            record = {'domain': 'venky.audio.cpu', 'native_abi': 1, 'ort_api_version': 29,
                      'library_sha256': build.sha(library), 'accelerate': True,
                      'fingerprint': {'architecture': 'arm64', 'deployment_target': '26.2',
                          'source_sha256': {'native/apple/native_kernels.h': build.sha(header)}}}
            self.assertEqual(build.validate_native(record, library, header), '26.2')
            for field, value in [('domain', 'venky.audio.cpu.portable'), ('native_abi', 2),
                                 ('ort_api_version', 28), ('accelerate', False), ('library_sha256', 'bad')]:
                changed = {**record, field: value}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    build.validate_native(changed, library, header)
            for field, value in [('architecture', 'x86_64'), ('deployment_target', '-unsafe'),
                                 ('source_sha256', {})]:
                changed = {**record, 'fingerprint': {**record['fingerprint'], field: value}}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    build.validate_native(changed, library, header)


if __name__ == '__main__':
    unittest.main()
