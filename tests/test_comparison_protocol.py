"""Pure protocol checks: tiny arrays and metadata only, no ORT model execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load_script('compare_decoders')
diagnostic = load_script('diagnose_cpu')
comparison.np = np


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.contract = {'sample_rate': 8, 'hop': 2, 'latent_channels': 2}
        self.z = np.zeros((1, 2, 17), np.float32)
        self.ref = np.zeros((1, 1, 34), np.float32)

    def write_cases(self, **arrays):
        path = self.root / 'cases.npz'
        np.savez(path, **(arrays or {'a__z': self.z, 'a__ref': self.ref}))
        path.with_suffix('.json').write_text(json.dumps({'timed_ids': ['a']}))
        return path

    def config(self):
        (self.root / 'model.onnx').write_bytes(b'protocol test model placeholder')
        self.write_cases()
        return {
            'models': [{'name': 'audio_stock', 'kind': 'audio', 'path': 'model.onnx', 'causal': True}],
            'kind_contracts': {'audio': self.contract}, 'audio_cases': 'cases.npz',
            'expected_case_count': 1, 'expected_uids': ['a'], 'timing_uids': ['a'],
            'artifact_sha256': {name: comparison.sha(self.root / name)
                               for name in ('model.onnx', 'cases.npz', 'cases.json')},
        }

    def read_config(self, config):
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        with patch.object(comparison, 'model_external_data', return_value=set()):
            return comparison.read_config(path, comparison.sha(path))

    def test_import_does_not_import_ort(self):
        self.assertIsNone(comparison.ort)
        self.assertEqual(comparison.SOURCE_CAMPAIGN_SHA256,
                         'cbd847dd70398c811cec1b9a6e412b3486fb14f7df678879817742f891ec3922')

    def test_config_and_all_artifact_hashes_are_verified(self):
        config = self.config()
        result, verified = self.read_config(config)
        self.assertEqual(result, config)
        self.assertEqual(verified, config['artifact_sha256'])
        (self.root / 'cases.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.read_config(config)

    def test_config_hash_is_required_before_parsing(self):
        path = self.root / 'config.json'
        path.write_text('not json')
        for value in ('missing', '0' * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.read_config(path, value)

    def test_missing_companion_hash_is_rejected(self):
        config = self.config()
        del config['artifact_sha256']['cases.json']
        with self.assertRaisesRegex(ValueError, 'every input artifact'):
            self.read_config(config)

    def test_non_ort_or_implicit_causality_is_rejected(self):
        for change in ({'backend': 'native'}, {'causal': None}, {'causal': 1}):
            with self.subTest(change=change):
                config = self.config()
                config['models'][0].update(change)
                with self.assertRaises(ValueError):
                    self.read_config(config)

    def test_absolute_and_windows_paths_are_rejected(self):
        for value in ('/model.onnx', 'C:/model.onnx', 'C:\\model.onnx', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.relative_path(self.root, value)
        self.assertEqual(comparison.relative_path(self.root, 'models/a.onnx'), self.root / 'models/a.onnx')

    def test_external_file_declarations_must_match_graph(self):
        config = self.config()
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        with patch.object(comparison, 'model_external_data', return_value={self.root / 'missing.data'}):
            with self.assertRaisesRegex(ValueError, 'exactly list'):
                comparison.read_config(path, comparison.sha(path))

    def test_external_tensor_visitor_includes_nested_attributes(self):
        class Tensor:
            EXTERNAL = 1
            data_location = 1
            external_data = [SimpleNamespace(key='location', value='weights.data')]
            def ListFields(self):
                return []

        field = SimpleNamespace(type=11, TYPE_MESSAGE=11, is_repeated=True)
        graph = SimpleNamespace(ListFields=lambda: [(field, [Tensor()])])
        fake_onnx = SimpleNamespace(TensorProto=Tensor, load=lambda *a, **k: graph)
        with patch.dict(sys.modules, {'onnx': fake_onnx}):
            self.assertEqual(comparison.model_external_data(self.root / 'model.onnx'),
                             {self.root / 'weights.data'})

    def test_exact_full_reference_is_retained(self):
        cases, meta = comparison.read_cases(self.write_cases(), self.contract)
        self.assertEqual(meta['timed_ids'], ['a'])
        self.assertEqual(cases['a']['ref'].shape, (1, 1, 34))
        self.assertTrue(np.array_equal(cases['a']['ref'], self.ref))

    def test_truncated_flat_wrong_dtype_or_nonfinite_reference_is_rejected(self):
        for ref in (self.ref[..., :-1], self.ref.ravel(), self.ref.astype(np.float64),
                    np.full_like(self.ref, np.nan)):
            with self.subTest(shape=ref.shape, dtype=ref.dtype), self.assertRaises(ValueError):
                comparison.validate_case(self.z, ref, self.contract, 'a')

    def test_wrong_latent_contract_is_rejected(self):
        for z in (self.z.astype(np.float64), self.z[0], self.z[:, :1], self.z[..., :0],
                  np.full_like(self.z, np.inf)):
            with self.subTest(shape=z.shape, dtype=z.dtype), self.assertRaises(ValueError):
                comparison.validate_case(z, self.ref, self.contract, 'a')

    def test_noncontiguous_cases_get_safe_copies(self):
        z = np.zeros((1, 2, 34), np.float32)[..., ::2]
        ref = np.zeros((1, 1, 68), np.float32)[..., ::2]
        result = comparison.validate_case(z, ref, self.contract, 'a')
        self.assertTrue(result['z'].flags.c_contiguous and result['ref'].flags.c_contiguous)

    def test_orphan_references_and_insufficient_probe_length_are_rejected(self):
        for arrays in ({'a__z': self.z, 'b__ref': self.ref},
                       {'a__z': self.z[..., :2], 'a__ref': self.ref[..., :4]}):
            with self.subTest(keys=list(arrays)), self.assertRaises(ValueError):
                comparison.read_cases(self.write_cases(**arrays), self.contract)

    def test_comparison_rejects_shape_dtype_and_nonfinite_errors(self):
        self.assertTrue(comparison.compare(self.ref, self.ref)['passed'])
        for actual in (self.ref[..., :-1], self.ref.astype(np.float64), np.full_like(self.ref, np.inf)):
            with self.subTest(shape=actual.shape, dtype=actual.dtype):
                self.assertFalse(comparison.compare(actual, self.ref)['passed'])

    def test_timing_subset_is_matched_and_does_not_mutate_metadata(self):
        data = {'audio': {'a': {}, 'b': {}}, 'mimi': {'b': {}, 'a': {}}}
        meta = {kind: {'timed_ids': ['a', 'b']} for kind in data}
        self.assertEqual(comparison.select_timing_ids(data, meta, 'all', ['b']),
                         {'audio': ['b'], 'mimi': ['b']})
        self.assertEqual(meta['audio']['timed_ids'], ['a', 'b'])
        for ids in ([], ['a', 'a'], ['missing']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                comparison.select_timing_ids(data, meta, 'all', ids)

    def test_provider_proof_rejects_non_cpu_and_missing_provider(self):
        def event(provider):
            return {'cat': 'Node', 'name': 'op_kernel_time', 'args': {'provider': provider}}
        self.assertTrue(comparison.cpu_profile_report([event('CPUExecutionProvider')])['passed'])
        for events in ([], [event(None)], [event('CUDAExecutionProvider')]):
            self.assertFalse(comparison.cpu_profile_report(events)['passed'])

    def test_cpu_visibility_is_explicit(self):
        with patch.dict(comparison.os.environ, comparison.CPU_ENV, clear=True):
            self.assertEqual(comparison.require_cpu_environment(), comparison.CPU_ENV)
        with patch.dict(comparison.os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            comparison.require_cpu_environment()

    def test_platform_counter_absence_does_not_crash(self):
        with patch.object(comparison, 'optional_text', return_value=None), \
                patch.object(comparison.os, 'getloadavg', side_effect=OSError, create=True), \
                patch.object(comparison.os, 'sysconf', side_effect=ValueError, create=True):
            snapshot = comparison.host_snapshot()
        self.assertIsNone(snapshot['cpu_ticks'])
        self.assertIsNone(snapshot['loadavg'])
        self.assertIsNone(comparison.host_delta(snapshot, snapshot)['steal_percent'])
        with patch.object(diagnostic.os, 'sched_getaffinity', side_effect=OSError, create=True), \
                patch.object(diagnostic.os, 'getloadavg', side_effect=NotImplementedError, create=True):
            self.assertIsNone(diagnostic.current_affinity())
            self.assertIsNone(diagnostic.load_average())

    def test_counter_reset_and_guest_accounting(self):
        before = {'cpu_ticks': [0] * 10}
        after = {'cpu_ticks': [10, 0, 0, 80, 5, 0, 0, 5, 100, 100]}
        result = comparison.host_delta(before, after)
        self.assertEqual(result['total_ticks'], 100)
        self.assertEqual(result['steal_percent'], 5)
        self.assertIsNone(comparison.host_delta(after, before)['steal_percent'])

    def test_affinity_plan_rejects_invalid_or_unavailable_masks(self):
        path = self.root / 'affinity.json'
        for value, original in (({'4': [0, 1]}, {0, 1}), ({'1': [7]}, {0}),
                                ({'1': [0, 0]}, {0}), ({'1': [False]}, {0}), ({'1': [0]}, None)):
            path.write_text(json.dumps(value))
            count = int(next(iter(value)))
            with self.subTest(value=value), self.assertRaises((ValueError, RuntimeError)):
                comparison.affinity_plan(path, [count], original)

    def test_diagnostic_cgroup_parsers_handle_v1_v2_and_malformed_data(self):
        self.assertEqual(diagnostic.quota_from_files({'cpu.max': '150000 100000'}), 1.5)
        self.assertEqual(diagnostic.quota_from_files({'cpu.cfs_quota_us': '200000', 'cpu.cfs_period_us': '100000'}), 2)
        self.assertIsNone(diagnostic.quota_from_files({'cpu.max': 'invalid 100000'}))
        self.assertEqual(diagnostic.cpu_list('0-2,4'), [0, 1, 2, 4])
        self.assertIsNone(diagnostic.resolve_cgroup({'root': '/', 'mount': '/cg'}, '/../escape')[0])


if __name__ == '__main__':
    unittest.main()
