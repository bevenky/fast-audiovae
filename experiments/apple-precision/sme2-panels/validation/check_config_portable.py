"""Small config metadata fixtures only: no ONNX, native library or model loading.

The injected helper hashes tiny stand-in files. Static ONNX external-data parsing
is owned by the unchanged pinned public helper and is not simulated as a result.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('portable_preparer', Path(__file__).with_name('prepare_config_portable.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class FixtureHelper:
    @staticmethod
    def relative_path(base, value):
        p.require(isinstance(value, str) and value and not Path(value).is_absolute() and '\\' not in value,
                  'Relative fixture path required')
        return (base / value).resolve()

    @classmethod
    def read_config(cls, path, expected):
        config = p.pinned_json(path, expected)
        for name, digest in config['artifact_sha256'].items():
            p.require(p.sha(cls.relative_path(path.parent, name)) == digest, 'Fixture artifact changed')
        required = set()
        for model in config['models']:
            required.add(model['path'])
            required.update(p.libraries(model))
            required.update(model.get('external_data', []))
            case = config[model['kind'] + '_cases']
            required.update((case, str(Path(case).with_suffix('.json'))))
        p.require(required <= set(config['artifact_sha256']), 'Fixture dependency missing')
        return config, config['artifact_sha256']

    @staticmethod
    def model_external_data(path):
        return set()


class ConfigChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        self.original_source_sha = p.SOURCE_GRAPH_SHA
        def file(name, content):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return path
        self.file = file
        stock = file('data/stock.graph', 'stock')
        fp32 = file('data/fp32.graph', 'fp32')
        old_int8 = file('data/old-int8.graph', 'old-int8')
        mimi = file('data/mimi.graph', 'mimi')
        native = file('data/native.library', 'native')
        old_ops = file('data/old-ops.library', 'old-ops')
        audio = file('data/audio.npz', 'audio case fixture')
        audio_meta = file('data/audio.json', '{}')
        mimi_cases = file('data/mimi.npz', 'mimi case fixture')
        mimi_meta = file('data/mimi.json', '{}')
        p.SOURCE_GRAPH_SHA = p.sha(fp32)
        source_file = file('package/native/precision.cpp', 'frozen source')
        file('package/validation/campaign_portable.py', 'frozen campaign fixture')
        file('package/validation/prepare_config_portable.py', 'frozen preparer fixture')
        self.core = file('build/core.library', 'core')
        ops = file('build/ops.library', 'ops')
        stage = file('fused/stage.library', 'stage')
        upsample = file('fused/upsample.library', 'upsample')
        self.graph = file('graphs/new.graph', 'new graph fixture')
        paths = [stock, fp32, old_int8, mimi, native, old_ops, audio, audio_meta, mimi_cases, mimi_meta]
        relative = lambda path: os.path.relpath(path, self.root).replace(os.sep, '/')
        self.source = {
            'expected_case_count': 60, 'expected_uids': ['uid' + str(i) for i in range(60)],
            'timing_uids': ['uid' + str(i) for i in range(0, 60, 6)],
            'frozen_manifest_sha256': '0' * 64, 'kind_contracts': copy.deepcopy(p.CONTRACTS),
            'models': [
                {'name': 'audio_stock', 'kind': 'audio', 'path': relative(stock), 'causal': True},
                {'name': 'fast_fp32', 'kind': 'audio', 'path': relative(fp32), 'causal': True,
                 'custom_library': relative(native)},
                {'name': 'int8_large', 'kind': 'audio', 'path': relative(old_int8), 'causal': True,
                 'approximate': True, 'custom_libraries': [relative(native), relative(old_ops)]},
                {'name': 'mimi', 'kind': 'mimi', 'path': relative(mimi), 'causal': True}],
            'audio_cases': relative(audio), 'mimi_cases': relative(mimi_cases),
            'artifact_sha256': {relative(path): p.sha(path) for path in paths}}
        self.reference = copy.deepcopy(self.source)
        core_build = {'ORT_API': 29, 'GPU_used': False, 'backend': 3, 'domain': p.DOMAINS[0],
                      'C_API_prefix': 'ipc_', 'core_path': str(self.core), 'ops_path': str(ops),
                      'libraries': {self.core.name: p.sha(self.core), ops.name: p.sha(ops)},
                      'sources': {'native/precision.cpp': p.sha(source_file)}}
        self.fused_build = {'ort_api': 29, 'gpu_used': False, 'tile_capacity': 512,
                            'no_internal_threads': True,
                            'libraries': {'stage': str(stage), 'upsample': str(upsample)},
                            'library_sha256': {'stage': p.sha(stage), 'upsample': p.sha(upsample)},
                            'sources_dependencies_sha256': {str(self.core): p.sha(self.core), str(native): p.sha(native)}}
        self.manifest = {'source_sha256': p.SOURCE_GRAPH_SHA, 'output_sha256': p.sha(self.graph),
                         'selected_products': 22, 'retained_FP32_products': 9, 'tile_time': 512,
                         'segments': 4, 'shards': 4, 'domains': list(p.DOMAINS),
                         'original_initializer_bytes_unchanged': True}
        self.args = SimpleNamespace(graph=self.graph, output=self.root / 'relocated/deep/config.json')
        for name, value in (('source_config', self.source), ('reference_config', self.reference),
                            ('core_build', core_build), ('fused_build', self.fused_build),
                            ('graph_manifest', self.manifest)):
            path = file(name + '.json', json.dumps(value))
            setattr(self.args, name, path)
            setattr(self.args, name + '_sha256', p.sha(path))

    def tearDown(self):
        p.SOURCE_GRAPH_SHA = self.original_source_sha
        self.temp.cleanup()

    def refresh(self, name, value):
        path = getattr(self.args, name)
        path.write_text(json.dumps(value))
        setattr(self.args, name + '_sha256', p.sha(path))

    def run_prepare(self):
        return p.prepare(self.args, FixtureHelper, self.package)

    def test_valid_relocation_preserves_cohort_and_all_pins(self):
        result = self.run_prepare()
        config = json.loads(self.args.output.read_text())
        self.assertEqual(config['expected_uids'], self.source['expected_uids'])
        self.assertEqual(config['timing_uids'], self.source['timing_uids'])
        self.assertEqual(config['kind_contracts'], self.source['kind_contracts'])
        self.assertFalse(result['model_execution'])
        for name, digest in config['artifact_sha256'].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(p.sha(self.args.output.parent / name), digest)
        for index in (0, 1, 3):
            self.assertEqual(p.model_identity(config['models'][index], self.args.output.parent, FixtureHelper),
                             p.model_identity(self.source['models'][index], self.root, FixtureHelper))
        self.assertEqual(len(config['models'][2]['custom_libraries']), 4)
        self.assertTrue(config['models'][2]['approximate'])
        self.assertEqual((self.args.output.parent / config['transitive_core_library']).resolve(), self.core.resolve())
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_baseline_or_cohort_change_rejected(self):
        original = copy.deepcopy(self.source)
        for mutate in (lambda x: x['timing_uids'].reverse(),
                       lambda x: x['expected_uids'].reverse(),
                       lambda x: x['models'][1].update(causal=False),
                       lambda x: x['models'][2].update(approximate=False),
                       lambda x: x['models'][0].update(extra_inputs={'new': 'array'}),
                       lambda x: x['kind_contracts']['audio'].update(hop=1919)):
            source = copy.deepcopy(original)
            mutate(source)
            self.refresh('source_config', source)
            with self.assertRaises(ValueError):
                self.run_prepare()
            self.assertFalse(self.args.output.exists())

    def test_graph_manifest_or_build_contract_change_rejected(self):
        for key, value in (('selected_products', 21), ('segments', 2), ('shards', 2),
                           ('original_initializer_bytes_unchanged', False), ('output_sha256', '0' * 64)):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = value
            self.refresh('graph_manifest', manifest)
            with self.assertRaises(ValueError):
                self.run_prepare()
        self.refresh('graph_manifest', self.manifest)
        broken = copy.deepcopy(self.fused_build)
        broken['sources_dependencies_sha256'].pop(str(self.core))
        self.refresh('fused_build', broken)
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_stale_json_hash_and_library_bytes_rejected(self):
        self.args.source_config_sha256 = '0' * 64
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.args.source_config_sha256 = p.sha(self.args.source_config)
        self.core.write_text('changed core')
        with self.assertRaises(ValueError):
            self.run_prepare()


if __name__ == '__main__':
    unittest.main()
