"""Static source identity and packaging checks. No compiler or model execution."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


class PackagingTests(unittest.TestCase):
    def test_native_sources_match_frozen_revision(self):
        record=json.loads((ROOT/'source-hashes.json').read_text())
        self.assertFalse(record['default_runtime_integration'])
        self.assertEqual(len(record['source_sha256']),5)
        for path,expected in record['source_sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected)

    def test_python_sources_parse_without_importing_probes(self):
        files=list(ROOT.rglob('*.py'))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(),filename=str(path))

    def test_headers_match_shared_dependency_record(self):
        tree=ast.parse((ROOT/'matrix/build.py').read_text())
        headers=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='HEADERS' for t in n.targets))
        pins=json.loads((ROOT/'dependency-pins.json').read_text())
        self.assertEqual(headers,pins['ort_headers']['sha256'])
        self.assertEqual(pins['onnxruntime'],'1.29.0')
        self.assertEqual(pins['onnx'],'1.22.0')

    def test_experiment_is_outside_packaged_source_and_defaults(self):
        self.assertFalse(ROOT.is_relative_to(REPO/'src'))
        for name in ('prepare.py','runtime.py','__init__.py'):
            text=(REPO/'src/fast_audiovae'/name).read_text()
            self.assertNotIn('PointwiseBiasResidualF32',text)
            self.assertNotIn('StageStackF32',text)
            self.assertNotIn('cpu-stage',text)


if __name__=='__main__':
    unittest.main()
