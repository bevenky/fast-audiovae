"""Small offline tests for explicit native build routing and pinned headers."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class BuildEntryPoint(unittest.TestCase):
    def test_import_has_no_network(self):
        with mock.patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
            self.assertTrue(callable(module('build').build))
            self.assertTrue(callable(module('fetch_headers').fetch))

    def test_header_pins_match_native_builders(self):
        headers = module('fetch_headers').HEADER_SHA256
        self.assertEqual(headers, module('build_x86').HEADERS)
        self.assertEqual(headers, module('build_apple').HEADER_SHA256)
        self.assertEqual(len(headers), 6)

    def test_offline_missing_header_is_rejected(self):
        fetcher = module('fetch_headers')
        with tempfile.TemporaryDirectory() as path, mock.patch.object(fetcher.urllib.request, 'urlopen') as network:
            with self.assertRaisesRegex(RuntimeError, 'Missing pinned ORT header'):
                fetcher.fetch(path, offline=True)
            network.assert_not_called()

    def test_mismatched_existing_header_is_not_overwritten(self):
        fetcher = module('fetch_headers')
        with tempfile.TemporaryDirectory() as path, mock.patch.object(fetcher.urllib.request, 'urlopen') as network:
            destination = Path(path)/'onnxruntime_c_api.h'
            destination.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                fetcher.fetch(path)
            self.assertEqual(destination.read_bytes(), b'changed')
            network.assert_not_called()

    def test_bad_download_is_never_installed(self):
        fetcher = module('fetch_headers')
        with tempfile.TemporaryDirectory() as path:
            with mock.patch.object(fetcher.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad response')):
                with self.assertRaisesRegex(RuntimeError, 'Downloaded ORT header SHA256 mismatch'):
                    fetcher.fetch(path)
            self.assertFalse((Path(path)/'onnxruntime_c_api.h').exists())

    def test_verified_header_reuse_makes_no_network_call(self):
        fetcher = module('fetch_headers')
        with tempfile.TemporaryDirectory() as path:
            data = b'fixture header'
            (Path(path)/'fixture.h').write_bytes(data)
            with mock.patch.object(fetcher, 'HEADER_SHA256', {'fixture.h': hashlib.sha256(data).hexdigest()}):
                with mock.patch.object(fetcher.urllib.request, 'urlopen') as network:
                    result = fetcher.fetch(path)
                    self.assertEqual(result['downloaded'], [])
                    network.assert_not_called()

    def test_wrong_platform_rejected_before_dependency_work(self):
        builder = module('build')
        with mock.patch.object(builder.platform, 'system', return_value='Windows'):
            with mock.patch.object(builder, '_module') as loader:
                with self.assertRaisesRegex(RuntimeError, 'Supported builds'):
                    builder.build()
                loader.assert_not_called()

    def test_amd_option_rejected_on_apple_before_dependency_work(self):
        builder = module('build')
        with mock.patch.object(builder.platform, 'system', return_value='Darwin'):
            with mock.patch.object(builder.platform, 'machine', return_value='arm64'):
                with mock.patch.object(builder, '_module') as loader:
                    with self.assertRaisesRegex(RuntimeError, '--amd-packed requires Linux'):
                        builder.build(amd_packed=True)
                    loader.assert_not_called()

    def test_amd_cpu_precheck_rejects_intel(self):
        builder = module('build')
        info = 'processor : 0\nvendor_id : GenuineIntel\nflags : '+' '.join(sorted(builder.AMD_FLAGS))+'\n'
        with mock.patch.object(builder.Path, 'read_text', return_value=info):
            with mock.patch.object(builder.os, 'sched_getaffinity', return_value={0}, create=True):
                with self.assertRaisesRegex(RuntimeError, 'requires an AMD CPU'):
                    builder._amd_cpu_check()

    def test_missing_dependency_offline_never_loads_builder(self):
        builder = module('build')
        with tempfile.TemporaryDirectory() as path, mock.patch.object(builder, 'ROOT', Path(path)):
            with mock.patch.object(builder, '_module') as loader:
                with self.assertRaisesRegex(RuntimeError, 'Missing verified SLEEF'):
                    builder._ensure_dependency('sleef', True, 2)
                loader.assert_not_called()

    def test_existing_dependency_hash_is_required(self):
        builder = module('build')
        with tempfile.TemporaryDirectory() as path:
            prefix = Path(path)
            (prefix/'include').mkdir(); (prefix/'lib').mkdir()
            (prefix/'include/sleef.h').write_bytes(b'header')
            (prefix/'lib/libsleef.a').write_bytes(b'library')
            record = {'commit': builder.SLEEF_PIN, 'tag': '3.9.0', 'openmp': False,
                      'header_sha256': hashlib.sha256(b'header').hexdigest(),
                      'library_sha256': hashlib.sha256(b'library').hexdigest()}
            (prefix/'dependency-build.json').write_text(json.dumps(record))
            self.assertTrue(builder._dependency_ready('sleef', prefix))
            (prefix/'lib/libsleef.a').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'artifact hash mismatch'):
                builder._dependency_ready('sleef', prefix)

    def test_linux_routing_keeps_amd_optional(self):
        builder = module('build')
        calls = []
        def load(name):
            calls.append(name)
            return types.SimpleNamespace(fetch=lambda **kw: {'offline': kw['offline']}, build=lambda: {'name': name})
        with tempfile.TemporaryDirectory() as path, mock.patch.object(builder, 'ROOT', Path(path)):
            with mock.patch.object(builder.platform, 'system', return_value='Linux'):
                with mock.patch.object(builder.platform, 'machine', return_value='x86_64'):
                    with mock.patch.object(builder, '_module', side_effect=load):
                        with mock.patch.object(builder, '_ensure_dependency', return_value={'prefix': 'fixture'}):
                            with mock.patch.object(builder, '_library_summary', side_effect=lambda v: v):
                                with mock.patch.object(builder, '_sha256', return_value='fixture'):
                                    result = builder.build(offline=True)
        self.assertEqual(set(result['libraries']), {'x86'})
        self.assertEqual(calls, ['fetch_headers', 'build_x86'])


if __name__ == '__main__':
    unittest.main()
