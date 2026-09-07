"""Dependency/build guard tests with no numerical packages or model execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class BuildGuards(unittest.TestCase):
    def test_standalone_imports(self):
        for name in ('build_x86', 'build_amd', 'build_sleef', 'build_aocl'):
            self.assertTrue(callable(module(name).build))

    def test_host_guard(self):
        builder = module('build_x86')
        with mock.patch.object(builder.platform, 'system', return_value='Darwin'):
            with self.assertRaisesRegex(RuntimeError, 'Linux x86_64'):
                builder.build()

    def test_missing_headers_fail_closed(self):
        builder = module('build_x86')
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError, 'ORT 1.29 header'):
                builder.require_headers(Path(path))

    def test_modified_header_fails_closed(self):
        builder = module('build_x86')
        with tempfile.TemporaryDirectory() as path:
            base = Path(path)
            (base/'onnxruntime_c_api.h').write_text('wrong header')
            with self.assertRaisesRegex(RuntimeError, 'ORT 1.29 header'):
                builder.require_headers(base)

    def test_gpu_environment_is_hidden(self):
        environment = module('build_x86').cpu_environment()
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(environment['NVIDIA_VISIBLE_DEVICES'], 'void')
        self.assertEqual(environment['ROCR_VISIBLE_DEVICES'], '-1')
        self.assertEqual(environment['HIP_VISIBLE_DEVICES'], '-1')
        self.assertEqual(environment['BLIS_NUM_THREADS'], '1')

    def test_dependency_jobs_reject_bool(self):
        for name in ('build_sleef', 'build_aocl'):
            builder = module(name)
            with mock.patch.object(builder.common, 'require_linux_x86'):
                with self.assertRaises(ValueError):
                    builder.build(jobs=True)


if __name__ == '__main__':
    unittest.main()
