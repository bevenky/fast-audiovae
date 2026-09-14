import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('intel_attribution_' + name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


RUN = module('run')
BUILD = module('build')


class AttributionTests(unittest.TestCase):
    def test_stage_call_counts_include_partial_tiles(self):
        expected = {
            'sp_stage_c256': ((1, 256, 480), 256, (6, 6, 6, 12, 0)),
            'upsample_stage_c128': ((1, 256, 480), 128, (32, 40, 40, 48, 0)),
            'sp_stage_c64': ((1, 64, 1920), 256, (0, 0, 0, 48, 24)),
            'sp_stage_c32': ((1, 32, 3840), 256, (0, 0, 0, 90, 45)),
        }
        for name, (shape, tile, counts) in expected.items():
            with self.subTest(stage=name):
                self.assertEqual(tuple(RUN.expected_counts(name, shape, {'tile_time': tile, 'segments': 1}).values()), counts)

    def test_multisegment_warm_overlap_cannot_masquerade_as_single_thread(self):
        with self.assertRaisesRegex(RuntimeError, 'One stage segment'):
            RUN.expected_counts('sp_stage_c256', (1, 256, 480), {'tile_time': 256, 'segments': 4})

    def test_nested_gemm_not_double_counted(self):
        counts = dict(prepare_ns=10, row_ns=50, nested_gemm_ns=40, snake_ns=20, fx_ns=0)
        result = RUN.pieces(counts, 100)
        self.assertEqual(result['row_remainder_ns'], 10)
        self.assertEqual(result['other_stage_and_dispatch_ns'], 20)
        self.assertEqual(sum(result.values()), 100)

    def test_bad_attribution_is_rejected(self):
        counts = dict(prepare_ns=10, row_ns=50, nested_gemm_ns=51, snake_ns=20, fx_ns=0)
        with self.assertRaisesRegex(RuntimeError, 'Nested GEMM'):
            RUN.pieces(counts, 100)
        counts['nested_gemm_ns'] = 40
        with self.assertRaisesRegex(RuntimeError, 'Overlapping'):
            RUN.pieces(counts, 79)

    def test_explicit_include_must_contain_vendor_header(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'mkl_cblas.h'):
                BUILD.mkl_include(directory)
            (Path(directory) / 'mkl_cblas.h').write_text('/* test fixture */')
            self.assertEqual(BUILD.mkl_include(directory), Path(directory).resolve())


if __name__ == '__main__':
    unittest.main()
