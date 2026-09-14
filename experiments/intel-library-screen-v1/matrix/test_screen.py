import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location('matrix_screen', Path(__file__).with_name('run.py'))
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


def test_bitwise_gate_rejects_signed_zero():
    result = SCREEN.comparison(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32))
    assert not result['bitwise_equal']
    assert result['different_values'] == 1


def test_bitwise_gate_rejects_nonfinite():
    result = SCREEN.comparison(np.array([np.inf], dtype=np.float32), np.array([np.inf], dtype=np.float32))
    assert not result['bitwise_equal']


def test_offset_transpose_preserves_signed_products():
    # Integer identity used by the candidate to move the U8 offset from the
    # weights to the input. Includes both extremes and cancellation.
    weights = np.array([[-127, 127, 0, 3], [127, 126, -126, -127]], dtype=np.int32)
    inputs = np.array([[-127, 3], [127, -3], [126, 0], [-126, 127]], dtype=np.int32)
    original = (weights + 128) @ inputs - 128 * inputs.sum(axis=0)
    transposed = ((inputs.T + 128) @ weights.T - 128 * weights.sum(axis=1)).T
    np.testing.assert_array_equal(original, transposed)


def test_capture_shape_mismatch_rejected():
    assert SCREEN.comparison(np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32)) == {
        'bitwise_equal': False, 'shape_mismatch': True}
