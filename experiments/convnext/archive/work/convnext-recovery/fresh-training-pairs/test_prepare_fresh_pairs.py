"""CPU-only crop and teacher-contract tests; no teacher or optimizer is loaded."""
from dataclasses import replace
import io

import pytest
import torch

from audiovae_student.cache import TrainingCrop
from prepare_fresh_pairs import (compare_latents, validate_source_geometry,
                                 validate_teacher_tensor, window)


def crop(*, start=0, context_start=0, context=0, scored=2, valid=1923):
    frames = context + scored
    return TrainingCrop(torch.zeros(1, 64, frames), torch.zeros(1, 1, frames * 1920),
                        torch.zeros(1, 1, frames * 640), "old", "source", start,
                        context_start, context, scored, valid)


def test_partial_frame_storage_padding_is_allowed_but_not_scored():
    c = crop()
    validate_source_geometry(c, 641)
    with pytest.raises(ValueError, match="exceeds authentic source"):
        validate_source_geometry(replace(c, valid_scored_samples=1926), 641)


def test_absolute_context_and_real_tail_are_both_checked():
    c = crop(start=40, context_start=11, context=29, valid=1920)
    validate_source_geometry(c, 41 * 640)
    with pytest.raises(ValueError, match="absolute crop"):
        validate_source_geometry(replace(c, context_start_frame=10), 41 * 640)
    with pytest.raises(ValueError, match="exceeds authentic source"):
        validate_source_geometry(c, 40 * 640 + 639)


@pytest.mark.parametrize("changes", [{"start_frame": -1}, {"context_start_frame": -1},
                                      {"start_frame": True}])
def test_bad_source_coordinates_rejected(changes):
    with pytest.raises(ValueError, match="nonnegative integers"):
        validate_source_geometry(replace(crop(), **changes), 641)


def test_zero_padded_reference_cannot_hide_scored_overrun():
    # Matching all-zero source/reference bytes alone do not prove validity.
    audio = torch.zeros(1, 1, 641)
    c = crop(valid=3840)
    assert window(audio, 0, 1280).equal(c.reference16k)
    with pytest.raises(ValueError, match="exceeds authentic source"):
        validate_source_geometry(c, audio.shape[-1])


def test_window_clones_compact_storage_and_masks_only_real_tail():
    original = torch.arange(10000, dtype=torch.float32).reshape(1, 1, -1)
    selected = window(original, 9997, 5)
    assert selected.flatten().tolist() == [9997, 9998, 9999, 0, 0]
    assert selected.untyped_storage().nbytes() == selected.numel() * 4
    selected[..., 0] = -1
    assert original[..., 9997].item() == 9997
    buffer = io.BytesIO()
    torch.save(selected, buffer)
    assert len(buffer.getvalue()) < 5000
    assert window(original, 10001, 3).equal(torch.zeros(1, 1, 3))


@pytest.mark.parametrize("start,length", [(-1, 3), (0, 0), (True, 3), (0, 2.5)])
def test_invalid_window_geometry_rejected(start, length):
    with pytest.raises(ValueError, match="Window needs"):
        window(torch.zeros(1, 1, 3), start, length)


@pytest.mark.parametrize("value", [torch.zeros(1, 64, 3, dtype=torch.float64),
                                   torch.zeros(2, 64, 3),
                                   torch.full((1, 64, 3), float("nan")),
                                   torch.full((1, 64, 3), float("inf"))])
def test_teacher_contract_rejects_dtype_shape_and_nonfinite(value):
    with pytest.raises(ValueError, match="shape/dtype/finite"):
        validate_teacher_tensor(value, (1, 64, 3), "Encoder")


def test_latent_reference_gate_catches_small_localized_channel_defect():
    reference = torch.zeros(1, 64, 128)
    validate_teacher_tensor(reference, (1, 64, 128), "Encoder")
    assert compare_latents(reference, reference)["passed"]
    changed = reference.clone()
    changed[:, 63, 0] = 1e-4
    result = compare_latents(changed, reference)
    assert result["rms"] < 2e-6
    assert not result["passed"]  # Local max gate remains independent of RMS.
