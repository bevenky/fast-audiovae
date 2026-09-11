"""CPU checks of the experimental projection's mathematical and state contract."""
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.fusion_architecture import FusionArchitectureConfig, FusionStudentDecoder
from audiovae_student.model import StudentConfig, StudentDecoder
from bounded_head import BoundedDecoderView, bounded_waveform, transient_errors, capture_teacher_pre_tanh


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(67021)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bound_interior_bitwise_empty_and_no_input_mutation(dtype):
    raw = torch.tensor([-3., -1., -.7, -0., 0., 1e-7, .9, 1., 1.3], dtype=dtype)
    before = raw.clone()
    bounded = bounded_waveform(raw)
    assert torch.equal(raw, before)
    assert bool((bounded.abs() <= 1).all())
    mask = raw.abs() <= 1
    byte_dtype = torch.int32 if dtype == torch.float32 else torch.int64
    assert torch.equal(bounded[mask].view(byte_dtype), raw[mask].view(byte_dtype))
    assert bounded_waveform(raw[:0]).shape == (0,)
    with pytest.raises(TypeError):
        bounded_waveform(torch.tensor([0, 1]))


def test_mae_mse_projection_guarantee_and_true_raw_mae_gradient_identity():
    raw = (torch.randn(1000, dtype=torch.float64) * 2).requires_grad_()
    target = torch.linspace(-1, 1, raw.numel(), dtype=raw.dtype)
    bounded = bounded_waveform(raw)
    assert bool(((bounded - target).abs() <= (raw - target).abs()).all())
    assert bool(((bounded - target).square() <= (raw - target).square()).all())
    raw_loss = (raw - target).abs().sum()
    decomposed = (bounded - target).abs().sum() + (raw - bounded).abs().sum()
    torch.testing.assert_close(raw_loss, decomposed, rtol=1e-15, atol=1e-12)
    raw_gradient, = torch.autograd.grad(raw_loss, raw, retain_graph=True)
    decomposed_gradient, = torch.autograd.grad(decomposed, raw)
    assert torch.equal(raw_gradient, decomposed_gradient)
    assert bool((raw_gradient[raw > 1] == 1).all())
    assert bool((raw_gradient[raw < -1] == -1).all())


def test_transient_metrics_do_not_cross_invalid_gaps_or_rows():
    prediction = torch.tensor([[0., 1., 4., 9., 999., -999., 12.],
                               [1000., 1001., 1004., 1009., -999., 999., 12.]])
    teacher = torch.zeros_like(prediction)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 0, 0, 1]], dtype=torch.bool)
    result = transient_errors(prediction, teacher, valid)
    assert result["first_difference"]["count"] == 6
    assert result["first_difference"]["absolute_error_sum"] == 18
    assert result["first_difference"]["squared_error_sum"] == 70
    assert result["second_difference"]["count"] == 4
    assert result["second_difference"]["absolute_error_sum"] == 8
    assert result["second_difference"]["squared_error_sum"] == 16
    for length in (0, 1):
        result = transient_errors(prediction[..., :length], teacher[..., :length], valid[..., :length])
        assert result["first_difference"]["count"] == 0
        assert result["second_difference"]["rmse"] is None
    with pytest.raises(ValueError):
        transient_errors(prediction, teacher, valid[0])
    with pytest.raises(ValueError):
        transient_errors(prediction, teacher, valid.float())
    with pytest.raises(ValueError):
        transient_errors(prediction * float("nan"), teacher, valid)


def test_real_fusion_control_forward_stream_and_model_state_preserved():
    base = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=16, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        base(torch.randn(2, 64, 11))
    base.freeze_normalization_statistics().eval()
    model = FusionStudentDecoder.from_decoder(base, FusionArchitectureConfig()).eval()
    with torch.no_grad():
        model.output.weight.mul_(1000)
    original = deepcopy(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    view = BoundedDecoderView(model)
    assert set(vars(view)) == {"decoder"}
    assert view.state_shapes(2) == model.state_shapes(2)
    z = torch.randn(2, 64, 13)
    with torch.no_grad():
        ordinary = model(z)
        assert bool((ordinary.abs() > 1).any())
        assert torch.equal(view(z), bounded_waveform(ordinary))
        original_state = model.initial_state(2)
        state = view.initial_state(2)
        output, empty_state = view.forward_stream(z[..., :0], state)
        assert empty_state is state and output.shape[-1] == 0
        all_raw, all_bounded = [], []
        for start, stop in [(0, 1), (1, 4), (4, 5), (5, 13)]:
            raw_chunk, original_state = model.forward_stream(z[..., start:stop], original_state)
            bounded_chunk, state = view.forward_stream(z[..., start:stop], state)
            assert torch.equal(bounded_chunk, bounded_waveform(raw_chunk))
            all_raw.append(raw_chunk)
            all_bounded.append(bounded_chunk)
            assert torch.equal(state.started, original_state.started)
            assert state.normalization_layout == original_state.normalization_layout
            assert all(torch.equal(a, b) for a, b in zip(state.histories, original_state.histories))
        raw_stream = torch.cat(all_raw, -1)
        bounded_stream = torch.cat(all_bounded, -1)
        assert torch.equal(bounded_stream, bounded_waveform(raw_stream))
        assert bounded_stream.shape[-1] == z.shape[-1] * 1920
        torch.testing.assert_close(bounded_stream, view(z), atol=2e-6, rtol=1e-4)
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_view_passes_through_exact_next_state_object():
    class Stub:
        def forward_stream(self, x, state):
            return x, state
    state = object()
    raw = torch.tensor([1.3])
    bounded, actual = BoundedDecoderView(Stub()).forward_stream(raw, state)
    assert actual is state
    assert bounded.item() == 1.


def _toy_teacher_and_crop():
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Upsample(scale_factor=1920, mode="nearest"),
                                            torch.nn.Conv1d(64, 1, 1), torch.nn.Tanh())

        def forward(self, z):
            return self.model(z)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def forward(self, z):
            return self.decoder(z)

    class Teacher:
        def __init__(self):
            self.model = Model()
            self.device = torch.device("cpu")
            self.provenance = {"source_sha256": SOURCE_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256}
            self.calls = []
            self.output_offset = 0.

        def decode(self, z):
            self.calls.append(z.shape)
            return self.model(z) + self.output_offset

    teacher = Teacher()
    with torch.no_grad():
        teacher.model.decoder.model[-2].weight.zero_()
        teacher.model.decoder.model[-2].bias.fill_(3.)
    teacher.model.train()
    teacher.model.decoder.model[-1].eval()
    for parameter in teacher.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.ones_like(parameter)
    z = torch.randn(1, 64, 4)
    with torch.no_grad():
        target = teacher.decode(z).detach()
    teacher.calls.clear()
    crop = TrainingCrop(z, target, None, "test-key", "test-source", 2, 1, 1, 3, 5700)
    return teacher, crop


def test_teacher_actual_pre_activation_singleton_capture_mask_and_state():
    from audiovae_student.quiet_audio import QuietAudioConfig
    teacher, crop = _toy_teacher_and_crop()
    modes = tuple(m.training for m in teacher.model.modules())
    values = deepcopy(teacher.model.state_dict())
    gradients = tuple(p.grad for p in teacher.model.parameters())
    result = capture_teacher_pre_tanh(teacher, crop, QuietAudioConfig())
    assert teacher.calls == [(1, 64, 4), (1, 64, 4)]
    assert bool((result["pre_tanh"] == 3.).all())
    assert torch.equal(torch.tanh(result["pre_tanh"]), result["post_tanh"])
    assert result["receipt"]["scored_start_sample"] == 1926
    assert result["receipt"]["scored_stop_sample"] == 7620
    assert result["receipt"]["scored_samples"] == 5694
    assert not bool(result["quiet"].any())
    assert tuple(m.training for m in teacher.model.modules()) == modes
    assert all(torch.equal(v, teacher.model.state_dict()[k]) for k, v in values.items())
    assert all(p.grad is g for p, g in zip(teacher.model.parameters(), gradients))
    assert len(teacher.model.decoder.model[-2]._forward_hooks) == 0
    assert not result["pre_tanh"].requires_grad and not result["post_tanh"].requires_grad


def test_teacher_capture_ignores_unscored_target_but_rejects_scored_mismatch_and_removes_hook():
    from audiovae_student.quiet_audio import QuietAudioConfig
    teacher, crop = _toy_teacher_and_crop()
    changed = crop.teacher_audio.clone()
    changed[..., :1926] = 0.
    changed[..., 7620:] = 0.
    capture_teacher_pre_tanh(teacher, replace(crop, teacher_audio=changed), QuietAudioConfig())
    changed[..., 1926] = 0.
    modes = tuple(m.training for m in teacher.model.modules())
    with pytest.raises(RuntimeError, match="canonical") as caught:
        capture_teacher_pre_tanh(teacher, replace(crop, teacher_audio=changed), QuietAudioConfig())
    failure = caught.value.diagnostic
    assert failure["source_id"] == "test-source"
    assert failure["nonzero_delta_count"] == 1
    assert failure["first_mismatch_sample_index_full_crop"] == 1926
    assert failure["last_mismatch_sample_index_full_crop"] == 1926
    assert failure["first_mismatch_offset_from_scored_start"] == 0
    assert failure["mismatches_in_first_480_scored_samples"] == 1
    assert failure["mismatches_after_first_480_scored_samples"] == 0
    assert failure["abs_delta_gt_1e_5_count"] == 1
    assert failure["tolerance_relaxed"] is False
    assert tuple(m.training for m in teacher.model.modules()) == modes
    assert len(teacher.model.decoder.model[-2]._forward_hooks) == 0
    teacher.output_offset = .01
    with pytest.raises(RuntimeError, match="native output"):
        capture_teacher_pre_tanh(teacher, crop, QuietAudioConfig())
    assert len(teacher.model.decoder.model[-2]._forward_hooks) == 0
    with pytest.raises(ValueError, match="singleton"):
        capture_teacher_pre_tanh(teacher, replace(crop, latents=crop.latents.expand(2, -1, -1)), QuietAudioConfig())
