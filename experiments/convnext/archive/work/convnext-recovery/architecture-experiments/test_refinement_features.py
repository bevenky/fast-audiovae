"""CPU-only checks for causal feature alignment and unchanged decoder replay."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.objective_comparison import state_fingerprint
import refinement_features as features
from test_run_joint_heads import tiny_decoder


@pytest.fixture(autouse=True)
def cpu_determinism():
    torch.set_num_threads(1)
    torch.manual_seed(79874)


def test_last_block_exact_replay_and_only_declared_gradients():
    model = tiny_decoder()
    baseline_state = deepcopy(model.state_dict())
    z = torch.randn(1, 64, 33)
    prefix, baseline = features.capture_last_block_input(model, z)
    assert not prefix.requires_grad
    selected = features.select_last_block_parameters(model)
    names = {n for n, _ in selected}
    assert names == features.HEAD_NAMES | {n for n, _ in model.named_parameters() if n.startswith("blocks.9.")}
    prediction, hidden = features.last_block_forward(model, prefix, return_features=True)
    assert torch.equal(prediction, baseline)
    assert hidden.shape == (1, 8, z.shape[-1] * 4)
    prediction.square().mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in selected)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in names)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline_state.items())
    assert not model.blocks[9]._forward_pre_hooks


def test_teacher_cell_mask_excludes_partial_context_and_tail_without_shifts():
    valid = torch.ones(1, 1, 1920, dtype=torch.bool)
    valid[..., :246] = False
    valid[..., -1] = False
    assert features.complete_teacher_feature_mask(valid).tolist() == [[[False, False, True, True, True, True, True, False]]]
    with pytest.raises(ValueError, match="whole teacher cells"):
        features.complete_teacher_feature_mask(valid[..., :-1])


def test_two_phase_output_is_chronological_and_has_no_future_student_frame():
    adapter = features.AuxiliaryPhaseReadout(1, 2)
    with torch.no_grad():
        adapter.projection.weight[:, 0, 0].copy_(torch.tensor([1., 10., 100., 1000.]))
        adapter.initialized.fill_(True)
    student = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    actual = adapter(student)
    expected = torch.tensor([[[1., 10., 2., 20., 3., 30.], [100., 1000., 200., 2000., 300., 3000.]]])
    assert torch.equal(actual, expected)
    actual[..., :2].sum().backward()
    assert student.grad[0, 0, 0] != 0
    assert torch.equal(student.grad[..., 1:], torch.zeros_like(student.grad[..., 1:]))
    assert not any(p.requires_grad for p in adapter.parameters())


def calibration_rows():
    rows = []
    w = torch.randn(6, 3)
    bias = torch.randn(6)
    for index, count in enumerate((21, 37, 29)):
        x = torch.randn(1, 3, count)
        packed = torch.einsum("oc,bct->bot", w, x) + bias[None, :, None]
        y = packed.reshape(1, 3, 2, count).permute(0, 1, 3, 2).reshape(1, 3, count * 2)
        valid = torch.ones(1, 1, count * 2, dtype=torch.bool)
        valid[..., :3] = False; valid[..., -1] = False
        rows.append({"source_id": str(index), "student_features": x,
                     "teacher_features": y, "feature_valid": valid})
    return rows


def test_normalization_is_sample_pooled_fixed_and_ignores_invalid_values():
    rows = calibration_rows()
    norm = features.fit_feature_normalization(rows)
    x = torch.cat([r["teacher_features"][0, :, r["feature_valid"][0, 0]] for r in rows], 1).double()
    torch.testing.assert_close(norm.mean, x.mean(1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(norm.scale, x.var(1, unbiased=False).sqrt(), rtol=1e-12, atol=1e-12)
    changed = deepcopy(rows)
    for row in changed:
        row["teacher_features"] = torch.where(row["feature_valid"], row["teacher_features"], 12345.)
    other = features.fit_feature_normalization(changed)
    assert torch.equal(norm.mean, other.mean) and torch.equal(norm.scale, other.scale)
    with pytest.raises(ValueError, match="Distinct"):
        features.fit_feature_normalization([rows[0], rows[0]])


def test_ridge_fit_frozen_map_is_accurate_and_student_gradient_nonzero():
    rows = calibration_rows()
    norm = features.fit_feature_normalization(rows)
    adapter = features.AuxiliaryPhaseReadout(3, 3)
    with pytest.raises(RuntimeError, match="Fit the auxiliary"):
        adapter(rows[0]["student_features"])
    report = features.fit_auxiliary_ridge(adapter, rows, norm, ridge_ratio=1e-6)
    assert report["student_updates"] == 0 and report["adapter_frozen_during_student_fit"]
    assert report["adapter_weight_norm"] > 0
    for row in rows:
        error = features.auxiliary_feature_loss(adapter, row["student_features"],
            row["teacher_features"], row["feature_valid"], norm)
        assert float(error) < 1e-9
    x = (rows[0]["student_features"] + .03).requires_grad_(True)
    loss = features.auxiliary_feature_loss(adapter, x, rows[0]["teacher_features"], rows[0]["feature_valid"], norm)
    loss.backward()
    assert x.grad.norm() > 0 and not any(p.grad is not None for p in adapter.parameters())
    assert torch.equal(x.grad[..., 0], torch.zeros_like(x.grad[..., 0]))


def test_feature_loss_batch_denominator_is_pooled_and_target_detached():
    rows = calibration_rows()
    norm = features.fit_feature_normalization(rows)
    adapter = features.AuxiliaryPhaseReadout(3, 3)
    features.fit_auxiliary_ridge(adapter, rows, norm)
    total = sum(int(r["feature_valid"].sum()) for r in rows)
    loss = sum(features.auxiliary_feature_loss(adapter, r["student_features"] + .1,
        r["teacher_features"], r["feature_valid"], norm, total_valid_cells=total) for r in rows)
    expected_sum = sum(features.auxiliary_feature_loss(adapter, r["student_features"] + .1,
        r["teacher_features"], r["feature_valid"], norm) * int(r["feature_valid"].sum()) for r in rows)
    torch.testing.assert_close(loss, expected_sum / total)
    target = rows[0]["teacher_features"].clone().requires_grad_(True)
    x = rows[0]["student_features"].clone().requires_grad_(True)
    features.auxiliary_feature_loss(adapter, x, target, rows[0]["feature_valid"], norm).backward()
    assert target.grad is None and x.grad is not None


def teacher_fixture(tmp_path):
    from test_full_source_teacher_head import fixture
    from full_source_teacher_head import authenticated_source, crop_window, full_source_cache_key
    teacher, crop, data, _, path = fixture(tmp_path)

    class CausalDecoderBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Upsample(scale_factor=8, mode="nearest"),
                                             torch.nn.Conv1d(64, 1024, 1))
        def forward(self, value):
            return self.block(value)

    teacher.model.decoder.model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
        CausalDecoderBlock(), torch.nn.Conv1d(1024, 1, 1), torch.nn.Upsample(scale_factor=240, mode="nearest"),
        torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(),
        torch.nn.Conv1d(1, 1, 1), torch.nn.Tanh())
    teacher.model.eval().requires_grad_(False)
    teacher.provenance["config"] = {"decoder_rates": [8, 6, 5, 2, 2, 2], "depthwise": True,
        "use_noise_block": False, "cond_type": "scale_bias", "cond_out_layer": False}
    audio, inventory = authenticated_source(data, crop)
    before = state_fingerprint(teacher.model.state_dict())
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        z = teacher.model.encode(audio, 16000)
        target = teacher.decode(z)
    crop = replace(crop, teacher_audio=crop_window(target, 1920, 7680),
                   cache_key=full_source_cache_key(inventory, before, z, target))
    teacher.calls.clear()
    return teacher, crop, data, before


def test_full_source_feature_capture_authenticates_then_slices_exact_clock(tmp_path):
    teacher, crop, data, before = teacher_fixture(tmp_path)
    result = features.capture_full_source_teacher_features(teacher, crop, data,
        expected_teacher_state_sha256=before)
    assert teacher.calls == [(1, 64, 5), (1, 64, 5)]
    assert result["teacher_features"].shape == (1, 1024, 32)
    assert result["receipt"]["feature_start_index"] == 8
    assert result["receipt"]["valid_teacher_cells"] == 23
    assert not bool(result["feature_valid"][..., :9].any())
    assert not teacher.model.decoder.model[2]._forward_hooks
    assert state_fingerprint(teacher.model.state_dict()) == before


def test_full_source_feature_capture_rejects_cache_and_target_mutation(tmp_path):
    teacher, crop, data, before = teacher_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="cache key"):
        features.capture_full_source_teacher_features(teacher, replace(crop, cache_key="0"*64), data,
            expected_teacher_state_sha256=before)
    bad = crop.teacher_audio.clone(); bad[..., 0] += .1
    with pytest.raises(RuntimeError, match="context/padding"):
        features.capture_full_source_teacher_features(teacher, replace(crop, teacher_audio=bad), data,
            expected_teacher_state_sha256=before)
    assert not teacher.model.decoder.model[2]._forward_hooks
    assert state_fingerprint(teacher.model.state_dict()) == before
