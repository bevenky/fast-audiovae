"""CPU correctness tests for the bounded spectral head comparison.

The tests exercise the actual objective and update helpers, using small decoder
blocks. They do not load retained production weights or connect to a GPU host.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config
import run_joint_heads as joint
import run_spectral_heads as spectral


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(813902)


@pytest.fixture
def small_spectral():
    return spectral.SpectralObjective(ReconstructionV2Config(
        fft_sizes=(64, 128), mel_bands=(3, 4)))


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=2, *, source="fixture", target_offset=.0002):
    features, baseline = joint.capture_prehead(model, torch.randn(1, 64, frames))
    target = baseline + target_offset
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :19] = False
    valid[..., -17:] = False
    quiet = valid.clone()
    quiet[..., target.shape[-1] // 3:] = False
    return {"features": features, "target": target, "baseline": baseline,
        "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
        "outside_samples": int((valid & ~quiet).sum()), "source_id": source,
        "start_frame": 0, "language": "fixture", "condition": "speech", "dataset": "unit"}


def pooled(objective, triples):
    counts = tuple(sum(spectral.spectral_counts(v, objective.config)[i] for _, _, v in triples)
                   for i in range(len(objective.config.fft_sizes)))
    return sum(spectral.pooled_spectral_loss(objective.sums(p, t, v), counts, objective.config)
               for p, t, v in triples)


def test_full_valid_pooled_mel_and_gradients_match_reconstruction_v2(small_spectral):
    predictions = [torch.randn(1, 1, n, requires_grad=True) * .1 for n in (320, 451)]
    teachers = [torch.randn_like(p, requires_grad=True) * .1 for p in predictions]
    triples = [(p, t, torch.ones_like(p, dtype=torch.bool)) for p, t in zip(predictions, teachers)]
    result = pooled(small_spectral, triples)
    oracle = ReconstructionV2(small_spectral.config).forward_groups(zip(predictions, teachers))
    torch.testing.assert_close(result, oracle.losses["teacher_mel"], rtol=1e-6, atol=1e-7)
    actual = torch.autograd.grad(result, predictions + teachers, allow_unused=True, retain_graph=True)
    expected = torch.autograd.grad(oracle.losses["teacher_mel"], predictions, retain_graph=True)
    for a, b in zip(actual[:2], expected):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-8)
    assert actual[2:] == (None, None), "Frozen teacher targets must not receive gradients"


def test_spectral_spans_do_not_join_recordings_or_invalid_gaps(small_spectral):
    p = torch.randn(1, 1, 630, requires_grad=True)
    t = torch.randn_like(p)
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 7:170] = True
    valid[..., 201:490] = True
    valid[..., 600:615] = True  # Too short for any configured FFT.
    assert spectral.contiguous_spans(valid) == [(7, 170), (201, 490), (600, 615)]
    result = pooled(small_spectral, [(p, t, valid)])
    oracle = ReconstructionV2(small_spectral.config).forward_groups(
        ((p[..., 7:170], t[..., 7:170]), (p[..., 201:490], t[..., 201:490])))
    torch.testing.assert_close(result, oracle.losses["teacher_mel"], rtol=1e-6, atol=1e-7)
    changed = p.detach().clone()
    changed[~valid] = 10000
    torch.testing.assert_close(pooled(small_spectral, [(changed, t, valid)]), result, rtol=0, atol=0)
    grad = torch.autograd.grad(result, p)[0]
    assert not bool(grad[~valid].any())
    assert not bool(grad[..., 600:615].any())
    assert bool(grad[..., 7:170].any())


def test_short_span_keeps_wave_samples_but_omits_all_mel_resolutions(small_spectral):
    # The unchanged objective requires every span to fit the largest FFT.
    # Both 80-sample and 19-sample tails stay in waveform counts, but no scale
    # gets shorter material than the others or pads it to make a new window.
    p = torch.randn(1, 1, 280, requires_grad=True)
    t = torch.randn_like(p)
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 3:83] = True
    valid[..., 111:251] = True
    valid[..., 260:279] = True
    counts = spectral.spectral_counts(valid, small_spectral.config)
    assert counts == (3 * 5, 4 * 1)
    result = pooled(small_spectral, [(p, t, valid)])
    oracle = ReconstructionV2(small_spectral.config)(p[..., 111:251], t[..., 111:251])
    torch.testing.assert_close(result, oracle.losses["teacher_mel"], rtol=1e-6, atol=1e-7)
    stats = small_spectral.sums(p, t, valid)
    assert tuple(stats["skipped_samples_by_resolution"]) == (99, 99)
    wave = joint.masked_sums(p, t, t, valid, valid)
    assert stats["scored_samples"] == wave["quiet_samples"] == int(valid.sum()) == 239


@pytest.mark.parametrize("shape", [(2, 1, 30), (1, 2, 30), (30,)])
def test_contiguous_spans_rejects_unsupported_waveform_geometry(shape):
    with pytest.raises(ValueError):
        spectral.contiguous_spans(torch.ones(shape, dtype=torch.bool))


def test_contiguous_spans_requires_boolean_mask():
    with pytest.raises(ValueError):
        spectral.contiguous_spans(torch.ones(1, 1, 30))


def test_no_available_spectral_windows_is_finite_differentiable_zero(small_spectral):
    p = torch.randn(1, 1, 90, requires_grad=True)
    t = torch.randn_like(p)
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 10:29] = True
    assert spectral.spectral_counts(valid, small_spectral.config) == (0, 0)
    result = pooled(small_spectral, [(p, t, valid)])
    assert torch.isfinite(result) and result.item() == 0
    result.backward()
    assert torch.equal(p.grad, torch.zeros_like(p))


def test_four_arms_only_train_projection_or_existing_joint_head():
    assert len(spectral.ARMS) == 4 and len(set(spectral.ARMS)) == 4
    model = tiny_decoder()
    initial = deepcopy(model.state_dict())
    states = []
    for arm in spectral.ARMS:
        names = spectral.trainable_names(arm)
        assert names in ({"output.weight"}, joint.HEAD_NAMES)
        named = spectral.select_parameters(model, arm)
        assert {n for n, p in named} == names
        assert {n for n, p in model.named_parameters() if p.requires_grad} == names
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
        states.append(frozenset(names))
    assert states.count(frozenset({"output.weight"})) == 2
    assert states.count(frozenset(joint.HEAD_NAMES)) == 2


def test_region_masks_classify_intact_frames_and_conserve_spectral_sums(small_spectral):
    p, t = torch.randn(1, 1, 550), torch.randn(1, 1, 550)
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 1:513] = True
    quiet = torch.zeros_like(valid)
    quiet[..., 1:193] = True
    quiet[..., 385:513] = True
    regions = small_spectral.regions(p, t, valid, quiet)
    whole = small_spectral.sums(p, t, valid)
    for i, (size, bands) in enumerate(zip(small_spectral.config.fft_sizes, small_spectral.config.mel_bands)):
        coverage = quiet[0, 0, 1:513].unfold(0, size, size//4).sum(-1)
        expected = {"quiet": int((coverage == size).sum())*bands,
                    "active": int((coverage == 0).sum())*bands,
                    "transition": int(((coverage > 0) & (coverage < size)).sum())*bands}
        assert regions["all"]["element_counts"][i] == whole["element_counts"][i]
        for name, count in expected.items():
            assert regions[name]["element_counts"][i] == count
        assert sum(expected.values()) == regions["all"]["element_counts"][i]
        for key in ("linear_sums", "log_sums"):
            assert sum(regions[r][key][i] for r in expected) == pytest.approx(
                regions["all"][key][i], rel=1e-12, abs=1e-12)
            assert regions["all"][key][i] == pytest.approx(float(whole[key][i]), rel=2e-6, abs=1e-8)
    # Quiet and transition categories are measurements only. Altering that mask
    # does not change the all-audio spectral loss or produce cut-up waveforms.
    other = small_spectral.regions(p, t, valid, torch.zeros_like(valid))
    assert other["all"] == regions["all"]
    assert other["quiet"]["mel"] is None
    assert other["transition"]["mel"] is None
    assert other["active"] == other["all"]


def test_region_quiet_mask_cannot_include_padding(small_spectral):
    p = torch.zeros(1, 1, 260)
    valid = torch.ones_like(p, dtype=torch.bool)
    valid[..., 0] = False
    with pytest.raises(ValueError, match="subset"):
        small_spectral.regions(p, p, valid, torch.ones_like(valid))


@pytest.mark.parametrize("arm", spectral.ARMS)
def test_actual_optimizer_step_preserves_all_frozen_weights_and_buffers(arm, small_spectral):
    model = tiny_decoder()
    named = spectral.select_parameters(model, arm)
    bank = [bank_row(model, source="a"), bank_row(model, frames=3, source="b", target_offset=-.0003)]
    before = deepcopy(model.state_dict())
    flags = tuple(m.training for m in model.modules())
    scales, _ = joint.baseline_scales(bank)
    optimizer = spectral.new_optimizer([p for _, p in named], 1e-8)
    update = spectral.train_batch(model, bank, [0, 1], optimizer, scales, arm, 100.,
                                   small_spectral, .5, device="cpu")
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert update["parameter_displacement"] > 0
    assert changed and changed.issubset(spectral.trainable_names(arm))
    assert tuple(m.training for m in model.modules()) == flags
    assert all(p.grad is None for n, p in model.named_parameters() if n not in spectral.trainable_names(arm))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
    assert update["quiet_samples"] == sum(r["quiet_samples"] for r in bank)
    assert update["outside_samples"] == sum(r["outside_samples"] for r in bank)
    if arm.endswith("wave"):
        assert update["spectral_loss"] == 0
    else:
        assert update["spectral_loss"] > 0


def spectral_source(source, mel, count=100):
    # Include empty regions to ensure absent quiet/transition evidence is never
    # fabricated as zero error or used to silently change the overall gate.
    return {"source_id": source, "crops": 1, "regions": {region: {
        "element_counts": [count, count] if region == "all" else [0, 0],
        "mel": mel if region == "all" else None,
        "linear_sums": [mel*count/2]*2 if region == "all" else [0., 0.],
        "log_sums": [mel*count/2]*2 if region == "all" else [0., 0.],
    } for region in spectral.REGIONS}}


def test_source_guard_rejects_12_7_percent_regression_despite_better_global_mean():
    baseline = [spectral_source("quiet-cantonese", .277836, 100),
                spectral_source("long-speech", 1., 10000)]
    candidate = deepcopy(baseline)
    candidate[0]["regions"]["all"]["mel"] *= 1.127
    candidate[1]["regions"]["all"]["mel"] *= .95
    old_mean = sum(s["regions"]["all"]["mel"] for s in baseline)/2
    new_mean = sum(s["regions"]["all"]["mel"] for s in candidate)/2
    assert new_mean < old_mean
    result = spectral.source_spectral_checks(candidate, baseline)
    assert not result["passed"]
    assert result["all_region_failed_sources"] == ["quiet-cantonese"]
    assert result["source_gates"] == 2
    assert spectral.source_spectral_checks(deepcopy(baseline), baseline)["passed"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "count", "coverage"])
def test_source_guard_rejects_comparison_coverage_changes(problem):
    baseline = [spectral_source("a", .5), spectral_source("b", 1.)]
    candidate = deepcopy(baseline)
    if problem == "missing":
        candidate.pop()
    elif problem == "duplicate":
        candidate.append(deepcopy(candidate[0]))
    elif problem == "count":
        candidate[0]["regions"]["all"]["element_counts"][0] += 1
    else:
        candidate[0]["regions"]["quiet"]["mel"] = .1
    with pytest.raises(ValueError):
        spectral.source_spectral_checks(candidate, baseline)


def test_coefficient_calibration_uses_only_fixed_training_audio_and_matches_output_gradients(monkeypatch, small_spectral):
    model = tiny_decoder()
    bank = [bank_row(model, frames=2+(i % 2), source=f"train-{i}",
                     target_offset=.0001*(i+1)) for i in range(8)]
    scales, _ = joint.baseline_scales(bank)
    initial = joint.head_state(model)
    full_state = deepcopy(model.state_dict())
    modes = tuple(m.training for m in model.modules())
    flags = tuple(p.requires_grad for p in model.parameters())
    rng = torch.random.get_rng_state().clone()
    # A fifth-batch sentinel makes accidental selection/validation or extra
    # calibration-data consumption fail immediately, without supplying a model.
    with_sentinel = bank + [{"source_id": "do-not-use", "valid": None}]
    def forbidden(*args, **kwargs):
        raise AssertionError("Output-gradient calibration may not update or execute the model")
    monkeypatch.setattr(spectral, "new_optimizer", forbidden)
    monkeypatch.setattr(spectral, "head_forward", forbidden)
    result = spectral.calibrate_spectral_weight(model, with_sentinel, initial, scales,
        SimpleNamespace(batch_size=2, preservation_weight=100.), small_spectral, "cpu")
    assert [r["source_ids"] for r in result["batches"]] == [
        [f"train-{i}", f"train-{i+1}"] for i in range(0, 8, 2)]
    assert result["optimizer_updates"] == 0
    energy = {"wave": 0., "spectral": 0.}
    for start in range(0, 8, 2):
        rows = bank[start:start+2]
        predictions = [row["baseline"].clone().requires_grad_(True) for row in rows]
        qcount = sum(row["quiet_samples"] for row in rows)
        wave = sum((p-row["target"])[row["quiet"]].square().sum()
                   for p, row in zip(predictions, rows)) / qcount / scales["quiet_mse"]
        pairs = [(p[..., 19:-17], row["target"][..., 19:-17]) for p, row in zip(predictions, rows)]
        mel = ReconstructionV2(small_spectral.config).forward_groups(pairs).losses["teacher_mel"]
        for name, loss in (("wave", wave), ("spectral", mel)):
            grads = torch.autograd.grad(loss, predictions, retain_graph=True)
            energy[name] += sum(float(g.double().square().sum()) for g in grads)
    assert result["spectral_weight"] == pytest.approx((energy["wave"]/energy["spectral"])**.5, rel=2e-6)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in full_state.items())
    assert tuple(m.training for m in model.modules()) == modes
    assert tuple(p.requires_grad for p in model.parameters()) == flags
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_rate_calibration_uses_only_first_training_batch_and_discards_updates(monkeypatch, small_spectral):
    model = tiny_decoder()
    bank = [bank_row(model, source="train-a"), bank_row(model, source="train-b")]
    initial = joint.head_state(model)
    state = deepcopy(model.state_dict())
    scales, _ = joint.baseline_scales(bank)
    rng = torch.random.get_rng_state().clone()
    real_new = spectral.new_optimizer
    real_score = spectral.bank_score
    created = []
    allowed = {id(row) for row in bank}
    def observe_new(*args, **kwargs):
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
        result = real_new(*args, **kwargs)
        assert not result.state
        created.append(result)
        return result
    def observe_score(model, selected, *args, **kwargs):
        assert {id(row) for row in selected} == allowed
        return real_score(model, selected, *args, **kwargs)
    monkeypatch.setattr(spectral, "new_optimizer", observe_new)
    monkeypatch.setattr(spectral, "bank_score", observe_score)
    result = spectral.calibrate_rate(model, bank + [{"source_id": "do-not-use"}], initial, scales,
        SimpleNamespace(batch_size=2, learning_rate=1e-10, preservation_weight=100.),
        small_spectral, .1, "cpu")
    assert result["chosen_learning_rate"] is not None
    assert len(created) == 4*len(result["trials"])
    assert len({id(opt) for opt in created}) == len(created)
    assert result["retained_calibration_optimizer_updates"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_rate_calibration_failure_restores_weights_and_transient_gradients(monkeypatch, small_spectral):
    model = tiny_decoder()
    bank = [bank_row(model)]
    initial = joint.head_state(model)
    state = deepcopy(model.state_dict())
    scales, _ = joint.baseline_scales(bank)
    def broken(*args, **kwargs):
        parameter = dict(model.named_parameters())["output.weight"]
        with torch.no_grad():
            parameter.add_(1.)
        parameter.grad = torch.ones_like(parameter)
        raise FloatingPointError("injected failure")
    monkeypatch.setattr(spectral, "train_batch", broken)
    with pytest.raises(FloatingPointError, match="injected"):
        spectral.calibrate_rate(model, bank, initial, scales,
            SimpleNamespace(batch_size=1, learning_rate=1e-7, preservation_weight=100.),
            small_spectral, .1, "cpu")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("arm", spectral.ARMS)
def test_microbatch_gradients_equal_joint_sample_and_mel_pooled_objective(arm, small_spectral):
    model = tiny_decoder()
    named = spectral.select_parameters(model, arm)
    parameters = [p for _, p in named]
    bank = [bank_row(model, frames=2, source="short"),
            bank_row(model, frames=4, source="long", target_offset=-.0003)]
    scales, _ = joint.baseline_scales(bank)
    # Move away from the preserved baseline so nonquiet preservation has a
    # nonzero gradient too; at exact initialization it cannot test that term.
    with torch.no_grad():
        model.output.weight.mul_(1.002)
    weight = .37 if arm.endswith("spectral") else 0.
    predictions = [joint.head_forward(model, row["features"]) for row in bank]
    qcount = sum(row["quiet_samples"] for row in bank)
    ocount = sum(row["outside_samples"] for row in bank)
    quiet = sum((p-row["target"])[row["quiet"]].square().sum()
                for p, row in zip(predictions, bank))/qcount/scales["quiet_mse"]
    preserve = sum((p-row["baseline"])[row["valid"] & ~row["quiet"]].square().sum()
                   for p, row in zip(predictions, bank))/ocount/scales["outside_mse"]
    mel = ReconstructionV2(small_spectral.config).forward_groups(
        (p[..., 19:-17], row["target"][..., 19:-17]) for p, row in zip(predictions, bank)).losses["teacher_mel"]
    expected = torch.autograd.grad(quiet+100.*preserve+weight*mel, parameters)
    result = spectral.backward_batch(model, bank, scales, small_spectral, device="cpu",
        preservation_weight=100., spectral_weight=weight)
    for parameter, reference in zip(parameters, expected):
        torch.testing.assert_close(parameter.grad, reference, rtol=5e-6, atol=2e-5)
    assert result["loss"] == pytest.approx(float((quiet+100.*preserve+weight*mel).detach()), rel=2e-6)
