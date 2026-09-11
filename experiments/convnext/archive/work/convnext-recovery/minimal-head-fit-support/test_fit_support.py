from types import SimpleNamespace

import pytest
import torch

from fit_support import (ReadoutStatistics, complete_scored_frames, select_regularization,
                         select_training_sources, validation_masks)


def make_crop(sid="x", index=0):
    return SimpleNamespace(source_id=sid, start_frame=index, context_start_frame=1,
        valid_scored_samples=1640, teacher_audio=torch.arange(2880, dtype=torch.float32).reshape(1, 1, -1),
        scored_slice=slice(960, 2600))


def test_split_is_deterministic_disjoint_and_deduplicates_audio():
    crops = [make_crop(str(i), j) for i in range(12) for j in range(2)]
    rows = {str(i): {"audio_sha256": f"{i:064x}"} for i in range(12)}
    rows["1"]["audio_sha256"] = rows["0"]["audio_sha256"]
    before = torch.random.get_rng_state().clone()
    a = select_training_sources(crops, rows, {"2"}, {rows["3"]["audio_sha256"]}, fit_count=4, validation_count=3)
    b = select_training_sources(crops, rows, {"2"}, {rows["3"]["audio_sha256"]}, fit_count=4, validation_count=3)
    assert a == b and torch.equal(before, torch.random.get_rng_state())
    selected = [r for split in a["splits"].values() for r in split["sources"]]
    assert len({r["source_id"] for r in selected}) == len({r["audio_sha256"] for r in selected}) == 7
    assert not {"2", "3"} & {r["source_id"] for r in selected}


def test_insufficient_sources_never_silently_shrink():
    with pytest.raises(ValueError, match="Need 5"):
        select_training_sources([make_crop("x")], {"x": {"audio_sha256": "a" * 64}}, set(), set(), fit_count=3, validation_count=2)


def test_full_block_mask_and_override_preserve_geometry():
    crop = make_crop()
    features = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    h, t, receipt = complete_scored_frames(crop, features)
    assert torch.equal(h, features[0, :, 3:5].T)
    assert torch.equal(t.flatten(), crop.teacher_audio.flatten()[1440:2400])
    assert receipt["fit_frames"] == 2 and receipt["full_score_samples"] == 1634
    assert receipt["fit_edge_excluded_samples"] == 674
    override = crop.teacher_audio + 1
    _, changed, proof = complete_scored_frames(crop, features, target_override=override)
    assert torch.equal(changed, t + 1) and proof["fit_target"] == "explicit_override_same_geometry"
    with pytest.raises(ValueError, match="geometry"):
        complete_scored_frames(crop, features, target_override=override[..., :-1])


def test_statistics_match_direct_least_squares_moments():
    generator = torch.Generator().manual_seed(17)
    w = torch.randn(480, 5, generator=generator)
    h = torch.randn(19, 5, generator=generator)
    t = torch.randn(19, 480, generator=generator)
    acc = ReadoutStatistics(w)
    acc.add(h[:7], t[:7], source_id="a")
    acc.add(h[7:], t[7:], source_id="b")
    result = acc.finalize()
    hd, td, wd = h.double(), t.double(), w.double()
    torch.testing.assert_close(result["A"], hd.T @ hd / 19, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(result["B"], hd.T @ (td - hd @ wd.T) / 19, rtol=1e-14, atol=1e-14)
    assert result["samples"] == 19 * 480
    with pytest.raises(ValueError, match="repeated"):
        acc.add(h, t, source_id="a")


def test_regularization_declared_gates_and_none_if_everyone_fails():
    baseline = dict(mse=1., mae=1., quiet_mse=1., valid_samples=100, quiet_samples=10)
    rows = [{**baseline, "lambda": lam, "mse": .8, "mae": 1.02} for lam in (.001, .01, .1, 1.)]
    assert select_regularization(rows, baseline)["selected_lambda"] is None
    rows[1].update(mae=1., quiet_mse=1.001)
    assert select_regularization(rows, baseline)["selected_lambda"] == .01
    rows[2].update(mae=.9, quiet_mse=1., mse=.7)
    assert select_regularization(rows, baseline)["selected_lambda"] == .1
    rows[3].update(mae=.9, quiet_mse=1., mse=.7)
    assert select_regularization(rows, baseline)["selected_lambda"] == 1.


def test_regularization_masks_must_match_and_no_nan():
    baseline = dict(mse=1., mae=1., quiet_mse=1., valid_samples=100, quiet_samples=10)
    rows = [{**baseline, "lambda": lam} for lam in (.001, .01, .1, 1.)]
    rows[0]["quiet_samples"] = 9
    with pytest.raises(ValueError, match="differ"):
        select_regularization(rows, baseline)
    rows[0].update(quiet_samples=10, mse=float("nan"))
    with pytest.raises(ValueError, match="Finite"):
        select_regularization(rows, baseline)


def test_quiet_mask_depends_only_on_teacher_and_original_grid():
    crop = make_crop()
    crop.teacher_audio.zero_()
    teacher, valid, quiet = validation_masks(crop, device="cpu")
    assert int(valid.sum()) == 1634 and torch.equal(valid, quiet)
    assert torch.equal(teacher, crop.teacher_audio)
