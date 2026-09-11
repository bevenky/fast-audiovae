from types import SimpleNamespace

import pytest
import torch

from diagnose_features import (
    collect_features, complete_block_indices, exact_output_change, fit_readout, identity_partition,
    matrix_spectrum, mean_feature_terms, probe_statistics, select_probe_crops,
    unique_peak_events,
)


def crop(source="a", start=0, context=0, valid=19200):
    return SimpleNamespace(source_id=source, start_frame=start + context,
        context_start_frame=start, context_frames=context, valid_scored_samples=valid,
        scored_slice=slice(context * 1920, context * 1920 + valid))


def test_complete_blocks_exclude_context_prefix_and_padded_partial_tail():
    c = crop(start=10, context=29, valid=1920 + 470)
    ids = complete_block_indices(c)
    assert ids.tolist() == [117, 118, 119]
    c = crop(valid=1920 + 470)
    assert complete_block_indices(c).tolist() == [0, 1, 2, 3]


def test_block_sampling_deterministic_and_without_replacement():
    c = crop(valid=640 * 480)
    ids = complete_block_indices(c, maximum=64)
    assert len(ids) == len(set(ids.tolist())) == 64
    assert torch.equal(ids, complete_block_indices(c, maximum=64))
    assert set((ids % 4).tolist()) == {0, 1, 2, 3}


def test_identity_grouping_keeps_transitive_speaker_session_parent_links():
    crops = [crop(str(i)) for i in range(80)]
    rows = {str(i): {"speaker_id": "speaker" + str(i), "session_id": "session" + str(i),
                    "parent_recording_id": "parent" + str(i), "audio_sha256": "hash" + str(i)} for i in range(80)}
    rows["1"]["speaker_id"] = rows["0"]["speaker_id"]
    rows["2"]["session_id"] = rows["1"]["session_id"]
    rows["3"]["audio_sha256"] = rows["2"]["audio_sha256"]
    rows["4"]["parent_recording_id"] = rows["3"]["parent_recording_id"]
    split, report = identity_partition(crops, rows)
    assert len({split[str(i)] for i in range(5)}) == 1
    assert report["group_sizes_max"] == 5
    assert report["connected_groups"] == 76
    assert set(split.values()) == {"fit", "tune"}


def test_training_source_cannot_be_heldout():
    with pytest.raises(ValueError, match="overlaps"):
        select_probe_crops([crop("same")], {}, [crop("same")])


def test_exact_symmetric_head_feature_change_has_cross_cancellation():
    torch.manual_seed(17)
    old_h, new_h = torch.randn(10, 7), torch.randn(10, 7)
    old_w, new_w = torch.randn(11, 7), torch.randn(11, 7)
    head, upstream, actual = exact_output_change(old_h, new_h, old_w, new_w)
    torch.testing.assert_close(head + upstream, actual, atol=1e-12, rtol=1e-12)
    h, u, d = exact_output_change(old_h, old_h, old_w, new_w)
    assert not torch.count_nonzero(u)
    torch.testing.assert_close(h, d)


def test_head_matrix_can_have_full_row_rank():
    w = torch.cat([torch.eye(5), torch.zeros(5, 3)], dim=1)
    result = matrix_spectrum(w)
    assert result["shape"] == [5, 8]
    assert result["rank_float64_threshold"] == result["rank_float32_threshold"] == 5
    assert result["condition_number"] == 1


def test_quiet_feature_mean_terms_close_exactly():
    torch.manual_seed(2)
    h = torch.randn(40, 12) * .0001 + .00005
    w = torch.randn(480, 12) * .1
    p = (h.double() @ w.double().T).reshape(1, 1, -1)
    t = torch.randn_like(p) * 1e-5
    result = mean_feature_terms(h, p, t, w)
    assert result["quiet_cycles"] == 10
    assert result["linear_reconstruction_max_error"] < 1e-16
    assert abs(result["cross_term"]) < 1e-20
    assert result["observed_residual_power"] == pytest.approx(
        result["fixed_mean_residual_power"] + result["varying_residual_power"], abs=1e-18)


def test_unique_peaks_merge_overlapping_source_timeline():
    a, b = crop("same", valid=1920 * 3), crop("same", start=1, valid=1920 * 2)
    p1, p2 = torch.zeros(1,1,5760), torch.zeros(1,1,3840)
    p1[...,2000:2002] = 1.1; p2[...,80:82] = 1.2
    result = unique_peak_events([{"crop":a,"prediction":p1,"target":torch.zeros_like(p1)},
                                {"crop":b,"prediction":p2,"target":torch.zeros_like(p2)}])
    assert result["sources"] == 1
    assert result["unique_samples"] == 2
    assert len(result["events"]) == 1
    assert result["events"][0]["start_sample"] == 2000
    assert result["events"][0]["student_peak_signed"] == pytest.approx(1.2)


@pytest.mark.parametrize("intercept", [False, True])
def test_ridge_recovers_external_readout_without_changing_inputs(intercept):
    torch.manual_seed(4)
    x = torch.randn(512, 8, dtype=torch.float64) * torch.linspace(.1,2,8)
    w = torch.randn(8, 11, dtype=torch.float64)
    b = torch.randn(11, dtype=torch.float64) if intercept else torch.zeros(11,dtype=torch.float64)
    y = x @ w + b
    items = [{"crop":crop("train"), "features":x, "target_blocks":y}]
    stats = probe_statistics(items,{"train":"fit"},"fit","cpu")
    frozen = {k:v.clone() for k,v in stats.items() if isinstance(v,torch.Tensor)}
    coefficient,bias,details = fit_readout(stats,ridge=1e-8,intercept=intercept)
    torch.testing.assert_close(x @ coefficient + bias,y,atol=1e-6,rtol=1e-6)
    for key,value in frozen.items():
        assert torch.equal(stats[key],value)
    assert details["external_probe_only"]


def test_affine_probe_can_identify_missing_intercept_without_changing_dimensions():
    torch.manual_seed(44)
    x = torch.randn(1000,6,dtype=torch.float64)
    y = x @ torch.randn(6,3,dtype=torch.float64) + 4
    stats = probe_statistics([{"crop":crop(),"features":x,"target_blocks":y}],{"a":"fit"},"fit","cpu")
    w0,b0,_ = fit_readout(stats,ridge=1e-8,intercept=False)
    w1,b1,_ = fit_readout(stats,ridge=1e-8,intercept=True)
    assert ((x@w0+b0-y)**2).mean() > 10
    assert ((x@w1+b1-y)**2).mean() < 1e-12
    assert w0.shape == w1.shape == (6,3)


def test_actual_features_reproduce_head_and_preserve_model_and_variable_lengths():
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.model import StudentConfig, StudentDecoder
    torch.manual_seed(88)
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
                                        head_channels=12, dilations=(1,)))
    model.train()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    crops = [TrainingCrop(latents=torch.randn(1,64,n),teacher_audio=torch.zeros(1,1,n*1920),
        reference16k=None,cache_key="key",source_id=str(n),start_frame=0,context_start_frame=0,
        context_frames=0,scored_frames=n,valid_scored_samples=n*1920-3) for n in (8,10)]
    engine = SimpleNamespace(model=model,device="cpu",step=0)
    full = collect_features(engine,crops,full=True)
    for item,c in zip(full,crops):
        projected = (item["features"] @ model.output.weight.detach()[:,:,0].T).reshape(1,1,-1)
        torch.testing.assert_close(projected,item["prediction"],rtol=1e-5,atol=1e-8)
        assert item["prediction"].shape == c.teacher_audio.shape
    assert model.training
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    subset = collect_features(engine,crops,full=False,max_blocks=12)
    assert all(len(item["features"])==12 for item in subset)
    assert all(int(item["block_indices"].max()) < c.scored_frames*4-1 for item,c in zip(subset,crops))
