"""Small CPU tests of paired state, unique exposure, and interruption handling."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from audiovae_student.balance_comparison import BalanceComparisonConfig, run_balance_comparison
from audiovae_student.distillation_training import DistillationEngine, DistillationTrainingConfig
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.teacher import CHECKPOINT_SHA256
from audiovae_student.training import _rng_state
from test_representative_pilot import fixture

torch.set_num_threads(1)


def parent_fixture():
    torch.manual_seed(345)
    data = fixture()
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
        dilations=(1,), layer_scale_init=.1, normalization_mode="masked_batch_norm"))
    engine = DistillationEngine(model, config=DistillationTrainingConfig(optimizer="adamw", total_steps=3,
        warmup_steps=0, freeze_normalization_step=1, reconstruction_waveform_share=.75, quiet_window_gate=True),
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=AudioDiscriminators(DiscriminatorConfig(periods=(2,),
            mpd_channels=(2,2,4,4,4), fft_sizes=(64,), mrd_channels=2)))
    # Parent-only fixture data is never part of comparison exposure.
    for _ in range(3):
        engine.train_step(data[5])
    parent = {"engine": deepcopy(engine.state_dict()), "rng": _rng_state(),
              "identity": {"data": {"teacher_checkpoint_sha256": CHECKPOINT_SHA256}}}
    return parent, data


def run(path, parent, data, *, resume=None, max_updates=None):
    return run_balance_comparison(*data, path, parent=parent,
        parent_checkpoint_sha256="b" * 64, data_identity={"teacher_checkpoint_sha256": CHECKPOINT_SHA256,
            "source_corpus": data[0].identity, "teacher_batch_qualification": {"fixture": True}},
        heldout_metadata={r.source_id: {"dataset": r.dataset, "language": r.language, "condition": "speech"} for r in data[6]},
        config=BalanceComparisonConfig(total_steps=4, batch_size=1, expected_parent_step=3,
            evaluation_interval=2, checkpoint_interval=2, min_free_bytes=0), device="cpu",
        resume_from=resume, max_updates=max_updates, heldout_corpus=data[0])


def test_paired_parent_state_and_shared_unique_targets(tmp_path):
    parent, data = parent_fixture()
    fingerprint = state_fingerprint(parent)
    result = run(tmp_path / "whole", parent, data)
    assert result["step"] == 4 and result["consumed_windows_per_arm"] == 4
    assert result["paired_start_verified"] and not result["perceptual_training_started"]
    assert state_fingerprint(parent) == fingerprint
    start = json.loads((tmp_path / "whole/paired-start.json").read_text())
    assert start["control"] == start["mel-cap"]
    checkpoint = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    for arm, version in (("control", 1), ("mel-cap", 2)):
        state = checkpoint["engines"][arm]
        assert state["balancer"]["format_version"] == version
        assert state["step"] == 4 and state["balancer"]["updates"] == 7
        assert state_fingerprint(state["discriminators"]) == state_fingerprint(parent["engine"]["discriminators"])
        for norm in ("stem_norm", "affine"):
            assert torch.equal(state["model"][norm + ".running_mean"], parent["engine"]["model"][norm + ".running_mean"])
    metrics = [json.loads(line) for line in (tmp_path / "whole/mel-cap-metrics.jsonl").read_text().splitlines()]
    assert all(m["actual_teacher_mel_share"] <= .25 + 1e-9 for m in metrics)
    assert len((tmp_path / "whole/exposure.jsonl").read_text().splitlines()) == 4


def test_clean_resume_exactly_matches_uninterrupted_pair(tmp_path):
    parent, data = parent_fixture()
    run(tmp_path / "whole", parent, data)
    run(tmp_path / "split", parent, data, max_updates=2)
    run(tmp_path / "split", parent, data, resume=tmp_path / "split/latest.pt")
    a = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    assert a["sampler"] == b["sampler"]
    assert state_fingerprint(a["engines"]) == state_fingerprint(b["engines"])


@pytest.mark.parametrize("damage", ["inflight", "journal", "arm_step"])
def test_uncertain_partial_pair_or_exposure_cannot_resume(tmp_path, damage):
    parent, data = parent_fixture()
    run(tmp_path / "run", parent, data, max_updates=2)
    if damage == "inflight":
        (tmp_path / "run/inflight.json").write_text("{}")
    elif damage == "journal":
        with (tmp_path / "run/exposure.jsonl").open("a") as f:
            f.write('{}\n')
    else:
        state = torch.load(tmp_path / "run/latest.pt", weights_only=True)
        state["engines"]["mel-cap"]["step"] += 1
        torch.save(state, tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="exposure|steps|in-flight"):
        run(tmp_path / "run", parent, data, resume=tmp_path / "run/latest.pt")


def test_duplicate_or_heldout_overlap_rejected_before_creating_run(tmp_path):
    parent, data = parent_fixture()
    bad = list(data)
    bad[1] = [data[1][0]] * 4
    with pytest.raises(ValueError):
        run(tmp_path / "duplicate", parent, bad)
    assert not (tmp_path / "duplicate").exists()
    bad = list(data)
    bad[6] = [replace(data[6][0], speaker_id=data[2][0].speaker_id)]
    with pytest.raises(ValueError):
        run(tmp_path / "heldout", parent, bad)
    assert not (tmp_path / "heldout").exists()


def test_changed_heldout_tensor_rejected_before_training(tmp_path):
    parent, data = parent_fixture()
    data[5][0].latents[..., 0] += .001
    with pytest.raises(ValueError, match="raw latent"):
        run(tmp_path / "changed", parent, data)
    assert not (tmp_path / "changed").exists()


def test_real_partial_pair_failure_preserves_reservation(tmp_path, monkeypatch):
    parent, data = parent_fixture()
    original = DistillationEngine.train_step
    def fail_candidate(self, crops):
        if self.balancer.mel_cap is not None:
            raise RuntimeError("injected between paired updates")
        return original(self, crops)
    with monkeypatch.context() as patch:
        patch.setattr(DistillationEngine, "train_step", fail_candidate)
        with pytest.raises(RuntimeError, match="injected"):
            run(tmp_path / "partial", parent, data)
    assert (tmp_path / "partial/inflight.json").exists()
    state = json.loads((tmp_path / "partial/status.json").read_text())
    assert state["arm_steps"] == {"control": 1, "mel-cap": 0}
    with pytest.raises(ValueError, match="in-flight"):
        run(tmp_path / "partial", parent, data, resume=tmp_path / "partial/latest.pt")


def test_component_mel_share_uses_actual_phase_rescaled_components():
    from audiovae_student.balance_comparison import component_mel_share
    assert component_mel_share({'teacher_mel/scaled_norm': 2., 'teacher_waveform/scaled_norm': 6.}) == .25
    assert component_mel_share({'teacher_mel/scaled_norm': 2., 'teacher_waveform/scaled_norm': 6.,
        'teacher_phase/base_scale': .98, 'teacher_phase/scaled_norm': .12,
        'teacher_phase/base_scaled_norm': 5.88}) == pytest.approx(1.96 / 7.96)


def test_quiet_phase_pair_preserves_parent_and_resumes_exactly(tmp_path):
    parent, data = parent_fixture()
    before = state_fingerprint(parent)
    def phase_run(path, resume=None, bound=None):
        return run_balance_comparison(*data, path, parent=parent,
            parent_checkpoint_sha256='b'*64, data_identity={
                'teacher_checkpoint_sha256': CHECKPOINT_SHA256, 'source_corpus': data[0].identity,
                'teacher_batch_qualification': {'fixture': True}}, heldout_metadata={},
            config=BalanceComparisonConfig(total_steps=4,batch_size=1,expected_parent_step=3,
                evaluation_interval=2,checkpoint_interval=2,min_free_bytes=0,candidate_kind='quiet_phase'),
            phase_policy={'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'teacher_state_unchanged':True,
                'parent_checkpoint_sha256':'b'*64,'evidence_sha256':'c'*64}, device='cpu',
            resume_from=resume,max_updates=bound,heldout_corpus=data[0])
    phase_run(tmp_path/'whole')
    phase_run(tmp_path/'split',bound=2)
    phase_run(tmp_path/'split',resume=tmp_path/'split/latest.pt')
    a=torch.load(tmp_path/'whole/latest.pt',weights_only=True)
    b=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    assert state_fingerprint(a['engines'])==state_fingerprint(b['engines'])
    assert a['paired_start']['control']==a['paired_start']['quiet-phase']
    assert a['engines']['quiet-phase']['quiet_phase']['config']['gradient_share']==.02
    assert a['engines']['control']['quiet_phase']['config']['gradient_share']==0
    assert state_fingerprint(parent)==before


def test_midpoint_required_and_active_phase_not_silently_dropped():
    from audiovae_student.balance_comparison import new_arm
    with pytest.raises(ValueError,match='midpoint'):
        BalanceComparisonConfig(total_steps=250,evaluation_interval=250)
    parent,_=parent_fixture()
    parent['engine']['quiet_phase']={'activation': {'any':'active'}}
    with pytest.raises(ValueError,match='objective transition'):
        new_arm(parent['engine'],BalanceComparisonConfig(), 'cpu', capped=False)
