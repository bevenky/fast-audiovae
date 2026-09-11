"""Small synthetic CPU integration tests; no real training or quality claim."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from audiovae_student.cache import TrainingCrop
from audiovae_student.distillation_training import DistillationEngine, evaluate_crops
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.perceptual_readiness import (assess_perceptual_readiness,
    collect_readiness_evidence, evaluation_evidence, _seal)
from audiovae_student.perceptual_trial_engine import PerceptualTrialConfig, PerceptualTrialEngine
from audiovae_student.teacher import CHECKPOINT_SHA256
from test_perceptual_readiness import fixture as readiness_fixture

torch.set_num_threads(1)


def fixture():
    data = readiness_fixture()
    base, crops, current, previous, evidence, *_ = data
    # Use a small synthetic generator step so the positive continuation control
    # stays inside its real quiet rollback bound. Production defaults stay intact.
    base.config = replace(base.config, learning_rate=2e-5, final_learning_rate=2e-5)
    for optimizer in base.optimizer.optimizers.values():
        for group in optimizer.param_groups:
            group['lr'] = 2e-5
    evidence = collect_readiness_evidence(base, crops, corpus=data[5], panel_rows=data[6], sample_counts=data[7])
    report = assess_perceptual_readiness(base, crops, current=current, previous=previous, evidence=evidence)
    assert report['ready_for_bounded_trial']
    policy = {'parent_checkpoint_sha256': 'b' * 64, 'data_plan_sha256': 'c' * 64,
              'teacher_checkpoint_sha256': CHECKPOINT_SHA256, 'readiness_sha256': report['sha256']}
    engine = new_engine(base)
    saved = deepcopy(base.state_dict())
    engine.load_state_dict(saved)
    return engine, base, crops, current, previous, evidence, report, policy, saved


def new_engine(base, *, config=None, trial_config=None):
    return PerceptualTrialEngine(deepcopy(base.model), config=config or base.config,
        loss_config=base.criterion.config, discriminators=deepcopy(base.discriminators),
        trial_config=trial_config or PerceptualTrialConfig(max_updates=4, review_update=2))


def enable(data, report=None):
    engine, base, crops, current, previous, evidence, original_report, policy, saved = data
    return engine.enable_trial(report or original_report, policy, crops=crops, current=current,
                               previous=previous, evidence=evidence)


def training_crop(base, seed=5):
    rng = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 64, 5, generator=rng)
    with torch.no_grad():
        teacher = base.model(latents).detach() * 1.5
    return TrainingCrop(latents, teacher, None, 'd' * 64, f'training-{seed}', 0, 0, 0, 5, 9600)


def current_evidence(engine, crops):
    return evaluation_evidence(crops, evaluate_crops(engine, crops, include_signal_checks=True),
        run_identity={'fixture': 'bounded-trial'}, model_config=engine.model.config.to_dict())


def test_actual_readiness_preserves_generator_history_and_explicitly_starts_fresh_discriminator():
    data = fixture()
    engine, base, *_ = data
    initial = deepcopy(engine.state_dict())
    rng = torch.get_rng_state().clone()
    activation = enable(data)
    assert not activation['readiness']['final_acceptance']['passed']
    assert 'passed' not in engine.gate
    assert engine.step == engine.perceptual_start == 0
    for key in ('model', 'optimizer', 'crop_rng', 'discriminators', 'balancer'):
        assert state_fingerprint(engine.state_dict()[key]) == state_fingerprint(initial[key])
    assert torch.equal(rng, torch.get_rng_state())
    assert engine.config.total_steps == 4 and engine.config.freeze_normalization_step == 0
    group = engine.discriminator_optimizer.param_groups[0]
    assert group['lr'] == .0002 and group['betas'] == (.8, .9) and group['weight_decay'] == 0
    assert not engine.discriminator_optimizer.state
    assert state_fingerprint(data[-1]) == state_fingerprint(base.state_dict())


def test_training_has_real_adversarial_and_feature_gradients_then_stops_at_review():
    data = fixture()
    engine, base, *_ = data
    enable(data)
    teacher = training_crop(base)
    target_before = teacher.teacher_audio.clone()
    before_d = state_fingerprint(engine.discriminators.state_dict())
    first = engine.train_step([teacher])
    assert first['adversarial_examples'] == 1 and first['discriminator'] > 0
    assert first['feature_matching/raw_norm'] > 0 and first['adversarial/raw_norm'] > 0
    assert first['teacher_waveform/target_share'] >= .3
    assert state_fingerprint(engine.discriminators.state_dict()) != before_d
    assert teacher.teacher_audio.grad is None and teacher.latents.grad is None
    torch.testing.assert_close(teacher.teacher_audio, target_before, rtol=0, atol=0)
    engine.train_step([training_crop(base, 6)])
    before = state_fingerprint(engine.state_dict())
    with pytest.raises(ValueError, match='review or completion'):
        engine.train_step([training_crop(base, 7)])
    assert state_fingerprint(engine.state_dict()) == before


def test_trial_checkpoint_resume_exactly_matches_next_real_update():
    data = fixture()
    engine, base, *_ = data
    enable(data)
    engine.train_step([training_crop(base)])
    saved = deepcopy(engine.state_dict())
    restored = new_engine(base, config=engine.config)
    restored.load_state_dict(saved)
    assert state_fingerprint(restored.state_dict()) == state_fingerprint(saved)
    item = training_crop(base, 6)
    assert engine.train_step([item]) == restored.train_step([item])
    assert state_fingerprint(engine.state_dict()) == state_fingerprint(restored.state_dict())
    plain = DistillationEngine(deepcopy(base.model), config=engine.config,
        loss_config=base.criterion.config, discriminators=deepcopy(base.discriminators))
    with pytest.raises(ValueError, match='perceptual gate'):
        plain.load_state_dict(saved)


def test_initial_trial_resume_keeps_readiness_bound_parent_weights_and_optimizer():
    data = fixture()
    engine, base, *_ = data
    enable(data)
    saved = deepcopy(engine.state_dict())
    restored = new_engine(base, config=engine.config)
    restored.load_state_dict(saved)
    assert state_fingerprint(restored.state_dict()) == state_fingerprint(saved)
    altered = deepcopy(saved)
    altered['model']['output.weight'][0, 0] += .01
    altered['perceptual_trial']['base_state_sha256'] = state_fingerprint(
        {k: v for k, v in altered.items() if k != 'perceptual_trial'})
    with pytest.raises(ValueError, match='initial weights'):
        new_engine(base, config=engine.config).load_state_dict(altered)


def test_explicit_review_unlocks_only_remaining_budget_and_remains_strictly_resumable():
    data = fixture()
    engine, base, crops, *_ = data
    enable(data)
    engine.train_step([training_crop(base)])
    engine.train_step([training_crop(base, 6)])
    review = engine.review_trial(current_evidence(engine, crops), crops=crops)
    assert review['continuation_authorized'], review['checks']
    saved = deepcopy(engine.state_dict())
    restored = new_engine(base, config=engine.config)
    restored.load_state_dict(saved)
    for seed in (7, 8):
        item = training_crop(base, seed)
        assert engine.train_step([item]) == restored.train_step([item])
    assert state_fingerprint(engine.state_dict()) == state_fingerprint(restored.state_dict())
    with pytest.raises(ValueError, match='completion bound'):
        engine.train_step([training_crop(base, 9)])


def test_fabricated_or_stale_readiness_cannot_start_a_trial():
    data = fixture()
    engine = data[0]
    with pytest.raises(ValueError):
        enable(data, {'passed': True, 'ready_for_bounded_trial': True})
    invented = deepcopy(data[6])
    invented.pop('sha256')
    invented['current']['waveform_error'] = 0
    with pytest.raises(ValueError, match='Actual current readiness'):
        enable(data, _seal(invented))
    with torch.no_grad():
        next(engine.model.parameters()).add_(.001)
    with pytest.raises(ValueError, match='parent changed'):
        enable(data)


@pytest.mark.parametrize('damage', ['d_settings', 'shares', 'fixed_stats', 'gate', 'budget', 'config', 'balancer_config'])
def test_trial_resume_rejects_changed_state_contract(damage):
    data = fixture()
    engine, base, *_ = data
    enable(data)
    engine.train_step([training_crop(base)])
    state = deepcopy(engine.state_dict())
    if damage == 'd_settings':
        state['discriminator_optimizer']['param_groups'][0]['betas'] = (.9, .999)
    elif damage == 'shares':
        state['balancer']['weights']['adversarial'] = .2
    elif damage == 'fixed_stats':
        state['model']['stem_norm.running_mean'][0] += .01
    elif damage == 'gate':
        state['gate']['passed'] = True
    elif damage == 'budget':
        state['step'] = 3
    elif damage == 'config':
        state['config']['gradient_clip'] = .5
    else:
        state['balancer']['config']['ema_decay'] = .9
    # Recompute the container checksum to test the deeper contract too.
    state['perceptual_trial']['base_state_sha256'] = state_fingerprint(
        {k: v for k, v in state.items() if k != 'perceptual_trial'})
    with pytest.raises(ValueError):
        new_engine(base, config=engine.config).load_state_dict(state)


def test_changed_panel_or_failing_review_cannot_unlock_training():
    data = fixture()
    engine, base, crops, *_ = data
    enable(data)
    engine.train_step([training_crop(base)])
    with pytest.raises(ValueError, match='exact declared safety'):
        engine.review_trial(current_evidence(engine, crops), crops=crops)
    engine.train_step([training_crop(base, 6)])
    evaluation = current_evidence(engine, crops)
    rows = deepcopy(evaluation['rows'])
    for row in rows:
        row['teacher_waveform'] = 100.
    failed = evaluation_evidence(crops, rows, run_identity=evaluation['run_identity'], model_config=engine.model.config.to_dict())
    review = engine.review_trial(failed, crops=crops)
    assert not review['continuation_authorized'] and engine.trial_review is None
    with pytest.raises(ValueError, match='review or completion'):
        engine.train_step([training_crop(base, 7)])


def test_plain_unactivated_or_phase_parent_cannot_train_or_skip_readiness():
    data = fixture()
    engine, base, *_ = data
    with pytest.raises(ValueError, match='No perceptual trial'):
        engine.train_step([training_crop(base)])
    with pytest.raises(ValueError, match='enable_trial'):
        engine.enable_perceptual({'passed': True})
    state = deepcopy(data[-1])
    state['quiet_phase'] = {'config': {'gradient_share': .02}}
    with pytest.raises(ValueError, match='separately declared transition'):
        engine.load_state_dict(state)


def test_assigning_a_fake_review_cannot_bypass_the_safety_boundary():
    data = fixture()
    engine, base, *_ = data
    enable(data)
    engine.trial_review = {'continuation_authorized': True}
    with pytest.raises(ValueError, match='Unverified review'):
        engine.train_step([training_crop(base)])
