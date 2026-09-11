"""Synthetic CPU evidence-policy tests; these establish no audio quality claim."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from audiovae_student.distillation_training import evaluate_crops
from audiovae_student.gradient_balancer import MelGradientCapConfig
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import fixed_crops, PreflightConfig
from audiovae_student.perceptual_readiness import (PerceptualReadinessPolicy,
    assess_perceptual_readiness, collect_readiness_evidence, evaluation_evidence,
    discriminator_optimizer_proposal, _seal)
from audiovae_student.teacher import FrozenAudioVAE2, CHECKPOINT_SHA256, SOURCE_SHA256
from test_balance_comparison import parent_fixture

torch.set_num_threads(1)


def fixture():
    from audiovae_student.balance_comparison import new_arm, BalanceComparisonConfig
    parent, data = parent_fixture()
    engine, _ = new_arm(parent['engine'], BalanceComparisonConfig(total_steps=4,evaluation_interval=2), 'cpu', capped=False)
    engine.step = 3
    corpus, panel_rows = data[0], data[6]
    # Give the synthetic decoder a learnable reference with imperfect level
    # and quiet reconstruction. No test pretends these are real teacher assets.
    original = corpus.get(panel_rows[0].source_id)
    with torch.no_grad():
        engine.model.eval()
        initial = engine.model(original.latents)
        engine.model.output.weight.mul_(.02 / initial.square().mean().sqrt())
        target = engine.model(original.latents).detach() * 1.5
    target[..., :960] = 0
    target[..., 3 * 1920:3 * 1920 + 960] = 0
    record = replace(original, teacher_audio=target)
    corpus.records[panel_rows[0].source_id] = record
    corpus.teacher_identity = {'checkpoint_sha256': CHECKPOINT_SHA256,
        'source_sha256': SOURCE_SHA256, 'dtype': 'float32'}
    corpus._teacher = FrozenAudioVAE2(torch.nn.Linear(1, 1), corpus.teacher_identity)
    crops = tuple(fixed_crops(record, role='sentinel', config=PreflightConfig(scored_frames=3), minimum_samples=256))
    current_rows = evaluate_crops(engine, crops, include_signal_checks=True)
    prior = deepcopy(engine)
    prior.step = 2
    previous_rows = evaluate_crops(prior, crops, include_signal_checks=True)
    kwargs = {'run_identity': {'synthetic_fixture': 'same-fixed-panel-run'},
              'model_config': engine.model.config.to_dict()}
    current = evaluation_evidence(crops, current_rows, **kwargs)
    previous = evaluation_evidence(crops, previous_rows, **kwargs)
    evidence = collect_readiness_evidence(engine, crops, corpus=corpus,
        panel_rows=panel_rows, sample_counts=data[3])
    return engine, crops, current, previous, evidence, corpus, panel_rows, data[3]


def assess(data, **changes):
    engine, crops, current, previous, evidence, *_ = data
    return assess_perceptual_readiness(engine, crops, current=changes.get('current', current),
        previous=changes.get('previous', previous), evidence=changes.get('evidence', evidence))


def test_eligibility_is_distinct_from_final_point99_gate_and_does_not_mutate_engine():
    data = fixture()
    before = state_fingerprint(data[0].state_dict())
    result = assess(data)
    assert result['ready_for_bounded_trial'], result['checks']
    assert not result['final_acceptance']['passed']
    assert result['final_acceptance']['criteria']['waveform_cosine_min'] == .99
    assert result['final_acceptance']['criteria']['all_quiet_windows_must_pass']
    assert 'passed' not in result and not result['automatic_activation']
    assert state_fingerprint(data[0].state_dict()) == before
    assert data[0].perceptual_start is None
    with pytest.raises(ValueError, match='gate'):
        data[0].enable_perceptual(result)


def test_json_roundtrip_preserves_bound_evidence_and_no_rtf_is_claimed():
    data = fixture()
    result = assess(data, current=json.loads(json.dumps(data[2])),
                    previous=json.loads(json.dumps(data[3])), evidence=json.loads(json.dumps(data[4])))
    assert result['ready_for_bounded_trial']
    assert all(s['report']['rtf_measured'] is False for s in data[4]['streaming'])


@pytest.mark.parametrize('change', ['step', 'model', 'balancer', 'target'])
def test_stale_current_model_optimizer_or_raw_targets_are_rejected(change):
    data = fixture()
    engine = data[0]
    if change == 'step':
        engine.step += 1
    elif change == 'model':
        with torch.no_grad():
            next(engine.model.parameters()).add_(.01)
    elif change == 'balancer':
        engine.balancer.updates += 1
    else:
        data[1][0].teacher_audio[..., 0] += .01
    with pytest.raises(ValueError, match='stale'):
        assess(data)


@pytest.mark.parametrize('field', ['streaming', 'gradients', 'teacher', 'alignment'])
def test_missing_checks_do_not_become_passes(field):
    data = fixture()
    evidence = deepcopy(data[4])
    evidence.pop('sha256')
    del evidence[field]
    with pytest.raises(ValueError):
        assess(data, evidence=_seal(evidence))


def test_changed_evidence_content_without_matching_hash_is_rejected():
    data = fixture()
    evidence = deepcopy(data[4])
    evidence['alignment']['latent_channels'] = 32
    with pytest.raises(ValueError, match='hash'):
        assess(data, evidence=evidence)


def test_same_step_or_different_run_is_not_a_stability_history():
    data = fixture()
    with pytest.raises(ValueError, match='earlier matched-run'):
        assess(data, previous=data[2])
    body = deepcopy(data[3])
    body.pop('sha256')
    body['run_identity'] = {'synthetic_fixture': 'unrelated-run'}
    with pytest.raises(ValueError, match='matched-run'):
        assess(data, previous=_seal(body))


def test_missing_signal_checks_or_nonfinite_measurements_are_rejected():
    data = fixture()
    for key in ('student_peak_abs', 'teacher_mel'):
        rows = deepcopy(data[2]['rows'])
        if key == 'student_peak_abs':
            del rows[0][key]
        else:
            rows[0][key] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            evaluation_evidence(data[1], rows, run_identity={'fixture': True},
                                model_config=data[0].model.config.to_dict())


def test_finite_but_collapsed_or_regressing_audio_fails_trial_eligibility():
    data = fixture()
    # These synthetic policy controls alter recorded scalar values only. Their
    # purpose is to exercise the decision rule, not to generate audio evidence.
    for key, value, check in (
        ('student_rms', 0., 'no_active_nearzero_collapse'),
        ('waveform_to_silence_error_ratio', 1.1, 'better_than_silence_overall'),
        ('rms_db_error', -40., 'active_volume_sanity'),
        ('teacher_waveform', 100., 'waveform_error_stable'),
        ('student_clipped_samples', 100, 'clipping_sanity'),
    ):
        rows = deepcopy(data[2]['rows'])
        for row in rows:
            row[key] = value
        changed = evaluation_evidence(data[1], rows, run_identity=data[2]['run_identity'],
                                      model_config=data[0].model.config.to_dict())
        result = assess(data, current=changed)
        assert not result['ready_for_bounded_trial']
        assert not result['checks'][check]


def test_live_teacher_gradient_or_cache_target_changes_block_evidence_collection():
    data = fixture()
    engine, crops, _, _, _, corpus, rows, counts = data
    next(corpus._teacher.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        collect_readiness_evidence(engine, crops, corpus=corpus, panel_rows=rows, sample_counts=counts)
    next(corpus._teacher.parameters()).requires_grad_(False)
    changed = replace(crops[0], latents=crops[0].latents.clone())
    changed.latents[..., 0] += .01
    with pytest.raises(ValueError, match='frozen cache'):
        collect_readiness_evidence(engine, (changed, *crops[1:]), corpus=corpus, panel_rows=rows, sample_counts=counts)


def test_collection_preserves_rng_model_optimizer_teacher_and_raw_targets():
    data = fixture()
    engine, crops, _, _, _, corpus, rows, counts = data
    before = state_fingerprint((engine.state_dict(), corpus._teacher.state_dict(), torch.get_rng_state(),
                                [c.latents for c in crops], [c.teacher_audio for c in crops]))
    evidence = collect_readiness_evidence(engine, crops, corpus=corpus, panel_rows=rows, sample_counts=counts)
    after = state_fingerprint((engine.state_dict(), corpus._teacher.state_dict(), torch.get_rng_state(),
                               [c.latents for c in crops], [c.teacher_audio for c in crops]))
    assert before == after
    assert evidence['teacher']['state_sha256'] == state_fingerprint(corpus._teacher.state_dict())


def test_cap_v2_probe_preserves_existing_parameter_gradients_and_mixed_training_modes():
    data = fixture()
    engine, crops, _, _, _, corpus, rows, counts = data
    engine.balancer = engine.balancer.fork_with_mel_cap(MelGradientCapConfig())
    engine.model.train()
    engine.model.stem_norm.eval()
    parameters = tuple(engine.model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, (index + 1) / 100)
    gradients = [parameter.grad.clone() for parameter in parameters]
    modes = [module.training for module in engine.model.modules()]
    state = state_fingerprint(engine.state_dict())
    evidence = collect_readiness_evidence(engine, crops, corpus=corpus, panel_rows=rows, sample_counts=counts)
    assert evidence['gradients']['output_gradients']['mel_cap/enabled'] == 1
    assert state_fingerprint(engine.state_dict()) == state
    assert [module.training for module in engine.model.modules()] == modes
    for parameter, gradient in zip(parameters, gradients):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    assert assess(data, evidence=evidence)['ready_for_bounded_trial']


@pytest.mark.parametrize('damage', ['samples', 'parity', 'missing_gradient_route'])
def test_incomplete_streaming_or_gradient_routes_fail_closed(damage):
    data = fixture()
    body = deepcopy(data[4])
    body.pop('sha256')
    if damage == 'samples':
        body['streaming'][0]['report']['samples'] -= 1
    elif damage == 'parity':
        body['streaming'][0]['report']['passed'] = False
    else:
        body['gradients']['parameters'][0]['waveform_scaled_norm'] = 0.
    with pytest.raises(ValueError, match='streaming|gradient route'):
        assess(data, evidence=_seal(body))


def test_trial_budget_and_optimizer_are_explicit_and_not_engine_mutations():
    with pytest.raises(ValueError, match='bounded'):
        PerceptualReadinessPolicy(max_trial_updates=1001)
    with pytest.raises(ValueError, match='early review'):
        PerceptualReadinessPolicy(safety_review_update=300)
    proposal = discriminator_optimizer_proposal()
    assert proposal['betas'] == [.8, .9] and proposal['weight_decay'] == 0
    assert proposal['current_engine_defaults']['betas'] == [.9, .999]
