"""Synthetic mechanics for the complete recipe, with no learned-quality claim."""
from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig, generator_losses
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine, calibration_batch, scored_batch_v2
from audiovae_student.training import _restore_rng, _rng_state


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    yield
    torch.set_num_threads(previous)


def tiny_engine(recipe=None):
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, dilations=(1, 2), layer_scale_init=.1,
        normalization_mode='masked_batch_norm', adapter_mode='raw_repeat_phase_bias'))
    recipe = recipe or RecipeV2Config(total_steps=6, reconstruction_warmup_steps=2,
        perceptual_ramp_steps=2, learning_rate_warmup_steps=1, optimizer='adamw')
    discriminator = AudioDiscriminators(DiscriminatorConfig(periods=(2, 3),
        mpd_channels=(2, 4, 4, 4, 4), fft_sizes=(128,), mrd_channels=2))
    return RecipeV2Engine(model, recipe=recipe, discriminators=discriminator)


def crops():
    result = []
    for index, (context, valid) in enumerate(((0, 6 * DECODER_HOP), (4, 10002))):
        frames = context + 6
        result.append(TrainingCrop(torch.randn(1, 64, frames, requires_grad=True),
            (torch.randn(1, 1, frames * DECODER_HOP) * .03).requires_grad_(), None,
            'a' * 64, 'fixture-' + str(index), context, 0, context, 6, valid))
    return result


def at_boundary(engine, items):
    engine.train_step(items)
    engine.train_step(items)
    assert engine.step == 2 and engine.calibration_due


def calibrated(engine, items):
    at_boundary(engine, items)
    return engine.calibrate([calibration_batch(items, 'cpu')],
        provenance={'split': 'train', 'manifest_sha256': 'f' * 64, 'purpose': 'synthetic-mechanics'})


def normalization_buffers(model):
    suffixes = ('running_mean', 'running_var', 'num_batches_tracked', 'statistics_frozen')
    return {name: value.clone() for name, value in model.named_buffers()
            if name.endswith(suffixes)}


def test_mel_is_active_from_first_update_while_discriminator_stays_unused():
    engine, items = tiny_engine(), crops()
    d_before = deepcopy(engine.discriminators.state_dict())
    latent_before = [item.latents.detach().clone() for item in items]
    target_before = [item.teacher_audio.detach().clone() for item in items]
    for expected_step in (1, 2):
        metrics = engine.train_step(items)
        assert metrics['step'] == expected_step
        assert metrics['teacher_mel'] > 0
        assert metrics['teacher_mel/raw_norm'] > 0
        assert metrics['teacher_mel/target_share'] == .5
        assert metrics['perceptual_fraction'] == 0
        assert metrics['discriminator_updates'] == 0
        assert 'adversarial' not in metrics and 'feature_matching' not in metrics
    assert engine.perceptual_start is None and engine.gate is None
    assert engine.discriminator_optimizer.state_dict()['state'] == {}
    assert state_fingerprint(engine.discriminators.state_dict()) == state_fingerprint(d_before)
    for i, item in enumerate(items):
        torch.testing.assert_close(item.latents, latent_before[i], atol=0, rtol=0)
        torch.testing.assert_close(item.teacher_audio, target_before[i], atol=0, rtol=0)
        assert item.latents.grad is None and item.teacher_audio.grad is None


def test_boundary_requires_real_calibration_and_rejects_a_quality_gate_shortcut():
    engine, items = tiny_engine(), crops()
    with pytest.raises(ValueError, match='warmup boundary'):
        engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})
    at_boundary(engine, items)
    before = state_fingerprint(engine.state_dict())
    with pytest.raises(ValueError, match='calibration is required'):
        engine.train_step(items)
    assert state_fingerprint(engine.state_dict()) == before
    with pytest.raises(ValueError, match='not a quality entry gate'):
        engine.enable_perceptual({'passed': True, 'waveform_cosine': 1.0})
    with pytest.raises(ValueError, match='training-only'):
        engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'dev'})
    assert engine.calibration is None and engine.perceptual_start is None


def test_calibration_preserves_weights_optimizer_rng_and_freezes_statistics():
    engine, items = tiny_engine(), crops()
    at_boundary(engine, items)
    weights = {name: parameter.clone() for name, parameter in engine.model.named_parameters()}
    optimizer = state_fingerprint(engine.optimizer.state_dict())
    rng = engine.crop_generator.get_state().clone()
    report = engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})
    assert report['completed_step'] == 2
    assert report['report']['parameters_unchanged'] is True
    assert report['report']['statistics_frozen'] is True
    assert engine.perceptual_start == 2 and engine.gate is None
    assert not engine.calibration_due
    assert state_fingerprint(engine.optimizer.state_dict()) == optimizer
    torch.testing.assert_close(engine.crop_generator.get_state(), rng, atol=0, rtol=0)
    for name, parameter in engine.model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
    frozen = normalization_buffers(engine.model)
    engine.train_step(items)
    for name, value in normalization_buffers(engine.model).items():
        torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match='exactly once'):
        engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})


def test_scheduled_gan_and_feature_matching_update_actual_weights_without_quality_gate():
    engine, items = tiny_engine(), crops()
    targets_before = [item.teacher_audio.detach().clone() for item in items]
    calibrated(engine, items)
    engine.model.eval()
    batch, _ = scored_batch_v2(engine.model, items, engine.reconstruction, 'cpu')
    cosine = torch.nn.functional.cosine_similarity(batch.predictions[0].flatten(),
        batch.targets[0].flatten(), dim=0)
    assert cosine < .99  # These random synthetic pairs are deliberately unqualified.
    predicted, teacher, count = engine._perceptual_audio(batch)
    assert count == len(items) and predicted.shape[-1] == 9120
    perceptual = generator_losses(engine.discriminators, predicted, teacher)
    output_gradient, = torch.autograd.grad(perceptual['feature_matching'], engine.model.output.weight)
    assert torch.isfinite(output_gradient).all() and output_gradient.norm() > 0
    assert all(parameter.grad is None for parameter in engine.discriminators.parameters())
    d_before = state_fingerprint(engine.discriminators.state_dict())
    output_before = engine.model.output.weight.detach().clone()
    first = engine.train_step(items)
    assert first['perceptual_fraction'] == .5
    assert first['discriminator_updates'] == 1
    assert first['feature_matching/raw_norm'] > 0
    assert first['adversarial/raw_norm'] > 0
    assert first['discriminator_gradient_norm'] > 0
    assert engine.gate is None
    assert state_fingerprint(engine.discriminators.state_dict()) != d_before
    assert not torch.equal(engine.model.output.weight, output_before)
    assert engine.discriminator_optimizer.state_dict()['state']
    assert all(parameter.grad is None for parameter in engine.discriminators.parameters())
    assert all(item.teacher_audio.grad is None and item.latents.grad is None for item in items)
    second = engine.train_step(items)
    assert second['perceptual_fraction'] == 1 and second['discriminator_updates'] == 2
    for item, before in zip(items, targets_before):
        torch.testing.assert_close(item.teacher_audio, before, atol=0, rtol=0)
    group = engine.discriminator_optimizer.param_groups[0]
    assert group['betas'] == (.8, .9) and group['weight_decay'] == 0


@pytest.mark.parametrize('saved_stage', ['warmup', 'calibration_due', 'calibrated', 'perceptual'])
def test_exact_resume_preserves_next_update_and_optimizer_balancer_crop_rng(tmp_path, saved_stage):
    engine, items = tiny_engine(), crops()
    if saved_stage == 'warmup':
        engine.train_step(items)
    else:
        at_boundary(engine, items)
        if saved_stage != 'calibration_due':
            engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})
        if saved_stage == 'perceptual':
            engine.train_step(items)
    saved = {'engine': deepcopy(engine.state_dict()), 'rng': _rng_state()}
    checkpoint = tmp_path / 'fixture.pt'
    torch.save(saved, checkpoint)
    if saved_stage == 'calibration_due':
        engine.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})
    expected_metrics = engine.train_step(items)
    expected_state = state_fingerprint(engine.state_dict())
    expected_rng = state_fingerprint(_rng_state())
    restored = tiny_engine()
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored.load_state_dict(loaded['engine'])
    assert state_fingerprint(restored.state_dict()) == state_fingerprint(saved['engine'])
    _restore_rng(loaded['rng'])  # Global RNG belongs to the runner checkpoint.
    if saved_stage == 'calibration_due':
        restored.calibrate([calibration_batch(items, 'cpu')], provenance={'split': 'train'})
    assert restored.train_step(items) == expected_metrics
    assert state_fingerprint(restored.state_dict()) == expected_state
    assert state_fingerprint(_rng_state()) == expected_rng


@pytest.mark.parametrize('field', ['recipe', 'model_config', 'reconstruction_v2', 'discriminator_config'])
def test_changed_recipe_components_are_not_exact_resumes(field):
    engine = tiny_engine()
    state = deepcopy(engine.state_dict())
    if field == 'recipe':
        state[field]['reconstruction_warmup_steps'] += 1
    elif field == 'model_config':
        state[field]['adapter_mode'] = 'learned_phase'
    elif field == 'reconstruction_v2':
        state[field]['mel_log_weight'] = 0
    else:
        state[field]['log_epsilon'] *= 2
    with pytest.raises(ValueError, match='not an exact resume'):
        tiny_engine().load_state_dict(state)


@pytest.mark.parametrize('change', ['balancer_share', 'report', 'norm_buffer', 'frozen_flag',
                                  'discriminator_count', 'discriminator_betas', 'discriminator_decay',
                                  'discriminator_rate'])
def test_calibration_schedule_and_discriminator_resume_bindings_reject_changes(change):
    engine, items = tiny_engine(), crops()
    calibrated(engine, items)
    engine.train_step(items)
    saved = deepcopy(engine.state_dict())
    if change == 'balancer_share':
        saved['balancer']['weights']['adversarial'] += .01
    elif change == 'report':
        saved['calibration']['report']['stem']['inputs_sha256'] = '0' * 64
    elif change == 'norm_buffer':
        saved['model']['affine.running_mean'].add_(.01)
    elif change == 'frozen_flag':
        saved['model']['stem_norm.statistics_frozen'].fill_(False)
    elif change == 'discriminator_count':
        saved['discriminator_updates'] += 1
    else:
        group = saved['discriminator_optimizer']['param_groups'][0]
        if change == 'discriminator_betas':
            group['betas'] = (.9, .999)
        elif change == 'discriminator_decay':
            group['weight_decay'] = .01
        else:
            group['lr'] *= 2
    restored = tiny_engine()
    before = state_fingerprint(restored.state_dict())
    with pytest.raises(ValueError):
        restored.load_state_dict(saved)
    assert state_fingerprint(restored.state_dict()) == before


def test_planned_crops_cannot_silently_skip_the_adversarial_stage_and_budget_is_hard():
    engine, items = tiny_engine(), crops()
    short = replace(items[0], valid_scored_samples=7998)
    initial = state_fingerprint(engine.state_dict())
    with pytest.raises(ValueError, match='same adversarial sample count'):
        engine.train_step([short])
    assert state_fingerprint(engine.state_dict()) == initial
    calibrated(engine, items)
    while engine.step < engine.recipe.total_steps:
        engine.train_step(items)
    before = state_fingerprint(engine.state_dict())
    with pytest.raises(ValueError, match='budget exhausted'):
        engine.train_step(items)
    assert state_fingerprint(engine.state_dict()) == before
