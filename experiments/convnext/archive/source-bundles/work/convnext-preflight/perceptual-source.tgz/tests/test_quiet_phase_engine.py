"""Focused CPU integration checks for an isolated, strictly resumable phase trial."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import DistillationEngine, DistillationTrainingConfig, scored_batch
from audiovae_student.gradient_balancer import MelGradientCapConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_phase_engine import QuietPhaseDistillationEngine, QuietPhaseEngineConfig
from audiovae_student.teacher import CHECKPOINT_SHA256

torch.set_num_threads(1)


def make_engine(cls=QuietPhaseDistillationEngine, *, config=None, phase_share=0, mel_cap=None):
    torch.manual_seed(217)
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
        dilations=(1,), layer_scale_init=.1, normalization_mode='masked_batch_norm'))
    options = dict(config=config or DistillationTrainingConfig(optimizer='adamw', total_steps=6,
        warmup_steps=1, freeze_normalization_step=1, reconstruction_waveform_share=.75),
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=AudioDiscriminators(DiscriminatorConfig(periods=(2,),
            mpd_channels=(2, 2, 4, 4, 4), fft_sizes=(64,), mrd_channels=2)))
    if cls is QuietPhaseDistillationEngine:
        options.update(phase_config=QuietPhaseEngineConfig(gradient_share=phase_share), mel_cap=mel_cap)
    result = cls(model, **options)
    if cls is DistillationEngine and mel_cap is not None:
        result.balancer = result.balancer.fork_with_mel_cap(mel_cap)
    return result


def crop(*, context=0, valid=5 * DECODER_HOP, loud=False):
    rng = torch.Generator().manual_seed(44)
    frames = context + 5
    return TrainingCrop(torch.randn(1, 64, frames, generator=rng),
        torch.full((1, 1, frames * DECODER_HOP), .03 if loud else 1e-5), None,
        'a' * 64, 'fixture', context, 0, context, 5, valid)


def policy():
    return dict(teacher_checkpoint_sha256=CHECKPOINT_SHA256, teacher_state_unchanged=True,
                parent_checkpoint_sha256='b' * 64, evidence_sha256='c' * 64)


def parent(mel_cap=None):
    engine = make_engine(DistillationEngine, mel_cap=mel_cap)
    engine.train_step([crop()])
    engine.train_step([crop()])
    return deepcopy(engine.state_dict())


def candidate(saved=None, *, mel_cap=None):
    saved = parent(mel_cap) if saved is None else saved
    engine = make_engine(config=DistillationTrainingConfig(**saved['config']), mel_cap=mel_cap)
    engine.load_state_dict(saved)
    engine.fork_reconstruction(replace(engine.config, total_steps=4, warmup_steps=0,
                                       freeze_normalization_step=0,
                                       learning_rate_schedule='constant_after_warmup'))
    engine.enable_phase(gradient_share=.02, policy=policy())
    return engine


def test_disabled_engine_has_exact_base_update_and_preserves_parent_payload():
    saved = parent()
    before = state_fingerprint(saved)
    base, extension = make_engine(DistillationEngine), make_engine()
    base.load_state_dict(deepcopy(saved))
    extension.load_state_dict(saved)
    assert base.train_step([crop()]) == extension.train_step([crop()])
    state = extension.state_dict()
    assert state.pop('quiet_phase')['activation'] is None
    assert state_fingerprint(state) == state_fingerprint(base.state_dict())
    assert state_fingerprint(saved) == before


def test_candidate_preserves_targets_masks_and_never_exceeds_share():
    engine = candidate()
    item = crop(context=4, valid=4 * DECODER_HOP + 123)
    poisoned = item.teacher_audio.clone()
    poisoned[..., :item.scored_slice.start] = float('nan')
    poisoned[..., item.scored_slice.stop:] = float('inf')
    item = replace(item, teacher_audio=poisoned.requires_grad_(), latents=item.latents.requires_grad_())
    teacher_before = item.teacher_audio.detach().clone()
    discriminators = state_fingerprint(engine.discriminators.state_dict())
    metrics = engine.train_step([item])
    assert 0 < metrics['teacher_phase/achieved_share'] <= .02 + 1e-12
    assert metrics['teacher_phase/eligible_windows'] == 8
    assert metrics['teacher_phase/active_examples'] == 1
    assert item.teacher_audio.grad is None and item.latents.grad is None
    torch.testing.assert_close(item.teacher_audio, teacher_before, equal_nan=True)
    assert state_fingerprint(engine.discriminators.state_dict()) == discriminators
    engine.model.eval()
    scored = scored_batch(engine.model, [item], engine.criterion, 'cpu')
    base = torch.randn_like(scored.prediction).masked_fill(~scored.score_mask, 0)
    combined, _ = engine._combine_quiet(scored, base)
    assert not combined.masked_select(~scored.score_mask).count_nonzero()


@pytest.mark.parametrize('mode', ['short_quiet', 'loud_teacher', 'zero_base', 'matching_phase_zero'])
def test_inactive_or_zero_budget_preserves_exact_base_gradient(mode):
    engine = candidate()
    item = crop(valid=3 * DECODER_HOP if mode == 'short_quiet' else 5 * DECODER_HOP,
                loud=mode == 'loud_teacher')
    batch = scored_batch(engine.model, [item], engine.criterion, 'cpu')
    if mode == 'matching_phase_zero':
        batch = replace(batch, prediction=batch.prediction * 0,
                        targets=tuple(torch.zeros_like(target) for target in batch.targets))
    base = torch.randn_like(batch.prediction).masked_fill(~batch.score_mask, 0)
    if mode == 'zero_base':
        base.zero_()
    combined, metrics = engine._combine_quiet(batch, base)
    assert combined is base
    assert metrics['teacher_phase/achieved_share'] == 0
    assert metrics['teacher_phase/base_scale'] == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_candidate_resume_is_exact_and_preserves_cap_balancer_version_two():
    cap = MelGradientCapConfig(max_share=.25)
    engine = candidate(mel_cap=cap)
    engine.train_step([crop()])
    saved = deepcopy(engine.state_dict())
    assert saved['balancer']['format_version'] == 2
    restored = make_engine(config=engine.config, phase_share=.02, mel_cap=cap)
    restored.load_state_dict(saved)
    assert state_fingerprint(restored.state_dict()) == state_fingerprint(saved)
    assert engine.train_step([crop()]) == restored.train_step([crop()])
    assert state_fingerprint(engine.state_dict()) == state_fingerprint(restored.state_dict())
    with pytest.raises(ValueError, match='phase configuration'):
        make_engine(config=engine.config, phase_share=.03, mel_cap=cap).load_state_dict(saved)
    changed = deepcopy(saved)
    changed['model']['stem_norm.running_mean'][0] += .001
    with pytest.raises(ValueError, match='fixed statistics'):
        make_engine(config=engine.config, phase_share=.02, mel_cap=cap).load_state_dict(changed)
    with pytest.raises(ValueError, match='balancer checkpoint'):
        make_engine(config=engine.config, phase_share=.02).load_state_dict(saved)


def test_activation_requires_exact_parent_fork_teacher_and_unchanged_objective():
    fresh = make_engine()
    with pytest.raises(ValueError, match='parent load'):
        fresh.enable_phase(gradient_share=.02, policy=policy())
    with pytest.raises(ValueError, match='Load the exact parent'):
        fresh.fork_reconstruction(replace(fresh.config, freeze_normalization_step=0))
    fresh.load_state_dict(parent())
    with pytest.raises(ValueError, match='comparison fork'):
        fresh.enable_phase(gradient_share=.02, policy=policy())
    fresh.fork_reconstruction(replace(fresh.config, freeze_normalization_step=0))
    with pytest.raises(ValueError, match='unchanged original teacher'):
        fresh.enable_phase(gradient_share=.02, policy={**policy(), 'teacher_state_unchanged': False})
    fresh.enable_phase(gradient_share=.02, policy=policy())
    with pytest.raises(ValueError, match='reconstruction objective'):
        fresh.set_reconstruction_objective(waveform_share=.9, quiet_share=0)
    with pytest.raises(ValueError, match='GAN activation'):
        fresh.enable_perceptual({})
    with pytest.raises(ValueError, match='plain parent'):
        fresh.load_state_dict(parent())
    fresh.phase_config = replace(fresh.phase_config, gradient_share=0)
    with pytest.raises(ValueError, match='configuration changed'):
        fresh.train_step([crop()])


def test_rejected_general_quiet_loss_cannot_be_combined():
    engine = make_engine()
    engine.load_state_dict(parent())
    engine.fork_reconstruction(replace(engine.config, freeze_normalization_step=0, quiet_gradient_share=.01))
    with pytest.raises(ValueError, match='general quiet'):
        engine.enable_phase(gradient_share=.02, policy=policy())


@pytest.mark.parametrize('share', [-.01, .051, float('nan'), True])
def test_share_config_is_bounded(share):
    with pytest.raises(ValueError, match='share'):
        QuietPhaseEngineConfig(gradient_share=share)
