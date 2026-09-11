"""Matched-mask, quiet/peak failure cases and read-only streaming diagnostics."""

from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.fusion_evaluation import diagnose_streaming, evaluate_fusion
from audiovae_student.losses_distillation import DistillationLossConfig, DistillationReconstructionLoss
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.training import _restore_rng, _rng_state


@pytest.fixture(autouse=True)
def single_thread_cpu():
    threads, rng = torch.get_num_threads(), _rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    yield
    torch.set_num_threads(threads)
    _restore_rng(rng)


class StoredWaveDecoder(nn.Module):
    def __init__(self, wave, *, consume_rng=False, mutate=False):
        super().__init__()
        self.register_buffer('wave', wave.clone())
        self.register_buffer('counter', torch.zeros(()))
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Identity()
        self.consume_rng, self.mutate = consume_rng, mutate

    def forward(self, latents):
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(())
        if self.mutate:
            self.counter.add_(1)
        return self.wave + self.anchor * 0


def setup(teacher, prediction=None, *, context_frames=0, context_start=0,
          source_id='fixture', consume_rng=False, mutate=False):
    frames = teacher.shape[-1] // DECODER_HOP
    crop = TrainingCrop(torch.zeros(1, 64, frames, requires_grad=True), teacher, None,
        'a' * 64, source_id, context_frames + context_start, context_start,
        context_frames, frames - context_frames, (frames - context_frames) * DECODER_HOP)
    model = StoredWaveDecoder(teacher if prediction is None else prediction,
                             consume_rng=consume_rng, mutate=mutate)
    criterion = DistillationReconstructionLoss(DistillationLossConfig())
    engine = SimpleNamespace(model=model, criterion=criterion, device=torch.device('cpu'), step=123,
                             config=SimpleNamespace(quiet_audio=QuietAudioConfig()))
    return engine, crop


def test_exact_teacher_has_zero_quality_errors_and_preserves_tensor_grad_modes_and_rng():
    wave = torch.zeros(1, 1, DECODER_HOP * 6, requires_grad=True)
    engine, crop = setup(wave, consume_rng=True)
    engine.model.train()
    engine.model.child.eval()
    engine.model.anchor.grad = torch.tensor(7.)
    modes = [module.training for module in engine.model.modules()]
    before = state_fingerprint(engine.model.state_dict()), state_fingerprint(_rng_state())
    result = evaluate_fusion(engine, [crop], {'fixture': {'group': 'silence'}})
    row, summary = result['rows'][0], result['summary']
    assert row['waveform_raw_mae'] == row['teacher_mel'] == 0
    assert row['high_frequency']['complex_residual_rms'] == 0
    assert row['quiet_phase480']['residual_template_rms'] == 0
    assert not row['waveform_cosine_defined']
    assert summary['quiet_fraction_passing'] == 1 and summary['nonquiet_cosine_mean'] is None
    assert result['groups']['group/silence']['crops'] == 1
    assert before == (state_fingerprint(engine.model.state_dict()), state_fingerprint(_rng_state()))
    assert [module.training for module in engine.model.modules()] == modes
    assert engine.model.anchor.grad == 7 and wave.grad is None and crop.latents.grad is None
    json.dumps(result, allow_nan=False)


def test_parameter_free_variant_migration_cannot_collide_with_control_identity():
    engine, crop = setup(torch.zeros(1, 1, DECODER_HOP * 6))
    engine.fusion_migration = {'variant': 'control', 'architecture': {'terminal_tanh': False}}
    control = evaluate_fusion(engine, [crop], {})
    engine.fusion_migration = {'variant': 'tanh', 'architecture': {'terminal_tanh': True}}
    variant = evaluate_fusion(engine, [crop], {})
    assert control['model_identity']['parameter_state_sha256'] == variant['model_identity']['parameter_state_sha256']
    assert control['model_state_sha256'] != variant['model_state_sha256']
    assert control['model_identity']['fusion_migration']['variant'] == 'control'
    assert variant['model_identity']['fusion_migration'] == engine.fusion_migration


@pytest.mark.parametrize('context_start,expected_excluded,expected_peaks', [(1, 6, 0), (0, 0, 6)])
def test_shared_context_prefix_mask_applies_to_every_metric(context_start, expected_excluded, expected_peaks):
    context, score = 29, 6
    teacher = torch.zeros(1, 1, DECODER_HOP * (context + score))
    prediction = teacher.clone()
    start = context * DECODER_HOP
    prediction[..., start:start + 6] = 2
    # Invalid teacher history must not enter any scored metric.
    teacher[..., :start] = float('nan')
    engine, crop = setup(teacher, prediction, context_frames=context, context_start=context_start)
    result = evaluate_fusion(engine, [crop], {})
    row = result['rows'][0]
    assert row['context_excluded_samples'] == expected_excluded
    assert row['student_overshoot_samples'] == expected_peaks
    assert row['samples'] == DECODER_HOP * score - expected_excluded
    assert row['quiet_windows']['valid_samples'] == row['samples']
    assert result['summary']['context_excluded_samples'] == expected_excluded
    assert row['absolute_scored_start_sample'] == crop.start_frame * DECODER_HOP + expected_excluded
    if expected_excluded:
        assert row['waveform_raw_mae'] == row['teacher_mel'] == 0
        assert row['high_frequency']['complex_residual_rms'] == 0
        assert row['quiet_windows']['quiet_failed_count'] == 0
        assert row['quiet_phase480']['residual_template_rms'] == 0
    else:
        assert row['waveform_raw_mae'] > 0
        assert row['quiet_windows']['quiet_failed_count'] == 1


def test_seventh_sample_is_retained_and_detected_even_with_the_common_prefix_exclusion():
    teacher = torch.zeros(1, 1, 35 * DECODER_HOP)
    prediction = teacher.clone()
    prediction[..., 29 * DECODER_HOP + 6] = 1.2
    engine, crop = setup(teacher, prediction, context_frames=29, context_start=4)
    row = evaluate_fusion(engine, [crop], {})['rows'][0]
    assert row['student_overshoot_samples'] == 1
    assert row['waveform_raw_mae'] > 0 and row['quiet_windows']['quiet_failed_count'] == 1


@pytest.mark.parametrize('change', ['mute', 'invert'])
def test_quiet_breath_failure_is_visible_even_without_loud_output_or_peak_overshoot(change):
    samples = torch.arange(DECODER_HOP * 6, dtype=torch.float64)
    teacher = (.0005 * torch.sin(2 * torch.pi * samples / 480)).float()[None, None]
    prediction = torch.zeros_like(teacher) if change == 'mute' else -teacher
    engine, crop = setup(teacher, prediction)
    result = evaluate_fusion(engine, [crop], {'fixture': {'condition': 'breathing'}})
    row = result['rows'][0]
    assert row['student_overshoot_samples'] == 0 and row['student_peak_abs'] < .001
    assert row['quiet_windows']['quiet_failed_count'] == 12
    assert row['quiet_phase480']['residual_template_rms'] > .0003
    assert result['summary']['quiet_fraction_passing'] == 0
    assert result['groups']['group/expressive']['crops'] == 1


def test_high_frequency_diagnostic_detects_hf_error_without_conflating_lowband_energy():
    samples = torch.arange(DECODER_HOP * 6, dtype=torch.float64)
    values = []
    for hz in (1500, 12000):
        prediction = (.01 * torch.sin(2 * torch.pi * hz * samples / 48000)).float()[None, None]
        engine, crop = setup(torch.zeros_like(prediction), prediction)
        values.append(evaluate_fusion(engine, [crop], {})['summary'])
    assert values[1]['high_frequency_complex_residual_rms'] > 1000 * values[0]['high_frequency_complex_residual_rms']
    assert values[0]['raw_mae_sample_weighted'] > .004
    assert values[1]['raw_mae_sample_weighted'] > .004


def test_overshoot_and_near_saturation_are_distinct_counts_and_teacher_is_reported():
    teacher = torch.zeros(1, 1, 6 * DECODER_HOP)
    prediction = teacher.clone()
    prediction[..., :6] = torch.tensor([1.001, 1., .999, -1.002, -.999, .998])
    teacher[..., 10] = 1
    engine, crop = setup(teacher, prediction)
    row = evaluate_fusion(engine, [crop], {})['rows'][0]
    assert row['student_overshoot_samples'] == 2
    assert row['student_full_scale_samples'] == 3
    assert row['student_near_saturation_samples'] == 5
    assert row['teacher_overshoot_samples'] == 0 and row['teacher_near_saturation_samples'] == 1


def test_failure_restores_mutated_model_rng_and_original_modes_instead_of_publishing_report():
    engine, crop = setup(torch.zeros(1, 1, DECODER_HOP * 6), consume_rng=True, mutate=True)
    engine.model.child.eval()
    modes = [module.training for module in engine.model.modules()]
    before = state_fingerprint(engine.model.state_dict()), state_fingerprint(_rng_state())
    with pytest.raises(RuntimeError, match='mutated module tensors'):
        evaluate_fusion(engine, [crop], {})
    assert before == (state_fingerprint(engine.model.state_dict()), state_fingerprint(_rng_state()))
    assert modes == [module.training for module in engine.model.modules()]


def test_bad_context_short_regions_and_invalid_scored_audio_are_rejected():
    engine, crop = setup(torch.zeros(1, 1, DECODER_HOP * 6))
    with pytest.raises(ValueError, match='context disagree'):
        evaluate_fusion(engine, [replace(crop, context_start_frame=1)], {})
    with pytest.raises(ValueError, match='29 context'):
        evaluate_fusion(engine, [replace(crop, context_start_frame=1, start_frame=1)], {})
    with pytest.raises(ValueError, match='too short'):
        evaluate_fusion(engine, [replace(crop, valid_scored_samples=2046)], {})
    bad_target = crop.teacher_audio.clone()
    bad_target[..., 100] = float('inf')
    with pytest.raises(ValueError, match='Retained evaluation samples'):
        evaluate_fusion(engine, [replace(crop, teacher_audio=bad_target)], {})


def test_real_decoder_stream_diagnostic_preserves_state_and_counts_partial_final_chunk():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, dilations=(1, 2), layer_scale_init=.1))
    model.train()
    latents = torch.randn(1, 64, 5, requires_grad=True)
    before = state_fingerprint(model.state_dict()), state_fingerprint(_rng_state())
    result = diagnose_streaming(model, latents, frames_per_chunk=2)
    assert result['sample_count_passed']
    assert result['expected_samples'] == 9600 and result['flush_samples'] == 0
    assert [row['input_frames'] for row in result['chunks']] == [2, 2, 1]
    assert result['max_abs_error'] < 2e-6
    assert before == (state_fingerprint(model.state_dict()), state_fingerprint(_rng_state()))
    assert model.training and latents.grad is None
    assert not result['rtf_measured']


class BadStreamDecoder(nn.Module):
    def __init__(self, fault):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.fault = fault

    def forward(self, latents):
        return latents[:, :1].repeat_interleave(DECODER_HOP, dim=-1) + self.anchor * 0

    def stream(self):
        model = self

        class Stream:
            def decode_chunk(self, latents):
                value = model(latents)
                return value[..., :-1] if model.fault == 'drop' else value.flip(-1)

            def flush(self):
                return model.anchor.new_zeros(1, 1, 0)

            def close(self):
                pass

        return Stream()


def test_lost_stream_samples_are_not_hidden_by_minimum_length_trimming():
    result = diagnose_streaming(BadStreamDecoder('drop'), torch.ones(1, 64, 5))
    assert not result['sample_count_passed']
    assert result['stream_samples'] == result['expected_samples'] - 3
    assert result['max_abs_error'] is None and result['rms_error'] is None


def test_wrong_chunk_order_is_visible_when_sample_counts_match():
    latents = torch.arange(5.).reshape(1, 1, 5).expand(1, 64, 5)
    result = diagnose_streaming(BadStreamDecoder('reverse'), latents)
    assert result['sample_count_passed'] and result['max_abs_error'] == 1


def test_empty_stream_is_consistent_and_invalid_chunk_size_rejected():
    model = BadStreamDecoder('drop')
    result = diagnose_streaming(model, torch.empty(1, 64, 0))
    assert result['sample_count_passed'] and result['max_abs_error'] == 0
    with pytest.raises(ValueError, match='positive integer'):
        diagnose_streaming(model, torch.empty(1, 64, 0), frames_per_chunk=True)
