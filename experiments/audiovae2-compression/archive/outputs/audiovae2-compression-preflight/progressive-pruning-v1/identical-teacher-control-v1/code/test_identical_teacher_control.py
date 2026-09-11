"""CPU positive and negative controls for the unchanged teacher pipeline.

The small decoder follows the real causal/WN/conditioning operation pattern.
Model tests exercise the new entry helpers, rather than an alternate decoder.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))

import identical_teacher_control as control
import group_model as gm
from test_group_model import TinyDecoder, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(707)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def teacher():
    return TinyDecoder().eval().requires_grad_(False)


def cached_crop(frames=3, context=1, tail=17):
    t = torch.arange(frames * 1920, dtype=torch.float32)
    return {
        'source_id': 'fixture', 'latents': torch.zeros(1, 64, frames),
        'teacher_audio': (.07 * torch.sin(t * .071) + .003 * torch.cos(t * .023))[None, None],
        'context_start_frame': 7, 'start_frame': 7 + context,
        'context_frames': context, 'valid_scored_samples': (frames-context)*1920-tail,
    }


def test_full_width_control_preserves_every_raw_weight_and_independent_storage(teacher):
    before = snapshot(teacher)
    rng = torch.get_rng_state().clone()
    model = control.build_control(teacher)
    assert torch.equal(torch.get_rng_state(), rng)
    assert_snapshot(model.decoder, before)
    for name, value in model.decoder.state_dict().items():
        assert value.data_ptr() != teacher.state_dict()[name].data_ptr(), name
    assert model.selections == {'stage2_indices': list(range(32)), 'stage3_indices': list(range(16))}
    assert len(model.group_named_parameters()) == 90
    for name, p in model.decoder.named_parameters():
        assert p.requires_grad == name.startswith(gm.GROUP_PREFIXES), name
    assert_snapshot(teacher, before)
    receipt = control.state_identity(teacher, model)
    assert receipt['equal'] and receipt['independent_storage']
    assert receipt['teacher_state_sha256'] == receipt['control_state_sha256']


@pytest.mark.parametrize('frames,mode', [(1, False), (3, False), (3, True)])
def test_native_trace_and_split_wrapper_match_all_boundaries_in_both_modes(teacher, frames, mode):
    model = control.build_control(teacher).train(mode)
    z = torch.randn(1, 8, frames) * .1
    sr = torch.tensor([48000], dtype=torch.int32)
    state = snapshot(teacher)
    with torch.no_grad():
        native = control.native_decoder_forward(teacher, z, sr)
        oracle = teacher(z, sr)
        trace = gm.teacher_trace(teacher, z, sr)
        actual = model.forward_from_latents(z, sr)
    torch.testing.assert_close(native, oracle, atol=0, rtol=0)
    torch.testing.assert_close(actual['waveform'], native, atol=0, rtol=0)
    assert set(actual) == set(trace)
    for key in trace:
        torch.testing.assert_close(actual[key], trace[key], atol=0, rtol=0, msg=key)
    comparisons = control.compare_forward(teacher, model, z, sr)
    assert comparisons['all_exact'] and comparisons['allclose_existing']
    assert comparisons['student_training'] is mode
    for stage in range(1, 7):
        assert comparisons['comparisons'][f'stage{stage}_output']['exact']
        assert comparisons['comparisons'][f'stage{stage}_conditioned_input']['exact']
    assert_snapshot(teacher, state)


def test_state_and_native_boundary_checks_detect_perturbations_and_shared_storage(teacher):
    alias = control.state_identity(teacher, SimpleNamespace(decoder=teacher))
    assert alias['equal'] and not alias['independent_storage']
    model = control.build_control(teacher)
    with torch.no_grad():
        model.decoder.sr_cond_model[4].bias_embed.weight.add_(.04)
    receipt = control.state_identity(teacher, model)
    assert not receipt['equal'] and receipt['independent_storage']
    result = control.compare_forward(teacher, model, torch.randn(1, 8, 2) * .1)
    assert not result['all_exact'] and not result['allclose_existing']
    assert result['comparisons']['stage2_output']['exact']
    assert not result['comparisons']['stage3_conditioned_input']['exact']
    assert all(not m._forward_hooks for m in teacher.modules())
    assert all(not m._forward_hooks for m in model.decoder.modules())


def test_cache_masks_keep_context_and_partial_tail_separate():
    crop = cached_crop()
    prediction = crop['teacher_audio'].clone()
    score = control.cache_check(crop, prediction)
    assert score['scored_span'] == [1920, 5743]
    assert score['valid']['elements'] == 3823
    assert score['context']['elements'] == 1920
    assert score['right_padding']['elements'] == 17
    assert all(score[key]['exact'] for key in ('valid', 'context', 'right_padding', 'full'))
    prediction[..., :1920] += .02
    prediction[..., -17:] -= .03
    changed = control.cache_check(crop, prediction)
    assert changed['valid']['exact']
    assert not changed['context']['allclose_existing']
    assert not changed['right_padding']['allclose_existing']
    assert not changed['full']['exact']


@pytest.mark.parametrize('damage', ['stale', 'shifted', 'crop_origin'])
def test_cache_checks_detect_wrong_target_or_time_alignment(damage):
    crop = cached_crop()
    prediction = crop['teacher_audio'].clone()
    if damage == 'stale': prediction[..., 2200:2400] += .01
    elif damage == 'shifted': prediction = prediction.roll(1, -1)
    else:
        prediction = torch.cat([prediction[..., 1920:], prediction[..., :1920]], -1)
    result = control.cache_check(crop, prediction)
    assert not result['valid']['exact'] and not result['valid']['allclose_existing']
    assert result['valid']['residual_rms'] > 0


@pytest.mark.parametrize('damage', ['length', 'duration', 'context', 'tail', 'dtype', 'latent_width', 'latent_dtype', 'nan'])
def test_cache_geometry_and_nonfinite_inputs_fail_closed(damage):
    crop = cached_crop()
    prediction = crop['teacher_audio'].clone()
    if damage == 'length': prediction = prediction[..., :-1]
    elif damage == 'duration': crop['latents'] = crop['latents'][..., :-1]
    elif damage == 'context': crop['start_frame'] += 1
    elif damage == 'tail': crop['valid_scored_samples'] += 18
    elif damage == 'dtype': prediction = prediction.double()
    elif damage == 'latent_width': crop['latents'] = crop['latents'][:, :8]
    elif damage == 'latent_dtype': crop['latents'] = crop['latents'].double()
    else: prediction[..., 0] = float('nan')
    with pytest.raises(ValueError): control.cache_check(crop, prediction)


def test_exact_and_numerical_parity_are_not_conflated():
    reference = torch.tensor([[[0., .5]]])
    actual = reference + torch.tensor([[[1e-6, 1e-6]]])
    close = control.compare_tensors(actual, reference)
    assert not close['exact'] and close['allclose_existing']
    assert close['residual_rms'] > 0
    empty = control.compare_tensors(actual, reference, torch.zeros_like(reference, dtype=torch.bool))
    assert empty['elements'] == 0 and empty['residual_rms'] is None


def test_zero_silence_has_no_invented_cosine_or_gain_and_passes_absolute_checks():
    target = torch.zeros(1, 1, 973)
    valid = torch.ones_like(target, dtype=torch.bool)
    score = control.zero_safe_metrics(target, target, valid)
    assert score['cosine'] is None and score['rms_ratio'] is None
    assert score['mae'] == score['residual_rms'] == 0
    assert score['quiet_windows'] == 2 and score['quiet_failed'] == 0
    assert score['quiet_passed'] is True
    assert [row['valid_samples'] for row in score['quiet']['windows']] == [960, 13]
    noisy = control.zero_safe_metrics(target + 1e-3, target, valid)
    assert noisy['cosine'] is None and noisy['rms_ratio'] is None
    assert noisy['quiet_failed'] == 2
    assert noisy['residual_rms'] == pytest.approx(1e-3)


def test_perfect_cosine_does_not_hide_amplitude_error_and_masked_samples_do_not_reweight():
    target = torch.full((1, 1, 1933), .03)
    prediction = target * 1.2
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[..., 960:973] = True
    # These excluded values must affect neither RMS nor source-window counts.
    prediction[..., :960] = -.8
    prediction[..., 973:] = .9
    score = control.zero_safe_metrics(prediction, target, valid)
    assert score['elements'] == 13
    assert score['cosine'] == pytest.approx(1.)
    assert score['rms_ratio'] == pytest.approx(1.2)
    assert score['mae'] == score['residual_rms'] == pytest.approx(.006)
    assert score['quiet_windows'] == 0 and score['quiet_failed'] == 0
    assert score['quiet_passed'] is None
    assert [(w['start_sample'], w['valid_samples']) for w in score['quiet']['windows']] == [(960, 13)]


def test_full_control_materializes_loaded_wn_values_without_stale_weight_cache(teacher):
    layer = teacher.model[3].block[1]
    stale = layer.weight.detach().clone()
    with torch.no_grad():
        layer.weight_g.mul_(1.3)
        layer.weight_v.add_(.017)
    model = control.build_control(teacher)
    assert not torch.allclose(gm.effective_weight(layer), stale)
    torch.testing.assert_close(gm.effective_weight(model.decoder.model[3].block[1]),
                               gm.effective_weight(layer), atol=0, rtol=0)
    z = torch.randn(1, 8, 2) * .1
    with torch.no_grad():
        torch.testing.assert_close(model.forward_from_latents(z)['waveform'],
                                   teacher(z), atol=0, rtol=0)


class TinyFrozenTeacher(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = decoder


def update_fixture(monkeypatch, teacher):
    """Use real B1/accum12 update, only adapting the tiny encoder width/device."""
    original_batch = control.base.batch
    monkeypatch.setattr(control.base, 'batch', lambda rows: original_batch(rows, device='cpu'))
    wrapped = TinyFrozenTeacher(teacher).eval().requires_grad_(False)

    @torch.no_grad()
    def target(instance, z):
        return gm.teacher_trace(instance.model.decoder, z[:, :8])

    monkeypatch.setattr(control.base, 'teacher_forward', target)
    crops = []
    for i in range(12):
        row = cached_crop(frames=3, context=i % 2, tail=i * 7)
        row['source_id'] = f'update-{i}'
        row['latents'] = torch.randn(1, 64, 3) * .05
        row['teacher_audio'] = target(wrapped, row['latents'])['waveform'].clone()
        crops.append(row)
    common = control.base.ReconstructionV2(control.base.ReconstructionV2Config(
        fft_sizes=(32, 64), mel_bands=(4, 8)))
    model = control.build_control(teacher)
    optimizer = torch.optim.AdamW(model.trainable_group_parameters(), lr=3e-5,
                                 betas=(.9, .99), eps=1e-8, weight_decay=0)
    return model, wrapped, crops, common, optimizer


def test_fresh_adam_at_exact_teacher_optimum_has_zero_gradients_and_no_weight_update(monkeypatch, teacher):
    model, wrapped, crops, common, optimizer = update_fixture(monkeypatch, teacher)
    model.train()
    assert not optimizer.state
    before, teacher_before = snapshot(model), snapshot(wrapped)
    rng = torch.get_rng_state().clone()
    control.stability_step(model, wrapped, crops, common, optimizer)
    assert_snapshot(model, before)
    assert_snapshot(wrapped, teacher_before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(optimizer.state) == 90
    for parameter in model.trainable_group_parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        state = optimizer.state[parameter]
        assert float(state['step']) == 1
        assert torch.count_nonzero(state['exp_avg']) == torch.count_nonzero(state['exp_avg_sq']) == 0
    assert all(p.grad is None for p in wrapped.parameters())


def test_perturbed_negative_control_generates_gradients_without_changing_teacher_or_outer_layers(monkeypatch, teacher):
    model, wrapped, crops, common, optimizer = update_fixture(monkeypatch, teacher)
    with torch.no_grad():
        model.decoder.model[5].block[4].block[3].bias.add_(.03)
    before, teacher_before = snapshot(model.decoder), snapshot(wrapped)
    control.stability_step(model, wrapped, crops, common, optimizer)
    changed = []
    for name, value in model.decoder.state_dict().items():
        if not torch.equal(value, before[name]):
            changed.append(name)
            assert name.startswith(gm.GROUP_PREFIXES)
    assert changed, 'The negative control must detect a real nonzero-error update'
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.trainable_group_parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.trainable_group_parameters())
    assert_snapshot(wrapped, teacher_before)


@pytest.mark.parametrize('damage', [None, 'still_running', 'receipt', 'source_order', 'protected'])
def test_control_cannot_run_before_completed_5000_or_against_another_source_ledger(tmp_path, damage):
    from progressive_continue_5000 import VERSION as TRAIN_VERSION
    source_ids = [f'source-{i}' for i in range(60000)]
    dependency = tmp_path / 'immutable.py'
    dependency.write_text('original input')
    launch = {'target_step': 5000, 'learning_rate': 3e-5, 'gradient_accumulation': 12,
              'execution_batch_size': 1, 'trainable_stages': [2, 3, 4],
              'coefficients': control.prior.COEFFICIENTS,
              'protected': {str(dependency): control.base.sha(dependency)}}
    payload = {'format': TRAIN_VERSION, 'identity': launch, 'cut_index': 1,
               'cut_updates': 5000, 'global_updates': 5000, 'accumulation': 12,
               'coefficients': control.prior.COEFFICIENTS, 'sources_seen': source_ids,
               'teacher_source_sha256': control.base.SOURCE_SHA256,
               'teacher_checkpoint_sha256': control.base.CHECKPOINT_SHA256}
    path = tmp_path / 'checkpoint-step5000.pt'
    torch.save(payload, path)
    checksum = control.base.sha(path)
    receipt = {'checkpoint_sha256': checksum, 'cut_updates': 5000, 'global_updates': 5000,
               'sources_seen': 60000, 'frozen_state_preserved': True}
    completed = {'version': TRAIN_VERSION, 'status': 'awaiting_review', 'failure': None,
                 'step': 5000, 'cut_updates': 5000, 'sources_seen': 60000,
                 'last_checkpoint_sha256': checksum, 'frozen_state_preserved': True,
                 'original_files_preserved': True}
    if damage == 'still_running': completed['status'] = 'running'
    elif damage == 'receipt': receipt['checkpoint_sha256'] = 'foreign'
    elif damage == 'source_order': source_ids = list(reversed(source_ids))
    elif damage == 'protected': dependency.write_text('changed input')
    for name, value in [('launch.json', launch), ('checkpoint-step5000.json', receipt),
                        ('completed.json', completed)]:
        (tmp_path / name).write_text(json.dumps(value))
    before = {str(p): control.base.sha(p) for p in tmp_path.iterdir()}
    if damage is None:
        result = control.final_run_guard(path, source_ids)
        assert result['step'] == 5000 and result['source_count'] == 60000
        assert result['checkpoint_sha256'] == checksum
    else:
        with pytest.raises(ValueError): control.final_run_guard(path, source_ids)
    assert {str(p): control.base.sha(p) for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('cold_grad_offset', [0., .01])
def test_warmup_retains_first_gradient_call_evidence_even_when_third_call_matches(monkeypatch, cold_grad_offset):
    calls = {'native': 0, 'nograd': 0, 'grad': 0}

    class WarmControl(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(.7))

        def group_from_input(self, z):
            mode = 'grad' if torch.is_grad_enabled() else 'nograd'
            calls[mode] += 1
            offset = cold_grad_offset if mode == 'grad' and calls[mode] == 1 else 0.
            return z * self.gain + offset

        def suffix_from_group(self, h):
            return h * 2

        def forward_from_latents(self, z):
            h = self.group_from_input(z)
            return {'group_output': h, 'waveform': self.suffix_from_group(h)}

    model = WarmControl().eval()
    model.gain.grad = torch.tensor(.25)
    teacher = TinyFrozenTeacher(nn.Identity()).eval()
    real_batch = control.base.batch
    monkeypatch.setattr(control.base, 'batch', lambda rows: real_batch(rows, device='cpu'))
    monkeypatch.setattr(control.base, 'teacher_forward', lambda t, z: {'group_input': z})
    def native(decoder, z):
        calls['native'] += 1
        return z
    monkeypatch.setattr(control, 'native_decoder_forward', native)
    crop = cached_crop()
    state, rng, grad = snapshot(model), torch.get_rng_state().clone(), model.gain.grad.clone()
    warmed, evidence = set(), []
    assert control.warm_shapes(teacher, model, [crop], warmed, evidence) == 13
    assert calls == {'native': 3, 'nograd': 3, 'grad': 3}
    report = evidence[0]
    assert report['all_exact'] is (cold_grad_offset == 0)
    assert report['parameter_updates'] == 0 and report['no_extra_forwards_for_comparison']
    compared = report['comparisons']
    for key, scale in [('group_output', 1), ('waveform', 2)]:
        assert compared['first_vs_third_nograd/' + key]['exact']
        assert compared['first_grad_vs_third_nograd/' + key]['max_abs'] == pytest.approx(cold_grad_offset * scale)
        assert compared['first_vs_third_grad/' + key]['max_abs'] == pytest.approx(cold_grad_offset * scale)
    assert_snapshot(model, state)
    assert model.training is False and teacher.training is False
    assert torch.equal(model.gain.grad, grad) and torch.equal(torch.get_rng_state(), rng)
    assert control.warm_shapes(teacher, model, [crop], warmed, evidence) == 0
    assert calls == {'native': 3, 'nograd': 3, 'grad': 3} and len(evidence) == 1
