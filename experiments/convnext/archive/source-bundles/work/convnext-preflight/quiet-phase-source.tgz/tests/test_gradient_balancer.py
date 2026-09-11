import copy
import json

import pytest
import torch

from audiovae_student.gradient_balancer import (
    GradientBalancer, GradientBalancerConfig, MelGradientCapConfig,
)

torch.set_num_threads(1)


def test_equal_shares_cancel_opposing_gradients_despite_scale_difference():
    prediction = torch.zeros(2, 1, 16, requires_grad=True)
    balancer = GradientBalancer(weights={'first': 1, 'second': 1})
    losses = {'first': (prediction - 1).abs().mean(), 'second': 100 * (prediction + 1).abs().mean()}
    result = balancer.combine(losses, prediction)
    torch.testing.assert_close(result.gradient, torch.zeros_like(prediction), atol=1e-7, rtol=0)
    assert result.metrics['first/scaled_norm'] == pytest.approx(.5)
    assert result.metrics['second/scaled_norm'] == pytest.approx(.5)
    assert result.metrics['second/raw_norm'] == pytest.approx(100 * result.metrics['first/raw_norm'])


def test_valid_mask_excludes_history_tails_and_uses_per_example_norm():
    prediction = torch.zeros(2, 1, 8, requires_grad=True)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 0, 2:4] = True
    mask[1, 0, 3:6] = True
    balancer = GradientBalancer(weights={'only': 1})
    result = balancer.combine({'only': prediction.sum()}, prediction, valid_mask=mask)
    assert torch.count_nonzero(result.gradient[~mask]) == 0
    assert result.metrics['valid_samples_mean'] == 2.5
    assert result.metrics['only/raw_norm'] == pytest.approx((2**.5 + 3**.5) / 2)
    assert result.metrics['combined_norm'] == pytest.approx(1)
    prediction.backward(result.gradient)
    torch.testing.assert_close(prediction.grad, result.gradient)


def test_zero_norm_does_not_initialize_ema_or_amplify_noise():
    prediction = torch.zeros(1, 1, 8, requires_grad=True)
    balancer = GradientBalancer(weights={'only': 1})
    result = balancer.combine({'only': prediction.sum() * 0}, prediction)
    assert torch.count_nonzero(result.gradient) == 0
    assert result.metrics['only/scale'] == 0 and result.metrics['only/saturated'] == 0
    assert balancer.state_dict()['ema']['only'] == {'total': 0., 'weight': 0.}


def test_coefficient_caps_are_visible_and_finite():
    prediction = torch.zeros(1, 1, 8, requires_grad=True)
    cfg = GradientBalancerConfig(min_scale=.001, max_scale=10)
    balancer = GradientBalancer(cfg, weights={'small': 1, 'big': 1})
    result = balancer.combine({'small': prediction.sum() * 1e-8,
                               'big': prediction.sum() * 1e8}, prediction)
    assert result.metrics['small/scale'] == 10
    assert result.metrics['big/scale'] == .001
    assert result.metrics['small/saturated'] == result.metrics['big/saturated'] == 1
    assert torch.isfinite(result.gradient).all()


def _losses(prediction, multiplier):
    return {'teacher_waveform': (prediction - .3).square().mean(),
            'teacher_mel': multiplier * (prediction + .2).abs().mean(),
            'feature_matching': (prediction + .7).square().mean(),
            'adversarial': (prediction - .9).square().mean()}


def test_checkpoint_resume_restores_phase_and_exact_next_combination():
    torch.manual_seed(9)
    first = GradientBalancer()
    p = torch.randn(2, 1, 16, requires_grad=True)
    first.combine(_losses(p, 4), p)
    first.set_weights({'teacher_waveform': .3, 'teacher_mel': .4,
                       'feature_matching': .2, 'adversarial': .1})
    first.combine(_losses(p, 8), p)
    state = json.loads(json.dumps(first.state_dict()))
    resumed = GradientBalancer()
    resumed.load_state_dict(state)
    next_p = torch.randn(2, 1, 16, requires_grad=True)
    continued = first.combine(_losses(next_p, 2), next_p)
    replay = resumed.combine(_losses(next_p, 2), next_p)
    torch.testing.assert_close(continued.gradient, replay.gradient, rtol=0, atol=0)
    assert continued.metrics == replay.metrics and first.state_dict() == resumed.state_dict()
    state['ema']['teacher_mel']['total'] = 999
    assert resumed.state_dict()['ema']['teacher_mel']['total'] != 999


def test_invalid_inputs_and_checkpoints_do_not_mutate_valid_state():
    balancer = GradientBalancer(weights={'only': 1})
    before = copy.deepcopy(balancer.state_dict())
    p = torch.zeros(1, 1, 8, requires_grad=True)
    with pytest.raises(ValueError, match='finite'):
        balancer.combine({'only': p.mean() * float('nan')}, p)
    assert balancer.state_dict() == before
    bad = copy.deepcopy(before)
    bad['config']['max_scale'] = 123
    with pytest.raises(ValueError, match='config'):
        balancer.load_state_dict(bad)
    assert balancer.state_dict() == before
    with pytest.raises(ValueError, match='scored samples'):
        balancer.combine({'only': p.mean()}, p, valid_mask=torch.zeros_like(p, dtype=torch.bool))
    with pytest.raises(ValueError, match='loss-name'):
        balancer.set_weights({'other': 1})
    with pytest.raises(ValueError, match='every active'):
        balancer.combine({}, p)


def test_prefix_lengths_and_backward_wrapper():
    p = torch.ones(2, 1, 8, requires_grad=True)
    balancer = GradientBalancer(weights={'only': 1})
    metrics = balancer.backward({'only': p.sum()}, p, valid_lengths=torch.tensor([4, 6]))
    assert torch.count_nonzero(p.grad[0, :, 4:]) == 0
    assert torch.count_nonzero(p.grad[1, :, 6:]) == 0
    assert metrics['valid_samples_mean'] == 5


def _pair_losses(prediction, mel_multiplier=1., waveform_multiplier=1.):
    # Different directions make it possible to detect both sides of a transfer.
    return {
        'teacher_waveform': prediction[..., ::2].sum() * waveform_multiplier,
        'teacher_mel': prediction[..., 1::2].sum() * mel_multiplier,
    }


def _pair_balancer(config=GradientBalancerConfig(), *, cap=False):
    return GradientBalancer(config, {'teacher_waveform': .75, 'teacher_mel': .25},
                            mel_cap=MelGradientCapConfig() if cap else None)


def test_current_batch_cap_corrects_lagging_ema_and_transfers_norm():
    control = _pair_balancer()
    initial = torch.zeros(2, 1, 16, requires_grad=True)
    control.combine(_pair_losses(initial), initial)
    candidate = control.fork_with_mel_cap(MelGradientCapConfig())
    prediction = torch.zeros(2, 1, 16, requires_grad=True)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, :, 2:8] = True
    mask[1, :, 4:15] = True
    losses = _pair_losses(prediction, 100.)
    baseline = control.combine(losses, prediction, valid_mask=mask)
    capped = candidate.combine(losses, prediction, valid_mask=mask)
    metrics = capped.metrics
    assert metrics['mel_cap/share_before'] > .25
    assert metrics['mel_cap/share_after'] == pytest.approx(.25)
    assert metrics['mel_cap/applied'] == metrics['mel_cap/satisfied'] == 1
    assert metrics['mel_cap/total_norm_after'] == pytest.approx(metrics['mel_cap/total_norm_before'])
    assert (metrics['teacher_waveform/scaled_norm'] - baseline.metrics['teacher_waveform/scaled_norm']
            == pytest.approx(baseline.metrics['teacher_mel/scaled_norm'] - metrics['teacher_mel/scaled_norm']))
    assert candidate.state_dict()['ema'] == control.state_dict()['ema']
    components = []
    for name, loss in losses.items():
        grad, = torch.autograd.grad(loss, prediction, retain_graph=True)
        components.append(grad.masked_fill(~mask, 0) * metrics[f'{name}/scale'])
    measured_norms = [float(value.flatten(1).norm(dim=1).mean()) for value in components]
    assert measured_norms[1] / sum(measured_norms) == pytest.approx(.25, abs=1e-7)
    torch.testing.assert_close(capped.gradient, sum(components), atol=1e-7, rtol=1e-7)
    assert torch.count_nonzero(capped.gradient[~mask]) == 0
    assert not torch.equal(capped.gradient, baseline.gradient)


def test_cap_below_limit_leaves_combination_and_history_exact():
    control = _pair_balancer()
    p = torch.zeros(2, 1, 16, requires_grad=True)
    control.combine(_pair_losses(p), p)
    candidate = control.fork_with_mel_cap(MelGradientCapConfig())
    losses = _pair_losses(p, .01)
    baseline, capped = control.combine(losses, p), candidate.combine(losses, p)
    torch.testing.assert_close(capped.gradient, baseline.gradient, rtol=0, atol=0)
    assert capped.metrics['mel_cap/requested'] == capped.metrics['mel_cap/applied'] == 0
    assert capped.metrics['mel_cap/satisfied'] == 1
    assert control.state_dict()['ema'] == candidate.state_dict()['ema']


def test_cap_uses_total_active_norm_and_keeps_other_loss_scales_fixed():
    weights = {'teacher_waveform': .5, 'teacher_mel': .25, 'feature_matching': .25}
    control = GradientBalancer(weights=weights)
    p = torch.zeros(1, 1, 12, requires_grad=True)
    warm_losses = _pair_losses(p)
    warm_losses['feature_matching'] = p[..., :3].sum()
    control.combine(warm_losses, p)
    candidate = control.fork_with_mel_cap(MelGradientCapConfig())
    losses = _pair_losses(p, 100.)
    losses['feature_matching'] = p[..., :3].sum()
    baseline, capped = control.combine(losses, p), candidate.combine(losses, p)
    assert capped.metrics['mel_cap/applied'] == 1
    for key in ('scale', 'scaled_norm'):
        assert capped.metrics[f'feature_matching/{key}'] == baseline.metrics[f'feature_matching/{key}']
    assert capped.metrics['teacher_mel/achieved_share'] == pytest.approx(.25)
    assert sum(capped.metrics[f'{name}/achieved_share'] for name in weights) == pytest.approx(1)


@pytest.mark.parametrize('waveform_weight', [0., .75])
def test_zero_waveform_cannot_silently_satisfy_cap(waveform_weight):
    weights = {'teacher_waveform': waveform_weight, 'teacher_mel': .25}
    control = GradientBalancer(weights=weights)
    candidate = control.fork_with_mel_cap(MelGradientCapConfig())
    p = torch.zeros(1, 1, 8, requires_grad=True)
    losses = _pair_losses(p, waveform_multiplier=0.)
    baseline, capped = control.combine(losses, p), candidate.combine(losses, p)
    assert capped.metrics['mel_cap/zero_waveform'] == 1
    assert capped.metrics['mel_cap/feasible'] == capped.metrics['mel_cap/satisfied'] == 0
    assert capped.metrics['mel_cap/applied'] == 0
    assert capped.metrics['mel_cap/share_after'] == 1
    torch.testing.assert_close(capped.gradient, baseline.gradient, rtol=0, atol=0)
    assert candidate.state_dict()['ema'] == control.state_dict()['ema']


def test_zero_mel_and_zero_total_are_reported_without_spurious_signal():
    candidate = _pair_balancer(cap=True)
    p = torch.zeros(1, 1, 8, requires_grad=True)
    result = candidate.combine(_pair_losses(p, mel_multiplier=0.), p)
    assert result.metrics['mel_cap/zero_mel'] == result.metrics['mel_cap/share_defined'] == 1
    assert result.metrics['mel_cap/share_after'] == 0
    assert candidate.state_dict()['ema']['teacher_mel'] == {'total': 0., 'weight': 0.}
    all_zero = candidate.combine(_pair_losses(p, mel_multiplier=0., waveform_multiplier=0.), p)
    assert all_zero.metrics['mel_cap/zero_total'] == 1
    assert all_zero.metrics['mel_cap/share_defined'] == 0
    assert torch.count_nonzero(all_zero.gradient) == 0


@pytest.mark.parametrize('config, warm_mel, current_mel, exception', [
    (GradientBalancerConfig(min_scale=.1, max_scale=1e5), 1., 1000., 'mel'),
    (GradientBalancerConfig(min_scale=1e-6, max_scale=.4), 1., 100., 'waveform'),
])
def test_bound_exceptions_retain_original_combination(config, warm_mel, current_mel, exception):
    control = _pair_balancer(config)
    p = torch.zeros(1, 1, 8, requires_grad=True)
    control.combine(_pair_losses(p, warm_mel), p)
    candidate = control.fork_with_mel_cap(MelGradientCapConfig())
    losses = _pair_losses(p, current_mel)
    baseline, capped = control.combine(losses, p), candidate.combine(losses, p)
    assert capped.metrics[f'mel_cap/{exception}_bound_exception'] == 1
    assert capped.metrics['mel_cap/feasible'] == capped.metrics['mel_cap/satisfied'] == 0
    assert capped.metrics['mel_cap/applied'] == 0
    torch.testing.assert_close(capped.gradient, baseline.gradient, rtol=0, atol=0)
    assert candidate.state_dict()['ema'] == control.state_dict()['ema']


def test_version_one_checkpoint_shape_is_unchanged_and_fork_does_not_alias_it():
    original = _pair_balancer()
    p = torch.zeros(1, 1, 8, requires_grad=True)
    original.combine(_pair_losses(p), p)
    before = copy.deepcopy(original.state_dict())
    assert set(before) == {'format_version', 'config', 'weights', 'updates', 'ema'}
    assert before['format_version'] == 1
    assert set(before['config']) == {'ema_decay', 'epsilon', 'total_norm', 'min_scale', 'max_scale'}
    candidate = original.fork_with_mel_cap(MelGradientCapConfig())
    assert original.state_dict() == before
    assert candidate.state_dict()['format_version'] == 2
    assert candidate.state_dict()['mel_cap'] == {'max_share': .25, 'format_version': 1}
    assert candidate.updates == original.updates == 1
    candidate.combine(_pair_losses(p, 100.), p)
    candidate.set_weights({'teacher_waveform': .8, 'teacher_mel': .2})
    assert original.state_dict() == before


def test_capped_checkpoint_resume_is_exact_and_policy_changes_require_explicit_fork():
    original = _pair_balancer(cap=True)
    p = torch.zeros(2, 1, 8, requires_grad=True)
    original.combine(_pair_losses(p), p)
    state = json.loads(json.dumps(original.state_dict()))
    resumed = _pair_balancer(cap=True)
    resumed.load_state_dict(state)
    losses = _pair_losses(p, 100.)
    continued, replay = original.combine(losses, p), resumed.combine(losses, p)
    torch.testing.assert_close(continued.gradient, replay.gradient, atol=0, rtol=0)
    assert continued.metrics == replay.metrics
    assert original.state_dict() == resumed.state_dict()
    uncapped = _pair_balancer()
    different = uncapped.fork_with_mel_cap(MelGradientCapConfig(max_share=.2))
    for receiver, incompatible in ((uncapped, state), (different, state),
                                   (resumed, uncapped.state_dict())):
        before = copy.deepcopy(receiver.state_dict())
        with pytest.raises(ValueError, match='format/config'):
            receiver.load_state_dict(incompatible)
        assert receiver.state_dict() == before
    assert resumed.fork_with_mel_cap(None).state_dict()['format_version'] == 1


@pytest.mark.parametrize('share', [0., 1., -.1, float('nan'), float('inf')])
def test_invalid_mel_cap_shares_are_rejected(share):
    with pytest.raises(ValueError, match='share'):
        MelGradientCapConfig(max_share=share)


def test_cap_requires_named_losses_and_supported_policy_version():
    with pytest.raises(ValueError, match='requires'):
        GradientBalancer(weights={'only': 1}, mel_cap=MelGradientCapConfig())
    with pytest.raises(ValueError, match='version'):
        MelGradientCapConfig(format_version=2)
