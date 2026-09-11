import copy
import json

import pytest
import torch

from audiovae_student.gradient_balancer import GradientBalancer, GradientBalancerConfig

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
