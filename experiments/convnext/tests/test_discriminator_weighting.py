"""Duration weighting preserves loss ownership and the legacy default path."""
import pytest
import torch
from torch import nn

from audiovae_student.discriminators import (AudioDiscriminators, DiscriminatorConfig,
    DiscriminatorOutput, discriminator_loss, generator_losses)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_num_threads(previous)


def small_discriminators():
    return AudioDiscriminators(DiscriminatorConfig(periods=(2, 3),
        mpd_channels=(2, 4, 4, 4, 4), fft_sizes=(32, 64), mrd_channels=2))


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, audio):
        value = self.scale * audio
        # Replicating feature channels must not increase that layer's weight.
        return (DiscriminatorOutput(value, (value, value.expand(-1, 3, -1))),)


def test_none_path_retains_exact_existing_arithmetic():
    model = small_discriminators()
    teacher = torch.randn(2, 1, 256) * .05
    prediction = (torch.randn(2, 1, 256) * .05).requires_grad_()
    real, fake = model(teacher), model(prediction)
    expected_d = torch.stack([(r.logits - 1).square().mean() + (f.logits + 1).square().mean()
                              for r, f in zip(real, fake)]).mean()
    expected_adv = torch.stack([(h.logits - 1).square().mean() for h in fake]).mean()
    expected_fm = torch.stack([torch.stack([(f - r).abs().mean()
        for f, r in zip(fh.features, rh.features)]).mean() for fh, rh in zip(fake, real)]).mean()
    actual = generator_losses(model, prediction, teacher, example_weights=None)
    for a, b in ((discriminator_loss(model, prediction, teacher), expected_d),
                 (actual['adversarial'], expected_adv), (actual['feature_matching'], expected_fm)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_uniform_weights_match_legacy_values_and_generator_gradients():
    model = small_discriminators()
    teacher = torch.randn(3, 1, 256) * .05
    prediction = (torch.randn(3, 1, 256) * .05).requires_grad_()
    legacy = generator_losses(model, prediction, teacher)
    weighted = generator_losses(model, prediction, teacher, example_weights=torch.ones(3))
    for key in legacy:
        torch.testing.assert_close(legacy[key], weighted[key])
    torch.testing.assert_close(discriminator_loss(model, prediction, teacher),
        discriminator_loss(model, prediction, teacher, example_weights=torch.ones(3)))
    old_grad, = torch.autograd.grad(sum(legacy.values()), prediction, retain_graph=True)
    new_grad, = torch.autograd.grad(sum(weighted.values()), prediction)
    torch.testing.assert_close(old_grad, new_grad, rtol=2e-5, atol=1e-8)


def test_duration_weights_give_expected_audio_gradient_ratio_and_are_detached():
    model = ScaleDiscriminator()
    prediction = torch.full((2, 1, 16), 2., requires_grad=True)
    teacher = torch.full_like(prediction, 3., requires_grad=True)
    weights = torch.tensor([100., 300.], requires_grad=True)
    losses = generator_losses(model, prediction, teacher, example_weights=weights)
    fm_grad, = torch.autograd.grad(losses['feature_matching'], prediction, retain_graph=True)
    adv_grad, = torch.autograd.grad(losses['adversarial'], prediction, retain_graph=True)
    expected = torch.tensor([.25, .75]).view(2, 1, 1).expand_as(prediction) / 16
    torch.testing.assert_close(fm_grad, -expected, atol=0, rtol=0)
    torch.testing.assert_close(adv_grad, 2 * expected, atol=0, rtol=0)
    sum(losses.values()).backward()
    assert teacher.grad is None and weights.grad is None and model.scale.grad is None


def test_discriminator_weights_apply_to_real_and_fake_and_only_update_discriminator():
    model = ScaleDiscriminator(scale=.5)
    teacher = torch.tensor([3., 5.]).view(2, 1, 1).expand(2, 1, 16).requires_grad_()
    prediction = torch.tensor([2., 4.]).view(2, 1, 1).expand(2, 1, 16).requires_grad_()
    weights = torch.tensor([1., 3.], requires_grad=True)
    value = discriminator_loss(model, prediction, teacher, example_weights=weights)
    value.backward()
    # At scale=.5, the per-example derivatives are 11 and 39 respectively.
    torch.testing.assert_close(model.scale.grad, torch.tensor(.25 * 11 + .75 * 39), atol=0, rtol=0)
    assert teacher.grad is None and prediction.grad is None and weights.grad is None


def test_weighted_multihead_losses_equal_weighted_single_example_results():
    model = small_discriminators()
    teacher, prediction = torch.randn(3, 1, 256) * .05, torch.randn(3, 1, 256) * .05
    weights = torch.tensor([9120, 12000, 30000])
    result = generator_losses(model, prediction, teacher, example_weights=weights)
    individual = [generator_losses(model, prediction[i:i+1], teacher[i:i+1]) for i in range(3)]
    normalized = weights / weights.sum()
    for key in result:
        expected = sum(normalized[i] * item[key] for i, item in enumerate(individual))
        torch.testing.assert_close(result[key], expected, rtol=2e-5, atol=1e-7)
    expected_d = sum(normalized[i] * discriminator_loss(model, prediction[i:i+1], teacher[i:i+1])
                     for i in range(3))
    torch.testing.assert_close(discriminator_loss(model, prediction, teacher, example_weights=weights), expected_d)
    scaled = generator_losses(model, prediction, teacher, example_weights=weights * 5)
    for key in result:
        torch.testing.assert_close(result[key], scaled[key], atol=0, rtol=0)


@pytest.mark.parametrize('weights', [torch.ones(2, 1), torch.ones(3), torch.tensor([0., 1.]),
    torch.tensor([-1., 2.]), torch.tensor([float('nan'), 1.]), torch.tensor([float('inf'), 1.]),
    torch.tensor([1e308, 1e308], dtype=torch.float64), torch.tensor([1j, 2j])])
def test_invalid_weights_fail_before_discriminator_forward(weights):
    class NeverCalled(nn.Module):
        def forward(self, audio):
            raise AssertionError('invalid weights reached the discriminator')
    audio = torch.ones(2, 1, 16)
    for function in (generator_losses, discriminator_loss):
        with pytest.raises(ValueError, match='weights'):
            function(NeverCalled(), audio, audio, example_weights=weights)
