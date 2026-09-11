"""MPD reflection equivalence and strict deterministic backward coverage."""
import pytest
import torch
from torch.nn import functional as F

from audiovae_student.discriminators import (AudioDiscriminators, DiscriminatorConfig,
    DiscriminatorOutput, PeriodDiscriminator, discriminator_loss, generator_losses)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(59)
    yield
    torch.set_num_threads(previous)


def torch_reflection_reference(model, audio):
    """Historical torch reflection implementation, used only for parity."""
    pad = (-audio.shape[-1]) % model.period
    if pad:
        audio = F.pad(audio, (0, pad), mode='reflect')
    value = audio.reshape(audio.shape[0], 1, -1, model.period)
    features = []
    for layer in model.layers:
        value = F.leaky_relu(layer(value), .1)
        features.append(value)
    return DiscriminatorOutput(model.output(value), tuple(features))


@pytest.mark.parametrize('period', [2, 3, 5, 7, 11])
@pytest.mark.parametrize('samples', [17, 9120])
def test_period_forward_and_input_gradient_equal_torch_reflection(period, samples):
    model = PeriodDiscriminator(period, (2, 4, 4, 4, 4))
    audio = torch.randn(2, 1, samples, requires_grad=True)
    seen = []
    hook = model.layers[0].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    try:
        actual = model(audio)
        reference = torch_reflection_reference(model, audio)
    finally:
        hook.remove()
    # Check the reflected samples themselves, not only their processed output.
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    actual_outputs = (actual.logits, *actual.features)
    reference_outputs = (reference.logits, *reference.features)
    for a, b in zip(actual_outputs, reference_outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual_loss = sum(value.square().mean() for value in actual_outputs)
    reference_loss = sum(value.square().mean() for value in reference_outputs)
    actual_gradient, = torch.autograd.grad(actual_loss, audio, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference_loss, audio)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)


def test_reflection_rejects_input_shorter_than_required_pad():
    model = PeriodDiscriminator(11, (2, 4, 4, 4, 4))
    with pytest.raises(ValueError, match='too short for MPD reflection'):
        model(torch.ones(1, 1, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA for strict determinism qualification')
def test_cuda_discriminator_and_generator_backward_with_strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        # Periods 7 and 11 need reflected tails for the production crop length.
        model = AudioDiscriminators(DiscriminatorConfig(periods=(2, 3, 5, 7, 11),
            mpd_channels=(2, 4, 4, 4, 4), fft_sizes=(128,), mrd_channels=2)).cuda()
        teacher = (torch.randn(2, 1, 9120, device='cuda') * .03).requires_grad_()
        prediction = (torch.randn_like(teacher) * .03).requires_grad_()
        weights = torch.tensor([9120, 18000], device='cuda')
        disc = discriminator_loss(model, prediction, teacher, example_weights=weights)
        disc.backward()
        assert prediction.grad is None and teacher.grad is None
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        model.zero_grad(set_to_none=True)
        losses = generator_losses(model, prediction, teacher, example_weights=weights)
        sum(losses.values()).backward()
        assert torch.isfinite(prediction.grad).all() and prediction.grad.norm() > 0
        assert teacher.grad is None
        assert all(parameter.grad is None for parameter in model.parameters())
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
