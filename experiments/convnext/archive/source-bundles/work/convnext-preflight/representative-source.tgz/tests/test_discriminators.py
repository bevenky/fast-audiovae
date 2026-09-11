from dataclasses import asdict
import json

import pytest
import torch
from torch import nn

from audiovae_student.discriminators import (
    AudioDiscriminators, DiscriminatorConfig, DiscriminatorOutput,
    discriminator_loss, frozen_parameters, generator_losses,
)

torch.set_num_threads(1)


def small_discriminators():
    return AudioDiscriminators(DiscriminatorConfig(periods=(2, 3), mpd_channels=(2, 4, 8, 8, 8),
                                                    fft_sizes=(32, 64), mrd_channels=2))


def test_published_dimensions_are_complete():
    model = AudioDiscriminators()
    assert [d.period for d in model.periods] == [2, 3, 5, 7, 11]
    assert [d.n_fft for d in model.resolutions] == [512, 1024, 2048]
    assert [layer.out_channels for layer in model.periods[0].layers] == [16, 64, 256, 512, 512]
    assert [layer.out_channels for layer in model.resolutions[0].layers] == [16] * 5
    assert model.periods[0].output.out_channels == model.resolutions[0].output.out_channels == 1
    assert [layer.stride for layer in model.resolutions[0].layers] == [(1, 1), (2, 1), (2, 1), (2, 1), (1, 1)]
    assert all(hasattr(layer, 'parametrizations') for layer in model.periods[0].layers)
    cfg = model.config
    assert DiscriminatorConfig(**json.loads(json.dumps(asdict(cfg)))) == cfg


def test_outputs_five_hidden_maps_and_identical_pair_matching():
    torch.manual_seed(3)
    model = small_discriminators()
    audio = torch.randn(2, 1, 256) * .1
    output = model(audio)
    assert len(output) == 4
    assert all(len(head.features) == 5 and torch.isfinite(head.logits).all() for head in output)
    losses = generator_losses(model, audio.requires_grad_(), audio.detach())
    assert losses['feature_matching'].item() == 0
    assert losses['adversarial'].item() > 0


def test_discriminator_then_generator_updates_keep_gradient_routes_separate():
    torch.manual_seed(4)
    model = small_discriminators()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    prediction = (torch.randn(2, 1, 256) * .1).requires_grad_()
    teacher = (torch.randn(2, 1, 256) * .1).requires_grad_()
    loss = discriminator_loss(model, prediction, teacher)
    loss.backward()
    assert prediction.grad is None and teacher.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original_flags = [p.requires_grad for p in model.parameters()]
    losses = generator_losses(model, prediction, teacher)
    assert [p.requires_grad for p in model.parameters()] == original_flags
    (losses['adversarial'] + losses['feature_matching']).backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0 and teacher.grad is None
    assert all(p.grad is None for p in model.parameters())


class IdentityHeads(nn.Module):
    def forward(self, audio):
        return (DiscriminatorOutput(audio, (audio,)),
                DiscriminatorOutput(audio.expand(-1, 3, -1), (audio.expand(-1, 7, -1),)))


def test_least_squares_labels_and_element_head_means():
    model = IdentityHeads()
    teacher = torch.full((1, 1, 8), 3.)
    prediction = torch.full((1, 1, 8), 2., requires_grad=True)
    # (real-1)^2 + (fake+1)^2 = 4+9, not the fake-zero variant.
    assert discriminator_loss(model, prediction, teacher) == 13
    losses = generator_losses(model, prediction, teacher)
    assert losses['adversarial'] == 1 and losses['feature_matching'] == 1


def test_frozen_flags_restore_on_error_and_invalid_crops_fail():
    model = small_discriminators()
    next(model.parameters()).requires_grad_(False)
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError, match='fixture'):
        with frozen_parameters(model):
            assert not any(p.requires_grad for p in model.parameters())
            raise RuntimeError('fixture')
    assert [p.requires_grad for p in model.parameters()] == flags
    with pytest.raises(ValueError, match='too short'):
        model(torch.zeros(1, 1, 16))
    with pytest.raises(ValueError, match='aligned'):
        generator_losses(model, torch.zeros(1, 1, 256), torch.zeros(1, 1, 128))
