"""Independent DAC/audiotools conventions and singleton accumulation checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from author_mel import AuthorMelConfig, AuthorMelLoss, center_reflect, periodic_hann, slaney_mel_bank


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def reference():
    path = HERE / "data/author-mel-reference.npz"
    receipt = json.loads((HERE / "data/author-mel-reference.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["archive_sha256"]
    assert receipt["archive_sha256"] == "ed96d468748ec75a2b7c7408c91c2a20baf8f77179103c83fd3a1c700d4b5072"
    with np.load(path) as archive:
        arrays = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    for row in receipt["constants"]:
        for kind in ("window", "mel"):
            a = arrays[f"{kind}_{row['fft']}"]
            assert hashlib.sha256(a.numpy().tobytes()).hexdigest() == row[f"{kind}_sha256"]
    return arrays


def audio(length, seed, batch=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, length, generator=generator) * .05


def independent_terms(prediction, target, reference, config=AuthorMelConfig()):
    """Direct upstream expression using independently generated constants."""
    sums, counts = [], []
    for size in config.fft_sizes:
        window = reference[f"window_{size}"]
        bank = reference[f"mel_{size}"]
        spectra = []
        for value in (prediction, target.detach()):
            stft = torch.stft(value[:, 0], n_fft=size, hop_length=size//4,
                             win_length=size, window=window, center=True,
                             pad_mode="reflect", normalized=False,
                             onesided=True, return_complex=True)
            mel = torch.matmul(stft.abs().transpose(1, 2), bank.T).transpose(1, 2)
            spectra.append(torch.log10(torch.clamp(mel, min=1e-5)))
        difference = (spectra[0]-spectra[1]).abs()
        sums.append(difference.sum())
        counts.append(difference.numel())
    return sums, counts


def test_constants_match_pinned_librosa_and_scipy_bitwise(reference):
    loss = AuthorMelLoss()
    for size, bands in zip(loss.config.fft_sizes, loss.config.mel_bands):
        assert torch.equal(periodic_hann(size), reference[f"window_{size}"])
        assert torch.equal(slaney_mel_bank(size, bands, 48000), reference[f"mel_{size}"])
    assert loss.provenance["empty_mel_rows"] == [[0], [0], [], [], [], [], []]
    assert loss.provenance["center"] is True
    assert loss.provenance["match_stride"] is False
    assert loss.provenance["resolution_reduction"] == "sum"


def test_all_seven_native_losses_and_gradients_match_direct_upstream_formula(reference):
    pred = audio(2601, 1, batch=2).requires_grad_()
    target = audio(2601, 2, batch=2).requires_grad_()
    objective = AuthorMelLoss()
    result = objective(pred, target)
    sums, counts = independent_terms(pred, target, reference)
    expected = sum(x/n for x, n in zip(sums, counts))
    torch.testing.assert_close(result.losses["teacher_mel"], expected, atol=0, rtol=0)
    got_grad = torch.autograd.grad(result.losses["total"], pred, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(got_grad, expected_grad, atol=0, rtol=0)
    assert result.counts["mel_elements_by_resolution"] == tuple(counts)
    assert result.counts["mel_frames_by_resolution"] == tuple(2*(2601//(n//4)+1) for n in objective.config.fft_sizes)
    assert target.grad is None
    assert not torch.isclose(expected, expected/7)


def test_centered_reflection_is_part_of_objective_not_center_false(reference):
    pred = torch.zeros(1, 1, 2049)
    pred[0, 0, 0], pred[0, 0, -1] = .7, -.3
    target = torch.zeros_like(pred)
    loss = AuthorMelLoss()
    terms = loss.group_terms(pred, target)
    size, bands = loss.config.fft_sizes[-1], loss.config.mel_bands[-1]
    assert terms.mel_element_counts[-1] == bands*(2049//512+1)
    stft = torch.stft(pred[:, 0], n_fft=size, hop_length=512,
                     window=reference[f"window_{size}"], center=False,
                     return_complex=True)
    assert stft.shape[-1] == 1
    assert loss(pred, target).losses["total"].item() > 0


def test_singleton_accumulation_uses_global_per_scale_counts_on_unequal_lengths(reference):
    lengths = [1401, 2048, 3077]
    predictions = [audio(n, 10+i).requires_grad_() for i, n in enumerate(lengths)]
    targets = [audio(n, 20+i) for i, n in enumerate(lengths)]
    objective = AuthorMelLoss()
    counts = tuple(sum(objective.element_counts(n)[i] for n in lengths) for i in range(7))
    for pred, target in zip(predictions, targets):
        objective.loss_from_terms(objective.group_terms(pred, target), counts).backward()
    actual_gradients = [p.grad.clone() for p in predictions]
    oracle_predictions = [p.detach().clone().requires_grad_() for p in predictions]
    reference_terms = [independent_terms(p, t, reference) for p, t in zip(oracle_predictions, targets)]
    expected = sum(sum(sums[i] for sums, _ in reference_terms)/counts[i] for i in range(7))
    expected.backward()
    for got, pred in zip(actual_gradients, oracle_predictions):
        # Overlapping reflections may sum three contributions in a different
        # FP32 order than native CPU reflection backward. Bound only that
        # arithmetic rounding; the loss and reflected forward samples are exact.
        rounding = 2*torch.finfo(got.dtype).eps*float(pred.grad.abs().max())
        torch.testing.assert_close(got, pred.grad, atol=rounding, rtol=0)
    combined = objective.aggregate([objective.group_terms(p.detach(), t) for p,t in zip(predictions,targets)])
    torch.testing.assert_close(combined.losses["total"], expected, atol=0, rtol=0)
    assert combined.counts["valid_samples"] == sum(lengths)
    assert combined.counts["examples"] == 3
    # An unweighted average of source losses has different unequal-length weighting.
    naive = sum(objective(p.detach(), t).losses["total"] for p,t in zip(predictions,targets))/3
    assert abs(float((naive-expected).detach())) > 1e-5


@pytest.mark.parametrize("length,padding", [(9,0),(9,1),(9,4),(9,8),(1401,1024)])
def test_explicit_reflection_matches_native_samples_and_analytic_gradient(length,padding):
    p = torch.arange(length, dtype=torch.float64).reshape(1,-1).requires_grad_()
    actual = center_reflect(p,padding)
    expected = torch.nn.functional.pad(p,(padding,padding),mode="reflect")
    assert torch.equal(actual,expected)
    # Integer weights make every derivative sum exact, including the case in
    # which both reflected sides map back to the same original sample.
    weights = torch.arange(actual.numel(),dtype=torch.float64).reshape_as(actual)
    gradient = torch.autograd.grad((actual*weights).sum(),p)[0]
    oracle = torch.zeros_like(p)
    index_map = list(range(padding,0,-1))+list(range(length))+list(range(length-2,length-padding-2,-1))
    for padded_index, original_index in enumerate(index_map):
        oracle[0,original_index] += weights[0,padded_index]
    assert torch.equal(gradient,oracle)
    native_gradient = torch.autograd.grad((expected*weights).sum(),p)[0]
    assert torch.equal(gradient,native_gradient)


def test_objective_keeps_strict_determinism_and_never_calls_native_reflection(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    native_pad = torch.nn.functional.pad

    def reject_native_reflection(input,pad,mode="constant",value=None):
        if mode == "reflect":
            raise AssertionError("Native reflection would reintroduce unsupported CUDA backward")
        return native_pad(input,pad,mode,value)

    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        monkeypatch.setattr(torch.nn.functional,"pad",reject_native_reflection)
        loss = AuthorMelLoss()
        gradients = []
        for _ in range(2):
            p = audio(1401,80).requires_grad_()
            loss(p,audio(1401,81)).losses["total"].backward()
            gradients.append(p.grad.clone())
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.equal(gradients[0],gradients[1])
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


@pytest.mark.parametrize("silence", [False, True])
def test_exact_teacher_has_zero_loss_and_gradient_without_teacher_grad(silence):
    target = (torch.zeros(1,1,1201) if silence else audio(1201, 30)).requires_grad_()
    pred = target.detach().clone().requires_grad_()
    result = AuthorMelLoss()(pred,target)
    assert result.losses["total"].item() == 0
    result.losses["total"].backward()
    assert torch.equal(pred.grad,torch.zeros_like(pred))
    assert target.grad is None


def test_magnitude_floor_and_no_linear_branch_make_subfloor_audio_equal():
    target = torch.zeros(1,1,1501)
    pred = torch.full_like(target,1e-12).requires_grad_()
    result = AuthorMelLoss()(pred,target)
    assert result.losses["total"].item() == 0
    result.losses["total"].backward()
    assert torch.equal(pred.grad,torch.zeros_like(pred))


@pytest.mark.parametrize("length", [0, 1, 1023, 1024])
def test_rejects_short_crops_instead_of_silently_altering_native_padding(length):
    with pytest.raises(ValueError, match="reflect padding"):
        AuthorMelLoss().element_counts(length)


def test_native_minimum_reflection_length_works():
    loss = AuthorMelLoss()
    p = audio(1025,40).requires_grad_()
    result = loss(p,audio(1025,41))
    assert torch.isfinite(result.losses["total"])
    result.losses["total"].backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("bad", [torch.zeros(1500), torch.zeros(1,2,1500), torch.zeros(0,1,1500),
                                 torch.zeros(1,1,1500,dtype=torch.int32),
                                 torch.full((1,1,1500),float("nan")),
                                 torch.full((1,1,1500),float("inf"))])
def test_invalid_audio_is_rejected(bad):
    with pytest.raises(ValueError):
        AuthorMelLoss().group_terms(bad,torch.zeros(1,1,1500))


def test_alignment_and_denominators_are_strict():
    loss = AuthorMelLoss()
    p,t = audio(1500,50),audio(1500,51)
    with pytest.raises(ValueError,match="aligned"):
        loss.group_terms(p,t[:,:,:-1])
    terms = loss.group_terms(p,t)
    with pytest.raises(ValueError,match="denominators"):
        loss.loss_from_terms(terms,tuple(n-1 for n in terms.mel_element_counts))
    with pytest.raises(ValueError,match="denominators"):
        loss.loss_from_terms(terms,terms.mel_element_counts[:-1])
    with pytest.raises(ValueError,match="nonempty"):
        loss.aggregate([])
    with pytest.raises(ValueError,match="Matching"):
        AuthorMelLoss(AuthorMelConfig(fft_sizes=(32,),mel_bands=(5,))).aggregate(terms)


@pytest.mark.parametrize("settings", [dict(fft_sizes=()),dict(fft_sizes=(31,),mel_bands=(5,)),
                                     dict(mel_bands=(5,)),dict(log_epsilon=0),dict(log_epsilon=float('nan')),
                                     dict(sample_rate=0),dict(format_version=True)])
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        AuthorMelConfig(**settings)
