"""Frozen-waveform diagnostic, never a student or optimizer update.

Use FrozenWaveformGradientProbe(engine_state, device="cuda").probe(crop, prediction).
Prediction must be the detached full-context waveform for the canonical crop.
Negative directional derivatives mean infinitesimal improvement of the named
metric in waveform space. They do not predict a Muon/AdamW parameter update.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig, generator_losses
from audiovae_student.gradient_balancer import (
    GradientBalancer, GradientBalancerConfig, MelGradientCapConfig,
)
from audiovae_student.quiet_audio import QuietAudioConfig, _window_values
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config


NAMES = ("teacher_waveform", "teacher_mel", "feature_matching", "adversarial")
VIEW_SAMPLES = 9120


def _fresh_balancer(saved):
    saved = deepcopy(saved)
    cap = MelGradientCapConfig(**saved["mel_cap"]) if saved.get("mel_cap") else None
    result = GradientBalancer(GradientBalancerConfig(**saved["config"]), saved["weights"], mel_cap=cap)
    result.load_state_dict(saved)
    return result


def _quiet_mask(prediction, teacher, valid, config):
    # Classify on the original tensor grid before selecting scored samples.
    with torch.no_grad():
        _, _, _, _, windows = _window_values(prediction, teacher, valid, config)
        expanded = windows.repeat_interleave(config.window_samples, dim=-1).unsqueeze(1)
        return expanded[..., :prediction.shape[-1]] & valid


def _metric_gradients(prediction, teacher, valid, quiet):
    count = int(valid.sum())
    excess = (prediction.abs() - 1).clamp_min(0).masked_fill(~valid, 0)
    peak_gradient = 2 * excess * prediction.sign() / count
    quiet_count = int(quiet.sum())
    quiet_gradient = (2 * (prediction - teacher).masked_fill(~quiet, 0) / quiet_count
                      if quiet_count else None)
    return {
        "peak_excess_mse": (float(excess.double().square().sum() / count), peak_gradient),
        "teacher_quiet_residual_mse": (
            float((prediction - teacher)[quiet].double().square().mean()) if quiet_count else None,
            quiet_gradient),
    }


def _direction_report(gradient, metrics):
    """First derivative of M(p - epsilon*g) at epsilon=0, not a finite step."""
    norm = float(gradient.double().norm())
    result = {"loss_gradient_l2": norm, "zero_loss_gradient": norm == 0, "metrics": {}}
    for name, (value, metric_gradient) in metrics.items():
        if metric_gradient is None:
            result["metrics"][name] = {"value": None, "defined": False,
                "reason": "no teacher-quiet scored samples", "directional_derivative": None,
                "unit_l2_directional_derivative": None, "gradient_cosine": None}
            continue
        metric_norm = float(metric_gradient.double().norm())
        dot = float((metric_gradient.double() * gradient.double()).sum())
        derivative = -dot
        result["metrics"][name] = {"value": value, "defined": True,
            "metric_gradient_l2": metric_norm, "zero_metric_gradient": metric_norm == 0,
            "directional_derivative": derivative,
            "unit_l2_directional_derivative": derivative / norm if norm else None,
            "gradient_cosine": dot / (metric_norm * norm) if metric_norm and norm else None,
            "infinitesimal_effect": ("decrease" if derivative < 0 else "increase"
                                     if derivative > 0 else "zero_first_derivative")}
    return result


class FrozenWaveformGradientProbe:
    """Restore only copied loss components, not a student or either optimizer.

    The checkpoint discriminator is held fixed. Actual training updates D before
    its generator loss; this diagnostic deliberately omits that update. Each
    crop/view gets an independently restored balancer and one hypothetical EMA
    observation. Consequently views do not contaminate each other's history.
    """

    def __init__(self, engine_state: dict, *, device="cuda"):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Use an explicit CPU or CUDA diagnostic device")
        if engine_state.get("fusion_variant") not in (None, "control", "fresh_magnitude"):
            raise ValueError("This probe requires the retained magnitude-discriminator recipe")
        if engine_state["recipe"]["adversarial_samples"] != VIEW_SAMPLES:
            raise ValueError("The diagnostic contract requires the checkpoint's 9120-sample view")
        if engine_state.get("perceptual_start") is None:
            raise ValueError("The diagnostic requires an active perceptual stage")
        self.step = int(engine_state["step"])
        self.balancer_state = deepcopy(engine_state["balancer"])
        if set(self.balancer_state["weights"]) != set(NAMES):
            raise ValueError("Unexpected loss identities")
        self.quiet_config = QuietAudioConfig(**engine_state["config"]["quiet_audio"])
        self.reconstruction_config = ReconstructionV2Config(**engine_state["reconstruction_v2"])
        self.discriminator_config = DiscriminatorConfig(**engine_state["discriminator_config"])
        # Initialization happens on CPU and restores CPU RNG. Loading copies the
        # checkpoint tensors into fresh parameters; no source tensors are shared.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.reconstruction = ReconstructionV2(self.reconstruction_config)
            self.discriminators = AudioDiscriminators(self.discriminator_config)
            self.discriminators.load_state_dict(engine_state["discriminators"], strict=True)
        self.reconstruction.to(self.device).eval()
        self.discriminators.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode(False)
    def probe(self, crop, prediction, *, include_first_view=True):
        _validate_crop(crop)
        if (prediction.shape != crop.teacher_audio.shape or prediction.dtype != torch.float32
                or prediction.requires_grad or prediction.grad_fn is not None):
            raise ValueError("Supply detached FP32 full-context crop prediction")
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Prediction contains nonfinite samples")
        copied_versions = tuple(t._version for t in self.discriminators.state_dict().values())
        p0 = prediction.detach().to(self.device).clone()
        teacher = crop.teacher_audio.detach().to(self.device).clone()
        if not bool(torch.isfinite(teacher).all()):
            raise ValueError("Teacher contains nonfinite samples")
        start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
        stop = crop.scored_slice.stop
        if stop - start < max(VIEW_SAMPLES, max(self.reconstruction_config.fft_sizes)):
            raise ValueError("Full valid scored region is too short")
        valid = torch.zeros_like(p0, dtype=torch.bool)
        valid[..., start:stop] = True
        quiet = _quiet_mask(p0, teacher, valid, self.quiet_config)
        metrics = _metric_gradients(p0, teacher, valid, quiet)
        has_peak = bool(((p0.abs() > 1) & valid).any())
        if has_peak:
            selection = (p0.abs() - 1).clamp_min(0).masked_fill(~valid, -1)
            reason = "largest_scored_full_scale_excess"
        elif bool(quiet.any()):
            selection = (p0 - teacher).square().masked_fill(~quiet, -1)
            reason = "largest_scored_error_in_teacher_quiet_window"
        else:
            selection = (p0 - teacher).square().masked_fill(~valid, -1)
            reason = "no_peak_or_quiet_samples_largest_scored_residual"
        center = int(selection.flatten().argmax())
        view_start = max(start, min(center - VIEW_SAMPLES // 2, stop - VIEW_SAMPLES))
        views = [("event_centered", view_start)]
        if include_first_view and view_start != start:
            views.append(("first_scored", start))
        reports = []
        with torch.enable_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            for view_name, offset in views:
                p = p0.clone().requires_grad_(True)
                terms = self.reconstruction(p[..., start:stop], teacher[..., start:stop]).losses
                losses = {name: terms[name] for name in NAMES[:2]}
                losses.update(generator_losses(self.discriminators,
                    p[..., offset:offset + VIEW_SAMPLES], teacher[..., offset:offset + VIEW_SAMPLES]))
                gradients = {name: torch.autograd.grad(losses[name], p, retain_graph=True)[0]
                             .detach().float().masked_fill(~valid, 0) for name in NAMES}
                if not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
                    raise FloatingPointError("Nonfinite loss gradient")
                balancer = _fresh_balancer(self.balancer_state)
                combined = balancer.combine(losses, p, valid_mask=valid)
                individual = {name: _direction_report(gradients[name], metrics) for name in NAMES}
                for name in NAMES:
                    scale = combined.metrics.get(name + "/scale", 0.0)
                    individual[name]["balance_scale"] = scale
                    individual[name]["balanced_component"] = _direction_report(gradients[name] * scale, metrics)
                reconstructed = sum(gradients[name] * combined.metrics.get(name + "/scale", 0.0)
                                    for name in NAMES)
                discrepancy = float((reconstructed - combined.gradient).abs().max())
                if not torch.allclose(reconstructed, combined.gradient, atol=2e-6, rtol=2e-5):
                    raise RuntimeError("Component scales do not reconstruct the combined gradient")
                pairwise = {}
                for i, left in enumerate(NAMES):
                    for right in NAMES[i + 1:]:
                        a, b = gradients[left].double(), gradients[right].double()
                        denominator = float(a.norm() * b.norm())
                        pairwise[left + "__" + right] = float((a * b).sum()) / denominator if denominator else None
                reports.append({"view": view_name, "perceptual_start_in_crop": offset,
                    "perceptual_stop_in_crop": offset + VIEW_SAMPLES,
                    "perceptual_absolute_start_sample": crop.context_start_frame * 1920 + offset,
                    "perceptual_contains_selected_sample": offset <= center < offset + VIEW_SAMPLES,
                    "quiet_samples_in_perceptual_view": int(quiet[..., offset:offset + VIEW_SAMPLES].sum()),
                    "overshoot_samples_in_perceptual_view": int((p0[..., offset:offset + VIEW_SAMPLES].abs() > 1).sum()),
                    "loss_values": {name: float(value.detach()) for name, value in losses.items()},
                    "individual": individual, "combined": _direction_report(combined.gradient, metrics),
                    "loss_gradient_cosines": pairwise, "balancer_metrics": combined.metrics,
                    "combined_reconstruction_max_error": discrepancy,
                    "copied_balancer_updates_before": self.balancer_state["updates"],
                    "copied_balancer_updates_after": balancer.updates})
        if (not torch.equal(prediction.detach().to(self.device), p0)
                or not torch.equal(crop.teacher_audio.detach().to(self.device), teacher)
                or copied_versions != tuple(t._version for t in self.discriminators.state_dict().values())
                or any(p.grad is not None for p in self.discriminators.parameters())):
            raise RuntimeError("A frozen source or discriminator changed")
        return {"format_version": 1, "source_id": crop.source_id, "start_frame": crop.start_frame,
            "engine_step": self.step, "diagnostic_space": "detached_waveform_only",
            "interpretation": "Derivative of metric(p-epsilon*loss_gradient) at epsilon=0; negative improves locally. No finite update, student Jacobian, parameter optimizer, or discriminator update is included.",
            "view_policy": "Fixed diagnostic event-centered and optional first view; not replay of the training crop RNG",
            "balancer_policy": "Checkpoint weights and EMA, independently copied for one hypothetical single-crop observation per view; not the actual batched training gradient",
            "valid_start_in_crop": start, "valid_stop_in_crop": stop, "valid_samples": stop - start,
            "context_excluded_samples": 6 if crop.context_start_frame > 0 else 0,
            "quiet_samples": int(quiet.sum()), "quiet_config": asdict(self.quiet_config),
            "reconstruction_config": asdict(self.reconstruction_config),
            "discriminator_config": asdict(self.discriminator_config),
            "checkpoint_loss_weights": deepcopy(self.balancer_state["weights"]),
            "selection_reason": reason, "selected_sample_in_crop": center,
            "selected_absolute_sample": crop.context_start_frame * 1920 + center,
            "views": reports, "source_tensors_preserved": True,
            "discriminator_parameters_and_buffers_preserved": True,
            "student_or_optimizer_updates": 0}


def self_test():
    """Tiny CPU fixtures only: signs, undefined cases, masks, copied state."""
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.objective_comparison import state_fingerprint
    torch.set_num_threads(1)
    dc = DiscriminatorConfig(periods=(2,), mpd_channels=(1, 1, 1, 1, 1), fft_sizes=(64,), mrd_channels=1)
    rc = ReconstructionV2Config(fft_sizes=(64, 128), mel_bands=(8, 16))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        bank = AudioDiscriminators(dc)
    balancer = GradientBalancer(weights={"teacher_waveform": .3, "teacher_mel": .4,
                                        "feature_matching": .2, "adversarial": .1})
    state = {"step": 8890, "recipe": {"adversarial_samples": VIEW_SAMPLES}, "perceptual_start": 500,
             "balancer": balancer.state_dict(), "discriminator_config": asdict(dc),
             "discriminators": bank.state_dict(), "reconstruction_v2": asdict(rc),
             "config": {"quiet_audio": asdict(QuietAudioConfig())}}
    target = torch.full((1, 1, 19200), .1)
    target[..., :1920] = 0
    prediction = target.clone(); prediction[..., :1920] = .003; prediction[..., 15000] = 1.2
    crop = TrainingCrop(torch.zeros(1, 64, 10), target, None, "fixture", "fixture", 0, 0, 0, 10, 19200)
    before = state_fingerprint(state); rng = torch.get_rng_state().clone()
    probe = FrozenWaveformGradientProbe(state, device="cpu")
    report = probe.probe(crop, prediction)
    assert torch.equal(rng, torch.get_rng_state()) and state_fingerprint(state) == before
    assert len(report["views"]) == 2 and report["quiet_samples"] == 1920
    assert report["selected_sample_in_crop"] == 15000
    for view in report["views"]:
        wave = view["individual"]["teacher_waveform"]["metrics"]
        assert all(wave[k]["directional_derivative"] < 0 for k in wave)
        assert view["copied_balancer_updates_after"] == view["copied_balancer_updates_before"] + 1
    quiet = torch.zeros_like(target, dtype=torch.bool)
    valid = torch.ones_like(quiet)
    undefined = _direction_report(torch.ones_like(target), _metric_gradients(target, target, valid, quiet))
    assert undefined["metrics"]["teacher_quiet_residual_mse"]["defined"] is False
    assert undefined["metrics"]["peak_excess_mse"]["zero_metric_gradient"]
    # Context exclusion and teacher-grid classification must retain partial windows.
    valid[..., :966] = False
    qm = _quiet_mask(prediction, target, valid, QuietAudioConfig())
    assert int(qm.sum()) == 954 and not bool(qm[..., :966].any())
    zero = _direction_report(torch.zeros_like(target), _metric_gradients(prediction, target, valid, qm))
    assert zero["zero_loss_gradient"] and zero["metrics"]["peak_excess_mse"]["gradient_cosine"] is None
    assert all(math.isfinite(v["loss_values"][name]) for v in report["views"] for name in NAMES)
    print("PASS: CPU fixture loss signs, view sensitivity, quiet/no-quiet masks, zero gradients, copied state/RNG; no student or optimizer updates")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
