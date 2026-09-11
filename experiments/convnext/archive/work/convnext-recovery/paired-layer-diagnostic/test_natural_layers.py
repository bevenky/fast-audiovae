"""Synthetic CPU fixtures for actual-forward observation and local gain VJPs."""
import math
import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from natural_layers import (capture_forward, gain_sensitivities, masked_feature_stats,
                            student_specs, _eval_preserved)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    valid = torch.zeros(1, 1, 960, dtype=torch.bool)
    valid[..., :720] = True
    result = masked_feature_stats(torch.tensor([[[1., 3.]]]), 100, valid)
    assert abs(result["scored_rms"] - math.sqrt(11 / 3)) < 1e-12
    assert result["scored_sample_coverage"] == 720

    x = torch.tensor([[[1., .5, .1]]], requires_grad=True)
    one = x + .1*x
    two = one + .2*one
    target = torch.zeros_like(two)
    valid = torch.ones_like(two, dtype=torch.bool)
    quiet = torch.zeros_like(valid); quiet[..., 2] = True
    directions = gain_sensitivities(two, target, valid, quiet, {0:x, 1:one}, {0:one, 1:two})
    assert directions["active_teacher_residual_mse"]["samples"] == 1  # Peak is not a control.
    assert directions["full_scale_excess_mse"]["fixed_baseline_overshoot_samples"] == 1
    expected = 2 * (1.32 - 1) / 3 * 1.2 * .1
    measured = directions["full_scale_excess_mse"]["blocks"][0]["d_metric_d_residual_gain_at_1"]
    assert abs(measured - expected) < 2e-8
    assert x.grad is None
    empty = gain_sensitivities(two, target, valid, valid, {0:x, 1:one}, {0:one, 1:two})
    assert not empty["active_teacher_residual_mse"]["defined"]

    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=8)).eval()
    parameter = next(model.parameters()); parameter.grad = torch.full_like(parameter, .25)
    z = torch.randn(1, 64, 8)
    valid = torch.ones(1, 1, 8*1920, dtype=torch.bool)
    hooks_before = [len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()]
    with _eval_preserved(model):
        with torch.no_grad(): native = model(z)
        with capture_forward(model, student_specs(model), valid, blocks=model.blocks,
                retain_modules=(("projection", model.output),)) as trace:
            observed = model(z.clone().requires_grad_(True))
            assert torch.equal(observed, native)
            assert torch.equal(model._waveform(trace["retained"]["projection"]), observed)
            quiet = valid.clone(); quiet[..., quiet.shape[-1]//2:] = False
            report = gain_sensitivities(observed, torch.zeros_like(observed), valid, quiet,
                                       trace["inputs"], trace["outputs"])
            assert len(report["teacher_quiet_residual_mse"]["blocks"]) == 10
            assert len([r for r in trace["rows"] if r["name"].endswith(".gelu")]) == 10
    assert hooks_before == [len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()]
    try:
        with capture_forward(model, student_specs(model), valid):
            raise RuntimeError("Synthetic hook cleanup failure")
    except RuntimeError:
        pass
    assert hooks_before == [len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()]
    print("PASS: physical masks, fixed peak exclusion, analytic residual-gain VJP, zero-mask reporting, actual-forward parity, ten-block hooks and state/gradient cleanup")


if __name__ == "__main__":
    main()
