"""CPU fixtures only. The official-source fixture uses random tiny weights."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from paired_layers import _eval_preserved, native_startup_check, self_test, student_stage_trace, teacher_stage_trace


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1907)
    self_test()
    class StartupFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.calls = 0

        def decode(self, z):
            self.calls += 1
            return z + (0.125 if self.calls == 1 else 0.)

    cold = StartupFixture()
    cold.weight.grad = torch.tensor(0.5)
    target = torch.ones(1, 1, 32)
    with torch.no_grad(), _eval_preserved(cold):
        stable, startup = native_startup_check(cold, target, target)
    assert cold.calls == 3 and cold.training and cold.weight.requires_grad
    assert startup["first_vs_second"]["max_abs"] == 0.125
    assert startup["second_vs_third"]["bitwise_equal"] and torch.equal(stable, target)
    cold.calls = 0
    try:
        native_startup_check(cold, target, target + .25)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A wrong canonical target was silently accepted")
    source = (Path(__file__).resolve().parents[2] / "convnext-natural-context-repair"
              / "upstream-source-audit" / "audio_vae_v2.py")
    spec = importlib.util.spec_from_file_location("official_teacher_cpu_fixture", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decoder = module.CausalDecoder(input_channel=64, channels=64, rates=[8, 6, 5, 2, 2, 2],
        depthwise=True, sr_bin_boundaries=[20000, 30000, 40000], cond_type="scale_bias").eval()

    class Fixture:
        sr_cond = 48000
        model = SimpleNamespace(decoder=decoder)
        provenance = {"config": {"decoder_rates": [8, 6, 5, 2, 2, 2], "depthwise": True,
            "use_noise_block": False, "cond_type": "scale_bias", "cond_out_layer": False}}

        def decode(self, z):
            return decoder(z, torch.tensor([48000], dtype=torch.int32))

    student = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=8)).eval()
    z = torch.randn(1, 64, 1).repeat(1, 1, 150)
    with torch.no_grad(), _eval_preserved(decoder), _eval_preserved(student):
        teacher = teacher_stage_trace(Fixture(), z)
        waveform, stages, _, _ = student_stage_trace(student, z)
    assert teacher["waveform"].shape == waveform.shape == (1, 1, 288000)
    assert teacher["sr_bucket"] == 3
    assert len([r for r in stages if r["name"].endswith(".gelu")]) == 10
    assert len(teacher["stages"]) == 120
    assert len(stages) == 80
    print("PASS: bounded cold-start reporting, strict canonical failure, exact official teacher leaf/SR-conditioning replay and ten-block student primitive replay; states preserved")


if __name__ == "__main__":
    main()
