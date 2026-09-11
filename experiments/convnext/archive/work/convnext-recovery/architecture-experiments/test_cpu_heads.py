"""CPU correctness checks only. No timing runs or remote model loading."""
from types import SimpleNamespace

import pytest
import torch

from cpu_heads import assert_cpu_tree, check_streaming, validate_heads


def test_artifact_contract_and_unknown_output_modes():
    original = torch.randn(4, 3, 1)
    artifact = {"format_version": 1, "base_checkpoint_sha256": "frozen",
                "head_weights": {"baseline": original[..., 0].clone()},
                "candidates": [["baseline", "raw"], ["baseline", "clamp"]]}
    weights, candidates = validate_heads(artifact, original, "frozen")
    assert candidates == [("baseline", "raw"), ("baseline", "clamp")]
    assert_cpu_tree(weights)
    with pytest.raises(ValueError):
        validate_heads(artifact, original, "wrong")
    artifact["candidates"].append(["baseline", "unrecognized"])
    with pytest.raises(ValueError):
        validate_heads(artifact, original, "frozen")


@pytest.mark.parametrize("mode", ["raw", "clamp", "tanh"])
def test_streaming_count_tail_empty_and_exact_state_contract(mode):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eval()

        def forward(self, z):
            return z[:, :1, :].repeat_interleave(1920, dim=-1)

        def initial_state(self):
            return SimpleNamespace(started=torch.zeros(1, dtype=torch.bool), histories=())

        def state_shapes(self):
            return ()

        def forward_stream(self, z, state):
            if z.shape[-1]:
                state = SimpleNamespace(started=torch.ones(1, dtype=torch.bool), histories=())
            return self(z), state

    z = torch.randn(1, 64, 5)
    z[:, 0, :] *= 3
    for frames in (2, 4):
        result = check_streaming(Toy(), z, mode, frames)
        assert result["expected_samples"] == result["emitted_samples"] == 9600
        assert result["last_chunk_samples"] == 1920
        assert result["max_abs_batch_stream_difference"] == 0.
        assert result["empty_calls_preserve_state"]


def test_parity_exceedance_remains_failed_when_collection_continues():
    class OffByTinyAmount(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eval()

        def forward(self, z):
            return z[:, :1, :].repeat_interleave(1920, dim=-1)

        def initial_state(self):
            return SimpleNamespace(started=torch.zeros(1, dtype=torch.bool), histories=())

        def state_shapes(self):
            return ()

        def forward_stream(self, z, state):
            if not z.shape[-1]:
                return self(z), state
            return self(z) + 3e-6, SimpleNamespace(started=torch.ones(1, dtype=torch.bool), histories=())

    z = torch.zeros(1, 64, 3)
    with pytest.raises(RuntimeError, match="unchanged 2e-6"):
        check_streaming(OffByTinyAmount(), z, "raw", 2)
    row = check_streaming(OffByTinyAmount(), z, "raw", 2, fail_on_tolerance=False)
    assert row["passed"] is False and row["tolerance_relaxed"] is False
    assert row["threshold_absolute"] == 2e-6
    assert row["max_abs_batch_stream_difference"] > 2e-6
    assert row["emitted_samples"] == row["expected_samples"] == 5760
    assert "exceeds 2e-6" in row["failure_reason"]
