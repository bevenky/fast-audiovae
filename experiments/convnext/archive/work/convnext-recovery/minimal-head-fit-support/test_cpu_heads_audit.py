"""Correctness-only checks of CPU candidate validation and functional streams."""
import pytest
import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from cpu_heads import check_streaming, validate_heads


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", ["raw", "clamp", "tanh"])
@pytest.mark.parametrize("chunk", [2, 4])
def test_cpu_streaming_counts_partial_tail_and_state_readonly(mode, chunk):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
            head_channels=12, adapter_mode="raw_repeat_phase_bias")).eval()
        z = torch.randn(1, 64, 7)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    row = check_streaming(model, z, mode, chunk)
    assert row["passed"] and row["emitted_samples"] == row["expected_samples"] == 7 * 1920
    assert row["last_chunk_samples"] == (7 % chunk) * 1920
    assert row["empty_calls_preserve_state"] and not model.training
    assert all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in before.items())
    assert not torch.cuda.is_initialized()


def test_head_artifact_duplicate_missing_and_nonfinite_guards():
    w = torch.zeros(480, 12, 1)
    artifact = {"format_version": 1, "base_checkpoint_sha256": "fixture",
                "head_weights": {"baseline": w[..., 0]}, "candidates": [("baseline", "raw")]}
    weights, rows = validate_heads(artifact, w, "fixture")
    assert rows == [("baseline", "raw")]
    artifact["candidates"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        validate_heads(artifact, w, "fixture")
    artifact["candidates"] = [("baseline", "clamp")]
    with pytest.raises(ValueError, match="Missing raw"):
        validate_heads(artifact, w, "fixture")
    artifact["candidates"] = [("baseline", "raw"), ("broken", "raw")]
    artifact["head_weights"]["broken"] = torch.full_like(w[..., 0], float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_heads(artifact, w, "fixture")
