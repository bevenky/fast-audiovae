"""Small CPU fixtures exercising the actual pinned streaming implementation."""
import hashlib
import os
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))

from audiovae_student.teacher import SOURCE_SHA256, _load_source
import group_model as gm
import streaming_preflight as streaming


@pytest.fixture(scope="module")
def source_module():
    source = Path(os.environ.get("AUDIOVAE2_SOURCE", str(
        HERE.parents[2] / "convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py")))
    if not source.is_file():
        pytest.skip("Set AUDIOVAE2_SOURCE to the pinned teacher source for streaming contract tests")
    payload = source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
    name = f"_audiovae2_teacher_{SOURCE_SHA256}"
    return sys.modules.get(name) or _load_source(source, payload)


@pytest.fixture(autouse=True)
def cpu_determinism():
    old_threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(9137)
    yield
    torch.set_num_threads(old_threads)
    torch.set_rng_state(rng)


@pytest.fixture
def teacher(source_module):
    return source_module.CausalDecoder(
        input_channel=8, channels=128, rates=[8,6,5,2,2,2], depthwise=True,
        sr_bin_boundaries=[20000,30000,40000], cond_type="scale_bias", cond_out_layer=False,
    ).requires_grad_(False).eval()


@pytest.mark.parametrize("narrow", [False, True])
def test_actual_source_streaming_matches_continuous_with_short_tails(teacher, source_module, narrow):
    student = gm.build_student(teacher, list(range(16 if narrow else 32)), list(range(8 if narrow else 16)))
    student.requires_grad_(False).eval()
    z = torch.randn(1,8,5)
    state = streaming.state_digest(student)
    with torch.no_grad():
        reference = student(z)
    for pattern in ((1,), (2,), (4,), (1,4,2,3)):
        waveform, report = streaming.stream_sequence(student, source_module.StreamingVAEDecoder, source_module, z, pattern, reference)
        assert report["passed"]
        assert waveform.shape == reference.shape == (1,1,9600)
        assert sum(row["output_samples"] for row in report["chunks"]) == 9600
        assert all(row["state"]["buffers"] == 26 for row in report["chunks"])
        assert len({row["state"]["elements"] for row in report["chunks"]}) == 1
        if pattern in ((2,), (4,)):
            assert report["chunks"][-1]["short_tail"]
    assert streaming.state_digest(student) == state


def test_reused_context_and_interleaved_source_streams_are_isolated(teacher, source_module):
    student = gm.build_student(teacher, list(range(16)), list(range(8)))
    report = streaming.run_model_checks(
        student, source_module.StreamingVAEDecoder, source_module,
        [torch.randn(1,8,5), torch.randn(1,8,7)],
    )
    assert report["passed"]
    assert len(report["cases"]) == 8
    assert report["repeated_source_bitwise_equal"]
    assert report["interleaved_independent_streams"]["passed"]
    assert report["interleaved_independent_streams"]["output_samples"] == [9600,13440]
    assert report["model_state_sha256_before"] == report["model_state_sha256_after"]


def test_missing_output_sample_fails_and_restores_all_forward_methods(teacher, source_module):
    class LosingStream(source_module.StreamingVAEDecoder):
        def decode_chunk(self, z):
            return super().decode_chunk(z)[..., :-1]
    student = gm.build_student(teacher, list(range(16)), list(range(8)))
    student.requires_grad_(False).eval()
    before = streaming.forward_snapshot(student.decoder)
    stream = LosingStream(streaming.wrap_decoder(student))
    with pytest.raises(RuntimeError, match="lost, duplicated or deferred"):
        streaming.stream_sequence(student, LosingStream, source_module, torch.randn(1,8,3), (2,), stream=stream)
    streaming.assert_restored(stream, before)


def test_growing_causal_state_fails_before_acceptance(teacher, source_module):
    class GrowingState(source_module.StreamingVAEDecoder):
        def decode_chunk(self, z):
            result = super().decode_chunk(z)
            key = next(iter(self._states))
            self._states[key] = torch.cat([self._states[key], self._states[key][..., :1]], -1)
            return result
    student = gm.build_student(teacher, list(range(16)), list(range(8)))
    student.requires_grad_(False).eval()
    with pytest.raises(RuntimeError, match="Incorrect or growing"):
        streaming.stream_sequence(student, GrowingState, source_module, torch.randn(1,8,3), (1,))


@pytest.mark.parametrize("pattern", [(), (0,), (-1,), (True,), (1.5,)])
def test_invalid_chunk_pattern_rejected(pattern):
    with pytest.raises(ValueError):
        streaming.chunk_plan(5, pattern)


def test_selected_real_inputs_are_bounded_unique_and_have_short_tails():
    crops = [{"source_id":str(i), "context_start_frame":start,
              "latents":torch.randn(1,64,frames)} for i,start,frames in ((0,30,40),(1,0,24),(2,0,18))]
    values, rows = streaming.selected_inputs(crops)
    assert [row["source_id"] for row in rows] == ["1","2"]
    assert [value.shape[-1] for value in values] == [17,17]
    assert all(value.shape[-1] <= 19 and value.shape[-1] % 2 for value in values)
