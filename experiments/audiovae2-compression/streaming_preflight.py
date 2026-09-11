"""Bounded CPU parity checks using the pinned teacher's streaming machinery.

Both an independent full-width copy and the initialized narrower decoder are
checked. This is state/sample correctness, not quality qualification or timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import torch

from audiovae_student.teacher import FrozenAudioVAE2
from export_preflight import authenticate_preflight, compare, sha, write_json
from group_model import CompressedDecoderGroup, build_student, clone_teacher
from run_pilot import load_data


VERSION = "audiovae2_group_streaming_v1"


def state_digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def wrap_decoder(student):
    def decode(z):
        sr = torch.tensor([48000], device=z.device, dtype=torch.int32)
        return student.decoder(z, sr)
    return SimpleNamespace(decoder=student.decoder, decode=decode)


def expected_states(decoder, source_module, batch_size=1):
    expected = {}
    for name, module in decoder.named_modules():
        if isinstance(module, source_module.CausalConv1d):
            history = module._CausalConv1d__padding * 2 - module._CausalConv1d__output_padding
        elif isinstance(module, source_module.CausalTransposeConv1d):
            history = (module.kernel_size[0] - 1) // module.stride[0]
        else:
            continue
        if history > 0:
            expected[id(module)] = {"module": name, "shape": (batch_size, module.in_channels, history)}
    if len(expected) != 26:
        raise RuntimeError(f"Expected 26 causal state buffers in the approved decoder, found {len(expected)}")
    return expected


def check_state(stream, decoder, source_module, batch_size=1):
    expected = expected_states(decoder, source_module, batch_size)
    if set(stream._states) != set(expected):
        raise RuntimeError("Streaming state keys do not match all causal modules")
    for key, spec in expected.items():
        tensor = stream._states[key]
        if tuple(tensor.shape) != spec["shape"] or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise RuntimeError(f"Incorrect or growing state at {spec['module']}")
        if tensor.requires_grad or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Nonfinite or attached streaming state at {spec['module']}")
    return {"buffers": len(expected), "elements": sum(t.numel() for t in stream._states.values()),
            "bytes": sum(t.numel() * t.element_size() for t in stream._states.values())}


def forward_snapshot(decoder):
    return [(module, module.forward) for module in decoder.modules()]


def assert_restored(stream, snapshot):
    if stream._states or stream._originals:
        raise RuntimeError("Streaming state or forward patches survived context exit")
    if any(module.forward != original for module, original in snapshot):
        raise RuntimeError("A decoder forward method was not restored")


def chunk_plan(frames, pattern):
    if frames <= 0 or not pattern or any(type(size) is not int or size <= 0 for size in pattern):
        raise ValueError("Positive input length and positive integer chunk sizes are required")
    result = []
    start = 0
    while start < frames:
        requested = pattern[len(result) % len(pattern)]
        stop = min(start + requested, frames)
        result.append((start, stop, requested))
        start = stop
    return result


def parity(reference, actual, label):
    return compare(reference.detach().cpu().numpy(), actual.detach().cpu().numpy(), tuple(reference.shape), label)


@torch.no_grad()
def stream_sequence(student, streaming_cls, source_module, z, pattern, reference=None, stream=None):
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] < 1 or z.device.type != "cpu" or z.dtype != torch.float32:
        raise ValueError("Expected one nonempty CPU FP32 latent sequence")
    if reference is None:
        reference = student(z)
    if tuple(reference.shape) != (1, 1, z.shape[-1] * 1920):
        raise RuntimeError("Continuous decoder violated sample accounting")
    if stream is None:
        stream = streaming_cls(wrap_decoder(student))
    elif stream._vae.decoder is not student.decoder:
        raise ValueError("Reusable stream belongs to another decoder")
    snapshot = forward_snapshot(student.decoder)
    outputs, chunks = [], []
    try:
        with stream:
            if stream._states:
                raise RuntimeError("New or reused streaming context did not reset its state")
            for start, stop, requested in chunk_plan(z.shape[-1], pattern):
                output = stream.decode_chunk(z[..., start:stop])
                expected_samples = (stop-start) * 1920
                if tuple(output.shape) != (1, 1, expected_samples):
                    raise RuntimeError("A streaming call lost, duplicated or deferred samples")
                state = check_state(stream, student.decoder, source_module)
                row = {"start_frame": start, "stop_frame": stop, "requested_frames": requested,
                       "input_frames": stop-start, "output_samples": output.shape[-1],
                       "short_tail": stop-start < requested, "state": state,
                       "parity": parity(reference[..., start*1920:stop*1920], output, "streaming chunk")}
                outputs.append(output)
                chunks.append(row)
    finally:
        assert_restored(stream, snapshot)
    waveform = torch.cat(outputs, dim=-1)
    report = {"pattern_latent_frames": list(pattern), "input_frames": z.shape[-1],
              "output_samples": waveform.shape[-1], "calls": len(chunks), "chunks": chunks,
              "full_parity": parity(reference, waveform, "concatenated streaming output"),
              "startup_parity": parity(reference[..., :1920], waveform[..., :1920], "startup"),
              "tail_parity": parity(reference[..., -1920:], waveform[..., -1920:], "tail"),
              "bounded_output": float(waveform.abs().max()) <= 1.0 + 2e-6,
              "reset_after_exit": not stream._states, "forwards_restored": True}
    report["passed"] = (report["full_parity"]["passed"] and report["startup_parity"]["passed"]
                        and report["tail_parity"]["passed"] and report["bounded_output"]
                        and all(row["parity"]["passed"] for row in chunks))
    return waveform, report


@torch.no_grad()
def interleaved_streams(student, streaming_cls, source_module, inputs, references):
    other = CompressedDecoderGroup(clone_teacher(student.decoder), student.selections, student.output_sample_rate)
    other.requires_grad_(False).eval()
    models = (student, other)
    streams = [streaming_cls(wrap_decoder(model)) for model in models]
    snapshots = [forward_snapshot(model.decoder) for model in models]
    plans = [chunk_plan(inputs[0].shape[-1], (2,)), chunk_plan(inputs[1].shape[-1], (4,))]
    outputs, counts = [[], []], [0, 0]
    states = [None, None]
    try:
        with streams[0], streams[1]:
            for i in range(max(map(len, plans))):
                for j in range(2):
                    if i >= len(plans[j]):
                        continue
                    start, stop, _ = plans[j][i]
                    output = streams[j].decode_chunk(inputs[j][..., start:stop])
                    if tuple(output.shape) != (1, 1, (stop-start)*1920):
                        raise RuntimeError("Interleaved stream changed sample counts")
                    outputs[j].append(output)
                    counts[j] += output.shape[-1]
                    states[j] = check_state(streams[j], models[j].decoder, source_module)
                if set(streams[0]._states) & set(streams[1]._states):
                    raise RuntimeError("Independent streams share module state keys")
    finally:
        for stream, snapshot in zip(streams, snapshots):
            assert_restored(stream, snapshot)
    rows = [parity(ref, torch.cat(values, -1), "interleaved stream") for ref, values in zip(references, outputs)]
    return {"passed": all(row["passed"] for row in rows), "parity": rows,
            "output_samples": counts, "states": states, "forwards_restored": True,
            "state_isolation": "Two independent decoder objects, both contexts active and chunk calls interleaved"}


@torch.no_grad()
def run_model_checks(student, streaming_cls, source_module, inputs):
    if len(inputs) != 2:
        raise ValueError("Two distinct input sequences are required")
    student.requires_grad_(False).eval()
    before = state_digest(student)
    references = [student(z) for z in inputs]
    cases = []
    for i, z in enumerate(inputs):
        for pattern in ((1,), (2,), (4,), (1, 4, 2, 3)):
            _, report = stream_sequence(student, streaming_cls, source_module, z, pattern, references[i])
            cases.append({"input_index": i, **report})
    # Reenter the same context object after another source. It must not retain
    # the previous source's hidden history, even after a short final chunk.
    stream = streaming_cls(wrap_decoder(student))
    reset_runs, reset_waves = [], []
    for i in (0, 1, 0):
        waveform, report = stream_sequence(student, streaming_cls, source_module, inputs[i], (4,), references[i], stream)
        reset_runs.append({"input_index": i, **report})
        reset_waves.append(waveform)
    reset_equal = torch.equal(reset_waves[0], reset_waves[2])
    interleaved = interleaved_streams(student, streaming_cls, source_module, inputs, references)
    after = state_digest(student)
    if after != before or any(m.training for m in student.modules()) or any(p.requires_grad for p in student.parameters()):
        raise RuntimeError("Streaming checks changed model state, modes or frozen parameters")
    return {"passed": all(c["passed"] for c in cases + reset_runs) and reset_equal and interleaved["passed"],
            "cases": cases, "reused_context_runs": reset_runs, "repeated_source_bitwise_equal": reset_equal,
            "interleaved_independent_streams": interleaved,
            "model_state_sha256_before": before, "model_state_sha256_after": after,
            "continuous_output_samples": [int(r.shape[-1]) for r in references]}


def selected_inputs(crops):
    eligible = [(i, crop) for i, crop in enumerate(crops) if crop["latents"].shape[-1] >= 5]
    eligible.sort(key=lambda item: (item[1]["context_start_frame"] != 0, item[0]))
    if len(eligible) < 2:
        raise ValueError("Need two real calibration sequences with at least five latent frames")
    inputs, metadata = [], []
    for limit, (index, crop) in zip((17, 19), eligible[:2]):
        frames = min(limit, crop["latents"].shape[-1])
        if frames % 2 == 0:
            frames -= 1  # Ensure a short final call for both 2- and 4-frame chunks.
        z = crop["latents"][..., :frames].detach().cpu().contiguous().clone()
        if tuple(z.shape) != (1, 64, frames) or z.dtype != torch.float32 or not torch.isfinite(z).all():
            raise ValueError("Invalid real calibration latent tensor")
        inputs.append(z)
        metadata.append({"calibration_index": index, "source_id": crop["source_id"],
                         "absolute_first_latent_frame": crop["context_start_frame"],
                         "available_frames": crop["latents"].shape[-1], "tested_frames": frames,
                         "latent_sha256": hashlib.sha256(memoryview(z.numpy()).cast("B")).hexdigest()})
    if metadata[0]["source_id"] == metadata[1]["source_id"]:
        raise ValueError("Streaming input sources must be distinct")
    return inputs, metadata


def execute(assets: Path, out: Path, manifest_path: Path):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    metadata, selection, initial = authenticate_preflight(out)
    pre = json.loads((out / "preflight.json").read_text())
    if sha(manifest_path) != pre["identity"]["manifest_sha256"]:
        raise ValueError("Streaming manifest does not match preflight")
    runner_sha = sha(Path(__file__).with_name("run_pilot.py"))
    if runner_sha != pre["identity"]["runner_sha256"]:
        raise ValueError("Pilot source changed before streaming validation")
    _, pools, receipt = load_data(manifest_path)
    if receipt["sha256"] != pre["identity"]["cache_sha256"]:
        raise ValueError("Streaming latent cache does not match preflight")
    inputs, input_metadata = selected_inputs(pools["calibration"])
    report = {**metadata, "version": VERSION, "passed": False, "device": "cpu",
              "torch": str(torch.__version__), "threads": 1,
              "streaming_checker_sha256": sha(__file__), "manifest_sha256": sha(manifest_path),
              "runner_sha256": runner_sha,
              "cache_sha256": receipt["sha256"], "inputs": input_metadata,
              "parity_tolerance": {"atol": 1e-5, "rtol": 1e-4},
              "chunk_sizes_latent_frames": [1, 2, 4], "chunk_sizes_ms": [40, 80, 160],
              "rtf_measured": False, "native_runtime_verified": False,
              "scope": "Pinned original PyTorch stateful decoder; continuous and streaming paths both reset to zero at each tested sequence start. This is output/state parity, not compressed-model quality or full-source teacher-target scoring.",
              "models": {}}
    path = out / "streaming-check.json"
    write_json(path, report)
    try:
        teacher = FrozenAudioVAE2.from_files(assets / "audio_vae_v2.py", assets / "audiovae.pth", device="cpu")
        source_module = sys.modules[type(teacher.model).__module__]
        streaming_cls = source_module.StreamingVAEDecoder
        report["implementation"] = "Pinned AudioVAE2 source StreamingVAEDecoder on independent cloned decoder objects"
        teacher_before = state_digest(teacher)
        for name, index2, index3 in (
            ("full_width_control", list(range(512)), list(range(256))),
            ("narrowed_initial", selection["stage2_indices"], selection["stage3_indices"]),
        ):
            student = build_student(teacher.model.decoder, index2, index3)
            if name == "narrowed_initial":
                student.load_group_state_dict(initial["group"])
            report["models"][name] = run_model_checks(student, streaming_cls, source_module, inputs)
            if name == "full_width_control":
                with torch.no_grad():
                    copies = [parity(teacher.decode(z), student(z), "full-width control vs original teacher") for z in inputs]
                report["models"][name]["original_teacher_parity"] = copies
                report["models"][name]["passed"] &= all(row["passed"] for row in copies)
            write_json(path, report)
            print(json.dumps({"stage": "streaming_correctness", "model": name, "passed": report["models"][name]["passed"]}), flush=True)
            del student
        teacher_after = state_digest(teacher)
        report["teacher_state_sha256_before"] = teacher_before
        report["teacher_state_sha256_after"] = teacher_after
        if teacher_after != teacher_before:
            raise RuntimeError("Original teacher state changed")
        current, _, _ = authenticate_preflight(out)
        if (current != metadata or sha(manifest_path) != report["manifest_sha256"]
                or sha(__file__) != report["streaming_checker_sha256"]
                or sha(Path(__file__).with_name("run_pilot.py")) != runner_sha):
            raise RuntimeError("Authenticated experiment inputs changed during the streaming check")
        report["passed"] = all(value["passed"] for value in report["models"].values()) and len(report["models"]) == 2
        write_json(path, report)
        return report
    except BaseException as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(path, report)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    report = execute(args.assets, args.out, args.manifest)
    print(json.dumps({"passed": report["passed"], "report": str(args.out / "streaming-check.json")}), flush=True)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
