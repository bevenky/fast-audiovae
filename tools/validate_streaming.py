"""Validate cached AudioVAE2 streaming against full and upstream decoding.

Inputs are local prepared bundles and NPZ files containing UID__z and,
optionally, UID__ref arrays. The companion JSON is recorded for provenance.
Nothing is downloaded. Timing is opt-in and follows all correctness checks.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time


CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
           "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
SOURCE_SHA256 = "2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8"
CHECKPOINT_SHA256 = "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"
HOP = 1920
RATE = 48000
np = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_environment():
    wrong = {key: os.environ.get(key) for key, value in CPU_ENV.items()
             if os.environ.get(key) != value}
    if wrong:
        raise RuntimeError("Set CPU-only visibility before launch: " +
                           " ".join(key + "=" + value for key, value in CPU_ENV.items()))
    return {key: os.environ[key] for key in CPU_ENV}


def chunks(length, pattern):
    """Return contiguous boundaries, including a shorter final chunk."""
    if length < 1 or not pattern or any(type(size) is not int or size < 1 for size in pattern):
        raise ValueError("Positive length and chunk sizes required")
    start = index = 0
    while start < length:
        end = min(length, start + pattern[index % len(pattern)])
        yield start, end
        start, index = end, index + 1


def wave_contract(value, frames):
    return (isinstance(value, np.ndarray) and value.dtype == np.float32 and
            value.shape == (1, 1, HOP * frames) and bool(np.isfinite(value).all()))


def discrepancy(reference, actual, *, atol=1e-5, rtol=1e-4, boundaries=()):
    """Strict waveform and boundary comparison; SNR null means undefined/infinite."""
    valid = (isinstance(reference, np.ndarray) and isinstance(actual, np.ndarray) and
             reference.dtype == actual.dtype == np.float32 and
             reference.shape == actual.shape and reference.ndim == 3 and
             reference.shape[:2] == (1, 1) and reference.size > 0 and
             bool(np.isfinite(reference).all()) and bool(np.isfinite(actual).all()))
    result = {"passed": False, "shape": list(actual.shape) if hasattr(actual, "shape") else None,
              "expected_shape": list(reference.shape), "finite_shape_dtype": valid,
              "atol": atol, "rtol": rtol}
    if not valid:
        return result
    error = actual.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(error * error))
    signal = float(np.mean(reference.astype(np.float64) ** 2))
    result.update(passed=bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
                  max_abs=float(np.max(np.abs(error))), rmse=float(np.sqrt(mse)),
                  snr_db=float(10 * np.log10(signal / mse)) if mse > 0 and signal > 0 else None,
                  exact=bool(np.array_equal(reference.view(np.uint32), actual.view(np.uint32))),
                  boundary_max_abs=max((float(np.max(np.abs(error[..., max(0, b - 32):min(error.shape[-1], b + 32)])))
                                        for b in boundaries if 0 < b < error.shape[-1]), default=0.0))
    return result


def decode_chunks(stream, latent, pattern):
    output, returned, boundaries = [], [], []
    for start, end in chunks(latent.shape[-1], pattern):
        part = stream.decode_chunk(latent[..., start:end])
        if not wave_contract(part, end - start):
            raise ValueError("decode_chunk returned an invalid waveform at latent frame " + str(start))
        output.append(part.copy())
        returned.append(part)
        boundaries.append(end * HOP)
    tail = stream.flush()
    if not wave_contract(tail, 0):
        raise ValueError("Decoder flush must emit no extra samples for complete latent frames")
    if any(not np.array_equal(original, snapshot) for original, snapshot in zip(returned, output)):
        raise ValueError("A later call changed a previously returned audio chunk")
    return np.concatenate(output, axis=-1), boundaries[:-1]


def load_cases(path, limit=None):
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(key for key in archive.files if key.endswith("__z") or key == "z")
        if not keys:
            raise ValueError("NPZ contains no UID__z or z latent arrays")
        if limit is not None:
            keys = keys[:limit]
        result = []
        for key in keys:
            z = archive[key]
            if z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (1, 64) or z.shape[-1] < 1 or not np.isfinite(z).all():
                raise ValueError("Invalid finite FP32 latent array: " + key)
            uid = key[:-3] if key.endswith("__z") else key
            reference_key = uid + "__ref" if key.endswith("__z") else "ref"
            reference = archive[reference_key] if reference_key in archive else None
            if reference is not None and not wave_contract(reference, z.shape[-1]):
                raise ValueError("Invalid stored full waveform: " + reference_key)
            result.append((uid, z, reference))
    return result


def validate_case(decoder, full_decode, uid, latent, stored_reference=None, upstream=None,
                  reference_policy="strict-upstream", waveform_callback=None):
    """Return all checks; exceptions are recorded by the campaign caller."""
    reference = full_decode(latent)
    if not wave_contract(reference, latent.shape[-1]):
        raise ValueError("Full decoder returned an invalid waveform")
    if waveform_callback:
        waveform_callback(uid, "full", reference)
    checks = []

    strict_upstream = reference_policy == "strict-upstream"
    if reference_policy not in ("strict-upstream", "same-model"):
        raise ValueError("Unknown reference policy")

    def compare(name, expected, actual, boundaries=(), gate=True):
        checks.append({"case": uid, "check": name, "gate": gate,
                       **discrepancy(expected, actual, boundaries=boundaries)})

    if stored_reference is not None:
        compare("full_vs_stored_upstream", stored_reference, reference, gate=strict_upstream)
    if upstream is not None:
        compare("full_vs_upstream", upstream.full(latent), reference, gate=strict_upstream)
    for name, pattern in (("one_frame", (1,)), ("two_frames", (2,)),
                          ("five_frames", (5,)), ("uneven", (1, 7, 2, 3, 11))):
        with decoder.streaming_decode() as stream:
            actual, boundaries = decode_chunks(stream, latent, pattern)
        compare(name, reference, actual, boundaries)
        if waveform_callback and name in ("one_frame", "five_frames"):
            waveform_callback(uid, name, actual)
        if upstream is not None:
            compare(name + "_vs_upstream_stream", upstream.stream(latent, pattern), actual, boundaries,
                    gate=strict_upstream)

    # Force a final partial chunk even when the original length is divisible by five.
    partial_length = min(7, latent.shape[-1])
    if partial_length > 1:
        partial_pattern = (partial_length - 1,)
        partial = latent[..., :partial_length]
        with decoder.streaming_decode() as stream:
            actual, boundaries = decode_chunks(stream, partial, partial_pattern)
        compare("final_partial", full_decode(partial), actual, boundaries)

    with decoder.streaming_decode() as stream:
        # Reset after unflushed history and after a completed stream.
        stream.decode_chunk(latent[..., :1])
        stream.reset()
        replay, boundaries = decode_chunks(stream, latent, (5,))
        compare("reset_incomplete", reference, replay, boundaries)
        stream.reset()
        replay, boundaries = decode_chunks(stream, latent, (5,))
        compare("reset_completed", reference, replay, boundaries)

    # Different input/history catches accidental state stored in a shared session.
    other = np.ascontiguousarray(latent[..., ::-1])
    other_reference = full_decode(other)
    outputs = [[], []]
    with ExitStack() as stack:
        streams = [stack.enter_context(decoder.streaming_decode()) for _ in range(2)]
        for start, end in chunks(latent.shape[-1], (2, 5, 1)):
            for index, z in enumerate((latent, other)):
                actual = streams[index].decode_chunk(z[..., start:end])
                if not wave_contract(actual, end - start):
                    raise ValueError("Invalid interleaved stream output")
                outputs[index].append(actual.copy())
        for stream in streams:
            if not wave_contract(stream.flush(), 0):
                raise ValueError("Interleaved flush emitted extra output")
    compare("independent_stream_a", reference, np.concatenate(outputs[0], axis=-1))
    compare("independent_stream_b", other_reference, np.concatenate(outputs[1], axis=-1))
    return checks


class UpstreamReference:
    """Serial reference only: pinned upstream temporarily patches model forwards."""

    def __init__(self, source, checkpoint, threads):
        if sha256(source) != SOURCE_SHA256 or sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Upstream source/checkpoint must match the validated AudioVAE2 pins")
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        name = "_fast_audiovae_streaming_reference"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.model = module.AudioVAE(module.AudioVAEConfig()).cpu().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("state_dict", state), strict=True)

    def full(self, z):
        with self.torch.inference_mode():
            return self.model.decode(self.torch.from_numpy(np.ascontiguousarray(z))).numpy().copy()

    def stream(self, z, pattern):
        output = []
        with self.torch.inference_mode(), self.model.streaming_decode() as stream:
            for start, end in chunks(z.shape[-1], pattern):
                output.append(stream.decode_chunk(self.torch.from_numpy(np.ascontiguousarray(z[..., start:end]))).numpy().copy())
        return np.concatenate(output, axis=-1)


def measure(decoder, cases, warmups, repeats, patterns):
    rows = []
    for uid, latent, _ in cases:
        for frames in patterns:
            for repeat in range(-warmups, repeats):
                timings = []
                with decoder.streaming_decode() as stream:
                    for start, end in chunks(latent.shape[-1], (frames,)):
                        before = time.perf_counter()
                        output = stream.decode_chunk(latent[..., start:end])
                        timings.append(time.perf_counter() - before)
                        if not wave_contract(output, end - start):
                            raise ValueError("Invalid output during timing")
                    before = time.perf_counter()
                    tail = stream.flush()
                    flush_seconds = time.perf_counter() - before
                    if not wave_contract(tail, 0):
                        raise ValueError("Invalid flush during timing")
                if repeat >= 0:
                    duration = latent.shape[-1] * HOP / RATE
                    rows.append({"case": uid, "frames_per_chunk": frames, "repeat": repeat,
                                 "audio_seconds": duration, "decode_seconds": sum(timings) + flush_seconds,
                                 "rtf": (sum(timings) + flush_seconds) / duration,
                                 "first_chunk_ms": timings[0] * 1000,
                                 "chunk_p50_ms": float(np.percentile(timings, 50)) * 1000,
                                 "chunk_p95_ms": float(np.percentile(timings, 95)) * 1000,
                                 "max_chunk_ms": max(timings) * 1000, "flush_ms": flush_seconds * 1000})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--cases-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Explicit smoke subset; omitted validates every case")
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--upstream-checkpoint", type=Path)
    parser.add_argument("--reference-policy", choices=("strict-upstream", "same-model"),
                        default="strict-upstream", help="Same-model gates streaming parity; upstream fidelity remains reported separately")
    parser.add_argument("--waveform-dir", type=Path, help="Save unmodified FLOAT WAVs for independent quality scoring")
    parser.add_argument("--expected-selection", choices=("native", "portable_onnx"),
                        help="Fail if this validation unexpectedly selects another backend")
    parser.add_argument("--timing-cases", type=int, default=0, help="Opt-in count of validated cases to time")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timing-frames", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1) or args.timing_cases < 0 or args.warmups < 1 or args.repeats < 1 or any(n < 1 for n in args.timing_frames):
        parser.error("Positive threads, limits, warmups, repeats and chunk sizes required")
    if bool(args.upstream_source) != bool(args.upstream_checkpoint):
        parser.error("Supply both upstream source and checkpoint")
    if args.output.exists():
        parser.error("Output already exists; use a fresh output path")
    if args.waveform_dir and args.waveform_dir.exists():
        parser.error("Waveform directory already exists; use a fresh destination")
    environment = cpu_environment()
    if args.waveform_dir:
        import soundfile  # Check the optional WAV dependency before any model execution.
    global np
    import numpy as np
    import onnxruntime as ort
    if ort.__version__ != "1.29.0":
        raise RuntimeError("ONNX Runtime 1.29.0 required")
    from fast_audiovae import load_decoder, load_streaming_decoder
    manifest = args.cases_manifest or args.cases.with_suffix(".json")
    artifacts = {str(args.cases.resolve()): sha256(args.cases)}
    for path in sorted(args.model_dir.rglob("*")):
        if path.is_file():
            artifacts[str(path.resolve())] = sha256(path)
    if manifest.exists():
        artifacts[str(manifest.resolve())] = sha256(manifest)
    if args.upstream_source:
        for path in (args.upstream_source, args.upstream_checkpoint):
            artifacts[str(path.resolve())] = sha256(path)
    result = {"status": "running", "passed": False, "gpu_used": False,
              "environment": environment, "python": platform.python_version(),
              "system": platform.platform(), "machine": platform.machine(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "onnxruntime": ort.__version__, "numpy": np.__version__, "threads": args.threads,
              "script_sha256": sha256(__file__), "artifact_sha256": artifacts,
              "protocol": {"atol": 1e-5, "rtol": 1e-4, "sample_rate": RATE, "decoder_hop": HOP,
                           "patterns": [[1], [2], [5], [1, 7, 2, 3, 11]],
                           "limit": args.limit, "warmups": args.warmups, "repeats": args.repeats,
                           "reference_policy": args.reference_policy,
                           "timing_cases": args.timing_cases,
                           "timing_scope": "Steady stream decode calls and flush; excludes load, stream creation, warmup and output concatenation",
                           "quality_scope": "Numerical waveform parity; no perceptual MOS claim"},
              "checks": [], "failures": [], "measurements": [], "waveforms": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    def save_waveform(uid, variant, audio):
        import re
        import soundfile as sf
        if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
            raise ValueError("Unsafe waveform case identifier")
        path = args.waveform_dir / variant / (uid + ".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.reshape(-1), RATE, subtype="FLOAT")
        result["waveforms"].append({"case": uid, "variant": variant, "path": str(path.resolve()),
                                    "sample_rate": RATE, "samples": audio.shape[-1], "sha256": sha256(path)})

    try:
        cases = load_cases(args.cases, args.limit)
        result["case_count"] = len(cases)
        result["case_uids"] = [case[0] for case in cases]
        result["timing_uids"] = [case[0] for case in cases[:args.timing_cases]]
        decoder, info = load_streaming_decoder(args.model_dir, threads=args.threads)
        full, full_info = load_decoder(args.model_dir, threads=args.threads)
        result["streaming_backend"], result["full_backend"] = info, full_info
        if args.expected_selection and (info.get("selected") != args.expected_selection
                                        or full_info.get("selected") != args.expected_selection):
            raise RuntimeError("Validation selected an unexpected decoder backend")
        if info.get("selected") != full_info.get("selected"):
            raise RuntimeError("Full and streaming decoder selected different backends")
        if full.get_providers() != ["CPUExecutionProvider"] or info.get("providers") != ["CPUExecutionProvider"]:
            raise RuntimeError("CPU-only providers required")
        full_decode = lambda z: full.run(None, {full.get_inputs()[0].name: z})[0]
        upstream = UpstreamReference(args.upstream_source, args.upstream_checkpoint, args.threads) if args.upstream_source else None
        for index, (uid, latent, stored) in enumerate(cases):
            try:
                result["checks"].extend(validate_case(decoder, full_decode, uid, latent, stored, upstream,
                                                       args.reference_policy, save_waveform if args.waveform_dir else None))
            except Exception as exc:
                result["failures"].append({"case": uid, "error": repr(exc)})
            save()
            print(json.dumps({"case": uid, "completed": index + 1, "total": len(cases)}), flush=True)
        result["passed"] = bool(result["checks"]) and not result["failures"] and all(
            row["passed"] for row in result["checks"] if row["gate"])
        if result["passed"] and args.timing_cases:
            result["measurements"] = measure(decoder, cases[:args.timing_cases], args.warmups, args.repeats, args.timing_frames)
        result["status"] = "complete"
    except Exception as exc:
        result["passed"] = False
        result["status"] = "failed"
        result["failures"].append({"error": repr(exc)})
    save()
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                      "failed_checks": sum(not row["passed"] for row in result["checks"] if row["gate"]),
                      "upstream_comparisons_outside_tolerance": sum(not row["passed"] for row in result["checks"] if not row["gate"]),
                      "failures": result["failures"], "output": str(args.output)}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
