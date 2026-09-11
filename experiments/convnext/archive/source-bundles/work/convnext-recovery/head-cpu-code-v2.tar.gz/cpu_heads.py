"""CPU-only streaming parity and bounded-head operation cost for frozen heads.

This is the experimental PyTorch student, not a production ONNX or Mimi test.
The script never loads a teacher or runs an encoder. Run after head calibration.
"""
import os

# These take effect before importing PyTorch or any project modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch

from bounded_head import bounded_waveform


def transform(audio, mode):
    if mode == "raw":
        return audio
    if mode == "clamp":
        return bounded_waveform(audio)
    if mode == "tanh":
        return torch.tanh(audio)
    raise ValueError("Unknown output mode: " + str(mode))


def assert_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError("Non-CPU tensor in this CPU-only experiment")
    elif isinstance(value, dict):
        for child in value.values():
            assert_cpu_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_cpu_tree(child)


def cpu_identity():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        info.update(affinity_cpu_count=len(affinity), affinity_cpus=affinity)
    proc = Path("/proc/cpuinfo")
    if proc.is_file():
        first = proc.read_text().split("\n\n", 1)[0]
        values = dict(line.split(":", 1) for line in first.splitlines() if ":" in line)
        values = {k.strip(): v.strip() for k, v in values.items()}
        for source, label in (("model name", "model_name"), ("vendor_id", "vendor"),
                              ("cpu family", "family"), ("model", "model"),
                              ("flags", "flags"), ("Features", "features")):
            if source in values:
                info[label] = values[source]
    elif platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info["model_name"] = result.stdout.strip()
    return info


def validate_heads(artifact, original_weight, checkpoint_sha):
    if artifact.get("format_version") != 1 or artifact.get("base_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Head artifact does not match the frozen checkpoint")
    weights = artifact.get("head_weights")
    candidates = artifact.get("candidates")
    if not isinstance(weights, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("Head artifact is incomplete")
    if not torch.equal(weights.get("baseline", torch.empty(0)), original_weight[..., 0]):
        raise ValueError("Artifact baseline readout differs from the frozen checkpoint")
    result = []
    for item in candidates:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Malformed candidate")
        name, mode = item
        if not isinstance(name, str) or mode not in ("raw", "clamp", "tanh") or name not in weights:
            raise ValueError("Unknown candidate readout or output mode")
        weight = weights[name]
        if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
                or weight.shape != original_weight.shape[:2] or weight.dtype != original_weight.dtype
                or not bool(torch.isfinite(weight).all())):
            raise ValueError("Readout shape, dtype, finite or CPU contract failed")
        result.append((name, mode))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate head candidates")
    if ("baseline", "raw") not in result:
        raise ValueError("Missing raw frozen baseline")
    return weights, result


def fixed_cases(panel):
    crops, metadata = panel["crops"], panel["metadata"]
    zeros = [c for c in crops if c.source_id == "encoded_zero"]
    laughter = [c for c in crops if c.source_id == "freesound:25794" and c.start_frame == 45]
    speech = [c for c in crops if metadata[c.source_id].get("condition") == "speech"]
    if len(zeros) != 1 or len(laughter) != 1 or not speech:
        raise ValueError("The declared encoded-zero, first-speech and laughter cases are unavailable")
    cases = [("encoded_zero", zeros[0]), ("first_speech_in_frozen_panel", speech[0]),
             ("previously_identified_laughter_peak", laughter[0])]
    timing_crop = next((c for c in speech if c.latents.shape[-1] >= 75),
                       max(speech, key=lambda c: c.latents.shape[-1]))
    timing_z = timing_crop.latents[..., :75].detach().contiguous()
    if timing_z.shape[-1] < 16:
        raise ValueError("Insufficient fixed speech context for the CPU timing screen")
    return cases, timing_crop, timing_z


@torch.inference_mode()
def check_streaming(model, z, mode, frames_per_chunk, *, fail_on_tolerance=True):
    if type(fail_on_tolerance) is not bool:
        raise ValueError("Tolerance failure policy must be Boolean")
    if z.device.type != "cpu" or model.training:
        raise ValueError("Streaming parity requires CPU evaluation mode")
    expected = transform(model(z), mode)
    if expected.shape != (1, 1, z.shape[-1] * 1920) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError("Batch output contract failed")
    state = model.initial_state()
    shapes = model.state_shapes()
    empty, returned = model.forward_stream(z[..., :0], state)
    if returned is not state or transform(empty, mode).shape != (1, 1, 0):
        raise RuntimeError("Empty streaming call changed state or emitted samples")
    outputs = []
    lengths = []
    for chunk in z.split(frames_per_chunk, dim=-1):
        raw, state = model.forward_stream(chunk, state)
        output = transform(raw, mode)
        if output.shape != (1, 1, chunk.shape[-1] * 1920) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Streaming call lost samples or produced nonfinite output")
        if tuple(tuple(h.shape) for h in state.histories) != shapes:
            raise RuntimeError("Streaming history geometry changed")
        if state.started.device.type != "cpu" or any(h.device.type != "cpu" for h in state.histories):
            raise RuntimeError("Streaming state escaped CPU")
        if any(not bool(torch.isfinite(h).all()) for h in state.histories):
            raise RuntimeError("Streaming history is nonfinite")
        outputs.append(output)
        lengths.append(output.shape[-1])
    actual = torch.cat(outputs, dim=-1)
    empty, returned = model.forward_stream(z[..., :0], state)
    if returned is not state or transform(empty, mode).numel() != 0:
        raise RuntimeError("Trailing empty call changed streaming behavior")
    difference = actual.double() - expected.double()
    maximum = float(difference.abs().max())
    if actual.shape != expected.shape or sum(lengths) != expected.numel():
        raise RuntimeError("Streaming sample count differed")
    passed = maximum <= 2e-6
    if not passed and fail_on_tolerance:
        raise RuntimeError("Streaming parity exceeded unchanged 2e-6 threshold: " + str(maximum))
    return {"frames_per_chunk": frames_per_chunk, "chunk_ms": frames_per_chunk * 40,
            "latent_frames": z.shape[-1], "emitted_samples": sum(lengths),
            "expected_samples": expected.numel(), "chunks": len(lengths),
            "last_chunk_samples": lengths[-1], "max_abs_batch_stream_difference": maximum,
            "rms_batch_stream_difference": float(difference.square().mean().sqrt()),
            "threshold_absolute": 2e-6, "empty_calls_preserve_state": True,
            "history_shapes": [list(shape) for shape in shapes], "passed": passed,
            "tolerance_relaxed": False,
            "failure_reason": None if passed else "Batch-versus-stream maximum absolute difference exceeds 2e-6"}


@torch.inference_mode()
def timed_decode(model, z, mode, frames_per_chunk):
    start = time.perf_counter()
    state = model.initial_state()
    count = 0
    for chunk in z.split(frames_per_chunk, dim=-1):
        output, state = model.forward_stream(chunk, state)
        output = transform(output, mode)
        count += output.shape[-1]
    elapsed = time.perf_counter() - start
    if count != z.shape[-1] * 1920:
        raise RuntimeError("Timing loop lost output samples")
    return {"seconds": elapsed, "audio_seconds": count / 48000,
            "rtf": elapsed / (count / 48000), "samples": count}


def run(args):
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.restart_data import file_sha
    from audiovae_student.training import _rng_state, _restore_rng
    from diagnostic_common import load_context, atomic_json, status
    from native_chain import QUARTER_SHA, restore_quarter
    from run_update_experiment import load_canonical_panel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized in the CPU-only process")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    head_path = Path(args.heads).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong quarter checkpoint")
    head_sha = file_sha(head_path)
    rng = _rng_state()
    rng_sha = state_fingerprint(rng)
    engine = model = original_weight = modes = original = None
    result = None
    try:
        ctx = load_context()
        panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        engine = ctx.engine("targeted", device="cpu")
        original = restore_quarter(engine, saved["engine"])
        model = engine.model
        assert_cpu_tree(engine.state_dict())
        original_weight = model.output.weight.detach().clone()
        modes = tuple((m, m.training) for m in model.modules())
        model.eval()
        artifact = torch.load(head_path, map_location="cpu", weights_only=True)
        assert_cpu_tree(artifact)
        weights, candidates = validate_heads(artifact, original_weight, QUARTER_SHA)
        cases, timing_crop, timing_z = fixed_cases(panel)
        for _, crop in cases:
            assert_cpu_tree((crop.latents, crop.teacher_audio))
        identity = {"checkpoint_sha256": QUARTER_SHA, "head_file_sha256": head_sha,
            "canonical_panel": panel["identity"], "script_sha256": file_sha(Path(__file__)),
            "bounded_head_sha256": file_sha(Path(__file__).parent / "bounded_head.py"),
            "cpu": cpu_identity(), "torch": str(torch.__version__),
            "torch_build": torch.__config__.show(), "device": "cpu", "cuda_visible_devices": "",
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "teacher_forwards": 0, "encoder_forwards": 0,
            "scope": "Experimental PyTorch student only; not production ONNX, a Mimi comparison or final deployment RTF"}
        atomic_json(out / "identity.json", identity)
        parity = []
        for name, mode in candidates:
            with torch.no_grad():
                model.output.weight.copy_(weights[name][..., None])
            for role, crop in cases:
                for frames in (2, 4):
                    status("cpu_head_stream_parity", candidate=name + "_" + mode,
                           case=role, frames_per_chunk=frames)
                    row = check_streaming(model, crop.latents, mode, frames, fail_on_tolerance=False)
                    parity.append({"candidate": name + "_" + mode, "case": role,
                        "source_id": crop.source_id, "start_frame": crop.start_frame, **row})
            atomic_json(out / "streaming-parity.json", parity)
        with torch.no_grad():
            model.output.weight.copy_(original_weight)
        conditions = [(mode, frames) for frames in (2, 4) for mode in ("raw", "clamp", "tanh")]
        for _ in range(2):
            for mode, frames in conditions:
                timed_decode(model, timing_z, mode, frames)
        order_rng = random.Random(9060923)
        timings = {str(frames) + "_" + mode: [] for mode, frames in conditions}
        order = []
        for repeat in range(5):
            group = conditions.copy()
            order_rng.shuffle(group)
            for mode, frames in group:
                measurement = timed_decode(model, timing_z, mode, frames)
                key = str(frames) + "_" + mode
                timings[key].append(measurement)
                order.append({"repeat": repeat, "condition": key})
        summary = {}
        for mode, frames in conditions:
            key = str(frames) + "_" + mode
            summary[key] = {"mode": mode, "frames_per_chunk": frames, "chunk_ms": frames * 40,
                            "rtf_median": statistics.median(v["rtf"] for v in timings[key]),
                            "seconds_median": statistics.median(v["seconds"] for v in timings[key]),
                            "measurements": timings[key]}
        for mode, frames in conditions:
            key = str(frames) + "_" + mode
            summary[key]["rtf_ratio_to_same_chunk_raw"] = summary[key]["rtf_median"] / summary[str(frames) + "_raw"]["rtf_median"]
        result = {"version": 2, "complete": True,
            "completion_means": "All diagnostics collected; not candidate qualification",
            "all_streaming_checks_passed": all(row["passed"] for row in parity),
            "streaming_failure_count": sum(not row["passed"] for row in parity),
            "identity": identity, "streaming_parity": parity,
            "timing": {"weights": "unchanged baseline for every mode", "warmup_per_condition": 2,
                       "repeats": 5, "interleaved_order": order, "conditions": summary,
                       "source_id": timing_crop.source_id, "start_frame": timing_crop.start_frame,
                       "latent_shape": list(timing_z.shape), "latent_sha256": state_fingerprint(timing_z),
                       "initial_state_included_once_per_utterance": True,
                       "concatenation_and_quality_checks_in_timing": False},
            "checkpoint_promoted": False, "teacher_or_encoder_execution": False}
    finally:
        if model is not None and original_weight is not None:
            with torch.no_grad():
                model.output.weight.copy_(original_weight)
        if modes is not None:
            for module, training in modes:
                module.training = training
        _restore_rng(rng)
        if state_fingerprint(_rng_state()) != rng_sha:
            raise RuntimeError("CPU experiment failed to restore RNG")
        if engine is not None and original is not None and state_fingerprint(engine.state_dict()) != original:
            raise RuntimeError("CPU experiment failed to restore original engine")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU experiment unexpectedly initialized CUDA")
    if file_sha(checkpoint) != QUARTER_SHA or file_sha(head_path) != head_sha:
        raise RuntimeError("Input checkpoint or head artifact changed")
    for field in ("receipt", "cache"):
        if file_sha(panel["identity"][field + "_path"]) != panel["identity"][field + "_sha256"]:
            raise RuntimeError("Canonical panel input changed")
    result.update(original_engine_state_restored=True, original_model_modes_restored=True,
                  rng_restored=True, cuda_initialized=False, all_runtime_tensors_cpu=True)
    atomic_json(out / "cpu-heads.json", result)
    status("cpu_head_checks_complete", out=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("heads", "checkpoint", "canonical-receipt", "out"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
