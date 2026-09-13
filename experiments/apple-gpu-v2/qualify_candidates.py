"""Bounded public-API qualification of isolated MPS candidates; no import-time GPU work.

Run at most two arms per invocation. Compilation/first-call preparation is
reported separately; ordinary completed GPU API calls plus synchronized state
inspection share a 15-second limit. This is qualification, not an RTF benchmark.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for key in ("PYTORCH_MPS_FAST_MATH", "PYTORCH_ENABLE_MPS_FALLBACK", "TORCHINDUCTOR_USE_FAST_MATH"):
    if os.environ.get(key, "0") != "0":
        raise RuntimeError(key + " must be disabled before Torch import")
    os.environ[key] = "0"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import experiment as exp
import numpy as np
import onnxruntime as ort
import torch
import torch._dynamo.config as dynamo_config
from torch._dynamo.utils import counters
from fast_audiovae.gpu import GPUDecoder
from fast_audiovae.assets import MODEL_FILES, verify_model


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def compare(value, reference):
    require(isinstance(value, np.ndarray) and isinstance(reference, np.ndarray)
            and value.dtype == reference.dtype == np.float32 and value.shape == reference.shape
            and np.isfinite(value).all() and np.isfinite(reference).all(), "Invalid numerical comparison")
    difference = np.abs(value.astype(np.float64) - reference)
    return dict(passed=bool(np.allclose(value, reference, atol=1e-5, rtol=1e-4)),
                max_abs=float(difference.max()) if difference.size else 0.0)


def counter_snapshot():
    return {str(group): {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()}


def main(output, arms):
    require(1 <= len(arms) <= 2 and len(set(arms)) == len(arms), "Choose one or two unique arms for the bounded invocation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write('{"status":"starting"}\n')
    result = dict(version="apple_gpu_candidate_qualification_v1", status="running", arms=arms,
                  calls=0, gpu_api_calls=0, cpu_reference_calls=0, compilation_first_calls=0,
                  gpu_api_seconds=0.0, state_inspection_seconds=0.0, cpu_reference_seconds=0.0,
                  compilation_first_call_seconds=0.0, preparation=[], checks=[], state_summary={},
                  files={}, state_snapshot_checks=[], per_call=[])

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def budget():
        require(result["calls"] < 250, "250-call qualification limit reached")
        require(result["gpu_api_seconds"] + result["state_inspection_seconds"] < 15,
                "15-second ordinary GPU-call/state-inspection budget reached")

    def call(kind, label, fn):
        budget()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed = time.perf_counter() - start
            result["calls"] += 1
            if kind == "compile":
                result["compilation_first_calls"] += 1
                result["compilation_first_call_seconds"] += elapsed
            elif kind == "cpu":
                result["cpu_reference_calls"] += 1
                result["cpu_reference_seconds"] += elapsed
            else:
                result["gpu_api_calls"] += 1
                result["gpu_api_seconds"] += elapsed
            result["per_call"].append(dict(kind=kind, label=label, seconds=elapsed))
            budget()

    def inspect(fn):
        budget()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            result["state_inspection_seconds"] += time.perf_counter() - start
            budget()

    def check(label, value, reference):
        row = dict(label=label, **compare(value, reference))
        result["checks"].append(row)
        require(row["passed"], "Numerical gate failed: " + label)

    def states_cpu(history, shapes):
        require(set(history) == set(shapes), "Incomplete state mapping")
        for name, shape in shapes.items():
            value = history[name]
            require(isinstance(value, torch.Tensor) and value.device.type == "mps"
                    and value.dtype == torch.float32 and tuple(value.shape) == tuple(shape)
                    and value.is_contiguous(), "State dtype/device/shape/contiguity mismatch: " + name)
        with torch.inference_mode():
            flat = torch.cat([history[name].reshape(-1) for name in shapes])
            ready = flat.to("cpu", non_blocking=False)
            torch.mps.synchronize()
            owned = ready.numpy().copy()
        require(np.isfinite(owned).all(), "Nonfinite state")
        return owned

    lock = (HERE / "gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        dynamo_config.suppress_errors = False
        dynamo_config.fail_on_recompile_limit_hit = True
        require(torch.__version__.split("+")[0] == "2.14.0" and torch.backends.mps.is_available()
                and torch.get_default_dtype() == torch.float32, "Wrong FP32/MPS runtime")
        config = json.loads(exp.CONFIG.read_text())
        paths = [Path(__file__), HERE / "experiment.py", HERE / "mps_decoder_before.py",
                 HERE / "snake_candidates.py", exp.CONFIG,
                 ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae/mps_decoder.py",
                 ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae/gpu.py"]
        result["files"] = {str(path): exp.sha(path) for path in paths}
        verify_model(exp.ORIGINAL)
        result["files"].update({str(exp.ORIGINAL.parent / name): digest for name, digest in MODEL_FILES.items()})
        result["runtime"] = dict(torch=torch.__version__, onnxruntime=ort.__version__,
                                 numpy=np.__version__, fast_math=False, cpu_fallback=False)
        for key in ("latents", "extra_latents"):
            require(exp.sha(config[key]) == config["inputs_sha256"][key], "Latent asset checksum mismatch")
            result["files"][config[key]] = config["inputs_sha256"][key]
        crops = {}
        with np.load(config["latents"], allow_pickle=False) as archive:
            for label, uid in exp.CLIPS:
                crops[label] = np.ascontiguousarray(archive[uid + "__z"][..., :7])
        with np.load(config["extra_latents"], allow_pickle=False) as archive:
            names = [name for name in archive.files if name.endswith("__z")]
            require(len(names) >= 2, "Missing existing expressive fixtures")
            for index, name in enumerate(names[:2]):
                crops[f"Expressive{index}"] = np.ascontiguousarray(archive[name][..., :7])
        crops["Zero latent"] = np.zeros((1, 64, 7), np.float32)
        crops["Quiet latent"] = crops["English"] * np.float32(.001)
        require(len(crops) == 7 and all(z.shape == (1, 64, 7) and z.dtype == np.float32
                and np.isfinite(z).all() for z in crops.values()), "Wrong seven-frame fixture inventory")
        result["inputs"] = [dict(label=label, frames=7, sha256=digest(z)) for label, z in crops.items()]
        result["protocol"] = dict(atol=1e-5, rtol=1e-4, precision="FP32", host_threads=1,
            fixed_compile_lengths=[1, 2], mixed_partition=[2, 1, 2, 2], empty_on_public_fast_path=True,
            cpu_reference="Original pinned full ONNX on the same seven-frame prefix.",
            state_reference="Separate eager model with the candidate's identical overlap representation; each arm evolves its own state.",
            state_checks="All26 states: exact keys/shape/device/FP32/contiguity, finite values, per-state tolerance; one noninitial pre-call original-byte snapshot for both streams per arm.",
            budget="Ordinary completed GPU API plus synchronized state-inspection wall time <15s; <250 total model/API calls including CPU references and preparation.",
            exclusions="Model construction/shader compilation and the four first full_compile/snake_compile calls are separately reported. Those first-call wall times include compilation AND decode, not pure compiler time.",
            benchmark=False, suppress_errors=False, fail_on_recompile_limit_hit=True)
        reference = exp.cpu_reference(exp.ORIGINAL)
        refs = {label: call("cpu", label + "/full", lambda z=z: reference.run(None, {"z": z})[0])
                for label, z in crops.items()}
        short_refs = {(label, count): call("cpu", f"{label}/prefix{count}",
                      lambda label=label, count=count: reference.run(None, {"z": crops[label][..., :count].copy()})[0])
                      for label, count in (("English", 2), ("English", 3), ("Spanish", 3), ("English", 6))}

        for arm in arms:
            before = time.perf_counter()
            model = exp.build(arm)
            eager_name = "before" if arm in ("before", "before_compile") else "overlap"
            eager_model = exp.build(eager_name)
            require(model.state_shapes == eager_model.state_shapes and len(model.state_shapes) == 26,
                    "Candidate/eager state representations differ")
            shapes = {name: tuple(shape) for name, shape in model.state_shapes.items()}
            candidate = GPUDecoder(model, torch, {"experiment": arm}, "streaming")
            eager = GPUDecoder(eager_model, torch, {"experiment": eager_name}, "streaming")
            torch.mps.synchronize()
            result["preparation"].append(dict(arm=arm, build_seconds=time.perf_counter() - before,
                                              state_shapes=shapes, state_bytes=model.state_bytes,
                                              source_identity=getattr(model, "model", model).source_identity))
            summary = {name: dict(comparisons=0, max_abs=0.0, passed=True) for name in shapes}
            result["state_summary"][arm] = summary
            snapshot_done = False

            def pair(label, cs, es, packet, preparing=False, snapshot=False):
                nonlocal snapshot_done
                originals = packet.tobytes()
                saved = None
                if snapshot and not snapshot_done:
                    old = (dict(cs._history), dict(es._history))
                    saved = tuple(inspect(lambda h=h: states_cpu(h, shapes)).tobytes() for h in old)
                kind = "compile" if preparing and arm in ("before_compile", "full_compile", "snake_compile") else "gpu"
                cy = call(kind, arm + "/" + label, lambda: cs.decode_chunk(packet))
                ey = call("gpu", eager_name + "/" + label, lambda: es.decode_chunk(packet))
                require(packet.tobytes() == originals, "Latent input was mutated")
                check(arm + "/" + label + "/audio-eager", cy, ey)
                a = inspect(lambda: states_cpu(cs._history, shapes))
                b = inspect(lambda: states_cpu(es._history, shapes))
                offset = 0
                for name, shape in shapes.items():
                    count = int(np.prod(shape))
                    compared = compare(a[offset:offset + count], b[offset:offset + count])
                    summary[name]["comparisons"] += 1
                    summary[name]["max_abs"] = max(summary[name]["max_abs"], compared["max_abs"])
                    summary[name]["passed"] &= compared["passed"]
                    require(compared["passed"], "State numerical gate failed: " + name)
                    offset += count
                if saved is not None:
                    require(all(inspect(lambda h=h: states_cpu(h, shapes)).tobytes() == value
                                for h, value in zip(old, saved)), "A pre-call original state tensor was mutated")
                    result["state_snapshot_checks"].append(dict(arm=arm, tensors_per_stream=26,
                        streams=2, before_sha256=[hashlib.sha256(value).hexdigest() for value in saved], exact=True))
                    snapshot_done = True
                return cy, ey

            # Both fixed shapes see fresh and carried states before case gates.
            with candidate.stream() as cs, eager.stream() as es:
                chunks, position = [], 0
                for count in (1, 2, 1, 2):
                    y, _ = pair(f"preparation/{position}", cs, es,
                                crops["English"][..., position:position + count].copy(), preparing=True)
                    chunks.append(y); position += count
                check(arm + "/preparation/full-reference", np.concatenate(chunks, -1), short_refs["English", 6])
            if isinstance(model, exp.CompiledModel):
                require(set(model.compiled) == {1, 2}, "Unexpected compiled packet specialization")
            result["preparation"][-1]["compiler_counters"] = counter_snapshot()
            for label, z in crops.items():
                with candidate.stream() as cs, eager.stream() as es:
                    parts, eager_parts, position = [], [], 0
                    for count in (2, 1, 2, 2):
                        cy, ey = pair(label + f"/{position}", cs, es, z[..., position:position + count].copy(),
                                      snapshot=position > 0)
                        parts.append(cy); eager_parts.append(ey); position += count
                    require(cs.frames_decoded == es.frames_decoded == 7, "Wrong decoded frame count")
                    check(arm + "/" + label + "/original-full", np.concatenate(parts, -1), refs[label])
                    check(eager_name + "/" + label + "/original-full", np.concatenate(eager_parts, -1), refs[label])
                save()
            # Public empty input/flush must not initialize or advance histories.
            with candidate.stream() as cs, eager.stream() as es:
                for label, stream in ((arm, cs), (eager_name, es)):
                    for operation, fn in (("initial-empty", lambda s=stream: s.decode_chunk(np.empty((1, 64, 0), np.float32))),
                                          ("flush", stream.flush)):
                        y = call("gpu", label + "/" + operation, fn)
                        require(y.shape == (1, 1, 0) and y.dtype == np.float32
                                and stream.frames_decoded == 0 and stream._history == {}, "Empty/flush mutated state")
            z = crops["English"][..., :2].copy()
            with candidate.stream() as cs, eager.stream() as es:
                original, eager_original = pair("reset/before", cs, es, z)
                cs.reset(); es.reset()
                require(cs._history == es._history == {} and cs.frames_decoded == es.frames_decoded == 0,
                        "Reset did not clear state")
                repeated, _ = pair("reset/after", cs, es, z)
                check(arm + "/reset-repeat", repeated, original)
                check(arm + "/reset-original-full", repeated, short_refs["English", 2])
            changed = z.copy(); changed[..., 1:] += np.float32(.375)
            with candidate.stream() as cs, eager.stream() as es:
                future, eager_future = pair("changed-future", cs, es, changed)
                check(arm + "/prefix-invariance", future[..., :1920], original[..., :1920])
                check(eager_name + "/prefix-invariance", eager_future[..., :1920], eager_original[..., :1920])
            with candidate.stream() as ca, candidate.stream() as cb, eager.stream() as ea, eager.stream() as eb:
                a1, _ = pair("interleave/A1", ca, ea, crops["English"][..., :1].copy())
                b1, _ = pair("interleave/B1", cb, eb, crops["Spanish"][..., :2].copy())
                a2, _ = pair("interleave/A2", ca, ea, crops["English"][..., 1:3].copy())
                b2, _ = pair("interleave/B2", cb, eb, crops["Spanish"][..., 2:3].copy())
                check(arm + "/interleave/A-full", np.concatenate((a1, a2), -1), short_refs["English", 3])
                check(arm + "/interleave/B-full", np.concatenate((b1, b2), -1), short_refs["Spanish", 3])
            require(snapshot_done and all(row["passed"] and row["comparisons"] == 39 for row in summary.values()),
                    "Incomplete per-state qualification")
            if isinstance(model, exp.CompiledModel):
                require(set(model.compiled) == {1, 2}, "A non-L1/L2 graph was compiled")
            result["preparation"][-1]["final_compiler_counters"] = counter_snapshot()
            save()
            print(arm, "passed", result["calls"], "calls", flush=True)
            del candidate, eager, model, eager_model
        require(result["calls"] == 11 + 82 * len(arms), "Incomplete fixed call accounting")
        require(all(exp.sha(path) == result["files"][str(path)] for path in paths), "Qualification source changed")
        require(all(digest(crops[row["label"]]) == row["sha256"] for row in result["inputs"]), "Fixture changed")
        result.update(status="passed", artifacts_unchanged=True, completed_arms=len(arms))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        lock.close()
    print(json.dumps({key: result[key] for key in ("status", "calls", "gpu_api_seconds", "state_inspection_seconds",
                                                   "compilation_first_call_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arms", required=True, nargs="+", choices=("before", "before_compile", "overlap", "full_compile", "snake_compile", "snake_metal"))
    args = parser.parse_args()
    main(args.output.resolve(), args.arms)
