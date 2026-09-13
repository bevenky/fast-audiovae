"""Four-packet numerical diagnostic, not a benchmark or acceptance relaxation.

Trace original eager versus V3 transpose-only at every operation. The additional
paired-projection arm is checked at all transposes, histories and final audio.
Each local transpose check uses the original arm's exact input and history.
Only aggregate errors and hashes are saved. No import-time model/GPU work.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import uuid

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "apple-gpu-v3"
V2 = HERE.parent / "apple-gpu-v2"
PARTITION = (2, 1, 2, 2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output, fixture="English"):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "TORCHINDUCTOR_USE_FAST_MATH"):
        if os.environ.get(key, "0") != "0":
            raise RuntimeError(key + " must be disabled before Torch import")
        os.environ[key] = "0"
    os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    sys.path[:0] = [str(V3), str(V2), str(HERE)]
    import experiment as exp
    import screen_transpose as screen
    import candidates
    import numpy as np
    import torch
    from fast_audiovae.gpu import GPUDecoder
    import fcntl

    files = {str(p): sha(p) for p in (Path(__file__).resolve(), HERE / "candidates.py",
             V3 / "screen_transpose.py", V2 / "experiment.py", V2 / "mps_decoder_before.py",
             exp.ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae/gpu.py", exp.CONFIG)}
    result = dict(version="apple_gpu_divergence_trace_v1", status="running", files=files,
        protocol=dict(frames=7, partition=list(PARTITION), precision="FP32", threads=1,
            atol=1e-5, rtol=1e-4, maximum_tensor_comparisons=999,
            original="Original eager literal-weight decoder with activated-input histories",
            v3="Exact existing transpose_mm builder, eager model plus four compiled calls",
            paired="V4 paired projection, eager only; original pointwise convolutions",
            local="Both alternative transposes receive original eager input and history",
            compiled="Separate four public API calls, compared with each saved eager reference endpoint",
            benchmark=False, fp64_oracle=False, cpu_inference_fallback=False,
            limitations="FP32 comparison locates divergence; it does not establish which implementation is closer to exact arithmetic."),
        comparisons=[], compiled_api_calls=0, manual_packets=0)

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def cpu(value):
        if isinstance(value, np.ndarray):
            return value
        assert value.device.type == "mps" and value.dtype == torch.float32
        ready = value.detach().to("cpu", non_blocking=False)
        torch.mps.synchronize()
        return ready.numpy().copy()

    def compare(packet, label, arm, scope, actual, reference):
        assert len(result["comparisons"]) < 999, "Comparison limit reached"
        a, b = cpu(actual), cpu(reference)
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite diagnostic tensor"
        difference = a.astype(np.float64) - b.astype(np.float64)
        absolute = np.abs(difference)
        ratio = absolute / (1e-5 + 1e-4 * np.abs(b.astype(np.float64)))
        failures = int(np.count_nonzero(ratio > 1))
        result["comparisons"].append(dict(packet=packet, label=label, arm=arm, scope=scope,
            shape=list(a.shape), elements=int(a.size), max_abs=float(absolute.max()) if a.size else 0.,
            rms=float(np.sqrt(np.mean(difference * difference))) if a.size else 0.,
            failing_elements=failures, max_tolerance_ratio=float(ratio.max()) if a.size else 0.,
            passed=failures == 0, bitwise_equal=a.tobytes() == b.tobytes()))

    lock = (V2 / "gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.__version__.split("+")[0] == "2.14.0" and torch.backends.mps.is_available()
        result["torch"] = torch.__version__
        # Capture the exact local builder after its CPU algebra gates. Raising
        # this private sentinel avoids the outer screen's performance sweep and
        # creates no temporary screen result. Its finally restores exp.build.
        captured = {}
        original_main, original_build = exp.main, exp.build
        class BuilderCaptured(Exception):
            pass
        def capture(args):
            assert args.arms == ["before_compile", "transpose_mm"]
            captured["build"] = exp.build
            raise BuilderCaptured()
        exp.main = capture
        try:
            screen.main(argparse.Namespace(name="trace-capture-" + uuid.uuid4().hex,
                                           cpu_only=False, combined=False))
        except BuilderCaptured:
            pass
        finally:
            exp.main, exp.build = original_main, original_build
        assert "build" in captured
        reference = original_build("before")
        compiled = captured["build"]("transpose_mm")
        paired = original_build("before")
        assert candidates.apply_paired_projection(paired) == 6
        models = (reference, compiled.model, paired)
        assert all(m.state_shapes == reference.state_shapes for m in models)
        assert len(reference.state_shapes) == 26
        result["source_identity"] = reference.source_identity
        result["state_shapes"] = reference.state_shapes
        config = json.loads(exp.CONFIG.read_text())
        asset = "latents" if fixture == "English" else "extra_latents"
        assert sha(config[asset]) == config["inputs_sha256"][asset]
        result["files"][config[asset]] = config["inputs_sha256"][asset]
        with np.load(config[asset], allow_pickle=False) as archive:
            if fixture == "English":
                uid = next(uid for label, uid in exp.CLIPS if label == "English")
                key = uid + "__z"
            else:
                assert fixture == "Expressive0"
                key = next(name for name in archive.files if name.endswith("__z"))
                uid = key.removesuffix("__z")
            z = np.ascontiguousarray(archive[key][..., :7])
        assert z.shape == (1, 64, 7) and z.dtype == np.float32 and np.isfinite(z).all()
        result["input"] = dict(fixture=fixture, uid=uid, frames=7, sha256=hashlib.sha256(z.tobytes()).hexdigest())
        histories = [{}, {}, {}]
        snapshots = []
        position = 0
        with torch.inference_mode(), torch.autocast("mps", enabled=False):
            for packet, length in enumerate(PARTITION):
                latent = torch.from_numpy(z[..., position:position+length].copy()).to("mps")
                old = [m._checked_states(latent, h) for m, h in zip(models, histories)]
                new = [{}, {}, {}]
                reference_states = {}

                def tensor(label, values, check_paired=False):
                    r = cpu(values[0])
                    compare(packet, label, "v3_transpose_mm", "propagated", values[1], r)
                    if check_paired:
                        compare(packet, label, "paired_projection", "propagated", values[2], r)
                    return values

                def causal(label, modules, values, key, transpose=False):
                    pairs = [m.decode(x, h[key]) for m, x, h in zip(modules, values, old)]
                    outputs = tensor(label, [p[0] for p in pairs], check_paired=transpose)
                    if transpose:
                        compare(packet, label + "/input", "paired_projection", "incoming", values[2], values[0])
                        for i, arm in ((1, "v3_transpose_mm"), (2, "paired_projection")):
                            local, _ = modules[i].decode(values[0], old[0][key])
                            compare(packet, label, arm, "local_same_input_history", local, outputs[0])
                    for i, pair in enumerate(pairs):
                        state = pair[1]
                        assert tuple(state.shape) == reference.state_shapes[key] and state.is_contiguous()
                        new[i][key] = state
                    r = cpu(new[0][key]); reference_states[key] = r
                    for i, arm in ((1, "v3_transpose_mm"), (2, "paired_projection")):
                        compare(packet, key, arm, "propagated_state", new[i][key], r)
                    return outputs

                xs = causal("stem.depthwise", [m.stem for m in models], [latent]*3, "stem.history")
                xs = tensor("stem.pointwise", [m.pointwise(x) for m, x in zip(models, xs)])
                for si in range(6):
                    stages = [m.stages[si] for m in models]
                    prefix = f"stage{si}"
                    xs = tensor(prefix+".scale", [x*s.scale for x, s in zip(xs, stages)])
                    xs = tensor(prefix+".offset", [x+s.offset for x, s in zip(xs, stages)])
                    xs = tensor(prefix+".snake", [s.snake(x) for x, s in zip(xs, stages)])
                    xs = causal(prefix+".transpose", [s.transpose for s in stages], xs,
                                prefix+".transpose.history", transpose=True)
                    for ri in range(3):
                        residuals = [s.residuals[ri] for s in stages]
                        name = prefix+f".residual{ri}"
                        branch = tensor(name+".pre_snake", [r.before(x) for r, x in zip(residuals, xs)])
                        branch = causal(name+".depthwise", [r.depthwise for r in residuals], branch, name+".history")
                        branch = tensor(name+".post_snake", [r.after(x) for r, x in zip(residuals, branch)])
                        branch = tensor(name+".pointwise", [r.pointwise(x) for r, x in zip(residuals, branch)])
                        xs = tensor(name+".residual_add", [x+b for x, b in zip(xs, branch)])
                xs = tensor("final.snake", [m.final_snake(x) for m, x in zip(models, xs)])
                xs = causal("final.convolution", [m.final_conv for m in models], xs, "final.history")
                xs = tensor("final.tanh", [torch.tanh(x) for x in xs], check_paired=True)
                assert all(set(h) == set(reference.state_shapes) for h in new)
                histories = new
                snapshots.append(dict(audio=cpu(xs[0]), states=reference_states))
                position += length; result["manual_packets"] += 1; save()
            assert position == 7
            # Separate compiled path; public API owns upload, all-state finite
            # checks, contiguity and CPU completion. Compilation is diagnostic,
            # not timed or confused with an eager per-operation measurement.
            decoder = GPUDecoder(compiled, torch, {"experiment": "v3-transpose-mm-trace"}, "streaming")
            position = 0
            with decoder.stream() as stream:
                for packet, length in enumerate(PARTITION):
                    y = stream.decode_chunk(z[..., position:position+length].copy())
                    result["compiled_api_calls"] += 1
                    compare(packet, "audio", "compiled_v3_transpose_mm", "compiled_propagated", y, snapshots[packet]["audio"])
                    for key, state in stream._history.items():
                        assert state.is_contiguous()
                        compare(packet, key, "compiled_v3_transpose_mm", "compiled_state", state, snapshots[packet]["states"][key])
                    position += length; save()
                assert stream.frames_decoded == 7 and set(stream._history) == set(reference.state_shapes)
            assert set(compiled.compiled) == {1, 2}
        assert result["manual_packets"] == result["compiled_api_calls"] == 4
        assert len(result["comparisons"]) == 892, len(result["comparisons"])
        result["compile_counters"] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()
                                      if k in ("stats", "frames", "graph_break", "unimplemented")}
        assert 0 < result["compile_counters"].get("stats", {}).get("unique_graphs", 0) <= 2
        assert all(sha(p) == h for p, h in files.items())
        result["comparison_count"] = len(result["comparisons"])
        result["failed_comparisons"] = sum(not r["passed"] for r in result["comparisons"])
        result["all_numerical_gates_passed"] = result["failed_comparisons"] == 0
        result["first_failure_by_arm_scope"] = {}
        for row in result["comparisons"]:
            if not row["passed"]:
                result["first_failure_by_arm_scope"].setdefault(row["arm"]+"/"+row["scope"], row)
        result.update(status="completed", artifacts_unchanged=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save(); lock.close()
    print(json.dumps({k: result[k] for k in ("status", "comparison_count", "failed_comparisons", "first_failure_by_arm_scope")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=("English", "Expressive0"), default="English")
    args = parser.parse_args()
    main(args.output.resolve(), args.fixture)
