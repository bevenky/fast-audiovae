"""Short public-package / actual upstream / Pocket Mimi MPS comparison.

No model is constructed at import. Run only under the parent's exclusive GPU
lease. Original artifacts and earlier experiment scripts are read-only.
"""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae"
for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "TORCHINDUCTOR_USE_FAST_MATH"):
    if os.environ.get(key, "0") != "0":
        raise RuntimeError(key + " must be disabled before Torch import")
    os.environ[key] = "0"
if os.environ.get("PYTORCH_MPS_PREFER_METAL", "0") != "0":
    raise RuntimeError("PYTORCH_MPS_PREFER_METAL must be unset or zero")
os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(HERE / "inductor-cache")
sys.path.insert(0, str(ROOT / "outputs/apple-gpu-v1"))
import benchmark_mps as common
import mimi_mps
import numpy as np
import onnxruntime as ort
import torch
import fast_audiovae
from fast_audiovae.assets import MODEL_FILES, verify_model
from fast_audiovae.gpu import _latents, _policy

require, sha = common.require, common.sha
ARMS = ("upstream_audiovae2_gpu", "fast_audiovae2_gpu", "pocket_mimi_gpu")
UPSTREAM_RECEIPT = Path("/Users/venky/tech/pockettts/benchmarks/results/fleurs-en-pilot/audiovae2.json")
UPSTREAM_SOURCE_SHA = "2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8"
UPSTREAM_WEIGHTS_SHA = "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"


def compiler_counters():
    return {k: dict(v) for k, v in torch._dynamo.utils.counters.items()
            if k in ("stats", "frames", "graph_break", "unimplemented")}


def public_model(mode):
    model = fast_audiovae.load(device="gpu", mode=mode, threads=1,
        source=common.ORIGINAL, offline=True, cache_dir=HERE / "cache")
    info = model.info
    require(info["device"] == "gpu" and info["backend"] == "mps"
            and info["selected"] == "torch_mps_hybrid"
            and info["recipe"] == "apple_mps_hybrid_v1"
            and info["precision"] == "FP32" and info["threads"] == 1
            and info["fallback"] is False and info["mps_cpu_fallback"] is False
            and info["mps_fast_math"] is False and info["mode"] == mode,
            "Public loader did not select the integrated MPS implementation")
    return model


def ready_upstream(upstream, z, stream=None):
    """V5 boundary, with the unmodified upstream class and weight-norm hooks."""
    _policy()
    owned = _latents(z)
    x = torch.from_numpy(owned).to("mps", dtype=torch.float32, non_blocking=False)
    y = upstream.decode(x) if stream is None else stream.decode_chunk(x)
    require(y.device.type == "mps" and y.dtype == torch.float32
            and tuple(y.shape) == (1, 1, z.shape[-1] * 1920), "Invalid upstream MPS output")
    values = [y]
    if stream is not None:
        require(len(stream._states) == 26 and all(v.device.type == "mps"
                and v.dtype == torch.float32 for v in stream._states.values()), "Invalid upstream states")
        values += list(stream._states.values())
    require(torch.isfinite(torch.cat([v.reshape(-1) for v in values])).all().item(),
            "Nonfinite upstream output/history")
    ready = y.detach().to("cpu", non_blocking=False)
    torch.mps.synchronize()
    result = ready.numpy().copy()
    common.audio_check(result, z.shape[-1] * 1920)
    return result


def main(output):
    output = output.expanduser().resolve()
    require(output.parent == HERE and output.suffix == ".json", "Use a new JSON output in apple-gpu-v6")
    with output.open("x") as handle:
        handle.write('{"status":"starting"}\n')
    result = dict(version="apple_gpu_public_matched_v6", status="running", checks=[],
        calls=[], rows=[], clips=[], preparation=[], ordinary_call_seconds=0., summary=[])
    lock = (ROOT / "outputs/apple-gpu-v2/gpu.lock").open("a")
    mimi = None
    started = time.perf_counter()

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def call(kind, label, fn):
        require(result["ordinary_call_seconds"] < 25, "25-second all-ordinary-call cap exhausted")
        row = dict(kind=kind, label=label, status="running")
        result["calls"].append(row)
        begin = time.perf_counter()
        try:
            answer = fn()
            row["status"] = "returned"
            return answer
        finally:
            row["seconds"] = time.perf_counter() - begin
            result["ordinary_call_seconds"] += row["seconds"]
            require(result["ordinary_call_seconds"] <= 25, "25-second all-ordinary-call cap exceeded")

    def check(label, value, reference):
        row = dict(label=label, **common.comparison(value, reference))
        result["checks"].append(row)
        require(row["passed"], "Original waveform gate failed: " + str(row))
        return row

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.__version__.split("+")[0] == "2.14.0" and ort.__version__ == "1.30.0"
                and torch.get_default_dtype() == torch.float32 and torch.backends.mps.is_available(),
                "Wrong runtime or FP32/MPS policy")
        require(Path(fast_audiovae.__file__).resolve().parent == SOURCE, "Wrong public package source")
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        torch._dynamo.config.error_on_recompile = False
        config = json.loads(common.CONFIG.read_text())
        provenance = json.loads(UPSTREAM_RECEIPT.read_text())["provenance"]
        source, checkpoint = Path(provenance["source_path"]), Path(provenance["checkpoint"])
        require(provenance["source_sha256"] == UPSTREAM_SOURCE_SHA
                and provenance["checkpoint_sha256"] == UPSTREAM_WEIGHTS_SHA, "Wrong actual upstream identity")
        verify_model(common.ORIGINAL)
        expected = {source: UPSTREAM_SOURCE_SHA, checkpoint: UPSTREAM_WEIGHTS_SHA,
                    Path(config["mimi_bundle"]) / "mimi_full.onnx": common.MIMI_FULL}
        for key in ("latents", "mimi_latents"):
            expected[Path(config[key])] = config["inputs_sha256"][key]
        for name, value in MODEL_FILES.items():
            expected[common.ORIGINAL.parent / name] = value
        for path, value in expected.items():
            require(sha(path) == value, "Changed input artifact: " + str(path))
        source_files = [Path(__file__).resolve(), Path(common.__file__), Path(mimi_mps.__file__),
                        common.CONFIG, UPSTREAM_RECEIPT, source] + sorted(SOURCE.glob("*.py"))
        files = {**expected, **{p: sha(p) for p in source_files}}
        stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        result["files"] = {str(p): h for p, h in files.items()}
        result["protocol"] = dict(arms=list(ARMS), clips=3, duration_ms=960,
            audio_packet_ms=[40, 80], mimi_packet_ms=[80], mimi_40_ms="unsupported: one latent represents80ms",
            audio_sample_rate=48000, mimi_sample_rate=24000, precision="FP32", host_threads=1,
            qualification_sweeps=1, warmup_sweeps=1, measured_sweeps=2,
            expected_streams=60, expected_stream_packets=1008, expected_flush_calls=60,
            short_panel_ordinary_calls=1085, public_load_internal_preparation_calls=2,
            ordinary_call_cap_seconds=25, atol=1e-5, rtol=1e-4,
            crops="Middle shared960ms,80ms-aligned; fresh state at the crop start, no context replay",
            order="Reverse complete arm order with clip+repetition parity; each measured matched group has both orders",
            references="Actual upstream AudioVAE2 full MPS on same crops; additionally original full CPU ONNX for each codec",
            timing="Owned CPU input through completed GPU decode and owned CPU output, including validation and synchronization; packet+flush times",
            exclusions="Model/session/stream construction, compilation in public load, initialization sync, close, hashing and numerical comparison. Lazy packet history allocation stays timed",
            boundary="Upstream and public Audio validate audio plus26 histories. Mimi wrapper validates input/audio, not all histories. All return completed CPU copies",
            upstream_flush="Official upstream has no flush method; a labeled empty host shim is timed, with no model call or pending audio",
            limitations="Two measured rounds are short evidence; no outlier removal, significance claim, or upstream-state equality claim")
        result["runtime"] = dict(torch=torch.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
            platform=platform.platform(), fast_math=False, cpu_fallback=False)
        crops = {}
        with np.load(config["latents"], allow_pickle=False) as a, np.load(config["mimi_latents"], allow_pickle=False) as m:
            for label, uid in common.CLIPS:
                az, mz = a[uid + "__z"], m[uid + "__z"]
                require(az.dtype == mz.dtype == np.float32 and az.shape[:2] == (1, 64)
                        and mz.shape[:2] == (1, 32), "Wrong latent schema")
                available = min(az.shape[-1] // 2, mz.shape[-1])
                require(available >= 12, "Clip shorter than960ms")
                start = (available - 12) // 2
                crops[label] = dict(audio=np.ascontiguousarray(az[..., 2*start:2*start+24]),
                                    mimi=np.ascontiguousarray(mz[..., start:start+12]))
                require(all(np.isfinite(v).all() for v in crops[label].values()), "Nonfinite latents")
                result["clips"].append(dict(label=label, uid=uid, start_ms=start*80,
                    audio_frames=24, mimi_frames=12, audio_samples=46080, mimi_samples=23040,
                    sha256={k: common.tensor_sha(v) for k, v in crops[label].items()}))
                if label == "English":
                    continuation = np.ascontiguousarray(az[..., :min(200, az.shape[-1])])
                    require(continuation.shape[-1] > 0, "Missing natural English continuation")
                    result["continuation"] = dict(uid=uid, available_frames=az.shape[-1],
                        frames=continuation.shape[-1], duration_seconds=continuation.shape[-1]*.04,
                        sha256=common.tensor_sha(continuation), source="Natural English prefix, no repetition or concatenation",
                        packet_ms=[40, 80], included_in_rtf=False, rows=[])
        save()
        with torch.inference_mode(), torch.autocast("mps", enabled=False):
            begin = time.perf_counter()
            spec = importlib.util.spec_from_file_location("_actual_upstream_audiovae2_v6", source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            upstream = module.AudioVAE(module.AudioVAEConfig())
            upstream.load_state_dict(weights.get("state_dict", weights), strict=True)
            upstream.eval().requires_grad_(False).to("mps", dtype=torch.float32)
            require(all(p.device.type == "mps" and p.dtype == torch.float32 for p in upstream.parameters()), "Upstream parameters not FP32 MPS")
            require(sum(hasattr(m, "weight_g") for m in upstream.decoder.modules()) == 45, "Weight normalization removed")
            del weights
            torch.mps.synchronize()
            result["preparation"].append(dict(arm=ARMS[0], seconds=time.perf_counter()-begin,
                class_name=spec.name + ".AudioVAE", provenance=provenance, retained_weight_norm_modules=45))
            for mode in ("streaming", "batch"):
                begin = time.perf_counter()
                loaded = public_model(mode)
                torch.mps.synchronize()
                result["preparation"].append(dict(arm=ARMS[1], mode=mode, seconds=time.perf_counter()-begin,
                    info=loaded.info, internal_preparation_calls=2 if mode == "streaming" else 0,
                    note="Load duration includes verification/construction and any public load-time compile+decode preparation"))
                if mode == "streaming":
                    fast = loaded
                    require(set(fast._model.compiled) == {1, 2}, "Public fixed-shape preparation incomplete")
                else:
                    batch = loaded
            begin = time.perf_counter()
            mimi = mimi_mps.MimiMPSAdapter(max_frames=12)
            torch.mps.synchronize()
            result["preparation"].append(dict(arm=ARMS[2], seconds=time.perf_counter()-begin, metadata=mimi.metadata))
            result["compile_counters_after_load"] = compiler_counters()
            torch._dynamo.config.error_on_recompile = True
            sessions = {"audio": common.cpu_reference(common.ORIGINAL),
                        "mimi": common.cpu_reference(Path(config["mimi_bundle"]) / "mimi_full.onnx")}
            refs = {}
            for label, values in crops.items():
                refs[label] = {}
                for codec, z in values.items():
                    session = sessions[codec]
                    refs[label][codec] = call("cpu_reference", label + "/" + codec,
                        lambda s=session, z=z: s.run(None, {s.get_inputs()[0].name: z})[0])
                refs[label]["upstream"] = call("upstream_full_reference", label,
                    lambda: ready_upstream(upstream, values["audio"]))
                check(label + "/upstream-full-vs-original-onnx", refs[label]["upstream"], refs[label]["audio"])
            # Explicit public batch and carried arbitrary streaming, without a timing sweep.
            short_references = {}
            for length in (3, 7):
                z = crops["English"]["audio"][..., :length].copy()
                reference = call("upstream_full_reference", f"English/short{length}", lambda: ready_upstream(upstream, z))
                short_references[length] = reference
                y = call("batch_gate", f"English/{length}", lambda: batch.decode(z))
                check(f"batch/{length}/actual-upstream", y, reference)
            z = crops["English"]["audio"][..., :16].copy()
            reference = call("upstream_full_reference", "English/carried3+13", lambda: ready_upstream(upstream, z))
            with fast.stream() as stream:
                first = call("arbitrary_stream_gate", "English/initial3", lambda: stream.decode_chunk(z[..., :3]))
                second = call("arbitrary_stream_gate", "English/carried13", lambda: stream.decode_chunk(z[..., 3:]))
                tail = call("arbitrary_stream_flush", "English/carried3+13", stream.flush)
                common.audio_check(tail, 0)
                require(stream.frames_decoded == 16, "Arbitrary chunk frame count")
            check("arbitrary-stream/initial3/actual-upstream", first, short_references[3])
            check("arbitrary-stream/carried3+13/actual-upstream", np.concatenate((first, second), -1), reference)
            require(set(fast._model.compiled) == {1, 2}, "Arbitrary chunks expanded compiled graph cache")
            for phase, repeats in (("qualification", 1), ("warmup", 1), ("measured", 2)):
                for repeat in range(repeats):
                    for ci, (label, values) in enumerate(crops.items()):
                        for ms in (40, 80):
                            order = list(ARMS if ms == 80 else ARMS[:2])
                            if (ci + repeat) % 2:
                                order.reverse()
                            for arm in order:
                                is_mimi = arm == ARMS[2]
                                z = values["mimi" if is_mimi else "audio"]
                                length = 1 if is_mimi else ms // 40
                                context = upstream.streaming_decode() if arm == ARMS[0] else None
                                stream = context.__enter__() if context else (common.BasicGPUStream(mimi, True) if is_mimi else fast.stream())
                                torch.mps.synchronize()
                                parts, times = [], []
                                try:
                                    for pos in range(0, z.shape[-1], length):
                                        packet = z[..., pos:pos+length]
                                        y = call("packet", f"{phase}/{repeat}/{label}/{ms}/{arm}/{pos}",
                                            lambda: ready_upstream(upstream, packet, stream) if context else stream.decode_chunk(packet))
                                        parts.append(y)
                                        times.append(result["calls"][-1]["seconds"])
                                    tail = call("upstream_flush_shim" if context else "flush", f"{phase}/{repeat}/{label}/{ms}/{arm}",
                                        lambda: np.empty((1, 1, 0), np.float32) if context else stream.flush())
                                    flush = result["calls"][-1]["seconds"]
                                    common.audio_check(tail, 0)
                                    if not context:
                                        require(stream.frames_decoded == z.shape[-1], "Wrong stream frame count")
                                finally:
                                    if context:
                                        context.__exit__(None, None, None)
                                    else:
                                        stream.close()
                                joined = np.concatenate(parts, -1)
                                reference = refs[label]["mimi" if is_mimi else "upstream"]
                                comparison = check(f"{phase}/{repeat}/{label}/{ms}/{arm}", joined, reference)
                                result["rows"].append(dict(phase=phase, repetition=repeat, clip=label, arm=arm,
                                    order=order, packet_ms=ms, packets=len(times), frames=z.shape[-1],
                                    samples=joined.shape[-1], packet_seconds=times, flush_seconds=flush,
                                    rtf=(sum(times)+flush)/.96, comparison=comparison))
                    save()
                    print(phase, repeat, "complete", flush=True)
            # One longer natural input, once per packet size/arm; quality only.
            continuation_start = len(result["calls"])
            long_reference = call("continuation_full_reference", "English",
                                  lambda: ready_upstream(upstream, continuation))
            for ms in (40, 80):
                length = ms // 40
                for arm in ARMS[:2]:
                    context = upstream.streaming_decode() if arm == ARMS[0] else None
                    stream = context.__enter__() if context else fast.stream()
                    parts = []
                    try:
                        for pos in range(0, continuation.shape[-1], length):
                            packet = continuation[..., pos:pos+length]
                            parts.append(call("continuation_packet", f"{ms}/{arm}/{pos}",
                                lambda: ready_upstream(upstream, packet, stream) if context else stream.decode_chunk(packet)))
                        tail = call("continuation_flush_shim" if context else "continuation_flush", f"{ms}/{arm}",
                                    lambda: np.empty((1, 1, 0), np.float32) if context else stream.flush())
                        common.audio_check(tail, 0)
                        if not context:
                            require(stream.frames_decoded == continuation.shape[-1], "Continuation frame count")
                    finally:
                        if context:
                            context.__exit__(None, None, None)
                        else:
                            stream.close()
                    joined = np.concatenate(parts, -1)
                    compared = check(f"continuation/{ms}/{arm}/upstream-full", joined, long_reference)
                    result["continuation"]["rows"].append(dict(arm=arm, packet_ms=ms, packets=len(parts),
                        samples=joined.shape[-1], comparison=compared))
            count = continuation.shape[-1]
            continuation_calls = 1 + 2*count + 2*((count+1)//2) + 4
            require(len(result["calls"])-continuation_start == continuation_calls, "Continuation call accounting")
            result["continuation"]["calls"] = continuation_calls
            result["compile_counters_final"] = compiler_counters()
            require(result["compile_counters_final"] == result["compile_counters_after_load"], "Recompilation during comparison")
        for arm in ARMS:
            for ms in ((80,) if arm == ARMS[2] else (40, 80)):
                rows = [r for r in result["rows"] if r["phase"] == "measured" and r["arm"] == arm and r["packet_ms"] == ms]
                times = [t for r in rows for t in r["packet_seconds"]]
                require(len(rows) == 6, "Missing measured streams")
                result["summary"].append(dict(arm=arm, packet_ms=ms, streams=len(rows),
                    rtf=sum(sum(r["packet_seconds"])+r["flush_seconds"] for r in rows)/(.96*len(rows)),
                    mean_packet_ms=1000*statistics.mean(times), median_packet_ms=1000*statistics.median(times),
                    p95_packet_ms=1000*float(np.percentile(times, 95)), stream_rtfs=[r["rtf"] for r in rows]))
        result["matched_audio_reductions"] = []
        for ms in (40, 80):
            pairs = []
            for repeat in range(2):
                for label, _ in common.CLIPS:
                    rows = {r["arm"]: r for r in result["rows"] if r["phase"] == "measured"
                            and r["packet_ms"] == ms and r["repetition"] == repeat and r["clip"] == label}
                    pairs.append(1-rows[ARMS[1]]["rtf"]/rows[ARMS[0]]["rtf"])
            result["matched_audio_reductions"].append(dict(packet_ms=ms, reductions=pairs,
                median_reduction=statistics.median(pairs), faster_pairs=sum(x > 0 for x in pairs)))
        require(len(result["rows"]) == 60 and len(result["calls"]) == 1085+continuation_calls
                and sum(r["packets"] for r in result["rows"]) == 1008, "Fixed accounting incomplete")
        require(all((p.stat().st_size, p.stat().st_mtime_ns) == stamps[str(p)] for p in files)
                and all(sha(p) == files[p] for p in source_files), "Input/source changed during run")
        result.update(status="passed", artifacts_unchanged=True, counted_ordinary_calls=len(result["calls"]),
                      public_load_internal_calls=2, total_calls_including_public_load=len(result["calls"])+2)
        print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if mimi is not None:
            mimi.close()
        result["wall_seconds"] = time.perf_counter()-started
        save()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
