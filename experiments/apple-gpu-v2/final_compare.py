"""Fixed short CPU/MPS comparison. Execute only under the parent's GPU lease."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(name, "0") != "0":
        raise RuntimeError(name + " must be disabled before importing Torch")
    os.environ[name] = "0"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs/apple-gpu-v1"))

import experiment as candidates
import benchmark_mps as common
import mimi_mps
import numpy as np
import onnxruntime as ort
import torch
from torch._dynamo.utils import counters
import fast_audiovae
from fast_audiovae.assets import MODEL_FILES, verify_model
from fast_audiovae.gpu import GPUDecoder

AUDIO = ("selected_cpu", "before", "before_compile", "full_compile")
COMPILED = ("before_compile", "full_compile")
SOURCE = ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae"
QUALIFICATION = HERE / "qualification-r1.json"
QUALIFICATION_SHA = "e83d10da5a841bf2d4c217eb3cadcdbadee8457241297c527607090382e8a272"
require, sha = common.require, common.sha


def counter_snapshot():
    return {str(group): {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()}


def main(output):
    output = output.expanduser().resolve()
    require(output.is_relative_to(HERE) and output.suffix == ".json",
            "Output must be a new JSON file under outputs/apple-gpu-v2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write('{"status":"starting"}\n')
    result = dict(version="apple_gpu_final_comparison_v2", status="running", rows=[],
                  calls=[], references=[], preparation=[], clips=[], models={},
                  ordinary_gpu_seconds=0.0, ordinary_decode_seconds=0.0,
                  compile_first_call_seconds=0.0)
    lock = (HERE / "gpu.lock").open("a")
    mimi = None
    started = time.perf_counter()

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def call(kind, label, fn, sink):
        require(result["ordinary_gpu_seconds"] < 20, "20-second ordinary GPU budget exhausted")
        require(result["ordinary_decode_seconds"] < 25, "25-second ordinary decode budget exhausted")
        record = dict(kind=kind, label=label, status="running")
        result["calls"].append(record)
        start = time.perf_counter()
        try:
            answer = fn()
            record["status"] = "returned"
            return answer
        finally:
            elapsed = time.perf_counter() - start
            record["seconds"] = elapsed
            sink.append(elapsed)
            if kind == "compile":
                result["compile_first_call_seconds"] += elapsed
            else:
                result["ordinary_decode_seconds"] += elapsed
                if kind == "gpu":
                    result["ordinary_gpu_seconds"] += elapsed
            require(result["ordinary_gpu_seconds"] <= 20, "20-second ordinary GPU budget exceeded")
            require(result["ordinary_decode_seconds"] <= 25, "25-second ordinary decode budget exceeded")

    def compare(label, value, reference):
        check = common.comparison(value, reference)
        require(check["passed"], "Original full CPU ONNX waveform gate failed: " + label)
        return check

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.__version__.split("+")[0] == "2.14.0" and ort.__version__ == "1.30.0",
                "Wrong runtime versions")
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1
                and torch.get_default_dtype() == torch.float32
                and torch.backends.mps.is_available(), "Wrong MPS/FP32/thread policy")
        require(Path(fast_audiovae.__file__).resolve().parent == SOURCE, "Wrong package source")
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        # Both fixed lengths and model variants are prepared before this is enabled.
        torch._dynamo.config.error_on_recompile = False

        config = json.loads(common.CONFIG.read_text())
        bundle = Path(config["bundle"])
        require(sha(bundle / "bundle.json") == config["bundle_manifest_sha256"]
                and config["stream_graph_sha256"] == common.SELECTED, "Wrong selected CPU baseline")
        manifest = json.loads((bundle / "bundle.json").read_text())
        native = manifest["native"]["Darwin/arm64"]
        spec = manifest["streaming"]["models"][native["model"]]
        require(spec["model_sha256"] == common.SELECTED and len(spec["states"]) == 26,
                "Wrong CPU stream graph/state inventory")
        expected = {bundle / "bundle.json": config["bundle_manifest_sha256"],
                    bundle / spec["model"]: common.SELECTED,
                    bundle / native["library"]: native["library_sha256"]}
        for item in spec["additional_libraries"] + spec["dependencies"]:
            expected[bundle / item["library"]] = item["sha256"]
        for key in ("latents", "latent_manifest", "mimi_latents", "mimi_latent_manifest"):
            expected[Path(config[key])] = config["inputs_sha256"][key]
        verify_model(common.ORIGINAL)
        for name, digest in MODEL_FILES.items():
            expected[common.ORIGINAL.parent / name] = digest
        mimi_reference_path = Path(config["mimi_bundle"]) / "mimi_full.onnx"
        expected[mimi_reference_path] = common.MIMI_FULL
        expected[mimi_mps.DEFAULT_CHECKPOINT] = mimi_mps.CHECKPOINT_SHA256
        for name, digest in mimi_mps.SOURCE_SHA256.items():
            expected[mimi_mps.DEFAULT_SOURCE / "pocket_tts" / name] = digest
        require(sha(QUALIFICATION) == QUALIFICATION_SHA, "Qualification receipt changed")
        qualification = json.loads(QUALIFICATION.read_text())
        require(qualification["status"] == "passed" and qualification["artifacts_unchanged"]
                and qualification["arms"] == list(COMPILED) and qualification["completed_arms"] == 2
                and len(qualification["checks"]) == 120 and all(c["passed"] for c in qualification["checks"])
                and all(len(qualification["state_summary"][arm]) == 26
                        and all(c["passed"] and c["comparisons"] == 39
                                for c in qualification["state_summary"][arm].values()) for arm in COMPILED),
                "Both compiled routes require the complete passed qualification")
        expected[QUALIFICATION] = QUALIFICATION_SHA
        for path in (HERE / "experiment.py", HERE / "mps_decoder_before.py", HERE / "qualify_candidates.py",
                     SOURCE / "gpu.py", SOURCE / "mps_decoder.py", common.CONFIG):
            expected[path] = qualification["files"][str(path)]
        for path, digest in expected.items():
            require(sha(path) == digest, "Artifact checksum mismatch: " + str(path))
        source_files = [Path(__file__), HERE / "experiment.py", HERE / "mps_decoder_before.py",
                        Path(common.__file__), Path(mimi_mps.__file__), common.CONFIG]
        source_files += [SOURCE / name for name in ("__init__.py", "automatic.py", "gpu.py",
                         "mps_decoder.py", "assets.py", "runtime.py", "streaming.py")]
        files = dict(expected)
        files.update({path: sha(path) for path in source_files})
        # Include all installed bundle payloads, not only ORT-registered libraries.
        files.update({path: sha(path) for path in bundle.rglob("*") if path.is_file() and path not in files})
        stamps = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in files}
        result["provenance"] = dict(files={str(path): digest for path, digest in files.items()},
            torch=torch.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
            platform=platform.platform(), machine=platform.machine(), host_threads=1,
            precision="FP32", fast_math=False, cpu_fallback=False,
            qualification=dict(path=str(QUALIFICATION), sha256=QUALIFICATION_SHA, status="passed"),
            postrun_artifact_check="All file sizes/mtimes unchanged; benchmark/package sources rehashed. Large model archives are not rehashed after timing.")
        result["protocol"] = dict(arms=list(AUDIO) + ["mimi_gpu"], qualification_sweeps=1,
            warmup_sweeps=1, measured_sweeps=2, packet_ms=[40, 80], mimi_40_ms="unsupported",
            duration_ms=960, clips=3, expected_streams=108, expected_stream_packets=1872,
            expected_stream_flushes=108, expected_reference_calls=8, expected_first_calls=6,
            order="For each clip and packet size, reverse complete arm order on clip+repetition parity. Each measured arm order occurs once in each direction.",
            reference="Each codec's original full CPU ONNX on exactly the cropped latents, with fresh state at the crop start.",
            sample_selection="Same shared-duration middle 960ms crop as benchmark_mps; aligned80ms, no context replay.",
            timing="Packet decode and flush calls, including owned input upload and completed CPU-ready output, API validation and synchronization. Audio first-packet lazy history allocation remains inside decode.",
            exclusions="Model/session/stream creation, stream-initialization synchronization, close, source hashing, reference comparison and reporting. First compiled public calls are separately reported including both compile and decode work.",
            boundary="All three Audio GPU arms use the same GPUDecoder API and finite validation of audio plus all26 histories. Mimi BasicGPUStream validates owned NumPy input/output and performs completed copy/sync, but omits all-history finite validation. CPU uses its selected streaming facade. These API validation boundaries are not identical across codecs.",
            tolerances=dict(atol=1e-5, rtol=1e-4), ordinary_gpu_cap_seconds=20,
            all_ordinary_decode_cap_seconds=25, error_on_recompile_after_preparation=True,
            aggregate="Pooled RTF uses packet+flush time divided by emitted duration; all repetitions retained.",
            short_evidence=True, performance_claim="Matched short comparison, no significance or overlap-only speed claim")
        save()

        crops = {}
        with np.load(config["latents"], allow_pickle=False) as a, np.load(config["mimi_latents"], allow_pickle=False) as m:
            for label, uid in common.CLIPS:
                az, mz = a[uid + "__z"], m[uid + "__z"]
                require(az.dtype == mz.dtype == np.float32 and az.shape[:2] == (1, 64)
                        and mz.shape[:2] == (1, 32), "Unexpected latent schema")
                available = min(az.shape[-1] // 2, mz.shape[-1])
                require(available >= 12, "Clip shorter than960ms")
                start = (available - 12) // 2
                crops[label] = dict(audio=np.ascontiguousarray(az[..., 2*start:2*start+24]),
                                    mimi=np.ascontiguousarray(mz[..., start:start+12]))
                require(all(np.isfinite(z).all() for z in crops[label].values()), "Nonfinite crop")
                result["clips"].append(dict(label=label, uid=uid, start_ms=80*start, duration_ms=960,
                    audio_frames=24, mimi_frames=12, audio_samples=46080, mimi_samples=23040,
                    crop_sha256={key: common.tensor_sha(z) for key, z in crops[label].items()}))
        sessions = {"audio": common.cpu_reference(common.ORIGINAL),
                    "mimi": common.cpu_reference(mimi_reference_path)}
        refs = {}
        for label, values in crops.items():
            refs[label] = {}
            for codec, z in values.items():
                row = dict(label=label, codec=codec, seconds=[])
                result["references"].append(row)
                session = sessions[codec]
                y = call("cpu", "reference/" + label + "/" + codec,
                         lambda: session.run(None, {session.get_inputs()[0].name: z})[0], row["seconds"])
                common.audio_check(y, z.shape[-1]*1920)
                refs[label][codec] = y
                row.update(samples=y.shape[-1], output_sha256=common.tensor_sha(y))
        prepared_refs = {}
        for length in (1, 2):
            z = crops["English"]["audio"][..., :length].copy()
            row = dict(label="English/prepared" + str(length), codec="audio", seconds=[])
            result["references"].append(row)
            y = call("cpu", row["label"], lambda: sessions["audio"].run(None, {"z": z})[0], row["seconds"])
            common.audio_check(y, length*1920)
            prepared_refs[length] = y
            row.update(samples=y.shape[-1], output_sha256=common.tensor_sha(y))

        cpu, cpu_info = fast_audiovae.load_streaming_decoder(bundle, threads=1, prefer_custom=True)
        require(cpu_info["selected"] == "native" and cpu_info["threads"] == 1
                and cpu_info["providers"] == ["CPUExecutionProvider"]
                and sha(bundle / cpu_info["model"]) == common.SELECTED, "CPU native selection failed")
        result["models"]["selected_cpu"] = cpu_info
        decoders = {}
        for arm in AUDIO[1:]:
            start = time.perf_counter()
            model = candidates.build(arm)
            decoder = GPUDecoder(model, torch, dict(experiment=arm, device="gpu", backend="mps",
                precision="FP32", fallback=False), "streaming")
            torch.mps.synchronize()
            entry = dict(arm=arm, model_creation_seconds=time.perf_counter()-start, first_calls=[])
            result["preparation"].append(entry)
            base = model.model if arm in COMPILED else model
            result["models"][arm] = dict(info=decoder.info, state_shapes=model.state_shapes,
                                         source_identity=base.source_identity)
            for length in (1, 2):
                row = dict(frames=length, seconds=[])
                entry["first_calls"].append(row)
                with decoder.stream() as stream:
                    torch.mps.synchronize()
                    z = crops["English"]["audio"][..., :length].copy()
                    y = call("compile" if arm in COMPILED else "gpu", arm + "/prepare" + str(length),
                             lambda: stream.decode_chunk(z), row["seconds"])
                    require(stream.frames_decoded == length, "Preparation frame count mismatch")
                    row["comparison"] = compare(arm + "/prepare" + str(length), y, prepared_refs[length])
                    row["output_sha256"] = common.tensor_sha(y)
                save()
                print("prepared", arm, length, round(row["seconds"][0], 4), flush=True)
            decoders[arm] = decoder
        mimi = common.MimiMPSAdapter(max_frames=12)
        torch.mps.synchronize()
        result["models"]["mimi_gpu"] = mimi.metadata
        result["compile_counters_after_preparation"] = counter_snapshot()
        torch._dynamo.config.error_on_recompile = True

        def run_stream(phase, repetition, label, arm, packet_ms, order):
            codec = "mimi" if arm == "mimi_gpu" else "audio"
            z = crops[label][codec]
            step = 1 if codec == "mimi" else packet_ms // 40
            if arm == "selected_cpu":
                stream = cpu.streaming_decode()
            elif arm == "mimi_gpu":
                stream = common.BasicGPUStream(mimi, True)
            else:
                stream = decoders[arm].stream()
            row = dict(phase=phase, repetition=repetition, clip=label, arm=arm, packet_ms=packet_ms,
                       order=list(order), packet_seconds=[], flush_seconds=[], status="running")
            result["rows"].append(row)
            kind = "cpu" if arm == "selected_cpu" else "gpu"
            try:
                if kind == "gpu":
                    torch.mps.synchronize()
                parts = []
                for position in range(0, z.shape[-1], step):
                    packet = z[..., position:position+step]
                    y = call(kind, phase + "/" + label + "/" + arm + "/" + str(position),
                             lambda: stream.decode_chunk(packet), row["packet_seconds"])
                    common.audio_check(y, packet.shape[-1]*1920)
                    parts.append(y)
                tail = call(kind, phase + "/" + label + "/" + arm + "/flush", stream.flush, row["flush_seconds"])
                common.audio_check(tail, 0)
                require(stream.frames_decoded == z.shape[-1], "Consumed frame count mismatch")
                joined = np.concatenate(parts + [tail], axis=-1)
                row["comparison"] = compare(arm + "/" + label, joined, refs[label][codec])
                row.update(status="passed", frames=z.shape[-1], samples=joined.shape[-1],
                           output_sha256=common.tensor_sha(joined),
                           seconds=sum(row["packet_seconds"]) + sum(row["flush_seconds"]))
                row["rtf"] = row["seconds"] / .96
            finally:
                stream.close()

        for phase, repetitions in (("qualification", 1), ("warmup", 1), ("measured", 2)):
            for repetition in range(repetitions):
                for index, (label, _) in enumerate(common.CLIPS):
                    for packet_ms in (40, 80):
                        order = list(AUDIO) + (["mimi_gpu"] if packet_ms == 80 else [])
                        if (index + repetition) % 2:
                            order.reverse()
                        for arm in order:
                            run_stream(phase, repetition, label, arm, packet_ms, order)
                require(counter_snapshot() == result["compile_counters_after_preparation"],
                        "Compiler counters changed during fixed execution sweeps")
                save()
                print(phase, repetition+1, "rows", len(result["rows"]), "ordinary GPU seconds",
                      round(result["ordinary_gpu_seconds"], 4), flush=True)

        require(len(result["rows"]) == 108 and len(result["calls"]) == 1994, "Incomplete fixed run accounting")
        require(sum(len(r["packet_seconds"]) for r in result["rows"]) == 1872
                and sum(len(r["flush_seconds"]) for r in result["rows"]) == 108, "Packet/flush accounting mismatch")
        require(all(r["status"] == "passed" for r in result["rows"]), "Incomplete stream")
        require(all([path.stat().st_size, path.stat().st_mtime_ns] == stamps[str(path)] for path in files),
                "An input artifact changed")
        require(all(sha(path) == files[path] for path in source_files), "Benchmark/package source changed")
        require(all(common.tensor_sha(crops[row["label"]][codec]) == digest for row in result["clips"]
                    for codec, digest in row["crop_sha256"].items()), "Input latents mutated")
        result["artifacts_unchanged"] = True
        result["models"]["mimi_gpu"]["numerical_qualification"] = (
            "Passed original CPU ONNX comparison for all four fixed960ms sweeps on three clips in this run")
        result["compile_counters_after_timing"] = counter_snapshot()
        require(result["compile_counters_after_timing"] == result["compile_counters_after_preparation"],
                "Compilation occurred during fixed sweeps")
        result["summary"] = []
        measured = [row for row in result["rows"] if row["phase"] == "measured"]
        for packet_ms in (40, 80):
            for arm in list(AUDIO) + (["mimi_gpu"] if packet_ms == 80 else []):
                rows = [row for row in measured if row["arm"] == arm and row["packet_ms"] == packet_ms]
                packets = [seconds for row in rows for seconds in row["packet_seconds"]]
                result["summary"].append(dict(arm=arm, packet_ms=packet_ms, streams=len(rows),
                    pooled_rtf=sum(row["seconds"] for row in rows)/(len(rows)*.96),
                    median_stream_rtf=statistics.median(row["rtf"] for row in rows),
                    stream_rtfs=[row["rtf"] for row in rows],
                    median_packet_ms=1000*statistics.median(packets),
                    p95_packet_ms=float(np.percentile(packets, 95)*1000),
                    median_first_packet_ms=1000*statistics.median(row["packet_seconds"][0] for row in rows)))
        result["paired_comparisons"] = []
        for packet_ms in (40, 80):
            for candidate, baseline in (("before_compile", "before"), ("full_compile", "before"),
                                         ("full_compile", "before_compile")):
                pairs = []
                for label, _ in common.CLIPS:
                    for repetition in range(2):
                        selected = {row["arm"]: row for row in measured if row["clip"] == label
                                    and row["repetition"] == repetition and row["packet_ms"] == packet_ms}
                        old, new = selected[baseline]["seconds"], selected[candidate]["seconds"]
                        pairs.append(dict(clip=label, repetition=repetition, baseline_seconds=old,
                                          candidate_seconds=new, reduction_percent=100*(1-new/old)))
                result["paired_comparisons"].append(dict(packet_ms=packet_ms, candidate=candidate,
                    baseline=baseline, pairs=pairs,
                    median_reduction_percent=statistics.median(pair["reduction_percent"] for pair in pairs)))
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if mimi is not None:
            mimi.close()
        result["wall_seconds"] = time.perf_counter() - started
        save()
        lock.close()
    print(json.dumps(dict(status=result["status"], ordinary_gpu_seconds=result["ordinary_gpu_seconds"],
                          ordinary_decode_seconds=result["ordinary_decode_seconds"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
