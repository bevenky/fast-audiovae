#!/usr/bin/env python3
"""Matched, single-thread CPU decoder timing; no encoder or playback timing.

Config: {"clip_ids": ["frozen_uid"], "adapters": [
  {"id": "base", "factory": "audiovae2", "kwargs": {
    "model_dir": "/absolute/bundle", "optimized": false},
   "latents": "/absolute/frozen.npz", "latent_manifest": "/absolute/frozen.json"},
  {"id": "mimi", "factory": "mimi_adapter:MimiOnnxAdapter", "kwargs": {
    "model_dir": "/absolute/mimi"}, "latents": "...", "latent_manifest": "..."}
]}

Run: python benchmark.py --config run.json --output result.json
Every adapter exposes metadata, sessions, full(z), and stream(). A stream
exposes decode_chunk(z), reset(), flush(), and close(). Factories receive
threads=1 explicitly. Config paths are relative to the config file.
"""

# Set these before importing any numerical package, including a plugin.
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
             "GOTO_NUM_THREADS", "OMP_THREAD_LIMIT", "TF_NUM_INTRAOP_THREADS",
             "TF_NUM_INTEROP_THREADS"):
    os.environ[_key] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
                  HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1",
                  OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")

import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_report(path, report):
    """Atomic incremental snapshot, preserving completed raw rows on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_artifacts(paths):
    """Hash selected graphs/libraries and ONNX external tensors, outside timers."""
    import onnx
    records, graph_attributes = {}, []
    queue = [Path(path).resolve() for path in paths]
    while queue:
        path = queue.pop()
        if str(path) in records:
            continue
        if not path.is_file():
            raise ValueError(f"Missing adapter artifact: {path}")
        records[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if path.suffix != ".onnx":
            continue
        model = onnx.load(str(path), load_external_data=False)
        graphs = [model.graph]
        while graphs:
            graph = graphs.pop()
            tensors = list(graph.initializer)
            for node in graph.node:
                selected = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        tensors.append(attr.t)
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        graphs.append(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        graphs.extend(attr.graphs)
                    if any(word in attr.name.lower() for word in ("thread", "segment", "shard")):
                        value = onnx.helper.get_attribute_value(attr)
                        if isinstance(value, bytes):
                            value = value.decode()
                        if not isinstance(value, (str, int, float, list)):
                            raise ValueError(f"Unrecognized concurrency attribute: {node.name}/{attr.name}")
                        selected[attr.name] = value
                        if "thread" in attr.name.lower() and isinstance(value, int) and value > 1:
                            raise ValueError(f"Graph hardcodes multiple threads: {node.name}/{attr.name}={value}")
                if selected:
                    graph_attributes.append({"model": str(path), "node": node.name,
                                             "operator": node.op_type, "attributes": selected})
            for tensor in tensors:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        queue.append((path.parent / entry.value).resolve())
    return records, graph_attributes


class AudioVAE2Adapter:
    def __init__(self, model_dir, *, optimized, threads=1):
        if threads != 1:
            raise ValueError("This protocol requires one thread")
        from fast_audiovae import load_decoder, load_streaming_decoder
        root = Path(model_dir).resolve()
        # Each session registers only its own library set. Do not merge the
        # stateless and streaming fused custom-op libraries into one session.
        self.full_session, full_info = load_decoder(root, threads=1, prefer_custom=optimized)
        self.decoder, stream_info = load_streaming_decoder(root, threads=1, prefer_custom=optimized)
        self.sessions = [self.full_session, self.decoder._session]
        expected = "native" if optimized else "portable_onnx"
        if full_info["selected"] != expected or stream_info["selected"] != expected:
            raise RuntimeError(f"Expected {expected}, got {full_info['selected']}/{stream_info['selected']}")
        if full_info["model"] != stream_info["full_call_model"]:
            raise RuntimeError("Full and streaming sessions do not share the same reference graph")
        manifest = json.loads((root / "bundle.json").read_text())
        artifacts = [root / "bundle.json", root / full_info["model"], root / stream_info["model"]]
        if optimized:
            native = manifest["native"][full_info["platform"]]
            artifacts.append(root / native["library"])
            entry = manifest["streaming"]["models"][full_info["model"]]
            for record in native.get("additional_libraries", []) + entry.get("additional_libraries", []):
                artifacts.append(root / record["library"])
        exact = bool(optimized and full_info.get("native_streaming_math_version") == 1)
        self.metadata = {"name": "AudioVAE2 optimized" if optimized else "AudioVAE2 base",
                         "codec": "audiovae2", "sample_rate": 48000, "hop_samples": 1920,
                         "latent_fps": 25, "channels": 64, "exact_streaming": exact,
                         "providers": ["CPUExecutionProvider"], "artifacts": list(map(str, artifacts)),
                         "full_runtime": full_info, "stream_runtime": stream_info,
                         "baseline": "Explicit supplied portable bundle; inspect artifact provenance to distinguish stock export from graph rewrites" if not optimized else None}
        inputs = self.full_session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError("Full decoder must have exactly one latent input")
        self._input = inputs[0].name
        self._output = self.full_session.get_outputs()[0].name

    def full(self, z):
        return self.full_session.run([self._output], {self._input: z})[0]

    def stream(self):
        return self.decoder.streaming_decode()


def validate_adapter(adapter):
    import onnxruntime as ort
    if ort.__version__ != "1.29.0":
        raise RuntimeError(f"Required ONNX Runtime 1.29.0, got {ort.__version__}")
    meta = adapter.metadata
    for key in ("name", "codec", "sample_rate", "hop_samples", "latent_fps", "channels", "artifacts"):
        if key not in meta:
            raise ValueError(f"Adapter metadata missing {key}")
    for key in ("sample_rate", "hop_samples", "channels"):
        if type(meta[key]) is not int or meta[key] <= 0:
            raise ValueError(f"Invalid adapter {key}")
    if not math.isclose(meta["sample_rate"] / meta["hop_samples"], meta["latent_fps"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Adapter sample rate, hop, and latent rate disagree")
    if not getattr(adapter, "sessions", None):
        raise ValueError("Adapter must expose all ORT sessions for provider/thread verification")
    for session in adapter.sessions:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Non-CPU execution provider in adapter")
        options = session.get_session_options()
        if options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1:
            raise RuntimeError("Adapter session is not explicitly one thread")
        if options.execution_mode != ort.ExecutionMode.ORT_SEQUENTIAL:
            raise RuntimeError("Adapter uses parallel graph execution")


def chunk_frames(milliseconds, metadata):
    frames = milliseconds * metadata["latent_fps"] / 1000
    if frames < 1 or not math.isclose(frames, round(frames), abs_tol=1e-12):
        raise ValueError(f"{milliseconds} ms is not a whole latent frame for {metadata['name']}")
    return round(frames)


def validate_waveform(value, samples):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError("Decoder output must be a float32 NumPy array")
    if value.shape != (1, 1, samples) or not np.isfinite(value).all():
        raise ValueError(f"Invalid waveform shape/finiteness: {value.shape}, expected (1, 1, {samples})")


def checked_affinity():
    """Linux measurements must be pinned to one logical CPU before launch."""
    if platform.system() == "Linux":
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("Linux timing requires sched_getaffinity to verify single-CPU pinning")
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) != 1:
            raise RuntimeError(f"Linux timing must be pinned to exactly one logical CPU, got {affinity}")
        return affinity
    # macOS does not expose this affinity API. The external sampler records
    # process CPU usage while actual ORT sessions still enforce one thread.
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def run_once(adapter, z, reference, mode, *, uid, repeat, phase, clock=time.perf_counter):
    """Time all calls and copies; compare the exact timed waveform afterwards."""
    row_started_epoch = time.time()
    meta = adapter.metadata
    if z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (1, meta["channels"]) or z.shape[2] < 1:
        raise ValueError("Invalid latent tensor")
    frames, hop = z.shape[2], meta["hop_samples"]
    samples = frames * hop
    validate_waveform(reference, samples)
    chunk = frames if mode == "full" else chunk_frames(int(mode.removeprefix("stream_")), meta)
    # Contiguous copies of input chunks are setup, excluded from decode latency.
    pieces = [(start, min(start + chunk, frames), np.ascontiguousarray(z[:, :, start:start + chunk]))
              for start in range(0, frames, chunk)]
    outputs, raw = [], []
    stream, creation_seconds, close_seconds = None, 0.0, 0.0
    if mode != "full":
        start = clock()
        stream = adapter.stream()
        creation_seconds = clock() - start
    # One process-time pair surrounds the entire loop, never each chunk.
    process_started = time.process_time()
    loop_start = clock()
    failure = None
    try:
        for index, (start_frame, end_frame, values) in enumerate(pieces):
            started = clock()
            output = adapter.full(values) if mode == "full" else stream.decode_chunk(values)
            ended = clock()
            copy_start = clock()
            # Protect a timed result from adapters that reuse output buffers.
            copied = output.copy() if isinstance(output, np.ndarray) else output
            copy_end = clock()
            outputs.append(copied)
            raw.append({"kind": "full" if mode == "full" else "chunk", "chunk_index": index,
                        "start_frame": start_frame, "end_frame": end_frame,
                        "start_sample": start_frame * hop, "end_sample": end_frame * hop,
                        "first": index == 0, "final": index == len(pieces) - 1,
                        "returned_samples": int(output.shape[-1]) if isinstance(output, np.ndarray) and output.ndim else None,
                        "start_seconds": started - loop_start, "end_seconds": ended - loop_start,
                        "duration_seconds": ended - started, "output_copy_seconds": copy_end - copy_start})
        if stream is not None:
            started = clock()
            output = stream.flush()
            ended = clock()
            copy_start = clock()
            copied = output.copy() if isinstance(output, np.ndarray) else output
            copy_end = clock()
            outputs.append(copied)
            raw.append({"kind": "flush", "chunk_index": len(pieces), "start_frame": frames,
                        "end_frame": frames, "start_sample": samples, "end_sample": samples,
                        "first": False, "final": True,
                        "returned_samples": int(output.shape[-1]) if isinstance(output, np.ndarray) and output.ndim else None,
                        "start_seconds": started - loop_start, "end_seconds": ended - loop_start,
                        "duration_seconds": ended - started, "output_copy_seconds": copy_end - copy_start})
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    loop_seconds = clock() - loop_start
    process_cpu_seconds = time.process_time() - process_started
    if stream is not None:
        start = clock()
        stream.close()
        close_seconds = clock() - start
    check_started = clock()
    check = {"passed": False, "exact_required": bool(meta.get("exact_streaming", False)),
             "atol": 0.0 if meta.get("exact_streaming") else 1e-5,
             "rtol": 0.0 if meta.get("exact_streaming") else 1e-4}
    if failure is None:
        try:
            for record, output in zip(raw, outputs):
                validate_waveform(output, record["end_sample"] - record["start_sample"])
            combined = np.concatenate(outputs, axis=-1)
            validate_waveform(combined, samples)
            difference = combined.astype(np.float64) - reference.astype(np.float64)
            # Equality of floating values treats +0 and -0 alike. Keep the
            # stronger bitwise claim tied to the actual float32 payloads.
            exact = bool(np.array_equal(combined.view(np.uint32), reference.view(np.uint32)))
            check.update(bitwise_equal=exact, max_abs=float(np.abs(difference).max(initial=0)),
                         rmse=float(np.sqrt(np.mean(difference ** 2))), returned_samples=combined.shape[-1],
                         waveform_sha256=hashlib.sha256(combined.tobytes()).hexdigest())
            check["passed"] = exact if check["exact_required"] else bool(np.allclose(combined, reference, atol=1e-5, rtol=1e-4))
            if not check["passed"]:
                failure = "Timed waveform differs from the same-model full reference"
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    check_seconds = clock() - check_started
    call_seconds = sum(record["duration_seconds"] for record in raw)
    duration = samples / meta["sample_rate"]
    row = {"uid": uid, "mode": mode, "repeat": repeat, "phase": phase,
           "row_started_epoch_seconds": row_started_epoch,
           "row_finished_epoch_seconds": time.time(),
           "latency_state": "sessions and weights already loaded; fresh stream state; OS disk cache uncontrolled",
           "first_call_after_warmups": phase == "measure",
           "latent_frames": frames, "chunk_frames": chunk, "samples": samples,
           "decoded_duration_seconds": duration, "stream_creation_seconds": creation_seconds,
           "stream_close_seconds": close_seconds, "sum_call_seconds": call_seconds,
           "whole_loop_seconds": loop_seconds, "sum_call_rtf": call_seconds / duration,
           "whole_loop_rtf": loop_seconds / duration,
           "process_cpu_seconds": process_cpu_seconds,
           "process_cpu_to_loop_wall_ratio": process_cpu_seconds / loop_seconds if loop_seconds > 0 else None,
           "output_copy_seconds": sum(record["output_copy_seconds"] for record in raw),
           "check_seconds": check_seconds, "first_call_seconds": raw[0]["duration_seconds"] if raw else None,
           "calls": raw, "check": check}
    if failure:
        row["error"] = failure
    return row


def summarize(rows):
    groups = {}
    for row in rows:
        if row["phase"] == "measure" and row["check"]["passed"]:
            groups.setdefault((row["adapter"], row["mode"]), []).append(row)
    summaries = []
    for (adapter, mode), group in sorted(groups.items()):
        # First median across repeats for each clip, then equal-clip mean.
        uids = sorted({row["uid"] for row in group})
        entry = {"adapter": adapter, "mode": mode, "clips": len(uids), "runs": len(group)}
        for metric in ("sum_call_rtf", "whole_loop_rtf", "first_call_seconds"):
            entry["mean_clip_median_" + metric] = statistics.mean(
                statistics.median(row[metric] for row in group if row["uid"] == uid) for uid in uids)
        entry["duration_weighted_sum_call_rtf"] = sum(r["sum_call_seconds"] for r in group) / sum(r["decoded_duration_seconds"] for r in group)
        summaries.append(entry)
    return summaries


def run(config, output, *, factory_loader=None):
    run_started_epoch = time.time()
    affinity = checked_affinity()
    warmups, repeats = config.get("warmups", 2), config.get("repeats", 3)
    if type(warmups) is not int or warmups < 2 or type(repeats) is not int or repeats < 3:
        raise ValueError("Protocol requires at least two warmups and three repeats")
    uids = config.get("clip_ids")
    if not isinstance(uids, list) or not uids or any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError("Provide explicit, unique frozen clip_ids")
    adapters = config.get("adapters", [])
    if not adapters or len({item["id"] for item in adapters}) != len(adapters):
        raise ValueError("Provide adapters with unique IDs")
    if any(item.get("kwargs", {}).get("threads", 1) != 1 for item in adapters):
        raise ValueError("Only one thread is permitted")
    import onnxruntime as ort
    if ort.__version__ != "1.29.0":
        raise RuntimeError(f"Required ONNX Runtime 1.29.0, got {ort.__version__}")
    report = {"version": 1, "status": "running", "config": config, "onnxruntime": ort.__version__,
              "run_started_epoch_seconds": run_started_epoch,
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "python": sys.version, "affinity": affinity},
              "protocol": {"threads": 1, "warmups": warmups, "repeats": repeats,
                           "cpu_only": True, "seed": config.get("seed", 20260908),
                           "cold_load": "Initial adapter construction only; no disk cache eviction; excluded from RTF",
                           "whole_loop": "All calls, flush, output copies and Python bookkeeping; excludes stream creation, close and checks",
                           "process_cpu": "Process CPU time across the whole loop; one pair of readings outside the loop, excludes creation/close/checks",
                           "duration": "Actual decoder samples including latent rounding; source duration reported separately",
                           "comparison": "Per-codec full reference; not an audio quality comparison between different codecs",
                           "environment": {key: value for key, value in os.environ.items() if key in (
                               "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}},
              "adapters": {}, "runs": [], "references": []}
    write_report(output, report)
    loaded, inputs = {}, {}
    rng = random.Random(report["protocol"]["seed"])
    try:
        for spec in adapters:
            if factory_loader is None:
                factory = AudioVAE2Adapter if spec["factory"] == "audiovae2" else getattr(importlib.import_module(spec["factory"].split(":")[0]), spec["factory"].split(":")[1])
            else:
                factory = factory_loader(spec["factory"])
            kwargs = dict(spec.get("kwargs", {})); kwargs["threads"] = 1
            start = time.perf_counter()
            adapter = factory(**kwargs)
            load_seconds = time.perf_counter() - start
            loaded[spec["id"]] = adapter
            validate_adapter(adapter)
            artifacts = list(adapter.metadata["artifacts"])
            source = inspect.getsourcefile(factory)
            if source:
                artifacts.append(source)
            artifacts.extend([__file__, spec["latents"], spec["latent_manifest"]])
            hashes, graph_attributes = model_artifacts(artifacts)
            manifest = json.loads(Path(spec["latent_manifest"]).read_text())
            cases = {case["uid"]: case for case in manifest["cases"]}
            with np.load(spec["latents"], allow_pickle=False) as archive:
                for uid in uids:
                    if uid not in cases or uid + "__z" not in archive:
                        raise ValueError(f"Frozen clip missing for {spec['id']}: {uid}")
                    z = np.array(archive[uid + "__z"], copy=True, order="C")
                    if list(z.shape) != cases[uid]["latent_shape"]:
                        raise ValueError(f"Latent shape differs from frozen manifest: {uid}")
                    if z.dtype != np.float32 or not np.isfinite(z).all():
                        raise ValueError(f"Invalid frozen latent: {uid}")
                    inputs[spec["id"], uid] = z
            report["adapters"][spec["id"]] = {"metadata": adapter.metadata, "initial_load_seconds": load_seconds,
                "artifacts": hashes, "graph_concurrency_attributes": graph_attributes,
                "clips": {uid: cases[uid] for uid in uids}}
            write_report(output, report)
        for uid in uids:
            sources = [report["adapters"][name]["clips"][uid]["source_duration_s"] for name in loaded]
            if max(sources) - min(sources) > 1e-9:
                raise ValueError(f"Codec manifests disagree on source duration: {uid}")
            references, jobs = {}, []
            for name, adapter in loaded.items():
                z = inputs[name, uid]
                reference = adapter.full(z).copy()
                validate_waveform(reference, z.shape[2] * adapter.metadata["hop_samples"])
                references[name] = reference
                report["references"].append({"adapter": name, "uid": uid, "samples": reference.shape[-1],
                    "waveform_sha256": hashlib.sha256(reference.tobytes()).hexdigest(), "timing_excluded": True})
                modes = ["full", "stream_80", "stream_160"]
                if config.get("include_40ms", False) and adapter.metadata["codec"] == "audiovae2":
                    modes.append("stream_40")
                jobs.extend((name, mode) for mode in modes)
            for phase, count in (("warmup", warmups), ("measure", repeats)):
                for repeat in range(count):
                    order = list(jobs)
                    rng.shuffle(order)
                    for name, mode in order:
                        row = run_once(loaded[name], inputs[name, uid], references[name], mode,
                                       uid=uid, repeat=repeat, phase=phase)
                        row.update(adapter=name, order=len(report["runs"]), source_duration_seconds=sources[0])
                        report["runs"].append(row)
                        report["summary"] = summarize(report["runs"])
                        write_report(output, report)
                        if not row["check"]["passed"]:
                            raise RuntimeError(f"Parity/count check failed: {name}/{uid}/{mode}: {row.get('error')}")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["run_finished_epoch_seconds"] = time.time()
        write_report(output, report)
        raise
    finally:
        for adapter in loaded.values():
            if hasattr(adapter, "close"):
                adapter.close()
    report["run_finished_epoch_seconds"] = time.time()
    write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    for spec in config["adapters"]:
        for key in ("latents", "latent_manifest"):
            spec[key] = str((root / spec[key]).resolve())
        if "model_dir" in spec.get("kwargs", {}):
            spec["kwargs"]["model_dir"] = str((root / spec["kwargs"]["model_dir"]).resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    result = run(config, args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
