#!/usr/bin/env python3
"""Paired one-thread CPU streaming candidate test against one accepted reference.

python paired_benchmark.py --config run.json --output results.json
Paths are absolute or relative to the config. See config-template.json.
"""
import benchmark_core as core  # Sets CPU/thread environment before NumPy.
import argparse
import hashlib
import inspect
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

CLIPS = ["bn_in_00151_1818", "en_us_00103_1779", "es_419_00060_1994"]
MODES = ["full", "stream_80", "stream_160"]


def require_one_worker(attributes):
    for record in attributes:
        for name, value in record["attributes"].items():
            if type(value) is not int or value != 1:
                raise ValueError(f"Expected one-worker graph: {record['node']}/{name}={value}")


def waveform_check(value, reference, exact):
    core.validate_waveform(value, reference.shape[-1])
    core.validate_waveform(reference, reference.shape[-1])
    delta = value.astype(np.float64) - reference.astype(np.float64)
    equal = bool(np.array_equal(value.view(np.uint32), reference.view(np.uint32)))
    return {"passed": equal if exact else bool(np.allclose(value, reference, atol=1e-5, rtol=1e-4)),
            "exact_required": exact, "bitwise_equal": equal, "atol": 0 if exact else 1e-5,
            "rtol": 0 if exact else 1e-4, "samples": value.shape[-1],
            "max_abs": float(np.abs(delta).max(initial=0)),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "waveform_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def decode_schedule(stream, z, hop, sizes):
    """Untimed state validation with explicit count checks at every boundary."""
    outputs, calls, start, index = [], [], 0, 0
    while start < z.shape[-1]:
        end = min(start + sizes[index % len(sizes)], z.shape[-1])
        value = stream.decode_chunk(np.ascontiguousarray(z[:, :, start:end])).copy()
        core.validate_waveform(value, (end - start) * hop)
        outputs.append(value)
        calls.append({"kind": "chunk", "start_frame": start, "end_frame": end,
                      "start_sample": start * hop, "end_sample": end * hop,
                      "returned_samples": value.shape[-1]})
        start, index = end, index + 1
    value = stream.flush().copy()
    core.validate_waveform(value, 0)
    outputs.append(value)
    calls.append({"kind": "flush", "start_frame": start, "end_frame": start,
                  "start_sample": start * hop, "end_sample": start * hop,
                  "returned_samples": value.shape[-1]})
    return np.concatenate(outputs, axis=-1), calls


def state_checks(adapter, accepted, z, *, exact, uid, record_sink=None):
    """Bounded 13-frame probes; all references come from accepted.full."""
    z = np.ascontiguousarray(z[:, :, :13])
    if z.shape[-1] < 9:
        raise ValueError("State probes require at least nine frozen frames")
    hop = adapter.metadata["hop_samples"]
    reference = accepted.full(z).copy()
    alternative = z.copy()
    alternative[:, :, 5:] *= np.float32(-1)
    alternative_reference = accepted.full(alternative).copy()
    records = []

    def add(name, value, target, calls=None):
        row = {"name": name, "uid": uid, "check": waveform_check(value, target, exact)}
        if calls is not None:
            row["calls"] = calls
        records.append(row)
        if record_sink is not None:
            record_sink(row)
        if not row["check"]["passed"]:
            raise RuntimeError(f"State/parity failure: {name}/{uid}: {row['check']}")

    # Confirm full-call causality on this probe and the candidate's full path.
    add("accepted_future_prefix", alternative_reference[:, :, :5 * hop], reference[:, :, :5 * hop])
    add("full_probe", adapter.full(z).copy(), reference)
    add("full_alternative", adapter.full(alternative).copy(), alternative_reference)
    for name, sizes in (("one_frame", [1]), ("two_frames", [2]),
                        ("four_frames", [4]), ("uneven_frames", [1, 4, 2, 3])):
        stream = adapter.stream()
        try:
            result, calls = decode_schedule(stream, z, hop, sizes)
            add(name, result, reference, calls)
        finally:
            stream.close()
    stream = adapter.stream()
    try:
        # Reset deliberately abandons a partially consumed sequence.
        abandoned = stream.decode_chunk(np.ascontiguousarray(alternative[:, :, :3])).copy()
        core.validate_waveform(abandoned, 3 * hop)
        stream.reset()
        result, calls = decode_schedule(stream, z, hop, [4, 1, 2])
        add("reset_after_partial", result, reference, calls)
        stream.reset()
        result, calls = decode_schedule(stream, alternative, hop, [2])
        add("reset_after_flush", result, alternative_reference, calls)
    finally:
        stream.close()
    streams = [adapter.stream(), adapter.stream()]
    values, callsets = [[], []], [[], []]
    try:
        for start in range(0, z.shape[-1], 2):
            end = min(start + 2, z.shape[-1])
            for index, source in enumerate((z, alternative)):
                value = streams[index].decode_chunk(np.ascontiguousarray(source[:, :, start:end])).copy()
                core.validate_waveform(value, (end - start) * hop)
                values[index].append(value)
                callsets[index].append({"kind": "chunk", "start_frame": start, "end_frame": end,
                    "start_sample": start * hop, "end_sample": end * hop, "returned_samples": value.shape[-1]})
        for index, target in enumerate((reference, alternative_reference)):
            value = streams[index].flush().copy()
            core.validate_waveform(value, 0)
            callsets[index].append({"kind": "flush", "start_frame": z.shape[-1], "end_frame": z.shape[-1],
                "start_sample": z.shape[-1] * hop, "end_sample": z.shape[-1] * hop, "returned_samples": 0})
            add(f"interleaved_{index}", np.concatenate(values[index] + [value], axis=-1), target, callsets[index])
        first, second = (np.concatenate(v, axis=-1) for v in values)
        add("stream_future_prefix", second[:, :, :5 * hop], first[:, :, :5 * hop])
    finally:
        for stream in streams:
            stream.close()
    return records


def paired_schedule(uids, warmups=2, repeats=5, seed=20260908):
    rng = random.Random(seed)
    jobs = []
    for uid in uids:
        for phase, count in (("warmup", warmups), ("measure", repeats)):
            for repeat in range(count):
                modes = list(MODES)
                rng.shuffle(modes)
                for mode in modes:
                    names = ["baseline", "candidate"]
                    rng.shuffle(names)
                    pair_id = f"{uid}/{phase}/{repeat}/{mode}"
                    jobs.extend({"uid": uid, "phase": phase, "repeat": repeat, "mode": mode,
                                 "adapter": name, "pair_id": pair_id, "pair_position": pos}
                                for pos, name in enumerate(names))
    return jobs


def summarize_pairs(rows):
    grouped, pairs = {}, {}
    for row in rows:
        if row["phase"] == "measure" and row["check"]["passed"]:
            pairs.setdefault(row["pair_id"], {})[row["adapter"]] = row
    for pair in pairs.values():
        if set(pair) != {"baseline", "candidate"}:
            continue
        baseline, candidate = pair["baseline"], pair["candidate"]
        grouped.setdefault(baseline["mode"], []).append({
            "uid": baseline["uid"], "pair_id": baseline["pair_id"],
            "baseline_rtf": baseline["sum_call_rtf"], "candidate_rtf": candidate["sum_call_rtf"],
            "reduction_percent": 100 * (1 - candidate["sum_call_seconds"] / baseline["sum_call_seconds"])})
    summaries = []
    for mode, values in sorted(grouped.items()):
        uids = sorted({value["uid"] for value in values})
        summary = {"mode": mode, "complete_pairs": len(values), "clips": len(uids), "pairs": values}
        for name in ("baseline_rtf", "candidate_rtf"):
            summary["mean_clip_median_" + name] = statistics.mean(
                statistics.median(v[name] for v in values if v["uid"] == uid) for uid in uids)
        summary["aggregate_reduction_percent"] = 100 * (1 - summary["mean_clip_median_candidate_rtf"] / summary["mean_clip_median_baseline_rtf"])
        summary["median_paired_reduction_percent"] = statistics.median(v["reduction_percent"] for v in values)
        summaries.append(summary)
    return summaries


def run(config, output, *, factory=None, artifact_reader=None):
    if config.get("warmups", 2) != 2 or config.get("repeats", 5) != 5:
        raise ValueError("This bounded protocol requires two warmups and five measured repetitions")
    if type(config.get("exact")) is not bool:
        raise ValueError("Specify exact=true for Intel and exact=false for Apple")
    uids = config.get("clip_ids", CLIPS)
    if not isinstance(uids, list) or not uids or len(set(uids)) != len(uids):
        raise ValueError("clip_ids must be explicit unique IDs")
    affinity = core.checked_affinity()
    if platform.system() == "Linux" and affinity != [0]:
        raise ValueError(f"This matched campaign requires Linux CPU0, got {affinity}")
    import onnxruntime as ort
    if ort.__version__ != "1.29.0":
        raise RuntimeError(f"Required ONNX Runtime 1.29.0, got {ort.__version__}")
    factory = factory or core.AudioVAE2Adapter
    artifact_reader = artifact_reader or core.model_artifacts
    report = {"version": 1, "status": "running", "config": config, "onnxruntime": ort.__version__,
              "run_started_epoch_seconds": time.time(),
              "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version, "affinity": affinity},
              "protocol": {"threads": 1, "cpu_only": True, "warmups": 2, "repeats": 5,
                "modes": MODES, "reference": "One accepted baseline full waveform for BOTH adapters",
                "ordering": "Adjacent randomized AB/BA pairs, shuffled modes in each repetition",
                "headline_rtf": "Sum decode/full and flush wall times / actual generated duration",
                "whole_loop": "Calls, output copies and Python bookkeeping; excludes stream construction/close and checks",
                "state_checks_enabled": not config.get("skip_state_checks", False),
                "state_checks": "Skipped by explicit timing-screen configuration" if config.get("skip_state_checks", False) else
                    "Untimed first 13 frames: 1/2/4, uneven, partial/complete reset, interleaved streams and perturbed future",
                "environment": {key: value for key, value in core.os.environ.items() if key in (
                    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "BLIS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}},
              "adapters": {}, "references": [], "state_checks": [], "runs": []}
    core.write_report(output, report)
    loaded, all_paths, original_hashes = {}, [], {}
    try:
        for name in ("baseline", "candidate"):
            started = time.perf_counter()
            adapter = factory(config[name + "_bundle"], optimized=True, threads=1)
            loaded[name] = adapter
            adapter.metadata["exact_streaming"] = config["exact"]
            core.validate_adapter(adapter)
            files = list(adapter.metadata["artifacts"]) + [__file__, core.__file__, config["latents"], config["latent_manifest"]]
            files.extend(config.get("extra_artifacts", []))
            factory_source = inspect.getsourcefile(factory)
            if factory_source:
                files.append(factory_source)
            for module_name, module in list(sys.modules.items()):
                if module_name == "fast_audiovae" or module_name.startswith("fast_audiovae."):
                    source = getattr(module, "__file__", None)
                    if source:
                        files.append(source)
            hashes, attributes = artifact_reader(files)
            require_one_worker(attributes)
            for path, value in hashes.items():
                if path in original_hashes and original_hashes[path] != value:
                    raise RuntimeError(f"Artifact changed during adapter setup: {path}")
                original_hashes[path] = value
            all_paths.extend(hashes)
            report["adapters"][name] = {"metadata": adapter.metadata, "load_and_provenance_seconds": time.perf_counter() - started,
                "artifacts_before": hashes, "graph_concurrency_attributes": attributes}
            core.write_report(output, report)
        fields = ("codec", "sample_rate", "hop_samples", "latent_fps", "channels")
        if any(loaded["baseline"].metadata[k] != loaded["candidate"].metadata[k] for k in fields):
            raise ValueError("Baseline and candidate interfaces differ")
        manifest = json.loads(Path(config["latent_manifest"]).read_text())
        cases = {case["uid"]: case for case in manifest["cases"]}
        frozen, references = {}, {}
        with np.load(config["latents"], allow_pickle=False) as archive:
            for uid in uids:
                z = np.array(archive[uid + "__z"], copy=True, order="C")
                if list(z.shape) != cases[uid]["latent_shape"] or z.dtype != np.float32 or not np.isfinite(z).all():
                    raise ValueError(f"Invalid frozen latent: {uid}")
                frozen[uid] = z
                reference = loaded["baseline"].full(z).copy()
                core.validate_waveform(reference, z.shape[-1] * loaded["baseline"].metadata["hop_samples"])
                references[uid] = reference
                report["references"].append({"uid": uid, "adapter": "baseline", "samples": reference.shape[-1],
                    "waveform_sha256": hashlib.sha256(reference.tobytes()).hexdigest(), "timing_excluded": True,
                    "source_duration_seconds": cases[uid]["source_duration_s"]})
                for name, adapter in loaded.items():
                    if not config.get("skip_state_checks", False):
                        def record_state(row):
                            row["adapter"] = name
                            report["state_checks"].append(row)
                            core.write_report(output, report)
                        state_checks(adapter, loaded["baseline"], z, exact=config["exact"], uid=uid,
                                     record_sink=record_state)
        schedule = paired_schedule(uids, seed=config.get("seed", 20260908))
        for job in schedule:
            uid, name = job["uid"], job["adapter"]
            row = core.run_once(loaded[name], frozen[uid], references[uid], job["mode"], uid=uid,
                               repeat=job["repeat"], phase=job["phase"])
            row.update(job, order=len(report["runs"]), source_duration_seconds=cases[uid]["source_duration_s"],
                       reference_adapter="baseline")
            report["runs"].append(row)
            report["summary"] = core.summarize(report["runs"])
            report["paired_summary"] = summarize_pairs(report["runs"])
            core.write_report(output, report)
            if not row["check"]["passed"]:
                raise RuntimeError(f"Accepted-reference parity/count failure: {name}/{uid}/{job['mode']}: {row.get('error')}")
        if len(report["runs"]) != len(schedule):
            raise RuntimeError("Final run count differs from the complete paired schedule")
        paired_summary = report["paired_summary"]
        if {row["mode"] for row in paired_summary} != set(MODES) or any(
                row["complete_pairs"] != len(uids) * 5 for row in paired_summary):
            raise RuntimeError("Measured pairs are incomplete")
        if not config.get("skip_state_checks", False) and len(report["state_checks"]) != len(uids) * 24:
            raise RuntimeError("State probes are incomplete")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Run even after a numerical failure, retaining all completed raw rows.
        try:
            after, attributes = artifact_reader(sorted(set(all_paths)))
            require_one_worker(attributes)
            report["artifacts_after"] = after
            report["artifacts_unchanged"] = after == original_hashes
            if not report["artifacts_unchanged"]:
                report["status"] = "failed"
                report["artifact_error"] = "Source, model, data or library artifact changed during the campaign"
        except BaseException as error:
            report["status"] = "failed"
            report["artifact_error"] = f"{type(error).__name__}: {error}"
        for adapter in loaded.values():
            if hasattr(adapter, "close"):
                adapter.close()
        report["run_finished_epoch_seconds"] = time.time()
        core.write_report(output, report)
    if report["status"] != "passed":
        raise RuntimeError(report.get("artifact_error", report.get("error", "Candidate campaign failed")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    for key in ("baseline_bundle", "candidate_bundle", "latents", "latent_manifest"):
        config[key] = str((root / config[key]).resolve())
    config["extra_artifacts"] = [str((root / item).resolve()) for item in config.get("extra_artifacts", [])]
    result = run(config, args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "paired_summary": result["paired_summary"]}, indent=2))


if __name__ == "__main__":
    main()
