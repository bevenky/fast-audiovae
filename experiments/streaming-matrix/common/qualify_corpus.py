#!/usr/bin/env python3
"""Untimed 60-clip streaming qualification against the accepted full decoder.

python qualify_corpus.py --config run.json --output qualification.json
Uses the paired campaign config but always validates ALL 60 manifest clips.
Only one accepted full session and one candidate streaming session are loaded.
"""
import benchmark_core as core  # Freezes CPU/thread environment before NumPy.
import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


class QualificationAdapter:
    """The public loading API, with only the session required by each role."""
    def __init__(self, model_dir, *, role, threads=1):
        if threads != 1 or role not in ("accepted_full", "candidate_stream"):
            raise ValueError("Expected one-thread full or streaming qualification role")
        from fast_audiovae import load_decoder, load_streaming_decoder
        self.role, root = role, Path(model_dir).resolve()
        manifest = json.loads((root / "bundle.json").read_text())
        if role == "accepted_full":
            self.full_session, info = load_decoder(root, threads=1, prefer_custom=True)
            session, full_model = self.full_session, info["model"]
            self._input = session.get_inputs()[0].name
            self._output = session.get_outputs()[0].name
        else:
            self.decoder, info = load_streaming_decoder(root, threads=1, prefer_custom=True)
            session, full_model = self.decoder._session, info["full_call_model"]
        if info["selected"] != "native":
            raise RuntimeError(f"Qualification requires selected native bundle, got {info['selected']}")
        native = manifest["native"][info["platform"]]
        stream_entry = manifest["streaming"]["models"][full_model]
        paths = [root / "bundle.json", root / full_model, root / stream_entry["model"], root / native["library"]]
        for item in native.get("additional_libraries", []) + stream_entry.get("additional_libraries", []):
            paths.append(root / item["library"])
        self.sessions = [session]
        self.metadata = {"name": role, "codec": "audiovae2", "sample_rate": 48000,
            "hop_samples": 1920, "latent_fps": 25, "channels": 64,
            "providers": ["CPUExecutionProvider"], "role": role, "runtime": info,
            "full_graph": str(root / full_model), "artifacts": list(map(str, paths))}

    def full(self, z):
        if self.role != "accepted_full":
            raise RuntimeError("The candidate full session is deliberately not loaded")
        return self.full_session.run([self._output], {self._input: z})[0]

    def stream(self):
        if self.role != "candidate_stream":
            raise RuntimeError("The accepted streaming session is deliberately not loaded")
        return self.decoder.streaming_decode()


def compare(value, reference, *, exact, gate=True):
    core.validate_waveform(value, reference.shape[-1])
    core.validate_waveform(reference, reference.shape[-1])
    difference = value.astype(np.float64) - reference.astype(np.float64)
    equal = bool(np.array_equal(value.view(np.uint32), reference.view(np.uint32)))
    close = bool(np.allclose(value, reference, atol=1e-5, rtol=1e-4))
    return {"passed": equal if exact else close, "gate": gate, "exact_required": exact,
        "bitwise_equal": equal, "atol": 0.0 if exact else 1e-5, "rtol": 0.0 if exact else 1e-4,
        "returned_samples": value.shape[-1], "max_abs": float(np.abs(difference).max(initial=0)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "waveform_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def check_stream(adapter, z, reference, stored, *, uid, frames, exact):
    """Capture every attempted chunk and count failure, with no speed claims."""
    hop = adapter.metadata["hop_samples"]
    row = {"uid": uid, "chunk_frames": frames, "latent_frames": z.shape[-1],
           "expected_samples": z.shape[-1] * hop, "calls": [], "checks": {}, "passed": False}
    outputs, stream = [], None
    try:
        stream = adapter.stream()
        for index, start in enumerate(range(0, z.shape[-1], frames)):
            end = min(start + frames, z.shape[-1])
            record = {"kind": "chunk", "chunk_index": index, "start_frame": start, "end_frame": end,
                "start_sample": start * hop, "end_sample": end * hop,
                "first": start == 0, "final": end == z.shape[-1], "returned_samples": None}
            row["calls"].append(record)
            value = stream.decode_chunk(np.ascontiguousarray(z[:, :, start:end]))
            record["returned_samples"] = int(value.shape[-1]) if isinstance(value, np.ndarray) and value.ndim else None
            core.validate_waveform(value, (end - start) * hop)
            outputs.append(value.copy())
        record = {"kind": "flush", "chunk_index": len(row["calls"]), "start_frame": z.shape[-1],
            "end_frame": z.shape[-1], "start_sample": row["expected_samples"], "end_sample": row["expected_samples"],
            "first": False, "final": True, "returned_samples": None}
        row["calls"].append(record)
        value = stream.flush()
        record["returned_samples"] = int(value.shape[-1]) if isinstance(value, np.ndarray) and value.ndim else None
        core.validate_waveform(value, 0)
        outputs.append(value.copy())
        combined = np.concatenate(outputs, axis=-1)
        row["checks"]["accepted_full"] = compare(combined, reference, exact=exact)
        if stored is not None:
            row["checks"]["stored_upstream"] = compare(combined, stored, exact=False, gate=not exact)
        row["passed"] = all(check["passed"] for check in row["checks"].values() if check["gate"])
        if not row["passed"]:
            row["error"] = "Candidate differs from required accepted/full or stored-upstream reference"
    except BaseException as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if row["calls"] and row["calls"][-1]["returned_samples"] is None:
            row["calls"][-1]["error"] = row["error"]
        # Preserve interruption semantics after persisting the enclosing result.
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            row["interrupted"] = True
    finally:
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                row["passed"] = False
                row["close_error"] = f"{type(error).__name__}: {error}"
    return row


def check_workers(attributes):
    for record in attributes:
        for name, value in record["attributes"].items():
            if type(value) is not int or value != 1:
                raise ValueError(f"Expected one-worker graph: {record['node']}/{name}={value}")


def validate_cohort(manifest, archive_keys):
    cases = manifest.get("cases", [])
    uids = [case.get("uid") for case in cases]
    if len(cases) != 60 or len(set(uids)) != 60 or any(not isinstance(uid, str) or not uid for uid in uids):
        raise ValueError("Qualification requires exactly 60 unique frozen manifest clips")
    frozen = {key[:-3] for key in archive_keys if key.endswith("__z")}
    if frozen != set(uids):
        raise ValueError("Frozen latent archive cohort differs from the full 60-clip manifest")
    return cases


def run(config, output, *, factory=None, artifact_reader=None):
    factory = factory or QualificationAdapter
    artifact_reader = artifact_reader or core.model_artifacts
    if type(config.get("exact")) is not bool:
        raise ValueError("Specify exact=true for Intel or exact=false for Apple")
    affinity = core.checked_affinity()
    if platform.system() == "Linux" and affinity != [0]:
        raise RuntimeError(f"Linux qualification must be pinned to CPU0, got {affinity}")
    report = {"version": 1, "status": "running", "config": config,
        "run_started_epoch_seconds": time.time(),
        "host": {"platform": platform.platform(), "machine": platform.machine(), "affinity": affinity},
        "protocol": {"cpu_only": True, "threads": 1, "onnxruntime": "1.29.0", "required_clips": 60,
            "candidate_stream_patterns": [1, 2, 4], "timing_claims": False,
            "reference": "Accepted baseline full output shared by all candidate chunk patterns",
            "upstream_policy": "Reported only; accepted integer approximation unchanged" if config["exact"] else
                "Every available stored upstream waveform is also a strict tolerance gate",
            "scope": "All complete frozen clips, including first output and last partial chunk; state isolation covered by paired campaign probes"},
        "adapters": {}, "references": [], "runs": []}
    core.write_report(output, report)
    loaded, before, paths = {}, {}, []
    try:
        for name, role in (("baseline", "accepted_full"), ("candidate", "candidate_stream")):
            adapter = factory(config[name + "_bundle"], role=role, threads=1)
            loaded[name] = adapter
            core.validate_adapter(adapter)
            files = list(adapter.metadata["artifacts"]) + [__file__, core.__file__, config["latents"], config["latent_manifest"]]
            files.extend(config.get("extra_artifacts", []))
            source = inspect.getsourcefile(factory)
            if source:
                files.append(source)
            for module_name, module in list(sys.modules.items()):
                if module_name == "fast_audiovae" or module_name.startswith("fast_audiovae."):
                    source = getattr(module, "__file__", None)
                    if source:
                        files.append(source)
            hashes, attributes = artifact_reader(files)
            check_workers(attributes)
            for path, value in hashes.items():
                if path in before and before[path] != value:
                    raise RuntimeError(f"Artifact changed during setup: {path}")
                before[path] = value
            paths.extend(hashes)
            report["adapters"][name] = {"metadata": adapter.metadata, "artifacts_before": hashes,
                "graph_concurrency_attributes": attributes}
            core.write_report(output, report)
        fields = ("codec", "sample_rate", "hop_samples", "latent_fps", "channels")
        if any(loaded["baseline"].metadata[key] != loaded["candidate"].metadata[key] for key in fields):
            raise ValueError("Candidate changed the codec interface")
        manifest = json.loads(Path(config["latent_manifest"]).read_text())
        with np.load(config["latents"], allow_pickle=False) as archive:
            cases = validate_cohort(manifest, archive.files)
            report["cohort"] = {"clips": len(cases), "uids": [case["uid"] for case in cases],
                "stored_upstream_references": sum(case["uid"] + "__ref" in archive for case in cases)}
            for case in cases:
                uid = case["uid"]
                report["active_clip"] = uid
                core.write_report(output, report)
                z = np.array(archive[uid + "__z"], copy=True, order="C")
                if list(z.shape) != case["latent_shape"] or z.dtype != np.float32 or not np.isfinite(z).all():
                    raise ValueError(f"Invalid frozen latent: {uid}")
                if z.ndim != 3 or z.shape[:2] != (1, loaded["baseline"].metadata["channels"]) or z.shape[-1] < 1:
                    raise ValueError(f"Invalid latent geometry: {uid}")
                samples = z.shape[-1] * loaded["baseline"].metadata["hop_samples"]
                if case.get("reference_samples", samples) != samples:
                    raise ValueError(f"Manifest output sample count differs: {uid}")
                reference = loaded["baseline"].full(z).copy()
                core.validate_waveform(reference, samples)
                stored = np.array(archive[uid + "__ref"], copy=True) if uid + "__ref" in archive else None
                record = {"uid": uid, "samples": samples, "latent_frames": z.shape[-1],
                    "waveform_sha256": hashlib.sha256(reference.tobytes()).hexdigest(), "adapter": "baseline"}
                if stored is not None:
                    record["stored_upstream"] = compare(reference, stored, exact=False, gate=not config["exact"])
                report["references"].append(record)
                core.write_report(output, report)
                if stored is not None and not config["exact"] and not record["stored_upstream"]["passed"]:
                    raise RuntimeError(f"Accepted baseline already fails stored-upstream gate: {uid}")
                for frames in (1, 2, 4):
                    row = check_stream(loaded["candidate"], z, reference, stored, uid=uid, frames=frames, exact=config["exact"])
                    report["runs"].append(row)
                    report["completed_streams"] = len(report["runs"])
                    core.write_report(output, report)
                    if not row["passed"]:
                        raise RuntimeError(f"Corpus qualification failure: {uid}/{frames} frames: {row.get('error', row.get('close_error'))}")
            if len(report["references"]) != 60 or len(report["runs"]) != 180:
                raise RuntimeError("Qualification did not complete all 60 references and 180 streams")
            for uid in report["cohort"]["uids"]:
                if {row["chunk_frames"] for row in report["runs"] if row["uid"] == uid} != {1, 2, 4}:
                    raise RuntimeError(f"Incomplete candidate chunk patterns: {uid}")
        report["status"] = "passed"
        report.pop("active_clip", None)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            after, attributes = artifact_reader(sorted(set(paths)))
            check_workers(attributes)
            report["artifacts_after"] = after
            report["artifacts_unchanged"] = after == before
            if not report["artifacts_unchanged"]:
                report["status"] = "failed"
                report["artifact_error"] = "Source, model, library or input artifact changed during qualification"
        except BaseException as error:
            report["status"] = "failed"
            report["artifact_error"] = f"{type(error).__name__}: {error}"
        for adapter in loaded.values():
            if hasattr(adapter, "close"):
                adapter.close()
        report["run_finished_epoch_seconds"] = time.time()
        core.write_report(output, report)
    if report["status"] != "passed":
        raise RuntimeError(report.get("artifact_error", "Qualification failed"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.output}")
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    for key in ("baseline_bundle", "candidate_bundle", "latents", "latent_manifest"):
        config[key] = str((root / config[key]).resolve())
    config["extra_artifacts"] = [str((root / path).resolve()) for path in config.get("extra_artifacts", [])]
    result = run(config, args.output)
    print(json.dumps({"status": result["status"], "clips": len(result["references"]),
        "streams": len(result["runs"]), "artifacts_unchanged": result["artifacts_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
