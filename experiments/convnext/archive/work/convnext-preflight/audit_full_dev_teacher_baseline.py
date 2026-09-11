"""Read-only CPU calibration of exact-teacher loss on the fixed full dev set."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics
import time

import torch
from audiovae_student.cache import UtteranceCache
from audiovae_student.corpus_training import _fixed_dev_crop
from audiovae_student.data import load_manifest
from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss, _magnitude


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expected-dev", type=int, default=2839)
    parser.add_argument("--scored-frames", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = args.base.resolve()
    out = base / "audit-full-dev-teacher-baseline"
    out.mkdir(exist_ok=True)
    if (out / "report.json").exists():
        raise ValueError("Final audit already exists; preserve it")
    manifest_path = base / "expanded-pilot/corpus/source-manifest.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    rows = [row for row in load_manifest(manifest_path) if row.split == "dev"]
    config = WarmupLossConfig()
    criterion = WarmupReconstructionLoss(config)
    with sqlite3.connect("file:" + str(base / "target-cache-500h/index.sqlite3") + "?mode=ro", uri=True) as db:
        entries = {}
        for lookup, encoded in db.execute("SELECT lookup_key,info FROM entries"):
            info = json.loads(encoded)
            row_id = info.get("row_id")
            if row_id is not None:
                if row_id in entries:
                    raise ValueError("Duplicate source row in cache index")
                entries[row_id] = (lookup, info)
    cache_identity = json.loads((base / "target-cache-500h/identity.json").read_text())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running", "device": "cpu", "threads": 1,
        "training_updates": 0, "model_forward_calls": 0, "teacher_forward_calls": 0,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "teacher_identity": cache_identity["teacher"],
        "loss_config": asdict(config), "scored_frames": args.scored_frames,
        "expected_dev_rows": args.expected_dev, "manifest_dev_rows": len(rows),
        "selection": "Every dev manifest row, in manifest order, using the same leading fixed crop and valid sample count as source_training.evaluate",
        "aggregation": "Arithmetic mean of one scalar loss per dev utterance, identical to source_training.evaluate; not duration weighting or group-macro weighting",
        "exact_teacher_policy": "Student prediction equals cached original-teacher waveform. Teacher spectral and waveform losses are analytically zero; only original-reference branch requires FFTs.",
        "missing": [], "results": [],
        "implementation_sha256": {
            name: hashlib.sha256((base / "audiovae_student" / name).read_bytes()).hexdigest()
            for name in ("cache.py", "corpus_training.py", "losses.py", "source_training.py")
        },
    }
    atomic_json(out / "progress.json", {"state": "running", "completed": 0, "expected": len(rows)})
    started = time.monotonic()
    try:
        if len(rows) != args.expected_dev:
            raise ValueError("Actual manifest dev count differs from requested full-set count")
        with torch.no_grad():
            for index, row in enumerate(rows):
                if row.source_id not in entries:
                    report["missing"].append({"source_id": row.source_id, "reason": "no_index_entry"})
                    continue
                lookup, info = entries[row.source_id]
                path = base / "target-cache-500h/targets" / (lookup + ".pt")
                if not path.exists():
                    report["missing"].append({"source_id": row.source_id, "reason": "cache_file_absent"})
                    continue
                raw = path.read_bytes()
                if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["file_sha256"]:
                    raise ValueError("Cached file differs from its existing verified index")
                payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                if payload["cache_key"] != info["cache_key"]:
                    raise ValueError("Cache key differs from existing index")
                record = UtteranceCache(payload["latents"], payload["teacher_audio"], payload["reference16k"], payload["metadata"], payload["cache_key"])
                record.validate()
                expected_source = row.to_dict()
                expected_source["teacher_cache_key"] = None
                identity = record.metadata["identity"]
                if identity["source"] != expected_source or identity["teacher"] != cache_identity["teacher"]:
                    raise ValueError("Cache source or original-teacher identity differs from fixed manifest")
                if record.metadata.get("source_preparation") != {
                    "reader": cache_identity["reader"], "lookup_key": lookup, "source_file_sha256": row.audio_sha256,
                }:
                    raise ValueError("Cache source preparation identity differs from fixed reader")
                crop = _fixed_dev_crop(record, args.scored_frames)
                target = crop.teacher_audio[..., crop.scored_slice]
                reference = crop.reference16k[..., crop.reference_scored_slice]
                terms = []
                for size in config.reference_fft_sizes_16k:
                    prediction = _magnitude(target.float(), size * 3)[:, :size // 2 + 1]
                    truth = _magnitude(reference.float(), size)
                    terms.append((prediction.clamp_min(config.log_epsilon).log()
                                  - truth.clamp_min(config.log_epsilon).log()).abs().mean())
                spectral = torch.stack(terms).mean()
                total = config.reference_spectral_weight * spectral
                if index == 0:
                    direct = criterion(target, target, reference)
                    if direct["teacher_spectral"].item() != 0 or direct["teacher_waveform"].item() != 0:
                        raise ValueError("Exact teacher baseline has nonzero teacher loss")
                    torch.testing.assert_close(total, direct["total"], rtol=0, atol=0)
                    torch.testing.assert_close(spectral, direct["reference_spectral"], rtol=0, atol=0)
                    report["first_clip_full_criterion_check"] = "bitwise equal scalar results"
                report["results"].append({
                    "source_id": row.source_id, "dataset": row.dataset, "language": row.language,
                    "cache_key": record.cache_key, "valid_scored_samples_48k": crop.valid_scored_samples,
                    "total": total.item(), "teacher_spectral": 0.0, "teacher_waveform": 0.0,
                    "reference_spectral": spectral.item(),
                })
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    progress = {"state": "running", "completed": len(report["results"]), "expected": len(rows),
                                "missing": len(report["missing"]), "elapsed_seconds": time.monotonic() - started}
                    atomic_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                del raw, payload, record, crop, target, reference
        def summarize(values):
            if not values:
                return {"examples": 0}
            return {"examples": len(values), "scored_samples": sum(v["valid_scored_samples_48k"] for v in values),
                    **{name: statistics.mean(v[name] for v in values)
                       for name in ("total", "teacher_spectral", "teacher_waveform", "reference_spectral")}}
        report["summary"] = summarize(report["results"])
        for field in ("dataset", "language"):
            groups = defaultdict(list)
            for value in report["results"]:
                groups[value[field]].append(value)
            report["by_" + field] = {key: summarize(values) for key, values in sorted(groups.items())}
        report["state"] = "complete" if not report["missing"] and len(report["results"]) == len(rows) else "incomplete"
    except BaseException as error:
        report["state"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_json(out / "report.json", report)
    atomic_json(out / "progress.json", {key: report[key] for key in ("state", "elapsed_seconds", "finished_utc")})
    print(json.dumps({key: report.get(key) for key in ("state", "summary", "error", "elapsed_seconds")}), flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
