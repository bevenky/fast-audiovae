#!/usr/bin/env python3
"""Score supplied precision reconstructions with the unchanged audited CPU scorer.

Examples (no codec execution):
  score_quality.py --sources ../codec-quality-inputs/manifest.json \
    --variant fp32=/data/fp32 --variant bf16=/data/bf16 --baseline fp32 \
    --assets /data/metrics --output comparison.json --prepare-only
  score_quality.py --sources sources.json --variant-manifest variants.json \
    --baseline fp32 --assets /data/metrics --output comparison.json --resume

A variant manifest uses the existing canonical format: {"stage_ready": true,
"models": {...}, "samples": [{"uid": ..., "outputs": {name:
{"path": ..., "sample_rate": 48000, "samples": ..., "sha256": ...}}}]}.
Directory mode reads <uid>.wav. Sources may be relocated via --source-root;
their immutable identities and hashes must still match all 60 frozen clips.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# This is set before any numerical imports, including imports in child processes.
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}
os.environ.update(CPU_ENV)
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
SOURCE_IDENTITIES_SHA256 = "829b8070356f2dd37c65abee1d19b4445b2a3b4769055f623af7bac431e599bb"
SCORER_HASHES = {
    "score.py": "797bddcba0ffbd9f1035d3166e4db4eabc7ebea1f5fb0ab83a429199fc880f03",
    "metrics.py": "c5e37fb3a6376b90ee89e9c7c69fb1f9540080f12c6d750864e83814da9be491",
    "metric_assets.json": "628c81cd349eb45819143a275efb5417da2b518db2c71e858d0c1a6fe7c713c0",
}
METRICS = ("pesq_wb", "stoi", "estoi", "utmos", "dnsmos_sig", "dnsmos_bak",
           "dnsmos_ovrl", "dnsmos_p808", "mrstft_spectral_convergence",
           "mrstft_log_magnitude_mae", "si_sdr_db", "snr_db")
LOWER_BETTER = {"mrstft_spectral_convergence", "mrstft_log_magnitude_mae"}
IDENTITY_KEYS = ("uid", "language", "reference_rate", "reference_samples", "reference_sha256")


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve(path: str | Path, base: Path) -> Path:
    result = Path(path)
    return (result if result.is_absolute() else base / result).resolve()


def finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def load_sources(path: Path, source_root: Path | None = None) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text())
    samples = []
    for row in raw["samples"]:
        canonical = "reference_path" in row
        item = {
            "uid": row["uid"], "language": row["language"],
            "reference_path": row["reference_path"] if canonical else row["path"],
            "reference_rate": row["reference_rate"] if canonical else row.get("sample_rate", 16000),
            "reference_samples": row["reference_samples"] if canonical else row["samples"],
            "reference_sha256": row["reference_sha256"] if canonical else row["sha256"],
        }
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["uid"]):
            raise ValueError("Unsafe source uid")
        if source_root:
            possibilities = [source_root / item["language"] / (item["uid"] + ".wav"),
                             source_root / (item["uid"] + ".wav")]
            existing = [p for p in possibilities if p.is_file()]
            if len(existing) != 1:
                raise ValueError(f"Source root requires one unambiguous file for {item['uid']}")
            item["reference_path"] = str(existing[0].resolve())
        else:
            item["reference_path"] = str(resolve(item["reference_path"], path.parent))
        samples.append(item)
    identities = sorted(({key: row[key] for key in IDENTITY_KEYS} for row in samples), key=lambda r: r["uid"])
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(samples) != 60 or len({r["uid"] for r in samples}) != 60 or digest != SOURCE_IDENTITIES_SHA256:
        raise ValueError("Sources differ from the frozen 60-clip identity contract")
    return raw, samples


def load_variants(args, samples: list[dict]) -> tuple[dict, dict, dict]:
    uids = {r["uid"] for r in samples}
    sources = {r["uid"]: r for r in samples}
    outputs = {uid: {} for uid in uids}
    models, provenance = {}, {}
    if args.variant_manifest:
        path = args.variant_manifest.resolve()
        raw = json.loads(path.read_text())
        if not (raw.get("stage_ready") is True or raw.get("status") in ("ready", "complete")):
            raise ValueError("Variant manifest must declare stage_ready or ready/complete status")
        rows = raw["samples"]
        if len(rows) != len(uids) or {r["uid"] for r in rows} != uids:
            raise ValueError("Variant manifest must contain exactly the frozen 60 unique clips")
        for row in rows:
            for key in IDENTITY_KEYS:
                if key in row and row[key] != sources[row["uid"]][key]:
                    raise ValueError(f"Variant manifest source metadata mismatch: {row['uid']}/{key}")
            for name, item in row["outputs"].items():
                record = dict(item)
                record["path"] = str(resolve(record["path"], path.parent))
                outputs[row["uid"]][name] = record
        models.update(raw.get("models", {}))
        provenance["variant_manifest"] = {"path": str(path), "sha256": sha(path)}
    for spec in args.variant:
        name, separator, folder = spec.partition("=")
        if not separator or not folder or any(name in outputs[uid] for uid in uids):
            raise ValueError(f"Invalid or duplicate --variant {spec!r}")
        directory = Path(folder).resolve()
        for uid in uids:
            outputs[uid][name] = {"path": str(directory / (uid + ".wav"))}
        models[name] = {"input": "supplied reconstruction directory", "directory": str(directory)}
    names = set(next(iter(outputs.values())))
    if len(names) < 2 or any(set(v) != names for v in outputs.values()):
        raise ValueError("Every clip requires the same two or more variant names")
    if args.baseline not in names:
        raise ValueError("--baseline must name a supplied variant")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "original_reference":
            raise ValueError(f"Invalid/reserved variant name {name!r}")
    return outputs, {name: models.get(name, {}) for name in sorted(names)}, provenance


def read_wave(path: Path, *, rate=None, samples=None, digest=None):
    import numpy as np
    import soundfile as sf
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite, nonempty mono audio: {path}")
    if info.format not in ("WAV", "WAVEX", "RF64") or info.subtype != "FLOAT":
        raise ValueError(f"Use FLOAT WAV exports, without PCM quantization: {path}")
    if rate is not None and actual_rate != rate:
        raise ValueError(f"Unexpected sample rate: {path}")
    if samples is not None and len(audio) != samples:
        raise ValueError(f"Unexpected sample count: {path}")
    actual_digest = sha(path)
    if digest is not None and actual_digest != digest:
        raise ValueError(f"Waveform SHA256 mismatch: {path}")
    return audio, {"path": str(path.resolve()), "sample_rate": actual_rate,
                   "samples": len(audio), "sha256": actual_digest}


def native_difference(actual, reference, rate: int) -> dict:
    import numpy as np
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    energy = float(np.dot(reference.astype(np.float64), reference.astype(np.float64)))
    error = float(np.dot(difference, difference))
    status = "positive_infinity_exact_match" if error == 0 and energy > 0 else (
        "undefined_zero_reference_energy" if energy == 0 else "finite")
    return {"status": "same_native_rate_and_shape", "sample_rate": rate, "samples": len(actual),
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "mean_signed_error": float(np.mean(difference)),
            "bitwise_equal": actual.tobytes() == reference.tobytes(),
            "snr_db": 10 * math.log10(energy / error) if status == "finite" else None,
            "snr_status": status}


def prepare(args) -> tuple[Path, dict, list[dict]]:
    raw, samples = load_sources(args.sources.resolve(), args.source_root)
    outputs, models, variant_provenance = load_variants(args, samples)
    native_rows = []
    for sample in samples:
        uid = sample["uid"]
        read_wave(Path(sample["reference_path"]), rate=sample["reference_rate"],
                  samples=sample["reference_samples"], digest=sample["reference_sha256"])
        audio_by_model, records = {}, {}
        for name in sorted(models):
            record = outputs[uid][name]
            audio, verified = read_wave(Path(record["path"]), rate=record.get("sample_rate"),
                                        samples=record.get("samples"), digest=record.get("sha256"))
            expected_count = (sample["reference_samples"] * verified["sample_rate"] + 15999) // 16000
            if args.trim_audiovae2_right_padding:
                import soundfile as sf
                expected_full = ((sample["reference_samples"] + 639) // 640) * 1920
                if verified["sample_rate"] != 48000 or len(audio) != expected_full:
                    raise ValueError(f"{uid}/{name}: AudioVAE2 full-output contract requires 48 kHz and {expected_full} samples")
                raw_info = dict(verified)
                audio = audio[:expected_count].copy()
                folder = args.output.parent / (args.output.stem + "-waveforms") / name
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (uid + ".wav")
                if target.resolve() == Path(record["path"]).resolve():
                    raise ValueError("Scoring copy must not overwrite its source waveform")
                temporary = target.with_suffix(".partial.wav")
                sf.write(temporary, audio, 48000, subtype="FLOAT")
                temporary.replace(target)
                _, verified = read_wave(target, rate=48000, samples=expected_count)
                record.update({"raw_path": raw_info["path"], "raw_sha256": raw_info["sha256"],
                               "full_decoder_samples": raw_info["samples"],
                               "right_padding_removed_samples": expected_full - expected_count,
                               "trim_policy": "Known AudioVAE2 640-input/1920-output hop; left trim zero"})
            if len(audio) != expected_count:
                raise ValueError(f"{uid}/{name}: expected already trimmed {expected_count} samples, got {len(audio)}")
            audio_by_model[name] = audio
            records[name] = {**record, **verified}
        baseline = audio_by_model[args.baseline]
        baseline_rate = records[args.baseline]["sample_rate"]
        for name, audio in audio_by_model.items():
            if name == args.baseline:
                continue
            info = native_difference(audio, baseline, baseline_rate) if (
                audio.shape == baseline.shape and records[name]["sample_rate"] == baseline_rate
            ) else {"status": "not_compared_different_native_rate_or_shape"}
            native_rows.append({"uid": uid, "language": sample["language"], "model": name,
                                "baseline": args.baseline, **info})
        sample["outputs"] = records
    manifest = {
        "stage_ready": True, "dataset": raw.get("dataset", {}), "models": models, "samples": samples,
        "protocol": {"purpose": "Precision comparison on exactly paired frozen source clips",
                     "baseline": args.baseline, "normalization": "None", "alignment": "Start aligned; pretrimmed known right padding only",
                     "metric_definition": "Unchanged pinned quality-comparison/score.py and metrics.py",
                     "cpu_only": True, "threads": args.threads, "native_differences": "No resampling, gain fit or alignment",
                     "codec_inference": False, "codec_timing": False},
        "preparation": {"sources_manifest": str(args.sources.resolve()), "sources_manifest_sha256": sha(args.sources),
                        "source_identities_sha256": SOURCE_IDENTITIES_SHA256, "wrapper_sha256": sha(Path(__file__)),
                        **variant_provenance},
    }
    path = args.output.with_name(args.output.stem + "-manifest.json").resolve()
    atomic_json(path, manifest)
    return path, manifest, native_rows


def comparisons(raw: dict, baseline: str, native_rows: list[dict]) -> dict:
    rows = raw["rows"]
    index = {(row["uid"], row["model"]): row for row in rows}
    names = sorted({r["model"] for r in rows} - {baseline, "original_reference"})
    uids = [r["uid"] for r in raw["manifest"]["samples"]]
    per_clip, aggregate, languages = [], {}, {}
    for name in names:
        for uid in uids:
            source = index.get((uid, baseline))
            candidate = index.get((uid, name))
            if source is None or candidate is None:
                raise ValueError(f"Missing completed score row {uid}/{name}")
            delta = {}
            for key in METRICS:
                a, b = candidate["metrics"].get(key), source["metrics"].get(key)
                delta[key] = a - b if finite(a) and finite(b) else None
            per_clip.append({"uid": uid, "language": candidate["language"], "model": name,
                             "baseline": baseline, "candidate_minus_baseline": delta,
                             "candidate_metric_errors": candidate["metrics"].get("errors", []),
                             "baseline_metric_errors": source["metrics"].get("errors", [])})
        first, second = sorted((name, baseline))
        pair = raw["summary"]["paired_differences"].get(first + " minus " + second, {})
        sign = 1 if first == name else -1
        aggregate[name] = {}
        for key in METRICS:
            original = pair.get(key)
            if original is None:
                aggregate[name][key] = {"pairs": 0, "expected_pairs": len(uids), "status": "no_finite_pairs"}
                continue
            record = dict(original)
            record["mean_candidate_minus_baseline"] = sign * record.pop("mean_first_minus_second")
            ci = record["paired_bootstrap_95_percentile_ci"]
            record["paired_bootstrap_95_percentile_ci"] = ci if sign == 1 else [-ci[1], -ci[0]]
            if sign == -1:
                record["positive_pairs"], record["negative_pairs"] = record["negative_pairs"], record["positive_pairs"]
            record.update({"expected_pairs": len(uids), "status": "complete" if record["pairs"] == len(uids) else "some_pairs_not_finite",
                           "improvement_direction": "negative" if key in LOWER_BETTER else "positive"})
            aggregate[name][key] = record
        languages[name] = {}
        for language in sorted({r["language"] for r in rows}):
            selected = [r for r in per_clip if r["model"] == name and r["language"] == language]
            fields = {}
            for key in METRICS:
                values = [r["candidate_minus_baseline"][key] for r in selected if finite(r["candidate_minus_baseline"][key])]
                fields[key] = {"mean_candidate_minus_baseline": sum(values)/len(values) if values else None,
                               "pairs": len(values), "expected_pairs": len(selected)}
            languages[name][language] = fields
    native_summary = {}
    for name in names:
        selected = [r for r in native_rows if r["model"] == name and r["status"] == "same_native_rate_and_shape"]
        native_summary[name] = {"compared_clips": len(selected), "expected_clips": len(uids),
                                "max_abs_across_clips": max((r["max_abs"] for r in selected), default=None),
                                "equal_clip_mean_rmse": sum(r["rmse"] for r in selected)/len(selected) if selected else None,
                                "bitwise_equal_clips": sum(r["bitwise_equal"] for r in selected)}
    return {"per_clip": per_clip, "aggregate_candidate_minus_baseline": aggregate,
            "per_language_candidate_minus_baseline": languages,
            "native_waveform_differences": {"summary": native_summary, "per_clip": native_rows,
                                            "meaning": "Numerical differences from supplied baseline at unchanged native rate; no quality threshold inferred"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--variant", action="append", default=[], metavar="NAME=DIRECTORY")
    p.add_argument("--variant-manifest", type=Path)
    p.add_argument("--baseline", required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scorer-dir", type=Path, default=BASE.parent / "quality-comparison")
    p.add_argument("--threads", type=int, choices=(1, 2), default=2)
    p.add_argument("--trim-audiovae2-right-padding", action="store_true",
                   help="Require full 1920*ceil(source16k_samples/640) output and write separate duration-trimmed FLOAT WAVs")
    p.add_argument("--prepare-only", action="store_true", help="Validate/hash WAVs and assets; create no predictor session or scores")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    for name, expected in SCORER_HASHES.items():
        if sha(args.scorer_dir / name) != expected:
            raise ValueError(f"Audited scorer changed: {name}")
    assets = json.loads((args.scorer_dir / "metric_assets.json").read_text())
    for name, record in assets.items():
        if sha(args.assets / name) != record["sha256"]:
            raise ValueError(f"Metric asset SHA256 mismatch: {name}")
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists; use --resume or a new output path")
    manifest_path, manifest, native_rows = prepare(args)
    preflight = {"status": "prepared_not_scored", "clips": 60, "variants": list(manifest["models"]),
                 "baseline": args.baseline, "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                 "assets_hashes_verified": True, "scorer_hashes": SCORER_HASHES,
                 "cpu_environment": {key: os.environ[key] for key in CPU_ENV},
                 "predictor_models_loaded": False, "native_difference_records": len(native_rows)}
    atomic_json(args.output.with_name(args.output.stem + "-preflight.json"), preflight)
    if args.prepare_only:
        print(json.dumps(preflight, indent=2))
        return
    import onnxruntime as ort
    if ort.__version__ != "1.29.0":
        raise RuntimeError(f"Scoring requires loaded ONNX Runtime 1.29.0, got {ort.__version__}")
    for name, expected in (("pesq", "0.0.4"), ("pystoi", "0.4.1")):
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Scoring requires {name}=={expected}")
    raw_path = args.output.with_name(args.output.stem + "-scores.json").resolve()
    command = [sys.executable, str((args.scorer_dir / "score.py").resolve()),
               "--manifest", str(manifest_path), "--output", str(raw_path),
               "--assets", str(args.assets.resolve()), "--threads", str(args.threads)]
    if args.resume:
        command.append("--resume")
    completed = subprocess.run(command, check=False)
    if not raw_path.exists():
        raise RuntimeError(f"Scorer exited {completed.returncode} without a results file")
    raw = json.loads(raw_path.read_text())
    if raw["provenance"]["manifest_sha256"] != sha(manifest_path):
        raise RuntimeError("Scorer result does not belong to this prepared manifest")
    if raw.get("status") not in ("complete", "metric_errors"):
        raise RuntimeError("Scoring was interrupted; preserve partial scores and resume the identical invocation")
    result = {"schema_version": 1, "status": "complete" if completed.returncode == 0 and not raw["errors"] else "metric_errors",
              "baseline": args.baseline, "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
              "raw_scores": str(raw_path), "raw_scores_sha256": sha(raw_path),
              "scorer_exit_code": completed.returncode, "provenance": raw["provenance"],
              "wrapper_sha256": sha(Path(__file__)), "absolute_quality": raw["summary"], "errors": raw["errors"],
              "interpretation": "Predicted quality and numerical deltas only. No human ratings, codec timing or automatic quality-equivalence decision.",
              **comparisons(raw, args.baseline, native_rows)}
    atomic_json(args.output, result)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        fields = ["uid", "language", "model", "baseline", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["per_clip"]:
            writer.writerow({**{k: row[k] for k in fields[:4]}, **row["candidate_minus_baseline"]})
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "csv_delta_columns": str(csv_path.resolve())}))
    if result["status"] != "complete":
        raise SystemExit("Metric errors remain explicit; inspect the report")


if __name__ == "__main__":
    main()
