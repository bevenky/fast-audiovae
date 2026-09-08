"""Score canonical reconstructions on CPU with explicit, fixed preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Set these before importing numerical packages. No automatic accelerator choice.
for _name, _value in {
    "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_name] = _value
os.environ["NUMBA_CACHE_DIR"] = str(Path(__file__).resolve().parent / "numba-cache")
sys.dont_write_bytecode = True

import numpy as np
from scipy import signal
import soundfile as sf
from metrics import QualityMetrics, METRIC_SR, resample_for_metrics

PRIMARY_KEYS = ("pesq_wb", "stoi", "estoi", "utmos", "dnsmos_sig", "dnsmos_bak",
                "dnsmos_ovrl", "dnsmos_p808", "mrstft_spectral_convergence",
                "mrstft_log_magnitude_mae", "si_sdr_db", "snr_db")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def energy_ratio_db(numerator: float, denominator: float) -> tuple[float | None, str]:
    if numerator <= 0:
        return None, "undefined_zero_target_energy"
    if denominator == 0:
        return None, "positive_infinity_exact_match"
    return 10.0 * math.log10(numerator / denominator), "finite"


def fidelity_metrics(reference: np.ndarray, output: np.ndarray) -> dict:
    """SI-SDR uses standard zero-mean projection, without modifying any other metric."""
    r, o = reference.astype(np.float64), output.astype(np.float64)
    snr, snr_status = energy_ratio_db(float(np.dot(r, r)), float(np.dot(r-o, r-o)))
    r, o = r-r.mean(), o-o.mean()
    rr = float(np.dot(r, r))
    if rr <= 0:
        return {"si_sdr_db": None, "si_sdr_status": "undefined_silent_reference",
                "snr_db": snr, "snr_status": snr_status}
    target = r * (float(np.dot(o, r)) / rr)
    residual = o-target
    sisdr, status = energy_ratio_db(float(np.dot(target, target)), float(np.dot(residual, residual)))
    return {"si_sdr_db": sisdr, "si_sdr_status": status,
            "snr_db": snr, "snr_status": snr_status}


def aligned_overlap(reference: np.ndarray, output: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive lag denotes output delay. Used only for labeled diagnostics."""
    if lag > 0:
        return reference[:-lag], output[lag:]
    if lag < 0:
        return reference[-lag:], output[:lag]
    return reference, output


def lag_diagnostic(reference: np.ndarray, output: np.ndarray, aligned_stoi: bool = False) -> dict:
    if len(reference) != len(output) or len(reference) < 2:
        raise ValueError("Lag diagnostic requires equal, nontrivial waveform lengths")
    r = reference.astype(np.float64)
    o = output.astype(np.float64)
    r -= r.mean()
    o -= o.mean()
    n = len(r)
    bound = min(int(.250 * METRIC_SR), n-1)
    lags = np.arange(-bound, bound+1)
    correlation = signal.correlate(o, r, mode="full", method="fft")[n-1+lags]
    renergy = np.r_[0., np.cumsum(r*r)]
    oenergy = np.r_[0., np.cumsum(o*o)]
    start_o = np.maximum(lags, 0)
    start_r = np.maximum(-lags, 0)
    length = n-np.abs(lags)
    energy_r = renergy[start_r+length]-renergy[start_r]
    energy_o = oenergy[start_o+length]-oenergy[start_o]
    denominator = np.sqrt(np.maximum(energy_r*energy_o, 0))
    valid = denominator > 1e-20
    normalized = np.zeros_like(correlation)
    normalized[valid] = correlation[valid]/denominator[valid]
    if not valid.any():
        return {"status": "undefined_silent_signal", "external_shift_applied_to_primary": False}
    index = int(np.argmax(normalized))
    lag = int(lags[index])
    peak, zero = float(normalized[index]), float(normalized[bound])
    candidate = abs(lag) >= 16 and peak > .2 and peak-zero > .02
    result = {"status": "ok", "lag_samples_16k": lag, "lag_ms": lag*1000/METRIC_SR,
              "positive_lag_means": "output delayed relative to reference",
              "max_search_ms": 250, "global_mean_removed_for_diagnostic": True,
              "zero_lag_correlation": zero, "peak_correlation": peak,
              "at_search_boundary": abs(lag) == bound,
              "possible_constant_delay": candidate,
              "external_shift_applied_to_primary": False}
    if aligned_stoi and candidate:
        from pystoi import stoi
        rr, oo = aligned_overlap(reference, output, lag)
        result["supplementary_aligned_stoi"] = float(stoi(rr, oo, METRIC_SR, extended=False))
        result["supplementary_aligned_estoi"] = float(stoi(rr, oo, METRIC_SR, extended=True))
        result["aligned_overlap_samples"] = len(rr)
        result["supplementary_warning"] = "Output-fitted constant shift and shorter overlap; not a primary score and not evidence of codec equality."
    return result


def load_wave(path_text: str, expected_rate: int, manifest_dir: Path, expected_samples: int | None = None,
              expected_sha256: str | None = None) -> tuple[np.ndarray, dict]:
    path = Path(path_text)
    if not path.is_absolute():
        path = manifest_dir / path
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if rate != expected_rate or audio.ndim != 1 or not len(audio):
        raise ValueError(f"Rate/mono/nonempty contract mismatch: {path}")
    if expected_samples is not None and len(audio) != expected_samples:
        raise ValueError(f"Length mismatch: {path}: {len(audio)} != {expected_samples}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    file_digest = sha256(path)
    if expected_sha256 is not None and file_digest != expected_sha256:
        raise ValueError(f"Manifest waveform hash mismatch: {path}")
    metric_audio = resample_for_metrics(audio, rate)
    return audio, {"path": str(path.resolve()), "file_sha256": file_digest,
                   "sample_rate": rate, "samples": len(audio), "duration_s": len(audio)/rate,
                   "wav_subtype": info.subtype, "peak_abs": float(np.max(np.abs(audio))),
                   "rms": float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                   "samples_16k": len(metric_audio),
                   "float32_16k_sha256": hashlib.sha256(metric_audio.tobytes()).hexdigest()}


def summarize(rows: list[dict]) -> dict:
    model_ids = sorted({row["model"] for row in rows})
    result = {"aggregation": "Equal mean across language means; each language mean is unweighted over utterances. No duration weighting.",
              "models": {}, "paired_differences": {}, "bootstrap": {"seed": 20260907, "resamples": 10000}}
    for model in model_ids:
        selected = [row for row in rows if row["model"] == model]
        fields = {}
        for key in PRIMARY_KEYS:
            values = [row["metrics"][key] for row in selected
                      if isinstance(row["metrics"].get(key), (float, int))]
            if values:
                array = np.asarray(values)
                by_language = {}
                for row in selected:
                    value = row["metrics"].get(key)
                    if isinstance(value, (float, int)):
                        by_language.setdefault(row.get("language", "unspecified"), []).append(value)
                fields[key] = {"mean": float(np.mean([np.mean(v) for v in by_language.values()])),
                               "utterance_mean": float(array.mean()),
                               "std_sample": float(array.std(ddof=1)) if len(array)>1 else None,
                               "minimum": float(array.min()), "maximum": float(array.max()), "count": len(values)}
        result["models"][model] = {"clips": len(selected), "metrics": fields,
                                    "metric_errors": sum(len(row["metrics"].get("errors", [])) for row in selected),
                                    "language_counts": {language: sum(row.get("language", "unspecified") == language for row in selected)
                                                        for language in sorted({row.get("language", "unspecified") for row in selected})},
                                    "constant_delay_flags": sum(bool((row.get("lag_diagnostic") or {}).get("possible_constant_delay")) for row in selected)}
    indexed = {(row["model"], row["uid"]): row for row in rows}
    for first, second in itertools.combinations(model_ids, 2):
        uids = sorted({row["uid"] for row in rows if row["model"] == first} &
                      {row["uid"] for row in rows if row["model"] == second})
        comparison = {}
        for key in PRIMARY_KEYS:
            paired = [(indexed[first, uid]["metrics"].get(key), indexed[second, uid]["metrics"].get(key),
                       indexed[first, uid].get("language", "unspecified")) for uid in uids]
            groups = {}
            for a, b, language in paired:
                if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                    groups.setdefault(language, []).append(a-b)
            differences = np.asarray([value for group in groups.values() for value in group])
            if differences.size:
                rng = np.random.default_rng(20260907)
                group_samples = []
                for language in sorted(groups):
                    values = np.asarray(groups[language])
                    group_samples.append(values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1))
                means = np.mean(group_samples, axis=0)
                comparison[key] = {"mean_first_minus_second": float(np.mean([np.mean(group) for group in groups.values()])),
                                   "bootstrap_method": "Paired within-language resampling, equal-weight mean of fixed language strata",
                                   "paired_bootstrap_95_percentile_ci": list(map(float, np.quantile(means, [.025, .975]))),
                                   "pairs": len(differences), "positive_pairs": int((differences>0).sum()),
                                   "negative_pairs": int((differences<0).sum()), "exact_ties": int((differences==0).sum())}
        result["paired_differences"][first + " minus " + second] = comparison
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1, choices=(1, 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supplementary-aligned-stoi", action="store_true")
    args = parser.parse_args()
    # Capture one immutable stage while a later model may be published atomically.
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not (manifest.get("stage_ready") is True or manifest.get("status") == "ready"):
        raise ValueError("Manifest must declare stage_ready or status='ready'")
    samples = manifest["samples"]
    uids = [entry["uid"] for entry in samples]
    if not uids or len(uids) != len(set(uids)):
        raise ValueError("Manifest requires nonempty samples with unique uid values")
    output_models = set(samples[0]["outputs"])
    if not output_models or any(set(entry["outputs"]) != output_models for entry in samples):
        raise ValueError("Every sample must have the same completed output model set at each scoring stage")
    started = time.perf_counter()
    metrics = QualityMetrics(cache_dir=args.assets, threads=args.threads)
    provenance = metrics.provenance()
    provenance.update({"python": sys.version, "platform": platform.platform(),
                       "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("score.py", "metrics.py", "metric_assets.json")},
                       "supplementary_aligned_stoi": args.supplementary_aligned_stoi,
                       "source_manifest": str(args.manifest.resolve()),
                       "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    provenance["paired_metric_installed_file_sha256"] = {}
    for package in ("pesq", "pystoi"):
        distribution = importlib.metadata.distribution(package)
        provenance["paired_metric_installed_file_sha256"][package] = {
            str(path): sha256(Path(distribution.locate_file(path)))
            for path in distribution.files
            if str(path).startswith(package + "/") and str(path).endswith((".py", ".so"))}
    configuration = {key: value for key, value in provenance.items() if key not in ("source_manifest", "manifest_sha256")}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    old = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["configuration_sha256"] != config_hash:
            raise ValueError("Resume refused: metric source, dependencies or configuration changed")
        old = {(row["uid"], row["model"]): row
               for row in [*previous.get("pending_cached_rows", []), *previous["rows"]]}
    report = {"schema_version": 1, "status": "running", "configuration_sha256": config_hash,
              "manifest": manifest, "provenance": provenance, "rows": [], "errors": []}
    expected = sum(1 + len(entry["outputs"]) for entry in samples)
    for sample in samples:
        uid = sample["uid"]
        ref, ref_info = load_wave(sample["reference_path"], sample["reference_rate"], args.manifest.parent,
                                  sample["reference_samples"], sample.get("reference_sha256"))
        if sample["reference_rate"] != METRIC_SR:
            raise ValueError("Canonical reference must be 16 kHz")
        for model in ["original_reference", *sample["outputs"]]:
            begin = time.perf_counter()
            if model == "original_reference":
                wave, wave_info = ref, ref_info
            else:
                record = sample["outputs"][model]
                wave, wave_info = load_wave(record["path"], record["sample_rate"], args.manifest.parent,
                                            record.get("samples"), record.get("sha256"))
                if abs(wave_info["duration_s"]-ref_info["duration_s"]) > 1/wave_info["sample_rate"]:
                    raise ValueError(f"Untrimmed duration mismatch for {uid}/{model}")
            prior = old.get((uid, model))
            if (prior and prior["input"]["file_sha256"] == wave_info["file_sha256"]
                    and prior["reference"]["file_sha256"] == ref_info["file_sha256"]
                    and prior.get("language", "unspecified") == sample.get("language", "unspecified")):
                row = prior
                if model == "original_reference":
                    metrics._reference_cache[ref_info["float32_16k_sha256"]] = row["metrics"]
            else:
                if model == "original_reference":
                    scores = metrics.score_reference(ref, sample["reference_rate"])
                    lag = None
                else:
                    scores = metrics.score(ref, sample["reference_rate"], wave, wave_info["sample_rate"])
                    rr = resample_for_metrics(ref, sample["reference_rate"])
                    oo = resample_for_metrics(wave, wave_info["sample_rate"])
                    if abs(len(rr)-len(oo)) > 1:
                        raise ValueError("Resampled lengths differ by more than one sample")
                    length = min(len(rr), len(oo))
                    scores.update(fidelity_metrics(rr[:length], oo[:length]))
                    lag = lag_diagnostic(rr[:length], oo[:length], args.supplementary_aligned_stoi)
                row = {"uid": uid, "language": sample.get("language", "unspecified"),
                       "model": model, "input": wave_info, "reference": ref_info,
                       "metrics": scores, "lag_diagnostic": lag, "scoring_seconds": time.perf_counter()-begin}
            report["rows"].append(row)
            old.pop((uid, model), None)
            # Retain unvisited earlier-stage results during an interrupted
            # resume. Each is hash-checked before entering the current rows.
            report["pending_cached_rows"] = list(old.values())
            if row["metrics"].get("errors"):
                report["errors"].append({"uid": uid, "model": model, "errors": row["metrics"]["errors"]})
            report["completed_signals"] = len(report["rows"])
            report["expected_signals"] = expected
            report["invocation_elapsed_seconds"] = time.perf_counter()-started
            atomic_json(args.output, report)
            print(f"{len(report['rows'])}/{expected} {uid} {model}: {'reused' if row is prior else 'scored'}", flush=True)
    report["summary"] = summarize(report["rows"])
    report.pop("pending_cached_rows", None)
    report["status"] = "complete" if not report["errors"] else "metric_errors"
    report["invocation_elapsed_seconds"] = time.perf_counter()-started
    atomic_json(args.output, report)
    atomic_json(args.output.with_name(args.output.stem + "-summary.json"), report["summary"])
    if report["errors"]:
        raise SystemExit("Requested metric failures were recorded; inspect results")


if __name__ == "__main__":
    main()
