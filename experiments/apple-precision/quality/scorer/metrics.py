"""CPU waveform reconstruction metrics, deliberately outside codec timing.

All comparisons and MOS inputs use a common 16 kHz sample rate, so these
metrics do not measure high-band reconstruction above 8 kHz. No gain fitting,
peak normalization, silence removal, lag search, or time warping is applied.
The caller must remove only known codec padding/delay before calling score.

UTMOS: pinned tarepan/SpeechMOS, an MIT reimplementation of the UTMOS22
strong learner (not the complete UTMOS competition ensemble).
DNSMOS: Microsoft's non-personalized P.835 model and P.808 model; its original
9.01-second overlapping-window/repetition protocol and calibration are used.
Source revisions, checkpoint URLs and SHA256 hashes are in metric_assets.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import signal

BASE = Path(__file__).resolve().parent
METRIC_SR = 16000
SPEECHMOS_COMMIT = "ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
DNSMOS_COMMIT = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"


def _waveform(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 1 or not result.size:
        raise ValueError("Expected a nonempty, mono, one-dimensional waveform")
    if not np.isfinite(result).all():
        raise ValueError("Waveform contains NaN or infinity")
    return np.ascontiguousarray(result)


def resample_for_metrics(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Deterministic polyphase resampling; no amplitude normalization."""
    waveform = _waveform(waveform)
    if int(sample_rate) != sample_rate or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer")
    sample_rate = int(sample_rate)
    if sample_rate == METRIC_SR:
        return waveform
    divisor = math.gcd(sample_rate, METRIC_SR)
    return np.asarray(signal.resample_poly(
        waveform, METRIC_SR // divisor, sample_rate // divisor,
        window=("kaiser", 5.0), padtype="constant",
    ), dtype=np.float32)


class QualityMetrics:
    """Instantiate once and reuse to avoid repeated neural model loading.

    Flat score dictionary: paired metrics, decoded MOS, reference_* MOS,
    plus explicit errors/warnings/metric_metadata. Failed metrics are None.
    Missing neural assets never fall back to a proxy or random checkpoint.
    """

    def __init__(self, include_neural: bool = True, cache_dir: str | Path | None = None,
                 threads: int = 1):
        if threads not in (1, 2):
            raise ValueError("Quality scoring supports one or two CPU threads")
        self.threads = threads
        self.include_neural = include_neural
        self.cache_dir = Path(cache_dir) if cache_dir else BASE / ".cache" / "metrics"
        self._reference_cache: dict[str, dict[str, Any]] = {}
        self._utmos = None
        self._dns = None
        self._neural_errors: dict[str, str] = {}
        self.assets = json.loads((BASE / "metric_assets.json").read_text())
        if include_neural:
            self._load_neural()
            if self._neural_errors:
                raise RuntimeError(f"Requested predictors failed initialization: {self._neural_errors}")

    def _asset(self, filename: str) -> Path:
        path = self.cache_dir / filename
        expected = self.assets[filename]["sha256"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"SHA256 mismatch for metric asset {filename}")
        return path

    def _load_neural(self) -> None:
        try:
            import torch
            torch.set_default_device("cpu")
            torch.set_num_threads(self.threads)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            archive = self._asset("speechmos_source.zip")
            source = self.cache_dir / f"SpeechMOS-{SPEECHMOS_COMMIT}"
            if not (source / "speechmos" / "utmos22" / "strong" / "model.py").exists():
                raise FileNotFoundError(f"Extract pinned SpeechMOS archive under {self.cache_dir}")
            with zipfile.ZipFile(archive) as contents:
                for filename in contents.namelist():
                    if filename.endswith(".py"):
                        if (self.cache_dir / filename).read_bytes() != contents.read(filename):
                            raise ValueError(f"Extracted SpeechMOS source differs from pinned archive: {filename}")
            # These are the inspected files from the pinned upstream archive.
            sys.path.insert(0, str(source))
            from speechmos.utmos22.strong.model import UTMOS22Strong
            state = torch.load(self._asset("utmos22_strong_step7459_v1.pt"),
                               map_location="cpu", weights_only=True)
            self._utmos = UTMOS22Strong().eval().cpu()
            self._utmos.load_state_dict(state, strict=True)
            self._utmos.requires_grad_(False)
            if any(t.device.type != "cpu" for t in list(self._utmos.parameters()) + list(self._utmos.buffers())):
                raise RuntimeError("UTMOS has a non-CPU tensor")
        except Exception as exc:
            self._utmos = None
            self._neural_errors["utmos"] = f"{type(exc).__name__}: {exc}"
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = True
            options.log_severity_level = 3
            self._dns = (
                ort.InferenceSession(str(self._asset("sig_bak_ovr.onnx")),
                                     sess_options=options, providers=["CPUExecutionProvider"]),
                ort.InferenceSession(str(self._asset("model_v8.onnx")),
                                     sess_options=options, providers=["CPUExecutionProvider"]),
            )
            for session in self._dns:
                session.disable_fallback()
                if session.get_providers() != ["CPUExecutionProvider"]:
                    raise RuntimeError("DNSMOS must use CPUExecutionProvider exclusively")
        except Exception as exc:
            self._dns = None
            self._neural_errors["dnsmos"] = f"{type(exc).__name__}: {exc}"

    def provenance(self) -> dict[str, Any]:
        packages = {}
        for name in ("numpy", "scipy", "librosa", "torch", "torchaudio", "onnxruntime", "pystoi", "pesq"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "metric_sample_rate": METRIC_SR,
            "sample_rate_limitation": "All reported quality metrics assess at most 0-8 kHz bandwidth",
            "resampling": "scipy.signal.resample_poly, Kaiser beta 5.0, constant padding",
            "alignment": "No external gain normalization, lag correction, time warping, or silence removal",
            "metric_internal_processing": "pesq 0.0.4 scales both inputs by their common maximum absolute sample, then runs PESQ with its standard level/alignment processing. STOI/ESTOI retain standard normalization and silent-frame exclusion. SI-SDR uses its defining zero-mean scale projection only internally. These metric-specific operations do not alter the shared waveforms or other metrics.",
            "mrstft": {"fft_sizes": [256, 512, 1024, 2048], "hop": "fft_size // 4",
                       "window": "periodic Hann", "boundary": "zeros", "padded": True,
                       "log": "natural log, magnitude floor 1e-7"},
            "utmos": {"implementation": "tarepan/SpeechMOS utmos22_strong",
                      "commit": SPEECHMOS_COMMIT, "device": "cpu",
                      "note": "Single strong learner via SpeechMOS reimplementation; not the full official UTMOS22 ensemble or UTMOSv2"},
            "dnsmos": {"implementation": "microsoft/DNS-Challenge non-personalized P.835 + P.808",
                       "commit": DNSMOS_COMMIT, "device": "CPUExecutionProvider",
                       "threads": self.threads, "window_seconds": 9.01, "hop_seconds": 1,
                       "short_clip_handling": "Repeat waveform by doubling, following upstream"},
            "assets": self.assets,
            "versions": packages,
            "initialization_errors": dict(self._neural_errors),
            "cpu": {"torch_threads": self.threads, "torch_interop_threads": 1,
                    "ort_providers": [s.get_providers() for s in self._dns] if self._dns else [],
                    "environment": {k: os.environ.get(k) for k in (
                        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK", "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS")}},
        }

    @staticmethod
    def _attempt(result: dict[str, Any], key: str, calculate: Callable[[], float]) -> None:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = float(calculate())
            if not math.isfinite(value):
                raise ValueError("Metric returned a non-finite value")
            result[key] = value
            for item in caught:
                result["warnings"].append(f"{key}: {item.message}")
        except Exception as exc:
            result[key] = None
            result["errors"].append(f"{key}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _mrstft(reference: np.ndarray, output: np.ndarray) -> tuple[float, float]:
        convergence, log_mae = [], []
        for n_fft in (256, 512, 1024, 2048):
            # Fixed absolute amplitude scale and transform normalization for both.
            kwargs = dict(fs=METRIC_SR, window="hann", nperseg=n_fft,
                          noverlap=n_fft - n_fft // 4, nfft=n_fft,
                          boundary="zeros", padded=True)
            if min(len(reference), len(output)) < n_fft:
                raise ValueError("MR-STFT requires at least 2048 samples (128 ms)")
            _, _, r = signal.stft(reference, **kwargs)
            _, _, o = signal.stft(output, **kwargs)
            r, o = np.abs(r), np.abs(o)
            denominator = float(np.linalg.norm(r))
            if denominator < 1e-12:
                raise ValueError("Spectral convergence is undefined for silent reference")
            convergence.append(float(np.linalg.norm(r-o)) / denominator)
            log_mae.append(float(np.mean(np.abs(np.log(np.maximum(r, 1e-7)) -
                                                  np.log(np.maximum(o, 1e-7))))))
        return float(np.mean(convergence)), float(np.mean(log_mae))

    def _dnsmos(self, audio: np.ndarray) -> dict[str, float | int | bool]:
        import librosa
        primary, p808 = self._dns
        original_length = len(audio)
        length = int(9.01 * METRIC_SR)
        while len(audio) < length:
            audio = np.concatenate((audio, audio))
        # Preserve upstream window counting exactly (including Python int truncation).
        num_hops = int(np.floor(len(audio) / METRIC_SR) - 9.01) + 1
        values = []
        for index in range(num_hops):
            segment = audio[index * METRIC_SR:int((index + 9.01) * METRIC_SR)]
            if len(segment) < length:
                continue
            mel = librosa.feature.melspectrogram(y=segment[:-160], sr=METRIC_SR,
                                                 n_fft=321, hop_length=160, n_mels=120)
            mel = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T
            inputs = {primary.get_inputs()[0].name: segment[None].astype(np.float32)}
            sig, bak, ovrl = primary.run(None, inputs)[0][0]
            inputs_p808 = {p808.get_inputs()[0].name: mel[None].astype(np.float32)}
            p808_score = float(p808.run(None, inputs_p808)[0][0][0])
            values.append([
                np.polyval([-0.08397278, 1.22083953, 0.0052439], sig),
                np.polyval([-0.13166888, 1.60915514, -0.39604546], bak),
                np.polyval([-0.06766283, 1.11546468, 0.04602535], ovrl),
                p808_score, float(sig), float(bak), float(ovrl),
            ])
        if not values:
            raise ValueError("No complete DNSMOS analysis window")
        keys = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808",
                "dnsmos_sig_raw", "dnsmos_bak_raw", "dnsmos_ovrl_raw")
        result = dict(zip(keys, map(float, np.mean(values, axis=0))))
        result["dnsmos_analysis_windows"] = len(values)
        result["dnsmos_repeated_short_clip"] = original_length < length
        return result

    def _mos(self, audio: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {"errors": [], "warnings": []}
        if not self.include_neural:
            result["neural_metrics_enabled"] = False
            return result
        if self._utmos is None:
            result["utmos"] = None
            result["errors"].append("utmos: " + self._neural_errors.get("utmos", "unavailable"))
        else:
            import torch
            def utmos() -> float:
                with torch.inference_mode():
                    return self._utmos(torch.from_numpy(audio.copy())[None], METRIC_SR).item()
            self._attempt(result, "utmos", utmos)
        dns_keys = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808")
        if self._dns is None:
            result.update({key: None for key in dns_keys})
            result["errors"].append("dnsmos: " + self._neural_errors.get("dnsmos", "unavailable"))
        else:
            try:
                result.update(self._dnsmos(audio))
            except Exception as exc:
                result.update({key: None for key in dns_keys})
                result["errors"].append(f"dnsmos: {type(exc).__name__}: {exc}")
        return result

    def score_reference(self, reference: np.ndarray, reference_sr: int) -> dict[str, Any]:
        """Return/cache reference MOS, using the same processing as decoded audio."""
        audio = resample_for_metrics(reference, reference_sr)
        key = hashlib.sha256(audio.tobytes()).hexdigest()
        if key not in self._reference_cache:
            self._reference_cache[key] = self._mos(audio)
        # Return a copy so caller mutation cannot alter cached errors or metrics.
        return json.loads(json.dumps(self._reference_cache[key]))

    def score(self, reference: np.ndarray, reference_sr: int,
              output: np.ndarray, output_sr: int) -> dict[str, Any]:
        result: dict[str, Any] = {"errors": [], "warnings": []}
        ref = resample_for_metrics(reference, reference_sr)
        out = resample_for_metrics(output, output_sr)
        result["metric_metadata"] = {
            "sample_rate": METRIC_SR, "reference_input_sr": int(reference_sr),
            "output_input_sr": int(output_sr), "reference_samples_16k": len(ref),
            "output_samples_16k": len(out), "common_samples": min(len(ref), len(out)),
            "resampling_rounding_trim_samples": 0,
        }
        paired_keys = ("stoi", "estoi", "pesq_wb", "mrstft_spectral_convergence",
                       "mrstft_log_magnitude_mae")
        if abs(len(ref) - len(out)) > 1:
            result.update({key: None for key in paired_keys})
            result["errors"].append(
                f"paired: unequal 16 kHz lengths ({len(ref)} vs {len(out)}); caller must "
                "remove documented codec padding/delay. No automatic alignment performed."
            )
        else:
            length = min(len(ref), len(out))
            result["metric_metadata"]["resampling_rounding_trim_samples"] = abs(len(ref)-len(out))
            r, o = ref[:length], out[:length]
            from pystoi import stoi
            from pesq import pesq
            self._attempt(result, "stoi", lambda: stoi(r, o, METRIC_SR, extended=False))
            self._attempt(result, "estoi", lambda: stoi(r, o, METRIC_SR, extended=True))
            self._attempt(result, "pesq_wb", lambda: pesq(METRIC_SR, r, o, "wb"))
            try:
                sc, log_mae = self._mrstft(r, o)
                result["mrstft_spectral_convergence"] = sc
                result["mrstft_log_magnitude_mae"] = log_mae
            except Exception as exc:
                result["mrstft_spectral_convergence"] = None
                result["mrstft_log_magnitude_mae"] = None
                result["errors"].append(f"mrstft: {type(exc).__name__}: {exc}")
        for prefix, mos in (("reference_", self.score_reference(reference, reference_sr)),
                            ("", self._mos(out))):
            for key, value in mos.items():
                if key in ("errors", "warnings"):
                    result[key].extend(prefix + message for message in value)
                else:
                    result[prefix + key] = value
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired metrics on actual speech")
    parser.add_argument("--validate-audio", type=Path, required=True,
                        help="A real clean speech recording; no synthetic substitute")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import soundfile as sf
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    waveform, sr = sf.read(args.validate_audio, dtype="float32")
    waveform = resample_for_metrics(waveform, sr)
    # Controlled damage: 1.5 kHz low-pass plus fixed-seed white noise at 5 dB SNR.
    lowpass = signal.sosfilt(signal.butter(6, 1500, fs=METRIC_SR, output="sos"), waveform)
    rng = np.random.default_rng(12345)
    noise = rng.normal(size=waveform.size)
    noise *= np.sqrt(np.mean(waveform ** 2) / (10 ** (5 / 10) * np.mean(noise ** 2)))
    degraded = np.asarray(lowpass + noise, dtype=np.float32)
    metrics = QualityMetrics(include_neural=not args.no_neural)
    identity = metrics.score(waveform, METRIC_SR, waveform, METRIC_SR)
    damaged = metrics.score(waveform, METRIC_SR, degraded, METRIC_SR)
    checks = {
        "identity_stoi_near_one": identity.get("stoi") is not None and identity["stoi"] > .999,
        "identity_estoi_near_one": identity.get("estoi") is not None and identity["estoi"] > .999,
        "identity_spectral_distance_zero": identity.get("mrstft_spectral_convergence") == 0,
        "damage_lowers_stoi": damaged.get("stoi") is not None and damaged["stoi"] < identity["stoi"],
        "damage_lowers_pesq": damaged.get("pesq_wb") is not None and damaged["pesq_wb"] < identity["pesq_wb"],
        "damage_increases_spectral_distance": (damaged.get("mrstft_spectral_convergence") or 0) > 0,
        "no_metric_errors": not identity["errors"] and not damaged["errors"],
    }
    if not args.no_neural:
        for name in ("utmos", "dnsmos_ovrl", "dnsmos_p808"):
            checks[f"damage_lowers_{name}"] = (damaged.get(name) is not None and
                                                identity.get(name) is not None and
                                                damaged[name] < identity[name])
    report = {"audio": str(args.validate_audio.resolve()), "damage": "1.5 kHz LPF plus 5 dB white noise, seed12345",
              "checks": checks, "identity": identity, "degraded": damaged,
              "provenance": metrics.provenance()}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not all(checks.values()):
        raise SystemExit("Metric validation failed; inspect report")


if __name__ == "__main__":
    main()
