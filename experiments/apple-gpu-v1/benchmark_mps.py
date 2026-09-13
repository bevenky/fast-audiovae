"""Short, completed-output MPS comparison; run only under an exclusive lease.

Original continuous latents, independent fresh 960 ms streams, no audio export.
The base-GPU arm is an experiment around the same model as the public API:
it omits the public API's all-history finite reduction, not model arithmetic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time

for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(name, "0") != "0":
        raise RuntimeError(name + " must be disabled before Torch import")
    os.environ[name] = "0"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_THREAD_LIMIT", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", ORT_DISABLE_TELEMETRY="1",
                  CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "work/fast-audiovae-apple-gpu/src"
CONFIG = ROOT / "outputs/apple-streaming-promotion-v1/config-r3.json"
ORIGINAL = ROOT / "work/audiovae2_onnx/audio_vae_decoder.onnx"
SELECTED = "5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841"
MIMI_FULL = "e023777a2cee98a7e4a293f6af7712d4fed6364d1523cfc9a6283f112981e2f1"
CLIPS = (("Bengali", "bn_in_00151_1818"), ("English", "en_us_00103_1779"),
         ("Spanish", "es_419_00060_1994"))
AUDIO = ("audiovae2_cpu", "audiovae2_public_gpu", "audiovae2_base_gpu")
sys.path.insert(0, str(SOURCE))

import numpy as np
import onnxruntime as ort
import torch
import fast_audiovae
from fast_audiovae.assets import MODEL_FILES, verify_model
from mimi_mps import MimiMPSAdapter


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audio_check(value, samples):
    require(isinstance(value, np.ndarray) and value.dtype == np.float32
            and value.shape == (1, 1, samples) and np.isfinite(value).all(), "Invalid CPU-ready audio")


def comparison(value, reference):
    audio_check(value, reference.shape[-1])
    audio_check(reference, reference.shape[-1])
    difference = value.astype(np.float64) - reference
    return dict(passed=bool(np.allclose(value, reference, atol=1e-5, rtol=1e-4)),
                max_abs=float(np.abs(difference).max()), rms=float(np.sqrt(np.mean(difference ** 2))))


class BasicGPUStream:
    """Identical input/output boundary for Mimi and direct Audio model calls."""
    def __init__(self, decoder, mimi):
        self.decoder, self.mimi = decoder, mimi
        self.upstream = decoder.stream(max_frames=12) if mimi else None
        self.states, self.frames_decoded = {}, 0

    def decode_chunk(self, value):
        channels = 32 if self.mimi else 64
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and value.ndim == 3 and value.shape[:2] == (1, channels)
                and np.isfinite(value).all(), "Invalid raw FP32 latent input")
        owned = np.array(value, copy=True, order="C")
        with self.decoder._lock, torch.inference_mode(), torch.autocast("mps", enabled=False):
            latent = torch.from_numpy(owned).to("mps", dtype=torch.float32, non_blocking=False)
            if self.mimi:
                result = self.upstream.decode_chunk(latent)
                state = None
            else:
                result, state = self.decoder._model.decode(latent, dict(self.states))
            require(result.device.type == "mps" and result.dtype == torch.float32
                    and tuple(result.shape) == (1, 1, value.shape[-1] * 1920), "Invalid MPS audio")
            ready = result.detach().to("cpu", non_blocking=False)
            torch.mps.synchronize()
            output = ready.numpy().copy()
            audio_check(output, value.shape[-1] * 1920)
            if not self.mimi:
                self.states = state
            self.frames_decoded += value.shape[-1]
            return output

    def flush(self):
        if not self.mimi:
            return np.empty((1, 1, 0), np.float32)
        with torch.inference_mode():
            ready = self.upstream.flush().to("cpu", non_blocking=False)
            torch.mps.synchronize()
            output = ready.numpy().copy()
            audio_check(output, 0)
            return output

    def close(self):
        if self.upstream is not None:
            self.upstream.close()
        self.states.clear()


def cpu_reference(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    require(session.get_providers() == ["CPUExecutionProvider"]
            and len(session.get_inputs()) == len(session.get_outputs()) == 1, "Invalid full CPU reference")
    return session


def main(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write('{"status":"starting"}\n')
    result = dict(version="apple_mps_matched_benchmark_v1", status="running", rows=[],
                  reference_calls=[], calls=0, decode_seconds=0.0, phase_seconds={}, provenance={}, clips=[])
    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    def call(phase, fn, sink):
        require(result["decode_seconds"] < 25, "25-second decode budget exhausted")
        start = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed = time.perf_counter() - start
            sink.append(elapsed)
            result["calls"] += 1
            result["decode_seconds"] += elapsed
            result["phase_seconds"][phase] = result["phase_seconds"].get(phase, 0.0) + elapsed
            require(result["decode_seconds"] <= 25, "25-second decode budget exceeded; stopping without repeats")
    try:
        require(torch.__version__.split("+")[0] == "2.14.0" and ort.__version__ == "1.30.0", "Wrong runtime versions")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1
                and torch.get_default_dtype() == torch.float32 and torch.backends.mps.is_available(), "Wrong MPS/thread policy")
        require(Path(fast_audiovae.__file__).resolve().is_relative_to(SOURCE), "Wrong AudioVAE2 package")
        config = json.loads(CONFIG.read_text())
        bundle = Path(config["bundle"])
        require(sha(bundle / "bundle.json") == config["bundle_manifest_sha256"]
                and config["stream_graph_sha256"] == SELECTED, "Wrong selected CPU baseline")
        manifest = json.loads((bundle / "bundle.json").read_text())
        native = manifest["native"]["Darwin/arm64"]
        stream_spec = manifest["streaming"]["models"][native["model"]]
        require(stream_spec["model_sha256"] == SELECTED and len(stream_spec["states"]) == 26, "Wrong CPU streaming contract")
        # Verify every selected library, including the unregistered LIBXSMM dependency.
        files = {bundle / "bundle.json": config["bundle_manifest_sha256"],
                 bundle / stream_spec["model"]: SELECTED,
                 bundle / native["library"]: native["library_sha256"]}
        for item in stream_spec["additional_libraries"] + stream_spec["dependencies"]:
            files[bundle / item["library"]] = item["sha256"]
        for key in ("latents", "latent_manifest", "mimi_latents", "mimi_latent_manifest"):
            files[Path(config[key])] = config["inputs_sha256"][key]
        verify_model(ORIGINAL)
        for name, digest in MODEL_FILES.items():
            files[ORIGINAL.parent / name] = digest
        mimi_path = Path(config["mimi_bundle"]) / "mimi_full.onnx"
        files[mimi_path] = MIMI_FULL
        for path, expected in files.items():
            require(sha(path) == expected, "Artifact checksum mismatch: " + str(path))
        source_files = [Path(__file__), HERE / "mimi_mps.py", CONFIG]
        source_files += [SOURCE / "fast_audiovae" / name for name in
                         ("__init__.py", "automatic.py", "gpu.py", "mps_decoder.py", "runtime.py", "streaming.py")]
        files.update({path: sha(path) for path in source_files})
        stamps = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
        result["provenance"] = dict(files={str(path): digest for path, digest in files.items()},
                                    torch=torch.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
                                    platform=platform.platform(), machine=platform.machine(), host_threads=1,
                                    fast_math=False, cpu_fallback=False, precision="FP32")
        cpu, cpu_info = fast_audiovae.load_streaming_decoder(bundle, threads=1, prefer_custom=True)
        require(cpu_info["selected"] == "native" and cpu_info["threads"] == 1
                and cpu_info["providers"] == ["CPUExecutionProvider"]
                and sha(bundle / cpu_info["model"]) == SELECTED, "Native CPU selection failed")
        audio = fast_audiovae.load(device="gpu", mode="streaming", threads=1, source=ORIGINAL,
                                 offline=True, cache_dir=HERE / "cache")
        require(audio.info["selected"] == "torch_mps" and audio.info["fallback"] is False, "GPU selection failed")
        mimi = MimiMPSAdapter(max_frames=12)
        result["models"] = dict(audiovae2_cpu=cpu_info, audiovae2_gpu=audio.info, mimi_gpu=mimi.metadata)
        references = {"audio": cpu_reference(ORIGINAL), "mimi": cpu_reference(mimi_path)}
        crops = {}
        with np.load(config["latents"], allow_pickle=False) as a, np.load(config["mimi_latents"], allow_pickle=False) as m:
            for label, uid in CLIPS:
                az, mz = a[uid + "__z"], m[uid + "__z"]
                require(az.dtype == mz.dtype == np.float32 and az.shape[:2] == (1, 64)
                        and mz.shape[:2] == (1, 32), "Unexpected latent schema")
                available = min(az.shape[-1] // 2, mz.shape[-1])
                require(available >= 12, "Clip shorter than 960 ms")
                start = (available - 12) // 2
                crops[label] = dict(audio=np.ascontiguousarray(az[..., 2 * start:2 * start + 24]),
                                    mimi=np.ascontiguousarray(mz[..., start:start + 12]))
                require(all(np.isfinite(z).all() for z in crops[label].values()), "Nonfinite crop")
                result["clips"].append(dict(label=label, uid=uid, start_ms=80 * start, duration_ms=960,
                    audio_frames=24, mimi_frames=12, audio_samples=46080, mimi_samples=23040,
                    crop_sha256={k: tensor_sha(v) for k, v in crops[label].items()}))
        result["protocol"] = dict(warmup_sweeps=2, measured_sweeps=3, packet_ms=[40, 80], mimi_40_ms="not supported",
            order="Within each clip/packet-size group, reverse all arms on alternating (sweep+clip) parity; 3 repetitions have a 2/1 direction imbalance.",
            sample_selection="Middle shared-duration 960 ms aligned to 80 ms, fresh state at the crop start; no previous context replay.",
            reference="Each codec's original full CPU ONNX on that exact crop, not a crop of a longer decoded waveform.",
            tolerances=dict(atol=1e-5, rtol=1e-4), budget_seconds=25,
            timing="Every packet includes new latent upload, completed model work, owned CPU-ready output; includes per-call validation and flush.",
            exclusions="Model/session creation, stream construction and its MPS synchronization, close, reference comparison and report writing. First-call lazy state initialization inside decode stays timed.",
            boundary="Public Audio validates audio and all 26 histories. Base Audio and Mimi validate NumPy input/output with identical upload/copy/synchronization; base Audio uses unchanged public-model weights/arithmetic and independent state.",
            short_evidence=True, aggregate="RTF=sum completed packet+flush times / emitted audio duration; all measured repetitions retained.")
        refs = {}
        for label, values in crops.items():
            refs[label] = {}
            for codec, z in values.items():
                session = references[codec]
                record = dict(label=label, codec=codec, seconds=[])
                result["reference_calls"].append(record)
                refs[label][codec] = call("reference", lambda: session.run(None, {session.get_inputs()[0].name: z})[0], record["seconds"])
                audio_check(refs[label][codec], z.shape[-1] * 1920)
                record["output_sha256"] = tensor_sha(refs[label][codec])

        def run_stream(phase, repetition, label, method, packet_ms, order):
            codec = "mimi" if method == "mimi_gpu" else "audio"
            z = crops[label][codec]
            step = 1 if codec == "mimi" else packet_ms // 40
            factory = {"audiovae2_cpu": cpu.streaming_decode, "audiovae2_public_gpu": audio.stream,
                       "audiovae2_base_gpu": lambda: BasicGPUStream(audio, False),
                       "mimi_gpu": lambda: BasicGPUStream(mimi, True)}[method]
            stream = factory()
            row = dict(phase=phase, repetition=repetition, clip=label, method=method, packet_ms=packet_ms,
                       order=list(order), packet_seconds=[], flush_seconds=[], status="running")
            result["rows"].append(row)
            try:
                if method != "audiovae2_cpu":
                    torch.mps.synchronize()  # Exclude queued stream-construction work, including Mimi KV initialization.
                parts = []
                for start in range(0, z.shape[-1], step):
                    packet = z[..., start:start + step]
                    parts.append(call(phase, lambda: stream.decode_chunk(packet), row["packet_seconds"]))
                tail = call(phase, stream.flush, row["flush_seconds"])
                audio_check(tail, 0)
                require(stream.frames_decoded == z.shape[-1], "Wrong consumed frame count")
                joined = np.concatenate(parts + [tail], axis=-1)
                row["comparison"] = comparison(joined, refs[label][codec])
                require(row["comparison"]["passed"], "Waveform gate failed: " + method + "/" + label)
                row.update(status="passed", frames=z.shape[-1], samples=joined.shape[-1],
                           seconds=sum(row["packet_seconds"]) + sum(row["flush_seconds"]))
                row["rtf"] = row["seconds"] / .96
            finally:
                stream.close()

        # All seven active routes on all three crops pass before warmup/timing.
        for phase, sweeps in (("qualification", 1), ("warmup", 2), ("measured", 3)):
            for repetition in range(sweeps):
                for clip_index, (label, _) in enumerate(CLIPS):
                    for packet_ms in (40, 80):
                        order = list(AUDIO) + (["mimi_gpu"] if packet_ms == 80 else [])
                        if (repetition + clip_index) % 2:
                            order.reverse()
                        for method in order:
                            run_stream(phase, repetition, label, method, packet_ms, order)
                save()
                print(f"{phase} sweep {repetition + 1}/{sweeps}: {result['calls']} calls, {result['decode_seconds']:.3f}s decode", flush=True)
        require(result["calls"] == 2292 and len(result["rows"]) == 126, "Incomplete fixed benchmark accounting")
        require(all((path.stat().st_size, path.stat().st_mtime_ns) == stamps[str(path)] for path in files), "An input artifact changed")
        require(all(sha(path) == files[path] for path in source_files), "Benchmark/package source changed")
        require(all(tensor_sha(crops[item["label"]][codec]) == digest for item in result["clips"]
                    for codec, digest in item["crop_sha256"].items()), "Latents mutated")
        result["artifacts_unchanged"] = True
        result["summary"] = []
        measured = [row for row in result["rows"] if row["phase"] == "measured"]
        for packet_ms in (40, 80):
            for method in list(AUDIO) + (["mimi_gpu"] if packet_ms == 80 else []):
                rows = [r for r in measured if r["method"] == method and r["packet_ms"] == packet_ms]
                packets = [t for r in rows for t in r["packet_seconds"]]
                result["summary"].append(dict(method=method, packet_ms=packet_ms, streams=len(rows),
                    median_stream_rtf=statistics.median(r["rtf"] for r in rows),
                    pooled_rtf=sum(r["seconds"] for r in rows) / (len(rows) * .96),
                    median_packet_ms=1000 * statistics.median(packets), p95_packet_ms=float(np.percentile(packets, 95) * 1000),
                    median_first_packet_ms=1000 * statistics.median(r["packet_seconds"][0] for r in rows)))
        result["status"] = "passed"
        mimi.close()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(dict(status=result["status"], calls=result["calls"], decode_seconds=result["decode_seconds"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.output.resolve())
