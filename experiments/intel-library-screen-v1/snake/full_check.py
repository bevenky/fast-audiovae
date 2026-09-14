"""Short, resident-session Intel Snake A/B screen; export aggregate metrics only."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

UIDS = ("bn_in_00151_1818", "en_us_00103_1779", "es_419_00060_1994")
MODES = (0, 2)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=int, choices=(40, 80), default=80)
    parser.add_argument("--bundle", type=Path, default=Path("/dev/shm/fast-audiovae-projection-candidate-20260908-r1"))
    parser.add_argument("--runtime-source", type=Path, default=Path("/var/tmp/intel-streaming-transfer-v1/src"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.build.read_text())
    assert cfg["status"] == "built"
    if not args.worker:
        if args.output.exists():
            raise FileExistsError("Use a fresh output path; earlier reports are preserved")
        assert not os.environ.get("LD_PRELOAD")
        env = dict(os.environ)
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT", "NUMEXPR_NUM_THREADS")})
        env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", LD_PRELOAD=cfg["shim"], **cfg["resolvers"])
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], env=env).returncode
    os.sched_setaffinity(0, {0})
    assert "GenuineIntel" in Path("/proc/cpuinfo").read_text()
    assert os.environ["LD_PRELOAD"] == cfg["shim"]
    pins = {**cfg["pins"], str(args.build.resolve()): sha(args.build), str(Path(__file__).resolve()): sha(__file__), str(args.bundle / "bundle.json"): sha(args.bundle / "bundle.json")}
    for path, expected in pins.items():
        assert sha(path) == expected, path
    import numpy as np
    import onnxruntime as ort
    assert ort.__version__ == cfg["ort"] == "1.29.0"
    shim = ctypes.CDLL(cfg["shim"], mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    shim.sk_error.argtypes = []; shim.sk_error.restype = ctypes.c_char_p
    for name in ("sk_initialize", "sk_clear"):
        getattr(shim, name).argtypes = []; getattr(shim, name).restype = ctypes.c_int
    shim.sk_set_mode.argtypes = [ctypes.c_int]; shim.sk_set_mode.restype = ctypes.c_int
    shim.sk_summary.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]; shim.sk_summary.restype = ctypes.c_int

    def ok(result):
        if result:
            raise RuntimeError(shim.sk_error().decode())

    def counters():
        out = (ctypes.c_uint64 * 5)()
        ok(shim.sk_summary(out, 5))
        return {k: int(v) for k, v in zip(("captures", "capture_bytes", "intercepted", "unsupported", "resolved"), out)}

    class Metrics:
        def __init__(self):
            self.arrays = self.elements = self.exact_arrays = self.different_elements = 0
            self.max_abs = self.error_square = self.reference_square = self.candidate_square = self.dot = 0.0

        def add(self, reference, candidate):
            assert reference.shape == candidate.shape and reference.dtype == candidate.dtype == np.float32
            assert np.isfinite(reference).all() and np.isfinite(candidate).all()
            x = reference.astype(np.float64); y = candidate.astype(np.float64); error = y - x
            different = np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32))
            self.arrays += 1; self.elements += x.size; self.exact_arrays += int(different == 0)
            self.different_elements += int(different)
            self.max_abs = max(self.max_abs, float(np.max(np.abs(error), initial=0)))
            self.error_square += float(np.sum(error * error)); self.reference_square += float(np.sum(x * x))
            self.candidate_square += float(np.sum(y * y)); self.dot += float(np.sum(x * y))

        def summary(self):
            denominator = (self.reference_square * self.candidate_square) ** .5
            return {"arrays": self.arrays, "elements": self.elements, "shape_and_finite_checks_passed": True,
                    "bitwise_exact_arrays": self.exact_arrays, "different_elements": self.different_elements,
                    "max_abs": self.max_abs, "rms_error": (self.error_square / max(self.elements, 1)) ** .5,
                    "relative_l2": self.error_square ** .5 / max(self.reference_square ** .5, 1e-12),
                    "cosine": self.dot / denominator if denominator else (1.0 if not self.error_square else None)}

    report = {"status": "preparing", "protocol_version": 2, "packet_ms": args.packet, "threads": 1, "cpu": 0, "cpu_only": True,
              "ort": ort.__version__, "graph": cfg["graph"], "pins": pins, "pairs": [], "gates": [],
              "timing_scope": "One resident combined-graph session, serial mode 0/mode 2 complete prefixes with separate reset streaming state. Alternating order. Completed Session.run time only; controls, state checks, comparisons and setup excluded.",
              "workload": "Three 1.6-second prefixes, one warm pass per prefix and variant, three measured repetitions, selected packet size.",
              "numerical_scope": "Mode 0 is canonical Snake; mode 2 uses oneMKL HA whole-call sine only for intercepted dynamic calls. Warm and gate passes compare all 18 states each packet; timed passes retain packet audio and final 18 states only. Tiny differences are measured, not assumed safe. No automatic quality promotion.",
              "timing_storage": "Timed calls release obsolete histories, as normal streaming does, and skip per-packet finite scans. Complete history traces and finite scans are retained only in untimed warm/gate passes; timed waveform/final histories are checked after each prefix. Earlier protocol1 timings retained every history and are not directly comparable.",
              "relative_l2_denominator_floor": 1e-12, "native_budget_seconds_per_mode": 25,
              "privacy": "Audio, latent, feature and history arrays remain in RAM; reports contain only aggregate errors, timing and counts."}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    native_ns = {mode: 0 for mode in MODES}; native_calls = {mode: 0 for mode in MODES}
    intercepted = {mode: 0 for mode in MODES}
    save()
    try:
        opts = ort.SessionOptions(); opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            opts.add_session_config_entry(key, "0")
        for lib in cfg["libraries"]:
            opts.register_custom_ops_library(lib)
        session = ort.InferenceSession(cfg["graph"], opts, providers=["CPUExecutionProvider"])
        session.disable_fallback(); assert session.get_providers() == ["CPUExecutionProvider"]
        ok(shim.sk_initialize()); ok(shim.sk_set_mode(0)); ok(shim.sk_clear())
        mkl = ctypes.CDLL(cfg["resolvers"]["SK_MKL_LIBRARY"])
        mkl.mkl_get_max_threads.argtypes = []; mkl.mkl_get_max_threads.restype = ctypes.c_int
        assert mkl.mkl_get_max_threads() == 1
        bundle = json.loads((args.bundle / "bundle.json").read_text())
        entry = bundle["streaming"]["models"][bundle["native"]["Linux/x86_64"]["model"]]
        states = entry["states"]; assert len(states) == 18
        names = [entry["audio_output"], *[s["output"] for s in states]]
        assert set(names) == {o.name for o in session.get_outputs()}
        latent = entry["latent_input"]
        assert {latent, *[s["input"] for s in states]} == {i.name for i in session.get_inputs()}
        with np.load(cfg["latents"], allow_pickle=False) as archive:
            zs = {uid: np.ascontiguousarray(archive[uid + "__z"][..., :40]) for uid in UIDS}
        assert all(z.shape == (1, 64, 40) and z.dtype == np.float32 for z in zs.values())
        all_wave = Metrics(); all_states = {s["input"]: Metrics() for s in states}

        def native_call(mode, output_names, feed):
            assert native_ns[mode] < 25_000_000_000, "Short-run native budget exhausted"
            start = time.perf_counter_ns()
            output = session.run(output_names, feed)
            elapsed = time.perf_counter_ns() - start
            native_ns[mode] += elapsed; native_calls[mode] += 1
            assert native_ns[mode] < 25_000_000_000, "Short-run native budget exhausted"
            return output, elapsed

        def run(mode, z, chunks, full_trace=False):
            before = counters(); ok(shim.sk_set_mode(mode))
            history = {s["input"]: np.zeros(s["shape"], dtype=s["dtype"]) for s in states}
            trace = []; audio = []; elapsed = pos = 0
            try:
                for size in chunks:
                    x = np.ascontiguousarray(z[..., pos:pos + size]); assert x.shape[-1] == size
                    values, ns = native_call(mode, names, {latent: x, **history}); elapsed += ns
                    assert len(values) == 19 and values[0].shape == (1, 1, size * 1920)
                    if full_trace:
                        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in values)
                    history = {s["input"]: v for s, v in zip(states, values[1:])}
                    assert all(list(history[s["input"]].shape) == s["shape"] for s in states)
                    audio.append(values[0])
                    if full_trace: trace.append(values[1:])
                    pos += size
                assert pos == z.shape[-1]
            finally:
                ok(shim.sk_set_mode(0))
            after = counters(); calls = after["intercepted"] - before["intercepted"]
            assert calls > 0 and after["unsupported"] == before["unsupported"] == 0
            assert after["captures"] == 0 and after["resolved"] == 1
            intercepted[mode] += calls
            return {"audio": audio, "states": trace if full_trace else [values[1:]], "full_trace": full_trace}, {"session_ns": elapsed, "packets": len(chunks), "output_samples": pos * 1920, "duration": pos / 25, "rtf": elapsed / 1e9 / (pos / 25), "intercepted_calls": calls}

        def compare(reference, candidate, accumulate=True):
            assert reference["full_trace"] == candidate["full_trace"]
            assert len(reference["audio"]) == len(candidate["audio"])
            assert len(reference["states"]) == len(candidate["states"])
            audio = Metrics(); history_metrics = Metrics()
            for rv, cv in zip(reference["audio"], candidate["audio"]):
                audio.add(rv, cv)
                if accumulate: all_wave.add(rv, cv)
            for rv, cv in zip(reference["states"], candidate["states"]):
                assert len(rv) == len(cv) == 18
                for s, x, y in zip(states, rv, cv):
                    history_metrics.add(x, y)
                    if accumulate: all_states[s["input"]].add(x, y)
            return {"waveform": audio.summary(), "all_states": history_metrics.summary(), "packets_compared": len(reference["audio"]), "state_checks": 18 * len(reference["states"]), "state_scope": "every_packet" if reference["full_trace"] else "final_packet"}

        chunks = [args.packet // 40] * (40 // (args.packet // 40))
        report["status"] = "warming"; save()
        report["warmup_comparisons"] = []
        for uid, z in zs.items():
            warm = {mode: run(mode, z, chunks, full_trace=True) for mode in MODES}
            report["warmup_comparisons"].append({"uid": uid, **compare(warm[0][0], warm[2][0])})
            del warm
        report["status"] = "timing"; save()
        for rep in range(3):
            for index, (uid, z) in enumerate(zs.items()):
                order = MODES if (rep + index) % 2 == 0 else tuple(reversed(MODES))
                report["current"] = {"stage": "timing", "uid": uid, "rep": rep, "order": order}; save()
                results = {mode: run(mode, z, chunks) for mode in order}
                row = {"uid": uid, "rep": rep, "order": order, "baseline": results[0][1], "candidate": results[2][1]}
                row["comparison"] = compare(results[0][0], results[2][0])
                row["time_reduction_percent"] = 100 * (1 - row["candidate"]["session_ns"] / row["baseline"]["session_ns"])
                report["pairs"].append(row); save()
                del results
        report["status"] = "gates"; save()
        cases = {**{uid: z[..., :5] for uid, z in zs.items()}, "zero": np.zeros((1, 64, 5), np.float32), "tiny": zs[UIDS[0]][..., :5] * np.float32(1e-5)}
        for uid, z in cases.items():
            for partition in ([1, 2, 2], [2, 1, 2], [1] * 5, [5]):
                report["current"] = {"stage": "gates", "case": uid, "chunks": partition}; save()
                base, br = run(0, z, partition, full_trace=True); candidate, cr = run(2, z, partition, full_trace=True)
                assert br["output_samples"] == cr["output_samples"] == 9600
                report["gates"].append({"case": uid, "chunks": partition, **compare(base, candidate)})
                del base, candidate
        sys.path.insert(0, str(args.runtime_source))
        from fast_audiovae.streaming import StreamingDecoder
        pins[str(args.runtime_source / "fast_audiovae/streaming.py")] = sha(args.runtime_source / "fast_audiovae/streaming.py")
        class Counter:
            def __init__(self, mode): self.mode = mode; self.calls = 0
            def __getattr__(self, name): return getattr(session, name)
            def run(self, output_names, feed):
                self.calls += 1
                return native_call(self.mode, output_names, feed)[0]
        report["public_api"] = {}
        for mode in MODES:
            ok(shim.sk_set_mode(mode)); wrapped = Counter(mode); stream = StreamingDecoder(wrapped, entry).streaming_decode()
            try:
                empty = np.empty((1, 64, 0), np.float32)
                assert stream.decode_chunk(empty).shape == (1, 1, 0) and wrapped.calls == 0
                first = stream.decode_chunk(zs[UIDS[0]][..., :1])
                saved = {k: v.copy() for k, v in stream._history.items()}; count = wrapped.calls
                assert stream.decode_chunk(empty).shape == (1, 1, 0) and wrapped.calls == count and stream.frames_decoded == 1
                assert all(np.array_equal(v.view(np.uint32), stream._history[k].view(np.uint32)) for k, v in saved.items())
                stream.reset(); assert stream.frames_decoded == 0 and all(not np.any(v) for v in stream._history.values())
                replay = stream.decode_chunk(zs[UIDS[0]][..., :1])
                assert np.array_equal(first.view(np.uint32), replay.view(np.uint32))
                assert all(np.array_equal(v.view(np.uint32), stream._history[k].view(np.uint32)) for k, v in saved.items())
                report["public_api"][str(mode)] = {"empty_before_after_no_native_call": True, "empty_preserves_all_states_and_frame_count": True, "reset_clears_all_states": True, "reset_replay_all_states_and_audio_exact": True}
            finally:
                stream.close(); ok(shim.sk_set_mode(0))
        summary = counters(); assert summary["unsupported"] == 0 and summary["resolved"] == 1
        report["interception"] = summary
        report["timing_summary"] = {}
        for label in ("baseline", "candidate"):
            rows = [r[label] for r in report["pairs"]]
            report["timing_summary"][label] = {"pooled_rtf": sum(r["session_ns"] for r in rows) / 1e9 / sum(r["duration"] for r in rows), "median_prefix_rtf": statistics.median(r["rtf"] for r in rows), "packets": sum(r["packets"] for r in rows)}
        report["timing_summary"]["time_reduction_percent"] = 100 * (1 - report["timing_summary"]["candidate"]["pooled_rtf"] / report["timing_summary"]["baseline"]["pooled_rtf"])
        report["timing_summary"]["improved_pairs"] = sum(r["time_reduction_percent"] > 0 for r in report["pairs"])
        report["comparison"] = {"waveform": all_wave.summary(), "states": {k: v.summary() for k, v in all_states.items()}}
        for path, expected in pins.items(): assert sha(path) == expected, path
        assert "torch" not in sys.modules
        report["status"] = "complete"; report.pop("current", None)
    except BaseException as error:
        report["status"] = "failed"; report["error"] = repr(error); raise
    finally:
        report["native_ns_per_mode"] = native_ns; report["native_calls_per_mode"] = native_calls
        report["intercepted_calls_in_prefix_runs_per_mode"] = intercepted
        save()
    print(json.dumps({"status": report["status"], "packet_ms": args.packet, "timing": report["timing_summary"], "waveform": report["comparison"]["waveform"], "public_api": report["public_api"], "interception": report["interception"], "native_seconds_per_mode": {k: v / 1e9 for k, v in native_ns.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
