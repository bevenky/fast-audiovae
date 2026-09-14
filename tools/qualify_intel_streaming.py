"""Bounded installed-wheel Intel qualification; aggregate metadata only.

Run with the target virtualenv's Python and -I, outside the checkout. Candidate
and reference live in separate resident processes to prevent library SONAME
interposition. The candidate uses public load() with CPU/streaming/thread defaults.
An optional reference config has graph/graph_sha256, ordered libraries containing
path/sha256 records, and optional extra_artifacts records (for external weights).
Its state specification still comes from --reference-bundle/bundle.json.

Both packet sizes use three 1.6-second prefixes, one warmup and three repetitions.
The 30-second combined Session.run budget cannot interrupt an executing call.
No audio, latents or histories leave their worker processes or enter reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time
import traceback

LATENTS_SHA = "9b709b787c040591de57e094a5ba5a4515cc74e256e8014530ff8c7bb54d2d24"
UIDS = ("bn_in_00151_1818", "en_us_00103_1779", "es_419_00060_1994")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")
NATIVE_NAMES = ("fast_audiovae", "precision", "stage_pipeline", "intel_phase",
                "rawhistory", "paired_projection", "sleef", "xsmm", "libdnnl", "libmkl", "intel_stream")
BUDGET_SECONDS = 30.0


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu_environment():
    require(not os.environ.get("LD_PRELOAD"), "LD_PRELOAD must be empty; Snake interception is excluded")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1",
                      ROCR_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
                      MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE")
    require(hasattr(os, "sched_setaffinity") and 0 in os.sched_getaffinity(0),
            "Qualification requires Linux CPU0 in the process affinity")
    require("GenuineIntel" in Path("/proc/cpuinfo").read_text(), "Intel CPU required")
    os.sched_setaffinity(0, {0})


def verify_records(records):
    require(isinstance(records, list), "Expected an artifact inventory")
    for record in records:
        require(set(record) == {"path", "sha256"} and sha(record["path"]) == record["sha256"],
                "Artifact differs from its pin: " + str(record.get("path")))
    return [record["path"] for record in records]


def verify_external_weights(graph, records):
    import onnx
    graph = Path(graph).resolve()
    paths = {Path(record["path"]).resolve() for record in records}
    model = onnx.load(graph, load_external_data=False)
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = {item.key: item.value for item in tensor.external_data}.get("location")
            require(location and (graph.parent / location).resolve() in paths,
                    "External weight file lacks an explicit reference pin")


def installed_package():
    import importlib.abc
    import importlib.metadata
    import sysconfig

    class NoGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch.") or fullname in (
                    "fast_audiovae.gpu", "fast_audiovae.mps_decoder", "fast_audiovae.mps_compiled"):
                raise RuntimeError("CPU qualification attempted a GPU/Torch import: " + fullname)
    sys.meta_path.insert(0, NoGPU())
    import fast_audiovae
    origin = Path(fast_audiovae.__file__).resolve()
    installed_roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    require(any(origin.is_relative_to(root) for root in installed_roots),
            "Use the installed wheel, not an imported checkout: " + str(origin))
    distribution = importlib.metadata.distribution("fast-audiovae")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable"), "Editable installs are not qualified")
    return fast_audiovae, {"path": str(origin), "version": distribution.version,
                           "module_sha256": sha(origin), "editable": False}


def mapped_native(bundle=None, required=()):
    required = {Path(path).resolve() for path in required}
    names = {path.name for path in required}
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        lower = Path(name).name.lower()
        require(not any(token in lower for token in ("libcuda", "libcudnn", "libcublas", "libhip", "librocblas")),
                "GPU dependency loaded during CPU qualification")
        if ".so" in name and (Path(name).name in names or any(token in lower for token in NATIVE_NAMES)):
            require(not name.endswith(" (deleted)"), "Loaded native file was deleted")
            paths.add(Path(name).resolve())
    missing = required - paths
    require(not missing, "A registered native library is absent from process maps; missing="
            + json.dumps(sorted(str(path) for path in missing)) + "; mapped="
            + json.dumps(sorted(str(path) for path in paths)))
    require(any(path.name.startswith("libdnnl.so") for path in paths), "oneDNN not loaded")
    require(any(path.name.startswith("libmkl_") for path in paths), "oneMKL fallback not loaded")
    if bundle is not None:
        require(all(path.is_relative_to(bundle) for path in paths),
                "Candidate loaded a codec dependency outside its bundle: " + ", ".join(
                    str(path) for path in paths if not path.is_relative_to(bundle)))
    return [{"path": str(path), "sha256": sha(path)} for path in sorted(paths)]


class Meter:
    def __init__(self, inner):
        self.inner, self.seconds, self.calls, self.command_limit = inner, 0.0, 0, 0.0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def run(self, *args, **kwargs):
        require(self.seconds < self.command_limit, "Combined native budget exhausted")
        start = time.perf_counter()
        try:
            return self.inner.run(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start
            self.calls += 1
            require(self.seconds <= self.command_limit, "Combined native budget exceeded after call")


def stream_specification(bundle):
    manifest = json.loads((bundle / "bundle.json").read_text())
    native = manifest["native"]["Linux/x86_64"]
    spec = manifest["streaming"]["models"][native["model"]]
    require(len(spec["states"]) == 18 and spec["state_bytes"] == 683264, "Expected accepted 18-state inventory")
    return manifest, native, spec


def worker(args):
    cpu_environment()
    package, package_info = installed_package()
    import contextlib
    import ctypes
    import gc
    import numpy as np
    import onnxruntime as ort
    from fast_audiovae.runtime import load_streaming_decoder
    from fast_audiovae.streaming import StreamingDecoder
    require(ort.__version__ == "1.29.0", "Intel selected recipe is qualified on ORT1.29.0")
    require(sha(args.latents) == LATENTS_SHA, "Latent archive pin mismatch")
    _, _, reference_spec = stream_specification(args.reference_bundle)
    bundle = None
    with contextlib.redirect_stdout(sys.stderr):
        if args.worker == "reference":
            if args.reference_config:
                cfg = json.loads(args.reference_config.read_text())
                require(sha(cfg["graph"]) == cfg["graph_sha256"], "Reference graph pin mismatch")
                required_libraries = verify_records(cfg["libraries"])
                verify_records(cfg.get("extra_artifacts", []))
                verify_external_weights(cfg["graph"], cfg.get("extra_artifacts", []))
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
                    options.add_session_config_entry(key, "0")
                for path in required_libraries:
                    options.register_custom_ops_library(path)
                session = ort.InferenceSession(cfg["graph"], options, providers=["CPUExecutionProvider"])
                session.disable_fallback()
                decoder = StreamingDecoder(session, reference_spec)
                info = {"selected": "pinned_reference", "model_sha256": cfg["graph_sha256"]}
            else:
                decoder, info = load_streaming_decoder(args.reference_bundle, threads=1)
                require(info["selected"] == "native", "Reference silently selected portable ONNX")
                _, native, spec = stream_specification(args.reference_bundle)
                required_libraries = [args.reference_bundle / native["library"], *[
                    args.reference_bundle / item["library"] for item in spec.get(
                        "additional_libraries", native.get("additional_libraries", []))]]
            cache_check = None
        else:
            if not args.reuse_cache:
                require(not list((args.cache / "receipts").glob("*.json")), "Use a fresh candidate cache or explicitly pass --reuse-cache")
            original_popen = subprocess.Popen
            def no_compiler(command, *positional, **keywords):
                words = command if isinstance(command, (list, tuple)) else [command]
                require(Path(str(words[0])).name not in ("gcc", "g++", "cc", "c++", "clang", "clang++", "cmake", "make", "ninja"),
                        "Installed-wheel load attempted native compilation")
                return original_popen(command, *positional, **keywords)
            subprocess.Popen = no_compiler
            stdin_before = os.fstat(sys.stdin.fileno())
            public = package.load(source=args.source, cache_dir=args.cache, offline=True)
            first_info = dict(public.info)
            require(first_info["recipe"] == "intel_stream_selected" and first_info["selected"] == "native"
                    and first_info["threads"] == 1 and public.mode == "streaming",
                    "Public defaults failed selected Intel contract")
            if not args.reuse_cache:
                require(not first_info["cache_hit"], "Fresh-cache qualification unexpectedly reused a cache")
            key = first_info["cache_key"]
            receipt = json.loads((args.cache / "receipts" / (key + ".json")).read_text())
            bundle = (args.cache / receipt["bundle"]).resolve()
            require(bundle.is_relative_to(args.cache), "Prepared bundle escaped cache")
            del public
            gc.collect()
            stdin_after = os.fstat(sys.stdin.fileno())
            require((stdin_after.st_dev, stdin_after.st_ino) == (stdin_before.st_dev, stdin_before.st_ino),
                    "Destroying the first public session replaced its input descriptor")
            public = package.load(source=args.source, cache_dir=args.cache, offline=True)
            require(public.info["cache_hit"] and public.info["cache_key"] == key
                    and public.info["recipe"] == "intel_stream_selected", "Installed-wheel cache reuse failed")
            decoder, info = public._decoder, dict(public.info)
            cache_check = {"first_cache_hit": bool(first_info["cache_hit"]), "second_cache_hit": True, "key": key,
                           "reuse_cache_requested": args.reuse_cache,
                           "defaults": {"device": "cpu", "mode": "streaming", "threads": 1}}
            save(args.output / "candidate-cache-check.json", {"cache_check": cache_check,
                 "bundle": str(bundle), "package": package_info, "recipe": public.info["recipe"]})
            _, native, spec = stream_specification(bundle)
            require(spec.get("intel_stream_selected"), "Selected Intel graph metadata missing")
            required_libraries = [bundle / native["library"], *[
                bundle / item["library"] for item in spec.get("additional_libraries", native.get("additional_libraries", []))]]
    session = decoder._session
    require(session.get_providers() == ["CPUExecutionProvider"], "CPU provider only")
    options = session.get_session_options()
    require(options.intra_op_num_threads == options.inter_op_num_threads == 1
            and options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL, "One sequential ORT thread required")
    require(len(decoder._states) == 18 and decoder.state_bytes == 683264, "Public state inventory changed")
    maps = mapped_native(bundle, required_libraries)
    mkl_path = next(item["path"] for item in maps if Path(item["path"]).name.startswith("libmkl_intel_lp64.so"))
    mkl = ctypes.CDLL(mkl_path, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    mkl.mkl_get_max_threads.argtypes = []; mkl.mkl_get_max_threads.restype = ctypes.c_int
    require(mkl.mkl_get_max_threads() == 1, "oneMKL internal thread count is not one")
    meter = Meter(session); decoder._session = meter
    zs = {}
    with np.load(args.latents, allow_pickle=False) as archive:
        for uid in UIDS:
            raw = archive[uid + "__z"]
            require(raw.dtype == np.float32 and raw.shape[:2] == (1, 64) and raw.shape[-1] >= 40
                    and np.isfinite(raw).all(), "Invalid frozen latent input")
            zs[uid] = np.ascontiguousarray(raw[..., :40])

    def digest(value):
        require(value.dtype == np.float32 and np.isfinite(value).all(), "Invalid output/state values")
        return {"shape": list(value.shape), "dtype": "float32", "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}

    def history(stream):
        require(len(stream._history) == 18, "Incomplete public history")
        return {name: digest(value) for name, value in sorted(stream._history.items())}

    def sequence(z, chunks, stream=None, trace=True):
        stream = stream if stream is not None else decoder.streaming_decode()
        require(sum(chunks) == z.shape[-1], "Invalid chunk partition")
        before, initial, pos, audio, packets = meter.seconds, stream.frames_decoded, 0, [], []
        for size in chunks:
            x = np.ascontiguousarray(z[..., pos:pos + size])
            y = stream.decode_chunk(x)
            require(y.shape == (1, 1, size * 1920), "Missing output samples")
            record = {"frames": size, "samples": size * 1920}
            if trace:
                record.update(audio=digest(y), state=history(stream))
            audio.append(y); packets.append(record); pos += size
        elapsed = meter.seconds - before
        require(stream.frames_decoded == initial + pos, "Frame counter mismatch")
        return {"audio": digest(np.concatenate(audio, axis=-1)), "state": history(stream),
                "packets": packets, "samples": pos * 1920, "seconds": elapsed,
                "duration_seconds": pos / 25, "rtf": elapsed / (pos / 25)}, stream

    def content(result):
        return {name: result[name] for name in ("audio", "state", "packets", "samples")}

    def gates():
        cases, checks = {}, []
        for uid in UIDS:
            for partition in ([1, 2, 2], [2, 1, 2], [3, 2], [5]):
                result, _ = sequence(zs[uid][..., :5], partition)
                cases[uid + "/" + "-".join(map(str, partition))] = content(result)
            standard = cases[uid + "/1-2-2"]
            for suffix in ("2-1-2", "3-2", "5"):
                other = cases[uid + "/" + suffix]
                checks.append({"name": "partition_invariance/" + uid + "/" + suffix,
                               "pass": other["audio"] == standard["audio"]
                               and other["state"] == standard["state"]})
        for name, z in (("zero", np.zeros((1, 64, 5), np.float32)),
                        ("tiny", zs[UIDS[0]][..., :5] * np.float32(1e-5))):
            result, _ = sequence(z, [1, 2, 2]); cases[name] = content(result)
        z = zs[UIDS[0]][..., :5]; stream = decoder.streaming_decode()
        empty = np.empty((1, 64, 0), np.float32)
        def empty_gate(name):
            state, calls, frames = history(stream), meter.calls, stream.frames_decoded
            y = stream.decode_chunk(empty); tail = stream.flush()
            checks.append({"name": name, "pass": y.shape == tail.shape == (1, 1, 0)
                           and history(stream) == state and meter.calls == calls and stream.frames_decoded == frames})
        empty_gate("empty_fresh")
        original, stream = sequence(z, [1, 2, 2], stream); empty_gate("empty_and_flush_after_audio")
        stream.reset()
        checks.append({"name": "reset_clears_state_and_counter", "pass": stream.frames_decoded == 0
                       and all(not value.view(np.uint32).any() for value in stream._history.values())})
        replay, _ = sequence(z, [1, 2, 2], stream)
        checks.append({"name": "reset_replay", "pass": content(original) == content(replay)})
        changed = z.copy(); changed[..., 3:] *= np.float32(-1)
        future, _ = sequence(changed, [1, 2, 2])
        checks.append({"name": "future_prefix_invariance", "pass": future["packets"][:2] == original["packets"][:2]})
        independent = decoder.streaming_decode(); old_frames = stream.frames_decoded
        a, _ = sequence(z[..., :1], [1], independent)
        sequence(z[..., :2], [2], stream)
        b, _ = sequence(z[..., 1:], [2, 2], independent)
        checks.append({"name": "interleaved_independent_streams", "pass": a["packets"] + b["packets"] == original["packets"]})
        checks.append({"name": "independent_stream_objects", "pass": stream._history is not independent._history
                       and stream.frames_decoded == old_frames + 2 and independent.frames_decoded == 5})
        cases.update(reset_replay=content(replay), future_mutation=content(future))
        require(all(check["pass"] for check in checks), "Public stream behavior failed")
        return {"cases": cases, "checks": checks, "native_maps": mapped_native(bundle, required_libraries)}

    contract = {kind: [{"name": value.name, "shape": list(value.shape), "type": value.type}
                       for value in getattr(session, "get_" + kind)()] for kind in ("inputs", "outputs")}
    print(json.dumps({"ready": True, "role": args.worker, "ort": ort.__version__, "package": package_info,
                      "pid": os.getpid(), "contract": contract, "info": info, "cache_check": cache_check,
                      "native_maps": maps, "bundle": str(bundle) if bundle else str(args.reference_bundle),
                      "affinity": sorted(os.sched_getaffinity(0)), "mkl_threads": 1}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["command"] == "quit":
            break
        before, calls = meter.seconds, meter.calls
        meter.command_limit = before + request["remaining_seconds"]
        try:
            if request["command"] == "gates":
                result = gates()
            elif request["command"] == "stream":
                size = request["packet_ms"] // 40
                require(size in (1, 2), "Only 40/80ms timing is allowed")
                result, _ = sequence(zs[request["uid"]], [size] * (40 // size), trace=request["trace"])
            else:
                raise ValueError("Unknown command")
            require("torch" not in sys.modules, "Torch imported during CPU qualification")
            answer = {"ok": True, "result": result}
        except Exception as error:
            answer = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
        answer.update(native_seconds=meter.seconds - before, native_calls=meter.calls - calls)
        print(json.dumps(answer, allow_nan=False), flush=True)


def compare(reference, candidate, trace):
    require(reference["audio"] == candidate["audio"], "Waveform is not bitwise identical")
    require(reference["state"] == candidate["state"] and len(reference["state"]) == 18,
            "Final state is not bitwise identical")
    require(reference["packets"] == candidate["packets"] and reference["samples"] == candidate["samples"],
            "Packet output or intermediate state mismatch")
    count = len(reference["packets"])
    return {"pass": True, "packets": count, "state_arrays": 18 * (count if trace else 1),
            "waveform_scope": "each_packet_and_full" if trace else "full_prefix",
            "state_scope": "every_packet" if trace else "final_only"}


def main(args):
    cpu_environment()
    require(not args.output.exists(), "Use a fresh output directory")
    require(not (Path.cwd() / "src/fast_audiovae").exists(), "Run qualification outside the source checkout")
    args.output.mkdir(parents=True)
    report = {"status": "running", "cpu_only": True, "threads": 1, "packet_ms": [40, 80],
              "native_budget_seconds": BUDGET_SECONDS, "native_seconds": 0.0, "native_calls": 0,
              "script_sha256": sha(__file__), "source_sha256": sha(args.source), "latents_sha256": sha(args.latents),
              "reuse_cache_requested": args.reuse_cache,
              "reference_manifest_sha256": sha(args.reference_bundle / "bundle.json"),
              "reference_config": json.loads(args.reference_config.read_text()) if args.reference_config else None,
              "arms": {}, "rows": [], "checks": [],
              "timing_boundary": "Completed Session.run within public streaming calls; setup, Python validation, hashing and IPC excluded",
              "scope": "Installed-wheel parity and short paired latency, not new perceptual scoring or broad hardware qualification"}
    children, logs = {}, []

    def read(process, timeout=60):
        require(select.select([process.stdout], [], [], timeout)[0], "Worker response timeout; inspect stderr")
        line = process.stdout.readline()
        require(line, "Worker exited; inspect stderr log")
        return json.loads(line)

    def command(role, payload):
        remaining = BUDGET_SECONDS - report["native_seconds"]
        require(remaining > 0, "Combined native budget exhausted")
        process = children[role]
        process.stdin.write(json.dumps({**payload, "remaining_seconds": remaining}) + "\n"); process.stdin.flush()
        answer = read(process)
        report["native_seconds"] += answer["native_seconds"]; report["native_calls"] += answer["native_calls"]
        save(args.output / "result.json", report)
        require(answer["ok"], answer.get("error", "Worker failed"))
        return answer["result"]

    try:
        # Prepare the candidate before retaining the reference session in RAM.
        # Timing still alternates the two already resident processes below.
        for role in ("candidate", "reference"):
            log = (args.output / (role + ".stderr.log")).open("w"); logs.append(log)
            command_line = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", role,
                            "--source", str(args.source), "--cache", str(args.cache),
                            "--reference-bundle", str(args.reference_bundle), "--latents", str(args.latents),
                            "--output", str(args.output)]
            if args.reference_config:
                command_line += ["--reference-config", str(args.reference_config)]
            if args.reuse_cache:
                command_line.append("--reuse-cache")
            children[role] = subprocess.Popen(command_line, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=log, text=True, bufsize=1)
            # First-load graph preparation can take longer than inference; it
            # remains outside the explicitly bounded Session.run workload.
            report["arms"][role] = read(children[role], timeout=600)
            require(report["arms"][role].get("ready"), "Worker initialization failed")
            save(args.output / "result.json", report)
        ref, actual = report["arms"]["reference"], report["arms"]["candidate"]
        require(ref["ort"] == actual["ort"] and ref["contract"] == actual["contract"], "Runtime/state contract mismatch")
        results = {role: command(role, {"command": "gates"}) for role in children}
        report["gates"] = results
        require(set(results["reference"]["cases"]) == set(results["candidate"]["cases"]), "Gate inventory mismatch")
        for name, value in results["reference"]["cases"].items():
            report["checks"].append({"case": name, **compare(value, results["candidate"]["cases"][name], True)})
        for packet_ms in (40, 80):
            for phase, count in (("warmup", 1), ("measured", 3)):
                for repeat in range(count):
                    for index, uid in enumerate(UIDS):
                        order = ["reference", "candidate"] if (repeat + index) % 2 == 0 else ["candidate", "reference"]
                        pair = {role: command(role, {"command": "stream", "uid": uid, "packet_ms": packet_ms,
                                                    "trace": phase == "warmup"}) for role in order}
                        check = compare(pair["reference"], pair["candidate"], phase == "warmup")
                        report["rows"].append({"phase": phase, "repeat": repeat, "uid": uid, "packet_ms": packet_ms,
                                               "order": order, "exact": check, **pair})
                        save(args.output / "result.json", report)
        report["summary"] = {}
        for packet_ms in (40, 80):
            rows = [row for row in report["rows"] if row["phase"] == "measured" and row["packet_ms"] == packet_ms]
            summary = {role: {"pooled_rtf": sum(row[role]["seconds"] for row in rows) / (len(rows) * 1.6),
                              "median_clip_rtf": statistics.median(row[role]["rtf"] for row in rows)} for role in children}
            summary.update(paired_rounds=len(rows), candidate_faster_rounds=sum(
                row["candidate"]["seconds"] < row["reference"]["seconds"] for row in rows),
                time_reduction_percent=100 * (1 - summary["candidate"]["pooled_rtf"] / summary["reference"]["pooled_rtf"]))
            report["summary"][str(packet_ms)] = summary
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    finally:
        save(args.output / "result.json", report)
        for process in children.values():
            if process.poll() is None:
                try:
                    process.stdin.write('{"command":"quit"}\n'); process.stdin.flush(); process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill(); process.wait()
        for log in logs:
            log.close()
    print(json.dumps({"status": report["status"], "result": str(args.output / "result.json"),
                      "native_seconds": report["native_seconds"], "summary": report.get("summary")}, allow_nan=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "reference-bundle", "latents", "output"):
        parser.add_argument("--" + name, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--reference-config", type=lambda value: Path(value).resolve())
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Allow an existing verified candidate cache and record its actual first-load cache hit")
    parser.add_argument("--worker", choices=("reference", "candidate"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(main(args))
