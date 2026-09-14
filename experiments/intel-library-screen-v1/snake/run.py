"""Capture actual combined-baseline Snake calls and run bounded CPU microchecks."""
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


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args(); cfg = json.loads(a.build.read_text())
    assert cfg["status"] == "built"
    if not a.worker:
        assert not os.environ.get("LD_PRELOAD")
        env = dict(os.environ)
        env.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                        "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT", "NUMEXPR_NUM_THREADS")})
        env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1",
                   NVIDIA_VISIBLE_DEVICES="void", LD_PRELOAD=cfg["shim"], **cfg["resolvers"])
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--build",
                               str(a.build.resolve()), "--output", str(a.output.resolve())], env=env).returncode
    os.sched_setaffinity(0, {0})
    assert "GenuineIntel" in Path("/proc/cpuinfo").read_text()
    assert os.environ["LD_PRELOAD"] == cfg["shim"]
    for path, expected in cfg["pins"].items(): assert sha(path) == expected, path
    import numpy as np
    import onnxruntime as ort
    assert ort.__version__ == cfg["ort"] == "1.29.0"
    probe = ctypes.CDLL(cfg["shim"], mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    probe.sk_error.argtypes = []; probe.sk_error.restype = ctypes.c_char_p
    for name in ("sk_initialize", "sk_clear"):
        getattr(probe, name).argtypes = []; getattr(probe, name).restype = ctypes.c_int
    probe.sk_set_mode.argtypes = [ctypes.c_int]; probe.sk_set_mode.restype = ctypes.c_int
    probe.sk_summary.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]; probe.sk_summary.restype = ctypes.c_int
    probe.sk_info.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]; probe.sk_info.restype = ctypes.c_int
    probe.sk_compare.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]; probe.sk_compare.restype = ctypes.c_int
    probe.sk_measure.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]; probe.sk_measure.restype = ctypes.c_int
    def ok(value):
        if value != 0: raise RuntimeError(probe.sk_error().decode())
    report = {"status": "preparing", "threads": 1, "cpu": 0, "cpu_only": True, "ort": ort.__version__,
              "graph": cfg["graph"], "source_sha256": cfg["pins"], "packet_ms": 80,
              "full_session_ns": 0, "micro_probe_wall_ns": 0, "samples": [],
              "scope": "One actual 80 ms packet after two warm packets from the accepted combined Intel baseline; captures stay in RAM. No audio, latent or feature values are exported.",
              "coverage": "Only dynamically linked ncc_snake_f32 calls inside fused stages; symbol-bound base/history kernels are unchanged.",
              "numeric_gate": "Finite and abs(error)<=1e-5+1e-4*abs(reference); bitwise differences are separately counted. This is not full-decoder quality qualification.",
              "timing_scope": "Seven alternating native micro-pairs per captured shape/coefficients. Timing is not decoder RTF.",
              "native_work_budget_seconds": 5}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    def budget(): assert report["full_session_ns"]+report["micro_probe_wall_ns"] < 5_000_000_000
    save()
    try:
        opts = ort.SessionOptions(); opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; opts.log_severity_level = 3
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        for library in cfg["libraries"]: opts.register_custom_ops_library(library)
        session = ort.InferenceSession(cfg["graph"], opts, providers=["CPUExecutionProvider"])
        session.disable_fallback(); assert session.get_providers() == ["CPUExecutionProvider"]
        ok(probe.sk_initialize()); ok(probe.sk_set_mode(0))
        mkl = ctypes.CDLL(cfg["resolvers"]["SK_MKL_LIBRARY"])
        mkl.mkl_get_max_threads.argtypes=[]; mkl.mkl_get_max_threads.restype=ctypes.c_int
        assert mkl.mkl_get_max_threads() == 1
        report["mkl_max_threads"] = 1
        state = {v.name: np.zeros(v.shape, np.float32) for v in session.get_inputs()[1:]}
        names = [v.name for v in session.get_outputs()]
        latent_name = session.get_inputs()[0].name
        with np.load(cfg["latents"], allow_pickle=False) as archive:
            z = np.ascontiguousarray(archive["bn_in_00151_1818__z"][..., :6])
        assert z.shape == (1,64,6)
        def call(feed):
            budget(); start=time.perf_counter_ns(); out=session.run(None, feed)
            report["full_session_ns"] += time.perf_counter_ns()-start; budget()
            assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in out)
            return out
        for index in range(2):
            feed = {latent_name: np.ascontiguousarray(z[...,2*index:2*index+2]), **state}
            output = call(feed); values=dict(zip(names,output))
            state = {key:values[key.removesuffix("_in")+"_out"] for key in state}
        feed = {latent_name:np.ascontiguousarray(z[...,4:6]), **state}
        original = call(feed)
        input_hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in feed.items()}
        ok(probe.sk_clear()); ok(probe.sk_set_mode(3)); report["status"]="capturing";save()
        try: captured = call(feed)
        finally: ok(probe.sk_set_mode(0))
        assert all(np.array_equal(x.view(np.uint32),y.view(np.uint32)) for x,y in zip(original,captured))
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in feed.items()} == input_hashes
        report["capture_preserves_complete_waveform_and_all_states"] = True
        summary=(ctypes.c_uint64*5)();ok(probe.sk_summary(summary,5))
        report["capture"] = dict(zip(("count","bytes","intercepted_calls","unsupported_calls","resolved"),map(int,summary)))
        assert summary[0]>0 and summary[4]==1
        report["status"]="microchecks";save()
        for index in range(summary[0]):
            info=(ctypes.c_int64*4)();ok(probe.sk_info(index,info,4))
            item={"index":index,"shape":list(info)[:3],"observed_calls":int(info[3]),"variants":{}}
            report["samples"].append(item)
            for mode,label in ((1,"vml_ha_tile256"),(2,"vml_ha_whole_call")):
                report["current"]={"index":index,"variant":label};save();budget()
                metrics=(ctypes.c_double*7)();start=time.perf_counter_ns();ok(probe.sk_compare(index,mode,metrics,7))
                report["micro_probe_wall_ns"]+=time.perf_counter_ns()-start
                row=dict(zip(("elements","bitwise_different","max_abs","max_relative","rms_error","tolerance_failures","nonfinite"),map(float,metrics)))
                samples=(ctypes.c_uint64*14)();start=time.perf_counter_ns();ok(probe.sk_measure(index,mode,7,samples,14))
                report["micro_probe_wall_ns"]+=time.perf_counter_ns()-start;budget()
                row["paired_ns"]=list(map(int,samples));row["reference_median_ns"]=statistics.median(samples[::2]);row["candidate_median_ns"]=statistics.median(samples[1::2])
                row["time_reduction_percent"]=100*(1-row["candidate_median_ns"]/row["reference_median_ns"])
                item["variants"][label]=row;save()
        report["weighted_summary"]={}
        for label in ("vml_ha_tile256","vml_ha_whole_call"):
            ref=sum(s["observed_calls"]*s["variants"][label]["reference_median_ns"] for s in report["samples"])
            cand=sum(s["observed_calls"]*s["variants"][label]["candidate_median_ns"] for s in report["samples"])
            report["weighted_summary"][label]={"reference_ns":ref,"candidate_ns":cand,"time_reduction_percent":100*(1-cand/ref),
                 "numeric_gate_passed":all(s["variants"][label]["tolerance_failures"]==s["variants"][label]["nonfinite"]==0 for s in report["samples"]),
                 "max_abs":max(s["variants"][label]["max_abs"] for s in report["samples"])}
        report["represented_calls"] = sum(s["observed_calls"] for s in report["samples"])
        report["all_supported_calls_represented"] = report["represented_calls"] == summary[2]-summary[3]
        assert report["all_supported_calls_represented"]
        for path,expected in cfg["pins"].items():assert sha(path)==expected,path
        assert "torch" not in sys.modules
        report["status"]="complete"
    except BaseException as error:
        report["status"]="failed";report["error"]=repr(error);raise
    finally: save()
    print(json.dumps({key:report[key] for key in ("status","capture","full_session_ns","micro_probe_wall_ns","weighted_summary")},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
