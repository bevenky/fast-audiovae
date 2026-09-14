"""Bounded operator qualification against the previously qualified INT8 bridge.

Uses two saved projection matrices, never a complete codec or audio decoder.
The experimental library is a test oracle only and is not shipped or linked.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

DOMAIN = "fast.audiovae.apple.firstpair.int8.v1"
OLD_DOMAIN = "fast.audiovae.apple.int8.screen.v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def session(library, weight, old=False, **attrs):
    domain = OLD_DOMAIN if old else DOMAIN
    values = {"k": 2048, "n": 8192, "max_m": 2, "matrix_abi": 1, "threads": 1}
    if old:
        values["backend"] = 1
    values.update(attrs)
    node = helper.make_node("FirstPairInt8", ["x", "w"], ["y"], domain=domain, **values)
    graph = helper.make_graph([node], "projection", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", 2048, "T"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", 8192, "T"])], [numpy_helper.from_array(weight, "w")])
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid(domain, 1)])
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.register_custom_ops_library(str(library))
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--experimental-library", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(3)
    fixture = np.load(args.fixture)
    rng = np.random.default_rng(9017)
    weights = [fixture[k] for k in fixture.files if fixture[k].shape == (8192, 2048)]
    real_inputs = [fixture[k] for k in fixture.files if fixture[k].shape in ((1, 2048, 2), (2048, 2))]
    assert len(weights) == 2 and real_inputs
    rows = []
    rejected = 0
    helper_source = args.output.parent / "fpcr-validation.c"
    helper_library = args.output.parent / "libfpcr_validation.dylib"
    helper_source.write_text('#include <stdint.h>\nuint64_t get_fpcr(void){uint64_t x;__asm__ volatile("mrs %0, fpcr":"=r"(x));return x;}\nvoid set_fpcr(uint64_t x){__asm__ volatile("msr fpcr, %0\\n\\tisb"::"r"(x):"memory");}\n')
    subprocess.run(["xcrun", "clang", "-dynamiclib", str(helper_source), "-o", str(helper_library)], check=True)
    fpcr = ctypes.CDLL(str(helper_library.resolve()))
    fpcr.get_fpcr.argtypes = []; fpcr.get_fpcr.restype = ctypes.c_uint64
    fpcr.set_fpcr.argtypes = [ctypes.c_uint64]; fpcr.set_fpcr.restype = None
    for index, weight in enumerate(weights):
        shipping = session(args.library, weight)
        experimental = session(args.experimental_library, weight, old=True)
        assert shipping.get_providers() == ["CPUExecutionProvider"]
        run = lambda x: shipping.run(None, {"x": np.ascontiguousarray(x)})[0]
        ref = lambda x: experimental.run(None, {"x": np.ascontiguousarray(x)})[0]
        for t in (1, 2):
            samples = [np.asarray(real_inputs[0]).reshape(1, 2048, 2)[:, :, :t],
                       rng.normal(size=(1, 2048, t)).astype(np.float32),
                       np.zeros((1, 2048, t), np.float32),
                       (rng.normal(size=(1, 2048, t)) * 1e-8).astype(np.float32)]
            for x in samples:
                assert exact(run(x), ref(x))
                rows.append({"projection": index, "frames": t, "check": "qualified_bridge_exact"})
        for batch, t in ((1, 3), (1, 7), (1, 24), (2, 5)):
            x = rng.normal(size=(batch, 2048, t)).astype(np.float32)
            y = run(x)
            reference = np.concatenate([ref(x[:, :, start:start + 2]) for start in range(0, t, 2)], axis=2)
            assert exact(y, reference)
            singles = np.concatenate([ref(x[:, :, start:start + 1]) for start in range(t)], axis=2)
            assert exact(y, singles)
            changed = x.copy(); changed[:, :, 1:] += np.float32(0.1)
            assert exact(run(changed)[:, :, :1], y[:, :, :1])
            rows.append({"projection": index, "frames": t, "batch": batch, "check": "pair_single_and_prefix_exact"})
        for shape in ((1, 2048, 0), (0, 2048, 2), (0, 2048, 0)):
            assert run(np.zeros(shape, np.float32)).shape == (shape[0], 8192, shape[2])
            rows.append({"projection": index, "check": "empty_shape", "shape": shape})
        x = rng.normal(size=(1, 2048, 7)).astype(np.float32)
        expected = run(x)
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(exact(y, expected) for y in pool.map(lambda _: run(x), range(16)))
        rows.append({"projection": index, "check": "same_session_concurrency", "calls": 16})
        for bad in (float("nan"), float("inf"), -float("inf")):
            x = np.zeros((1, 2048, 3), np.float32); x[0, 17, 2] = bad
            try:
                run(x)
            except Exception as exc:
                assert "Activation quantization failed" in str(exc)
                rejected += 1
            else:
                raise AssertionError("Nonfinite input accepted")
        if index == 0:
            x = np.zeros((1, 2048, 1), np.float32)
            saved = fpcr.get_fpcr()
            assert saved & ((3 << 22) | (1 << 24) | 3) == 0
            for bit in (1 << 22, 1 << 24, 1, 2):
                failed = False
                try:
                    fpcr.set_fpcr(saved | bit)
                    try:
                        run(x)
                    except Exception:
                        failed = True
                finally:
                    fpcr.set_fpcr(saved)
                assert failed, "Modified FPCR accepted"
                rejected += 1
            assert fpcr.get_fpcr() == saved and np.isfinite(run(x)).all()
            rows.append({"projection": index, "check": "fpcr_rejection_and_restore", "rejections": 4})
            for attrs in ({"threads": 4}, {"matrix_abi": 2}, {"k": 1024}, {"max_m": 4}):
                try:
                    session(args.library, weight, **attrs)
                except Exception:
                    rejected += 1
                else:
                    raise AssertionError("Invalid attributes accepted")
        del shipping, experimental
    lib = ctypes.CDLL(str(args.library))
    counts = {}
    for name in ("calls", "packs", "quant_calls", "empty_calls"):
        func = getattr(lib, "av8_ort_" + name)
        func.argtypes = [ctypes.c_int]; func.restype = ctypes.c_uint64
        counts[name] = func(0)
    result = {"passed": True, "library": str(args.library), "library_sha256": sha(args.library),
        "oracle_sha256": sha(args.experimental_library), "fixture_sha256": sha(args.fixture),
        "checks": rows, "rejected_invalid_cases": rejected, "counters": counts,
        "cpu_only": True, "threads_per_operator": 1, "complete_decoder_executed": False,
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(rows), "rejected_invalid_cases": rejected, "output": str(args.output)}))


if __name__ == "__main__":
    main()
