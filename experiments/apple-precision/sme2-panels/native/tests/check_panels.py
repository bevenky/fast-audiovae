#!/usr/bin/env python3
"""Exact native panel checks only. No inference timing or performance claims."""
import argparse
import concurrent.futures
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite native evidence"
    assert digest(args.library) == args.library_sha256
    lib = C.CDLL(str(args.library.resolve()))
    ptr = C.c_void_p
    f32 = C.POINTER(C.c_float)
    i8 = C.POINTER(C.c_int8)
    i32 = C.POINTER(C.c_int32)
    lib.ipc_capabilities.restype = C.c_int
    assert lib.ipc_capabilities() & 16, "Apple SME2 capability required"
    lib.ipc_last_error.restype = C.c_char_p
    specs = {
        "create": ([C.c_int, C.c_int, f32, C.c_int, C.c_int], ptr),
        "destroy_plan": ([ptr], None), "prepare": ([ptr, f32, C.c_int], ptr),
        "destroy_input": ([ptr], None),
        "run_rows": ([ptr, ptr, f32, C.c_int, C.c_int], C.c_int),
        "inspect_input": ([ptr, C.POINTER(i8), C.POINTER(f32), C.POINTER(i32)], C.c_int),
        "create_workspace": ([ptr, C.c_int], ptr), "destroy_workspace": ([ptr], None),
        "prepare_panel": ([ptr, f32, C.c_int, C.c_int, C.c_int], C.c_int),
        "run_panel": ([ptr, ptr, f32, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int),
        "copy_panel": ([ptr, i8, f32, i32, C.c_int], C.c_int),
        "workspace_bytes": ([ptr], C.c_size_t),
        "input_bytes": ([ptr], C.c_size_t), "row_step": ([ptr], C.c_int),
        "allocate_input": ([ptr, f32, C.c_int, C.c_int], ptr),
        "prepare_jobs": ([ptr], C.c_int), "prepare_job": ([ptr, C.c_int], C.c_int),
        "check_packed_input": ([ptr], C.c_int),
    }
    for name, (inputs, result) in specs.items():
        fn = getattr(lib, "ipc_" + name)
        fn.argtypes, fn.restype = inputs, result

    def fp(a):
        return a.ctypes.data_as(f32)

    def ok(result):
        assert result == 0, lib.ipc_last_error().decode()

    def same(a, b):
        assert a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8)), "Byte mismatch"

    def create(w, backend):
        p = lib.ipc_create(*w.shape, fp(w), 8, backend)
        assert p, lib.ipc_last_error().decode()
        return p

    def reference(w, x):
        p = create(w, 0)
        inp = lib.ipc_prepare(p, fp(x), x.shape[1])
        assert inp, lib.ipc_last_error().decode()
        try:
            y = np.empty((w.shape[0], x.shape[1]), np.float32)
            ok(lib.ipc_run_rows(p, inp, fp(y), 0, w.shape[0]))
            q, scales, sums = i8(), f32(), i32()
            ok(lib.ipc_inspect_input(inp, C.byref(q), C.byref(scales), C.byref(sums)))
            qs = np.ctypeslib.as_array(q, shape=(x.size,)).copy().reshape(x.shape) if x.size else np.empty(x.shape, np.int8)
            ss = np.ctypeslib.as_array(scales, shape=(x.shape[1],)).copy() if x.shape[1] else np.empty(0, np.float32)
            zs = np.ctypeslib.as_array(sums, shape=(x.shape[1],)).copy() if x.shape[1] else np.empty(0, np.int32)
            return y, qs, ss, zs
        finally:
            lib.ipc_destroy_input(inp)
            lib.ipc_destroy_plan(p)

    rng = np.random.default_rng(20260908)
    records = []
    shapes = [(17, k, n) for k in (1, 3, 4, 31, 32, 127) for n in (0, 1, 15, 16, 17, 63, 64, 65)]
    shapes += [(m, k, n) for m, k in ((32, 128), (65, 256)) for n in (127, 128, 129, 511, 512, 513, 1023, 1024, 1031)]
    for index, (m, k, n) in enumerate(shapes):
        w = rng.uniform(-2, 2, (m, k)).astype(np.float32)
        source = rng.uniform(-3, 3, (k, n + 11)).astype(np.float32)
        source[:, :5] = np.nan
        source[:, n + 5:] = np.inf
        if k >= 3 and n:
            # Scale exactly1, full-range S8, half-integer ties and signed zero.
            source[0, 5:n + 5] = 127
            source[1, 5:n + 5] = np.resize(np.array([-127, -126.5, -2.5, -.5, -0., 0., .5, 2.5, 126.5, 127], np.float32), n)
        if index % 4 == 0 and n:
            source[:, 5] = np.nextafter(np.float32(0), np.float32(1))
            w[0] = np.nextafter(np.float32(0), np.float32(1))
        x = source[:, 5:n + 5].copy()
        y_ref, q_ref, scales_ref, sums_ref = reference(w, x)
        p = create(w, 3)
        ws = lib.ipc_create_workspace(p, max(1, n))
        assert ws, lib.ipc_last_error().decode()
        try:
            size_before = lib.ipc_workspace_bytes(ws)
            ok(lib.ipc_prepare_panel(ws, fp(source), source.shape[1], 5, n))
            q = np.empty((k, n), np.int8)
            scales, sums = np.empty(n, np.float32), np.empty(n, np.int32)
            ok(lib.ipc_copy_panel(ws, q.ctypes.data_as(i8), fp(scales), sums.ctypes.data_as(i32), n))
            same(q, q_ref); same(scales, scales_ref); same(sums, sums_ref)
            y = np.full((m, n + 13), np.float32(-12345))
            ok(lib.ipc_run_panel(p, ws, fp(y), y.shape[1], 7, 0, m))
            same(y[:, 7:n + 7], y_ref)
            assert np.all(y[:, :7] == -12345) and np.all(y[:, n + 7:] == -12345)
            assert lib.ipc_workspace_bytes(ws) == size_before, "Workspace grew during use or inspection"
            # MR-aligned split rows must have identical output and coverage.
            split = lib.ipc_row_step(p)
            if split < m:
                y.fill(-12345)
                ok(lib.ipc_run_panel(p, ws, fp(y), y.shape[1], 7, 0, split))
                ok(lib.ipc_run_panel(p, ws, fp(y), y.shape[1], 7, split, m))
                same(y[:, 7:n + 7], y_ref)
            records.append({"M": m, "K": k, "T": n, "exact_quantization": True, "exact_output": True, "strided_output": True})
        finally:
            lib.ipc_destroy_workspace(ws); lib.ipc_destroy_plan(p)

    # An immutable prepared panel is usable by different M/weights with equal K.
    x = rng.normal(size=(128, 129)).astype(np.float32)
    weights = [rng.normal(size=(m, 128)).astype(np.float32) for m in (17, 32, 65)]
    expected = [reference(w, x)[0] for w in weights]
    plans = [create(w, 3) for w in weights]
    ws = lib.ipc_create_workspace(plans[0], 256)
    malformed = 0
    def rejected(value):
        nonlocal malformed
        assert value == -1, "Malformed panel request unexpectedly accepted"
        malformed += 1
    try:
        y = np.empty((65, 129), np.float32)
        rejected(lib.ipc_run_panel(plans[2], ws, fp(y), 129, 0, 0, 65))
        ok(lib.ipc_prepare_panel(ws, fp(x), 129, 0, 129))
        def worker(i):
            which = i % 3
            y = np.empty_like(expected[which])
            ok(lib.ipc_run_panel(plans[which], ws, fp(y), 129, 0, 0, y.shape[0]))
            same(y, expected[which])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(48)))
        rejected(lib.ipc_run_panel(plans[0], ws, fp(x), 129, 0, 0, 17))
        rejected(lib.ipc_run_panel(plans[0], ws, fp(y), 128, 0, 0, 17))
        rejected(lib.ipc_run_panel(plans[0], ws, fp(y), 129, 1, 0, 17))
        rejected(lib.ipc_run_panel(plans[0], ws, fp(y), 129, 0, 1, 17))
        other_weights = np.zeros((17, 64), np.float32)
        other = create(other_weights, 3)
        try: rejected(lib.ipc_run_panel(other, ws, fp(y), 129, 0, 0, 17))
        finally: lib.ipc_destroy_plan(other)
        for first, length in ((-1, 1), (130, 0), (0, 130), (0, -1)):
            rejected(lib.ipc_prepare_panel(ws, fp(x), 129, first, length))
            rejected(lib.ipc_run_panel(plans[0], ws, fp(y), 129, 0, 0, 17))
            ok(lib.ipc_prepare_panel(ws, fp(x), 129, 0, 129))
        bad = x.copy(); bad[5, 5] = np.nan
        rejected(lib.ipc_prepare_panel(ws, fp(bad), 129, 0, 129))
        rejected(lib.ipc_run_panel(plans[0], ws, fp(y), 129, 0, 0, 17))
        ok(lib.ipc_prepare_panel(ws, fp(x), 129, 0, 129))
        # The full-sequence compatibility API must stay packed-only until its
        # explicit inspection call, and all preparation-state guards remain.
        inp = lib.ipc_allocate_input(plans[0], fp(x), 129, 4)
        assert inp
        try:
            packed_only_bytes = lib.ipc_input_bytes(inp)
            rejected(lib.ipc_run_rows(plans[0], inp, fp(y), 0, 17))
            jobs = lib.ipc_prepare_jobs(inp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda j: ok(lib.ipc_prepare_job(inp, j)), range(jobs)))
            out = np.empty_like(expected[0]); ok(lib.ipc_run_rows(plans[0], inp, fp(out), 0, 17)); same(out, expected[0])
            assert lib.ipc_input_bytes(inp) == packed_only_bytes
            rejected(lib.ipc_prepare_job(inp, 0))
            ok(lib.ipc_check_packed_input(inp))
            assert lib.ipc_input_bytes(inp) > packed_only_bytes
        finally: lib.ipc_destroy_input(inp)
    finally:
        lib.ipc_destroy_workspace(ws)
        for p in plans: lib.ipc_destroy_plan(p)
    assert digest(args.library) == args.library_sha256
    report = {"status": "complete", "GPU_used": False, "scope": "native exact panel checks only; no timings",
              "library_sha256": args.library_sha256, "script_sha256": digest(__file__),
              "cases": records, "exact_cases": len(records), "shared_workspace_calls": 48,
              "malformed_rejections": malformed, "all_passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: json.dump(report, stream, indent=2)
    print(json.dumps({k: report[k] for k in ("status", "exact_cases", "shared_workspace_calls", "malformed_rejections")}))


if __name__ == "__main__":
    main()
