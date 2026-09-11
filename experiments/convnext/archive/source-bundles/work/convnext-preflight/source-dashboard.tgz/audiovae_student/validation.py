"""CPU-only structural qualification of a freshly initialized exported student.

This is neither speech-quality scoring nor a trained-model benchmark. It checks
all waveform samples, including streaming boundaries, against the same seed's
PyTorch implementation. Report files include the exact exported graph hashes.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import onnxruntime as ort
import torch

from .export import OnnxStudentStream
from .model import StudentConfig, StudentDecoder


def validate_bundle(directory: Path) -> dict:
    torch.set_num_threads(1)
    spec = json.loads((directory / "bundle.json").read_text())
    if spec["provenance"].get("qualification") != "untrained_architecture_only":
        raise ValueError("This seed-replay validator is for untrained architecture exports only")
    torch.manual_seed(spec["provenance"]["seed"])
    model = StudentDecoder(StudentConfig(**spec["architecture"]["config"])).cpu().eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    import hashlib
    full_path = directory / "decoder.onnx"
    if hashlib.sha256(full_path.read_bytes()).hexdigest() != spec["graphs"][full_path.name]:
        raise ValueError("Full decoder checksum mismatch")
    full = ort.InferenceSession(str(full_path), sess_options=options, providers=["CPUExecutionProvider"])
    stream = OnnxStudentStream(directory)
    checks = []
    atol, rtol = 1e-5, 1e-4

    def compare(name, expected, actual):
        correct_shape = actual.shape == expected.shape
        finite = bool(np.isfinite(actual).all())
        passed = correct_shape and finite and bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        checks.append({"name": name, "passed": passed, "shape": list(actual.shape),
                       "max_abs": float(np.max(np.abs(actual - expected))) if correct_shape and actual.size else None})

    for frames in (1, 7, 33, 67):
        z = np.random.default_rng(frames).normal(size=(1, 64, frames)).astype(np.float32)
        with torch.inference_mode():
            reference = model(torch.from_numpy(z)).numpy()
        compare(f"frames_{frames}/full", reference, full.run(["audio"], {"latents": z})[0])
        for pattern in ([1], [2], [4], [1, 4, 2, 9]):
            stream.reset()
            offset, index, pieces, snapshots = 0, 0, [], []
            while offset < frames:
                end = min(frames, offset + pattern[index % len(pattern)])
                piece = stream.decode_chunk(z[..., offset:end])
                if piece.shape[-1] != (end - offset) * 1920:
                    raise AssertionError("Chunk lost or duplicated samples")
                pieces.append(piece)
                snapshots.append(piece.copy())
                offset, index = end, index + 1
            if stream.frames_decoded != frames or stream.flush().size:
                raise AssertionError("Incorrect frame count or flush tail")
            if any(not np.array_equal(a, b) for a, b in zip(pieces, snapshots)):
                raise AssertionError("A later call changed a previous output")
            compare(f"frames_{frames}/stream_{pattern}", reference, np.concatenate(pieces, axis=-1))
    stream.close()
    return {"qualification": "untrained_architecture_only", "passed": all(c["passed"] for c in checks),
            "checks": checks, "atol": atol, "rtol": rtol, "graphs": spec["graphs"],
            "architecture": spec["architecture"], "parameters": sum(p.numel() for p in model.parameters()),
            "hardware": {"system": platform.system(), "machine": platform.machine()},
            "runtime": {"torch": torch.__version__, "onnxruntime": ort.__version__,
                        "providers": full.get_providers(), "intra_op_threads": 1, "inter_op_threads": 1},
            "speech_quality_evaluated": False, "rtf_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "parameters": report["parameters"], "runtime": report["runtime"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
