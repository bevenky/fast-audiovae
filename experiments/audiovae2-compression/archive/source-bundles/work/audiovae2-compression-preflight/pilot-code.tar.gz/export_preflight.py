"""Export the authenticated initial group candidate and check CPU correctness.

This checks a generic ONNX graph, not the packed native runtime or its speed.
Run after run_pilot.py preflight and before any retained fitting updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from torch import nn

from audiovae_student.teacher import FrozenAudioVAE2, SOURCE_SHA256, CHECKPOINT_SHA256
from group_model import CompressedDecoderGroup, build_student, clone_teacher, effective_weight


VERSION = "audiovae2_group_export_v1"
EXPECTED_UPSAMPLING_WEIGHTS = (
    (2048, 1024, 16), (1024, 256, 12), (256, 128, 10),
    (128, 128, 4), (128, 64, 4), (64, 32, 4),
)
EXPECTED_STRIDES = (8, 6, 5, 2, 2, 2)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def authenticate_preflight(out: Path) -> tuple[dict, dict, dict]:
    """Authenticate all selectors and state before constructing a student."""
    pre = json.loads((out / "preflight.json").read_text())
    if pre.get("passed") is not True:
        raise ValueError("A successful numerical and objective preflight is required")
    identity = pre["identity"]
    paths = {
        "initial_sha256": out / "initial.pt",
        "selection_sha256": out / "channel-selection.json",
    }
    for key, path in paths.items():
        if sha(path) != pre[key]:
            raise ValueError(f"Preflight {key} changed")
    model_sha = sha(Path(__file__).with_name("group_model.py"))
    if identity["model_sha256"] != model_sha:
        raise ValueError("Group model source changed after preflight")
    if identity["teacher_source_sha256"] != SOURCE_SHA256 or identity["teacher_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Preflight did not use the pinned original teacher")
    selection = json.loads(paths["selection_sha256"].read_text())
    if len(selection["stage2_indices"]) != 256 or len(selection["stage3_indices"]) != 128:
        raise ValueError("This export preflight requires the approved 256/128/128 group")
    initial = torch.load(paths["initial_sha256"], map_location="cpu", weights_only=True, mmap=True)
    if initial["format"] != "audiovae2_group_width_v1" or initial["step"] != 0 or initial["identity"] != identity:
        raise ValueError("Initial group checkpoint does not match preflight identity")
    if initial["optimizer"] is not None:
        raise ValueError("Expected the unfitted initialized candidate, without optimizer state")
    metadata = {
        "version": VERSION,
        "preflight_sha256": sha(out / "preflight.json"),
        "initial_sha256": pre["initial_sha256"],
        "selection_sha256": pre["selection_sha256"],
        "model_sha256": model_sha,
        "teacher_source_sha256": SOURCE_SHA256,
        "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "exporter_sha256": sha(__file__),
    }
    return metadata, selection, initial


class ExportDecoder(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, latents):
        # Both outputs permit an independent boundary check after export.
        return self.student.forward_latents(latents)


@torch.no_grad()
def folded_export_copy(student):
    """Materialize legacy WN on an independent copy, leaving reference intact.

    Legacy ONNX export does not consistently constant-fold WN in every Torch
    version. Removing this weight parametrization once does not change the
    convolution or activation operations used at inference.
    """
    decoder = clone_teacher(student.decoder)
    folded = []
    for name, module in decoder.named_modules():
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            expected = effective_weight(module).detach()
            torch.nn.utils.remove_weight_norm(module)
            if not isinstance(module.weight, nn.Parameter) or not torch.equal(module.weight, expected):
                raise RuntimeError(f"Export weight normalization changed an effective matrix: {name}")
            folded.append(name)
    if not folded:
        raise ValueError("Expected original legacy weight normalization on the export input")
    if any(hasattr(module, "weight_g") or hasattr(module, "weight_v") for module in decoder.modules()):
        raise RuntimeError("An export convolution still has a weight-normalization parametrization")
    copied = CompressedDecoderGroup(decoder, student.selections, student.output_sample_rate)
    copied.requires_grad_(False).eval()
    return ExportDecoder(copied).eval(), {
        "method": "Materialize legacy weight normalization on an independent export-only decoder copy",
        "folded_convolutions": len(folded), "modules": folded,
        "effective_weights_bitwise_equal": True,
        "reference": "Original unfolded student with unchanged g/v weights",
    }


def graph_dimensions(graph) -> list[dict]:
    """Check that the graph actually contains the narrower matrix dimensions."""
    import onnx

    weights = {item.name: tuple(item.dims) for item in graph.graph.initializer}
    nodes = [node for node in graph.graph.node if node.op_type == "ConvTranspose"]
    if len(nodes) != 6:
        raise ValueError(f"Expected six exported causal upsamplers, found {len(nodes)}")
    records = []
    for stage, (node, expected, stride) in enumerate(zip(nodes, EXPECTED_UPSAMPLING_WEIGHTS, EXPECTED_STRIDES), 1):
        dims = weights.get(node.input[1])
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if dims != expected or tuple(attrs.get("strides", ())) != (stride,):
            raise ValueError(f"Stage {stage} did not export its expected smaller weight/stride: {dims}, {attrs}")
        records.append({"stage": stage, "node": node.name, "weight_shape": list(dims), "stride": stride})
    return records


def compare(reference: np.ndarray, actual: np.ndarray, expected_shape: tuple[int, ...], label: str) -> dict:
    if reference.shape != expected_shape or actual.shape != expected_shape:
        raise RuntimeError(f"{label}: sample or feature count changed")
    if reference.dtype != np.float32 or actual.dtype != np.float32:
        raise RuntimeError(f"{label}: expected FP32 arrays")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise RuntimeError(f"{label}: nonfinite output")
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    limit = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    failed = int(np.count_nonzero(difference > limit))
    result = {
        "shape": list(actual.shape), "elements": int(actual.size),
        "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "outside_tolerance": failed, "passed": failed == 0,
    }
    return result


def execute(assets: Path, out: Path) -> dict:
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    metadata, selection, initial = authenticate_preflight(out)
    report = {
        **metadata, "passed": False, "device": "cpu",
        "torch": str(torch.__version__), "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "opset": 18, "exporter": "torch.onnx.export legacy, dynamo=False",
        "torch_threads": 1, "ort_intra_threads": 1, "ort_inter_threads": 1,
        "rtf_measured": False, "streaming_verified": False,
        "native_integration": {
            "status": "pending",
            "reason": "Generic ONNX export only; packed native kernels and bounded streaming state require separate validation.",
        },
        "parity_tolerance": {"atol": 1e-5, "rtol": 1e-4}, "parity_records": [],
    }
    report_path = out / "export-check.json"
    write_json(report_path, report)
    try:
        teacher = FrozenAudioVAE2.from_files(assets / "audio_vae_v2.py", assets / "audiovae.pth", device="cpu")
        student = build_student(teacher.model.decoder, selection["stage2_indices"], selection["stage3_indices"])
        student.load_group_state_dict(initial["group"])
        student.requires_grad_(False).eval()
        del initial, teacher
        wrapper = ExportDecoder(student).eval()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in wrapper.parameters()):
            raise RuntimeError("Export must use CPU FP32 weights exclusively")
        export_wrapper, report["weight_normalization"] = folded_export_copy(student)
        generator = torch.Generator(device="cpu").manual_seed(20260910)
        inputs = {frames: torch.randn(1, 64, frames, generator=generator, dtype=torch.float32) for frames in (2, 5)}
        graph_path = out / "experimental-candidate.onnx"
        temp_path = out / "experimental-candidate.onnx.tmp"
        with torch.no_grad():
            torch.onnx.export(
                export_wrapper, (inputs[2],), temp_path,
                input_names=["latents"], output_names=["waveform", "group_output"],
                dynamic_axes={"latents": {2: "latent_frames"}, "waveform": {2: "audio_samples"}, "group_output": {2: "group_frames"}},
                opset_version=18, dynamo=False, export_params=True,
                do_constant_folding=True, keep_initializers_as_inputs=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
        graph = onnx.load(str(temp_path))
        onnx.checker.check_model(graph, full_check=True)
        report["upsampling_dimensions"] = graph_dimensions(graph)
        report["node_count"] = len(graph.graph.node)
        report["initializer_count"] = len(graph.graph.initializer)
        del graph
        os.replace(temp_path, graph_path)
        report["onnx_path"] = str(graph_path.resolve())
        report["onnx_sha256"] = sha(graph_path)
        report["onnx_bytes"] = graph_path.stat().st_size
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
        report["providers"] = session.get_providers()
        if report["providers"] != ["CPUExecutionProvider"]:
            raise RuntimeError("Unexpected ONNX execution provider")
        with torch.no_grad():
            for frames, latents in inputs.items():
                reference = wrapper(latents)
                folded_reference = export_wrapper(latents)
                actual = session.run(["waveform", "group_output"], {"latents": latents.numpy()})
                row = {"latent_frames": frames, "input_shape": list(latents.shape)}
                row["folded_pytorch_parity"] = {}
                for key, expected, p, t, folded in zip(
                    ("waveform", "group_output"), ((1, 1, frames * 1920), (1, 128, frames * 480)), actual, reference, folded_reference,
                ):
                    row[key] = compare(t.numpy(), p, expected, key)
                    row["folded_pytorch_parity"][key] = compare(t.numpy(), folded.numpy(), expected, "folded " + key)
                row["waveform_peak"] = float(np.abs(actual[0]).max())
                row["bounded_output"] = row["waveform_peak"] <= 1.0 + 2e-6
                report["parity_records"].append(row)
                write_json(report_path, report)
        # Bind the receipt to the exact files used throughout this check.
        current, _, _ = authenticate_preflight(out)
        if current != metadata or sha(graph_path) != report["onnx_sha256"]:
            raise RuntimeError("An authenticated input or graph changed during export validation")
        report["passed"] = all(
            row["waveform"]["passed"] and row["group_output"]["passed"] and row["bounded_output"]
            and all(value["passed"] for value in row["folded_pytorch_parity"].values())
            for row in report["parity_records"]
        ) and len(report["parity_records"]) == 2
        write_json(report_path, report)
        return report
    except BaseException as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(report_path, report)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.out.is_dir():
        raise FileNotFoundError("Use the existing successful pilot preflight directory")
    report = execute(args.assets, args.out)
    print(json.dumps({"passed": report["passed"], "onnx_sha256": report["onnx_sha256"], "report": str(args.out / "export-check.json")}), flush=True)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
