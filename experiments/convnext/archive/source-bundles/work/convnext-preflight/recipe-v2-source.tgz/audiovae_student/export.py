"""Export this student's full and explicit-state ONNX graphs for CPU checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import DecoderState, StudentConfig, StudentDecoder, architecture_summary


class FullGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents):
        return self.model._forward_nonempty(latents)


class StreamGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, started, *histories):
        audio, state = self.model._step_nonempty(
            latents, DecoderState(started, histories, self.model.normalization_layout))
        return audio, state.started, *state.histories


def export_decoder(model: StudentDecoder, directory: str | Path, *, provenance: dict) -> dict:
    """Serialize B=1 FP32 graphs, dynamic positive latent lengths and provenance.

    Empty chunks are handled outside ONNX. The initial boolean state seeds
    per-layer edge replication, rather than incorrectly seeding all layers zero.
    This export is experimental and does not replace the stable runtime bundle.
    Fixed normalization is folded into convolutions on a separate model copy;
    the caller's weights, statistics and live streams are not modified.
    """
    if model.training or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Export requires an evaluation-mode CPU FP32 student")
    if not isinstance(provenance, dict) or not provenance.get("qualification"):
        raise ValueError("Export must explicitly identify its qualification status")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in ("decoder.onnx", "decoder-stream.onnx", "bundle.json")]
    if any(p.exists() for p in paths):
        raise FileExistsError("Use a new output directory; existing model artifacts are not overwritten")
    source_normalization = model.config.normalization_mode
    model = model.fold_normalization()
    z = torch.zeros(1, 64, 2)
    state = model.initial_state()
    state_inputs = [f"history_{i}" for i in range(len(state.histories))]
    state_outputs = [f"next_history_{i}" for i in range(len(state.histories))]
    common = {"opset_version": 18, "dynamo": False, "do_constant_folding": True}
    with torch.inference_mode():
        torch.onnx.export(FullGraph(model).eval(), (z,), str(paths[0]), input_names=["latents"], output_names=["audio"],
                          dynamic_axes={"latents": {2: "latent_frames"}, "audio": {2: "samples"}}, **common)
        torch.onnx.export(StreamGraph(model).eval(), (z, state.started, *state.histories), str(paths[1]),
                          input_names=["latents", "started", *state_inputs],
                          output_names=["audio", "next_started", *state_outputs],
                          dynamic_axes={"latents": {2: "latent_frames"}, "audio": {2: "samples"}}, **common)
    import onnx
    for path in paths[:2]:
        onnx.checker.check_model(str(path), full_check=True)
    result = {"schema_version": 1, "architecture": architecture_summary(model.config), "provenance": provenance,
              "torch_version": torch.__version__, "onnx_version": onnx.__version__, "opset": 18,
              "normalization": {"source_mode": source_normalization, "exported_mode": "folded",
                                "requires_new_streaming_state": True},
              "graphs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]},
              "states": [{"input": i, "output": o, "shape": list(h.shape), "dtype": "float32"}
                         for i, o, h in zip(state_inputs, state_outputs, state.histories)],
              "started": {"input": "started", "output": "next_started", "shape": [1], "dtype": "bool"},
              "minimum_latent_frames": 1, "batch_size": 1, "default_mode": "streaming"}
    paths[2].write_text(json.dumps(result, indent=2) + "\n")
    return result


class OnnxStudentStream:
    """Experimental one-thread CPU runner with transactional bounded state."""
    def __init__(self, directory: str | Path):
        import onnxruntime as ort
        if ort.__version__ != "1.29.0":
            raise RuntimeError("This experiment requires ONNX Runtime 1.29.0")
        directory = Path(directory)
        self.manifest = json.loads((directory / "bundle.json").read_text())
        if (self.manifest.get("schema_version") != 1 or self.manifest.get("batch_size") != 1
                or self.manifest.get("architecture", {}).get("samples_per_latent") != 1920):
            raise ValueError("Unsupported student bundle contract")
        graph = directory / "decoder-stream.onnx"
        if hashlib.sha256(graph.read_bytes()).hexdigest() != self.manifest["graphs"][graph.name]:
            raise ValueError("Student graph checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        self.closed = False
        self.reset()

    def _open(self):
        if self.closed:
            raise RuntimeError("Stream is closed")

    def reset(self):
        self._open()
        self.history = {s["input"]: np.zeros(s["shape"], dtype=np.float32) for s in self.manifest["states"]}
        self.started = np.zeros(1, dtype=np.bool_)
        self.frames_decoded = 0

    def decode_chunk(self, latents: np.ndarray) -> np.ndarray:
        self._open()
        if (not isinstance(latents, np.ndarray) or latents.dtype != np.float32 or latents.ndim != 3
                or latents.shape[:2] != (1, 64) or not np.isfinite(latents).all()):
            raise ValueError("Expected finite FP32 latents shaped [1,64,T]")
        if latents.shape[-1] == 0:
            return np.empty((1, 1, 0), dtype=np.float32)
        names = ["audio", "next_started", *[s["output"] for s in self.manifest["states"]]]
        values = self.session.run(names, {"latents": np.ascontiguousarray(latents), "started": self.started, **self.history})
        audio, started, *histories = values
        if (audio.shape != (1, 1, latents.shape[-1] * 1920) or audio.dtype != np.float32 or not np.isfinite(audio).all()
                or started.shape != (1,) or started.dtype != np.bool_ or not started.all()
                or len(histories) != len(self.manifest["states"])):
            raise RuntimeError("Invalid student output; state not advanced")
        next_history = {}
        for spec, value in zip(self.manifest["states"], histories):
            if value.shape != tuple(spec["shape"]) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError("Invalid student history; state not advanced")
            next_history[spec["input"]] = value
        self.history, self.started = next_history, started
        self.frames_decoded += latents.shape[-1]
        return audio

    def flush(self):
        self._open()
        return np.empty((1, 1, 0), dtype=np.float32)

    def close(self):
        self.closed = True
        self.history.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--untrained", action="store_true", required=True,
                        help="Explicitly export freshly initialized weights for architecture checks only")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = StudentDecoder().cpu().eval()
    manifest = export_decoder(model, args.output, provenance={"qualification": "untrained_architecture_only", "seed": args.seed})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
