"""Three fixed original-history MPS arms using the unchanged V2 short harness."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import types

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "apple-gpu-v2"


def main(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("Use a simple result name without directories")
    output = HERE / (args.name + ".json")
    if output.exists():
        raise FileExistsError(output)
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(name, "0") != "0":
            raise RuntimeError(name + " must be disabled before importing Torch")
        os.environ[name] = "0"
    if args.prefer_metal:
        os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
    else:
        os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sys.path.insert(0, str(V2))
    import experiment as exp
    import torch

    original_build = exp.build
    replacement_counts = {}

    class LoweredModel(exp.CompiledModel):
        def decode(self, z, state):
            history = self.model._checked_states(z, state)
            length = z.shape[-1]
            if length == 0:
                return self.model.decode(z, state)
            if length not in self.compiled:
                self.compiled[length] = torch.compile(
                    self.step, backend="inductor", fullgraph=True, dynamic=False,
                    options={"conv_1x1_as_mm": True})
            result = self.compiled[length](z, *(history[name] for name in self.names))
            return result[0], {name: value.contiguous() for name, value in zip(self.names, result[1:])}

    def pointwise_mm(self, x):
        return torch.mm(self.weight[:, :, 0], x[0]).unsqueeze(0) + self.bias[None, :, None]

    def build(arm):
        if arm == "before_compile":
            return original_build(arm)
        model = exp.legacy()
        if arm == "mm_lowering":
            return LoweredModel(model)
        if arm == "mm_left":
            changed = 0
            for module in model.modules():
                if module.__class__.__name__ == "_Pointwise":
                    assert module.weight.ndim == 3 and module.weight.shape[-1] == 1
                    module.forward = types.MethodType(pointwise_mm, module)
                    changed += 1
            assert changed == 19, "Expected exactly19 original pointwise modules"
            replacement_counts[arm] = changed
            return exp.CompiledModel(model)
        raise ValueError(arm)

    exp.build = build
    try:
        exp.main(argparse.Namespace(name="../apple-gpu-v3/" + args.name,
            arms=["before_compile", "mm_lowering", "mm_left"], repetitions=2))
        result = json.loads(output.read_text())
        prepared = result["compile_counters_after_preparation"].get("stats", {}).get("unique_graphs", 0)
        if not 0 < prepared <= 6:
            result.update(status="failed", error="Preparation exceeded the6 fixed compile-graph bound")
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            raise RuntimeError(result["error"])
    finally:
        exp.build = original_build
        if output.exists():
            result = json.loads(output.read_text())
            result["matmul_screen"] = dict(
                script=str(Path(__file__).resolve()), script_sha256=source_sha,
                script_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha,
                prefer_metal=args.prefer_metal,
                PYTORCH_MPS_PREFER_METAL=os.environ.get("PYTORCH_MPS_PREFER_METAL"),
                environment_set_before_torch_import=True,
                source_model="original activated-input history, all three arms",
                modifications={"before_compile": "Unchanged experiment.build('before_compile')",
                    "mm_lowering": "torch.compile options conv_1x1_as_mm=True",
                    "mm_left": "19pointwise forwards use contiguous BCT weight-left torch.mm then bias"},
                pointwise_replacements=replacement_counts, compile_graph_limit=6,
                harness="Unchanged V2 experiment.main:1qualification,1warmup,2measured;40/80ms;3clips960ms",
                note="Prefer-Metal flag applies to all three matched arms. Source/weight transforms and compilation are outside ordinary decode timing.")
            result.setdefault("files", {})[str(Path(__file__).resolve())] = source_sha
            if not result["matmul_screen"]["script_unchanged"]:
                result.update(status="failed", error="Screen source changed during execution")
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prefer-metal", action="store_true")
    main(parser.parse_args())
