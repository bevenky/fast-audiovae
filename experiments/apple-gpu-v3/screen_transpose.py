"""Fixed transpose-as-matrix screen; original input histories and V2 harness."""
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
        raise ValueError("Use a simple output name")
    output = HERE / (args.name + ".json")
    if output.exists():
        raise FileExistsError(output)
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(key, "0") != "0":
            raise RuntimeError(key + " must be disabled before importing Torch")
        os.environ[key] = "0"
    os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(V2 / "inductor-cache"),
                      TORCHINDUCTOR_COMPILE_THREADS="1", ORT_DISABLE_TELEMETRY="1")
    import torch
    torch.set_num_threads(1)
    if args.cpu_only:
        torch.set_num_interop_threads(1)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def projected_decode(self, x, history):
        length = x.shape[-1]
        channels, stride = self.weight.shape[1], self.stride
        if length == 0:
            return x.new_empty((1, channels, 0)), history.clone()
        joined = torch.cat((history, x), dim=-1)
        p = torch.mm(self._packed_transpose, joined[0]).reshape(channels, 2*stride, length+1)
        y = (p[:, :stride, 1:] + p[:, stride:, :-1]).permute(0, 2, 1)
        y = y.reshape(1, channels, length*stride) + self.bias[None, :, None]
        return y.contiguous(), x[..., -1:].clone()

    def pack(module):
        weight = module.weight
        assert weight.dtype == torch.float32 and weight.shape[-1] == 2*module.stride
        packed = weight.permute(1, 2, 0).reshape(weight.shape[1]*weight.shape[2], weight.shape[0]).contiguous()
        module.register_buffer("_packed_transpose", packed, persistent=False)
        module.decode = types.MethodType(projected_decode, module)
        return packed.numel()*packed.element_size()

    def reference(x, history, weight, bias, stride):
        if not x.shape[-1]:
            return x.new_empty((1, weight.shape[1], 0))
        y = torch.nn.functional.conv_transpose1d(torch.cat((history, x), -1), weight, bias, stride=stride)
        return y[..., stride:stride+x.shape[-1]*stride]

    # Tiny CPU algebra gate, before importing the GPU harness or loading models.
    generator = torch.Generator(device="cpu").manual_seed(301)
    checks = []; maximum = 0.0
    for stride in (2, 5, 6, 8):
        module = torch.nn.Module()
        module.stride = stride
        module.register_buffer("weight", torch.randn(3, 2, 2*stride, generator=generator)*.1)
        module.register_buffer("bias", torch.tensor([.25, -.125], dtype=torch.float32))
        pack(module)
        for pattern, scale in (("signed", 1.0), ("quiet", 1e-5), ("zero", 0.0)):
            x = torch.randn(1, 3, 5, generator=generator)*scale
            initial = torch.randn(1, 3, 1, generator=generator)*scale
            for length in (0, 1, 2, 5):
                value = x[..., :length].clone(); old_x = value.clone(); old_h = initial.clone()
                actual, state = module.decode(value, initial)
                expected = reference(value, initial, module.weight, module.bias, stride)
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
                error = (actual-expected).abs().max().item() if length else 0.0
                maximum = max(maximum, error)
                assert torch.equal(value, old_x) and torch.equal(initial, old_h)
                assert torch.equal(state, value[..., -1:] if length else initial)
                assert state.is_contiguous() and state.data_ptr() != initial.data_ptr()
                checks.append(dict(stride=stride, pattern=pattern, frames=length, max_abs=error))
            state = initial.clone(); parts = []; position = 0
            for length in (1, 0, 2, 2):
                y, state = module.decode(x[..., position:position+length].clone(), state)
                parts.append(y); position += length
            joined = torch.cat(parts, -1)
            torch.testing.assert_close(joined, reference(x, initial, module.weight, module.bias, stride),
                                       atol=2e-7, rtol=2e-6)
            changed = x.clone(); changed[..., 3:] += .5
            changed_y, _ = module.decode(changed, initial)
            torch.testing.assert_close(changed_y[..., :3*stride], joined[..., :3*stride], atol=2e-7, rtol=2e-6)
    math_receipt = dict(status="passed", device="cpu", independent_single_call_cases=checks,
                        cases=len(checks), uneven_stream_and_future_cases=12, max_abs=maximum,
                        atol=2e-7, rtol=2e-6, model_weights_loaded=False)
    if args.cpu_only:
        with output.open("x") as handle:
            json.dump(dict(status="passed", cpu_math=math_receipt, script_sha256=source_sha,
                           gpu_executed=False), handle, indent=2)
            handle.write("\n")
        print("CPU algebra passed:", len(checks), "cases; max_abs", maximum)
        return

    sys.path.insert(0, str(V2))
    import experiment as exp
    original_build = exp.build
    packed_bytes = {}

    def pointwise_mm(self, x):
        return torch.mm(self.weight[:, :, 0], x[0]).unsqueeze(0) + self.bias[None, :, None]

    def build(arm):
        if arm == "before_compile":
            return original_build(arm)
        model = exp.legacy(); count = 0; size = 0; pointwise_count = 0
        for module in model.modules():
            if module.__class__.__name__ == "_CausalTranspose1d":
                size += pack(module); count += 1
            elif arm == "transpose_pointwise_mm" and module.__class__.__name__ == "_Pointwise":
                module.forward = types.MethodType(pointwise_mm, module); pointwise_count += 1
        assert count == 6 and pointwise_count == (19 if arm == "transpose_pointwise_mm" else 0)
        packed_bytes[arm] = size
        return exp.CompiledModel(model)

    arms = ["before_compile", "transpose_mm"] + (["transpose_pointwise_mm"] if args.combined else [])
    exp.build = build
    try:
        exp.main(argparse.Namespace(name="../apple-gpu-v3/" + args.name, arms=arms, repetitions=2))
        result = json.loads(output.read_text())
        graphs = result["compile_counters_after_preparation"].get("stats", {}).get("unique_graphs", 0)
        if not 0 < graphs <= 2*len(arms):
            result.update(status="failed", error="Exceeded fixed compile-graph bound")
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            raise RuntimeError(result["error"])
    finally:
        exp.build = original_build
        if output.exists():
            result = json.loads(output.read_text())
            result["transpose_screen"] = dict(script_sha256=source_sha, cpu_math=math_receipt,
                original_input_history=True, packed_weight_bytes=packed_bytes,
                packing="Once before compilation; literal FP32 permutation. Original weight buffers also retained.",
                prefer_metal=None, combined_pointwise=args.combined, compile_graph_limit=2*len(arms),
                harness="UnchangedV2:1qualification,1warmup,2measured;three960msclips;40/80ms")
            result.setdefault("files", {})[str(Path(__file__).resolve())] = source_sha
            unchanged = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha
            result["transpose_screen"]["script_unchanged"] = unchanged
            if not unchanged:
                result.update(status="failed", error="Screen source changed during run")
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    main(parser.parse_args())
