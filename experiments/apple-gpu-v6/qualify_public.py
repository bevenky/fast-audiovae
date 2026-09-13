"""Apply the unchanged V2 26-state panel to the integrated public GPU loader.

Only experiment-module factories are temporarily redirected. The candidate
decoder itself is returned by fast_audiovae.load, without production patches.
No model is loaded at import. This is qualification, not timing evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "TORCHINDUCTOR_USE_FAST_MATH"):
    if os.environ.get(key, "0") != "0":
        raise RuntimeError(key + " must be zero before Torch import")
    os.environ[key] = "0"
if os.environ.get("PYTORCH_MPS_PREFER_METAL", "0") != "0":
    raise RuntimeError("Unexpected MPS matrix override")
os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
sys.path.insert(0, str(HERE.parent / "apple-gpu-v2"))
import experiment as exp
import qualify_candidates as qualifier
import torch
import fast_audiovae

# experiment.py configures its historical cache at import. Override it before
# any model construction, using the same dedicated cache as v6 compare.py.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(HERE / "inductor-cache")
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"


def counters():
    return {k: dict(v) for k, v in torch._dynamo.utils.counters.items()
            if k in ("stats", "frames", "graph_break", "unimplemented")}


def main(output):
    output = output.expanduser().resolve()
    if output.parent != HERE or output.suffix != ".json" or output.exists():
        raise ValueError("Choose a new JSON output under apple-gpu-v6")
    source = ROOT / "work/fast-audiovae-apple-gpu/src/fast_audiovae"
    if Path(fast_audiovae.__file__).resolve().parent != source:
        raise RuntimeError("Wrong public package source")
    files = [Path(__file__).resolve(), Path(qualifier.__file__), Path(exp.__file__),
             HERE.parent / "apple-gpu-v2/mps_decoder_before.py"] + sorted(source.glob("*.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    original_build, original_factory = exp.build, qualifier.GPUDecoder
    loaded = None
    metadata = dict(true_arm="integrated_public_gpu", raw_alias="before_compile",
        reference="Original eager literal-ONNX-weight FP32 MPS port; actual upstream class waveform checks are in compare.py",
        candidate_factory="fast_audiovae.load(device='gpu', mode='streaming', threads=1, offline=True)",
        public_load_internal_preparation_calls=2, unchanged_tolerances=True)

    def build(arm):
        nonlocal loaded
        if arm == "before":
            return original_build(arm)
        if arm != "before_compile" or loaded is not None:
            raise RuntimeError("Unexpected qualification factory request")
        begin = time.perf_counter()
        loaded = fast_audiovae.load(device="gpu", mode="streaming", threads=1,
            source=exp.ORIGINAL, offline=True, cache_dir=HERE / "cache")
        torch.mps.synchronize()
        info = loaded.info
        if (info["selected"] != "torch_mps_hybrid" or info["backend"] != "mps"
                or info["fallback"] or info["precision"] != "FP32"
                or set(loaded._model.compiled) != {1, 2}):
            raise RuntimeError("Integrated public model was not prepared")
        metadata.update(load_seconds=time.perf_counter()-begin, info=info,
                        counters_after_public_load=counters())
        torch._dynamo.config.error_on_recompile = True
        return loaded._model

    def decoder_factory(model, *args):
        if loaded is not None and model is loaded._model:
            return loaded
        return original_factory(model, *args)

    exp.build, qualifier.GPUDecoder = build, decoder_factory
    try:
        qualifier.main(output, ["before_compile"])
    finally:
        exp.build, qualifier.GPUDecoder = original_build, original_factory
        if output.exists():
            result = json.loads(output.read_text())
            metadata["counters_final"] = counters()
            metadata["sources_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                                for p, h in hashes.items())
            result["public_integration"] = metadata
            result.setdefault("files", {}).update(hashes)
            result["protocol"]["state_reference"] = metadata["reference"]
            result["protocol"]["public_load_exclusion"] = "The public loader performs two zero-input compile+decode preparations; their combined load time is separately reported, outside the unchanged93-call panel"
            # The old qualifier labels four candidate preparation calls compile.
            # Here compilation happened in load; count those four ordinary calls
            # toward the same15s ordinary-work limit as the remaining panel.
            ordinary = result.get("gpu_api_seconds", 0) + result.get("state_inspection_seconds", 0) + result.get("compilation_first_call_seconds", 0)
            metadata["ordinary_panel_gpu_and_inspection_seconds"] = ordinary
            metadata["total_calls_including_public_load"] = result.get("calls", 0) + 2
            if result.get("status") == "passed":
                state = result["state_summary"]["before_compile"]
                valid = (result["calls"] == 93 and len(state) == 26
                    and all(v["passed"] and v["comparisons"] == 39 for v in state.values())
                    and metadata["counters_final"] == metadata["counters_after_public_load"]
                    and metadata["sources_unchanged"] and ordinary < 15
                    and set(loaded._model.compiled) == {1, 2})
                if not valid:
                    result.update(status="failed", error="Public integration accounting, recompilation, source, or time guard failed")
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result = json.loads(output.read_text())
    if result.get("status") != "passed":
        raise RuntimeError(result.get("error", "Public state qualification failed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
