"""Portable CPU fallback for hosts without a verified native recipe."""
from pathlib import Path
from .common import completed_bundle


def build_recipe(work_dir, source, platform_info, mode, threads):
    from ..prepare_streaming import prepare_streaming
    destination = Path(work_dir) / ("bundles/portable-" + mode)
    # No build records in this private directory can enable a native recipe.
    import onnx
    import json
    from ..graph.upsampling import rewrite
    from ..assets import sha256
    def create(staged):
        model, _ = rewrite(onnx.load(source), mode="split-matmul")
        staged.mkdir(parents=True)
        onnx.external_data_helper.convert_model_from_external_data(model)
        onnx.save_model(model, staged / "decoder_portable.onnx")
        manifest = {"onnxruntime": "1.29.0", "fallback": "decoder_portable.onnx", "native": {},
                    "source_sha256": sha256(source), "providers": ["CPUExecutionProvider"]}
        (staged / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if mode == "streaming":
            prepare_streaming(staged)
    return completed_bundle(destination, create)
