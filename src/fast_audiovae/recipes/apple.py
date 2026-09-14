"""Assemble the retained Apple native graph and streaming projection layout."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .common import completed_bundle


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_recipe(work_dir, source, platform_info, mode, threads):
    from ..prepare import prepare
    from ..prepare_streaming import prepare_streaming
    root = Path(work_dir).resolve()
    prebuilt = platform_info.get("prebuilt")
    if prebuilt:
        build_manifest = prebuilt["base_build"]
    else:
        _module(root / "tools/fetch_headers.py", "_apple_recipe_headers").fetch(
            offline=platform_info.get("offline", False))
        built = _module(root / "tools/build_apple.py", "_apple_recipe_build").build()
        build_manifest = built["build_manifest"]
    base = root / "bundles/apple-base"
    def create_base(destination):
        prepare(destination, source=source, native_build=build_manifest)
        prepare_streaming(destination)
        manifest_path = destination / "bundle.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["onnxruntime"] = "1.30.0"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed_bundle(base, create_base)
    if mode == "batch" and platform_info.get("recipe") != "apple_batch_selected":
        return base
    result = root / "bundles/apple-stream-projection"
    def create_projection(destination):
        process = subprocess.run([sys.executable, str(root / "experiments/streaming-matrix/apple/prepare.py"),
                                  "--source", str(base), "--output", str(destination)],
                                 capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError("Apple streaming preparation failed: " + process.stderr)
        # The source completion marker describes the source graph, not this derivative.
        (destination / ".recipe-ready.json").unlink(missing_ok=True)
    projection = completed_bundle(result, create_projection)
    if platform_info.get("recipe") not in ("apple_stream_selected", "apple_batch_selected", "apple_stream_int8"):
        return projection
    if threads not in (1, 4):
        raise ValueError("Selected Apple streaming recipe supports ORT threads 1 or 4")
    if prebuilt:
        streaming_build = prebuilt["streaming_build"]
    else:
        streaming_build = _module(root / "tools/build_apple_streaming.py", "_apple_streaming_build").build(
            offline=platform_info.get("offline", False))["build_manifest"]
    from .apple_selected import prepare as prepare_selected
    selected = root / "bundles/apple-stream-selected"
    selected = completed_bundle(selected, lambda destination: prepare_selected(projection, destination, streaming_build))
    if mode == "batch":
        from .apple_batch import prepare as prepare_batch
        return completed_bundle(root / "bundles/apple-batch-selected",
                                lambda destination: prepare_batch(selected, destination))
    if platform_info.get("recipe") == "apple_stream_int8":
        if threads != 1:
            raise ValueError("Apple INT8 streaming uses one worker")
        if prebuilt:
            int8_build = prebuilt["int8_build"]
        else:
            int8_build = _module(root / "tools/build_apple_int8.py", "_apple_int8_build").build(
                offline=platform_info.get("offline", False))["build_manifest"]
        from .apple_int8 import prepare as prepare_int8
        return completed_bundle(root / "bundles/apple-stream-int8",
                                lambda destination: prepare_int8(selected, destination, int8_build))
    return selected
