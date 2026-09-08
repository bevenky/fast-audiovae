"""Assemble the retained Apple native graph and streaming projection layout."""
from __future__ import annotations

import importlib.util
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
    completed_bundle(base, create_base)
    if mode == "batch":
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
    return completed_bundle(result, create_projection)
