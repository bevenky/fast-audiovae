"""Build a model bundle from the pinned export and locally built CPU operators."""
import json
from pathlib import Path
import platform
import shutil
import tempfile

import onnx

from .assets import fetch_model, verify_model, sha256
from .graph import elementwise, upsampling, pointwise, phase


def _save(model, destination):
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, destination)


def _verified_build(record_path, expected_domain):
    build = json.loads(Path(record_path).read_text())
    if (not isinstance(build, dict)
            or type(build.get("native_abi")) is not int or build["native_abi"] != 1
            or type(build.get("ort_api_version")) is not int or build["ort_api_version"] != 29
            or build.get("domain") != expected_domain):
        raise ValueError("Build record has an incompatible CPU domain, native ABI or ORT API")
    if not isinstance(build.get("library"), str) or not build["library"]:
        raise ValueError("Build record must identify its native library")
    library = Path(build["library"]).resolve()
    if not library.is_file() or sha256(library) != build.get("library_sha256"):
        raise ValueError("Native library does not match its build record")
    return library


def prepare(output="artifacts", *, source=None, native_build=None, amd_build=None):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be an empty or nonexistent directory")
    source_path = Path(source).resolve() if source else Path(".deps/models/audio_vae_decoder.onnx").resolve()
    if source_path.is_relative_to(output):
        raise ValueError("The source model must be outside the output directory")
    system, machine = platform.system(), platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    key = system + "/" + machine
    target = {"Darwin/arm64": "apple", "Linux/x86_64": "x86"}.get(key)
    if native_build is None and target:
        candidate = Path(".build") / target / "build.json"
        if candidate.is_file():
            native_build = candidate
    # Finish platform, record and library validation before creating or writing
    # the destination. Failed preflight cannot leave a partial model bundle.
    library = packed_library = None
    if amd_build and target != "x86":
        raise ValueError("AMD matrix packing requires the Linux x86 backend")
    if native_build:
        if target is None:
            raise ValueError("No native preparation recipe is available for this platform")
        expected_domain = "venky.audio.cpu" if target == "apple" else "venky.audio.cpu.portable"
        library = _verified_build(native_build, expected_domain)
    if amd_build:
        if library is None:
            raise ValueError("AMD packing also requires a base native build")
        packed_library = _verified_build(amd_build, "venky.audio.cpu.aocl.rows")
        if packed_library.name == library.name:
            raise ValueError("Base and AMD libraries must have different filenames")
    source = verify_model(source_path if source else fetch_model(source_path.parent))
    output.mkdir(parents=True, exist_ok=True)
    stock = onnx.load(source, load_external_data=True)
    fallback, _ = upsampling.rewrite(stock, mode="split-matmul")
    _save(fallback, output / "decoder_portable.onnx")
    manifest = {"onnxruntime": "1.29.0", "fallback": "decoder_portable.onnx", "native": {},
                "interface": "FP32 [1,64,L] to [1,1,1920*L] at 48000 Hz; fresh causal calls",
                "source_sha256": sha256(source), "precision": "FP32", "providers": ["CPUExecutionProvider"]}
    if native_build:
        (output / "runtime").mkdir(exist_ok=True)
        shutil.copy2(library, output / "runtime" / library.name)
        with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
            temporary = Path(temporary)
            fused, audit = elementwise.rewrite(stock, source.parent, "fused", enable_snake=True)
            if audit["fused_depthwise"] != 18 or audit["matched_snakes"] != 43:
                raise ValueError("Expected the validated 18 fused depthwise and 43 Snake patterns")
            up, _ = upsampling.rewrite(fused, mode="split-matmul")
            _save(up, temporary / "up.onnx")
            pointwise.rewrite(temporary / "up.onnx", temporary / "pointwise.onnx", "bct")
            phase.rewrite(temporary / "pointwise.onnx", temporary / "native.onnx")
            if target == "x86":
                from .graph.portable import convert
                convert(temporary / "native.onnx", output / "decoder_native.onnx", library)
            else:
                shutil.copy2(temporary / "native.onnx", output / "decoder_native.onnx")
        entry = {"library": "runtime/" + library.name, "model": "decoder_native.onnx",
                 "math": "vforce" if target == "apple" else "sleef_u10",
                 "experiment": "phase_finish_fused" if target == "apple" else "phase_fused",
                 "tested_cpu": "Apple M5 Max" if target == "apple" else "AMD EPYC 9654",
                 "library_sha256": sha256(library), "model_sha256": sha256(output / "decoder_native.onnx")}
        manifest["native"][key] = entry
        if amd_build:
            from .graph.packed import rewrite, SELECTED_NODES
            rewrite(output / "decoder_native.onnx", output / "decoder_amd.onnx", SELECTED_NODES, 4)
            shutil.copy2(packed_library, output / "runtime" / packed_library.name)
            entry["packed"] = {"library": "runtime/" + packed_library.name, "model": "decoder_amd.onnx",
                               "experiment": "phase_aocl_rows", "tested_cpu": "AMD EPYC 9654",
                               "validated_threads": [1, 4], "default": False}
    manifest["fallback_sha256"] = sha256(output / manifest["fallback"])
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
