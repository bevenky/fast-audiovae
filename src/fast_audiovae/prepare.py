"""Build a model bundle from the pinned export and locally built CPU operators."""
import json
from pathlib import Path
import platform
import shutil
import tempfile

import onnx

from .assets import fetch_model, verify_model, sha256
from .graph import elementwise, upsampling, pointwise, phase
from .graph import block_fusion as block_fusion_graph


BLOCK_FUSIONS = ("none", "adds", "chain", "both")


def _fusion_operators(variant):
    required = {"SnakeF32", "CausalDW7SnakeF32", "PhaseSumBiasInterleaveF32"}
    if variant in ("chain", "both"):
        required.add("SnakeDW7SnakeF32")
    if variant in ("adds", "both"):
        required.add("BiasResidualF32")
    return required


def _save(model, destination):
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, destination)


def _verified_build(record_path, expected_domain, *, required_operators=()):
    build = json.loads(Path(record_path).read_text())
    if (not isinstance(build, dict)
            or type(build.get("native_abi")) is not int or build["native_abi"] != 1
            or type(build.get("ort_api_version")) is not int or build["ort_api_version"] != 29
            or build.get("domain") != expected_domain):
        raise ValueError("Build record has an incompatible CPU domain, native ABI or ORT API")
    if required_operators:
        operators = build.get("operators")
        if (not isinstance(operators, list) or not all(isinstance(op, str) and op for op in operators)
                or len(set(operators)) != len(operators)):
            raise ValueError("Block fusion requires a new native build declaring its supported operators")
        missing = set(required_operators) - set(operators)
        if missing:
            raise ValueError("Native build lacks required block-fusion operators: " + ", ".join(sorted(missing)))
    if not isinstance(build.get("library"), str) or not build["library"]:
        raise ValueError("Build record must identify its native library")
    library = Path(build["library"]).resolve()
    if not library.is_file() or sha256(library) != build.get("library_sha256"):
        raise ValueError("Native library does not match its build record")
    return library


def prepare(output="artifacts", *, source=None, native_build=None, amd_build=None, block_fusion="none"):
    if not isinstance(block_fusion, str) or block_fusion not in BLOCK_FUSIONS:
        raise ValueError("block_fusion must be none, adds, chain or both")
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
    required_operators = _fusion_operators(block_fusion) if block_fusion != "none" else set()
    if required_operators and not native_build:
        raise ValueError("Block fusion requires a compatible native build; build the local native library first")
    if amd_build and target != "x86":
        raise ValueError("AMD matrix packing requires the Linux x86 backend")
    if native_build:
        if target is None:
            raise ValueError("No native preparation recipe is available for this platform")
        expected_domain = "venky.audio.cpu" if target == "apple" else "venky.audio.cpu.portable"
        library = _verified_build(native_build, expected_domain, required_operators=required_operators)
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
                "source_sha256": sha256(source), "precision": "FP32", "providers": ["CPUExecutionProvider"],
                "block_fusion": block_fusion}
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
            destination = temporary / "native-platform.onnx" if required_operators else output / "decoder_native.onnx"
            if target == "x86":
                from .graph.portable import convert
                convert(temporary / "native.onnx", destination, library)
            else:
                shutil.copy2(temporary / "native.onnx", destination)
            if required_operators:
                fusion_audit = block_fusion_graph.rewrite(
                    destination, output / "decoder_native.onnx", variant=block_fusion, backend=None,
                    expected_chains=18 if block_fusion in ("chain", "both") else 0,
                    expected_adds=18 if block_fusion in ("adds", "both") else 0)
        entry = {"library": "runtime/" + library.name, "model": "decoder_native.onnx",
                 "math": "vforce" if target == "apple" else "sleef_u10",
                 "experiment": "phase_finish_fused" if target == "apple" else "phase_fused",
                 "tested_cpu": "Apple M5 Max" if target == "apple" else "AMD EPYC 9654",
                 "library_sha256": sha256(library), "model_sha256": sha256(output / "decoder_native.onnx")}
        if required_operators:
            audit_path = output / "decoder_native.block-fusion.json"
            entry["block_fusion"] = {
                "variant": block_fusion, "counts": fusion_audit["counts"],
                "backend_override": None, "required_operators": sorted(required_operators),
                "audit_file": audit_path.name, "audit_sha256": sha256(audit_path),
                "source_model_sha256": fusion_audit["source_sha256"],
                "model_sha256": fusion_audit["output_sha256"],
                "native_build_record_sha256": sha256(Path(native_build)),
                "validation": "Preparation performs checked graph rewriting only; it runs no numerical, quality or timing validation"}
            entry["baseline_experiment"] = entry["experiment"]
            entry["baseline_tested_cpu"] = entry["tested_cpu"]
            entry["experiment"] = "block_fusion_" + block_fusion
            entry["tested_cpu"] = "Not recorded for this prepared fusion variant"
        manifest["native"][key] = entry
        if amd_build:
            from .graph.packed import rewrite, SELECTED_NODES
            rewrite(output / "decoder_native.onnx", output / "decoder_amd.onnx", SELECTED_NODES, 4)
            shutil.copy2(packed_library, output / "runtime" / packed_library.name)
            entry["packed"] = {"library": "runtime/" + packed_library.name, "model": "decoder_amd.onnx",
                               "experiment": "phase_aocl_rows", "tested_cpu": "AMD EPYC 9654",
                               "validated_threads": [1, 4], "default": False}
            if required_operators:
                entry["packed"].update(block_fusion=block_fusion,
                    baseline_experiment="phase_aocl_rows", baseline_tested_cpu="AMD EPYC 9654",
                    experiment="phase_aocl_rows_block_fusion_" + block_fusion,
                    tested_cpu="Not recorded for this prepared fusion variant with AMD packing",
                    thread_gate_origin="Existing AMD packing policy; prepare does not validate the fusion combination")
    manifest["fallback_sha256"] = sha256(output / manifest["fallback"])
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
