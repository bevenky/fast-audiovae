"""Add explicit-state streaming graphs without changing full-call artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import onnx

from .assets import sha256


def _checked_file(root, relative, digest=None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Decoder artifact is missing or outside the bundle: " + relative)
    if digest is not None and sha256(path) != digest:
        raise ValueError("Decoder artifact differs from its manifest: " + relative)
    return path


def _external_hashes(root, source):
    metadata = onnx.load(source, load_external_data=False)
    tensors = list(metadata.graph.initializer)
    for node in metadata.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
    records = {}
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            data = {item.key: item.value for item in tensor.external_data}
            location = data.get("location")
            if not location:
                raise ValueError("External tensor has no file location")
            path = (source.parent / location).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("External weights must be inside the model bundle")
            name = str(path.relative_to(root))
            if name not in records:
                records[name] = sha256(path)
    return records


def prepare_streaming(model_dir="artifacts", *, canonical_precision=False):
    """Derive every decoder variant in a bundle, then publish its manifest.

    The original graph files and native libraries are retained byte-for-byte.
    Quantized recipes require an explicit canonical-precision opt-in, which
    selects a derived full-call reference with consistent short-chunk math.
    Existing streaming bundles must be prepared in a new destination so active
    sessions cannot see a partially replaced set of artifacts.
    """
    from .graph.streaming import rewrite_model
    root = Path(model_dir).resolve()
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    destination = root / "streaming"
    if destination.exists() or "streaming" in manifest:
        raise ValueError("Bundle already contains streaming artifacts; prepare a fresh bundle")
    models = [manifest["fallback"]]
    _checked_file(root, manifest["fallback"], manifest.get("fallback_sha256"))
    for entry in manifest["native"].values():
        _checked_file(root, entry["model"], entry.get("model_sha256"))
        _checked_file(root, entry["library"], entry.get("library_sha256"))
        for record in entry.get("additional_libraries", []):
            _checked_file(root, record["library"], record["sha256"])
        models.append(entry["model"])
        if entry.get("packed"):
            models.append(entry["packed"]["model"])
    models = list(dict.fromkeys(models))
    sources = {}
    for model in models:
        sources[model] = _checked_file(root, model)
    with tempfile.TemporaryDirectory(dir=root, prefix=".streaming-") as staging_name:
        staging = Path(staging_name)
        records, replacements = {}, {}
        for index, (name, source) in enumerate(sources.items()):
            external = _external_hashes(root, source)
            model = onnx.load(source, load_external_data=True)
            precision = any(n.domain.startswith("fast.audiovae.precision.") for n in model.graph.node)
            if precision and not canonical_precision:
                raise ValueError("Precision recipes require --canonical-precision and a rebuilt streaming math library")
            if precision and name == manifest["fallback"]:
                raise ValueError("A precision recipe cannot be the portable fallback")
            if precision:
                from .graph.canonical import validate_precision_sine_backends
                validate_precision_sine_backends(model)
            derived, audit = rewrite_model(model)
            source_name, source_hash = name, sha256(source)
            if precision:
                from .graph.canonical import canonicalize_stem
                canonical, geometry = canonicalize_stem(model)
                derived, _ = canonicalize_stem(derived)
                onnx.external_data_helper.convert_model_from_external_data(canonical)
                full_name = f"decoder_{index}_full.onnx"
                onnx.save_model(canonical, staging / full_name)
                source_name = "streaming/" + full_name
                replacements[name] = (source_name, sha256(staging / full_name))
                audit.update(canonical_stem=geometry, original_full_call_model=name,
                             original_source_sha256=source_hash,
                             required_math_version=1, required_backend=5)
                source_hash = replacements[name][1]
                external = {}
            onnx.external_data_helper.convert_model_from_external_data(derived)
            target_name = f"decoder_{index}.onnx"
            target = staging / target_name
            onnx.save_model(derived, target)
            records[source_name] = {**audit, "model": "streaming/" + target_name,
                             "source_sha256": source_hash, "model_sha256": sha256(target),
                             "source_external_sha256": external}
        for entry in manifest["native"].values():
            for variant in [entry, *([entry["packed"]] if entry.get("packed") else [])]:
                if variant["model"] in replacements:
                    before = variant["model"]
                    variant["model"], variant["model_sha256"] = replacements[before]
                    variant.update(original_full_call_model=before, required_math_version=1,
                                   required_backend=5, canonical_precision=True)
        manifest["streaming"] = {"version": 1, "models": records}
        manifest["interface"] = "FP32 [1,64,L] to [1,1,1920*L] at 48000 Hz; full-call and stateful streaming"
        next_manifest = staging / "bundle.json"
        next_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        # Publish graphs first. Only the final atomic manifest change exposes
        # them to new readers; a failure leaves the old decoder usable.
        published = False
        try:
            destination.mkdir()
            published = True
            for graph in staging.glob("*.onnx"):
                os.replace(graph, destination / graph.name)
            os.replace(next_manifest, manifest_path)
        except BaseException:
            if published:
                shutil.rmtree(destination)
            raise
    return manifest["streaming"]
