"""Offline packaging checks for the accepted x86 recipe, with no inference."""
import errno
import hashlib
import os
import io
import json
from pathlib import Path
import tarfile

import onnx
import pytest
from onnx import helper, TensorProto

from fast_audiovae.recipes import dependencies, x86


@pytest.mark.parametrize("vendor,mode,threads", [
    ("GenuineIntel", "streaming", 1), ("GenuineIntel", "batch", 1),
    ("GenuineIntel", "batch", 2), ("AuthenticAMD", "streaming", 1),
    ("AuthenticAMD", "streaming", 4), ("AuthenticAMD", "batch", 4),
])
def test_only_recorded_cpu_schedules(vendor, mode, threads):
    assert x86.validate_selection({"vendor": vendor}, mode, threads) in ("intel", "amd")


@pytest.mark.parametrize("info,mode,threads", [
    ({"vendor": "GenuineIntel"}, "streaming", 2),
    ({"vendor": "AuthenticAMD"}, "batch", 2),
    ({"vendor": "GenuineIntel"}, "batch", True),
    ({"vendor": "Apple"}, "streaming", 1),
    ({"vendor": "GenuineIntel"}, "automatic-tune", 1),
])
def test_unmeasured_schedule_is_not_silently_enabled(info, mode, threads):
    with pytest.raises(ValueError):
        x86.validate_selection(info, mode, threads)


def _record(root, name, data=b"artifact"):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def test_prebuilt_payload_hashes_and_paths(tmp_path):
    libraries = {name: _record(tmp_path, "libs/" + name + ".so") for name in
                 ("native", "core", "ops", "stage", "upsample", "fp32_stage")}
    manifest = {"schema_version": 1, "vendor": "intel", "libraries": libraries,
                "native_build": _record(tmp_path, "native-build.json", b"{}"),
                "runtime_files": [_record(tmp_path, "libs/dependency.so")], "license_files": []}
    assert x86._read_payload({"root": tmp_path, "manifest": manifest}, "intel")[1] == manifest
    (tmp_path / "libs/dependency.so").write_bytes(b"changed")
    with pytest.raises(ValueError):
        x86._read_payload({"root": tmp_path, "manifest": manifest}, "intel")
    with pytest.raises(ValueError, match="outside"):
        x86._payload_file(tmp_path, {"path": "../outside.so", "sha256": "0" * 64})


def test_relocated_payload_preserves_roles_and_notices(tmp_path):
    root, bundle = tmp_path / "payload", tmp_path / "bundle"
    root.mkdir(); bundle.mkdir()
    libraries = {name: _record(root, "libs/" + name + ".so") for name in
                 ("native", "core", "ops", "stage", "upsample", "fp32_stage", "pair")}
    manifest = {"libraries": libraries, "runtime_files": [_record(root, "libs/dep.so")],
                "license_files": [_record(root, "licenses/vendor/LICENSE.txt", b"terms")]}
    base, additional = x86._payload_libraries((root, manifest), bundle)
    assert base["library"] == "libs/native.so"
    assert [x["library"] for x in additional] == ["libs/fp32_stage.so", "libs/ops.so", "libs/stage.so", "libs/upsample.so"]
    assert (bundle / "libs/dep.so").read_bytes() == b"artifact"
    assert (bundle / "licenses/vendor/LICENSE.txt").read_bytes() == b"terms"


def _schedule_graph():
    matrix = "fast.audiovae.precision.matrix.experimental"
    stage = "fast.audiovae.precision.stage.experimental"
    nodes = []
    previous = "x"
    for i in range(14):
        name = "matrix_" + str(i)
        nodes.append(helper.make_node("PrecisionMatMulF32", ["w", previous], [name], domain=matrix,
                                      name=name, shards=2, backend=1, precision_mode=8, native_abi=1, M=1, K=1))
        previous = name
    for i in range(4):
        name = "stage_" + str(i)
        nodes.append(helper.make_node("StageStackF32", [previous], [name], domain=stage,
                                      name=name, segments=2, channels=128, tile_time=256))
        previous = name
    graph = helper.make_graph(nodes, "schedule", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1])],
                              [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, 1, 1])],
                              [helper.make_tensor("w", TensorProto.FLOAT, [1, 1], [0.125])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18),
                              helper.make_opsetid(matrix, 1), helper.make_opsetid(stage, 1)])


def test_reschedule_changes_only_the_eighteen_worker_fields(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.onnx", tmp_path / "one.onnx"
    original = _schedule_graph()
    onnx.save(original, source)
    monkeypatch.setattr(x86, "_verify", lambda *args: None)
    changes = x86.reschedule(source, destination, 1)
    assert len(changes) == 18
    result = onnx.load(destination)
    assert [x.SerializeToString() for x in original.graph.initializer] == [x.SerializeToString() for x in result.graph.initializer]
    for before, after in zip(original.graph.node, result.graph.node):
        for attr in after.attribute:
            if attr.name in ("shards", "segments"):
                assert attr.i == 1
                attr.i = 2
        assert after.SerializeToString() == before.SerializeToString()


def test_archive_extraction_allows_only_contained_documentation_link(tmp_path):
    archive = tmp_path / "sources.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        file = tarfile.TarInfo("source/LICENSE")
        file.size = 5
        out.addfile(file, io.BytesIO(b"terms"))
        link = tarfile.TarInfo("source/docs/LICENSE")
        link.type = tarfile.SYMTYPE
        link.linkname = "../LICENSE"
        out.addfile(link)
    source = dependencies.unpack(archive, tmp_path / "source")
    assert (source / "docs/LICENSE").read_bytes() == b"terms"
    assert dependencies.unpack(archive, source) == source


def test_archive_traversal_fails_and_removes_partial_source(tmp_path):
    archive = tmp_path / "sources.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        member = tarfile.TarInfo("source/../../escaped")
        member.size = 1
        out.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "source"
    with pytest.raises(ValueError):
        dependencies.unpack(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escaped").exists()


def test_cpu_build_environment_does_not_mutate_parent(monkeypatch):
    monkeypatch.setenv("MKL_ENABLE_INSTRUCTIONS", "OTHER")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    env = dependencies.cpu_environment()
    assert "MKL_ENABLE_INSTRUCTIONS" not in env
    assert env["OMP_NUM_THREADS"] == "1"
    assert __import__("os").environ["OMP_NUM_THREADS"] == "99"


def test_prebuilt_pipeline_never_runs_toolchain_or_dependency_downloads(tmp_path, monkeypatch):
    # Model graph generation is mocked here; its exact hash gates are tested
    # separately. This checks the zero-toolchain end-user installation boundary.
    work, payload, source = tmp_path / "work", tmp_path / "payload", tmp_path / "model.onnx"
    payload.mkdir(); source.write_bytes(b"source")
    libraries = {name: _record(payload, "libs/" + name + ".so") for name in
                 ("native", "core", "ops", "stage", "upsample", "fp32_stage")}
    native_record = {"library_sha256": libraries["native"]["sha256"]}
    manifest = {"schema_version": 1, "vendor": "amd", "libraries": libraries,
                "native_build": _record(payload, "native-build.json", json.dumps(native_record).encode()),
                "runtime_files": [], "license_files": []}
    monkeypatch.setattr(x86, "_host_gate", lambda *args: None)
    monkeypatch.setattr(x86, "verify_model", lambda *args: None)
    monkeypatch.setattr(x86.shutil, "which", lambda *args: pytest.fail("No toolchain probing for a prebuilt recipe"))
    monkeypatch.setattr(x86.subprocess, "check_output", lambda *args, **kw: pytest.fail("No native tool commands for a prebuilt recipe"))
    monkeypatch.setattr(x86.deps, "mkl", lambda *args, **kw: pytest.fail("No dependency provisioner"))
    monkeypatch.setattr(x86.deps, "aocl", lambda *args, **kw: pytest.fail("No dependency provisioner"))
    monkeypatch.setattr(x86.deps, "libxsmm", lambda *args, **kw: pytest.fail("No dependency provisioner"))
    import fast_audiovae.prepare as prepare_module
    def fake_prepare(destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "decoder_portable.onnx").write_bytes(b"portable")
        (destination / "decoder_native.onnx").write_bytes(b"native")
        (destination / "bundle.json").write_text(json.dumps({"fallback": "decoder_portable.onnx"}))
    monkeypatch.setattr(prepare_module, "prepare", fake_prepare)
    monkeypatch.setattr(x86, "assemble_graphs", lambda *args: source)
    def fake_bundle(*args, **kwargs):
        destination = args[8]
        destination.mkdir(parents=True)
        (destination / "decoder_portable.onnx").write_bytes(b"portable")
        (destination / "bundle.json").write_text(json.dumps({"fallback": "decoder_portable.onnx"}))
    monkeypatch.setattr(x86, "_bundle", fake_bundle)
    result = x86.build_recipe(work, source, {"vendor": "AMD", "offline": True,
                              "prebuilt": {"root": payload, "manifest": manifest}}, "batch", 1)
    record = json.loads((result / "bundle.json").read_text())
    assert record["automatic_recipe"]["prebuilt"] is True
    assert result.name == "amd-batch-1"
    # A completed bundle survives a crash before the outer setup receipt is
    # published, without rebuilding or attempting to invoke a compiler.
    assert x86.build_recipe(work, source, {"vendor": "AMD", "offline": True,
                              "prebuilt": {"root": payload, "manifest": manifest}}, "batch", 1) == result


def test_immutable_library_copy_reuses_storage(tmp_path):
    source = tmp_path / "payload/libcpu.so"
    source.parent.mkdir()
    source.write_bytes(b"verified immutable CPU library")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    record = x86._copy_library(source, bundle)
    target = bundle / source.name
    assert source.stat().st_ino == target.stat().st_ino
    assert record["sha256"] == x86.sha256(source) == x86.sha256(target)
    assert x86._copy_library(source, bundle) == record


@pytest.mark.parametrize("code", [errno.EXDEV, errno.ENOTSUP, errno.EPERM])
def test_immutable_library_copy_falls_back_when_link_unavailable(tmp_path, monkeypatch, code):
    source = tmp_path / "libcpu.so"
    source.write_bytes(b"verified CPU library")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    def unavailable(*args):
        raise OSError(code, "hardlink unavailable")
    monkeypatch.setattr(x86.os, "link", unavailable)
    x86._copy_library(source, bundle)
    target = bundle / source.name
    assert target.read_bytes() == source.read_bytes()
    assert source.stat().st_ino != target.stat().st_ino


def test_library_storage_error_is_not_silently_retried(tmp_path, monkeypatch):
    source = tmp_path / "libcpu.so"
    source.write_bytes(b"CPU library")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    def full(*args):
        raise OSError(errno.ENOSPC, "no space")
    monkeypatch.setattr(x86.os, "link", full)
    with pytest.raises(OSError) as error:
        x86._copy_library(source, bundle)
    assert error.value.errno == errno.ENOSPC
    assert not (bundle / source.name).exists()


@pytest.mark.parametrize("checkpoint", ["composed", "accepted", "selected"])
def test_graph_checkpoint_retry_finishes_cleanup_and_preserves_audits(tmp_path, monkeypatch, checkpoint):
    original = tmp_path / "original.onnx"
    original.write_bytes(b"original")
    output = tmp_path / "graphs"
    output.mkdir()
    contents = {"composed": b"composed", "accepted": b"accepted", "selected": b"selected"}
    for key, constant in (("composed", "COMPOSED_SHA256"), ("accepted", "ACCEPTED_SHA256"),
                          ("selected", "SELECTIVE_SHA256")):
        monkeypatch.setattr(x86, constant, hashlib.sha256(contents[key]).hexdigest())
    monkeypatch.setattr(x86, "ORIGINAL_SHA256", x86.sha256(original))
    for name in x86._COMPOSITION_INPUTS:
        (output / name).write_bytes(b"obsolete graph")
    paths = {"composed": "composed.onnx", "accepted": "accepted.onnx", "selected": "precision/int8_large.onnx"}
    path = output / paths[checkpoint]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents[checkpoint])
    audit = output / "composed.compose.json"
    audit.write_bytes(b'{"proof": "retain exactly"}')
    if checkpoint == "selected":
        (output / "precision/manifest.json").write_text("{}")
        (output / "precision/fp16_all.onnx").write_bytes(b"unselected")
        (output / "accepted.onnx").write_bytes(contents["accepted"])
    calls = []
    def writer(work, relative, args, commands):
        calls.append(relative)
        if "upsample" in relative:
            (output / "accepted.onnx").write_bytes(contents["accepted"])
            (output / "accepted.upsample.json").write_text("{}")
        else:
            assert "intel-precision" in relative
            precision = output / "precision"
            precision.mkdir(exist_ok=True)
            (precision / "int8_large.onnx").write_bytes(contents["selected"])
            for name in ("int8_all.onnx", "fp16_all.onnx"):
                (precision / name).write_bytes(b"unselected")
            (precision / "manifest.json").write_text("{}")
    monkeypatch.setattr(x86, "_script", writer)
    discard = x86._discard_graphs
    def interrupted(directory, names):
        discard(directory, names[:1])
        raise OSError("interrupted during cleanup")
    monkeypatch.setattr(x86, "_discard_graphs", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        x86.assemble_graphs(tmp_path, original, output, [])
    assert calls == []
    monkeypatch.setattr(x86, "_discard_graphs", discard)
    selected = x86.assemble_graphs(tmp_path, original, output, [])
    assert selected.read_bytes() == contents["selected"]
    assert len(calls) == {"composed": 2, "accepted": 1, "selected": 0}[checkpoint]
    assert {p.relative_to(output).as_posix() for p in output.rglob("*.onnx")} == {"precision/int8_large.onnx"}
    assert audit.read_bytes() == b'{"proof": "retain exactly"}'
    assert original.read_bytes() == b"original"
    assert not list(tmp_path.glob("graphs.incomplete-*"))
    previous_calls = len(calls)
    assert x86.assemble_graphs(tmp_path, original, output, []) == selected
    assert len(calls) == previous_calls


def test_bad_checkpoint_is_not_used_to_delete_predecessors(tmp_path, monkeypatch):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"source")
    monkeypatch.setattr(x86, "ORIGINAL_SHA256", x86.sha256(source))
    output = tmp_path / "graphs"
    output.mkdir()
    (output / "base.onnx").write_bytes(b"keep")
    (output / "composed.onnx").write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="differs"):
        x86.assemble_graphs(tmp_path, source, output, [])
    assert (output / "base.onnx").read_bytes() == b"keep"


def test_pair_refuses_to_rewrite_published_bundle(tmp_path):
    (tmp_path / ".recipe-ready.json").write_text("{}")
    with pytest.raises(RuntimeError, match="published"):
        x86._pair(tmp_path, tmp_path, [])


def test_intel_pair_stays_unpublished_on_failure_and_retries(tmp_path, monkeypatch):
    work, payload, source = tmp_path / "work", tmp_path / "payload", tmp_path / "model.onnx"
    payload.mkdir()
    source.write_bytes(b"source")
    libraries = {name: _record(payload, "libs/" + name + ".so") for name in
                 ("native", "core", "ops", "stage", "upsample", "fp32_stage", "pair")}
    metadata = {"library_sha256": libraries["native"]["sha256"]}
    manifest = {"schema_version": 1, "vendor": "intel", "libraries": libraries,
                "native_build": _record(payload, "native-build.json", json.dumps(metadata).encode()),
                "runtime_files": [], "license_files": []}
    info = {"vendor": "Intel", "prebuilt": {"root": payload, "manifest": manifest}}
    monkeypatch.setattr(x86, "_host_gate", lambda *args: None)
    monkeypatch.setattr(x86, "verify_model", lambda *args: None)
    import fast_audiovae.prepare as prepare_module
    def canonical(destination):
        destination.mkdir(parents=True)
        for name in ("decoder_portable.onnx", "decoder_native.onnx", "stream.onnx"):
            (destination / name).write_bytes(name.encode())
        (destination / "bundle.json").write_text(json.dumps({"fallback": "decoder_portable.onnx"}))
    monkeypatch.setattr(prepare_module, "prepare", lambda destination, **kw: canonical(destination))
    monkeypatch.setattr(x86, "assemble_graphs", lambda *args: source)
    monkeypatch.setattr(x86, "_bundle", lambda *args, **kw: canonical(args[8]))
    final = work / "bundles/intel-streaming-1"
    def failing_pair(work_dir, staged, commands, **kwargs):
        assert not final.exists()
        assert staged.parent.name.startswith(".recipe-")
        assert not (staged / ".recipe-ready.json").exists()
        (staged / "stream.onnx").write_bytes(b"partially rewritten")
        raise RuntimeError("pair interrupted")
    monkeypatch.setattr(x86, "_pair", failing_pair)
    with pytest.raises(RuntimeError, match="pair interrupted"):
        x86.build_recipe(work, source, info, "streaming", 1)
    assert not final.exists()
    assert not (work / ".build/automatic-intel-canonical-1").exists()
    original = work / ".build/automatic-original"
    original_files = {p.name: x86.sha256(p) for p in original.iterdir() if p.is_file()}
    def successful_pair(work_dir, staged, commands, **kwargs):
        assert not final.exists()
        assert (staged / "stream.onnx").read_bytes() == b"stream.onnx"
        (staged / "stream.onnx").write_bytes(b"paired")
    monkeypatch.setattr(x86, "_pair", successful_pair)
    assert x86.build_recipe(work, source, info, "streaming", 1) == final
    assert (final / "stream.onnx").read_bytes() == b"paired"
    assert (final / "decoder_native.onnx").read_bytes() == b"decoder_native.onnx"
    assert original_files == {p.name: x86.sha256(p) for p in original.iterdir() if p.is_file()}
    monkeypatch.setattr(x86, "_pair", lambda *args, **kw: pytest.fail("completed pair was rebuilt"))
    assert x86.build_recipe(work, source, info, "streaming", 1) == final
