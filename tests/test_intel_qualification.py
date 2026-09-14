"""Failures in the release checker must reject parity rather than hide gaps."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("intel_qualification", ROOT / "tools/qualify_intel_streaming.py")
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def result():
    state = {"state" + str(index): {"sha256": str(index), "shape": [1, 2, 3], "dtype": "float32"}
             for index in range(18)}
    wave = {"sha256": "audio", "shape": [1, 1, 1920], "dtype": "float32"}
    return {"audio": wave, "state": state, "samples": 1920,
            "packets": [{"frames": 1, "samples": 1920, "audio": wave, "state": state}]}


@pytest.mark.parametrize("mutation", ["waveform", "state", "state_inventory", "packet", "sample_count"])
def test_mismatch_cannot_pass_installed_wheel_gate(mutation):
    reference = result(); actual = copy.deepcopy(reference)
    if mutation == "waveform":
        actual["audio"]["sha256"] = "different"
    elif mutation == "state":
        actual["state"]["state0"]["sha256"] = "different"
    elif mutation == "state_inventory":
        actual["state"].pop("state17")
    elif mutation == "packet":
        actual["packets"] = []
    else:
        actual["samples"] -= 1
    with pytest.raises(RuntimeError):
        qualification.compare(reference, actual, True)


def test_wave_and_all_eighteen_state_checks_are_counted():
    reference = result()
    answer = qualification.compare(reference, copy.deepcopy(reference), True)
    assert answer == {"pass": True, "packets": 1, "state_arrays": 18,
                      "waveform_scope": "each_packet_and_full", "state_scope": "every_packet"}


def test_reference_artifact_tampering_is_rejected(tmp_path):
    path = tmp_path / "library.so"; path.write_bytes(b"fixture")
    records = [{"path": str(path), "sha256": qualification.sha(path)}]
    assert qualification.verify_records(records) == [str(path)]
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from its pin"):
        qualification.verify_records(records)


def test_external_weight_file_must_be_in_pinned_reference_inventory(tmp_path):
    weight = numpy_helper.from_array(np.ones((2, 2), np.float32), "weight")
    graph = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["y"])], "external",
        [], [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])], [weight]))
    path = tmp_path / "decoder.onnx"
    onnx.save_model(graph, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    with pytest.raises(RuntimeError, match="lacks an explicit reference pin"):
        qualification.verify_external_weights(path, [])
    weights = tmp_path / "weights.bin"
    records = [{"path": str(weights), "sha256": qualification.sha(weights)}]
    qualification.verify_records(records)
    qualification.verify_external_weights(path, records)


def test_mapping_failure_reports_missing_and_observed_paths(tmp_path, monkeypatch):
    expected = tmp_path / "bundle/libs/libintel_stream_matrix_ops.so"
    observed = tmp_path / "old-prebuilt/libintel_stream_matrix_ops.so"
    native_map = f"7000-8000 r-xp 00000000 00:01 12 {observed}\n"
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: native_map if str(path) == "/proc/self/maps" else original(path, *a, **kw))
    with pytest.raises(RuntimeError) as error:
        qualification.mapped_native(tmp_path / "bundle", [expected])
    message = str(error.value)
    assert 'missing=' in message and 'mapped=' in message
    assert str(expected) in message and str(observed) in message


def test_mapping_diagnostic_does_not_accept_equal_bytes_outside_bundle(tmp_path, monkeypatch):
    expected = tmp_path / "bundle/libs/libintel_stream_matrix_ops.so"
    observed = tmp_path / "outside/libintel_stream_matrix_ops.so"
    for path in (expected, observed):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"same bytes, different inode")
    original = Path.read_text
    native_map = f"7000-8000 r-xp 00000000 00:01 12 {observed}\n"
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: native_map if str(path) == "/proc/self/maps" else original(path, *a, **kw))
    with pytest.raises(RuntimeError, match="absent from process maps"):
        qualification.mapped_native(tmp_path / "bundle", [expected])


@pytest.mark.parametrize("reuse", [False, True])
def test_cache_reuse_flag_reaches_worker_and_report(tmp_path, monkeypatch, reuse):
    import io
    import json
    from types import SimpleNamespace
    source = tmp_path / "source.onnx"; source.write_bytes(b"model")
    latents = tmp_path / "latents.npz"; latents.write_bytes(b"not loaded by parent")
    reference = tmp_path / "reference"; reference.mkdir()
    (reference / "bundle.json").write_text("{}")
    args = SimpleNamespace(source=source, latents=latents, reference_bundle=reference, reference_config=None,
                           cache=tmp_path / "cache", output=tmp_path / "result", reuse_cache=reuse)
    calls = []
    class InertWorker:
        # Stop at initialization: this test exercises only the parent/child CLI contract.
        stdout = io.StringIO('{"ready":false}\n')
        stdin = io.StringIO()
        def poll(self):
            return 0
    def launch(command, **kwargs):
        calls.append(command)
        return InertWorker()
    monkeypatch.setattr(qualification, "cpu_environment", lambda: None)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(qualification.subprocess, "Popen", launch)
    monkeypatch.setattr(qualification.select, "select", lambda reads, writes, errors, timeout: (reads, [], []))
    assert qualification.main(args) == 1
    assert len(calls) == 1 and ("--reuse-cache" in calls[0]) is reuse
    report = json.loads((args.output / "result.json").read_text())
    assert report["reuse_cache_requested"] is reuse
    assert report["error"] == "Worker initialization failed"
    assert report["native_calls"] == 0
