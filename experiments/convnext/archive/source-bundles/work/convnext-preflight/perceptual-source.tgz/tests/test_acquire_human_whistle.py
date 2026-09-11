"""Small synthetic preparation fixtures, without network or accelerator use."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audiovae_student.acquire import _json_bytes
from audiovae_student.acquire_expressive import StorageBudget
from audiovae_student.acquire_fsd_vocal import clip_identity
from audiovae_student import acquire_human_whistle as module


def wav(*, frequency=400, stereo=False):
    mono = (0.1 * np.sin(np.arange(8000, dtype=np.float32) * (2 * np.pi * frequency / 16000)))
    samples = np.column_stack((mono, -mono)) if stereo else mono
    output = io.BytesIO()
    sf.write(output, samples, 16000, format="WAV", subtype="FLOAT")
    return output.getvalue(), mono


def selection(tmp_path, count=1):
    page = tmp_path / "evidence.html"
    page.write_bytes(b"synthetic source evidence fixture")
    rows = []
    for index in range(count):
        identifier = str(80000 + index)
        rows.append({"id": identifier, **clip_identity(identifier, "fixture-uploader"),
            "uploader": "fixture-uploader", "dataset": "freesound_human_whistle_cc0",
            "license": "CC0-1.0", "license_url": module.LICENSES["CC0-1.0"][1],
            "source_url": f"https://freesound.org/s/{identifier}/", "title": "Human whistle fixture",
            "duration_seconds": .5, "sample_rate_hz": 16000,
            "source_page_path": str(page), "source_page_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
            "mirror_metadata": {"freesound_id": int(identifier), "shard": "fixture.parquet",
                                "row_group": 0, "row_within_group": index}})
    return {"format_version": 1, "preparation_version": 1, "repo": module.REPO,
            "revision": module.REVISION, "selected": rows, "forbidden_fsd_clip_ids": [],
            "selection_policy": {"scope": "synthetic tests"}}


def digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def test_reviewed_selection_rejects_hash_license_and_forbidden_ids(tmp_path):
    value = selection(tmp_path)
    assert module.validate_selection(value, digest(value)) == digest(value)
    with pytest.raises(ValueError, match="SHA-256 changed"):
        module.validate_selection(value, "0" * 64)
    bad = copy.deepcopy(value)
    bad["selected"][0]["license"] = "CC-BY-NC-4.0"
    with pytest.raises(ValueError, match="Reviewed license"):
        module.validate_selection(bad, digest(bad))
    bad = copy.deepcopy(value)
    bad["forbidden_fsd_clip_ids"] = [bad["selected"][0]["id"]]
    with pytest.raises(ValueError, match="Forbidden"):
        module.validate_selection(bad, digest(bad))


def test_original_http_cc_link_is_retained_but_other_hosts_are_rejected(tmp_path):
    value = selection(tmp_path)
    value["selected"][0]["license_url"] = "http://creativecommons.org/publicdomain/zero/1.0/"
    module.validate_selection(value, digest(value))
    assert value["selected"][0]["license_url"].startswith("http://")
    value["selected"][0]["license_url"] = "http://creativecommons.org.evil.example/publicdomain/zero/1.0/"
    with pytest.raises(ValueError, match="Reviewed license"):
        module.validate_selection(value, digest(value))


def test_stereo_uses_unscaled_channel_zero_and_preserves_encoded_source(tmp_path):
    value = selection(tmp_path)
    payload, expected = wav(stereo=True)
    root = tmp_path / "prepared-source"
    budget = StorageBudget(root, 2_000_000, 0)
    row, receipt = module.prepare_clip(value["selected"][0], payload, root, budget, {}, digest(value))
    actual, rate = sf.read(row.audio_path, dtype="float32")
    assert rate == 16000
    np.testing.assert_array_equal(actual, expected)
    assert Path(receipt["original_path"]).read_bytes() == payload
    assert receipt["original_channels"] == 2
    assert "channel0" in receipt["channel_policy"]
    assert not row.enhanced and not row.native_recording
    assert not receipt["original_freesound_byte_identity_verified"]


def test_complete_resume_never_refetches_or_rewrites_final_audit(tmp_path):
    value = selection(tmp_path)
    payload, _ = wav()
    calls = []
    def fetch(item):
        calls.append(item["id"])
        return payload, {"fixture": True}
    kwargs = {"expected_selection_sha256": digest(value), "reserve_bytes": 0, "fetch": fetch}
    root = tmp_path / "source"
    first = module.acquire(value, root, **kwargs)
    marker = root / "prepared/provenance/complete.json"
    saved = marker.read_bytes()
    assert first["state"] == "complete" and first["rows"] == 1
    assert module.acquire(value, root, **kwargs) == first
    assert marker.read_bytes() == saved
    assert len(calls) == 1
    path = next((root / "prepared/audio").glob("*.wav"))
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.acquire(value, root, **kwargs)
    assert len(calls) == 1


def test_bad_clip_is_quarantined_durably_without_refetch(tmp_path):
    value = selection(tmp_path, 2)
    payload, _ = wav()
    calls = []
    def fetch(item):
        calls.append(item["id"])
        return (b"not audio" if item["id"] == "80000" else payload), {}
    root = tmp_path / "source"
    kwargs = {"expected_selection_sha256": digest(value), "reserve_bytes": 0, "fetch": fetch}
    first = module.acquire(value, root, **kwargs)
    assert first["rows"] == 1 and len(first["quarantined"]) == 1
    assert module.acquire(value, root, **kwargs) == first
    assert len(calls) == 2


def test_whole_file_guard_rejects_long_recording_before_any_writes(tmp_path, monkeypatch):
    value = selection(tmp_path)
    monkeypatch.setattr(module.sf, "info", lambda _: SimpleNamespace(channels=1, samplerate=16000,
                                                                    frames=180 * 16000 + 1))
    root = tmp_path / "source"
    with pytest.raises(module.ClipRejected, match="180-second"):
        module.prepare_clip(value["selected"][0], b"fixture", root, StorageBudget(root, 1000, 0), {}, digest(value))
    assert not list(root.rglob("*.wav"))


def test_rate_or_duration_mismatch_is_not_silently_resampled_or_trimmed(tmp_path):
    value = selection(tmp_path)
    payload, _ = wav()
    item = value["selected"][0]
    root = tmp_path / "source"
    budget = StorageBudget(root, 2_000_000, 0)
    item["sample_rate_hz"] = 48000
    with pytest.raises(module.ClipRejected, match="sample rate differs"):
        module.prepare_clip(item, payload, root, budget, {}, digest(value))
    item["sample_rate_hz"], item["duration_seconds"] = 16000, 2
    with pytest.raises(module.ClipRejected, match="duration differs"):
        module.prepare_clip(item, payload, root, budget, {}, digest(value))


def test_infrastructure_storage_failure_is_not_quarantined(tmp_path):
    value = selection(tmp_path)
    payload, _ = wav()
    root = tmp_path / "source"
    with pytest.raises(ValueError, match="storage byte cap"):
        module.prepare_clip(value["selected"][0], payload, root, StorageBudget(root, 100, 0), {}, digest(value))
