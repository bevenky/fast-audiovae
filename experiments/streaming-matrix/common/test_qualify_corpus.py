"""Qualification accounting with synthetic arrays, never neural inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import qualify_corpus as qualify


class FakeSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_session_options(self):
        class Options:
            intra_op_num_threads = 1
            inter_op_num_threads = 1
            execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return Options()


class FakeAdapter:
    roles = []

    def __init__(self, model_dir, *, role, threads=1, fault=None):
        assert threads == 1
        self.role, self.fault = role, fault
        self.roles.append(role)
        self.metadata = {"name": role, "codec": "audiovae2", "sample_rate": 1000,
            "hop_samples": 40, "latent_fps": 25, "channels": 2, "artifacts": []}
        self.sessions = [FakeSession()]
        self.streams = []

    def full(self, z):
        assert self.role == "accepted_full", "Candidate full session must never run"
        return np.repeat(z[:, :1, :], 40, axis=-1)

    def stream(self):
        assert self.role == "candidate_stream", "Accepted streaming session must never run"
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream


class FakeStream:
    def __init__(self, adapter):
        self.adapter, self.closed, self.index = adapter, False, 0

    def decode_chunk(self, z):
        self.index += 1
        if self.adapter.fault == "exception" and self.index == 2:
            raise RuntimeError("broken-second-chunk")
        value = np.repeat(z[:, :1, :], 40, axis=-1)
        if self.adapter.fault == "missing":
            value = value[:, :, :-1]
        elif self.adapter.fault == "wrong":
            value += np.float32(1)
        return value

    def flush(self):
        return np.zeros((1, 1, 1 if self.adapter.fault == "flush" else 0), np.float32)

    def close(self):
        self.closed = True


def latent(frames=7):
    return np.arange(1, 2 * frames + 1, dtype=np.float32).reshape(1, 2, frames)


class AccountingTests(unittest.TestCase):
    def test_whole_clip_and_tail(self):
        adapter, z = FakeAdapter("candidate", role="candidate_stream"), latent()
        reference = np.repeat(z[:, :1, :], 40, axis=-1)
        for frames, counts in ((1, [40] * 7 + [0]), (2, [80, 80, 80, 40, 0]), (4, [160, 120, 0])):
            row = qualify.check_stream(adapter, z, reference, reference, uid="a", frames=frames, exact=True)
            self.assertTrue(row["passed"])
            self.assertEqual([r["returned_samples"] for r in row["calls"]], counts)
            self.assertEqual(row["calls"][-2]["end_sample"], 280)
            self.assertEqual(row["calls"][-1]["kind"], "flush")
            self.assertTrue(row["calls"][0]["first"])
            self.assertTrue(row["calls"][-2]["final"])
        self.assertTrue(all(s.closed for s in adapter.streams))

    def test_count_exception_and_parity_failures_are_retained(self):
        z, reference = latent(), np.repeat(latent()[:, :1, :], 40, axis=-1)
        for fault in ("missing", "wrong", "flush", "exception"):
            adapter = FakeAdapter("candidate", role="candidate_stream", fault=fault)
            row = qualify.check_stream(adapter, z, reference, None, uid="a", frames=2, exact=True)
            self.assertFalse(row["passed"])
            self.assertIn("error", row)
            self.assertTrue(adapter.streams[0].closed)
            if fault == "exception":
                self.assertEqual(len(row["calls"]), 2)
                self.assertEqual(row["calls"][0]["returned_samples"], 80)
                self.assertIsNone(row["calls"][1]["returned_samples"])
                self.assertIn("broken-second-chunk", row["calls"][1]["error"])

    def test_upstream_gate_is_strict_on_apple_only(self):
        adapter, z = FakeAdapter("candidate", role="candidate_stream"), latent()
        reference = np.repeat(z[:, :1, :], 40, axis=-1)
        stored = reference + np.float32(1)
        intel = qualify.check_stream(adapter, z, reference, stored, uid="a", frames=2, exact=True)
        apple = qualify.check_stream(adapter, z, reference, stored, uid="a", frames=2, exact=False)
        self.assertTrue(intel["passed"])
        self.assertFalse(intel["checks"]["stored_upstream"]["gate"])
        self.assertFalse(apple["passed"])
        self.assertTrue(apple["checks"]["stored_upstream"]["gate"])

    def test_whole_cohort_no_implicit_subset(self):
        cases = [{"uid": str(i)} for i in range(60)]
        keys = [str(i) + "__z" for i in range(60)]
        self.assertEqual(len(qualify.validate_cohort({"cases": cases}, keys)), 60)
        for invalid in (cases[:3], cases + [cases[0]], cases[:59] + [cases[0]]):
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                qualify.validate_cohort({"cases": invalid}, keys)
        for invalid in (keys[:59], keys + ["extra__z"]):
            with self.assertRaisesRegex(ValueError, "archive cohort"):
                qualify.validate_cohort({"cases": cases}, invalid)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cases, archive = [], {}
        for index in range(60):
            uid, z = f"clip{index:02d}", latent(3)
            archive[uid + "__z"] = z
            archive[uid + "__ref"] = np.repeat(z[:, :1, :], 40, axis=-1)
            self.cases.append({"uid": uid, "latent_shape": [1, 2, 3], "reference_samples": 120})
        np.savez(self.root / "latents.npz", **archive)
        (self.root / "latents.json").write_text(json.dumps({"cases": self.cases}))
        self.config = {"baseline_bundle": "baseline", "candidate_bundle": "candidate", "exact": True,
            "latents": str(self.root / "latents.npz"), "latent_manifest": str(self.root / "latents.json"),
            "clip_ids": ["clip00"]}  # Must not silently narrow qualification.

    def tearDown(self):
        self.temp.cleanup()

    def test_all_60_and_only_two_required_session_roles(self):
        FakeAdapter.roles = []
        with patch.object(qualify.core, "checked_affinity", return_value=[0]):
            result = qualify.run(self.config, self.root / "result.json", factory=FakeAdapter,
                                 artifact_reader=lambda files: ({}, []))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["artifacts_unchanged"])
        self.assertEqual(FakeAdapter.roles, ["accepted_full", "candidate_stream"])
        self.assertEqual(len(result["references"]), 60)
        self.assertEqual(len(result["runs"]), 180)
        self.assertEqual(result["cohort"]["stored_upstream_references"], 60)

    def test_failure_persists_partial_attempt(self):
        def factory(model_dir, **kwargs):
            return FakeAdapter(model_dir, fault="missing" if model_dir == "candidate" else None, **kwargs)
        with patch.object(qualify.core, "checked_affinity", return_value=[0]):
            with self.assertRaisesRegex(RuntimeError, "Corpus qualification failure"):
                qualify.run(self.config, self.root / "result.json", factory=factory, artifact_reader=lambda files: ({}, []))
        result = json.loads((self.root / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 1)
        self.assertFalse(result["runs"][0]["passed"])
        self.assertEqual(result["runs"][0]["calls"][0]["returned_samples"], 39)
        self.assertEqual(result["active_clip"], "clip00")

    def test_source_change_fails_closed(self):
        calls = []
        def artifacts(files):
            calls.append(files)
            return {"source": {"sha256": "first" if len(calls) < 3 else "changed", "bytes": 1}}, []
        with patch.object(qualify.core, "checked_affinity", return_value=[0]):
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                qualify.run(self.config, self.root / "result.json", factory=FakeAdapter, artifact_reader=artifacts)
        result = json.loads((self.root / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
