"""Synthetic protocol tests only. These never load or run a neural model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import paired_benchmark as paired


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
    def __init__(self, model_dir="baseline", *, optimized=True, threads=1, fault=None):
        assert threads == 1 and optimized
        self.metadata = {"name": model_dir, "codec": "audiovae2", "sample_rate": 1000,
                         "hop_samples": 40, "latent_fps": 25, "channels": 2,
                         "exact_streaming": True, "artifacts": []}
        self.sessions = [FakeSession()]
        self.fault = fault
        self.streams = []

    def full(self, z):
        value = np.repeat(z[:, :1, :], 40, axis=-1)
        if self.fault == "full_and_stream_wrong":
            value += np.float32(1)
        return value

    def stream(self):
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream


class FakeStream:
    def __init__(self, adapter):
        self.adapter = adapter
        self.closed = False
        self.reset()

    def reset(self):
        self.frames = 0

    def decode_chunk(self, z):
        self.frames += z.shape[-1]
        value = self.adapter.full(z)
        if self.adapter.fault == "missing":
            value = value[:, :, :-1]
        return value

    def flush(self):
        return np.zeros((1, 1, 0), np.float32)

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += .001
        return self.value


def latent(frames=13):
    return np.arange(1, 2 * frames + 1, dtype=np.float32).reshape(1, 2, frames)


class ProtocolTests(unittest.TestCase):
    def test_schedule_adjacent_and_reproducible(self):
        schedule = paired.paired_schedule(["a", "b", "c"])
        self.assertEqual(schedule, paired.paired_schedule(["a", "b", "c"]))
        self.assertEqual(len(schedule), 126)
        self.assertEqual(sum(row["phase"] == "measure" for row in schedule), 90)
        for first, second in zip(schedule[::2], schedule[1::2]):
            self.assertEqual(first["pair_id"], second["pair_id"])
            self.assertEqual({first["adapter"], second["adapter"]}, {"baseline", "candidate"})
            self.assertEqual([first["pair_position"], second["pair_position"]], [0, 1])

    def test_exact_counts_and_timing_reuse_audited_core(self):
        adapter, z = FakeAdapter(), latent(7)
        row = paired.core.run_once(adapter, z, adapter.full(z), "stream_80", uid="a", repeat=0,
                                   phase="measure", clock=Clock())
        self.assertTrue(row["check"]["passed"])
        self.assertEqual([c["returned_samples"] for c in row["calls"]], [80, 80, 80, 40, 0])
        self.assertEqual([(c["start_frame"], c["end_frame"]) for c in row["calls"]],
                         [(0, 2), (2, 4), (4, 6), (6, 7), (7, 7)])
        self.assertAlmostEqual(row["sum_call_seconds"], .005)
        self.assertAlmostEqual(row["sum_call_rtf"], .005 / .28)
        self.assertGreater(row["whole_loop_seconds"], row["sum_call_seconds"])

    def test_missing_sample_rejected(self):
        adapter, z = FakeAdapter(fault="missing"), latent(7)
        row = paired.core.run_once(adapter, z, FakeAdapter().full(z), "stream_80", uid="a", repeat=0,
                                   phase="measure", clock=Clock())
        self.assertFalse(row["check"]["passed"])
        self.assertIn("Invalid waveform", row["error"])

    def test_candidate_cannot_self_reference(self):
        z, baseline = latent(7), FakeAdapter()
        candidate = FakeAdapter("candidate", fault="full_and_stream_wrong")
        for mode in ("full", "stream_80", "stream_160"):
            row = paired.core.run_once(candidate, z, baseline.full(z), mode, uid="a", repeat=0,
                                       phase="measure", clock=Clock())
            self.assertFalse(row["check"]["passed"])

    def test_state_modes_and_failure_recording(self):
        adapter = FakeAdapter()
        rows = paired.state_checks(adapter, FakeAdapter(), latent(), exact=True, uid="a")
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["check"]["passed"] for row in rows))
        self.assertTrue(all(stream.closed for stream in adapter.streams))
        rows = []
        with self.assertRaisesRegex(RuntimeError, "State/parity"):
            paired.state_checks(FakeAdapter(fault="full_and_stream_wrong"), FakeAdapter(), latent(),
                                exact=True, uid="a", record_sink=rows.append)
        self.assertFalse(rows[-1]["check"]["passed"])

    def test_exact_distinguishes_signed_zero_and_apple_tolerance(self):
        ref = np.zeros((1, 1, 80), np.float32)
        value = np.full_like(ref, -0.0)
        self.assertFalse(paired.waveform_check(value, ref, True)["passed"])
        self.assertTrue(paired.waveform_check(value, ref, False)["passed"])
        value[:] = np.float32(1e-6)
        self.assertFalse(paired.waveform_check(value, ref, True)["passed"])
        self.assertTrue(paired.waveform_check(value, ref, False)["passed"])

    def test_worker_check(self):
        record = {"node": "kernel", "attributes": {"shards": 1, "segments": 1}}
        paired.require_one_worker([record])
        for value in (2, 0, [1], "1"):
            with self.assertRaisesRegex(ValueError, "one-worker"):
                paired.require_one_worker([dict(record, attributes={"shards": value})])

    def test_pair_summary_excludes_warmup_failed_and_incomplete(self):
        rows = []
        for uid, seconds in (("a", 1.), ("b", 2.)):
            for repeat in range(5):
                for name, multiplier in (("baseline", 1.), ("candidate", .8)):
                    rows.append({"pair_id": uid + str(repeat), "uid": uid, "adapter": name,
                        "mode": "stream_80", "phase": "measure", "check": {"passed": True},
                        "sum_call_seconds": seconds * multiplier, "sum_call_rtf": seconds * multiplier / 10})
        rows.append(dict(rows[0], pair_id="incomplete"))
        rows.append(dict(rows[0], phase="warmup", pair_id="warmup"))
        rows.append(dict(rows[0], check={"passed": False}, pair_id="failed"))
        result = paired.summarize_pairs(rows)[0]
        self.assertEqual(result["complete_pairs"], 10)
        self.assertAlmostEqual(result["aggregate_reduction_percent"], 20)
        self.assertAlmostEqual(result["median_paired_reduction_percent"], 20)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        np.savez(self.root / "latents.npz", frozen__z=latent())
        (self.root / "latents.json").write_text(json.dumps({"cases": [{"uid": "frozen",
            "latent_shape": [1, 2, 13], "source_duration_s": .52}]}))
        self.config = {"baseline_bundle": "baseline", "candidate_bundle": "candidate", "exact": True,
            "latents": str(self.root / "latents.npz"), "latent_manifest": str(self.root / "latents.json"),
            "clip_ids": ["frozen"]}

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_and_immutable(self):
        with patch.object(paired.core, "checked_affinity", return_value=[0]):
            result = paired.run(self.config, self.root / "result.json", factory=FakeAdapter,
                                artifact_reader=lambda paths: ({}, []))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["artifacts_unchanged"])
        self.assertEqual(len(result["runs"]), 42)
        self.assertEqual(len(result["state_checks"]), 24)
        self.assertTrue(all(r["reference_adapter"] == "baseline" for r in result["runs"]))

    def test_candidate_error_incrementally_preserved(self):
        def factory(model_dir, **kwargs):
            return FakeAdapter(model_dir, fault="full_and_stream_wrong" if model_dir == "candidate" else None, **kwargs)
        self.config["skip_state_checks"] = True
        with patch.object(paired.core, "checked_affinity", return_value=[0]):
            with self.assertRaisesRegex(RuntimeError, "Accepted-reference"):
                paired.run(self.config, self.root / "result.json", factory=factory, artifact_reader=lambda paths: ({}, []))
        result = json.loads((self.root / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["runs"][-1]["check"]["passed"])
        self.assertTrue(result["artifacts_unchanged"])

    def test_artifact_mutation_rejected(self):
        calls = []
        def artifacts(paths):
            calls.append(paths)
            return {"frozen": {"sha256": "before" if len(calls) < 3 else "after", "bytes": 1}}, []
        with patch.object(paired.core, "checked_affinity", return_value=[0]):
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                paired.run(self.config, self.root / "result.json", factory=FakeAdapter, artifact_reader=artifacts)
        result = json.loads((self.root / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
