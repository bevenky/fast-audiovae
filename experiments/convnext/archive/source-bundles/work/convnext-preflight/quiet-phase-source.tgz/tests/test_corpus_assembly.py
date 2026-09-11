from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audiovae_student.acquire import FLEURS_REVISION
from audiovae_student.acquire_indic import LANGUAGES
from audiovae_student.corpus_assembly import FrozenCorpusError, assemble_corpus
from audiovae_student.data import ManifestRow, load_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.access = root / "access.json"
        write_json(self.access, {"policy": "synthetic unchanged mono16k fixture"})
        self.counter = 0
        self.fleurs = self.row("fleurs", "fr_fr:train:fr.wav", language="fr_fr", filename="fr.wav")
        self.libri = self.row("librispeech", "libri-core", language="en_US")
        self.indic = [self.row("indicvoices", "indic-" + code, language=code) for code in LANGUAGES.values()]
        self.expressive = self.row("thorsten_emotional", "expressive", language="de")
        self.dev = self.row("librispeech", "heldout", split="dev")
        self.prior = self.row("librispeech", "prior-used")
        self.source("core", "acquire", [self.fleurs, self.libri])
        self.source("indic", "indic", self.indic)
        self.source("expressive", "expressive", [self.expressive])
        write_manifest(root / "dev.jsonl", [self.dev])
        write_manifest(root / "prior.jsonl", [self.prior])
        write_json(root / "legacy.json", {"samples": []})
        metadata = "10\tfr.wav\ttext\ttext\ttext\t16000\t0\n".encode()
        path = root / "core/provenance/fr_fr/train.tsv"
        path.parent.mkdir(parents=True)
        path.write_bytes(metadata)
        write_json(root / "core/provenance/acquisition-plan.json", {
            "configuration": {"fleurs_revision": FLEURS_REVISION, "languages": ["fr_fr"]},
            "language_seconds": {"fr_fr": 1}, "metadata_sha256": {"fr_fr": hashlib.sha256(metadata).hexdigest()},
        })
        self.config = {"format_version": 1, "minimum_train_hours": 25 / 3600,
                       "minimum_indic_hours_per_language": 1 / 3600,
                       "minimum_fleurs_configurations": 1, "maximum_utterance_seconds": 180,
                       "fleurs_plan": "core/provenance/acquisition-plan.json",
                       "initial_source_names": ["core", "indic", "expressive"],
                       "sources": [self.spec(name, kind) for name, kind in
                                   (("core", "acquire"), ("indic", "indic"), ("expressive", "expressive"))],
                       "dev_manifests": ["dev.jsonl"], "prior_training_manifests": ["prior.jsonl"],
                       "reserved_manifests": [], "legacy_evaluation_manifests": ["legacy.json"]}

    def row(self, dataset, source_id, language="en", split="train", filename=None):
        self.counter += 1
        path = self.root / "audio" / (filename or source_id + ".wav")
        path.parent.mkdir(exist_ok=True)
        sf.write(path, np.full(16000, self.counter / 100, dtype=np.float32), 16000, subtype="FLOAT")
        return ManifestRow(
            dataset=dataset, source_revision=FLEURS_REVISION if dataset == "fleurs" else "pinned-release-1",
            source_id=source_id, source_url="https://example.test/source", audio_path="../audio/" + path.name,
            audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), parent_recording_id=dataset + ":parent:" + source_id,
            parent_start_seconds=0, speaker_id=dataset + ":speaker:" + source_id,
            session_id=dataset + ":session:" + source_id, language=language, sample_rate_hz=16000,
            original_sample_rate_hz=16000, bandwidth_hz=8000, bandwidth_class="speech_band",
            bandwidth_evidence="Synthetic fixture's mono16k delivery defines conservative speech-band supervision",
            native_recording=False, enhanced=False, duration_seconds=1, split=split,
            source_split="dev-clean" if split == "dev" else "train",
            license="CC0-1.0" if dataset == "thorsten_emotional" else "CC-BY-4.0",
            license_url="https://example.test/license", attribution="Synthetic test fixture",
            access_record="../access.json", gain_policy="none: unchanged",
            resampler_policy="none: original mono 16000 Hz", teacher_cache_key=None,
        )

    def source(self, name, kind, rows):
        write_manifest(self.root / name / "train.jsonl", [row for row in rows if row.split == "train"])
        write_manifest(self.root / name / "dev.jsonl", [row for row in rows if row.split == "dev"])
        summary = {"rows": len(rows), "hours": sum(row.duration_seconds for row in rows) / 3600,
                   "hours_by": {"split": {split: sum(row.duration_seconds for row in rows if row.split == split) / 3600
                                           for split in {row.split for row in rows}}}}
        marker = {"state": "ready", "summary": summary} if kind == "acquire" else (
            {"status": "complete", **summary} if kind == "indic" else {"state": "complete", "preparation_version": 2, **summary})
        write_json(self.root / name / "complete.json", marker)

    def spec(self, name, kind):
        return {"name": name, "kind": kind, "manifests": [name + "/train.jsonl", name + "/dev.jsonl"],
                "readiness": name + "/complete.json"}

    def run(self, name="assembly.json", output="assembled"):
        # Fixed dev/prior manifests live at the config root, so use root-relative
        # paths there while source shards retain manifest-relative paths.
        for name_fixed, row in (("dev", self.dev), ("prior", self.prior)):
            write_manifest(self.root / (name_fixed + ".jsonl"), [replace(row,
                audio_path="audio/" + Path(row.audio_path).name, access_record="access.json")])
        path = self.root / name
        write_json(path, self.config)
        return assemble_corpus(path, self.root / output)


class CorpusAssemblyTests(unittest.TestCase):
    def test_expressive_source_requires_its_explicit_preparation_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            emogator = replace(fixture.row("emogator", "emogator:test"), license="Apache-2.0")
            fixture.source("emogator", "expressive", [emogator])
            marker_path = fixture.root / "emogator/complete.json"
            marker = json.loads(marker_path.read_text())
            marker["preparation_version"] = 1
            write_json(marker_path, marker)
            spec = fixture.spec("emogator", "expressive")
            fixture.config["sources"].append(spec)
            self.assertEqual(fixture.run()["state"], "pending_sources")
            spec["preparation_version"] = 1
            self.assertEqual(fixture.run()["state"], "ready")

    def test_ready_uses_actual_frames_and_combines_dev_with_resolved_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            report = fixture.run()
            self.assertEqual(report["state"], "ready", report["issues"])
            self.assertAlmostEqual(report["verified_train_hours"], 25 / 3600)
            self.assertEqual(report["verified_dev_rows"], 1)
            self.assertEqual(report["coverage"]["verified_indic_languages"], 22)
            self.assertIn("en", report["train_hours_by_normalized_language"])
            rows = load_manifest(fixture.root / "assembled/source-manifest.jsonl")
            self.assertEqual({row.split for row in rows}, {"train", "dev"})
            self.assertTrue(all(Path(row.audio_path).is_absolute() for row in rows))
            self.assertEqual(len(load_manifest(fixture.root / "assembled/filler-exclusion.jsonl")), 26)

    def test_real_500_hour_gate_and_filler_only_after_all_sources_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            fixture.config["minimum_train_hours"] = 500
            report = fixture.run()
            self.assertEqual(report["state"], "insufficient_hours")
            self.assertAlmostEqual(report["remaining_train_hours"], 500 - 25 / 3600)
            self.assertTrue(report["filler_permitted"])
            self.assertFalse((fixture.root / "assembled/source-manifest.jsonl").exists())
            marker = fixture.root / "indic/complete.json"
            value = json.loads(marker.read_text()); value["status"] = "downloading"; write_json(marker, value)
            report = fixture.run()
            self.assertEqual(report["state"], "pending_sources")
            self.assertFalse(report["initial_sources_complete"])
            self.assertFalse(report["filler_permitted"])

    def test_ready_audit_and_manifest_are_byte_stable_and_cannot_be_reconfigured(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            first = fixture.run()
            audit = fixture.root / "assembled/readiness.json"
            before, stat = audit.read_bytes(), audit.stat().st_mtime_ns
            second = assemble_corpus(fixture.root / "assembly.json", fixture.root / "assembled")
            self.assertEqual(first, second)
            self.assertEqual(audit.read_bytes(), before)
            self.assertEqual(audit.stat().st_mtime_ns, stat)
            fixture.config["minimum_train_hours"] = 500
            write_json(fixture.root / "changed.json", fixture.config)
            with self.assertRaises(FrozenCorpusError):
                assemble_corpus(fixture.root / "changed.json", fixture.root / "assembled")
            self.assertEqual(audit.read_bytes(), before)

    def test_metadata_fingerprint_ignores_manifest_row_order_but_ready_inputs_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            first = fixture.run()
            write_manifest(fixture.root / "core/train.jsonl", [fixture.libri, fixture.fleurs])
            with self.assertRaises(FrozenCorpusError):
                assemble_corpus(fixture.root / "assembly.json", fixture.root / "assembled")
            second = assemble_corpus(fixture.root / "assembly.json", fixture.root / "independent-assembly")
            self.assertEqual(second["state"], "ready", second["issues"])
            self.assertEqual(first["metadata_fingerprint"], second["metadata_fingerprint"])
            self.assertEqual(first["source_manifest_sha256"], second["source_manifest_sha256"])

    def test_filler_phase_uses_new_config_then_freezes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            fixture.config["minimum_train_hours"] = 26 / 3600
            self.assertEqual(fixture.run()["state"], "insufficient_hours")
            filler = fixture.row("librispeech", "fresh-filler")
            fixture.source("filler", "acquire", [filler])
            fixture.config["sources"].append(fixture.spec("filler", "acquire"))
            report = fixture.run("assembly-with-fill.json")
            self.assertEqual(report["state"], "ready", report["issues"])
            with self.assertRaises(FrozenCorpusError):
                assemble_corpus(fixture.root / "assembly.json", fixture.root / "assembled")

    def test_missing_indic_and_fleurs_coverage_do_not_trigger_english_filler(self):
        for source in ("indic", "fleurs"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                fixture.config["minimum_train_hours"] = 500
                if source == "indic":
                    fixture.source("indic", "indic", fixture.indic[:-1])
                else:
                    fixture.source("core", "acquire", [fixture.libri])
                report = fixture.run()
                self.assertEqual(report["state"], "insufficient_coverage", report["issues"])
                self.assertFalse(report["filler_permitted"])

    def test_hash_duration_source_split_and_long_file_checks_fail_closed(self):
        for violation in ("hash", "duration", "split", "maximum", "summary"):
            with self.subTest(violation=violation), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                if violation == "hash":
                    (fixture.root / "audio/fr.wav").write_bytes(b"changed")
                elif violation == "duration":
                    fixture.source("core", "acquire", [replace(fixture.fleurs, duration_seconds=1000000), fixture.libri])
                elif violation == "split":
                    fixture.source("indic", "indic", [replace(fixture.indic[0], source_split="VALID")] + fixture.indic[1:])
                elif violation == "maximum":
                    fixture.config["maximum_utterance_seconds"] = 0.5
                else:
                    marker = json.loads((fixture.root / "core/complete.json").read_text())
                    marker["summary"]["hours"] = 500
                    write_json(fixture.root / "core/complete.json", marker)
                report = fixture.run()
                self.assertEqual(report["state"], "invalid_sources")
                self.assertFalse(report["filler_permitted"])
                self.assertFalse((fixture.root / "assembled/source-manifest.jsonl").exists())
                if violation == "maximum":
                    self.assertEqual(report["actual_maximum_utterance_seconds"], 1)

    def test_heldout_and_prior_identity_leakage_is_rejected_across_revisions(self):
        for violation in ("speaker", "session", "parent", "source"):
            with self.subTest(violation=violation), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                if violation == "source":
                    changed = replace(fixture.libri, source_id=fixture.prior.source_id, source_revision="another-release")
                else:
                    field = {"speaker": "speaker_id", "session": "session_id", "parent": "parent_recording_id"}[violation]
                    changed = replace(fixture.libri, **{field: getattr(fixture.dev, field)})
                fixture.source("core", "acquire", [fixture.fleurs, changed])
                report = fixture.run()
                self.assertEqual(report["state"], "invalid_sources")

    def test_legacy_original_hash_and_normalized_fleurs_text_ids_are_excluded(self):
        for violation in ("hash", "text_id", "filename", "prepared_original"):
            with self.subTest(violation=violation), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                entry = {"language": "FR-FR"}
                if violation == "hash":
                    entry["sha256"] = fixture.fleurs.audio_sha256
                elif violation == "text_id":
                    entry["dataset_id"] = 10
                elif violation == "filename":
                    entry["original_audio_path"] = "/legacy/fr.wav"
                else:
                    entry["sha256"] = "a" * 64
                    trace = fixture.root / "expressive/provenance.json"
                    write_json(trace, {"prepared_audio_sha256": fixture.expressive.audio_sha256,
                                       "prepared_frames": 16000, "original_audio_sha256": "a" * 64})
                    fixture.source("expressive", "expressive", [replace(fixture.expressive, access_record="provenance.json")])
                write_json(fixture.root / "legacy.json", {"samples": [entry]})
                report = fixture.run()
                self.assertEqual(report["state"], "invalid_sources")


if __name__ == "__main__":
    unittest.main()
