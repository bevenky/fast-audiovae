"""Synthetic expressive corpus preparation; no network or real speech tests."""

import hashlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import soxr

from audiovae_student.acquire_expressive import (
    GAIN_POLICY, NATIVE_POLICY, RESAMPLE_POLICY, StorageBudget, archive_members,
    acquire_crema, audit_nonverbal_labels, deduplicate_prepared, parse_lfs_pointer, prepare_archive, prepare_audio, prepare_member, source_identity,
)
from audiovae_student.data import load_manifest, validate_manifest, validate_mixture


def wav(rate=48000, seconds=0.4, frequency=500, channels=1):
    audio = (0.03 * np.sin(2 * np.pi * frequency * np.arange(round(rate * seconds)) / rate)).astype(np.float32)
    if channels > 1:
        audio = np.stack([audio] * channels, axis=-1)
    stream = io.BytesIO()
    sf.write(stream, audio, rate, format="WAV", subtype="PCM_24")
    return stream.getvalue()


class ExpressiveTests(unittest.TestCase):
    def test_vhq_whole_utterance_matches_explicit_resampler_without_normalizing(self):
        original = wav(rate=22050)
        payload, trace = prepare_audio(original)
        raw, rate = sf.read(io.BytesIO(original), dtype="float32")
        prepared, output_rate = sf.read(io.BytesIO(payload), dtype="float32")
        np.testing.assert_array_equal(prepared, soxr.resample(raw, rate, 16000, quality="VHQ"))
        self.assertEqual(output_rate, 16000)
        self.assertEqual(sf.info(io.BytesIO(payload)).subtype, "FLOAT")
        self.assertEqual(payload[:4], b"RIFF")
        self.assertEqual(payload[12:16], b"fmt ")
        self.assertEqual(payload[36:40], b"fact")
        self.assertEqual(payload[48:52], b"data")
        self.assertEqual(payload, prepare_audio(original)[0])
        self.assertEqual(trace["resampler_policy"], RESAMPLE_POLICY)
        self.assertEqual(trace["original_audio_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(trace["original_subtype"], "PCM_24")
        self.assertEqual(trace["gain_policy"], GAIN_POLICY)
        self.assertLess(np.max(np.abs(prepared)), 0.04)

    def test_native_rate_decode_is_sample_exact_and_short_nv_is_retained(self):
        original = wav(rate=16000, seconds=0.1)
        payload, trace = prepare_audio(original)
        np.testing.assert_array_equal(sf.read(io.BytesIO(original), dtype="float32")[0],
                                      sf.read(io.BytesIO(payload), dtype="float32")[0])
        self.assertEqual(trace["resampler_policy"], NATIVE_POLICY)
        self.assertIsNone(trace["soxr_version"])
        self.assertEqual(trace["prepared_frames"], 1600)
        with self.assertRaisesRegex(ValueError, "too short"):
            prepare_audio(wav(seconds=0.01))
        with self.assertRaisesRegex(ValueError, "non-mono"):
            prepare_audio(wav(channels=2))

    def test_labels_and_cross_corpus_speaker_groups_are_conservative(self):
        a = source_identity("jnv", "JNV/M2/M2_angry_00_R.wav")
        b = source_identity("jvnv", "jvnv_v1/M2/anger/free/M2_anger_free_04.wav")
        self.assertEqual(a.speaker, b.speaker)
        self.assertEqual(a.split, "dev")
        self.assertEqual(b.split, "dev")
        self.assertEqual(source_identity("crema_d", "AudioWAV/1010_DFA_ANG_HI.wav").split, "dev")
        self.assertEqual(source_identity("crema_d", "AudioWAV/1001_DFA_ANG_HI.wav").split, "train")
        thorsten = source_identity("thorsten_emotional", "base/whispering/" + "a" * 32 + ".wav")
        self.assertEqual(thorsten.emotion, "whispering")
        self.assertEqual(thorsten.split, "train")
        self.assertEqual(source_identity("thorsten_emotional", "base/whisper/" + "a" * 32 + ".wav").emotion, "whispering")
        self.assertNotIn("shouting", source_identity("crema_d", "1001_DFA_ANG_HI.wav").labels.values())
        for invalid in ("../M2_angry_00_R.wav", "JNV/M2/M2_whistling_00_R.wav"):
            with self.assertRaises(ValueError):
                source_identity("jnv", invalid)

    def test_jnv_stereo_policy_selects_original_channel_without_mixing(self):
        raw = np.stack([np.linspace(-0.1, 0.1, 4800, dtype=np.float32),
                        np.linspace(0.4, -0.4, 4800, dtype=np.float32)], axis=-1)
        stream = io.BytesIO()
        sf.write(stream, raw, 48000, format="WAV", subtype="FLOAT")
        prepared, trace = prepare_audio(stream.getvalue(), stereo_first_channel=True)
        np.testing.assert_array_equal(sf.read(io.BytesIO(prepared), dtype="float32")[0],
                                      soxr.resample(raw[:, 0], 48000, 16000, quality="VHQ"))
        self.assertEqual(trace["original_channels"], 2)
        self.assertIn("channel 0", trace["channel_policy"])

    def test_provenance_keeps_original_and_prepared_hashes_distinct_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = StorageBudget(root, 10_000_000, 0)
            receipt = {"url": "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip", "archive_sha256": "a" * 64}
            original = wav()
            row, trace = prepare_member("jnv", "JNV/F1/F1_happy_01_F.wav", original, root, budget, receipt)
            validate_manifest([row])
            self.assertNotEqual(row.audio_sha256, trace["original_audio_sha256"])
            self.assertEqual(row.original_sample_rate_hz, 48000)
            self.assertEqual(row.bandwidth_class, "speech_band")
            self.assertEqual(row.bandwidth_hz, 8000)
            self.assertIn("not a measured", row.bandwidth_evidence)
            self.assertEqual(json.loads(Path(row.access_record).read_text())["source_receipt"]["archive_sha256"], "a" * 64)
            used = budget.used
            repeated, _ = prepare_member("jnv", "JNV/F1/F1_happy_01_F.wav", original, root, budget, receipt)
            self.assertEqual(repeated, row)
            self.assertEqual(budget.used, used)
            with self.assertRaisesRegex(ValueError, "disagree"):
                prepare_member("jnv", "JNV/F1/F1_happy_01_F.wav", wav(frequency=750), root, budget, receipt)

    def test_archive_checks_receipt_retains_short_clips_and_audits_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "jnv_corpus_ver3.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("JNV/F1/F1_happy_01_F.wav", wav(seconds=0.1))
                archive.writestr("JNV/M2/M2_angry_02_R.wav", wav(seconds=0.5, frequency=600))
                archive.writestr("JNV/F1/F1_sad_03_R.wav", wav(seconds=0.01))
            receipt = {"state": "complete", "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "url": "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip", "license": "CC-BY-SA-4.0"}
            (root / "jnv-ver3.json").write_text(json.dumps(receipt))
            result = prepare_archive("jnv", root, StorageBudget(root, 10_000_000, 0))
            self.assertEqual(result["rows"], 2)
            self.assertEqual(len(result["excluded"]), 1)
            self.assertEqual(len(load_manifest(root / "prepared/manifests/jnv-train.jsonl")), 1)
            self.assertEqual(len(load_manifest(root / "prepared/manifests/jnv-dev.jsonl")), 1)
            receipt["archive_sha256"] = "b" * 64
            (root / "jnv-ver3.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_archive("jnv", root, StorageBudget(root, 10_000_000, 0))

    def test_budget_and_archive_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = StorageBudget(root, 10, 0)
            with self.assertRaisesRegex(ValueError, "cap"):
                budget.write(root / "audio.wav", b"a" * 11)
            self.assertFalse((root / "audio.wav").exists())
            archive_path = root / "malformed.tar"
            with tarfile.open(archive_path, "w") as archive:
                member = tarfile.TarInfo("F1_happy_01_F.wav")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, "non-regular"):
                list(archive_members(archive_path))

    def test_additional_sources_do_not_rewrite_original_recipe_quota_contract(self):
        recipe = Path(__file__).parents[1] / "configs/data-mixture.json"
        validate_mixture(json.loads(recipe.read_text()))

    def test_crema_lfs_pointer_is_verified_against_git_tree_and_audio(self):
        payload = wav(rate=16000)
        digest = hashlib.sha256(payload).hexdigest()
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(payload)}\n".encode()
        blob_sha = hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest()
        self.assertEqual(parse_lfs_pointer(pointer, blob_sha), (digest, len(payload)))
        with self.assertRaisesRegex(ValueError, "Git blob"):
            parse_lfs_pointer(pointer, "a" * 40)
        item = {"path": "AudioWAV/1001_DFA_ANG_HI.wav", "sha256": digest,
                "bytes": len(payload), "git_blob_sha1": blob_sha}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("audiovae_student.acquire_expressive.crema_inventory", return_value=[item]), \
                 patch("audiovae_student.acquire_expressive._open_url", side_effect=lambda _: io.BytesIO(payload)) as request:
                result = acquire_crema(root, root / "tree.json", StorageBudget(root, 10_000_000, 0), 1)
                self.assertEqual(result["rows"], 1)
                self.assertEqual(request.call_count, 1)
                acquire_crema(root, root / "tree.json", StorageBudget(root, 10_000_000, 0), 1)
                self.assertEqual(request.call_count, 1)
            (root / "crema-originals" / item["path"]).write_bytes(payload[:-1] + b"x")
            with patch("audiovae_student.acquire_expressive.crema_inventory", return_value=[item]), \
                 self.assertRaisesRegex(ValueError, "pinned LFS"):
                acquire_crema(root, root / "tree.json", StorageBudget(root, 10_000_000, 0), 1)

    def test_action_audit_does_not_relabel_emotions_as_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = StorageBudget(root, 10_000_000, 0)
            rows = []
            for dataset, member in (("jnv", "JNV/F1/F1_sad_00_R.wav"),
                                    ("jvnv", "jvnv_v1/F1/anger/regular/F1_anger_regular_01.wav"),
                                    ("jvnv", "jvnv_v1/F1/happy/regular/F1_happy_regular_01.wav")):
                row, _ = prepare_member(dataset, member, wav(), root, budget, {"url": "https://example.test/source.zip"})
                rows.append(row)
            (root / "prepared/train.jsonl").write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
            (root / "prepared/provenance/expressive-complete.json").write_text('{"state":"complete"}')
            with zipfile.ZipFile(root / "jnv_corpus_ver3.zip", "w") as archive:
                archive.writestr("JNV/phrases.txt", "sad_00_R|うぅ\n")
            with zipfile.ZipFile(root / "jvnv_ver1.zip", "w") as archive:
                archive.writestr("jvnv_v1/transcription.csv", "anger_regular_01|おい|text\nhappy_regular_01|あはは！|text\n")
                archive.writestr("jvnv_v1/nv_label/F1/F1_anger_regular_01.txt", "0.1\t0.2\tNV\n")
                archive.writestr("jvnv_v1/nv_label/F1/F1_happy_regular_01.txt", "0.15\t0.35\tNV\n")
            audit = audit_nonverbal_labels(root)
            self.assertEqual(audit["groups"]["jvnv_generic_nv_annotation"]["train_utterances"], 2)
            self.assertAlmostEqual(audit["groups"]["laughter_like_phrase_candidate"]["annotated_nv_seconds"], 0.2)
            self.assertNotIn("sniffle_or_sob_phrase_candidate", audit["groups"])
            self.assertNotIn("shouting", audit["groups"])
            self.assertIn("crying", audit["not_separately_verified"])

    def test_duplicate_audio_does_not_count_twice_or_leak_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, _ = prepare_member("crema_d", "1001_DFA_ANG_HI.wav", wav(rate=16000), root,
                                    StorageBudget(root, 10_000_000, 0), {"url": "https://example.test/source.wav"})
            twin = replace(row, source_id="crema_d:1001_DFA_NEU_HI.wav", parent_recording_id="different-parent")
            excluded = []
            self.assertEqual(deduplicate_prepared([row, twin], excluded), [row])
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["retained_source_id"], row.source_id)
            excluded = []
            self.assertEqual(deduplicate_prepared([row, replace(twin, split="dev")], excluded), [])
            self.assertEqual(len(excluded), 2)


if __name__ == "__main__":
    unittest.main()
