"""Synthetic licensed-source preparation checks; no network or model execution."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audiovae_student.acquire_emogator import HELDOUT, CATEGORIES, identity, load_tree, prepare_member, verify_blob
from audiovae_student.acquire_expressive import StorageBudget
from audiovae_student.data import validate_manifest


def mp3():
    stream = io.BytesIO()
    audio = np.sin(np.arange(9600) * 0.07).astype(np.float32) * 0.03
    sf.write(stream, audio, 48000, format="MP3", subtype="MPEG_LAYER_III")
    payload = stream.getvalue()
    return payload, {"path": "data/mp3/000001-02-1.mp3", "size": len(payload),
                     "sha": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()}


class EmoGatorTests(unittest.TestCase):
    def test_contributor_split_and_category_are_explicit_and_complete(self):
        self.assertEqual(len(HELDOUT), 20)
        self.assertEqual(len(CATEGORIES), 30)
        self.assertEqual(identity("000357-30-3.mp3"), ("000357", 30, "3"))
        for name in ("../000001-01-1.mp3", "000358-01-1.mp3", "000001-31-1.mp3", "000001-01-4.mp3"):
            with self.assertRaises(ValueError):
                identity(name)

    def test_pinned_bytes_are_verified_and_tree_cannot_be_truncated(self):
        payload, entry = mp3()
        verify_blob(payload, entry)
        with self.assertRaisesRegex(ValueError, "pinned Git"):
            verify_blob(payload[:-1], entry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps({"truncated": True, "tree": []}))
            with self.assertRaisesRegex(ValueError, "truncated"):
                load_tree(path)

    def test_prepared_mp3_is_float_mono16k_with_original_provenance_and_no_language_guess(self):
        payload, entry = mp3()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, trace = prepare_member(entry["path"], payload, entry, root, StorageBudget(root, 10_000_000, 0),
                                        {"archive_sha256": "a" * 64, "url": "https://example.test/archive"})
            validate_manifest([row])
            info = sf.info(row.audio_path)
            self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "FLOAT"))
            self.assertEqual(row.language, "und")
            self.assertEqual(row.original_sample_rate_hz, 48000)
            self.assertEqual(trace["original_audio_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(trace["original_subtype"], "MPEG_LAYER_III")
            self.assertEqual(trace["intended_emotion"], "Amusement")
            self.assertIn("not separately annotated", trace["action_label"])
            self.assertIn("source_zero_fraction", trace)
            self.assertIn("source_at_or_over_full_scale_fraction", trace)
            self.assertEqual(row.split, "dev" if "000001" in HELDOUT else "train")
            self.assertEqual(row.resampler_policy, "prepared-soxr-vhq-to-16000-v1")
            repeated, _ = prepare_member(entry["path"], payload, entry, root, StorageBudget(root, 10_000_000, 0),
                                         {"archive_sha256": "a" * 64, "url": "https://example.test/archive"})
            self.assertEqual(row, repeated)


if __name__ == "__main__":
    unittest.main()
