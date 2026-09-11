from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from audiovae_student.indic_staging import NativeShardStager, OVERHEAD_BYTES, XET_OVERHEAD_BYTES, configure_native_transfer


PAYLOAD = b"pinned complete original parquet bytes" * 10


def item(path="hindi/train-00001.parquet", payload=PAYLOAD):
    return SimpleNamespace(path=path, size=len(payload), lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()))


def download(repo, name, *, local_dir, revision, repo_type):
    path = Path(local_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(PAYLOAD)
    return path


def stager(root, **kwargs):
    return NativeShardStager(root, repo="synthetic/repo", revision="a" * 40, lock=threading.RLock(),
                             stop=kwargs.pop("stop", threading.Event()), reserve_bytes=0,
                             download=kwargs.pop("download", download), **kwargs)


class IndicStagingTests(unittest.TestCase):
    def test_exact_bytes_and_cleanup_after_reader_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = stager(root)
            with staging.shard(item()) as path:
                with path.open("rb") as stream:
                    self.assertEqual(stream.read(), PAYLOAD)
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())
            self.assertTrue((root / "owner.json").exists())
            self.assertEqual(staging.entries, {})
            staging.close()

    def test_stop_retains_verified_cache_and_resumes_without_losing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = threading.Event();first = stager(Path(tmp), stop=stop)
            with first.shard(item()) as path:
                stop.set()
            self.assertTrue(path.exists())
            first.close()
            called = []
            def resumed_download(*args, **kwargs):
                existing = Path(kwargs["local_dir"]) / args[1]
                self.assertTrue(existing.exists())
                called.append(existing.read_bytes())
                return existing
            second = stager(Path(tmp), download=resumed_download)
            with second.shard(item()) as again:
                self.assertEqual(path, again)
            self.assertEqual(called, [PAYLOAD])
            self.assertFalse(path.exists())
            second.close()

    def test_partial_interruption_is_retained_and_accounted_at_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def interrupted(*args, **kwargs):
                (Path(kwargs["local_dir"]) / "partial.incomplete").write_bytes(PAYLOAD[:10])
                raise OSError("synthetic interrupted transport")
            first = stager(Path(tmp), download=interrupted)
            with self.assertRaises(OSError):
                with first.shard(item()): pass
            first.close()
            second = stager(Path(tmp))
            self.assertEqual(len(second.entries), 1)
            self.assertGreater(second.pending_bytes(), XET_OVERHEAD_BYTES)
            with second.shard(item()) as path:
                self.assertEqual(path.read_bytes(), PAYLOAD)
            second.close()

    def test_size_sha_ownership_and_symlink_fail_closed(self):
        for bad in (b"short", b"x" * len(PAYLOAD)):
            with self.subTest(payload=bad[:5]), tempfile.TemporaryDirectory() as tmp:
                def wrong(*args, **kwargs):
                    path=Path(kwargs["local_dir"])/args[1];path.parent.mkdir(parents=True);path.write_bytes(bad);return path
                staging = stager(Path(tmp), download=wrong)
                with self.assertRaisesRegex(ValueError, "size|SHA-256"):
                    with staging.shard(item()): pass
                self.assertEqual(len(staging.entries), 1)
                staging.close()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "unrelated").write_text("keep me")
            with self.assertRaisesRegex(ValueError, "ownership"):
                stager(Path(tmp))
            self.assertTrue((Path(tmp) / "unrelated").exists())

    def test_shared_cap_waits_until_first_owned_shard_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging = stager(Path(tmp), max_bytes=XET_OVERHEAD_BYTES + OVERHEAD_BYTES + len(PAYLOAD) + 100)
            attempting, entered, errors = threading.Event(), threading.Event(), []
            def other():
                attempting.set()
                try:
                    with staging.shard(item("tamil/train-00001.parquet")):
                        entered.set()
                except BaseException as error: errors.append(error)
            with staging.shard(item()):
                thread = threading.Thread(target=other);thread.start()
                self.assertTrue(attempting.wait(1))
                self.assertFalse(entered.wait(.05))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(entered.is_set());self.assertEqual(errors, [])
            staging.close()

    def test_audio_writes_preserve_reserved_staging_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging = stager(Path(tmp))
            with staging.shard(item()):
                pending = staging.pending_bytes()
                fake = SimpleNamespace(free=pending + 9)
                with patch("audiovae_student.indic_staging.shutil.disk_usage", return_value=fake):
                    staging.check_audio_write(9)
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        staging.check_audio_write(10)
            staging.close()

    def test_stop_before_transfer_and_native_options_do_not_change_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop=threading.Event();staging=stager(Path(tmp), stop=stop);stop.set()
            with self.assertRaises(InterruptedError):
                with staging.shard(item()): pass
            self.assertFalse(staging.entries);staging.close()
            with patch.dict(os.environ, {"HF_HOME": "/existing/login"}):
                configure_native_transfer(Path(tmp))
                self.assertEqual(os.environ["HF_HOME"], "/existing/login")
                self.assertEqual(os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"], "0")
                self.assertEqual(os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"], "16")


if __name__ == "__main__":
    unittest.main()
