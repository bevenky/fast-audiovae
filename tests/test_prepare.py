"""Preparation preflight tests use tiny fake files and never execute a model."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

prepare = importlib.import_module("fast_audiovae.prepare")


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.onnx"
        self.source.write_bytes(b"test model")
        self.output = self.root / "output"
        self.patches = [patch.object(prepare.platform, "system", return_value="Linux"),
                        patch.object(prepare.platform, "machine", return_value="x86_64"),
                        patch.object(prepare, "fetch_model"),
                        patch.object(prepare, "verify_model", return_value=self.source),
                        patch.object(prepare.onnx, "load")]
        self.system, self.machine, self.fetch, self.verify, self.load = [p.start() for p in self.patches]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def build_record(self, name="base", **overrides):
        library = self.root / (name + ".so")
        library.write_bytes(name.encode())
        record = {"library": str(library), "library_sha256": prepare.sha256(library),
                  "native_abi": 1, "ort_api_version": 29, "domain": "venky.audio.cpu.portable"}
        record.update(overrides)
        path = self.root / (name + ".json")
        path.write_text(json.dumps(record))
        return path

    def assert_preflight_untouched(self):
        self.assertFalse(self.output.exists())
        self.fetch.assert_not_called()
        self.verify.assert_not_called()
        self.load.assert_not_called()

    def test_nonempty_output_and_files_are_never_overwritten(self):
        self.output.mkdir()
        existing = self.output / "decoder_portable.onnx"
        existing.write_bytes(b"keep this")
        with self.assertRaisesRegex(ValueError, "empty or nonexistent"):
            prepare.prepare(self.output, source=self.source)
        self.assertEqual(existing.read_bytes(), b"keep this")
        self.fetch.assert_not_called()
        self.verify.assert_not_called()
        self.load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "empty or nonexistent"):
            prepare.prepare(self.source, source=self.source)
        self.assertEqual(self.source.read_bytes(), b"test model")

    def test_source_output_collision_is_rejected_before_download_or_load(self):
        with self.assertRaisesRegex(ValueError, "outside the output"):
            prepare.prepare(self.output, source=self.output / "decoder_portable.onnx")
        self.assert_preflight_untouched()

    def test_bad_native_domain_abi_api_or_hash_never_creates_output(self):
        for override in ({"native_abi": 2}, {"native_abi": True}, {"ort_api_version": 28},
                         {"ort_api_version": None}, {"domain": "venky.audio.cpu"},
                         {"library_sha256": "wrong"}):
            with self.subTest(override=override):
                build = self.build_record(**override)
                with self.assertRaises(ValueError):
                    prepare.prepare(self.output, source=self.source, native_build=build)
                self.assert_preflight_untouched()

    def test_bad_amd_domain_abi_api_or_hash_never_creates_output(self):
        base = self.build_record()
        for override in ({"native_abi": 2}, {"ort_api_version": 28},
                         {"domain": "venky.audio.cpu.portable"}, {"library_sha256": "wrong"}):
            with self.subTest(override=override):
                fields = {"domain": "venky.audio.cpu.aocl.rows", **override}
                amd = self.build_record("amd", **fields)
                with self.assertRaises(ValueError):
                    prepare.prepare(self.output, source=self.source, native_build=base, amd_build=amd)
                self.assert_preflight_untouched()

    def test_amd_requires_linux_x86_and_base_native_build(self):
        amd = self.build_record("amd", domain="venky.audio.cpu.aocl.rows")
        # A missing explicit base record must not be satisfied by a developer's
        # unrelated default build directory.
        with patch.object(prepare.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "base native build"):
                prepare.prepare(self.output, source=self.source, amd_build=amd)
        self.assert_preflight_untouched()
        self.system.return_value = "Darwin"
        self.machine.return_value = "arm64"
        with self.assertRaisesRegex(ValueError, "Linux x86"):
            prepare.prepare(self.output, source=self.source, amd_build=amd)
        self.assert_preflight_untouched()

    def test_library_filename_collision_is_rejected(self):
        base = self.build_record()
        amd = self.build_record("amd", domain="venky.audio.cpu.aocl.rows")
        data = json.loads(amd.read_text())
        data.update(library=str(self.root / "base.so"), library_sha256=prepare.sha256(self.root / "base.so"))
        amd.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "different filenames"):
            prepare.prepare(self.output, source=self.source, native_build=base, amd_build=amd)
        self.assert_preflight_untouched()

    def test_empty_output_allows_standard_onnx_bundle(self):
        self.output.mkdir()
        self.system.return_value = "Windows"
        model = object()
        self.load.return_value = model
        with patch.object(prepare.upsampling, "rewrite", return_value=(model, [])) as rewrite, \
             patch.object(prepare, "_save", side_effect=lambda m, path: path.write_bytes(b"graph")):
            result = prepare.prepare(self.output, source=self.source)
        rewrite.assert_called_once_with(model, mode="split-matmul")
        self.assertEqual(result["native"], {})
        self.assertEqual(result["providers"], ["CPUExecutionProvider"])
        self.assertTrue((self.output / "bundle.json").is_file())


if __name__ == "__main__":
    unittest.main()
