"""Preparation preflight tests use tiny fake files and never execute a model."""
import importlib
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

prepare = importlib.import_module("fast_audiovae.prepare")


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
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

    @contextmanager
    def native_pipeline(self):
        """Exercise bundle routing with fake files, no CPU libraries or models."""
        model = object()
        self.load.return_value = model

        def derived(source, output, *args):
            Path(output).write_bytes(b"native graph")

        def fusion(source, output, *, variant, backend, expected_chains, expected_adds):
            source, output = Path(source), Path(output)
            output.write_bytes(b"fused " + variant.encode())
            audit = {"variant": variant, "backend_override": backend,
                     "counts": {"chain": expected_chains, "adds": expected_adds},
                     "source_sha256": prepare.sha256(source), "output_sha256": prepare.sha256(output)}
            output.with_suffix(".block-fusion.json").write_text(json.dumps(audit))
            return audit

        with ExitStack() as stack:
            stack.enter_context(patch.object(prepare, "_save", side_effect=lambda m, p: p.write_bytes(b"graph")))
            stack.enter_context(patch.object(prepare.upsampling, "rewrite", return_value=(model, {})))
            stack.enter_context(patch.object(prepare.elementwise, "rewrite", return_value=(model,
                {"fused_depthwise": 18, "matched_snakes": 43})))
            stack.enter_context(patch.object(prepare.pointwise, "rewrite", side_effect=derived))
            stack.enter_context(patch.object(prepare.phase, "rewrite", side_effect=derived))
            convert = stack.enter_context(patch("fast_audiovae.graph.portable.convert", side_effect=derived))
            packed = stack.enter_context(patch("fast_audiovae.graph.packed.rewrite", side_effect=derived))
            rewrite = stack.enter_context(patch.object(prepare.block_fusion_graph, "rewrite", side_effect=fusion))
            yield rewrite, convert, packed

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
        self.assertEqual(result["block_fusion"], "none")
        self.assertTrue((self.output / "bundle.json").is_file())

    def test_invalid_block_fusion_is_rejected_before_output_or_model_access(self):
        for variant in ("all", "", None, True, []):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "block_fusion must"):
                prepare.prepare(self.output, source=self.source, block_fusion=variant)
            self.assert_preflight_untouched()

    def test_block_fusion_requires_native_build_before_output_or_download(self):
        with patch.object(prepare.Path, "is_file", return_value=False):
            for variant in ("adds", "chain", "both"):
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "requires a compatible native build"):
                    prepare.prepare(self.output, source=self.source, block_fusion=variant)
                self.assert_preflight_untouched()

    def test_old_or_incomplete_native_operator_records_fail_fusion_preflight(self):
        baseline = ["SnakeF32", "CausalDW7SnakeF32", "PhaseSumBiasInterleaveF32"]
        for operators in (None, "BiasResidualF32", [], baseline,
                          baseline + ["BiasResidualF32"], baseline + ["SnakeDW7SnakeF32"],
                          baseline + ["SnakeDW7SnakeF32", "BiasResidualF32", "BiasResidualF32"]):
            with self.subTest(operators=operators):
                build = self.build_record(operators=operators)
                with self.assertRaisesRegex(ValueError, "operators"):
                    prepare.prepare(self.output, source=self.source, native_build=build, block_fusion="both")
                self.assert_preflight_untouched()
        build = self.build_record()  # Old build records had no operators key.
        with self.assertRaisesRegex(ValueError, "declaring its supported operators"):
            prepare.prepare(self.output, source=self.source, native_build=build, block_fusion="adds")
        self.assert_preflight_untouched()

    def test_default_none_keeps_old_native_recipe_and_does_not_rewrite(self):
        build = self.build_record()
        with self.native_pipeline() as (rewrite, convert, packed):
            result = prepare.prepare(self.output, source=self.source, native_build=build)
        rewrite.assert_not_called()
        packed.assert_not_called()
        self.assertEqual(convert.call_args.args[1], self.output / "decoder_native.onnx")
        entry = result["native"]["Linux/x86_64"]
        self.assertEqual(result["block_fusion"], "none")
        self.assertNotIn("block_fusion", entry)
        self.assertEqual(entry["experiment"], "phase_fused")
        self.assertEqual(entry["tested_cpu"], "AMD EPYC 9654")

    def test_requested_fusions_use_exact_counts_auto_policy_and_hashed_audit(self):
        for system, machine, domain in (("Linux", "x86_64", "venky.audio.cpu.portable"),
                                        ("Darwin", "arm64", "venky.audio.cpu")):
            self.system.return_value, self.machine.return_value = system, machine
            for variant in ("adds", "chain", "both"):
                with self.subTest(platform=system, variant=variant):
                    output = self.root / (system + "_" + variant)
                    operators = sorted(prepare._fusion_operators(variant))
                    build = self.build_record(domain=domain, operators=operators)
                    with self.native_pipeline() as (rewrite, convert, packed):
                        result = prepare.prepare(output, source=self.source, native_build=build, block_fusion=variant)
                    rewrite.assert_called_once()
                    self.assertEqual(rewrite.call_args.kwargs, {
                        "variant": variant, "backend": None,
                        "expected_chains": 18 if variant in ("chain", "both") else 0,
                        "expected_adds": 18 if variant in ("adds", "both") else 0})
                    self.assertNotEqual(rewrite.call_args.args[0], rewrite.call_args.args[1])
                    packed.assert_not_called()
                    if system == "Darwin":
                        convert.assert_not_called()
                    else:
                        convert.assert_called_once()
                        self.assertEqual(convert.call_args.args[1], rewrite.call_args.args[0])
                    entry = result["native"][system + "/" + machine]
                    metadata = entry["block_fusion"]
                    self.assertEqual(result["block_fusion"], variant)
                    self.assertEqual(metadata["variant"], variant)
                    self.assertIsNone(metadata["backend_override"])
                    self.assertEqual(metadata["audit_sha256"], prepare.sha256(output / metadata["audit_file"]))
                    self.assertEqual(metadata["model_sha256"], entry["model_sha256"])
                    self.assertEqual(metadata["native_build_record_sha256"], prepare.sha256(build))
                    self.assertEqual(entry["experiment"], "block_fusion_" + variant)
                    self.assertIn("Not recorded", entry["tested_cpu"])
                    self.assertIn("no numerical, quality or timing validation", metadata["validation"])

    def test_fusion_keeps_amd_packing_explicit_and_uses_the_fused_source(self):
        from fast_audiovae.graph.packed import SELECTED_NODES
        build = self.build_record(operators=sorted(prepare._fusion_operators("both")))
        amd = self.build_record("amd", domain="venky.audio.cpu.aocl.rows")
        with self.native_pipeline() as (fusion, convert, packed):
            result = prepare.prepare(self.output, source=self.source, native_build=build,
                                     amd_build=amd, block_fusion="both")
        packed.assert_called_once_with(self.output / "decoder_native.onnx", self.output / "decoder_amd.onnx", SELECTED_NODES, 4)
        entry = result["native"]["Linux/x86_64"]["packed"]
        self.assertFalse(entry["default"])
        self.assertEqual(entry["validated_threads"], [1, 4])
        self.assertEqual(entry["block_fusion"], "both")
        self.assertEqual(entry["baseline_experiment"], "phase_aocl_rows")
        self.assertIn("Not recorded", entry["tested_cpu"])

    def test_cli_passes_explicit_fusion_and_default_none(self):
        from fast_audiovae import cli
        for flags, variant in (([], "none"), (["--block-fusion", "both"], "both")):
            with patch("sys.argv", ["fast-audiovae", "prepare", *flags]), \
                 patch.object(prepare, "prepare", return_value={}) as selected, patch("builtins.print"):
                cli.main()
            selected.assert_called_once_with("artifacts", source=None, native_build=None,
                                             amd_build=None, block_fusion=variant)


if __name__ == "__main__":
    unittest.main()
