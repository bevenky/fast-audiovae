"""CPU selection tests use synthetic evidence; no native probes or inference."""
import subprocess
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fast_audiovae import platforms


def evidence(vendor="intel", *, system="Linux", machine="x86_64", maximum=8, proven=True):
    return {"vendor": vendor, "system": system, "machine": machine,
            "platform": system + "/" + machine, "max_threads": maximum,
            "usable": {"neon": True, "avx2": True, "avx512": True, "avx512_vnni": True},
            "probe": {"status": "ok" if proven else "unavailable"}}


class SelectionTests(unittest.TestCase):
    def test_mode_and_vendor_select_separate_recipes(self):
        cases = [
            (evidence("apple", system="Darwin", machine="arm64"), "apple_native", "apple_stream_projection"),
            (evidence("intel"), "intel_precision", "intel_stream_projection"),
            (evidence("amd"), "amd_precision", "amd_precision"),
            (evidence("unknown"), "portable", "portable"),
            (evidence("mixed"), "portable", "portable"),
            (evidence("apple", system="Darwin", machine="x86_64"), "portable", "portable"),
            (evidence("amd", system="Windows"), "portable", "portable"),
            (evidence("unknown", machine="arm64"), "portable", "portable"),
        ]
        for cpu, batch, streaming in cases:
            with self.subTest(cpu=cpu):
                self.assertEqual(platforms.select_recipe(cpu, "batch")["recipe"], batch)
                self.assertEqual(platforms.select_recipe(cpu, "streaming")["recipe"], streaming)
                self.assertEqual(platforms.select_recipe(cpu)["recipe"], streaming)
                self.assertEqual(platforms.select_recipe(cpu)["threads"], 1)
                self.assertEqual(platforms.select_recipe(cpu, threads=None)["threads"], 1)

    def test_vendor_and_cpu_flags_without_os_proof_are_insufficient(self):
        for cpu in (evidence(proven=False), evidence("amd", proven=False)):
            self.assertEqual(platforms.select_recipe(cpu, "streaming")["recipe"], "portable")
        for feature in ("avx2", "avx512", "avx512_vnni"):
            for value in (False, None, "true", 1):
                cpu = evidence()
                cpu["usable"][feature] = value
                self.assertEqual(platforms.select_recipe(cpu)["recipe"], "portable")

    def test_validated_worker_schedules_and_cpu_budget(self):
        self.assertEqual(platforms.select_recipe(evidence(), "batch", threads=8)["threads"], 2)
        self.assertEqual(platforms.select_recipe(evidence(), "streaming", threads=8)["threads"], 1)
        self.assertEqual(platforms.select_recipe(evidence("amd"), threads=8)["threads"], 4)
        self.assertEqual(platforms.select_recipe(evidence("amd", maximum=3), threads=8)["threads"], 1)
        self.assertEqual(platforms.select_recipe(evidence("amd"), "streaming", threads=3)["threads"], 1)
        apple = evidence("apple", system="Darwin", machine="arm64")
        self.assertEqual(platforms.select_recipe(apple, "batch", threads=4)["threads"], 4)
        self.assertEqual(platforms.select_recipe(apple, "batch", threads=2)["threads"], 1)
        self.assertEqual(platforms.select_recipe(apple, "streaming", threads=4)["threads"], 1)
        portable = platforms.select_recipe(evidence("unknown", maximum=3), threads=8)
        self.assertEqual(portable["threads"], 3)
        self.assertTrue(portable["threads_capped"])
        self.assertIsNone(portable["validated_threads"])

    def test_invalid_overrides_rejected(self):
        for threads in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                platforms.select_recipe(evidence(), threads=threads)
        for mode in ("full", "stream", None, "gpu"):
            with self.assertRaises(ValueError):
                platforms.select_recipe(evidence(), mode=mode)


class CgroupTests(unittest.TestCase):
    def quota(self, files):
        with patch.object(platforms, "_read", side_effect=lambda path: files.get(str(path))):
            return platforms._cgroup_quota()

    def test_v2_parent_limit_and_fractional_quota(self):
        files = {"/proc/self/cgroup": "0::/tenant/worker\n",
                 "/proc/self/mountinfo": "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
                 "/sys/fs/cgroup/tenant/worker/cpu.max": "max 100000",
                 "/sys/fs/cgroup/tenant/cpu.max": "250000 100000",
                 "/sys/fs/cgroup/cpu.max": "400000 100000"}
        self.assertEqual(self.quota(files), 2.5)
        files["/sys/fs/cgroup/tenant/worker/cpu.max"] = "50000 100000"
        self.assertEqual(self.quota(files), .5)

    def test_v1_mount_root_mapping_and_parent_limit(self):
        files = {"/proc/self/cgroup": "3:cpu,cpuacct:/docker/abc/job\n",
                 "/proc/self/mountinfo": "39 25 0:35 /docker/abc /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
                 "/sys/fs/cgroup/cpu/job/cpu.cfs_quota_us": "-1",
                 "/sys/fs/cgroup/cpu/job/cpu.cfs_period_us": "100000",
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "150000",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}
        self.assertEqual(self.quota(files), 1.5)

    def test_unreadable_malformed_unlimited_or_missing_quota(self):
        for value in (None, "max 100000", "100000 0", "bad value", "-1 100000", "100000"):
            self.assertIsNone(self.quota({"/sys/fs/cgroup/cpu.max": value}))

    def test_mount_escapes_and_outside_membership(self):
        locations = platforms._cgroup_locations("0::/tenant/work\n",
            "36 25 0:32 /tenant /sys/fs/cgroup\\040space rw - cgroup2 cgroup rw\n")
        self.assertEqual(str(locations[0][1]), "/sys/fs/cgroup space/work")
        self.assertEqual(platforms._cgroup_locations("0::/other\n",
            "36 25 0:32 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"), [])
        self.assertEqual(platforms._cgroup_locations("0::/../../outside\n",
            "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"), [])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        platforms._cached_x86_probe.cache_clear()

    def tearDown(self):
        platforms._cached_x86_probe.cache_clear()

    def test_no_compiler_never_runs_an_instruction_probe(self):
        with patch.object(platforms.shutil, "which", return_value=None), patch.object(platforms.subprocess, "run") as run, \
             patch.object(platforms.platform, "system", return_value="Linux"), \
             patch.object(platforms.platform, "machine", return_value="x86_64"):
            result, info = platforms._x86_probe([0])
        self.assertEqual(result, {})
        self.assertEqual(info["status"], "unavailable")
        run.assert_not_called()

    def probe(self, response, affinity=(0, 2)):
        with patch.object(platforms, "build_probe"), \
             patch.object(platforms.subprocess, "run", return_value=response) as run:
            result = platforms._x86_probe(list(affinity))
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("shell", run.call_args_list[0].kwargs)
        return result

    def test_features_verified_for_exact_allowed_cpu_set(self):
        import json
        expected = {"vendor": "intel", "avx2": True, "avx512": True, "avx512_vnni": True, "cpus": [0, 2]}
        result, info = self.probe(SimpleNamespace(returncode=0, stdout=json.dumps(expected)))
        self.assertEqual(result, expected)
        self.assertEqual(info["status"], "ok")
        for replacement in ({"cpus": [0]}, {"cpus": [0, 0]}, {"avx512": False}, {"avx512_vnni": 1}):
            result, info = self.probe(SimpleNamespace(returncode=0, stdout=json.dumps(dict(expected, **replacement))))
            self.assertEqual(result, {})
            self.assertEqual(info["status"], "unavailable")

    def test_crash_timeout_and_malformed_probe_are_conservative(self):
        for response in (SimpleNamespace(returncode=-4, stdout=""), SimpleNamespace(returncode=0, stdout="not-json")):
            self.assertEqual(self.probe(response)[0], {})
        with patch.object(platforms, "build_probe"), \
             patch.object(platforms.subprocess, "run", side_effect=subprocess.TimeoutExpired("cc", 15)):
            self.assertEqual(platforms._x86_probe([0])[0], {})

    def test_detect_normalizes_architecture_rechecks_quota_and_caches_isa(self):
        probe = ({"vendor": "amd", "avx2": True, "avx512": True, "avx512_vnni": True, "cpus": [1, 2, 3, 4]},
                 {"status": "ok", "method": "child_cpuid_xgetbv", "cpus_checked": [1, 2, 3, 4]})
        with patch.object(platforms.platform, "system", return_value="Linux"), \
             patch.object(platforms.platform, "machine", return_value="AMD64"), \
             patch.object(platforms.os, "cpu_count", return_value=32), \
             patch.object(platforms.os, "sched_getaffinity", return_value={1, 2, 3, 4}, create=True), \
             patch.object(platforms, "_cgroup_quota", side_effect=[2.5, .5]), \
             patch.object(platforms, "_packaged_probe", return_value=(None, "absent")), \
             patch.object(platforms, "_x86_probe", return_value=probe) as native, \
             patch.object(platforms, "_read", return_value="model name : AMD EPYC\n"):
            first = platforms.detect_cpu(allow_compile=True)
            first["probe"]["cpus_checked"].append(99)
            second = platforms.detect_cpu(allow_compile=True)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(first["machine"], "x86_64")
        self.assertEqual(first["max_threads"], 2)
        self.assertEqual(second["max_threads"], 1)
        self.assertEqual(second["probe"]["cpus_checked"], [1, 2, 3, 4])
        self.assertEqual(second["cpu_budget"], .5)

    def test_affinity_change_requires_new_proof(self):
        with patch.object(platforms, "_x86_probe", return_value=({}, {"status": "unavailable"})) as native:
            platforms._cached_x86_probe((0,))
            platforms._cached_x86_probe((0,))
            platforms._cached_x86_probe((1,))
        self.assertEqual(native.call_count, 2)

    def test_apple_native_abi_and_rosetta_are_distinct(self):
        with patch.object(platforms.platform, "system", return_value="Darwin"), \
             patch.object(platforms.platform, "machine", side_effect=["aarch64", "aarch64", "x86_64", "x86_64"]), \
             patch.object(platforms.os, "cpu_count", return_value=16), \
             patch.object(platforms.os, "sched_getaffinity", side_effect=AttributeError, create=True), \
             patch.object(platforms.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Apple M5 Max\n")), \
             patch.object(platforms, "_x86_probe") as native:
            apple, rosetta = platforms.detect_cpu(), platforms.detect_cpu()
        self.assertEqual(platforms.select_recipe(apple, "streaming")["recipe"], "apple_stream_projection")
        self.assertEqual(platforms.select_recipe(rosetta, "streaming")["recipe"], "portable")
        native.assert_not_called()


class PackagedProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "cpu_probe"
        self.binary.write_bytes(b"synthetic packaged probe, never executed")
        self.digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.write_manifest()
        platforms._cached_x86_probe.cache_clear()

    def tearDown(self):
        platforms._cached_x86_probe.cache_clear()
        self.temp.cleanup()

    def write_manifest(self, **record):
        value = {"path": "cpu_probe", "sha256": self.digest}
        value.update(record)
        (self.root / "manifest.json").write_text(json.dumps({"version": 1, "wheel_platform": "linux_x86_64",
            "files": {"cpu_probe": self.digest}, "probe": value}))

    def test_verified_packaged_probe_runs_without_compilation(self):
        response = {"vendor": "intel", "avx2": True, "avx512": True, "avx512_vnni": True, "cpus": [0]}
        with patch.object(platforms, "_NATIVE_DIR", self.root), \
             patch.object(platforms, "build_probe") as build, \
             patch.object(platforms.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(response))) as run:
            record, status = platforms._packaged_probe()
            self.assertEqual(status, "verified")
            result, info = platforms._x86_probe([0], record)
        build.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], [str(self.binary.resolve())])
        self.assertEqual(result, response)
        self.assertEqual(info["method"], "packaged_child_cpuid_xgetbv")
        self.assertEqual(info["probe_sha256"], self.digest)

    def test_corrupt_digest_and_paths_are_rejected_before_execution(self):
        for replacement in ({"sha256": "0" * 64}, {"path": "../outside"}, {"path": str(self.binary)},
                            {"sha256": "BAD"}, {"path": "missing"}):
            self.write_manifest(**replacement)
            with patch.object(platforms, "_NATIVE_DIR", self.root), patch.object(platforms.subprocess, "run") as run:
                record, status = platforms._packaged_probe()
            self.assertIsNone(record)
            self.assertEqual(status, "invalid")
            run.assert_not_called()

    def test_corrupt_package_does_not_fall_back_to_compiler(self):
        self.binary.write_bytes(b"corrupt")
        with patch.object(platforms, "_NATIVE_DIR", self.root), \
             patch.object(platforms.platform, "system", return_value="Linux"), \
             patch.object(platforms.platform, "machine", return_value="x86_64"), \
             patch.object(platforms.os, "sched_getaffinity", return_value={0}, create=True), \
             patch.object(platforms, "_cgroup_quota", return_value=None), \
             patch.object(platforms, "_read", return_value=None), \
             patch.object(platforms, "_x86_probe") as probe, patch.object(platforms, "build_probe") as build:
            result = platforms.detect_cpu()
        build.assert_not_called()
        probe.assert_not_called()
        self.assertEqual(platforms.select_recipe(result)["recipe"], "portable")
        self.assertIn("failed verification", result["probe"]["reason"])

    def test_no_packaged_probe_never_compiles_for_ordinary_users(self):
        with patch.object(platforms, "_packaged_probe", return_value=(None, "absent")), \
             patch.object(platforms.platform, "system", return_value="Linux"), \
             patch.object(platforms.platform, "machine", return_value="x86_64"), \
             patch.object(platforms.os, "sched_getaffinity", return_value={0}, create=True), \
             patch.object(platforms, "_cgroup_quota", return_value=None), \
             patch.object(platforms, "_read", return_value=None), \
             patch.object(platforms, "_x86_probe") as probe, patch.object(platforms, "build_probe") as build:
            result = platforms.detect_cpu()
        build.assert_not_called()
        probe.assert_not_called()
        self.assertEqual(platforms.select_recipe(result)["recipe"], "portable")
        self.assertIn("without compiling", result["probe"]["reason"])

    def test_identity_change_invalidates_cached_proof(self):
        with patch.object(platforms, "_x86_probe", return_value=({}, {"status": "unavailable"})) as probe:
            platforms._cached_x86_probe((0,), "path", "first", "manifest")
            platforms._cached_x86_probe((0,), "path", "first", "manifest")
            platforms._cached_x86_probe((0,), "path", "second", "newmanifest")
        self.assertEqual(probe.call_count, 2)

    def test_packaged_probe_requires_inventory_platform_and_plain_files(self):
        original = json.loads((self.root / "manifest.json").read_text())
        for change in ({"version": True}, {"version": 2}, {"wheel_platform": "any"},
                       {"files": {}}, {"files": {"cpu_probe": "0" * 64}}):
            (self.root / "manifest.json").write_text(json.dumps(dict(original, **change)))
            with patch.object(platforms, "_NATIVE_DIR", self.root):
                self.assertEqual(platforms._packaged_probe(), (None, "invalid"))
        (self.root / "manifest.json").write_text(json.dumps(original))
        target = self.root / "real_probe"
        self.binary.rename(target)
        self.binary.symlink_to(target)
        with patch.object(platforms, "_NATIVE_DIR", self.root):
            self.assertEqual(platforms._packaged_probe(), (None, "invalid"))
        self.binary.unlink()
        target.rename(self.binary)
        manifest = self.root / "manifest.json"
        manifest.rename(self.root / "real_manifest.json")
        manifest.symlink_to(self.root / "real_manifest.json")
        with patch.object(platforms, "_NATIVE_DIR", self.root):
            self.assertEqual(platforms._packaged_probe(), (None, "invalid"))

    def test_maintainer_builder_only_compiles_and_returns_manifest_record(self):
        destination = self.root / "build"
        def compile_only(command, **kwargs):
            self.assertIn("-march=x86-64", command)
            self.assertIn("-mtune=generic", command)
            self.assertNotIn("shell", kwargs)
            Path(command[-1]).write_bytes(b"synthetic compiled artifact")
            return SimpleNamespace(returncode=0)
        with patch.object(platforms.platform, "system", return_value="Linux"), \
             patch.object(platforms.platform, "machine", return_value="x86_64"), \
             patch.object(platforms.shutil, "which", return_value="/usr/bin/cc"), \
             patch.object(platforms.subprocess, "run", side_effect=compile_only) as run:
            record = platforms.build_probe(destination)
            self.assertEqual(run.call_count, 1)
            with self.assertRaises(FileExistsError):
                platforms.build_probe(destination)
        self.assertEqual(record["path"], "cpu_probe")
        self.assertEqual(record["sha256"], hashlib.sha256((destination / "cpu_probe").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
