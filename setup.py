"""Package source recipes and optional maintainer-built CPU libraries."""
import os
from pathlib import Path
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
resources = runpy.run_path(str(ROOT / "src/fast_audiovae/build_resources.py"))
PAYLOAD = os.environ.get("FAST_AUDIOVAE_NATIVE_PAYLOAD")
PAYLOAD_ROOT = Path(PAYLOAD).resolve() if PAYLOAD else None
PAYLOAD_MANIFEST = resources["validate_native_payload"](PAYLOAD_ROOT) if PAYLOAD_ROOT else None


class BuildWithResources(build_py):
    def get_source_files(self):
        # Only maintained source files enter an sdist. External binary payloads
        # are supplied by maintainers when building a platform wheel.
        return super().get_source_files() + [str(path.relative_to(ROOT))
            for path in resources["resource_files"](ROOT)]

    def get_outputs(self, include_bytecode=1):
        answer = super().get_outputs(include_bytecode) + [str(Path(self.build_lib) /
            "fast_audiovae/_build_resources" / name) for name in resources["RESOURCE_FILES"]]
        if PAYLOAD_MANIFEST:
            answer += [str(Path(self.build_lib) / "fast_audiovae/_native" / name)
                       for name in ["manifest.json", *PAYLOAD_MANIFEST["files"]]]
        return answer

    def run(self):
        super().run()
        destination = Path(self.build_lib) / "fast_audiovae/_build_resources"
        # Stale resources from previous builds must not leak into either wheel.
        if destination.exists():
            shutil.rmtree(destination)
        for source in resources["resource_files"](ROOT):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        native = Path(self.build_lib) / "fast_audiovae/_native"
        if native.exists():
            shutil.rmtree(native)
        if PAYLOAD_ROOT:
            current = resources["validate_native_payload"](PAYLOAD_ROOT)
            if current != PAYLOAD_MANIFEST:
                raise RuntimeError("Native payload manifest changed during wheel build")
            for name in ["manifest.json", *current["files"]]:
                target = native / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = PAYLOAD_ROOT / name
                shutil.copyfile(source, target)
                # A packaged CPU probe is executed in an isolated child. Keep
                # executable permission while discarding special mode bits.
                target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
            # Verify the copied bytes, not only their source files.
            resources["validate_native_payload"](native)


class CPUWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if PAYLOAD_MANIFEST:
            self.root_is_pure = False

    def get_tag(self):
        if PAYLOAD_MANIFEST:
            return "py3", "none", PAYLOAD_MANIFEST["wheel_platform"]
        return super().get_tag()


setup(cmdclass={"build_py": BuildWithResources, "bdist_wheel": CPUWheel})
