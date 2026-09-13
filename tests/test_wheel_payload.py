"""Build inert platform/pure wheels and sdist; no compiler or native loading."""
from email.parser import Parser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement

from fast_audiovae import build_resources
from test_apple_payload import builds, inert_audit, packager

ROOT = Path(__file__).resolve().parents[1]
GPU_MODULES = ("gpu.py", "mps_decoder.py", "mps_compiled.py")


def assert_optional_gpu_metadata(raw):
    metadata = Parser().parsestr(raw.decode())
    assert "gpu" in metadata.get_all("Provides-Extra", [])
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    torch = [value for value in requirements if value.name.lower() == "torch"]
    assert torch, "The GPU extra must declare its Torch runtime"
    # An encoder extra can also require Torch. Neither optional dependency may
    # become active in a base installation, including another wheel platform.
    for system, machine in (("darwin", "arm64"), ("linux", "x86_64"), ("win32", "AMD64")):
        environment = {**default_environment(), "sys_platform": system, "platform_machine": machine}
        assert all(value.marker is not None and not value.marker.evaluate({**environment, "extra": ""})
                   for value in torch), "Base installation must not require Torch"
        enabled = [value for value in torch if value.marker.evaluate({**environment, "extra": "gpu"})]
        assert len(enabled) == 1, "The GPU extra must select exactly one Torch requirement"
        assert "2.14.0" in enabled[0].specifier
        assert "2.13.0" not in enabled[0].specifier
        assert "2.15.0" not in enabled[0].specifier


def assert_wheel_gpu_sources_and_cpu_import(archive, tree, install):
    # Platform wheels may place Python modules in .data/purelib; resolve the
    # actual package instead of accidentally testing an installed source tree.
    initializers = [name for name in archive.namelist()
                    if name == "fast_audiovae/__init__.py"
                    or name.endswith(".data/purelib/fast_audiovae/__init__.py")
                    or name.endswith(".data/platlib/fast_audiovae/__init__.py")]
    assert len(initializers) == 1
    package_prefix = initializers[0][:-len("__init__.py")]
    for name in GPU_MODULES:
        assert archive.read(package_prefix + name) == (tree / "src/fast_audiovae" / name).read_bytes()
    metadata, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert_optional_gpu_metadata(archive.read(metadata))
    archive.extractall(install)
    import_root = (install / initializers[0]).parent.parent.resolve()
    code = """
import importlib.abc
from pathlib import Path
import sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname in (
                'fast_audiovae.gpu', 'fast_audiovae.mps_decoder', 'fast_audiovae.mps_compiled'):
            raise AssertionError('CPU wheel import reached optional GPU module: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
sys.path.insert(0, sys.argv[1])
import fast_audiovae
import fast_audiovae.automatic
assert Path(fast_audiovae.__file__).resolve() == Path(sys.argv[1]) / 'fast_audiovae' / '__init__.py'
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(import_root)],
                               cwd=install, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_native_wheel_pure_rebuild_and_sdist_have_exact_payload_boundaries(tmp_path):
    tree = tmp_path / "source"
    tree.mkdir()
    for name in ("setup.py", "pyproject.toml", "README.md"):
        shutil.copyfile(ROOT / name, tree / name)
    for path in (ROOT / "src/fast_audiovae").rglob("*.py"):
        if "_build_resources" in path.parts:
            continue
        target = tree / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for path in build_resources.resource_files(ROOT):
        target = tree / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    a, b = builds(tmp_path / "inert-inputs")
    payload = tmp_path / "payload"
    manifest = packager.package(a, b, payload, inspect_library=inert_audit)
    env = dict(os.environ, FAST_AUDIOVAE_NATIVE_PAYLOAD=str(payload))
    def build(command, destination, environment):
        subprocess.run([sys.executable, "setup.py", command, "--dist-dir", str(destination)],
                       cwd=tree, env=environment, check=True, capture_output=True, text=True)
    build("bdist_wheel", tmp_path / "platform-dist", env)
    platform_wheel, = (tmp_path / "platform-dist").glob("*.whl")
    assert platform_wheel.name.endswith("-py3-none-macosx_26_0_arm64.whl")
    with zipfile.ZipFile(platform_wheel) as archive:
        assert_wheel_gpu_sources_and_cpu_import(archive, tree, tmp_path / "platform-install")
        wheel_metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
        assert "Root-Is-Purelib: false" in wheel_metadata
        assert "Tag: py3-none-macosx_26_0_arm64" in wheel_metadata
        names = [n for n in archive.namelist() if n.endswith("fast_audiovae/_native/manifest.json")]
        assert len(names) == 1
        prefix = names[0][:-len("manifest.json")]
        assert {n[len(prefix):] for n in archive.namelist() if n.startswith(prefix)} == set(manifest["files"]) | {"manifest.json"}
        for name in manifest["files"]:
            assert archive.read(prefix + name) == (payload / name).read_bytes()
        assert json.loads(archive.read(prefix + "manifest.json")) == manifest
    # Reuse the exact same source/build directory: stale native files must not
    # leak into a later pure wheel when the payload environment is removed.
    env.pop("FAST_AUDIOVAE_NATIVE_PAYLOAD")
    build("bdist_wheel", tmp_path / "pure-dist", env)
    pure_wheel, = (tmp_path / "pure-dist").glob("*.whl")
    assert pure_wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(pure_wheel) as archive:
        assert_wheel_gpu_sources_and_cpu_import(archive, tree, tmp_path / "pure-install")
        assert not any("/_native/" in n for n in archive.namelist())
        assert any(n.endswith("/_build_resources/native/apple/streaming/sources.json") for n in archive.namelist())
    # Even an explicitly configured binary payload must not enter an sdist.
    build("sdist", tmp_path / "source-dist", {**env, "FAST_AUDIOVAE_NATIVE_PAYLOAD": str(payload)})
    source_dist, = (tmp_path / "source-dist").glob("*.tar.gz")
    with tarfile.open(source_dist) as archive:
        names = archive.getnames()
        assert not any("/_native/" in n or n.endswith((".dylib", ".so", ".onnx", ".npz")) for n in names)
        assert any(n.endswith("native/apple/streaming/sources.json") for n in names)
        root, = {name.split("/", 1)[0] for name in names}
        for name in GPU_MODULES:
            with archive.extractfile(f"{root}/src/fast_audiovae/{name}") as handle:
                assert handle.read() == (tree / "src/fast_audiovae" / name).read_bytes()
        with archive.extractfile(f"{root}/PKG-INFO") as handle:
            assert_optional_gpu_metadata(handle.read())
