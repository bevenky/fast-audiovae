"""Build inert platform/pure wheels and sdist; no compiler or native loading."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from fast_audiovae import build_resources
from test_apple_payload import builds, inert_audit, packager

ROOT = Path(__file__).resolve().parents[1]


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
        assert not any("/_native/" in n for n in archive.namelist())
        assert any(n.endswith("/_build_resources/native/apple/streaming/sources.json") for n in archive.namelist())
    # Even an explicitly configured binary payload must not enter an sdist.
    build("sdist", tmp_path / "source-dist", {**env, "FAST_AUDIOVAE_NATIVE_PAYLOAD": str(payload)})
    source_dist, = (tmp_path / "source-dist").glob("*.tar.gz")
    with tarfile.open(source_dist) as archive:
        names = archive.getnames()
        assert not any("/_native/" in n or n.endswith((".dylib", ".so", ".onnx", ".npz")) for n in names)
        assert any(n.endswith("native/apple/streaming/sources.json") for n in names)
