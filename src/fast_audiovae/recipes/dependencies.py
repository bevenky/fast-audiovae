"""Pinned user-cache dependencies for the existing x86 CPU recipes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

from ..assets import download, sha256


def cpu_environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
               HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               BLIS_NUM_THREADS="1")
    # Build subprocesses own their environment. The embedding process is untouched.
    for name in ("LIBXSMM_TARGET", "MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS",
                 "AOCL_DLP_ENABLE_INSTRUCTIONS"):
        env.pop(name, None)
    return env


def run(command, cwd, commands):
    command = [str(value) for value in command]
    commands.append(command)
    log = Path(cwd) / ".build" / "automatic-setup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("\n" + repr(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=cpu_environment(),
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"CPU setup command failed; see {log}")


def fetch(url, destination, digest, *, offline):
    destination = Path(destination)
    if offline and not destination.is_file():
        raise RuntimeError("Offline setup needs cached dependency: " + destination.name)
    return download(url, destination, digest)


def unpack(archive, destination):
    """Extract a verified source archive once, rejecting unsafe members."""
    destination = Path(destination)
    marker = destination / ".source-archive-sha256"
    digest = sha256(archive)
    if marker.exists():
        if marker.read_text().strip() != digest:
            raise RuntimeError("Cached source archive identity changed")
        return destination
    if destination.exists():
        raise RuntimeError("Incomplete dependency source cache: " + str(destination))
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
            if len(roots) != 1:
                raise ValueError("Expected one top-level source directory")
            links = []
            for member in members:
                parts = Path(member.name).parts
                if (Path(member.name).is_absolute() or ".." in parts
                        or not (member.isdir() or member.isfile() or member.issym())):
                    raise ValueError("Unsupported source archive member: " + member.name)
                relative = Path(*parts[1:])
                if not relative.parts:
                    continue
                target = destination / relative
                if member.issym():
                    linked = (target.parent / member.linkname).resolve()
                    if Path(member.linkname).is_absolute() or not linked.is_relative_to(destination.resolve()):
                        raise ValueError("Source symlink leaves the cache")
                    links.append((target, member.linkname))
                elif member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    target.chmod(member.mode & 0o777)
            for target, link in links:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link)
        marker.write_text(digest + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


def file_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "dependency-receipt.json"}


def _reuse(root, identity):
    receipt = root / "dependency-receipt.json"
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record.get("identity") != identity:
        raise RuntimeError("Dependency cache uses a different pinned recipe")
    for name, digest in record["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != digest:
            raise RuntimeError("Cached dependency changed: " + name)
    return True


def _receipt(root, identity, commands, files):
    record = {"identity": identity, "commands": commands, "files": files,
              "cpu_only": True, "historical_binary_identity_claim": False}
    (root / "dependency-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def libxsmm(work, *, offline, jobs, commands):
    pin = json.loads((work / "experiments/intel-precision/pins/dependencies.json").read_text())["libxsmm"]
    root = work / ".deps/automatic-libxsmm"
    identity = {"commit": pin["commit"], "archive_sha256": pin["archive_sha256"], "static": True}
    if _reuse(root, identity):
        return root / "source"
    root.mkdir(parents=True, exist_ok=True)
    archive = fetch(pin["archive_url"], root / "source.tar.gz", pin["archive_sha256"], offline=offline)
    source = unpack(archive, root / "source")
    local_commands = []
    run(["make", "-C", source, "-j" + str(jobs), "STATIC=1", "lib"], work, local_commands)
    commands.extend(local_commands)
    library = source / "lib/libxsmm.a"
    if not library.is_file():
        raise RuntimeError("LIBXSMM did not produce its static CPU archive")
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(source.rglob("*"))
             if p.is_file() and (p.suffix in (".h", ".c", ".py", ".sh") or p == library)}
    files["source.tar.gz"] = sha256(archive)
    _receipt(root, identity, local_commands, files)
    return source


def aocl(work, *, offline, jobs, commands):
    pin = json.loads((work / "experiments/amd-precision/pins/rebuild.json").read_text())
    source_pin = pin["AOCL_source_pin"]
    root = work / ".deps/automatic-aocl-dlp"
    identity = {"commit": source_pin["commit"], "archive_sha256": source_pin["archive_sha256"],
                "configuration": pin["AOCL_CMake_configuration"]}
    if _reuse(root, identity):
        return root / "install"
    root.mkdir(parents=True, exist_ok=True)
    url = "https://codeload.github.com/amd/aocl-dlp/tar.gz/" + source_pin["commit"]
    archive = fetch(url, root / "source.tar.gz", source_pin["archive_sha256"], offline=offline)
    source = unpack(archive, root / "source")
    install = root / "install"
    local_commands = []
    run(["cmake", "-S", source, "-B", root / "build", "-DCMAKE_INSTALL_PREFIX=" + str(install),
         *["-D" + key + "=" + value for key, value in pin["AOCL_CMake_configuration"].items()]],
        work, local_commands)
    run(["cmake", "--build", root / "build", "--parallel", jobs], work, local_commands)
    run(["cmake", "--install", root / "build"], work, local_commands)
    commands.extend(local_commands)
    if not (install / "lib/libaocl-dlp.so").is_file():
        raise RuntimeError("AOCL-DLP did not produce its CPU library")
    files = {"source.tar.gz": sha256(archive)}
    files.update({str(p.relative_to(root)): sha256(p) for p in sorted(install.rglob("*")) if p.is_file()})
    _receipt(root, identity, local_commands, files)
    return install


def mkl(work, *, offline, commands):
    pins = json.loads((work / "experiments/intel-precision/pins/mkl.json").read_text())
    root = work / ".deps/automatic-mkl"
    identity = {"version": pins["version"], "packages": pins["packages"], "threading": "sequential_lp64"}
    if _reuse(root, identity):
        return root
    root.mkdir(parents=True, exist_ok=True)
    (root / "include").mkdir(exist_ok=True)
    (root / "lib").mkdir(exist_ok=True)
    # Retain the complete relevant CPU dispatch families, including newer ISA
    # dispatch files, rather than the original VM's eleven-library subset.
    cpu_library = re.compile(r"libmkl_(?:intel_lp64|sequential|core|def|mc3|avx[0-9][a-z0-9_]*|vml_[a-z0-9_]+)\.so(?:\.\d+)*$")
    for package in pins["packages"]:
        archive_path = fetch(package["url"], root / package["filename"], package["sha256"], offline=offline)
        if archive_path.stat().st_size != package["wheel_bytes"]:
            raise ValueError("Pinned oneMKL package length changed")
        with zipfile.ZipFile(archive_path) as archive:
            if package["filename"].startswith("mkl_include-"):
                by_name = {}
                for member in archive.namelist():
                    if member.endswith(".h"):
                        by_name.setdefault(Path(member).name, []).append(member)
                pending, seen = list(pins["header_sha256"]), set()
                while pending:
                    name = pending.pop()
                    if name in seen:
                        continue
                    if Path(name).name != name or len(by_name.get(name, [])) != 1:
                        raise ValueError("Ambiguous oneMKL header: " + name)
                    data = archive.read(by_name[name][0])
                    (root / "include" / name).write_bytes(data)
                    pending.extend(re.findall(r'^\s*#\s*include\s*"([^"]+)"', data.decode(), re.M))
                    seen.add(name)
            else:
                seen = set()
                for member in archive.namelist():
                    name = Path(member).name
                    if not cpu_library.fullmatch(name):
                        continue
                    if name in seen:
                        raise ValueError("Duplicate oneMKL CPU library: " + name)
                    seen.add(name)
                    with archive.open(member) as src, (root / "lib" / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    for directory, expected in (("include", pins["header_sha256"]), ("lib", pins["cpu_library_sha256"])):
        for name, digest in expected.items():
            if sha256(root / directory / name) != digest:
                raise ValueError("Pinned oneMKL member hash changed: " + name)
    _receipt(root, identity, [], file_hashes(root))
    return root
