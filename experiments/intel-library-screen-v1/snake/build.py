"""Build the optional Intel oneMKL HA Snake screen; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "mkl-prefix", "native-library", "bundle", "graph", "history-library", "phase-library", "latents"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert "GenuineIntel" in Path("/proc/cpuinfo").read_text()
    a.output.mkdir(parents=True, exist_ok=False)
    graph_manifest = json.loads((a.graph.parent / "manifest.json").read_text())
    assert a.graph.name == graph_manifest["variants"]["combined"]["model"]
    assert sha(a.graph) == graph_manifest["variants"]["combined"]["sha256"]
    bundle = json.loads((a.bundle / "bundle.json").read_text())
    native = bundle["native"]["Linux/x86_64"]
    stream = bundle["streaming"]["models"][native["model"]]
    libs = [a.bundle / native["library"]] + [a.bundle / row["library"] for row in stream["additional_libraries"]]
    assert libs[0].resolve() == a.native_library.resolve()
    assert sha(libs[0]) == native["library_sha256"]
    for path, record in zip(libs[1:], stream["additional_libraries"]): assert sha(path) == record["sha256"]
    libs += [a.history_library, a.phase_library]
    library = (a.output / "libintel_snake_screen.so").resolve()
    mkl = (a.mkl_prefix / "lib/libmkl_intel_lp64.so.3").resolve()
    pins = {str(path.resolve()): sha(path) for path in [a.graph, a.graph.parent / "weights.bin", a.latents,
            *libs, mkl, a.mkl_prefix / "lib/libmkl_core.so.3", a.mkl_prefix / "lib/libmkl_sequential.so.3",
            HERE / "snake_screen.cpp", HERE / "build.py", HERE / "run.py"]}
    assert sha(a.graph.parent / "weights.bin") == graph_manifest["shared_weights"]["sha256"]
    for path in sorted((a.mkl_prefix / "include").glob("mkl*vml*.h")): pins[str(path.resolve())] = sha(path)
    command = ["g++", "-shared", "-fPIC", "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off",
               "-fvisibility=hidden", "-I"+str(a.mkl_prefix / "include"), str(HERE / "snake_screen.cpp"),
               "-Wl,-z,defs", "-ldl", "-pthread", "-o", str(library)]
    completed = subprocess.run(command, capture_output=True, text=True)
    result = {"version": 1, "status": "built" if completed.returncode == 0 else "failed", "ort": "1.29.0",
              "graph": str(a.graph.resolve()), "latents": str(a.latents.resolve()), "libraries": [str(x.resolve()) for x in libs],
              "resolvers": {"SK_NATIVE_LIBRARY": str(a.native_library.resolve()), "SK_MKL_LIBRARY": str(mkl)},
              "pins": pins, "shim": str(library), "command": command, "stdout": completed.stdout,
              "stderr": completed.stderr, "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]}
    if completed.returncode == 0: result["pins"][str(library)] = sha(library)
    (a.output / "build.json").write_text(json.dumps(result, indent=2)+"\n")
    if completed.returncode: raise RuntimeError("Build failed; see build.json")
    print(json.dumps({"build": str(a.output / "build.json"), "shim": str(library)}))


if __name__ == "__main__": main()
