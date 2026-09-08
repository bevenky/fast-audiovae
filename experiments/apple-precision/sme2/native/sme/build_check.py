"""Compile the pinned SME2 closure and run small exactness checks, without timing."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL = ROOT / "upstream/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp"
NAME = "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--compile-only", action="store_true")
    a = p.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Apple ARM required")
    out = a.output_dir.resolve()
    if out.exists():
        raise ValueError("Fresh output directory required")
    pins = json.loads((ROOT / "upstream-provenance.json").read_text())
    for name, record in pins["files"].items():
        if sha(ROOT / name) != record["sha256"]:
            raise ValueError("Upstream source changed: " + name)
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    flags = ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
             "-isysroot", sdk, "-I" + str(ROOT / "upstream")]
    commands = []
    objects = []
    out.mkdir(parents=True)
    for name, src, is_sme in [("wrapper", ROOT / "sme_backend.cpp", False),
                             ("kernel_c", KERNEL / (NAME + ".c"), True),
                             ("kernel_asm", KERNEL / (NAME + "_asm.S"), True),
                             ("common_sme", ROOT / "upstream/kai/kai_common_sme_asm.S", True)]:
        obj = out / (name + ".o")
        cxx = src.suffix == ".cpp"
        command = ["c++" if cxx else "cc", *flags]
        if cxx:
            command += ["-std=c++17", "-isystem", str(Path(sdk) / "usr/include/c++/v1")]
        elif src.suffix == ".c":
            command += ["-std=c11"]
        if is_sme:
            command += ["-march=armv9.2-a+sme2"]
        command += ["-c", str(src), "-o", str(obj)]
        subprocess.run(command, check=True); commands.append(command); objects.append(obj)
    command = ["c++", *flags, "-std=c++17", "-isystem", str(Path(sdk) / "usr/include/c++/v1"),
               str(ROOT / "check_sme.cpp"), *map(str, objects), "-o", str(out / "check_sme")]
    subprocess.run(command, check=True); commands.append(command)
    record = {"status": "compiled", "commands": commands,
              "objects": {str(x): sha(x) for x in objects}, "upstream_commit": pins["commit"],
              "source_sha256": {str(x.relative_to(ROOT)): sha(x) for x in ROOT.rglob("*")
                                if x.is_file() and out not in x.parents and x.suffix in {".cpp", ".c", ".h", ".S", ".py"}},
              "GPU_used": False, "timings_collected": False}
    if not a.compile_only:
        result = subprocess.run([str(out / "check_sme")], check=True, text=True, capture_output=True)
        record["check"] = json.loads(result.stdout); record["status"] = "passed"
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "output": str(out), "check": record.get("check")}))


if __name__ == "__main__":
    main()
