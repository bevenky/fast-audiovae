"""Conservative CPU recipe selection without importing an inference runtime.

CPUID and guarded XGETBV execute only in a short-lived Linux child process.
Missing compiler, denied execution, mixed CPUs or an incomplete probe select
the portable recipe. The actual model loader must still validate its native
libraries and their ABI; CPU support does not prove that a library is present.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import platform
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile


_X86_PROBE = r"""
#define _GNU_SOURCE
#include <cpuid.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned usable(void) {
    unsigned a,b,c,d;
    if (!__get_cpuid(1,&a,&b,&c,&d)) return 0;
    const unsigned required=(1u<<26)|(1u<<27)|(1u<<28);
    if ((c&required)!=required) return 0;
    const unsigned fma=c&(1u<<12);
    uint32_t lo,hi;
    __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));
    (void)hi;
    if ((lo&6u)!=6u || !__get_cpuid_count(7,0,&a,&b,&c,&d)) return 0;
    unsigned flags=(b&(1u<<5))?1:0;
    const unsigned required512=(1u<<5)|(1u<<16)|(1u<<17)|(1u<<30)|(1u<<31);
    if ((lo&0xe6u)==0xe6u && fma && (b&required512)==required512) {
        flags|=2;
        if (c&(1u<<11)) flags|=4;
    }
    return flags;
}

static int vendor(void) {
    unsigned a,b,c,d;
    if (!__get_cpuid(0,&a,&b,&c,&d)) return 0;
    char text[13]={0};
    memcpy(text,&b,4);memcpy(text+4,&d,4);memcpy(text+8,&c,4);
    if (!strcmp(text,"GenuineIntel")) return 1;
    if (!strcmp(text,"AuthenticAMD")) return 2;
    return 0;
}

int main(void) {
    cpu_set_t allowed,one;
    if (sched_getaffinity(0,sizeof(allowed),&allowed)) return 2;
    unsigned common=7;int count=0,first_vendor=-1,mixed=0;
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&allowed)) {
        CPU_ZERO(&one);CPU_SET(cpu,&one);
        if (sched_setaffinity(0,sizeof(one),&one)) return 3;
        int current=vendor();
        if (first_vendor<0) first_vendor=current;
        if (current!=first_vendor) mixed=1;
        common&=usable();++count;
    }
    if (!count) return 4;
    const char *name=mixed?"mixed":first_vendor==1?"intel":first_vendor==2?"amd":"unknown";
    printf("{\"vendor\":\"%s\",\"avx2\":%s,\"avx512\":%s,\"avx512_vnni\":%s,\"cpus\":[",
        name,(common&1)?"true":"false",(common&2)?"true":"false",(common&4)?"true":"false");
    int separator=0;
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&allowed)) {
        printf("%s%d",separator?",":"",cpu);separator=1;
    }
    puts("]}");return 0;
}
"""
_NATIVE_DIR = Path(__file__).resolve().parent / "_native"
_PROBE_FLAGS = ["-O2", "-std=c11", "-march=x86-64", "-mtune=generic"]


def _read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def _unescape_mount(value):
    for source, target in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(source, target)
    return value


def _cgroup_locations(cgroups, mounts):
    """Resolve this process's CPU cgroup inside its mounted namespace."""
    memberships = []
    for line in (cgroups or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            memberships.append((set(fields[1].split(",")), PurePosixPath(fields[2])))
    result = []
    for line in (mounts or "").splitlines():
        sections = line.split(" - ", 1)
        if len(sections) != 2:
            continue
        left, right = sections[0].split(), sections[1].split()
        if len(left) < 5 or len(right) < 3 or right[0] not in ("cgroup", "cgroup2"):
            continue
        version = 2 if right[0] == "cgroup2" else 1
        if version == 1 and "cpu" not in set(right[2].split(",")):
            continue
        root, mount = PurePosixPath(_unescape_mount(left[3])), Path(_unescape_mount(left[4]))
        for controllers, member in memberships:
            if (version == 2 and controllers != {""}) or (version == 1 and "cpu" not in controllers):
                continue
            if not member.is_absolute() or ".." in member.parts or ".." in root.parts:
                continue
            try:
                relative = member.relative_to(root)
            except ValueError:
                continue
            result.append((version, mount / str(relative), mount))
    return result


def _cgroup_quota():
    """The tightest quota from the leaf and all visible CPU-cgroup parents."""
    locations = _cgroup_locations(_read("/proc/self/cgroup"), _read("/proc/self/mountinfo"))
    # Root paths also cover environments that hide /proc membership metadata.
    if not locations:
        locations = [(2, Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup")),
                     (1, Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu")),
                     (1, Path("/sys/fs/cgroup/cpu,cpuacct"), Path("/sys/fs/cgroup/cpu,cpuacct"))]
    quotas, seen = [], set()
    for version, leaf, mount in locations:
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(mount):
                break
            if (version, directory) in seen:
                continue
            seen.add((version, directory))
            try:
                if version == 2:
                    value = _read(directory / "cpu.max")
                    if value is None:
                        continue
                    quota, period = value.strip().split()
                    if quota == "max":
                        continue
                    quota, period = int(quota), int(period)
                else:
                    quota = int(_read(directory / "cpu.cfs_quota_us") or "-1")
                    period = int(_read(directory / "cpu.cfs_period_us") or "0")
                if quota > 0 and period > 0:
                    quotas.append(quota / period)
            except (ValueError, OverflowError):
                continue
    return min(quotas) if quotas else None


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_probe(destination):
    """Maintainer-only build of the baseline-ISA probe for a Linux x86 wheel.

Write ``cpu_probe`` into a fresh destination and return its manifest record.
No CPU instruction probe or inference is executed by this build function.
Published wheels carry this binary, so end users do not need a compiler.
"""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("Build the Linux CPU probe on a Linux x86_64 packaging host")
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / "cpu_probe"
    if executable.exists():
        raise FileExistsError(executable)
    compiler = next((value for name in ("cc", "gcc", "clang") if (value := shutil.which(name))), None)
    if compiler is None:
        raise RuntimeError("A C compiler is required on the wheel packaging host")
    with tempfile.TemporaryDirectory(prefix=".probe-build-", dir=directory) as staging:
        source, binary = Path(staging) / "probe.c", Path(staging) / "cpu_probe"
        source.write_text(_X86_PROBE)
        process = subprocess.run([compiler, *_PROBE_FLAGS, str(source), "-o", str(binary)],
                                 capture_output=True, text=True, timeout=15, check=False)
        if process.returncode or not binary.is_file():
            raise RuntimeError("The baseline-ISA CPU probe could not be compiled")
        binary.replace(executable)
    return {"path": "cpu_probe", "sha256": _digest(executable),
            "source_sha256": hashlib.sha256(_X86_PROBE.encode()).hexdigest(),
            "compiler": Path(compiler).name, "flags": list(_PROBE_FLAGS)}


def _packaged_probe():
    """Return a verified packaged record, absent, or invalid without executing it."""
    manifest = _NATIVE_DIR / "manifest.json"
    if manifest.is_symlink():
        return None, "invalid"
    if not manifest.exists():
        return None, "absent"
    try:
        raw = manifest.read_bytes()
        document = json.loads(raw)
        if (not isinstance(document, dict) or type(document.get("version")) is not int
                or document["version"] != 1 or document.get("wheel_platform") != "linux_x86_64"
                or not isinstance(document.get("files"), dict)):
            return None, "invalid"
        record = document["probe"]
        relative, expected = record["path"], record["sha256"]
        if (not isinstance(relative, str) or not relative or "\\" in relative or ".." in Path(relative).parts
                or Path(relative).is_absolute()
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                or document["files"].get(relative) != expected):
            return None, "invalid"
        root = _NATIVE_DIR.resolve()
        if (root / relative).is_symlink():
            return None, "invalid"
        binary = (root / relative).resolve()
        if not binary.is_relative_to(root) or not binary.is_file() or _digest(binary) != expected:
            return None, "invalid"
        return {"path": str(binary), "sha256": expected,
                "manifest_sha256": hashlib.sha256(raw).hexdigest()}, "verified"
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None, "invalid"


def _x86_probe(affinity, packaged=None):
    failure = {"status": "unavailable", "method": "child_cpuid_xgetbv", "reason": "No C compiler available for the CPU/OS capability probe"}
    try:
        if packaged is not None:
            if _digest(packaged["path"]) != packaged["sha256"]:
                return {}, dict(failure, reason="Packaged CPU probe changed after verification")
            process = subprocess.run([packaged["path"]], capture_output=True, text=True, timeout=15, check=False)
        else:
            # Source-tree development fallback only. Supported wheels carry a
            # verified binary and never enter this compiler path.
            with tempfile.TemporaryDirectory(prefix="fast-audiovae-cpu-") as directory:
                build_probe(directory)
                process = subprocess.run([str(Path(directory) / "cpu_probe")], capture_output=True,
                                         text=True, timeout=15, check=False)
        if process.returncode:
            return {}, dict(failure, reason="The isolated CPU/OS probe did not complete successfully")
        result = json.loads(process.stdout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, UnicodeError, json.JSONDecodeError):
        return {}, dict(failure, reason="The isolated CPU/OS probe is unavailable")
    if (not isinstance(result, dict) or result.get("vendor") not in ("intel", "amd", "mixed", "unknown")
            or any(type(result.get(key)) is not bool for key in ("avx2", "avx512", "avx512_vnni"))
            or not isinstance(result.get("cpus"), list) or not result["cpus"]
            or any(type(cpu) is not int or cpu < 0 for cpu in result["cpus"])
            or len(set(result["cpus"])) != len(result["cpus"])
            or (affinity is not None and sorted(result["cpus"]) != affinity)):
        return {}, dict(failure, reason="CPU probe output or affinity coverage could not be verified")
    if result["avx512_vnni"] and not result["avx512"] or result["avx512"] and not result["avx2"]:
        return {}, dict(failure, reason="CPU probe returned inconsistent capability evidence")
    return result, {"status": "ok", "method": "packaged_child_cpuid_xgetbv" if packaged else "child_cpuid_xgetbv",
                    "probe_sha256": packaged["sha256"] if packaged else None, "cpus_checked": result["cpus"],
                    "reason": "CPU features and OS vector-state support verified on every allowed CPU"}


@lru_cache(maxsize=8)
def _cached_x86_probe(affinity, probe_path=None, probe_digest=None, manifest_digest=None):
    # ISA evidence is reusable while the set of CPUs on which this process can
    # execute is unchanged. Quotas and requested worker counts are not cached.
    packaged = {"path": probe_path, "sha256": probe_digest, "manifest_sha256": manifest_digest} if probe_path else None
    return _x86_probe(list(affinity), packaged)


def detect_cpu(*, allow_compile=False):
    """Return actual process architecture, usable features and CPU budget.

No neural models are loaded and no network access is used. On Linux x86 the
    probe uses the wheel's hash-verified helper in its own process. Source-tree
    maintainers may explicitly allow a temporary compile with allow_compile=True.
    Ordinary users never enter that path. Failure to prove support leaves all
    advanced features false.
"""
    system = platform.system()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    logical = max(1, os.cpu_count() or 1)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    if affinity == []:
        affinity = None
    quota = _cgroup_quota() if system == "Linux" else None
    budget = min(float(len(affinity) if affinity else logical), quota if quota is not None else math.inf)
    result = {"system": system, "machine": machine, "platform": system + "/" + machine,
        "vendor": "unknown", "model": "", "logical_cpus": logical, "affinity": affinity,
        "cgroup_quota_cpus": quota, "cpu_budget": budget, "max_threads": max(1, math.floor(budget)),
        "usable": {"neon": False, "avx2": False, "avx512": False, "avx512_vnni": False},
        "probe": {"status": "unavailable", "method": "none", "reason": "No validated native recipe for this process architecture"}}
    if system == "Darwin" and machine == "arm64":
        result["vendor"] = "apple"
        result["usable"]["neon"] = True
        result["probe"] = {"status": "ok", "method": "native_arm64_abi",
                           "reason": "Native macOS arm64 ABI guarantees Advanced SIMD support"}
        try:
            query = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=2, check=False)
            if query.returncode == 0:
                result["model"] = query.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif system == "Linux" and machine == "x86_64":
        packaged, status = _packaged_probe()
        if status == "invalid":
            evidence = {}
            result["probe"] = {"status": "unavailable", "method": "packaged_child_cpuid_xgetbv",
                               "reason": "Packaged CPU probe manifest or binary failed verification"}
        elif packaged is None and not allow_compile:
            evidence = {}
            result["probe"] = {"status": "unavailable", "method": "none",
                               "reason": "No packaged CPU capability probe; portable ONNX selected without compiling"}
        elif affinity is not None:
            identity = (packaged["path"], packaged["sha256"], packaged["manifest_sha256"]) if packaged else (None, None, None)
            evidence, result["probe"] = deepcopy(_cached_x86_probe(tuple(affinity), *identity))
        else:
            evidence, result["probe"] = _x86_probe(None, packaged)
        if evidence:
            result["vendor"] = evidence["vendor"]
            result["usable"].update({key: evidence[key] for key in ("avx2", "avx512", "avx512_vnni")})
        text = _read("/proc/cpuinfo") or ""
        models = {line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name") and ":" in line}
        result["model"] = next(iter(models)) if len(models) == 1 else "mixed" if models else ""
    return result


def select_recipe(cpu, mode="streaming", threads=1):
    """Choose a supported recipe and cap an explicit worker request.

The default always remains one worker. A larger requested value is bounded
by process affinity and any visible cgroup quota. Fractional quotas round
down, with one worker as the minimum. Selection establishes CPU eligibility;
the bundle loader separately checks that compatible native artifacts exist.
"""
    if mode not in ("batch", "streaming"):
        raise ValueError("mode must be 'batch' or 'streaming'")
    if threads is None:
        threads = 1
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    maximum = cpu.get("max_threads", 1)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("CPU evidence contains an invalid thread budget")
    available_budget = min(threads, maximum)
    recipe, reason = "portable", "CPU/OS support for a validated native recipe was not proven"
    system, machine, vendor = cpu.get("system"), cpu.get("machine"), cpu.get("vendor")
    usable, proven = cpu.get("usable", {}), cpu.get("probe", {}).get("status") == "ok"
    if proven and system == "Darwin" and machine == "arm64" and vendor == "apple" and usable.get("neon") is True:
        recipe = "apple_stream_projection" if mode == "streaming" else "apple_native"
        reason = "Native Apple arm64 recipe selected for " + mode
    elif (proven and system == "Linux" and machine == "x86_64"
          and all(usable.get(key) is True for key in ("avx2", "avx512", "avx512_vnni"))):
        if vendor == "intel":
            recipe = "intel_stream_projection" if mode == "streaming" else "intel_precision"
            reason = "Usable AVX512-VNNI verified; Intel recipe selected for " + mode
        elif vendor == "amd":
            recipe = "amd_precision"
            reason = "Usable AVX512-VNNI verified; retained AMD recipe selected for " + mode
    validated = {"apple_native": [1, 4], "apple_stream_projection": [1],
                 "intel_precision": [1, 2], "intel_stream_projection": [1],
                 "amd_precision": [1, 4]}.get(recipe)
    selected_threads = max(value for value in validated if value <= available_budget) if validated else available_budget
    if selected_threads != threads:
        reason += "; worker request reduced to fit the available CPU budget and validated recipe schedules"
    return {"recipe": recipe, "mode": mode, "threads": selected_threads, "requested_threads": threads,
        "thread_budget": maximum, "threads_capped": selected_threads != threads,
        "validated_threads": validated,
        "thread_policy": "one_worker_default_or_explicit_budget_capped_to_available_cpu_and_validated_schedule",
        "cpu_platform": cpu.get("platform", str(system) + "/" + str(machine)),
        "cpu_vendor": vendor, "fallback": recipe == "portable", "reason": reason}
