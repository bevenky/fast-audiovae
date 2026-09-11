"""Start the reviewed bounded whistle acquisition on the existing pod."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

base = Path("/workspace/fast-audiovae-convnext-20260908-r1")
launch = base / "corpus-human-whistling.launch.json"
root = base / "data-human-whistling"
root.mkdir(exist_ok=True)
lock = (root / "acquisition.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if launch.exists():
    raise SystemExit("Existing whistle launch record: inspect before starting another process")
command = [str(base / ".train-venv/bin/python"), "-u", "-m", "audiovae_student.acquire_human_whistle",
    "--selection", str(root / "research/selection.json"),
    "--selection-sha256", "0cda4731405495190656a4f20fa76d5180be15ac342f4d542820c7380f8da36a",
    "--output-dir", str(root), "--exclude-training-manifest", str(base / "data-bootstrap/train.jsonl"),
    "--reserved-manifest", str(base / "data-bootstrap/dev.jsonl"),
    "--reserved-evaluation-manifest", str(base / "reserved-evaluation.json")]
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
with (root / "run.log").open("ab", buffering=0) as log:
    process = subprocess.Popen(command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
time.sleep(.2)
actual = Path(f"/proc/{process.pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
if actual != command:
    raise RuntimeError("Detached whistle process argv did not match")
stat = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()
boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
started = boot + int(stat[19]) / os.sysconf("SC_CLK_TCK")
value = {"pid": process.pid, "command": actual, "started_at": started, "recorded_at": time.time(),
         "record_source": "actual /proc argv and process start ticks", "cwd": str(base),
         "log": str(root / "run.log"), "readiness": str(root / "prepared/provenance/complete.json"),
         "preparation_version": 1, "cpu_only": True}
temporary = launch.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n")
os.replace(temporary, launch)
print(json.dumps(value, indent=2))
