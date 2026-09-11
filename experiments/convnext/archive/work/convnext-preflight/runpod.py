"""Task-local SSH transport; credentials stay in the ignored repository .env."""
from pathlib import Path
import shlex
import subprocess
import sys

base = Path(__file__).resolve().parent
values = {}
for line in (base.parent / "fast-audiovae/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, raw = line.split("=", 1)
        parsed = shlex.split(raw)
        if len(parsed) == 1:
            values[key.strip()] = parsed[0]
destination = values["AMD_DIRECT_SSH_USER"] + "@" + values["AMD_DIRECT_SSH_HOST"]
options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "UserKnownHostsFile=" + str(base / "known_hosts"),
           "-i", str(Path(values["SSH_IDENTITY_FILE"]).expanduser())]
if sys.argv[1] == "upload":
    command = ["scp", *options, "-P", values["AMD_DIRECT_SSH_PORT"],
               sys.argv[2], destination + ":" + sys.argv[3]]
elif sys.argv[1] == "download":
    command = ["scp", *options, "-P", values["AMD_DIRECT_SSH_PORT"],
               destination + ":" + sys.argv[2], sys.argv[3]]
else:
    command = ["ssh", *options, "-p", values["AMD_DIRECT_SSH_PORT"],
               destination, sys.argv[1]]
raise SystemExit(subprocess.call(command))
