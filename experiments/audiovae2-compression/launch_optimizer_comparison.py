"""Launch one approved finite comparison phase in a detached process."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('qualification', 'recovery'), required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['root'])
    receipt = root / (args.phase + '-dispatch.json')
    log_path = root / (args.phase + '-controller.log')
    if any(p.exists() for p in (receipt, log_path, root / (args.phase + '-controller-launch.json'))):
        raise FileExistsError('This phase has already been dispatched; preserve its results')
    os.umask(0o077)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    argv = [plan['python'], str(Path(__file__).parent / 'optimizer_comparison_controller.py'),
            '--plan', str(args.plan.resolve()), '--phase', args.phase]
    with log_path.open('xb') as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
    fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(') ', 1)[1].split()
    value = {'phase': args.phase, 'pid': process.pid, 'start_ticks': fields[19],
             'dispatched_at': datetime.now(timezone.utc).isoformat(),
             'argv': argv, 'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    with receipt.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
