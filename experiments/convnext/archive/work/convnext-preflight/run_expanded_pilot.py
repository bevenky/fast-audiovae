"""Complete the authorized 500-hour acquisition, then continue the H100 student.

Runs only on the existing pod. It cannot provision hardware, accept dataset
terms, delete source audio, or start a second training process. Acquisition
failures stop the handoff for review instead of lowering the corpus gates.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_matches(record):
    """Check both PID and recorded argv; never confuse a recycled PID with work."""
    try:
        actual = Path('/proc', str(record['pid']), 'cmdline').read_bytes().split(b'\0')
        command = [x.decode() for x in actual if x]
        return command == record['command']
    except (FileNotFoundError, ProcessLookupError):
        return False


def source_status(base, name, source_spec=None):
    specifications = {
        'core': ('data-expanded-core/provenance/sources.json', 'state', 'ready', 'corpus-core-440h.launch.json'),
        'indic': ('data-expanded-indic/progress.json', 'status', 'complete', 'corpus-indic-44h.launch.json'),
        'expressive': ('data-expressive-originals/prepared/provenance/expressive-complete.json',
                       'state', 'complete', 'expressive-preparation.launch.json'),
    }
    if source_spec is not None and 'launch_record' in source_spec:
        path, launch = source_spec['readiness'], source_spec['launch_record']
        field = 'status' if source_spec['kind'] == 'indic' else 'state'
        ready = 'ready' if source_spec['kind'] == 'acquire' else 'complete'
    else:
        path, field, ready, launch = specifications[name]
    record = read_json(base / path) if (base / path).exists() else {}
    value = record.get(field, 'pending')
    if name == 'core':
        hours = record.get('fresh_training_hours', 0)
    elif name == 'indic':
        hours = record.get('hours', 0)
    else:
        hours = record.get('hours_by', {}).get('split', {}).get('train', 0)
    if value == ready:
        if name == 'expressive' or (source_spec and source_spec['kind'] == 'expressive'):
            expected = (source_spec or {}).get('preparation_version', 2)
            if record.get('preparation_version') != expected:
                raise RuntimeError(f'{name} completion marker has an unexpected preparation version')
        return dict(state='ready', hours=hours)
    if 'fail' in value or 'insufficient' in value:
        raise RuntimeError(f'{name} acquisition reports {value}')
    if not (base / launch).exists() or not process_matches(read_json(base / launch)):
        raise RuntimeError(f'{name} acquisition is not ready and its recorded process is not running')
    return dict(state=value, hours=hours)


def run_child(command, base, record_path, log_path, report, *, lock_fd):
    """Record launches before waiting and refuse to duplicate an existing child."""
    if record_path.exists() and process_matches(read_json(record_path)):
        raise RuntimeError(f'Existing child remains active: {record_path.name}; do not start a duplicate')
    environment = dict(os.environ, PYTHONPATH=str(base), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        write_json(record_path, dict(pid=process.pid, command=command, started_at=time.time(), log=str(log_path)))
        while process.poll() is None:
            report()
            time.sleep(30)
        result = process.returncode
    record = read_json(record_path)
    record.update(returncode=result, finished_at=time.time())
    write_json(record_path, record)
    if result:
        raise RuntimeError(f'{record_path.name} exited with status {result}; inspect its saved log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--assembly-config', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    directory = base / 'expanded-pilot'
    directory.mkdir(exist_ok=True)
    lock = (directory / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from torch.utils.tensorboard import SummaryWriter
    from audiovae_student.corpus_assembly import assemble_corpus

    writer = SummaryWriter(str(base / 'runs/corpus-preparation-500h'), flush_secs=10)
    completed_config = directory / 'assembly-with-fill.json'
    config_path = completed_config if completed_config.exists() else args.assembly_config.resolve()
    configuration = read_json(config_path)
    initial_names = configuration.get('initial_source_names', ['core', 'indic', 'expressive'])
    specifications = {item['name']: item for item in configuration['sources']}
    assembled = directory / 'corpus'
    python = str(base / '.train-venv/bin/python')
    training_dir = base / 'training-runs/warmup-speech-500h-v1'
    state = dict(phase='waiting_for_sources', started_at=time.time(), pid=os.getpid(),
                 target_training_hours=500, target_step=10000, batch_size=64)
    tick = int(time.time())

    def publish(**changes):
        nonlocal tick
        state.update(changes, updated_at=time.time())
        write_json(directory / 'status.json', state)
        tick = max(tick + 1, int(time.time()))
        for name, value in state.get('sources', {}).items():
            writer.add_scalar('corpus/' + name + '_verified_hours', value['hours'], tick)
        if 'verified_train_hours' in state:
            writer.add_scalar('corpus/verified_train_hours', state['verified_train_hours'], tick)
        if state.get('phase') == 'training' and (training_dir / 'status.json').exists():
            current = read_json(training_dir / 'status.json')
            state['training_step'] = current['step']
            state['training_state'] = current['state']
            state['training_unique_hours'] = current['unique_scored_hours_seen']
            write_json(directory / 'status.json', state)

    try:
        while True:
            sources = {name: source_status(base, name, specifications[name]) for name in initial_names}
            publish(sources=sources)
            if all(source['state'] == 'ready' for source in sources.values()):
                break
            time.sleep(60)

        publish(phase='verifying_corpus')
        report = assemble_corpus(config_path, assembled)
        publish(verified_train_hours=report['verified_train_hours'], gate_state=report['state'])
        if report['state'] == 'insufficient_hours' and report.get('filler_permitted'):
            # Whole fresh utterances cover the remaining duration. Add two
            # minutes for numerical/quota boundaries; this is real audio.
            minutes = math.ceil(report['remaining_train_hours'] * 60) + 2
            if minutes > 1200:
                raise RuntimeError('Unexpected corpus shortfall exceeds the 20-hour fill bound')
            fill_root = base / 'data-expanded-fill'
            command = [python, '-m', 'audiovae_student.acquire',
                       '--output-root', str(fill_root),
                       '--audio-root', '/tmp/fast-audiovae-corpus-500h/fill',
                       '--minutes-per-language', '0', '--librispeech-train-minutes', '0',
                       '--librispeech-other-minutes', str(minutes), '--librispeech-dev-minutes', '0',
                       '--librispeech-speaker-minutes', '30',
                       '--exclude-training-manifest', str(assembled / 'filler-exclusion.jsonl'),
                       '--reserved-manifest', str(assembled / 'candidate-dev.jsonl'),
                       '--reserved-evaluation-manifest', str(base / 'reserved-evaluation.json'),
                       '--minimum-fresh-hours', str(minutes / 60),
                       '--max-audio-gb', '3', '--min-free-audio-gb', '8', '--max-download-gb', '30']
            if (fill_root / 'provenance/acquisition-plan.json').exists():
                command.append('--resume')
            publish(phase='acquiring_shortfall', fill_minutes=minutes)
            run_child(command, base, directory / 'fill.launch.json', directory / 'fill.log', publish, lock_fd=lock.fileno())
            config = read_json(config_path)
            if not any(item['name'] == 'fill' for item in config['sources']):
                config['sources'].append(dict(name='fill', kind='acquire',
                    manifests=[str(fill_root / 'train.jsonl')],
                    readiness=str(fill_root / 'provenance/sources.json')))
            final_config = directory / 'assembly-with-fill.json'
            # The deployed initial config uses absolute paths so relocation
            # here cannot change their meaning.
            write_json(final_config, config)
            report = assemble_corpus(final_config, assembled)
            publish(verified_train_hours=report['verified_train_hours'], gate_state=report['state'])
        if report['state'] != 'ready':
            raise RuntimeError(f'Corpus readiness gate failed: {report["state"]}')

        manifest = assembled / 'source-manifest.jsonl'
        warmup = base / 'training-runs/warmup-speech-muon-v1/latest.pt'
        if digest(warmup) != 'ba999a89e8d87f9da91fd00fa1ddde04398a353a5a1ecc6e23aa36f9bd30605f':
            raise ValueError('Original step-1000 checkpoint differs from the verified backup')
        command = [python, '-m', 'audiovae_student.source_training',
                   '--manifest', str(manifest), '--corpus-audit', str(assembled / 'readiness.json'),
                   '--teacher-source', str(base / 'assets/audio_vae_v2.py'),
                   '--teacher-checkpoint', str(base / 'assets/audiovae.pth'),
                   '--cache-dir', str(base / 'target-cache-500h'), '--cache-gib', '12',
                   '--cache-memory-utterances', '256', '--output-dir', str(training_dir),
                   '--device', 'cuda', '--steps', '10000', '--batch-size', '64',
                   '--log-dir', str(base / 'runs'), '--run-name', 'warmup-speech-500h-v1']
        checkpoint = training_dir / 'latest.pt'
        command.extend(['--resume', str(checkpoint)] if checkpoint.exists() else ['--initialize-from', str(warmup)])
        publish(phase='training', manifest_sha256=digest(manifest))
        run_child(command, base, directory / 'training.launch.json', directory / 'training.log', publish, lock_fd=lock.fileno())
        result = read_json(training_dir / 'status.json')
        if result['state'] != 'completed' or result['step'] != 10000:
            raise RuntimeError('Training process exited without completing step 10000')
        publish(phase='completed_reconstruction', training_step=10000,
                next_action='Review reconstruction progress and run the deferred perceptual quality evaluation once')
    except BaseException as error:
        publish(phase='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        writer.close()
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
