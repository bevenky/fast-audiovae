"""Assemble the audited sources and launch a bounded fresh reconstruction stage."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from audiovae_student.cache import TrainingCrop, DECODER_HOP
from audiovae_student.data import load_manifest, validate_manifest
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.representative_pilot import load_restart_plan, run_representative_pilot, RepresentativePilotConfig
from audiovae_student.restart_data import canonical, file_sha, normalize_language
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import FrozenAudioVAE2, CHECKPOINT_SHA256

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r5')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PLAN = Path('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1')
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')


def load_dev_panel(directory):
    """Read the exact metadata-selected panel without inventing aliases or crops."""
    directory = Path(directory)
    audit = json.loads((directory / 'ready.json').read_text())
    body = {key: value for key, value in audit.items() if key != 'identity_sha256'}
    # The panel builder's identity JSON has no trailing newline.
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    required = {'manifest.jsonl', 'input-sample-counts.json', 'groups.json',
                'windows.jsonl', 'source-checks.json'}
    if (audit.get('state') != 'ready' or audit.get('identity_sha256') != expected
            or not required <= set(audit.get('files_sha256', {}))):
        raise ValueError('Development panel is not checksum-complete and ready')
    for name, expected in audit['files_sha256'].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or file_sha(path) != expected:
            raise ValueError('Development panel metadata changed')
    rows = load_manifest(directory / 'manifest.jsonl')
    by_id = {row.source_id: row for row in rows}
    counts = json.loads((directory / 'input-sample-counts.json').read_text())
    groups = json.loads((directory / 'groups.json').read_text())
    if (len(by_id) != len(rows) or len(rows) != audit['rows'] or set(counts) != set(by_id)
            or any(row.split != 'dev' for row in rows)):
        raise ValueError('Development identities, splits or counts differ from panel')
    memberships = defaultdict(list)
    for group, source_ids in groups.items():
        if not isinstance(group, str) or not group or not isinstance(source_ids, list):
            raise ValueError('Development coverage group is invalid')
        for source_id in source_ids:
            memberships[source_id].append(group)
    if set(memberships) != set(by_id) or any(len(value) != 1 for value in memberships.values()):
        raise ValueError('Every development source must have exactly its published coverage group')
    metadata = {}
    for source_id, row in by_id.items():
        count = counts[source_id]
        if (type(count) is not int or count < 1
                or abs(row.duration_seconds - count / 16000) > 1 / 16000 + 1e-10):
            raise ValueError('Development exact sample count differs from source duration')
        group = memberships[source_id][0]
        condition = (group.split(':', 1)[1] if group.startswith('event:')
                     else 'speech' if group.startswith('speech:') else group)
        metadata[source_id] = {'dataset': row.dataset, 'language': row.language,
                              'condition': condition, 'coverage_group': group}
    windows = [json.loads(line) for line in (directory / 'windows.jsonl').read_text().splitlines()]
    seen = set()
    for spec in windows:
        source_id = spec['source_id']
        row = by_id[source_id]
        start, frames, valid = (spec[key] for key in
                               ('start_frame', 'scored_frames', 'valid_input_samples16k'))
        if (any(type(value) is not int for value in (start, frames, valid)) or start < 0 or frames < 1
                or spec['row_id'] != source_id or spec['dataset'] != row.dataset
                or spec['language'] != normalize_language(row.language)
                or valid != min(frames * 640, counts[source_id] - start * 640)
                or spec['valid_output_samples48k'] != valid * 3 or valid * 3 < 4096
                or (source_id, start) in seen):
            raise ValueError('Development window differs from its real source or repeats a crop')
        seen.add((source_id, start))
    if len(windows) != audit['windows'] or {source for source, _ in seen} != set(by_id):
        raise ValueError('Development window coverage is incomplete')
    return rows, counts, metadata, windows, audit


def panel_crop(record, spec, context_frames=29):
    """Slice one published development window from continuous frozen targets."""
    start, scored_frames = spec['start_frame'], spec['scored_frames']
    source_id = record.metadata['identity']['source']['source_id']
    valid = min(scored_frames * DECODER_HOP, record.valid_output_samples - start * DECODER_HOP)
    if source_id != spec['source_id'] or valid != spec['valid_output_samples48k'] or valid < 4096:
        raise ValueError('Development target does not match the published valid sample interval')
    context_start = max(0, start - context_frames)
    context = start - context_start
    frames = context + scored_frames
    def window(value, begin, size):
        part = value[..., begin:begin + size]
        return F.pad(part, (0, size - part.shape[-1])).contiguous().clone()
    return TrainingCrop(window(record.latents, context_start, frames),
        window(record.teacher_audio, context_start * DECODER_HOP, frames * DECODER_HOP),
        None, record.cache_key, source_id, start, context_start, context, scored_frames, valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int, default=200)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    plan = load_restart_plan(PLAN)
    panel_rows, panel_counts, panel_metadata, panel_windows, panel_audit = load_dev_panel(PANEL)
    if panel_audit['old_ledger_identity'] != plan['ledger'].identity_sha256:
        raise ValueError('Development panel was not checked against this consumed-audio ledger')
    validate_manifest(plan['rows'], reserved_rows=panel_rows, training_only=True)
    for row in panel_rows:
        if not plan['ledger'].whole_untouched(row):
            raise ValueError('Development panel overlaps previously trained source')
    print(json.dumps({'phase': 'data_verified', 'pilot_windows': len(plan['windows']),
                      'pilot_rows': len(plan['rows']), 'dev_rows': len(panel_rows)}), flush=True)
    rows = tuple(plan['rows']) + tuple(panel_rows)
    counts = {r.source_id: plan['counts'][r.source_id] for r in plan['rows']}
    counts.update(panel_counts)
    teacher = FrozenAudioVAE2.from_files(OLD / 'assets/audio_vae_v2.py', OLD / 'assets/audiovae.pth', device='cuda')
    first = plan['rows'][0]
    teacher = _target_teacher(teacher, first)
    teacher_start_sha = state_fingerprint(teacher.model.state_dict())
    payload = Path(first.audio_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != first.audio_sha256:
        raise ValueError('Teacher warmup source changed')
    warmup = read_native_16k(payload, first).to(teacher.device)
    for _ in range(2):
        teacher.decode(teacher.encode(warmup))
    del warmup
    corpus = SourceCorpus(rows, teacher, cache_dir=BASE / 'teacher-cache-pilot-v1',
        input_sample_counts=counts, max_disk_bytes=3 * 1024**3, min_free_bytes=4 * 1024**3,
        max_memory_utterances=64, allow_prepared_source=True)
    try:
        print(json.dumps({'phase': 'qualifying_and_caching_teacher', 'dev_rows': len(panel_rows)}), flush=True)
        prefill = corpus.prefetch([r.source_id for r in panel_rows], max_batch_size=8,
                                 max_total_input_samples=1_920_000)
        # Serial fallback is explicit and retains the pinned original targets.
        preparation = BASE / 'initial-teacher-prefill.json'
        if not preparation.exists():
            preparation.write_bytes(canonical({'source_corpus_identity_sha256': corpus.identity_sha256,
                                              'report': prefill}))
        elif json.loads(preparation.read_text())['source_corpus_identity_sha256'] != corpus.identity_sha256:
            raise ValueError('Teacher prefill identity changed')
        heldout = tuple(panel_crop(corpus.get(spec['source_id']), spec) for spec in panel_windows)
        if teacher_start_sha != state_fingerprint(teacher.model.state_dict()):
            raise ValueError('Teacher parameters changed while preparing targets')
        identity = {'teacher_checkpoint_sha256': CHECKPOINT_SHA256,
            'teacher_batch_qualification': json.loads(preparation.read_text()),
            'source_corpus': corpus.identity, 'restart_plan': plan['identity'],
            'dev_panel': panel_audit, 'dev_panel_sha256': file_sha(PANEL / 'ready.json'),
            'quiet_calibration': {'selected_quiet_share': 0.0,
                'reason': '10 percent candidate failed its declared training-side waveform regression limit'},
            'source_archive_sha256': file_sha(BASE / 'source.tgz'),
            'launcher_sha256': file_sha(Path(__file__)), 'teacher_state_sha256': teacher_start_sha}
        config = RepresentativePilotConfig(total_steps=(len(plan['windows']) + 31) // 32,
            quiet_gradient_share=0.0, evaluation_interval=200, checkpoint_interval=100)
        output = BASE / 'training-runs/representative-reconstruction-v1'
        print(json.dumps({'phase': 'starting_student', 'config': config.__dict__,
                          'max_updates_this_launch': args.max_updates, 'fresh_start': not args.resume}), flush=True)
        result = run_representative_pilot(corpus, plan['windows'], plan['rows'], counts, plan['ledger'],
            heldout, panel_rows, output, data_identity=identity, config=config, device='cuda',
            resume_from=output / 'latest.pt' if args.resume else None, max_updates=args.max_updates,
            heldout_metadata=panel_metadata, reserved_rows=plan['reserved'] + plan['sentinel'],
            excluded_sources=plan['diagnostic'], log_dir=OLD / 'runs',
            run_name='representative-reconstruction-v1')
        result['teacher_state_unchanged'] = teacher_start_sha == state_fingerprint(teacher.model.state_dict())
        if not result['teacher_state_unchanged']:
            raise ValueError('Frozen teacher state changed during student training')
        (BASE / 'stage-result.json').write_bytes(canonical(result))
        print(json.dumps(result), flush=True)
    finally:
        corpus.close()


if __name__ == '__main__':
    main()
