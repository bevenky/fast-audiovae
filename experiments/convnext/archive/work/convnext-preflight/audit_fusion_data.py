"""Read-only provenance and audio-exposure audit of the finished paired screen.

No teacher/student inference, optimizer construction, benchmarks or training.
Only the requested audit JSON is written. Natural input quiet statistics are
computed from original prepared PCM, not substituted for teacher-output metrics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
sys.path.insert(0, str(BASE / 'remediation/fusion-code'))
from audiovae_student.continuation_data import load_continuation_plan, _IntervalIndex, _check_windows
from audiovae_student.comparison_data import EVENTS, INDIC22, known_identities
from audiovae_student.data import validate_manifest
from audiovae_student.restart_data import digest, file_sha
from audiovae_student.sampling import normalize_language


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def overlap_pairs(index):
    total = 0
    for spans in index.values():
        furthest = -math.inf
        for start, stop in sorted(spans):
            total += start < furthest
            furthest = max(furthest, stop)
    return total


def exposure(windows, rows):
    n = sum(w.valid_input_samples16k for w in windows)
    groups = {}
    for name in ('language', 'dataset', 'condition'):
        acc = defaultdict(lambda: {'windows': 0, 'input_samples16k': 0, 'source_ids': set()})
        for w in windows:
            key = getattr(w, name) or 'unspecified'
            acc[key]['windows'] += 1
            acc[key]['input_samples16k'] += w.valid_input_samples16k
            acc[key]['source_ids'].add(w.source_id)
        groups[name] = {k: {'windows': v['windows'], 'sources': len(v['source_ids']),
            'seconds': v['input_samples16k'] / 16000,
            'duration_percent': 100 * v['input_samples16k'] / n}
            for k, v in sorted(acc.items())}
    explicit = sum(w.valid_input_samples16k for w in windows if w.condition in EVENTS)
    generic = sum(w.valid_input_samples16k for w in windows if w.condition == 'emotional_nonverbal')
    starts = [w for w in windows if w.start_frame == 0]
    languages = {w.language for w in windows} - {'und', 'none'}
    return {'windows': len(windows), 'sources': len({w.source_id for w in windows}),
        'scored_input_samples16k': n, 'scored_output_samples48k': n * 3,
        'seconds': n / 16000, 'hours': n / 16000 / 3600,
        'known_languages': len(languages), 'indic_languages_present': sorted(set(INDIC22) & languages),
        'indic_languages_missing': sorted(set(INDIC22) - languages),
        'genuine_source_start_windows': len(starts), 'source_start_window_percent': 100 * len(starts) / len(windows),
        'source_start_scored_duration_percent': 100 * sum(w.valid_input_samples16k for w in starts) / n,
        'less_than_30_context_frames': sum(w.start_frame < 30 for w in windows),
        'full_64_frame_windows': sum(w.valid_input_samples16k == 40960 for w in windows),
        'explicit_source_label_seconds': explicit / 16000,
        'explicit_source_label_duration_percent': 100 * explicit / n,
        'generic_emogator_seconds': generic / 16000,
        'generic_emogator_duration_percent': 100 * generic / n,
        'nominal_broad_expressive_duration_percent': 100 * (explicit + generic) / n,
        'synthetic_fixture_windows': sum(rows[w.source_id].dataset == 'synthetic_fixture' for w in windows),
        'dedicated_silence_label_windows': sum('silence' in (w.condition or '').lower() for w in windows),
        'by': groups}


def source_check(item):
    """Rehash source bytes, inspect shape and selected natural input quiet windows."""
    import numpy as np
    import soundfile as sf
    row, groups, expected_count = item
    raw = Path(row.audio_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row.audio_sha256, 'Source bytes changed: ' + row.source_id)
    audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    require(rate == 16000 and audio.shape == (expected_count, 1), 'Prepared source shape changed: ' + row.source_id)
    require(bool(np.isfinite(audio).all()), 'Nonfinite source PCM: ' + row.source_id)
    results = {}
    for label, windows in groups.items():
        q = Counter()
        for w in windows:
            start = w.start_frame * 640
            a = audio[start:start + w.valid_input_samples16k, 0]
            require(len(a) == w.valid_input_samples16k, 'Scored input beyond source')
            q['scored_samples'] += len(a)
            q['exact_zero_samples'] += int(np.count_nonzero(a == 0))
            count = len(a) // 320
            if count:
                frame = a[:count * 320].reshape(count, 320)
                rms = np.sqrt(np.mean(frame.astype('float64') ** 2, axis=1))
                q['complete_20ms_windows'] += count
                q['quiet_20ms_windows_rms_le_0_001'] += int(np.count_nonzero(rms <= .001))
                q['very_quiet_20ms_windows_rms_le_0_00001'] += int(np.count_nonzero(rms <= .00001))
                q['exact_zero_20ms_windows'] += int(np.count_nonzero(np.all(frame == 0, axis=1)))
            q['tail_samples_excluded_from_window_statistics'] += len(a) % 320
        results[label] = dict(q)
    return row.source_id, {'sha256': row.audio_sha256, 'bytes': len(raw), 'input_samples16k': len(audio)}, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=BASE / 'remediation/fusion-screen')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    identity = read(root / 'experiment-identity.json')
    receipt = read(root / 'parent-receipt.json')
    summary = read(root / 'summary-compact.json')
    plan_path = Path(receipt['plan_path'])
    plan = load_continuation_plan(plan_path)
    require(plan['identity']['identity_sha256'] == receipt['plan_identity_sha256'], 'Wrong parent plan')
    require(identity['parent_step'] == receipt['step'] == summary['parent_step'] == 8090, 'Wrong parent step')
    require(identity['parent_cursor'] == receipt['cursor'] == 70080, 'Wrong parent sampler boundary')
    require(identity['parent_sha256'] == receipt['checkpoint_sha256'] == summary['parent_sha256'], 'Wrong parent checkpoint')
    require(digest(identity) == summary['experiment_identity_sha256'], 'Unmatched result summary')
    require(not summary['pending_arms'], 'Incomplete screen')
    n = identity['generator_steps'] * identity['batch_size']
    nd = identity['discriminator_warmup_steps'] * identity['batch_size']
    cursor = identity['parent_cursor']
    windows = plan['windows'][cursor:cursor + n + nd]
    require([w.to_dict() for w in windows] == identity['training_windows'], 'Experiment does not match sealed plan windows')
    require(n == 6400 and nd == 640 and len(windows) == n + nd, 'Unexpected arm budget')
    rows = {r.source_id: r for r in plan['rows']}
    selected = {w.source_id: rows[w.source_id] for w in windows}
    validate_manifest(selected.values(), training_only=True)
    forbidden = _IntervalIndex.from_ledger(plan['ledger'])
    # from_ledger seals to JSON-style lists, while add emits tuples. Normalize
    # the equivalent interval containers before extending the immutable index.
    for key, spans in forbidden.intervals.items():
        forbidden.intervals[key] = [tuple(pair) for pair in spans]
    for w in plan['windows'][:cursor]:
        forbidden.add(rows[w.source_id], w.start_frame * 640, w.start_frame * 640 + w.valid_input_samples16k)
    forbidden.seal()
    _check_windows(windows, rows, plan['counts'], forbidden=forbidden,
                   reserved=plan['reserved'], excluded=plan['excluded'])
    for w, crop in zip(windows, identity['target_crops'], strict=True):
        require(crop['source_id'] == w.source_id and crop['start_frame'] == w.start_frame
                and crop['scored_frames'] == w.scored_frames
                and crop['valid_scored_samples'] == w.valid_output_samples48k
                and crop['context_frames'] == min(w.start_frame, 30), 'Mismatched frozen training target crop')
        require(w.dataset == rows[w.source_id].dataset and w.language == normalize_language(rows[w.source_id].language),
                'Condition join changed dataset or language')
    intervals = {'source': defaultdict(list), 'audio_sha256': defaultdict(list), 'absolute_parent': defaultdict(list)}
    for w in windows:
        row = rows[w.source_id]
        lo, hi = w.start_frame * 1920, w.start_frame * 1920 + w.valid_output_samples48k
        intervals['source'][w.source_id].append((lo, hi))
        intervals['audio_sha256'][row.audio_sha256].append((lo, hi))
        origin = row.parent_start_seconds * 48000
        intervals['absolute_parent'][row.parent_recording_id].append((math.floor(origin + lo), math.ceil(origin + hi)))
    overlaps = {k: overlap_pairs(v) for k, v in intervals.items()}
    require(not any(overlaps.values()), 'Repeated scored audio within generator or D warmup')
    # Check every completed artifact against the already tensor-verified summary.
    arm_checks = {}
    base_original = summary['arms']['control']['original_student_state_sha256']
    reports = {}
    for arm in identity['arms']:
        for name in ('before.json', 'after.json', 'migration.json', 'complete.json'):
            require(file_sha(root / arm / name) == summary['source_hashes'][arm][name], 'Saved arm metadata changed')
        before, after, migration = (read(root / arm / name) for name in ('before.json', 'after.json', 'migration.json'))
        require(before['evaluated_step'] == 8090 and after['evaluated_step'] == 8290, 'Different training start or update count')
        require(migration['original_student_state_sha256'] == base_original
                and migration['original_student_state_preserved'] and migration['original_student_optimizers_preserved'],
                'Original student or optimizer was not preserved')
        require(summary['arms'][arm]['generator_updates'] == 200, 'Unequal generator updates')
        expected_warmup = 20 if arm in ('fresh_magnitude', 'complex') else 0
        require(summary['arms'][arm]['discriminator_only_updates'] == expected_warmup, 'Wrong discriminator warmup')
        arm_checks[arm] = {'start_step': 8090, 'end_step': 8290, 'generator_windows': n,
            'discriminator_only_windows': nd if expected_warmup else 0,
            'original_student_parameter_state_sha256': base_original,
            'initial_parameter_state_sha256': summary['arms'][arm]['initial_parameter_state_sha256'],
            'original_weights_preserved': True, 'original_optimizers_preserved': True,
            'new_filter': 'seven causal taps initialized [0,0,0,0,0,0,1]' if arm == 'filter' else None}
        reports[arm] = before
    natural = [r for r in reports['control']['rows'] if r['metadata']['dataset'] != 'synthetic_fixture']
    synthetic = [r for r in reports['control']['rows'] if r['metadata']['dataset'] == 'synthetic_fixture']
    reserved = {r.source_id: r for r in plan['reserved']}
    heldout = {r['source_id']: reserved[r['source_id']] for r in natural}
    for arm, report in reports.items():
        projected = lambda r: (r['source_id'], r['start_frame'], r['absolute_scored_start_sample'], r['samples'], r['metadata'])
        require([projected(r) for r in report['rows']] == [projected(r) for r in reports['control']['rows']], 'Arm panels differ')
    disjoint = {}
    for kind, getter in [('source_id', lambda r: r.source_id), ('audio_sha256', lambda r: r.audio_sha256),
                         ('parent_recording_id', lambda r: r.parent_recording_id)]:
        count = len({getter(r) for r in selected.values()} & {getter(r) for r in heldout.values()})
        require(count == 0, 'Training/heldout leakage by ' + kind)
        disjoint[kind] = count
    known_train = set().union(*(known_identities(r) for r in selected.values()))
    known_dev = set().union(*(known_identities(r) for r in heldout.values()))
    require(not known_train & known_dev, 'Known heldout identity used in training')
    counts = dict(plan['counts'])
    for sid, row in heldout.items():
        counts[sid] = round(row.duration_seconds * 16000)
    all_rows = {**selected, **heldout}
    grouped = defaultdict(lambda: defaultdict(list))
    for label, ws in [('generator', windows[:n]), ('discriminator_warmup', windows[n:])]:
        for w in ws:
            grouped[w.source_id][label].append(w)
    verified = {}
    pcm = defaultdict(Counter)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sid, result, stats in pool.map(source_check,
                ((row, grouped[sid], counts[sid]) for sid, row in sorted(all_rows.items()))):
            verified[sid] = result
            for label, value in stats.items():
                pcm[label].update(value)
    for values in pcm.values():
        denom = values['complete_20ms_windows']
        for key in ('quiet_20ms_windows_rms_le_0_001', 'very_quiet_20ms_windows_rms_le_0_00001', 'exact_zero_20ms_windows'):
            values[key + '_seconds'] = values[key] * .02
            values[key + '_percent'] = values[key] * 100 / denom
    expressive = BASE / 'remediation/expressive'
    metadata = plan['metadata']
    labels = {}
    for name in ('fresh-conditions.json', 'emogator-conditions.json'):
        require(file_sha(expressive / name) == metadata['supplement_identity']['files_sha256'][name], 'Supplement label source changed')
        labels[name] = read(expressive / name)
    condition_matches = Counter()
    for w in windows:
        if w.source_id in labels['fresh-conditions.json']:
            require(w.condition in labels['fresh-conditions.json'][w.source_id], 'Fresh action label mismatch')
            condition_matches['fresh_explicit_source_labels'] += 1
        elif w.source_id in labels['emogator-conditions.json']['by_source']:
            require(w.condition == labels['emogator-conditions.json']['by_source'][w.source_id]['condition'], 'Generic emotion label mismatch')
            condition_matches['generic_emotion_source_labels'] += 1
        else:
            condition_matches['original_sealed_plan_labels'] += 1
    panel_groups = {}
    for key in ('language', 'condition', 'dataset'):
        buckets = defaultdict(lambda: {'crops': 0, 'samples': 0, 'sources': set()})
        for r in natural:
            value = normalize_language(r['metadata'].get(key, 'unspecified')) if key == 'language' else r['metadata'].get(key, 'unspecified')
            buckets[value]['crops'] += 1
            buckets[value]['samples'] += r['samples']
            buckets[value]['sources'].add(r['source_id'])
        panel_groups[key] = {k: {'crops': v['crops'], 'sources': len(v['sources']),
            'scored_seconds_including_overlapping_crops': v['samples'] / 48000} for k, v in sorted(buckets.items())}
    audit = {'format_version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': 'Completed six-arm 200-update paired screen; read-only source bytes and saved reports, no inference or training',
        'experiment_identity_sha256': digest(identity), 'summary_sha256': file_sha(root / 'summary-compact.json'),
        'audit_script_sha256': file_sha(Path(__file__)), 'parent_sha256': receipt['checkpoint_sha256'],
        'plan_identity_sha256': receipt['plan_identity_sha256'], 'plan_path': str(plan_path),
        'matched_generator_exposure': exposure(windows[:n], rows),
        'matched_new_discriminator_only_exposure': exposure(windows[n:], rows),
        'all_selected_exposure': exposure(windows, rows),
        'quiet_source_pcm': dict(pcm),
        'heldout': {'natural_crops': len(natural), 'natural_sources': len(heldout),
            'natural_20ms_quiet_windows_by_teacher_rms': sum(r['quiet_windows']['quiet_window_count'] for r in natural),
            'synthetic_fixtures': [{'name': r['source_id'], 'seconds': r['samples'] / 48000} for r in synthetic],
            'by': panel_groups, 'indic_languages_missing': sorted(set(INDIC22) - set(panel_groups['language'])),
            'split_counts_by_source': dict(Counter(r.split for r in heldout.values())),
            'heldout_crops_may_overlap': True},
        'integrity': {'selected_source_count': len(selected), 'heldout_source_count': len(heldout),
            'audio_files_rehashed_and_pcm_shape_verified': len(verified),
            'audio_bytes_rehashed': sum(v['bytes'] for v in verified.values()),
            'verified_audio_identity_digest': digest(verified),
            'all_selected_rows_training_split': True,
            'selected_source_split_counts': dict(Counter(r.source_split for r in selected.values())),
            'no_official_dev_or_test_partition_used_for_training': True,
            'all_reserved_rows_checked': len(plan['reserved']),
            'heldout_source_audio_parent_identity_intersections': disjoint,
            'known_speaker_and_session_intersections': 0,
            'scored_interval_overlap_counts': overlaps,
            'no_overlap_with_parent_prior_scored_exposure_or_calibration': True,
            'all_target_crop_keys_lengths_and_context_match': True,
            'label_provenance_window_counts': dict(condition_matches),
            'unknown_fleurs_session_placeholders_are_not_actual_shared_speakers': True,
            'all_six_saved_panels_have_matching_crop_and_sample_identity': True},
        'initialization': arm_checks,
        'interpretation': [
            'All arms continue the same trained step-8090 student; this is not training a new randomly initialized decoder.',
            'Only fresh_magnitude and complex receive the matched extra 640 crops for 20 discriminator-only warmup updates; their student is frozen during warmup.',
            'The 5% whole-continuation quota counts explicit named source labels plus generic Emogator emotional_nonverbal; it is not 5% named edge cases.',
            'Named event durations are durations of crops drawn from source-labelled recordings, not measured time occupied by the event within each crop. No listening relabeling was performed.',
            'No synthetic silence, synthetic low-noise or fade fixture entered training. Natural recordings can contain exact-zero or quiet intervals; the separate PCM statistics quantify these.',
            'Raw input quiet statistics use complete 20 ms windows at 16 kHz and cannot substitute for teacher-output quiet metrics at 48 kHz.',
            'Scored audio is not repeated within an arm, including its warmup. Causal context can overlap by design. Debugging arms deliberately reuse matched audio.',
            'Exact source bytes and known parent/speaker/session identities are checked. This does not prove absence of undetected re-encoded acoustic duplicates or undocumented speaker identities.',
            'Validation audio is already used for model selection. These are development diagnostics, not an untouched final test result.'
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'generator': {k: audit['matched_generator_exposure'][k]
        for k in ('windows', 'sources', 'hours', 'known_languages', 'explicit_source_label_duration_percent',
                  'generic_emogator_duration_percent', 'nominal_broad_expressive_duration_percent')},
        'verified_audio_files': len(verified), 'heldout_crops': len(natural) + len(synthetic)}))


if __name__ == '__main__':
    main()
