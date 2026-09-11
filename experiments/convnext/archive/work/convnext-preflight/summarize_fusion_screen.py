"""Verify and summarize saved paired fusion-screen results without inference.

Checkpoint reads verify provenance and tensor fingerprints; no model is built.
Differences are descriptive, not statistical significance or winner selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path


DEFAULT = Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation/fusion-screen')


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()


def file_sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path):
    result = json.loads(path.read_text())
    json.dumps(result, allow_nan=False)
    return result


def json_metadata(value):
    """Match checkpoint configuration tuples to their JSON array encoding.

    Migration receipts contain only JSON metadata, never tensors or optimizer
    state. Tensor/state fingerprints retain their exact type-aware comparison.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def state_fingerprint(value):
    """Identical to the experiment's nested-state fingerprint, without imports."""
    import torch
    checksum = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            checksum.update(b'tensor')
            checksum.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            checksum.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            checksum.update(type(item).__name__.encode()); checksum.update(str(len(item)).encode())
            for child in item: visit(child)
        elif item is None or type(item) in (bool, int, float, str):
            checksum.update(type(item).__name__.encode()); checksum.update(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError('Unsupported state fingerprint type: ' + type(item).__name__)
    visit(value)
    return checksum.hexdigest()


def crop_key(row):
    return row['source_id'], row['start_frame']


def row_map(report):
    rows = report['rows']
    mapping = {crop_key(row): row for row in rows}
    if not rows or len(mapping) != len(rows):
        raise ValueError('Empty or duplicated evaluation crops')
    for row in rows:
        if type(row['samples']) is not int or row['samples'] <= 0:
            raise ValueError('Invalid scored sample count')
        if row['original_scored_samples'] - row['context_excluded_samples'] != row['samples']:
            raise ValueError('Evaluation exclusion/sample count mismatch')
    return mapping


def assert_matching_reports(left, right):
    for key in ('criterion_config', 'mask_policy', 'quiet_config'):
        if left[key] != right[key]:
            raise ValueError('Different common evaluation policy: ' + key)
    a, b = row_map(left), row_map(right)
    if a.keys() != b.keys():
        raise ValueError('Evaluation crop identities differ')
    invariant = ('context_start_frame', 'original_scored_samples', 'context_excluded_samples', 'samples',
        'scored_start_sample_in_crop', 'scored_stop_sample_in_crop', 'absolute_scored_start_sample',
        'teacher_rms', 'teacher_peak_abs', 'teacher_overshoot_samples', 'teacher_near_saturation_samples',
        'near_saturation_threshold', 'metadata', 'cohort')
    quiet_invariant = ('batch_index', 'window_index', 'start_sample', 'stop_sample', 'valid_samples', 'teacher_rms', 'is_quiet')
    for key in a:
        if any(a[key][field] != b[key][field] for field in invariant):
            raise ValueError('Crop targets, metadata or retained samples differ: ' + str(key))
        aw, bw = a[key]['quiet_windows']['windows'], b[key]['quiet_windows']['windows']
        if len(aw) != len(bw) or any(any(x[field] != y[field] for field in quiet_invariant) for x, y in zip(aw, bw)):
            raise ValueError('Quiet target/window grids differ: ' + str(key))


def synthetic(row):
    return row['metadata'].get('dataset') == 'synthetic_fixture'


def summarize_rows(rows, quiet_threshold):
    rows = list(rows)
    if not rows: return None
    samples = sum(row['samples'] for row in rows)
    quiet = [window for row in rows for window in row['quiet_windows']['windows'] if window['is_quiet']]
    quiet_samples = sum(window['valid_samples'] for window in quiet)
    active = [row for row in rows if row['teacher_rms'] >= quiet_threshold and row.get('waveform_cosine_defined', True)]
    hf = sum(row['high_frequency']['elements'] for row in rows)
    phase = sorted(row['quiet_phase480']['residual_template_rms'] for row in rows
                   if row['quiet_phase480']['residual_template_rms'] is not None)
    def weighted(name):
        return sum(row['high_frequency'][name] * row['high_frequency']['elements'] for row in rows) / hf
    return {'crops': len(rows), 'sources': len({row['source_id'] for row in rows}), 'scored_samples': samples,
        'raw_mae_sample_weighted': sum(row['waveform_raw_mae'] * row['samples'] for row in rows) / samples,
        'fixed_mel_equal_crop_mean': sum(row['teacher_mel'] for row in rows) / len(rows),
        'nonquiet_crops': len(active),
        'nonquiet_cosine_mean': sum(row['waveform_cosine'] for row in active) / len(active) if active else None,
        'quiet_windows': len(quiet), 'quiet_scored_samples': quiet_samples,
        'quiet_failed_windows': sum(window['passed'] is False for window in quiet),
        'quiet_residual_rms_pooled': math.sqrt(sum(window['residual_rms'] ** 2 * window['valid_samples'] for window in quiet) / quiet_samples) if quiet_samples else None,
        'quiet_residual_rms_mean': sum(window['residual_rms'] for window in quiet) / len(quiet) if quiet else None,
        'quiet_phase480_residual_median': ((phase[(len(phase)-1)//2] + phase[len(phase)//2]) / 2) if phase else None,
        'peak_abs_max': max(row['student_peak_abs'] for row in rows),
        'overshoot_samples_gt_1': sum(row['student_overshoot_samples'] for row in rows),
        'overshoot_crops': sum(row['student_overshoot_samples'] > 0 for row in rows),
        'overshoot_sources': len({row['source_id'] for row in rows if row['student_overshoot_samples'] > 0}),
        'near_saturation_samples': sum(row['student_near_saturation_samples'] for row in rows),
        'high_frequency_magnitude_mae': weighted('magnitude_mae'),
        'high_frequency_log_magnitude_mae': weighted('log_magnitude_mae'),
        'high_frequency_complex_residual_rms': math.sqrt(sum(row['high_frequency']['complex_residual_rms'] ** 2 * row['high_frequency']['elements'] for row in rows) / hf)}


def cohorts(report):
    rows = report['rows']; threshold = report['quiet_config']['quiet_teacher_rms_max']
    natural = [row for row in rows if not synthetic(row)]
    fixtures = [row for row in rows if synthetic(row)]
    return {'natural': summarize_rows(natural, threshold),
        'speech': summarize_rows((row for row in natural if row['cohort'] == 'speech'), threshold),
        'expressive': summarize_rows((row for row in natural if row['cohort'] == 'expressive'), threshold),
        'synthetic': summarize_rows(fixtures, threshold),
        'synthetic_cases': {row['source_id']: summarize_rows([row], threshold) for row in fixtures}}


def delta(candidate, reference):
    if candidate is None or reference is None: return None
    keys = ('raw_mae_sample_weighted', 'fixed_mel_equal_crop_mean', 'nonquiet_cosine_mean',
        'quiet_residual_rms_pooled', 'quiet_residual_rms_mean', 'quiet_phase480_residual_median',
        'peak_abs_max', 'overshoot_samples_gt_1', 'overshoot_crops', 'overshoot_sources', 'near_saturation_samples',
        'high_frequency_magnitude_mae', 'high_frequency_log_magnitude_mae', 'high_frequency_complex_residual_rms')
    result = {}
    for key in keys:
        now, old = candidate.get(key), reference.get(key)
        result[key] = {'difference': now - old if now is not None and old is not None else None,
            'relative_percent': 100 * (now / old - 1) if now is not None and old not in (None, 0) else None}
    return result


def paired(candidate, reference):
    assert_matching_reports(candidate, reference)
    now, old = row_map(candidate), row_map(reference)
    rows, groups = [], defaultdict(list)
    for key, row in now.items():
        base = old[key]
        change = row['waveform_raw_mae'] - base['waveform_raw_mae']
        entry = {'source_id': key[0], 'start_frame': key[1], 'samples': row['samples'],
            'synthetic': synthetic(row), 'metadata': row['metadata'],
            'raw_mae_difference': change,
            'raw_mae_relative_percent': 100 * change / base['waveform_raw_mae'] if base['waveform_raw_mae'] else None,
            'fixed_mel_difference': row['teacher_mel'] - base['teacher_mel'],
            'cosine_difference': row['waveform_cosine'] - base['waveform_cosine'],
            'overshoot_samples_difference': row['student_overshoot_samples'] - base['student_overshoot_samples']}
        rows.append(entry)
        if not synthetic(row):
            language = row['metadata'].get('language') or 'unverified'
            event = row['metadata'].get('condition') or row['metadata'].get('event') or 'unverified'
            groups['language/' + str(language)].append(key)
            groups['event/' + str(event)].append(key)
    threshold = candidate['quiet_config']['quiet_teacher_rms_max']
    group_rows = []
    for name, keys in groups.items():
        a, b = summarize_rows([now[k] for k in keys], threshold), summarize_rows([old[k] for k in keys], threshold)
        group_rows.append({'group': name, 'crops': len(keys), 'sources': a['sources'],
            'candidate': a, 'reference': b, 'change': delta(a, b)})
    def worst(prefix):
        return sorted([row for row in group_rows if row['group'].startswith(prefix)],
            key=lambda row: row['change']['raw_mae_sample_weighted']['relative_percent']
                if row['change']['raw_mae_sample_weighted']['relative_percent'] is not None else -math.inf,
            reverse=True)[:5]
    def counts(selected):
        return {'crops': len(selected), 'raw_mae_improved': sum(r['raw_mae_difference'] < -1e-12 for r in selected),
            'raw_mae_regressed': sum(r['raw_mae_difference'] > 1e-12 for r in selected),
            'raw_mae_tied_within_1e_12': sum(abs(r['raw_mae_difference']) <= 1e-12 for r in selected),
            'fixed_mel_improved': sum(r['fixed_mel_difference'] < -1e-9 for r in selected),
            'fixed_mel_regressed': sum(r['fixed_mel_difference'] > 1e-9 for r in selected)}
    return {'natural_paired_counts': counts([r for r in rows if not r['synthetic']]),
        'synthetic_paired_counts': counts([r for r in rows if r['synthetic']]),
        'worst_five_language_groups_by_relative_mae': worst('language/'),
        'worst_five_event_groups_by_relative_mae': worst('event/'),
        'worst_five_natural_crops_by_absolute_mae': sorted([r for r in rows if not r['synthetic']], key=lambda r:r['raw_mae_difference'], reverse=True)[:5],
        'paired_per_crop_changes': rows}


def validate_report(report, expected_crops, expected_step, expected_parameter_sha, *, variant, base_config, migration):
    wrapped = variant in ('tanh', 'filter', 'zero_padding')
    expected_identity = json_metadata({
        'class': ('audiovae_student.fusion_architecture.FusionStudentDecoder' if wrapped
                  else 'audiovae_student.model.StudentDecoder'),
        'base_config': base_config,
        'architecture_config': migration['architecture'] if wrapped else None,
        'fusion_migration': migration,
        'parameter_state_sha256': expected_parameter_sha})
    expected_model_sha = state_fingerprint(expected_identity)
    if (report['evaluated_step'] != expected_step or report.get('model_identity') != expected_identity
            or report['model_state_sha256'] != expected_model_sha):
        raise ValueError('Evaluation step, parameter fingerprint or full model identity differs')
    for key in ('model_tensors_preserved', 'module_modes_preserved', 'global_rng_preserved'):
        if report.get(key) is not True: raise ValueError('Missing read-only evaluation evidence: ' + key)
    if report.get('optimizer_updates') != 0: raise ValueError('Evaluation includes optimizer updates')
    rows = row_map(report)
    if rows.keys() != expected_crops.keys(): raise ValueError('Panel membership differs from experiment identity')
    for key, row in rows.items():
        crop = expected_crops[key]
        if (row['original_scored_samples'] != crop['valid_scored_samples']
                or row['context_start_frame'] != crop['start_frame'] - crop['context_frames']
                or row['model_state_sha256'] != expected_model_sha or row['evaluated_step'] != expected_step):
            raise ValueError('Evaluation row differs from frozen crop or model identity')


def summarize(root, allow_partial=False):
    import torch
    torch.set_num_threads(1)
    identity = load_json(root / 'experiment-identity.json'); identity_sha = digest(identity)
    receipt = load_json(root / 'parent-receipt.json')
    parent_path = root / 'parent.pt'
    if file_sha(parent_path) != receipt['checkpoint_sha256'] or identity['parent_sha256'] != receipt['checkpoint_sha256']:
        raise ValueError('Parent checkpoint identity differs')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    parent_engine_sha = state_fingerprint(parent['engine'])
    original = parent['engine']['model']; original_sha = state_fingerprint(original)
    expected = {(row['source_id'], row['start_frame']): row for row in identity['heldout']['crops']}
    if len(expected) != identity['heldout']['count']: raise ValueError('Duplicated heldout identity')
    if identity['parent_step'] != parent['engine']['step']: raise ValueError('Parent step changed')
    for path, expected_sha in identity['heldout']['source_hashes'].items():
        if file_sha(Path(path)) != expected_sha: raise ValueError('Frozen panel source changed: ' + path)
    arms, reports, source_hashes, pending = {}, {}, {}, []
    for arm in identity['arms']:
        directory = root / arm
        if not (directory / 'complete.json').is_file():
            pending.append(arm)
            if allow_partial: continue
            raise ValueError('Arm is incomplete: ' + arm)
        if (directory / 'inflight.json').exists(): raise ValueError('Completed arm still has in-flight exposure')
        values = {name: load_json(directory / (name + '.json')) for name in ('complete', 'migration', 'before', 'after')}
        complete, migration, before, after = (values[name] for name in ('complete', 'migration', 'before', 'after'))
        checkpoint_path = directory / 'final.pt'
        if file_sha(checkpoint_path) != complete['checkpoint_sha256']: raise ValueError('Arm checkpoint hash differs: ' + arm)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        steps = identity['generator_steps']; final_step = identity['parent_step'] + steps
        expected_d_warmup = identity['discriminator_warmup_steps'] if arm in ('fresh_magnitude', 'complex') else 0
        if (checkpoint.get('format_version') != 'fusion_screen_v1'
                or checkpoint['engine'].get('format_version') != 'fusion_recipe_v1'
                or checkpoint['engine'].get('fusion_variant') != arm
                or json_metadata(checkpoint['engine'].get('fusion_migration')) != migration
                or checkpoint['experiment_identity_sha256'] != identity_sha
                or checkpoint['parent_sha256'] != receipt['checkpoint_sha256'] or checkpoint['variant'] != arm
                or checkpoint['training_target_identity_sha256'] != digest(identity['target_crops'])
                or json_metadata(checkpoint['migration']) != migration or checkpoint['generator_updates'] != steps
                or checkpoint['discriminator_only_updates'] != expected_d_warmup
                or complete['arm'] != arm or complete['updates'] != steps or complete['step'] != final_step
                or checkpoint['engine']['step'] != final_step):
            raise ValueError('Arm experiment/parent/target/update identity differs: ' + arm)
        if (migration['variant'] != arm or migration['parent_engine_state_sha256'] != parent_engine_sha
                or migration['original_student_state_sha256'] != original_sha
                or migration['parent_step'] != identity['parent_step']
                or any(migration.get(key) is not True for key in ('original_student_state_preserved',
                    'original_student_optimizers_preserved', 'calibration_preserved', 'crop_rng_preserved',
                    'balancer_preserved', 'training_step_preserved', 'discriminator_updates_preserved'))):
            raise ValueError('Original parameter/optimizer preservation differs: ' + arm)
        initial = dict(original)
        if arm == 'filter':
            weight = torch.zeros((1, 1, 7), dtype=original['output.weight'].dtype)
            weight[..., -1] = 1
            initial['output_filter.conv.weight'] = weight
        initial_sha = state_fingerprint(initial)
        validate_report(before, expected, identity['parent_step'], initial_sha, variant=arm,
                        base_config=parent['engine']['model_config'], migration=migration)
        final_parameter_sha = state_fingerprint(checkpoint['engine']['model'])
        validate_report(after, expected, final_step, final_parameter_sha, variant=arm,
                        base_config=checkpoint['engine']['model_config'], migration=migration)
        assert_matching_reports(before, after)
        if complete['before_summary'] != before['summary'] or complete['after_summary'] != after['summary']:
            raise ValueError('Completion summary differs from full report')
        reports[arm] = (before, after)
        arms[arm] = {'generator_updates': steps, 'discriminator_only_updates': checkpoint['discriminator_only_updates'],
            'training_seconds': checkpoint['seconds'], 'before': cohorts(before), 'after': cohorts(after),
            'within_arm': paired(after, before), 'initial_parameter_state_sha256': initial_sha,
            'initial_model_state_sha256': before['model_state_sha256'],
            'original_student_state_sha256': original_sha, 'final_parameter_state_sha256': final_parameter_sha,
            'final_model_state_sha256': after['model_state_sha256']}
        source_hashes[arm] = {name + '.json': file_sha(directory / (name + '.json')) for name in values}
        source_hashes[arm]['final.pt'] = complete['checkpoint_sha256']
        del checkpoint; gc.collect()
    comparisons = {}
    if reports:
        first = next(iter(reports.values()))[0]
        for before, _after in reports.values(): assert_matching_reports(first, before)
    for arm, reference in [('tanh', 'control'), ('short_mel', 'control'), ('filter', 'control'), ('complex', 'fresh_magnitude')]:
        if arm not in reports or reference not in reports: continue
        a, b = cohorts(reports[arm][1]), cohorts(reports[reference][1])
        comparisons[arm + '_vs_' + reference] = {'candidate': arm, 'reference': reference,
            'after_changes': {name: delta(a[name], b[name]) for name in ('natural', 'speech', 'expressive', 'synthetic')},
            'paired': paired(reports[arm][1], reports[reference][1])}
    timing = load_json(root / 'cpu-diagnostics.json') if (root / 'cpu-diagnostics.json').exists() else None
    compact_timing = {name: {key: value for key, value in row.items() if key in
        ('one_thread_unfused_stream_rtf_median', 'max_abs_error', 'sample_count_passed', 'input_frames')}
        for name, row in (timing or {}).items() if isinstance(row, dict)}
    return {'format_version': 1, 'experiment_identity_sha256': identity_sha,
        'parent_sha256': receipt['checkpoint_sha256'], 'parent_step': identity['parent_step'],
        'heldout_crops': len(expected), 'pending_arms': pending, 'arms': arms, 'comparisons': comparisons,
        'source_hashes': source_hashes, 'initial_cpu_diagnostics': compact_timing,
        'interpretation': {'relative_percent': '100 * (candidate/reference - 1); negative reduces an error metric',
            'paired_counts': 'Descriptive crop differences only. Overlapping crops are not independent trials.',
            'overshoot_counts': 'Sample counts sum overlapping scored crops, not unique physical events. Prefer peak maximum and affected crop/source counts.',
            'worst_groups': 'Largest relative raw-MAE change; counts shown. These may be improvements if every group improved.',
            'fixed_mel': 'Common unchanged diagnostic criterion, equal crop means; not candidate training loss.',
            'quiet': 'Natural and synthetic reported separately; residual RMS pooled by valid quiet samples.',
            'cpu_timing': 'Initial untrained-change, unfused one-thread CPU screen; not final student or production kernel RTF.',
            'zero_padding_startup': 'Initial startup table is not used; its padded-tail handling requires the separately corrected diagnostic.',
            'statistical_significance_established': False, 'winner_selected': False,
            'inference_or_optimizer_updates_performed_by_summarizer': False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = summarize(args.root, args.allow_partial)
    output = args.output or args.root / 'summary-compact.json'
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, output)
    print(json.dumps({'summary': str(output), 'completed_arms': list(result['arms']),
        'pending_arms': result['pending_arms'], 'comparisons': list(result['comparisons'])}), flush=True)


if __name__ == '__main__': main()
