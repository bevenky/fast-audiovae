"""Read-only, valid-prefix startup comparison for the preserved fusion parent.

Run with the frozen fusion package available on PYTHONPATH. This script uses
CPU inference only, never updates models, and does not measure runtime or RTF.
It writes a new report without modifying the running screen or its diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.fusion_evaluation import evaluate_fusion
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.restart_data import file_sha
from audiovae_student.training import _restore_rng, _rng_state


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
SCREEN = BASE / 'remediation' / 'fusion-screen'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_panels(base, screen, parent):
    """Verify the same archived sources plus fixed synthetic teacher targets."""
    paths = [(base / 'heldout-targets.pt', parent['identity']['heldout_metadata'], 'original')]
    for name in ('validation-appendix', 'language-validation'):
        directory = base / 'remediation' / name
        paths.append((directory / 'heldout-targets.pt',
                      json.loads((directory / 'groups.json').read_text()), name))
    crops, metadata, source_hashes = [], {}, {}
    for path, groups, name in paths:
        actual_sha = file_sha(path)
        source_hashes[str(path)] = actual_sha
        if name == 'original':
            expected_sha = parent['identity']['data']['heldout_targets']['sha256']
            expected_crops = parent['identity']['data']['heldout']['crops']
        else:
            descriptor_path = path.with_suffix('.json')
            descriptor = json.loads(descriptor_path.read_text())
            source_hashes[str(descriptor_path)] = file_sha(descriptor_path)
            source_hashes[str(path.parent / 'groups.json')] = file_sha(path.parent / 'groups.json')
            _require(descriptor['teacher_state_sha256'] == parent['identity']['data']['teacher_state_sha256'],
                     'Appendix teacher identity differs from the preserved parent')
            expected_sha, expected_crops = descriptor['sha256'], descriptor['crops']
        _require(actual_sha == expected_sha, 'Archived target file hash differs: ' + str(path))
        saved = torch.load(path, map_location='cpu', weights_only=True)
        loaded = [TrainingCrop(**value) for value in saved['crops']]
        _require(_crop_identity(loaded) == expected_crops, 'Archived crop identity differs: ' + str(path))
        for source_id, value in groups.items():
            _require(source_id not in metadata or metadata[source_id] == value,
                     'Conflicting metadata for a repeated source')
            metadata[source_id] = value
        crops.extend(loaded)
    synthetic_path = screen / 'synthetic-targets.pt'
    source_hashes[str(synthetic_path)] = file_sha(synthetic_path)
    saved = torch.load(synthetic_path, map_location='cpu', weights_only=True)
    synthetic = [TrainingCrop(**value) for value in saved['crops']]
    for crop in synthetic:
        _require(crop.source_id not in metadata, 'Synthetic source collides with archived metadata')
        metadata[crop.source_id] = {'dataset': 'synthetic_fixture', 'event': crop.source_id,
            'condition': crop.source_id, 'language': 'none', 'speech': False}
    crops.extend(synthetic)
    keys = [(crop.source_id, crop.start_frame) for crop in crops]
    _require(len(keys) == len(set(keys)), 'Duplicate source/start crop in evaluation panel')
    for crop in crops:
        _validate_crop(crop)
        _require(crop.start_frame == crop.context_start_frame + crop.context_frames,
                 'Crop causal context does not match its start')
    return crops, metadata, source_hashes


def compare_startup(control, zero, crops):
    """Compare real utterance prefixes only, clipping by original valid length."""
    rows = []
    modes = {module: module.training for engine in (control, zero) for module in engine.model.modules()}
    before = {name: state_fingerprint(engine.model.state_dict()) for name, engine in
              (('control', control), ('zero_padding', zero))}
    try:
        control.model.eval()
        zero.model.eval()
        with torch.no_grad():
            for crop in crops:
                _require(crop.context_start_frame == 0, 'Only true utterance prefixes belong in startup diagnostics')
                latents = crop.latents.detach().cpu()
                _require(bool(torch.isfinite(latents).all()), 'Nonfinite startup latents')
                a, b = control.model(latents), zero.model(latents)
                expected_shape = (1, 1, latents.shape[-1] * DECODER_HOP)
                _require(tuple(a.shape) == tuple(b.shape) == expected_shape,
                         'Decoder output sample count changed; do not truncate to hide it')
                _require(tuple(crop.teacher_audio.shape) == expected_shape,
                         'Teacher cached frame shape differs from decoder output')
                # The cached frame tensor can include a zero-padded final latent.
                # Context is real at context_start==0; stop excludes every invalid tail sample.
                total_valid = crop.scored_slice.stop
                _require(0 < total_valid <= expected_shape[-1], 'Invalid true prefix length')
                a, b = a[..., :total_valid], b[..., :total_valid]
                target = crop.teacher_audio.detach().cpu()[..., :total_valid]
                _require(all(bool(torch.isfinite(value).all()) for value in (a, b, target)),
                         'Nonfinite audio inside the genuine valid prefix')
                boundary = min(29 * DECODER_HOP, total_valid)
                control_mae = float((a[..., :boundary] - target[..., :boundary]).abs().mean())
                zero_mae = float((b[..., :boundary] - target[..., :boundary]).abs().mean())
                row = {'source_id': crop.source_id, 'start_frame': crop.start_frame,
                    'context_start_frame': crop.context_start_frame, 'context_frames': crop.context_frames,
                    'scored_frames': crop.scored_frames, 'crop_frames': latents.shape[-1],
                    'valid_scored_samples': crop.valid_scored_samples,
                    'full_generated_samples': expected_shape[-1],
                    'total_valid_prefix_samples': total_valid,
                    'padded_tail_samples_excluded': expected_shape[-1] - total_valid,
                    'startup_samples': boundary, 'mature_samples': total_valid - boundary,
                    'control_startup_mae': control_mae, 'zero_startup_mae': zero_mae,
                    'zero_minus_control_startup_mae': zero_mae - control_mae,
                    'startup_max_difference': float((a[..., :boundary] - b[..., :boundary]).abs().max()),
                    'mature_max_difference': None, 'control_mature_mae': None, 'zero_mature_mae': None}
                if total_valid > boundary:
                    row.update(mature_max_difference=float((a[..., boundary:] - b[..., boundary:]).abs().max()),
                        control_mature_mae=float((a[..., boundary:] - target[..., boundary:]).abs().mean()),
                        zero_mature_mae=float((b[..., boundary:] - target[..., boundary:]).abs().mean()))
                rows.append(row)
    finally:
        for module, mode in modes.items():
            module.training = mode
    for name, engine in (('control', control), ('zero_padding', zero)):
        _require(state_fingerprint(engine.model.state_dict()) == before[name],
                 'Startup comparison changed model tensors')
    return rows


def unique_source_summary(rows):
    """Avoid repeatedly weighting the same prefix in overlapping heldout crops."""
    longest = {}
    for row in rows:
        old = longest.get(row['source_id'])
        if old is None or row['total_valid_prefix_samples'] > old['total_valid_prefix_samples']:
            longest[row['source_id']] = row
    items = list(longest.values())
    samples = sum(row['startup_samples'] for row in items)
    return {'reduction': 'longest_valid_prefix_per_source_then_sample_weighted_startup_mae',
        'sources': len(items), 'startup_samples': samples,
        'control_startup_mae': sum(row['control_startup_mae'] * row['startup_samples'] for row in items) / samples,
        'zero_startup_mae': sum(row['zero_startup_mae'] * row['startup_samples'] for row in items) / samples,
        'mature_max_difference': max((row['mature_max_difference'] for row in items
                                      if row['mature_max_difference'] is not None), default=None),
        'selected_crops': [{'source_id': row['source_id'], 'start_frame': row['start_frame']} for row in items]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--screen', type=Path, default=SCREEN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.screen / 'corrected-startup.json'
    _require(not output.exists(), 'Refusing to overwrite an existing corrected startup report')
    receipt_path, parent_path = args.screen / 'parent-receipt.json', args.screen / 'parent.pt'
    receipt = json.loads(receipt_path.read_text())
    parent_sha = file_sha(parent_path)
    _require(parent_sha == receipt['checkpoint_sha256'], 'Preserved parent hash mismatch')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    _require(parent['engine']['step'] == 8090, 'This corrected diagnostic is for preserved step 8090')
    rng, previous_threads = _rng_state(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        all_crops, metadata, target_hashes = load_panels(args.base, args.screen, parent)
        # If preparation finished, bind synthetic data and all archived crops to
        # the already sealed screen. This does not wait on or modify its process.
        identity_path = args.screen / 'experiment-identity.json'
        experiment_sha = None
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            _require(identity['parent_sha256'] == parent_sha, 'Screen uses a different parent')
            _require(identity['heldout']['crops'] == _crop_identity(all_crops),
                     'Corrected panel differs from the screen target identity')
            experiment_sha = file_sha(identity_path)
        crops = [crop for crop in all_crops if crop.context_start_frame == 0]
        _require(bool(crops), 'No genuine utterance-start crops are available')
        engines, migrations = {}, {}
        for variant in ('control', 'zero_padding'):
            engines[variant], migrations[variant] = build_fusion_engine(parent['engine'], variant, device='cpu')
        rows = compare_startup(engines['control'], engines['zero_padding'], crops)
        # These quality rows score each crop's heldout scored region, not its
        # unscored history. Both arms use the evaluator's same explicit mask.
        quality = {variant: evaluate_fusion(engine, crops, metadata) for variant, engine in engines.items()}
        _require(file_sha(parent_path) == parent_sha, 'Preserved parent changed while diagnosing')
        source_hashes = {str(Path(__file__).resolve()): file_sha(Path(__file__))}
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('audiovae_student.') and path and Path(path).suffix == '.py':
                source_hashes[str(Path(path).resolve())] = file_sha(Path(path))
        result = {'format_version': 1, 'kind': 'corrected_valid_prefix_startup_diagnostic',
            'parent_checkpoint': str(parent_path), 'parent_checkpoint_sha256': parent_sha,
            'parent_receipt_sha256': file_sha(receipt_path), 'parent_step': parent['engine']['step'],
            'experiment_identity_sha256': experiment_sha,
            'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'torch_version': str(torch.__version__),
            'model_updates': 0, 'teacher_inference': False, 'runtime_benchmark': False,
            'source_hashes': source_hashes, 'target_hashes': target_hashes,
            'all_panel_crops': len(all_crops), 'selected_utterance_start_crops': len(crops),
            'selected_crop_identity': _crop_identity(crops), 'migrations': migrations,
            'startup_policy': {'context_start_frame': 0, 'maximum_startup_samples': 29 * DECODER_HOP,
                'valid_prefix_stop': 'crop.scored_slice.stop', 'padded_tail_included': False,
                'sample_count_mismatch': 'fail_without_truncation',
                'overlapping_crop_prefixes': 'reported_per_crop; summary_selects_longest_per_source'},
            'rows': rows, 'unique_source_summary': unique_source_summary(rows),
            'heldout_scored_region_quality': quality,
            'model_tensors_preserved': True, 'global_rng_preserved': True}
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    finally:
        _restore_rng(rng)
        torch.set_num_threads(previous_threads)
    temporary = output.with_suffix(output.suffix + '.tmp')
    _require(not temporary.exists(), 'An unfinished corrected report already exists')
    with temporary.open('x') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({'output': str(output), 'parent_step': result['parent_step'],
        'selected_crops': len(rows), 'unique_source_summary': result['unique_source_summary'],
        'control_quality': quality['control']['summary'],
        'zero_quality': quality['zero_padding']['summary'], 'runtime_benchmark': False}, allow_nan=False))


if __name__ == '__main__':
    main()
