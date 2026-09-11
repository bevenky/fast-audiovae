"""Bounded train-side quiet-loss calibration; discard both fitted models."""
from copy import deepcopy
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from audiovae_student.cached_comparison_data import load_parent_crops
from audiovae_student.corpus_training import _atomic_json
from audiovae_student.distillation_training import evaluate_crops, waveform_gate
from audiovae_student.objective_comparison import ObjectiveComparisonConfig, _new_arm, _common_start, _validate_inputs, state_fingerprint
from audiovae_student.training import _restore_rng

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r4')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PARENT = Path('/workspace/fast-audiovae-convnext-20260909-r3/training-runs/objective-comparison-v1/waveform-only/checkpoint-step001900.pt')
ORIGINAL = Path('/workspace/fast-audiovae-convnext-20260909-r2/training-runs/corrected-waveform-preflight-v1/checkpoint-step000500.pt')
EXPECTED = '08076f2b2c0bc853e6c6b6c80120d86a3ae209c91830d04ef7975e45cf6f34d0'
EXPECTED_ORIGINAL = 'df21671f774892cf5f2af0179da9fdfdc3df62ae6b195d66a11de125bb1a9fdd'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    nonquiet = [r for r in rows if r['teacher_rms'] >= 1e-3]
    quiet = [w for r in rows for w in r['quiet_windows']['windows'] if w['is_quiet']]
    if not nonquiet or not quiet:
        raise ValueError('The declared calibration requires both speech and quiet windows')
    return {'crops': len(rows), 'nonquiet_crops': len(nonquiet),
            'nonquiet_cosine_min': min(r['waveform_cosine'] for r in nonquiet),
            'nonquiet_cosine_mean': sum(r['waveform_cosine'] for r in nonquiet) / len(nonquiet),
            'nonquiet_cosine_099_count': sum(r['waveform_cosine'] >= .99 for r in nonquiet),
            'teacher_waveform_mean': sum(r['teacher_waveform'] for r in rows) / len(rows),
            'teacher_mel_mean': sum(r['teacher_mel'] for r in rows) / len(rows),
            'quiet_windows': len(quiet), 'quiet_failed': sum(w['passed'] is False for w in quiet),
            'quiet_residual_rms_mean': sum(w['residual_rms'] for w in quiet) / len(quiet),
            'quiet_normalized_residual_mean': sum(w['residual_rms'] / max(.001, w['teacher_rms']) for w in quiet) / len(quiet)}


def main():
    output = BASE / 'training-runs/quiet-calibration-v1'
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if sha(PARENT) != EXPECTED or sha(ORIGINAL) != EXPECTED_ORIGINAL:
        raise ValueError('Original checkpoint changed')
    original = torch.load(ORIGINAL, map_location='cpu', weights_only=True)
    fitted, sentinel, evidence = load_parent_crops(original, ORIGINAL.parents[2] / 'teacher-cache-preflight-v1')
    del original
    parent = torch.load(PARENT, map_location='cpu', weights_only=True)
    parent_fingerprint = state_fingerprint(parent)
    config = ObjectiveComparisonConfig(total_steps=150, evaluation_interval=50, checkpoint_interval=150,
        warmup_steps=10, learning_rate=2e-4, warmup_start_learning_rate=2e-5,
        parameter_update_metrics_interval=50, expected_parent_step=1900)
    _validate_inputs(parent, fitted, sentinel, evidence, config)
    protocol = {'kind': 'fixed_train_side_quiet_calibration', 'parent_sha256': EXPECTED,
        'original_crop_parent_sha256': EXPECTED_ORIGINAL, 'config': asdict(config),
        'quiet_shares': [0.0, 0.1], 'teacher_inference_calls': 0,
        'trained_weights_discarded': True, 'new_main_scored_hours': 0,
        'selection_rule': {'fit_quiet_normalized_residual_ratio_max': .9,
            'fit_waveform_loss_ratio_max': 1.1, 'fit_nonquiet_cosine_min': .99,
            'sentinel_used_to_select': False}, 'data': evidence}
    _atomic_json(output / 'protocol.json', protocol)
    starts, final = {}, {}
    for name, share in (('control', 0.0), ('quiet10', .1)):
        _restore_rng(deepcopy(parent['rng']))
        engine, _ = _new_arm(parent['engine'], 1.0, config, 'cuda')
        engine.fork_reconstruction(replace(engine.config, quiet_gradient_share=share, quiet_window_gate=True))
        _restore_rng(deepcopy(parent['rng']))
        starts[name] = _common_start(engine)
        if len(starts) == 2 and starts['control'] != starts['quiet10']:
            raise ValueError('Paired calibration starts differ')
        arm = output / name
        arm.mkdir()
        writer = SummaryWriter(str(OLD / 'runs' / ('quiet-calibration-v1-' + name)), flush_secs=5)
        started = time.monotonic()
        with (arm / 'train.jsonl').open('x') as log:
            while True:
                if engine.step % 50 == 0:
                    groups = {role: evaluate_crops(engine, crops) for role, crops in (('fitted', fitted), ('sentinel', sentinel))}
                    summary = {role: summarize(rows) for role, rows in groups.items()}
                    gates = {role: waveform_gate(engine, rows) for role, rows in groups.items()}
                    _atomic_json(arm / f'evaluation-{engine.step:06d}.json', {'rows': groups, 'summary': summary, 'gates': gates})
                    for role, measures in summary.items():
                        for key, value in measures.items():
                            writer.add_scalar(f'evaluation/{role}/{key}', value, engine.step)
                    _atomic_json(output / 'status.json', {'state': 'running', 'arm': name, 'step': engine.step,
                        'total_steps_per_arm': 150, 'elapsed_seconds': time.monotonic() - started, 'summary': summary})
                    print(json.dumps({'arm': name, 'step': engine.step, 'summary': summary}), flush=True)
                if engine.step == 150:
                    final[name] = summary
                    break
                metrics = engine.train_step(fitted)
                log.write(json.dumps(metrics, allow_nan=False) + '\n')
                log.flush()
                for key, value in metrics.items():
                    writer.add_scalar('train/' + key, value, engine.step)
                if share and metrics['teacher_quiet/achieved_share'] > share + 1e-6:
                    raise ValueError('Quiet gradient exceeded its declared budget')
        writer.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    control, candidate = final['control']['fitted'], final['quiet10']['fitted']
    checks = {'quiet_residual_improves_10pct': candidate['quiet_normalized_residual_mean'] <= .9 * control['quiet_normalized_residual_mean'],
        'waveform_within_10pct': candidate['teacher_waveform_mean'] <= 1.1 * control['teacher_waveform_mean'],
        'nonquiet_min_099': candidate['nonquiet_cosine_min'] >= .99}
    if state_fingerprint(parent) != parent_fingerprint or sha(PARENT) != EXPECTED or sha(ORIGINAL) != EXPECTED_ORIGINAL:
        raise ValueError('Parent state was mutated')
    result = {'state': 'completed', 'arms': final, 'selection_checks': checks,
        'quiet10_selected': all(checks.values()), 'sentinel_used_to_select': False,
        'same_initial_state': starts['control'] == starts['quiet10'],
        'teacher_inference_calls': 0, 'parent_unchanged': True,
        'trained_weights_discarded': True, 'source_archive_sha256': sha(BASE / 'source.tgz')}
    _atomic_json(output / 'summary.json', result)
    _atomic_json(output / 'status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
