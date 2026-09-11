"""Native upsampler delta ridge with calibration-only startup equalities.

The declared FP32-stability rank policy may make these hidden equalities
infeasible. That is a reported diagnostic outcome, never a reason to tune the
ridge, fit development data, or assert waveform reconstruction is impossible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import reconstruction_aware_init as ordinary

audit, common = ordinary.audit, ordinary.common
group, base, control, progressive = ordinary.group, ordinary.base, ordinary.control, ordinary.progressive
VERSION = 'audiovae2_startup_constrained_initialization_v1'
ATOL, RTOL = 1e-5, 1e-4
RIDGE = ordinary.RIDGE


def startup_cells(target, valid, crop, output_length):
    """Select only full native 8-sample cells inside actual source startup."""
    if (target.ndim != 3 or target.shape[:2] != (1, 1) or target.shape != valid.shape
            or valid.dtype != torch.bool or target.shape[-1] != output_length * 8
            or crop['start_frame'] - crop['context_start_frame'] != crop['context_frames']):
        raise ValueError('Startup mask geometry differs from original source and native output')
    start = crop['context_frames'] * 1920
    stop = start + crop['valid_scored_samples']
    if not 0 <= start < stop <= target.shape[-1]:
        raise ValueError('Invalid scored extent')
    expected = torch.zeros_like(valid); expected[..., start:stop] = True
    if not torch.equal(valid, expected):
        raise ValueError('Startup mask must use the original contiguous scored validity')
    scored = target[..., start:stop]
    raw = common.diagnostic_quiet_metrics(scored, scored, torch.ones_like(scored, dtype=torch.bool))
    mask = torch.zeros_like(valid); windows = []
    for window in raw['windows']:
        lo, hi = window['start_sample'], window['stop_sample']
        absolute_lo = crop['start_frame'] * 1920 + lo
        absolute_hi = crop['start_frame'] * 1920 + hi
        if absolute_lo == 0 and absolute_hi <= 960 and window['teacher_rms'] <= 1e-5:
            mask[..., start + lo:start + hi] = True
            windows.append({'source_id': crop['source_id'], 'source_start_sample': absolute_lo,
                            'source_stop_sample': absolute_hi, 'teacher_rms': window['teacher_rms'],
                            'valid_samples': hi - lo})
    selected = common.cell_weights(mask & valid, output_length) == 8
    samples = int((mask & valid).sum())
    return selected, {'source_id': crop['source_id'], 'windows': windows,
                      'observed_near_startup_samples': samples,
                      'constraint_cells': int(selected.sum()),
                      'excluded_partial_cell_samples': samples - 8 * int(selected.sum()),
                      'samples_per_cell': 8, 'previous_input_context_preserved': True}


def equality_metrics(predicted, expected):
    if predicted.shape != expected.shape or not predicted.numel():
        raise ValueError('Nonempty equality tensors with matching shape are required')
    residual = predicted.double() - expected.double()
    finite = bool(torch.isfinite(residual).all())
    limit = ATOL + RTOL * expected.double().abs()
    return {'passed': finite and bool((residual.abs() <= limit).all()),
            'max_abs': float(residual.abs().max()) if finite else None,
            'residual_rms': float(residual.square().mean().sqrt()) if finite else None,
            'max_fraction_of_original_tolerance': float((residual.abs() / limit).max()) if finite else None,
            'elements': residual.numel(), 'atol': ATOL, 'rtol': RTOL}


@torch.no_grad()
def solve_constrained_delta(stats, design, delta, teacher_output, ridge=RIDGE):
    """Project ordinary delta ridge in its Hessian metric using a fixed-rank SVD.

    Columns are ordinary native design coordinates. The last whitened coordinate
    represents mean response; this keeps the single bias unpenalized. All inputs
    are frozen observations, with no trainable projector or optimizer.
    """
    if ridge != RIDGE:
        raise ValueError('The original fixed ridge must remain unchanged')
    a0, c0, report = common.fit_affine(stats, ridge)
    d = stats['mean_x'].numel(); outputs = stats['mean_d'].numel()
    if (design.ndim != 2 or design.shape[1] != d or delta.shape != (design.shape[0], outputs)
            or teacher_output.shape != delta.shape or not design.shape[0]
            or not all(bool(torch.isfinite(x).all()) for x in (design, delta, teacher_output))):
        raise ValueError('Invalid or empty startup equality observations')
    x, y, reference = design.double(), delta.double(), teacher_output.double()
    mx, md = stats['mean_x'].double(), stats['mean_d'].double()
    covariance = stats['centered_xx'].double() / stats['n']
    covariance = (covariance + covariance.T) / 2
    lam = report['ridge_lambda']
    report.update(constraint_rows=x.shape[0], constraint_outputs=outputs,
                  rank_policy='max(whitened_design.shape) * float32 epsilon; absolute cutoff0; fixed before development',
                  ridge_unchanged=True, shared_bias=True,
                  interpretation='Stable feasibility at declared rank and original interface tolerance; hidden equality is not necessary for waveform fidelity')
    if not lam:
        report.update(feasible=False, status='no_stable_feasible_solution',
                      reason='All-valid covariance is degenerate under the original fixed-ridge policy')
        return None, None, report
    q = covariance + lam * torch.eye(d, device=x.device, dtype=torch.float64)
    chol, info = torch.linalg.cholesky_ex(q)
    if int(info): raise RuntimeError('Original regularized covariance is not positive definite')
    xc = x - mx
    whitened = torch.cat((torch.linalg.solve_triangular(chol, xc.T, upper=False).T,
                          torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float64)), 1)
    u, singular, vh = torch.linalg.svd(whitened, full_matrices=False,
        **({'driver': 'gesvd'} if whitened.is_cuda else {}))
    rank_rtol = max(whitened.shape) * torch.finfo(torch.float32).eps
    cutoff = float(singular[0]) * rank_rtol
    keep = singular > cutoff
    residual = y - (xc @ a0.T + md)
    correction = vh[keep].T @ ((u[:, keep].T @ residual) / singular[keep, None])
    a = (a0.T + torch.linalg.solve_triangular(chol.T, correction[:-1], upper=True)).T
    mean = md + correction[-1]
    c = mean - a @ mx
    predicted_delta = x @ a.T + c
    # Compare full teacher-interface values, not tiny correction magnitudes.
    current = reference - y
    compatibility = equality_metrics(current + predicted_delta, reference)
    ordinary_equality = equality_metrics(current + x @ a0.T + c0, reference)
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(c).all())
    report.update(feasible=finite and compatibility['passed'],
                  status='stable_feasible_solution' if finite and compatibility['passed'] else 'no_stable_feasible_solution',
                  rank_relative_cutoff=rank_rtol, rank_absolute_cutoff=cutoff,
                  retained_rank=int(keep.sum()), discarded_modes=int((~keep).sum()),
                  singular_values=singular.cpu().tolist(), cholesky_positive=True,
                  equality_before=ordinary_equality, equality_fp64=compatibility,
                  ordinary_delta_frobenius=float(a0.norm()), constrained_delta_frobenius=float(a.norm()),
                  change_from_ordinary_frobenius=float((a-a0).norm()),
                  ordinary_bias_delta_l2=float(c0.norm()), constrained_bias_delta_l2=float(c.norm()),
                  whitened_projection_norm=float(correction.norm()),
                  reason=None if finite and compatibility['passed'] else 'No stable feasible solution under the declared FP32-stability rank policy')
    if not report['feasible']: return None, None, report
    return a, c, report


@torch.no_grad()
def fit_constrained_delta(module, observations, chunk_rows=ordinary.CHUNK_ROWS):
    if ordinary._validate_module(module) is not True:
        raise ValueError('Only the existing stride5 native upsampler is fitted')
    if type(chunk_rows) is not int or chunk_rows < 1: raise ValueError('Positive chunk size required')
    stats = None; designs = []; deltas = []; targets = []; receipts = []; source_ids = []
    energy = None; weighted_samples = 0; feature_rows = 0
    for row in observations:
        x, target, current, weights, constraints = (row[k] for k in
            ('input', 'target', 'current_output', 'weights', 'constraint_mask'))
        length = x.shape[-1] * 5
        if (x.ndim != 3 or target.shape != current.shape or target.shape != (x.shape[0], module.out_channels, length)
                or weights.shape != (x.shape[0], 1, length) or constraints.shape != weights.shape
                or constraints.dtype != torch.bool or not torch.isfinite(weights).all()
                or (weights < 0).any() or (weights > 8).any() or (weights[constraints] != 8).any()):
            raise ValueError('Native all-valid or whole-cell constraint geometry differs')
        source_ids.append(row['source_id']); receipts.append(row.get('constraint_receipt', {}))
        change = target.detach().double() - current.detach().double()
        for start in range(0, length, chunk_rows):
            stop = min(start + chunk_rows, length)
            design = ordinary.native_design(x, start=start, stop=stop)
            w, z = weights[..., start:stop], change[..., start:stop]
            stats = common.accumulate_affine(stats, design, z, w)
            energy = ordinary._target_energy(energy, z, w)
            mask = constraints[..., start:stop].reshape(-1)
            if mask.any():
                designs.append(design.movedim(1, -1).reshape(-1, design.shape[1])[mask].double())
                deltas.append(z.movedim(1, -1).reshape(-1, z.shape[1])[mask])
                targets.append(target[..., start:stop].movedim(1, -1).reshape(-1, target.shape[1])[mask].double())
            weighted_samples += int(w.sum()); feature_rows += int((w > 0).sum())
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError('Calibration sources must be nonempty and unique')
    if not designs:
        return None, None, {'feasible': False, 'status': 'no_stable_feasible_solution',
                            'reason': 'No whole valid near-startup calibration cells',
                            'constraint_receipts': receipts, 'source_ids': source_ids}
    a, c, report = solve_constrained_delta(stats, torch.cat(designs), torch.cat(deltas), torch.cat(targets))
    report.update(source_ids=source_ids, observations=len(source_ids), constraint_receipts=receipts,
                  weighted_sample_count=weighted_samples, valid_feature_rows=feature_rows,
                  all_valid_rows_included=True, samples_per_cell=8, chunk_rows=chunk_rows,
                  weighted_error_before=float(energy['raw']) / (weighted_samples * module.out_channels))
    if a is None: return None, None, report
    centered = energy['centered'] - 2*(a*stats['centered_xd'].T).sum() + (a@stats['centered_xx']*a).sum()
    mean_error = stats['mean_d'] - a@stats['mean_x'] - c
    error = float(centered + stats['n']*mean_error.square().sum())
    roundoff = 128*torch.finfo(torch.float64).eps*max(float(energy['raw']), 1.)
    if not torch.isfinite(torch.tensor(error)) or error < -roundoff:
        raise RuntimeError('Invalid constrained reconstruction energy')
    report.update(weighted_error_after=max(0., error)/(weighted_samples*module.out_channels),
                  raw_after_error_sum=error, error_after_scope='FP64 constrained fit prediction before native FP32 writeback')
    return ordinary.native_weight_from_affine(a, module.in_channels), c, report


@torch.no_grad()
def observe_startup(model, teacher, crop, site):
    z, cached, valid, _ = base.batch([crop])
    with audit.capture_sites(teacher.model.decoder, [site]) as teacher_capture:
        trace = base.teacher_forward(teacher, z)
    common.warm_student(model, trace['group_input'])
    with audit.capture_sites(model.decoder, [site]) as student_capture:
        model.group_from_input(trace['group_input'])
    t, s = teacher_capture[site.name], student_capture[site.name]
    selected, receipt = startup_cells(cached, valid, crop, s['output'].shape[-1])
    native_selected, native_receipt = startup_cells(trace['waveform'], valid, crop, s['output'].shape[-1])
    if not torch.equal(selected, native_selected):
        raise RuntimeError('Native teacher startup membership differs from the authenticated cached selection')
    cache = control.compare_tensors(trace['waveform'], cached, valid)
    if not cache['allclose_existing']: raise RuntimeError('Calibration teacher and authenticated cached target differ')
    receipt['teacher_cache'] = cache
    receipt['native_mask_matches_cached_selection'] = True
    receipt['native_teacher_windows'] = native_receipt['windows']
    return {'source_id': crop['source_id'], 'input': s['input'], 'target': t['output'],
            'current_output': s['output'], 'weights': common.cell_weights(valid, s['output'].shape[-1]),
            'constraint_mask': selected, 'constraint_receipt': receipt}


@torch.no_grad()
def main():
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0', 'checkpoint', 'manifest', 'assets', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a fresh isolated constrained initialization directory')
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.step0.parent, args.checkpoint.parent)):
        raise ValueError('Output must not overlap preserved training runs')
    zero, _, authenticated = audit.checkpoint_inputs(args)
    if base.sha(args.manifest) not in authenticated['launch']['protected'].values():
        raise ValueError('Use the original authenticated fit/development manifest')
    process = control.gpu_idle_snapshot()
    manifest, pools, _ = base.load_data(args.manifest)
    calibration, development = pools['calibration'], pools['development']
    if len(calibration) != 72 or len(development) != 96: raise ValueError('Expected original72 fit and96 heldout sources')
    split = common.validate_fit_sources(calibration, development)
    paths = [args.step0, args.checkpoint, args.manifest, args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth',
             Path(manifest['cache_path']), Path(manifest['overlay_receipt_path']), args.step0.parent/'development-step0.json',
             Path(__file__), Path(ordinary.__file__), Path(audit.__file__), Path(common.__file__),
             Path(control.__file__), Path(group.__file__), Path(progressive.__file__), Path(base.__file__)]
    protected = {**authenticated['protected'], **{str(p.resolve()): base.sha(p) for p in paths}}
    base.policy()
    if str(torch.__version__) != authenticated['launch']['torch'] or torch.backends.cudnn.version() != authenticated['launch']['cudnn']:
        raise ValueError('Use the preserved FP32 runtime and backend')
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json', {'version': VERSION, 'protected': protected, **split,
        'selection': zero['selection'], 'original_step0_sha256': audit.STEP0_SHA,
        'preserved_step5000_sha256': audit.STEP5000_SHA, 'ridge_factor': RIDGE,
        'rank_rtol': 'max(whitened_constraint_design.shape)*float32_epsilon', 'rank_atol': 0,
        'interface_atol': ATOL, 'interface_rtol': RTOL, 'constraint_source': 'Original teacher near-startup calibration windows; full8sample cells only',
        'process_snapshot': process, 'neural_training_updates': 0, 'optimizer_created': False,
        'automatic_promotion': False, 'hidden_equality_required_for_waveform': False})
    teacher = initial = candidate = None; hashes = {}; results = {}; report = {}; failure = None
    status = 'initializing'; started = time.monotonic()
    try:
        teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth', device='cuda')
        initial = progressive.initialize_from_teacher(teacher.model.decoder, zero['selection'])
        initial.load_group_state_dict(zero['group']); initial.eval()
        pristine = audit.validate_pristine(teacher.model.decoder, initial, zero['selection'])
        hashes = {'teacher': control.state_hash(teacher.model), 'initial': control.state_hash(initial.decoder)}
        candidate = progressive.initialize_from_teacher(teacher.model.decoder, zero['selection'])
        candidate.load_group_state_dict(zero['group']); candidate.eval()
        if control.state_hash(candidate.decoder) != hashes['initial']: raise RuntimeError('Candidate initial state differs')
        before = audit._versions(candidate.decoder); site = audit.four_sites(teacher.model.decoder, zero['selection'])[-1]
        objective = base.objective(); metadata = {r['source_id']: r for r in manifest['splits']['development']['rows']}
        audit.warm_models(teacher, [initial], development)
        baseline = ordinary.evaluate_candidate(initial, teacher, development, objective, metadata, args.out, 'baseline-step0')
        results['baseline'] = baseline
        parity = numeric_reference_check(baseline, json.loads((args.step0.parent/'development-step0.json').read_text()))
        base.write_json(args.out/'baseline-parity.json', parity)
        if not parity['passed']: raise RuntimeError('Original step0 baseline did not reproduce')
        def observations():
            for i, crop in enumerate(calibration, 1):
                row = observe_startup(candidate, teacher, crop, site)
                if i % 12 == 0: base.event('startup_constraint_calibration', sources=i, expected=72)
                yield row
        module = candidate.decoder.get_submodule(site.path)
        dw, db, report = fit_constrained_delta(module, observations())
        base.write_json(args.out/'constraint-fit.json', report)
        if not report['feasible']:
            status = 'no_stable_feasible_solution'
            if control.state_hash(candidate.decoder) != hashes['initial']: raise RuntimeError('Infeasible solve changed candidate')
        else:
            desired = (group.effective_weight(module).detach().double()+dw).to(group.effective_weight(module))
            report['effective_writeback'] = ordinary.apply_delta(module, dw, db)
            if not all(x['allclose_existing'] for x in report['effective_writeback'].values()):
                raise RuntimeError('Native coefficient writeback differs from intended FP32 operator')
            native_checks = []; equality_checks = []
            # Only calibration inputs are revisited to measure installed FP32 equalities.
            for crop in calibration:
                row = observe_startup(candidate, teacher, crop, site)
                output = row['current_output']; direct = common.linear_response(module, row['input'], desired, True)
                native_checks.append(control.compare_tensors(output, direct))
                mask = row['constraint_mask'].expand_as(output)
                if mask.any(): equality_checks.append(equality_metrics(output[mask], row['target'][mask]))
            report.update(native_fold_allclose=all(x['allclose_existing'] for x in native_checks),
                          native_fold_checks=native_checks, equality_native_fp32=equality_checks,
                          native_fp32_constraints_passed=bool(equality_checks) and all(x['passed'] for x in equality_checks))
            base.write_json(args.out/'constraint-fit.json', report)
            ordinary._assert_only_sites_changed(before, candidate, [site])
            if not report['native_fold_allclose']: raise RuntimeError('Native operation differs from intended coefficients')
            # Waveform evidence remains useful even if rounding breaks a hidden equality.
            status = 'evaluated' if report['native_fp32_constraints_passed'] else 'evaluated_native_equality_not_met'
            candidate_hash = control.state_hash(candidate.decoder)
            audit.warm_models(teacher, [candidate], development)
            results['constrained_A'] = ordinary.evaluate_candidate(candidate, teacher, development, objective, metadata, args.out, 'constrained-A')
            if control.state_hash(candidate.decoder) != candidate_hash: raise RuntimeError('Evaluation changed fitted candidate')
            artifact = args.out/'constrained-A-native-operator.pt'
            torch.save({'format': VERSION, 'original_step0_sha256': audit.STEP0_SHA,
                        'selection': zero['selection'], 'fit_source_ids': split['fit_source_ids'],
                        'operators': {site.path: {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}},
                        'automatic_promotion': False}, artifact)
            base.write_json(args.out/'candidate-receipt.json', {'operator_sha256': base.sha(artifact),
                'candidate_state_sha256': candidate_hash, 'original_step0_pristine': audit._public_pristine(pristine),
                'native_fp32_constraints_passed': report['native_fp32_constraints_passed'],
                'changed_native_paths': [site.path], 'automatic_promotion': False})
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        files = all(base.sha(p) == checksum for p, checksum in protected.items())
        states = {'teacher': teacher is None or not hashes or control.state_hash(teacher.model) == hashes['teacher'],
                  'original_step0': initial is None or not hashes or control.state_hash(initial.decoder) == hashes['initial']}
        complete = failure is None and status in ('evaluated', 'evaluated_native_equality_not_met', 'no_stable_feasible_solution') and files and all(states.values())
        base.write_json(args.out/'completed.json', {'version': VERSION, 'complete': complete, 'status': status,
            'failure': failure, 'files_preserved': files, 'states_preserved': states,
            'stable_constraint_solution': report.get('feasible', False),
            'native_fp32_constraints_passed': report.get('native_fp32_constraints_passed', False),
            'calibration_sources': 72, 'development_sources': 96, 'neural_training_updates': 0,
            'automatic_promotion': False, 'elapsed_seconds': time.monotonic()-started,
            'results': {name: {key: result[key] for key in ('aggregate', 'quiet_regions', 'overview_window_metrics', 'recovery_window_metrics')}
                        for name,result in results.items()}})
        if not files or not all(states.values()): raise RuntimeError('Constrained calibration changed preserved data or models')


if __name__ == '__main__': main()
