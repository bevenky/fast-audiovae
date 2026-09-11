"""GRAIL-like zero-intercept hidden reconstruction in existing native operators.

Fresh original teacher, original pivoted384/256 support, four sequential maps.
The map is never deployed: it folds into pointwise weights or all10 native taps.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

import reconstruction_aware_init as ordinary

audit, common = ordinary.audit, ordinary.common
group, base, control, progressive = ordinary.group, ordinary.base, ordinary.control, ordinary.progressive
VERSION = 'audiovae2_grail_hidden_initialization_v1'
VARIANT = 'grail_shared_hidden'
RIDGE = 1e-6
CHUNK_ROWS = 1024
PATHS = ('model.3.block.2.block.3', 'model.3.block.3.block.3',
         'model.3.block.4.block.3', 'model.4.block.1')
ORDINARY_SHA = 'de5362361b1dfd387c9506fefa93e13435d772f784fcc344390791265efa1913'


def hidden_usage_weights(output_weights, *, native=False):
    """Count only scored output uses of each actual hidden input cell.

    The previous cell of a scored upsample frame is included even if that cell
    itself is context. There is no extra cell before actual tensor startup.
    """
    if (output_weights.ndim != 3 or output_weights.shape[1] != 1
            or not torch.isfinite(output_weights).all() or (output_weights < 0).any()
            or (output_weights > (8 if native else 40)).any()):
        raise ValueError('Invalid valid-waveform output-cell counts')
    if not native:
        return output_weights.clone()
    if output_weights.shape[-1] % 5:
        raise ValueError('Native stage3 output must contain complete five-phase frames')
    frame = output_weights.reshape(output_weights.shape[0], 1, -1, 5).sum(-1)
    result = frame.clone()
    result[..., :-1] += frame[..., 1:]
    return result


@torch.no_grad()
def accumulate_hidden(stats, x, h, weights):
    if (x.ndim != 3 or h.ndim != 3 or x.shape[0] != h.shape[0] or x.shape[-1] != h.shape[-1]
            or weights.shape != (x.shape[0], 1, x.shape[-1])
            or not x.is_floating_point() or not h.is_floating_point()
            or x.device != h.device or x.device != weights.device
            or not torch.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError('Invalid paired student/teacher post-Snake hidden inputs')
    a = x.detach().movedim(1, -1).reshape(-1, x.shape[1]).double()
    b = h.detach().movedim(1, -1).reshape(-1, h.shape[1]).double()
    q = weights.reshape(-1).double(); selected = q > 0
    a, b, q = a[selected], b[selected], q[selected]
    if not len(q):
        return stats
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite hidden values on a used cell')
    if stats is None:
        stats = {'xx': a.new_zeros((x.shape[1], x.shape[1])),
                 'hx': a.new_zeros((h.shape[1], x.shape[1])),
                 'hh_energy': a.new_zeros(()), 'weight': a.new_zeros(()), 'rows': 0}
    if stats['xx'].shape != (x.shape[1], x.shape[1]) or stats['hx'].shape != (h.shape[1], x.shape[1]):
        raise ValueError('Hidden-map dimensions changed within a fit')
    stats['xx'].add_(a.T @ (a * q[:, None]))
    stats['hx'].add_(b.T @ (a * q[:, None]))
    stats['hh_energy'].add_((b.square() * q[:, None]).sum())
    stats['weight'].add_(q.sum()); stats['rows'] += len(q)
    return stats


def solve_hidden_map(stats, *, ridge=RIDGE):
    if stats is None or ridge != RIDGE:
        raise ValueError('A nonempty hidden fit and the fixed ridge are required')
    xx, hx = stats['xx'], stats['hx']
    if (xx.dtype != torch.float64 or hx.dtype != torch.float64 or xx.ndim != 2
            or xx.shape[0] != xx.shape[1] or hx.shape[1] != xx.shape[0]
            or not torch.isfinite(xx).all() or not torch.isfinite(hx).all()
            or not torch.isfinite(stats['hh_energy']) or float(stats['weight']) <= 0):
        raise ValueError('Invalid FP64 uncentered hidden statistics')
    xx = (xx + xx.T) / 2
    scale = float(xx.diag().mean())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Hidden Gram needs positive finite trace; no new ridge fallback')
    lam = ridge * scale
    regularized = xx + lam * torch.eye(xx.shape[0], device=xx.device, dtype=xx.dtype)
    factor, info = torch.linalg.cholesky_ex(regularized)
    if int(info) != 0:
        raise RuntimeError('Regularized hidden Gram is not positive definite')
    mapping = torch.cholesky_solve(hx.T, factor).T
    if not torch.isfinite(mapping).all():
        raise RuntimeError('Nonfinite hidden map')
    error = float(stats['hh_energy'] - 2 * (mapping * hx).sum() + (mapping @ xx * mapping).sum())
    terms = float(stats['hh_energy'].abs() + 2 * (mapping * hx).abs().sum()
                  + (mapping @ xx * mapping).abs().sum())
    bound = 4096 * torch.finfo(torch.float64).eps * terms
    if error < -bound:
        raise RuntimeError('Materially negative hidden reconstruction energy')
    normal_residual = float((mapping @ regularized - hx).norm())
    report = {'ridge_factor': ridge, 'ridge_lambda': lam, 'uncentered': True,
              'intercept': False, 'student_channels': xx.shape[0], 'teacher_channels': hx.shape[0],
              'valid_hidden_rows': stats['rows'], 'weighted_hidden_uses': float(stats['weight']),
              'weighted_hidden_mse_after': max(error, 0.) / (float(stats['weight']) * hx.shape[0]),
              'raw_hidden_error_after': error, 'negative_roundoff_bound': bound,
              'normal_equation_relative_residual': normal_residual / max(float(hx.norm()), 1e-300),
              'regularized_condition_number': float(torch.linalg.cond(regularized)),
              'map_norm': float(mapping.norm()), 'cholesky_positive': True,
              'scope': 'FP64 post-Snake feature fit before native FP32 folding; not waveform quality'}
    return mapping, report


@torch.no_grad()
def folded_coefficients(teacher_module, mapping, kept_outputs):
    native = ordinary._validate_module(teacher_module)
    weight = group.effective_weight(teacher_module).detach().double()
    ids = group._indices(kept_outputs, teacher_module.out_channels, 'consumer outputs').to(weight.device)
    if (mapping.ndim != 2 or mapping.shape[0] != teacher_module.in_channels
            or not torch.isfinite(mapping).all() or mapping.device != weight.device):
        raise ValueError('Hidden map must reconstruct all teacher consumer inputs')
    mapping = mapping.double()
    if native:
        desired = torch.einsum('cot,ck->kot', weight[:, ids, :], mapping)
    else:
        desired = (weight[ids, :, 0] @ mapping).unsqueeze(-1)
    # No intercept is fitted or folded. This is the original shared native bias.
    return desired, teacher_module.bias.detach()[ids].clone()


@torch.no_grad()
def install_hidden_map(student_module, teacher_module, mapping, kept_outputs):
    if ordinary._validate_module(student_module) != ordinary._validate_module(teacher_module):
        raise ValueError('Student and teacher consumer types differ')
    desired, bias = folded_coefficients(teacher_module, mapping, kept_outputs)
    current = group.effective_weight(student_module)
    if desired.shape != current.shape or bias.shape != student_module.bias.shape:
        raise ValueError('Folded hidden map changed native student geometry')
    desired = desired.to(current); bias = bias.to(student_module.bias)
    group.assign_effective_weight(student_module, desired, bias)
    result = {'effective_weight': control.compare_tensors(group.effective_weight(student_module), desired),
              'bias': control.compare_tensors(student_module.bias, bias),
              'bias_bitwise_equal': torch.equal(student_module.bias, bias),
              'shared_all_native_taps': isinstance(student_module, torch.nn.ConvTranspose1d),
              'extra_inference_modules': 0}
    if not result['effective_weight']['allclose_existing'] or not result['bias_bitwise_equal']:
        raise RuntimeError('Native hidden-map fold differs from intended FP32 coefficients/bias')
    return result


@torch.no_grad()
def observe_hidden(model, teacher, crop, site):
    z, _, valid, _ = base.batch([crop])
    with audit.capture_sites(teacher.model.decoder, [site]) as tc:
        trace = base.teacher_forward(teacher, z)
    common.warm_student(model, trace['group_input'])
    with audit.capture_sites(model.decoder, [site]) as sc:
        model.group_from_input(trace['group_input'])
    t, s = tc[site.name], sc[site.name]
    output_weights = common.cell_weights(valid, s['output'].shape[-1])
    return {'source_id': crop['source_id'], 'input': s['input'], 'teacher_input': t['input'],
            'weights': hidden_usage_weights(output_weights, native=site.stride > 1)}


@torch.no_grad()
def sequential_hidden_fit(model, teacher, calibration, sites, observe_fn=observe_hidden, on_report=None):
    reports, maps = [], {}
    for site in sites:
        before = control.state_hash(model.decoder); stats = None; ids = []; last = None
        for crop in calibration:
            row = observe_fn(model, teacher, crop, site); ids.append(row['source_id'])
            x, h, q = (row[key] for key in ('input', 'teacher_input', 'weights'))
            for start in range(0, x.shape[-1], CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, x.shape[-1])
                stats = accumulate_hidden(stats, x[..., start:stop], h[..., start:stop], q[..., start:stop])
            last = row
        if len(ids) != len(set(ids)) or len(ids) != len(calibration):
            raise ValueError('Repeated or missing calibration source within hidden fit')
        mapping, report = solve_hidden_map(stats)
        sm = model.decoder.get_submodule(site.path); tm = teacher.model.decoder.get_submodule(site.path)
        desired, bias = folded_coefficients(tm, mapping, site.kept_outputs)
        report['native_writeback'] = install_hidden_map(sm, tm, mapping, site.kept_outputs)
        x = last['input']; actual = sm(x)
        expected = common.linear_response(sm, x, desired.to(x), False) + bias.to(x)[None, :, None]
        report['last_source_native_fold_parity'] = control.compare_tensors(actual, expected)
        if not report['last_source_native_fold_parity']['allclose_existing']:
            raise RuntimeError('Native consumer does not reproduce intended FP32 hidden fold')
        report.update(site=site.name, path=site.path, observations=len(ids), source_ids=ids,
                      input_candidate_state_sha256=before, output_candidate_state_sha256=control.state_hash(model.decoder),
                      inputs_recomputed_after_previous_fit=True, target='complete original post-Snake teacher consumer input',
                      residual_skip='actual student skip remains unchanged; no explicit skip-error target',
                      output_coordinates=list(site.kept_outputs))
        maps[site.path] = mapping.cpu(); reports.append(report)
        if on_report is not None: on_report(reports)
    return reports, maps


def export_artifact(model, teacher_decoder, calibration_ids, *, base_selected_state_sha256, maps_sha256):
    if len(calibration_ids) != 72 or len(set(calibration_ids)) != 72:
        raise ValueError('Export requires the fixed72 distinct calibration sources')
    sites = audit.four_sites(teacher_decoder, model.selections)
    if tuple(s.path for s in sites) != PATHS:
        raise ValueError('Only the four declared native consumers may be exported')
    states = {s.path: {k: v.detach().cpu().clone() for k, v in model.decoder.get_submodule(s.path).state_dict().items()}
              for s in sites}
    if any(not torch.isfinite(t).all() for values in states.values() for t in values.values()):
        raise ValueError('Nonfinite native artifact')
    return {'format': VERSION, 'variant': VARIANT, 'original_step0_sha256': audit.STEP0_SHA,
            'teacher_checkpoint_sha256': base.CHECKPOINT_SHA256, 'teacher_source_sha256': base.SOURCE_SHA256,
            'teacher_state_sha256': control.state_hash(teacher_decoder), 'selection': model.selections,
            'fit_source_ids': list(calibration_ids), 'ridge': RIDGE, 'operators': states,
            'base_selected_state_sha256': base_selected_state_sha256,
            'candidate_state_sha256': control.state_hash(model.decoder), 'hidden_maps_sha256': maps_sha256,
            'hidden_maps_exported': False, 'automatic_promotion': False}


def _cache_gate(records):
    if not records or not all(r['allclose_original_tolerance'] for r in records):
        raise RuntimeError('Original teacher differs from authenticated scored targets')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0', 'checkpoint', 'manifest', 'assets', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a new isolated GRAIL initializer directory')
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.step0.parent, args.checkpoint.parent)):
        raise ValueError('Output must not overlap preserved training runs')
    if base.sha(Path(ordinary.__file__)) != ORDINARY_SHA:
        raise ValueError('The pinned native fitting/helper source changed')
    zero, _, authenticated = audit.checkpoint_inputs(args)
    if base.sha(args.manifest) not in authenticated['launch']['protected'].values():
        raise ValueError('Use the original authenticated calibration/development manifest')
    process = control.gpu_idle_snapshot()
    manifest, pools, _ = base.load_data(args.manifest)
    calibration, development = pools['calibration'], pools['development']
    if len(calibration) != 72 or len(development) != 96:
        raise ValueError('Expected original72 calibration and96 development sources')
    split = common.validate_fit_sources(calibration, development)
    paths = [args.step0, args.checkpoint, args.manifest, args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth',
             Path(manifest['cache_path']), Path(manifest['overlay_receipt_path']), Path(__file__), Path(ordinary.__file__),
             Path(audit.__file__), Path(common.__file__), Path(control.__file__), Path(group.__file__),
             Path(progressive.__file__), Path(base.__file__), Path(common.replay.__file__)]
    protected = {**authenticated['protected'], **{str(p.resolve()): base.sha(p) for p in paths}}
    base.policy()
    if (str(torch.__version__) != authenticated['launch']['torch']
            or torch.backends.cudnn.version() != authenticated['launch']['cudnn']):
        raise ValueError('Use the preserved singleton FP32 runtime and backend')
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json', {'version': VERSION, 'variant': VARIANT, 'protected': protected, **split,
        'original_step0_sha256': audit.STEP0_SHA, 'preserved_step5000_sha256': audit.STEP5000_SHA,
        'teacher_checkpoint_sha256': base.CHECKPOINT_SHA256, 'teacher_source_sha256': base.SOURCE_SHA256,
        'selection': zero['selection'], 'torch': str(torch.__version__), 'cudnn': torch.backends.cudnn.version(),
        'python': sys.version, 'argv': sys.argv, 'backend': common.replay.backend_state(), 'process_snapshot': process,
        'source_pass_budget': {'sequential_hidden_fits': 4*72, 'development': 96},
        'warmups': 'Existing per-shape teacher/student warmups separate from scored passes',
        'ridge_factor': RIDGE, 'uncentered_hidden_map': True, 'intercept': False,
        'shared_map_across_native_taps': True, 'bias_policy': 'Original selected teacher bias, no offsets',
        'native_paths': list(PATHS), 'widths': [384, 256, 128], 'neural_training_updates': 0,
        'optimizer_created': False, 'extra_inference_modules': 0, 'automatic_promotion': False,
        'candidate_start': 'Fresh original teacher factory slice; original step0 tensors checked but never installed'})
    teacher = model = None; teacher_hash = None; before = None; sites = None
    results = {}; receipt = None; cache_records = []; failure = None; status = 'initializing'
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    try:
        teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth', device='cuda')
        teacher_hash = control.state_hash(teacher.model)
        if tuple(teacher.model.decoder.model[i].block[1].out_channels for i in (3, 4, 5)) != (512, 256, 128):
            raise ValueError('Expected original teacher512/256/128 geometry')
        model = progressive.initialize_from_teacher(teacher.model.decoder, zero['selection']).eval()
        pristine = audit.validate_pristine(teacher.model.decoder, model, zero['selection'])
        current = model.group_state_dict()
        if (set(current) != set(zero['group'])
                or any(not torch.equal(current[k].detach().cpu(), zero['group'][k].detach().cpu()) for k in current)):
            raise RuntimeError('Fresh teacher factory group differs from matched original pristine step0')
        if len(model.trainable_group_parameters()) != 90:
            raise ValueError('Native90-parameter group scope differs')
        base_hash = control.state_hash(model.decoder); before = audit._versions(model.decoder)
        sites = audit.four_sites(teacher.model.decoder, zero['selection'])
        if tuple(s.path for s in sites) != PATHS:
            raise ValueError('Unexpected native consumer sequence')
        base.write_json(args.out/'calibration-coverage.json', ordinary.calibration_coverage(calibration))
        def on_report(reports):
            base.write_json(args.out/'grail-fits.json', reports)
            row = reports[-1]
            base.event('grail_hidden_consumer_fitted', site=row['site'], sources=row['observations'],
                       hidden_mse=row['weighted_hidden_mse_after'])
        with common.replay.observe_teacher_cache(calibration*4) as checks:
            reports, maps = sequential_hidden_fit(model, teacher, calibration, sites, on_report=on_report)
        cache_records.extend(checks); _cache_gate(checks)
        ordinary._assert_only_sites_changed(before, model, sites)
        maps_path = args.out/'grail-hidden-maps.pt'
        torch.save({'format': VERSION, 'maps': maps, 'fit_source_ids': split['fit_source_ids'],
                    'deployed': False, 'teacher_state_sha256': control.state_hash(teacher.model.decoder)}, maps_path)
        payload = export_artifact(model, teacher.model.decoder, split['fit_source_ids'],
                                  base_selected_state_sha256=base_hash, maps_sha256=base.sha(maps_path))
        artifact_path = args.out/'grail-native-operators.pt'; torch.save(payload, artifact_path)
        candidate_hash = payload['candidate_state_sha256']
        objective = base.objective(); metadata = {r['source_id']: r for r in manifest['splits']['development']['rows']}
        audit.warm_models(teacher, [model], development)
        with common.replay.observe_teacher_cache(development) as checks:
            results['grail'] = ordinary.evaluate_candidate(model, teacher, development, objective, metadata, args.out, 'grail')
        cache_records.extend(checks); _cache_gate(checks)
        if control.state_hash(model.decoder) != candidate_hash:
            raise RuntimeError('Evaluation changed the candidate')
        ordinary._assert_only_sites_changed(before, model, sites)
        receipt = {'operators_sha256': base.sha(artifact_path), 'base_selected_state_sha256': base_hash,
            'candidate_state_sha256': candidate_hash, 'teacher_state_sha256': control.state_hash(teacher.model.decoder),
            'original_selected_pristine': audit._public_pristine(pristine), 'selection': model.selections,
            'fresh_factory_matches_original_step0': True, 'original_checkpoint_weights_installed': False,
            'hidden_maps_sha256': base.sha(maps_path), 'hidden_maps_exported': False,
            'changed_native_paths': list(PATHS), 'full_group_widths': [384, 256, 128],
            'all9_residual_units_preserved': True, 'unchanged_frozen_and_unselected_tensors': True,
            'native_writeback_passed': True, 'original_bias_bitwise_preserved': True,
            'neural_training_updates': 0, 'extra_inference_modules': 0, 'automatic_promotion': False}
        base.write_json(args.out/'grail-receipt.json', receipt)
        status = 'evaluated'; base.event('grail_hidden_evaluated', development_sources=96)
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        files = all(base.sha(path) == digest for path, digest in protected.items())
        states = {'teacher': teacher is None or teacher_hash is None or control.state_hash(teacher.model) == teacher_hash,
                  'frozen_and_unselected': True}
        if model is not None and before is not None:
            try: ordinary._assert_only_sites_changed(before, model, sites)
            except RuntimeError as exc:
                states['frozen_and_unselected'] = False
                if failure is None: failure = repr(exc)
        control._write_rows(args.out/'teacher-cache-checks.jsonl.gz', cache_records)
        complete = (status == 'evaluated' and failure is None and files and all(states.values())
                    and len(cache_records) == 384 and set(results) == {'grail'} and receipt is not None)
        base.write_json(args.out/'completed.json', {'version': VERSION, 'variant': VARIANT, 'complete': complete,
            'status': status, 'failure': failure, 'files_preserved': files, 'states_preserved': states,
            'calibration_sources': 72, 'development_sources': 96, 'neural_training_updates': 0,
            'optimizer_created': False, 'automatic_promotion': False, 'candidate_receipt': receipt,
            'teacher_cache_comparisons': len(cache_records),
            'teacher_cache_all_pass': bool(cache_records) and all(r['allclose_original_tolerance'] for r in cache_records),
            'teacher_cache_nonexact': sum(not r['bitwise_equal'] for r in cache_records),
            'elapsed_seconds': time.monotonic()-started, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'results': {name: {key: r[key] for key in ('aggregate', 'quiet_regions', 'overview_window_metrics', 'recovery_window_metrics')}
                        for name, r in results.items()}})
        if not files or not all(states.values()):
            raise RuntimeError('Initializer altered protected original state/files')


if __name__ == '__main__':
    main()
