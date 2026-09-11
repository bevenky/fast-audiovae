"""Refit B's native upsampler on B inputs with unchanged startup constraints.

All three fitted B residual mixers remain exact. This produces a separate
initialization artifact, without optimization, extra inference layers, or any
transplant from the earlier A candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import startup_constrained_init as constrained
import reconstruction_b_recovery as b_recovery

ordinary = constrained.ordinary
audit, common = constrained.audit, constrained.common
group, base, control, progressive = constrained.group, constrained.base, constrained.control, constrained.progressive
VERSION = 'audiovae2_combined_startup_initialization_v1'
VARIANT = 'combined_B_startup'
PATHS = b_recovery.OPERATOR_PATHS


@torch.no_grad()
def initialize_b(teacher_decoder, zero, artifact, receipt, calibration_ids, *, artifact_sha256):
    """Independently reconstruct the sealed B state from original teacher step0."""
    if receipt['original_step0_pristine']['teacher_state_sha256'] != control.state_hash(teacher_decoder):
        raise ValueError('B receipt belongs to a different original teacher')
    model = progressive.initialize_from_teacher(teacher_decoder, zero['selection'])
    model.load_group_state_dict(zero['group']); model.eval()
    installed = b_recovery.install_b_operators(model, artifact, receipt,
        artifact_sha256=artifact_sha256, step0_sha256=audit.STEP0_SHA, calibration_ids=calibration_ids)
    if installed['candidate_state_sha256'] != control.state_hash(model.decoder):
        raise RuntimeError('Complete sealed B state did not reproduce')
    return model, installed


def operator_hashes(model, paths=PATHS[:3]):
    return {path: control.state_hash(model.decoder.get_submodule(path)) for path in paths}


@torch.no_grad()
def refit_b_upsampler(model, teacher, calibration, *, observe_fn=constrained.observe_startup):
    """Only the existing upsampler changes; every observation uses actual B inputs."""
    sites = audit.four_sites(teacher.model.decoder, model.selections)
    site = sites[-1]; module = model.decoder.get_submodule(site.path)
    before = audit._versions(model.decoder)
    b_hash = control.state_hash(model.decoder); stage2 = operator_hashes(model)
    dw, db, report = constrained.fit_constrained_delta(module,
        (observe_fn(model, teacher, crop, site) for crop in calibration))
    report.update(base_b_state_sha256=b_hash, stage2_b_operator_hashes=stage2,
                  observations_from_actual_b_chain=True, transplanted_a_coefficients=False)
    if dw is None:
        if control.state_hash(model.decoder) != b_hash:
            raise RuntimeError('Infeasible solve changed the sealed B candidate')
        report['stage2_b_operators_preserved'] = True
        return report
    desired = (group.effective_weight(module).detach().double() + dw).to(group.effective_weight(module))
    report['effective_writeback'] = ordinary.apply_delta(module, dw, db)
    if not all(row['allclose_existing'] for row in report['effective_writeback'].values()):
        raise RuntimeError('Combined native coefficient writeback differs')
    native_checks = []; equalities = []; input_receipts = []
    for crop in calibration:
        row = observe_fn(model, teacher, crop, site)
        output = row['current_output']
        native = common.linear_response(module, row['input'], desired, True)
        native_checks.append(control.compare_tensors(output, native))
        mask = row['constraint_mask'].expand_as(output)
        if mask.any(): equalities.append(constrained.equality_metrics(output[mask], row['target'][mask]))
        input_receipts.append(row.get('constraint_receipt', {}))
    ordinary._assert_only_sites_changed(before, model, [site])
    if operator_hashes(model) != stage2: raise RuntimeError('Combined refit changed a stage2 B operator')
    report.update(native_fold_allclose=all(row['allclose_existing'] for row in native_checks),
                  native_fold_checks=native_checks, equality_native_fp32=equalities,
                  native_fp32_constraints_passed=bool(equalities) and all(row['passed'] for row in equalities),
                  stage2_b_operators_preserved=True, native_constraint_receipts=input_receipts,
                  candidate_state_sha256=control.state_hash(model.decoder))
    if not report['native_fold_allclose']: raise RuntimeError('Combined native forward differs from intended coefficients')
    return report


def export_artifact(model, calibration_ids, b_sha256):
    if len(calibration_ids) != 72 or len(set(calibration_ids)) != 72:
        raise ValueError('Export requires the original72 distinct calibration sources')
    operators = {path: {name: value.detach().cpu().clone() for name, value in model.decoder.get_submodule(path).state_dict().items()}
                 for path in PATHS}
    if not all(bool(torch.isfinite(value).all()) for state in operators.values() for value in state.values()):
        raise ValueError('Nonfinite native operator cannot be exported')
    return {'format': VERSION, 'variant': VARIANT, 'base_step0_sha256': audit.STEP0_SHA,
            'b_operators_sha256': b_sha256, 'selection': model.selections,
            'fit_source_ids': list(calibration_ids), 'ridge': constrained.RIDGE,
            'rank_policy': 'max(whitened_constraint_design.shape)*float32_epsilon, absolute cutoff0',
            'interface_atol': constrained.ATOL, 'interface_rtol': constrained.RTOL,
            'operators': operators, 'automatic_promotion': False}


def authenticate_b(directory, calibration_ids, development_ids, selection):
    artifact_path = directory/'variant-B-native-operators.pt'
    receipt = json.loads((directory/'variant-B-receipt.json').read_text())
    completion = json.loads((directory/'completed.json').read_text())
    launch = json.loads((directory/'launch.json').read_text())
    if (base.sha(artifact_path) != b_recovery.B_SHA or receipt.get('operators_sha256') != b_recovery.B_SHA
            or receipt.get('candidate_state_sha256') != b_recovery.B_STATE_SHA
            or receipt.get('full_group_widths') != [384, 256, 128]
            or completion.get('complete') is not True or completion.get('failure') is not None
            or completion.get('files_preserved') is not True or not completion.get('states_preserved')
            or not all(completion['states_preserved'].values())
            or completion.get('calibration_sources') != 72 or completion.get('development_sources') != 96
            or completion.get('neural_training_updates') != 0
            or launch.get('fit_source_ids') != list(calibration_ids)
            or launch.get('development_source_ids') != list(development_ids)
            or launch.get('selection') != selection or launch.get('original_step0_sha256') != audit.STEP0_SHA):
        raise ValueError('Sealed B initialization provenance or original source partition differs')
    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
    reference = json.loads((directory/'variant-B-development.json').read_text())
    if completion['results']['B']['aggregate'] != reference['aggregate']:
        raise ValueError('B baseline report differs from its completion')
    paths = [artifact_path, directory/'variant-B-receipt.json', directory/'completed.json',
             directory/'launch.json', directory/'variant-B-development.json']
    protected = {**launch['protected'], **{str(path.resolve()): base.sha(path) for path in paths}}
    if not all(base.sha(path) == digest for path, digest in protected.items()):
        raise ValueError('Protected B initialization assets changed')
    return artifact, receipt, reference, protected


@torch.no_grad()
def main():
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0', 'checkpoint', 'manifest', 'assets', 'b-dir', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a fresh isolated combined initializer directory')
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.step0.parent, args.checkpoint.parent, args.b_dir)):
        raise ValueError('Output must not overlap preserved source runs')
    zero, _, original_auth = audit.checkpoint_inputs(args)
    if base.sha(args.manifest) not in original_auth['launch']['protected'].values():
        raise ValueError('Use the original authenticated calibration/development manifest')
    process = control.gpu_idle_snapshot()
    manifest, pools, _ = base.load_data(args.manifest)
    calibration, development = pools['calibration'], pools['development']
    if len(calibration) != 72 or len(development) != 96: raise ValueError('Expected original72 fit and96 development sources')
    split = common.validate_fit_sources(calibration, development)
    artifact, b_receipt, reference, b_protected = authenticate_b(args.b_dir, split['fit_source_ids'], split['development_source_ids'], zero['selection'])
    paths = [args.step0, args.checkpoint, args.manifest, args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth',
             Path(manifest['cache_path']), Path(manifest['overlay_receipt_path']), Path(__file__), Path(constrained.__file__),
             Path(ordinary.__file__), Path(b_recovery.__file__), Path(audit.__file__), Path(common.__file__),
             Path(control.__file__), Path(group.__file__), Path(base.__file__), Path(progressive.__file__)]
    protected = {**original_auth['protected'], **b_protected, **{str(p.resolve()): base.sha(p) for p in paths}}
    base.policy()
    if str(torch.__version__) != original_auth['launch']['torch'] or torch.backends.cudnn.version() != original_auth['launch']['cudnn']:
        raise ValueError('Use the preserved FP32 runtime and backend')
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json', {'version': VERSION, 'protected': protected, **split,
        'selection': zero['selection'], 'original_step0_sha256': audit.STEP0_SHA,
        'preserved_step5000_sha256': audit.STEP5000_SHA, 'b_operators_sha256': b_recovery.B_SHA,
        'base_b_state_sha256': b_recovery.B_STATE_SHA, 'ridge_factor': constrained.RIDGE,
        'rank_rtol': 'max(whitened_constraint_design.shape)*float32_epsilon', 'rank_atol': 0,
        'interface_atol': constrained.ATOL, 'interface_rtol': constrained.RTOL,
        'input_chain': 'Actual sealed B stage2 chain; no transplanted A coefficients',
        'changed_relative_to_b': [PATHS[-1]], 'process_snapshot': process,
        'neural_training_updates': 0, 'optimizer_created': False, 'automatic_promotion': False})
    teacher = model = None; teacher_hash = None; b_versions = None; stage2_hashes = None
    results = {}; fit = {}; receipt = None; status = 'initializing'; failure = None; started = time.monotonic()
    try:
        teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth', device='cuda')
        teacher_hash = control.state_hash(teacher.model)
        model, installation = initialize_b(teacher.model.decoder, zero, artifact, b_receipt, split['fit_source_ids'], artifact_sha256=b_recovery.B_SHA)
        if installation['candidate_state_sha256'] != b_recovery.B_STATE_SHA: raise RuntimeError('Sealed B full-state hash differs')
        b_versions = audit._versions(model.decoder); stage2_hashes = operator_hashes(model)
        objective = base.objective(); metadata = {r['source_id']: r for r in manifest['splits']['development']['rows']}
        audit.warm_models(teacher, [model], development)
        results['baseline_B'] = ordinary.evaluate_candidate(model, teacher, development, objective, metadata, args.out, 'baseline-B')
        parity = numeric_reference_check(results['baseline_B'], reference)
        base.write_json(args.out/'baseline-parity.json', parity)
        if not parity['passed'] or control.state_hash(model.decoder) != b_recovery.B_STATE_SHA:
            raise RuntimeError('B full state or development baseline did not reproduce')
        def observed(candidate, original, crop, site):
            row = constrained.observe_startup(candidate, original, crop, site)
            return row
        fit = refit_b_upsampler(model, teacher, calibration, observe_fn=observed)
        base.write_json(args.out/'constraint-fit.json', fit)
        if not fit['feasible']:
            status = 'no_stable_feasible_solution'
        else:
            status = 'evaluated' if fit['native_fp32_constraints_passed'] else 'evaluated_native_equality_not_met'
            candidate_hash = control.state_hash(model.decoder)
            audit.warm_models(teacher, [model], development)
            results['combined'] = ordinary.evaluate_candidate(model, teacher, development, objective, metadata, args.out, 'combined')
            if control.state_hash(model.decoder) != candidate_hash: raise RuntimeError('Development evaluation changed combined state')
            payload = export_artifact(model, split['fit_source_ids'], b_recovery.B_SHA)
            path = args.out/'combined-native-operators.pt'; torch.save(payload, path)
            receipt = {'operators_sha256': base.sha(path), 'candidate_state_sha256': candidate_hash,
                'base_b_state_sha256': b_recovery.B_STATE_SHA, 'b_operators_sha256': b_recovery.B_SHA,
                'original_step0_pristine': b_receipt['original_step0_pristine'],
                'changed_native_paths': list(PATHS), 'changed_relative_to_b': [PATHS[-1]],
                'stage2_b_operators_preserved': operator_hashes(model) == stage2_hashes,
                'unchanged_frozen_and_unselected_tensors': True, 'native_fp32_constraints_passed': fit['native_fp32_constraints_passed'],
                'full_group_widths': [384, 256, 128], 'all9_residual_units_preserved': True,
                'neural_training_updates': 0, 'extra_inference_modules': 0, 'automatic_promotion': False}
            base.write_json(args.out/'combined-receipt.json', receipt)
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        files = all(base.sha(path) == digest for path, digest in protected.items())
        unchanged = {'teacher': teacher is None or teacher_hash is None or control.state_hash(teacher.model) == teacher_hash,
                     'stage2_b_operators': model is None or stage2_hashes is None or operator_hashes(model) == stage2_hashes}
        unchanged['outer_and_unselected_tensors'] = True
        if model is not None and b_versions is not None:
            try:
                ordinary._assert_only_sites_changed(b_versions, model, [audit.four_sites(teacher.model.decoder, model.selections)[-1]])
            except RuntimeError as guard_failure:
                unchanged['outer_and_unselected_tensors'] = False
                if failure is None: failure = repr(guard_failure)
                status = 'failed'
        complete = failure is None and status in ('evaluated', 'evaluated_native_equality_not_met', 'no_stable_feasible_solution') and files and all(unchanged.values())
        base.write_json(args.out/'completed.json', {'version': VERSION, 'complete': complete, 'status': status,
            'failure': failure, 'files_preserved': files, 'states_preserved': unchanged,
            'stable_constraint_solution': fit.get('feasible', False), 'native_fp32_constraints_passed': fit.get('native_fp32_constraints_passed', False),
            'calibration_sources': 72, 'development_sources': 96, 'neural_training_updates': 0,
            'automatic_promotion': False, 'candidate_receipt': receipt, 'elapsed_seconds': time.monotonic()-started,
            'results': {name: {key: value[key] for key in ('aggregate', 'quiet_regions', 'overview_window_metrics', 'recovery_window_metrics')}
                        for name, value in results.items()}})
        if not files or not all(unchanged.values()): raise RuntimeError('Combined fit altered preserved inputs or model state')


if __name__ == '__main__': main()
