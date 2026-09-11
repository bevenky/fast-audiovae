"""Fresh original teacher to G plus a constrained native upsampler refit.

Only the stage3 upsampler changes relative to the sealed, untrained G state.
Its new fit is unrestricted native delta ridge plus the existing startup
equalities; it need not retain G's shared hidden-map or original-bias structure.
No trained checkpoint, saved group, optimizer, or extra inference module is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import startup_constrained_init as constrained
import grail_candidate_recovery as grail
import independent_trial_aggregate as aggregate

ordinary, audit, common = constrained.ordinary, constrained.audit, constrained.common
group, base, control, progressive = constrained.group, constrained.base, constrained.control, constrained.progressive
VERSION = 'audiovae2_grail_startup_initialization_v1'
VARIANT = 'grail_plus_constrained_upsampler'
PATHS = grail.OPERATOR_PATHS
G_FILES = ('grail-native-operators.pt','grail-receipt.json','launch.json',
           'completed.json','grail-development.json','grail-hidden-maps.pt')
G_PINS = dict(zip(G_FILES,(
    '54a08e3733dbf53090bb3ebc11f65c2c0fac5f625d001b1b40a4455d1d3cf543',
    '65da007ce8c87d8fed2dedd3005ce146b75c15b19020fd7dd4d5b21ca4f906a1',
    '2fd467ff47d5585635542d184ab38accb00a99dbd43035b1a5d2046035236309',
    '72647b417abfbf37a9a11fbdbdd24c7de9fe0c57fa2d00922592abd0f0250ced',
    '8b2b6a528857f0fbb55b24f9bca4244f9889d2e5dee222064e4424aab7029180',
    'de765769a3693d6a00609821439f77162f132a328af8994fdaf4e00b421cecb6')))
G_STATE_SHA = 'c2499384e2981fa5ffbaddc945209b9cf1a5c02552f3d58d709c47cf07ee3b2c'
G_BASE_SHA = '01fdc44c20b0dad261fbdf7d8138f1cd4917d3df351943a0d93fba1885928a32'
TEACHER_STATE_SHA = '863109cec1a3cb1f17a90c9d2062dc7f781ee1362069d562567570970ffc07a2'


@torch.no_grad()
def initialize_g(teacher_decoder,artifact,receipt,calibration_ids,*,artifact_sha256):
    """Fresh teacher factory plus native G operators; no checkpoint group input."""
    teacher_hash = control.state_hash(teacher_decoder)
    if artifact.get('teacher_state_sha256') != teacher_hash or receipt.get('teacher_state_sha256') != teacher_hash:
        raise ValueError('Sealed G belongs to a different original teacher')
    model = progressive.initialize_from_teacher(teacher_decoder,artifact['selection']).eval()
    installed = grail.install_candidate_operators(model,artifact,receipt,
        artifact_sha256=artifact_sha256,calibration_ids=calibration_ids)
    if control.state_hash(teacher_decoder) != teacher_hash:
        raise RuntimeError('Fresh G construction changed the original teacher')
    return model,installed


def operator_hashes(model):
    return {path:control.state_hash(model.decoder.get_submodule(path)) for path in PATHS[:3]}


@torch.no_grad()
def refit_g_upsampler(model,teacher,calibration,*,observe_fn=constrained.observe_startup):
    site = audit.four_sites(teacher.model.decoder,model.selections)[-1]
    if site.path != PATHS[-1]:raise ValueError('Only the declared native upsampler may change')
    module = model.decoder.get_submodule(site.path)
    before = audit._versions(model.decoder)
    g_hash,stage2 = control.state_hash(model.decoder),operator_hashes(model)
    dw,db,report = constrained.fit_constrained_delta(module,
        (observe_fn(model,teacher,crop,site) for crop in calibration))
    report.update(base_g_state_sha256=g_hash,stage2_g_operator_hashes=stage2,
        observations_from_actual_g_chain=True,ordinary_reconstruction_refit=True,
        original_g_hidden_map_family_required=False,original_g_bias_required=False)
    if dw is None:
        if control.state_hash(model.decoder) != g_hash:
            raise RuntimeError('Infeasible fit changed the original G candidate')
        report['stage2_g_operators_preserved'] = True
        return report
    desired = (group.effective_weight(module).detach().double()+dw).to(group.effective_weight(module))
    report['effective_writeback'] = ordinary.apply_delta(module,dw,db)
    if not all(row['allclose_existing'] for row in report['effective_writeback'].values()):
        raise RuntimeError('Native constrained coefficient writeback differs')
    native_checks,equalities,receipts = [],[],[]
    for crop in calibration:
        row = observe_fn(model,teacher,crop,site)
        output = row['current_output']
        native = common.linear_response(module,row['input'],desired,True)
        native_checks.append(control.compare_tensors(output,native))
        mask = row['constraint_mask'].expand_as(output)
        if mask.any():equalities.append(constrained.equality_metrics(output[mask],row['target'][mask]))
        receipts.append(row.get('constraint_receipt',{}))
    ordinary._assert_only_sites_changed(before,model,[site])
    if operator_hashes(model) != stage2:raise RuntimeError('The refit changed a G residual mixer')
    report.update(native_fold_allclose=all(row['allclose_existing'] for row in native_checks),
        native_fold_checks=native_checks,equality_native_fp32=equalities,
        native_fp32_constraints_passed=bool(equalities) and all(row['passed'] for row in equalities),
        stage2_g_operators_preserved=True,native_constraint_receipts=receipts,
        candidate_state_sha256=control.state_hash(model.decoder))
    if not report['native_fold_allclose']:raise RuntimeError('Native execution differs from intended coefficients')
    return report


def export_artifact(model,teacher_decoder,calibration_ids,*,g_sha256,base_g_state_sha256,base_selected_state_sha256):
    if len(calibration_ids) != 72 or len(set(calibration_ids)) != 72:
        raise ValueError('Export requires the original72 distinct calibration sources')
    operators = {path:{key:value.detach().cpu().clone() for key,value in
        model.decoder.get_submodule(path).state_dict().items()} for path in PATHS}
    if any(not torch.isfinite(value).all() for state in operators.values() for value in state.values()):
        raise ValueError('Nonfinite native operator cannot be exported')
    return {'format':VERSION,'variant':VARIANT,'original_step0_sha256':audit.STEP0_SHA,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'teacher_source_sha256':base.SOURCE_SHA256,
        'teacher_state_sha256':control.state_hash(teacher_decoder),'selection':model.selections,
        'calibration_sources':72,'calibration_source_ids_sha256':grail.screen.digest(list(calibration_ids)),
        'g_operators_sha256':g_sha256,'base_g_state_sha256':base_g_state_sha256,
        'base_selected_state_sha256':base_selected_state_sha256,
        'candidate_state_sha256':control.state_hash(model.decoder),'operators':operators,
        'ridge':constrained.RIDGE,'rank_policy':'max(whitened_design.shape)*float32_epsilon; absolute cutoff0',
        'interface_atol':constrained.ATOL,'interface_rtol':constrained.RTOL,
        'stage2_g_operator_hashes':operator_hashes(model),'changed_relative_to_g':[PATHS[-1]],
        'original_g_hidden_map_family_required':False,'original_g_bias_required':False,
        'neural_training_updates':0,'extra_inference_modules':0,'automatic_promotion':False}


def summarize_fit(report):
    """Only named scalar fit evidence and pooled equality statistics leave memory."""
    keys = ('feasible','status','observations','constraint_rows','constraint_outputs','weighted_sample_count',
        'valid_feature_rows','all_valid_rows_included','samples_per_cell','chunk_rows','weighted_error_before',
        'weighted_error_after','raw_after_error_sum','ridge_lambda','retained_rank','discarded_modes',
        'rank_relative_cutoff','rank_absolute_cutoff','cholesky_positive','ordinary_delta_frobenius',
        'constrained_delta_frobenius','change_from_ordinary_frobenius','ordinary_bias_delta_l2',
        'constrained_bias_delta_l2','whitened_projection_norm','native_fold_allclose',
        'native_fp32_constraints_passed','stage2_g_operators_preserved','observations_from_actual_g_chain')
    result = {key:report[key] for key in keys if key in report}
    for key in ('equality_before','equality_fp64'):
        if key in report:result[key] = dict(report[key])
    checks = report.get('equality_native_fp32',[])
    result['native_equalities'] = {'windows':len(checks),'passed':sum(c['passed'] for c in checks),
        'elements':sum(c['elements'] for c in checks),
        'max_abs':max((c['max_abs'] for c in checks),default=None),
        'max_fraction_of_original_tolerance':max((c['max_fraction_of_original_tolerance'] for c in checks),default=None)}
    receipts = report.get('constraint_receipts',[])
    result['calibration_startup'] = {'windows':sum(len(r.get('windows',[])) for r in receipts),
        'samples':sum(r.get('observed_near_startup_samples',0) for r in receipts),
        'native_constraint_cells':sum(r.get('constraint_cells',0) for r in receipts),
        'excluded_partial_cell_samples':sum(r.get('excluded_partial_cell_samples',0) for r in receipts)}
    result['error_after_scope'] = 'FP64 solver prediction before installed FP32 native execution'
    return result


def summarize_quality(report):
    result = aggregate.quality(report)
    result['teacher_window_identity_sha256'] = report['quiet_regions']['window_identity_sha256']
    if result['regions']['near_startup_first20ms']['windows'] != 13:
        raise ValueError('Original development startup support differs')
    return result


def authenticate_g(directory,calibration_ids,development_ids,*,pins,state_sha256):
    """Read only the sealed untrained G artifact; never load any saved group."""
    if pins != G_PINS or state_sha256 != G_STATE_SHA:
        raise ValueError('Expected the declared original G artifact and state pins')
    paths = {name:directory/name for name in G_FILES}
    if any(base.sha(paths[name]) != digest for name,digest in pins.items()):
        raise ValueError('A pinned original G input changed')
    receipt,launch,done,reference = (json.loads(paths[name].read_text()) for name in
        ('grail-receipt.json','launch.json','completed.json','grail-development.json'))
    artifact = torch.load(paths['grail-native-operators.pt'],map_location='cpu',weights_only=True)
    selection = artifact.get('selection',{})
    if (done.get('complete') is not True or done.get('failure') is not None
            or done.get('version') != grail.INIT_VERSION or done.get('variant') != grail.VARIANT
            or done.get('status') != 'evaluated' or done.get('files_preserved') is not True
            or not done.get('states_preserved') or not all(done['states_preserved'].values())
            or done.get('neural_training_updates') != 0 or done.get('optimizer_created') is not False
            or done.get('calibration_sources') != 72 or done.get('development_sources') != 96
            or done.get('teacher_cache_comparisons') != 384 or done.get('teacher_cache_all_pass') is not True
            or done.get('candidate_receipt') != receipt
            or launch.get('version') != grail.INIT_VERSION or launch.get('variant') != grail.VARIANT
            or launch.get('fit_source_ids') != list(calibration_ids)
            or launch.get('development_source_ids') != list(development_ids)
            or launch.get('original_step0_sha256') != audit.STEP0_SHA
            or launch.get('selection') != selection or receipt.get('selection') != selection
            or len(selection.get('stage2_indices',[])) != 384
            or selection.get('stage3_indices') != list(range(256))
            or launch.get('intercept') is not False or launch.get('uncentered_hidden_map') is not True
            or launch.get('shared_map_across_native_taps') is not True or launch.get('ridge_factor') != constrained.RIDGE
            or receipt.get('candidate_state_sha256') != state_sha256
            or artifact.get('candidate_state_sha256') != state_sha256
            or receipt.get('base_selected_state_sha256') != G_BASE_SHA
            or artifact.get('base_selected_state_sha256') != G_BASE_SHA
            or receipt.get('teacher_state_sha256') != TEACHER_STATE_SHA
            or artifact.get('teacher_state_sha256') != TEACHER_STATE_SHA
            or receipt.get('hidden_maps_sha256') != pins['grail-hidden-maps.pt']
            or artifact.get('hidden_maps_sha256') != pins['grail-hidden-maps.pt']
            or receipt.get('native_writeback_passed') is not True):
        raise ValueError('Original G provenance, teacher, support or source partition differs')
    if any(done.get('results',{}).get('grail',{}).get(key) != reference.get(key)
           for key in ('aggregate','quiet_regions','overview_window_metrics','recovery_window_metrics')):
        raise ValueError('G completion metrics differ from the sealed full development reference')
    protected = {**launch['protected'],**{str(p.resolve()):base.sha(p) for p in paths.values()}}
    if any(base.sha(path) != digest for path,digest in protected.items()):
        raise ValueError('An original G protected file changed')
    return artifact,receipt,reference,launch,protected


@torch.no_grad()
def main():
    from compare_accumulation_v1 import numeric_reference_check
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    import startup_retention_probe as startup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if args.out.exists():raise FileExistsError('Use a new isolated G-plus-startup output directory')
    assets,manifest_path,step0,g_dir = (Path(config[k]) for k in ('assets','manifest','original_step0','g_dir'))
    if any(args.out.resolve().is_relative_to(path.resolve()) for path in (g_dir,step0.parent)):
        raise ValueError('Output overlaps an immutable original run')
    if (base.sha(step0) != audit.STEP0_SHA or base.sha(assets/'audio_vae_v2.py') != base.SOURCE_SHA256
            or base.sha(assets/'audiovae.pth') != base.CHECKPOINT_SHA256):
        raise ValueError('Original teacher assets or step0 provenance differs')
    # Historical step0 is byte-authenticated only; its tensors are never loaded.
    manifest,pools,_ = base.load_data(manifest_path)
    calibration,development = pools['calibration'],pools['development']
    if len(calibration) != 72 or len(development) != 96:
        raise ValueError('Require the original72 calibration and96 development crops')
    split = common.validate_fit_sources(calibration,development)
    artifact,g_receipt,g_reference,g_launch,protected = authenticate_g(g_dir,split['fit_source_ids'],
        split['development_source_ids'],pins=config['g_pins'],state_sha256=config['g_state_sha256'])
    if base.sha(manifest_path) not in g_launch['protected'].values():
        raise ValueError('Manifest was not part of the original G initialization')
    process = control.gpu_idle_snapshot(); base.policy()
    if str(torch.__version__) != g_launch['torch'] or torch.backends.cudnn.version() != g_launch['cudnn']:
        raise ValueError('Use the original qualified singleton FP32 runtime')
    modules = (constrained,ordinary,audit,common,grail,grail.b,control,group,base,progressive,
               aggregate,startup,common.replay)
    paths = [args.config,Path(__file__),step0,manifest_path,assets/'audio_vae_v2.py',assets/'audiovae.pth',
             Path(manifest['cache_path']),Path(manifest['overlay_receipt_path']),
             *(Path(module.__file__) for module in modules)]
    protected.update({str(path.resolve()):base.sha(path) for path in paths})
    args.out.mkdir(parents=True)
    launch = {'version':VERSION,'variant':VARIANT,'config_sha256':base.sha(args.config),
        'original_step0_sha256':audit.STEP0_SHA,'original_step0_tensors_loaded':False,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'teacher_source_sha256':base.SOURCE_SHA256,
        'g_pins':G_PINS,'base_g_state_sha256':G_STATE_SHA,'base_selected_state_sha256':G_BASE_SHA,
        'teacher_state_sha256':TEACHER_STATE_SHA,'selection':artifact['selection'],
        'calibration_sources':72,'development_sources':96,
        'calibration_source_ids_sha256':grail.screen.digest(split['fit_source_ids']),
        'development_source_ids_sha256':grail.screen.digest(split['development_source_ids']),
        'source_sha256':{p.name:base.sha(p) for p in paths if p.suffix=='.py'},
        'protected_inputs_identity_sha256':grail.screen.digest(protected),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),
        'backend':common.replay.backend_state(),'process_snapshot':process,
        'ridge_factor':constrained.RIDGE,'rank_policy':'max(whitened_design.shape)*float32_epsilon; absolute cutoff0',
        'native_geometry':{'input':384,'output':256,'stride':5,'kernel':10,'trim':5,'shared_bias':True},
        'changed_relative_to_g':[PATHS[-1]],'ordinary_reconstruction_refit':True,
        'original_g_hidden_map_family_required':False,'original_g_bias_required':False,
        'neural_training_updates':0,'optimizer_created':False,'automatic_promotion':False,
        'warmups':'Original per-shape teacher/student warmups separate from scored evaluations',
        'output_policy':'Aggregate JSON and native parameter artifact; no source IDs, window rows, audio or latents'}
    base.write_json(args.out/'launch.json',launch)
    teacher=model=None; teacher_hash=None; versions=None; stage2=None; receipt=None
    results={}; fit={}; cache_records=[]; anchor_checks={}; status='initializing'; failure=None
    started=time.monotonic(); initial_rng=grail.screen.rng_state()
    try:
        teacher=base.FrozenAudioVAE2.from_files(assets/'audio_vae_v2.py',assets/'audiovae.pth',device='cuda')
        teacher_hash=control.state_hash(teacher.model)
        model,installation=initialize_g(teacher.model.decoder,artifact,g_receipt,split['fit_source_ids'],
            artifact_sha256=G_PINS['grail-native-operators.pt'])
        if control.state_hash(model.decoder) != G_STATE_SHA or len(model.trainable_group_parameters()) != 90:
            raise RuntimeError('Original untrained G state or native90-tensor scope differs')
        versions=audit._versions(model.decoder); stage2=operator_hashes(model)
        objective=base.objective(); metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
        evaluator=UnifiedMonitor(None,{},metadata)
        def evaluate():
            audit.warm_models(teacher,[model],development)
            with common.replay.observe_teacher_cache(development) as checks:
                report=control.joint.evaluate_review(evaluator,model,teacher,development,objective,summarize=summarize_regions)
            if not all(c['allclose_original_tolerance'] for c in checks):raise RuntimeError('Development cached teacher target mismatch')
            cache_records.extend(checks)
            return report
        baseline=evaluate()
        parity=numeric_reference_check(baseline,g_reference)
        base.write_json(args.out/'baseline-parity.json',{'passed':parity['passed'],'failure_count':len(parity['failures']),
            'relative_tolerance':parity['relative_tolerance'],'absolute_tolerance':parity['absolute_tolerance'],
            'complete_G_state_exact':control.state_hash(model.decoder)==G_STATE_SHA})
        if not parity['passed'] or control.state_hash(model.decoder)!=G_STATE_SHA:
            raise RuntimeError('Sealed original G full development baseline did not reproduce')
        results['baseline_G']=summarize_quality(baseline)
        anchors,anchor_checks=startup.startup_panel(teacher,calibration,6)
        results['baseline_G']['calibration_startup']=startup.score_panel(model,anchors)
        base.write_json(args.out/'baseline-G-development-aggregate.json',results['baseline_G'])
        def observed(candidate,original,crop,site):
            with common.replay.observe_teacher_cache([crop]) as checks:
                row=constrained.observe_startup(candidate,original,crop,site)
            if not all(c['allclose_original_tolerance'] for c in checks):raise RuntimeError('Calibration cached teacher target mismatch')
            cache_records.extend(checks)
            return row
        fit=refit_g_upsampler(model,teacher,calibration,observe_fn=observed)
        base.write_json(args.out/'constraint-fit-aggregate.json',summarize_fit(fit))
        if not fit['feasible']:
            status='no_stable_feasible_solution'
        else:
            if summarize_fit(fit)['native_equalities']['windows']!=6:
                raise RuntimeError('Native calibration constraint support differs from six starts')
            candidate_hash=control.state_hash(model.decoder)
            result=evaluate()
            results['grail_startup']=summarize_quality(result)
            results['grail_startup']['calibration_startup']=startup.score_panel(model,anchors)
            results['grail_startup']['per_recording_comparison_counts']=aggregate.paired(result,baseline)
            if control.state_hash(model.decoder)!=candidate_hash:raise RuntimeError('Evaluation changed the fitted candidate')
            status='evaluated' if fit['native_fp32_constraints_passed'] else 'evaluated_native_equality_not_met'
            payload=export_artifact(model,teacher.model.decoder,split['fit_source_ids'],
                g_sha256=G_PINS['grail-native-operators.pt'],base_g_state_sha256=G_STATE_SHA,base_selected_state_sha256=G_BASE_SHA)
            path=args.out/'grail-startup-native-operators.pt'; torch.save(payload,path)
            receipt={key:payload[key] for key in ('format','variant','original_step0_sha256','teacher_state_sha256',
                'teacher_checkpoint_sha256','teacher_source_sha256','selection','calibration_sources',
                'calibration_source_ids_sha256','g_operators_sha256','base_g_state_sha256','base_selected_state_sha256',
                'candidate_state_sha256','stage2_g_operator_hashes','changed_relative_to_g')}
            receipt.update(operators_sha256=base.sha(path),changed_native_paths=list(PATHS),
                stage2_g_operators_preserved=operator_hashes(model)==stage2,
                native_fp32_constraints_passed=fit['native_fp32_constraints_passed'],
                calibration_waveform_startup_passed=results['grail_startup']['calibration_startup']['passed'],
                development_waveform_startup_passed=13-results['grail_startup']['regions']['near_startup_first20ms']['failed'],
                full_group_widths=[384,256,128],all9_residual_units_preserved=True,
                fresh_original_teacher_factory=True,original_checkpoint_weights_installed=False,
                unchanged_frozen_and_unselected_tensors=True,neural_training_updates=0,
                extra_inference_modules=0,automatic_promotion=False)
            base.write_json(args.out/'grail-startup-receipt.json',receipt)
            base.write_json(args.out/'grail-startup-development-aggregate.json',results['grail_startup'])
    except BaseException as exc:
        failure=type(exc).__name__; status='failed'
        raise
    finally:
        grail.screen.restore_rng(initial_rng)
        states={'teacher':teacher is None or teacher_hash is None or control.state_hash(teacher.model)==teacher_hash,
            'stage2_g_operators':model is None or stage2 is None or operator_hashes(model)==stage2,
            'outer_and_unselected_tensors':True,'rng':grail.replay.compare_tree(grail.screen.rng_state(),initial_rng)['equal']}
        if model is not None and versions is not None:
            try:ordinary._assert_only_sites_changed(versions,model,[audit.four_sites(teacher.model.decoder,model.selections)[-1]])
            except RuntimeError:states['outer_and_unselected_tensors']=False
        files=all(base.sha(path)==digest for path,digest in protected.items())
        expected=336 if fit.get('feasible') else 168
        cache_ok=len(cache_records)==expected and all(c['allclose_original_tolerance'] for c in cache_records)
        done=failure is None and status in ('evaluated','evaluated_native_equality_not_met','no_stable_feasible_solution') and files and all(states.values()) and cache_ok
        base.write_json(args.out/'completed.json',{'version':VERSION,'variant':VARIANT,'complete':done,'status':status,
            'failure_category':failure,'files_preserved':files,'states_preserved':states,
            'calibration_sources':72,'development_sources':96,'neural_training_updates':0,'optimizer_created':False,
            'stable_constraint_solution':fit.get('feasible',False),'native_fp32_constraints_passed':fit.get('native_fp32_constraints_passed',False),
            'teacher_cache_comparisons':len(cache_records),'teacher_cache_all_pass':cache_ok,
            'teacher_cache_nonexact':sum(not c['bitwise_equal'] for c in cache_records),
            'calibration_anchor_preparation':anchor_checks,'candidate_receipt':receipt,
            'results':results,'automatic_promotion':False,'elapsed_seconds':time.monotonic()-started})
        if not files or not all(states.values()):raise RuntimeError('Initializer changed a protected state or input')
        if failure is None and not done:raise RuntimeError('Initializer source accounting is incomplete')


if __name__=='__main__':main()
