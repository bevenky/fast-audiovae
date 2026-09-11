"""Independent 2,000-update recovery from GRAIL-like native hidden-map folds.

Historical checkpoints authenticate RNG, empty Adam settings and source order.
Their learned or sliced group tensors are never installed into this candidate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
import reconstruction_b_recovery as b

base,screen,replay,resume,joint = b.base,b.screen,b.replay,b.resume,b.joint
prior,progressive,control = b.prior,b.progressive,b.control
VERSION = 'audiovae2_grail_candidate_recovery_v1'
INIT_VERSION = 'audiovae2_grail_hidden_initialization_v1'
VARIANT = 'grail_shared_hidden'
OPERATOR_PATHS = b.OPERATOR_PATHS
ACCUMULATION,UPDATES = 12,2000
REVIEW_STEPS = (0,250,500,1000,1500,2000)
CHECKPOINT_STEPS = (0,1000,2000)
perform_update = prior.perform_update


def install_candidate_operators(model,artifact,receipt,*,artifact_sha256,calibration_ids):
    """Authenticate every state before writing to the new pristine teacher slice."""
    if (artifact.get('format') != INIT_VERSION or artifact.get('variant') != VARIANT
            or artifact.get('original_step0_sha256') != b.STEP0_SHA
            or artifact.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256
            or artifact.get('teacher_source_sha256') != base.SOURCE_SHA256
            or artifact.get('selection') != model.selections
            or artifact.get('fit_source_ids') != list(calibration_ids)
            or len(calibration_ids) != 72 or len(set(calibration_ids)) != 72
            or artifact.get('ridge') != 1e-6 or artifact.get('automatic_promotion') is not False
            or receipt.get('operators_sha256') != artifact_sha256
            or receipt.get('changed_native_paths') != list(OPERATOR_PATHS)
            or receipt.get('all9_residual_units_preserved') is not True
            or receipt.get('neural_training_updates') != 0 or receipt.get('extra_inference_modules') != 0
            or receipt.get('unchanged_frozen_and_unselected_tensors') is not True
            or receipt.get('automatic_promotion') is not False
            or artifact.get('hidden_maps_exported') is not False
            or receipt.get('hidden_maps_exported') is not False
            or not artifact.get('hidden_maps_sha256')
            or artifact.get('hidden_maps_sha256') != receipt.get('hidden_maps_sha256')
            or receipt.get('original_bias_bitwise_preserved') is not True
            or receipt.get('fresh_factory_matches_original_step0') is not True
            or receipt.get('original_checkpoint_weights_installed') is not False
            or set(artifact.get('operators',{})) != set(OPERATOR_PATHS)):
        raise ValueError('Candidate provenance, source split or native operator scope differs')
    state = model.decoder.state_dict()
    pristine_hash = b._state_hash(state)
    if (artifact.get('base_selected_state_sha256') != pristine_hash
            or receipt.get('base_selected_state_sha256') != pristine_hash):
        raise ValueError('Candidate must start on its freshly selected original-teacher basis')
    proposed = state.copy(); modules = {}
    for path in OPERATOR_PATHS:
        module = model.decoder.get_submodule(path); expected = module.state_dict()
        supplied = artifact['operators'][path]
        if supplied.keys() != expected.keys(): raise ValueError('Native state keys differ: '+path)
        for key,value in supplied.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
                raise ValueError('Native tensor geometry/dtype/finiteness differs: '+path+'.'+key)
            if key == 'bias' and not torch.equal(value.to(expected[key].device),expected[key]):
                raise ValueError('GRAIL must preserve the selected original shared native bias: '+path)
            proposed[path+'.'+key] = value
        modules[path] = module
    wanted = b._state_hash(proposed)
    if wanted != receipt.get('candidate_state_sha256') or wanted != artifact.get('candidate_state_sha256'):
        raise ValueError('Candidate operators do not reconstruct the sealed complete decoder')
    objects = [(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
    for path,module in modules.items(): module.load_state_dict(artifact['operators'][path],strict=True)
    if (objects != [(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
            or control.state_hash(model.decoder) != wanted):
        raise RuntimeError('Candidate installation changed parameter objects or native state')
    return {'candidate_state_sha256':wanted,'base_selected_state_sha256':pristine_hash,
            'parameter_objects_preserved':True,'only_four_native_operators_installed':True,
            'old_group_state_loaded':False,'fits_repeated':False,'optimizer_updates':0}


def restore_candidate_start(model,teacher,zero,artifact,receipt,calibration_ids,*,artifact_sha256):
    """Use historical state solely as the authenticated fresh RNG/Adam template."""
    if (zero.get('cut_updates') != 0 or zero.get('global_updates') != 0
            or zero.get('sources_seen') != [] or zero.get('optimizer',{}).get('state') != {}):
        raise ValueError('Historical template must have zero updates and empty optimizer state')
    if model.selections != zero.get('selection') or artifact.get('selection') != zero.get('selection'):
        raise ValueError('GRAIL must use the exact original pivoted support, not the new downstream selector')
    teacher_hash = control.state_hash(teacher.model.decoder)
    if artifact.get('teacher_state_sha256') != teacher_hash or receipt.get('teacher_state_sha256') != teacher_hash:
        raise ValueError('Candidate used a different original teacher')
    installed = install_candidate_operators(model,artifact,receipt,artifact_sha256=artifact_sha256,
                                           calibration_ids=calibration_ids)
    optimizer = progressive.fresh_optimizer(model,lr=3e-5)
    screen.restore_rng(zero['rng'])
    if (optimizer.state or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
            or not replay.compare_tree(screen.rng_state(),zero['rng'])['equal']
            or control.state_hash(teacher.model.decoder) != teacher_hash):
        raise RuntimeError('Original RNG, fresh Adam recipe or frozen teacher differs')
    return optimizer,{**installed,'original_step0_rng_restored_exactly':True,'fresh_adam_matches_original':True}


def authenticate_inputs(args,data,pools):
    """Reuse B's verified historical ledger; authenticate the new basis separately."""
    zero,_,_,source_ids,recipe,_,references,protected = b.authenticate_inputs(args,data,pools)
    folder = args.candidate_dir
    artifact_path = folder/'grail-native-operators.pt'
    if base.sha(artifact_path) != args.candidate_artifact_sha256:
        raise ValueError('Unexpected fresh candidate artifact bytes')
    artifact = torch.load(artifact_path,map_location='cpu',weights_only=True)
    receipt = json.loads((folder/'grail-receipt.json').read_text())
    launch = json.loads((folder/'launch.json').read_text())
    done = json.loads((folder/'completed.json').read_text())
    reference = json.loads((folder/'grail-development.json').read_text())
    calibration_ids = [c['source_id'] for c in pools['calibration']]
    development_ids = [c['source_id'] for c in pools['development']]
    selection = artifact.get('selection',{})
    if (done.get('complete') is not True or done.get('failure') is not None
            or done.get('files_preserved') is not True or not done.get('states_preserved')
            or not all(done['states_preserved'].values())
            or done.get('neural_training_updates') != 0 or done.get('calibration_sources') != 72
            or done.get('development_sources') != 96
            or launch.get('fit_source_ids') != calibration_ids or launch.get('development_source_ids') != development_ids
            or launch.get('original_step0_sha256') != b.STEP0_SHA
            or done.get('version') != INIT_VERSION or done.get('variant') != VARIANT
            or launch.get('version') != INIT_VERSION or launch.get('variant') != VARIANT
            or done.get('status') != 'evaluated'
            or done.get('teacher_cache_comparisons') != 384 or done.get('teacher_cache_all_pass') is not True
            or launch.get('intercept') is not False or launch.get('uncentered_hidden_map') is not True
            or launch.get('shared_map_across_native_taps') is not True or launch.get('ridge_factor') != 1e-6
            or selection != zero.get('selection') or receipt.get('selection') != selection
            or len(selection.get('stage2_indices',[])) != 384
            or selection.get('stage3_indices') != list(range(256))
            or receipt.get('candidate_state_sha256') != args.candidate_state_sha256
            or artifact.get('candidate_state_sha256') != args.candidate_state_sha256
            or done.get('results',{}).get('grail',{}).get('aggregate') != reference['aggregate']):
        raise ValueError('Fresh GRAIL initialization, original support or completion receipt differs')
    maps_path = folder/'grail-hidden-maps.pt'
    maps_sha = base.sha(maps_path)
    if maps_sha != artifact.get('hidden_maps_sha256') or maps_sha != receipt.get('hidden_maps_sha256'):
        raise ValueError('GRAIL analysis-only hidden maps changed')
    maps_payload = torch.load(maps_path,map_location='cpu',weights_only=True)
    if (maps_payload.get('format') != INIT_VERSION or maps_payload.get('deployed') is not False
            or maps_payload.get('fit_source_ids') != calibration_ids
            or maps_payload.get('teacher_state_sha256') != artifact.get('teacher_state_sha256')
            or set(maps_payload.get('maps',{})) != set(OPERATOR_PATHS)
            or any(not isinstance(t,torch.Tensor) or t.shape != (512,384) or t.dtype != torch.float64
                   or not torch.isfinite(t).all() for t in maps_payload['maps'].values())):
        raise ValueError('GRAIL hidden map topology or calibration provenance differs')
    if any(done['results']['grail'].get(k) != reference.get(k)
           for k in ('quiet_regions','overview_window_metrics','recovery_window_metrics')):
        raise ValueError('GRAIL fixed-panel completion metrics differ from its full reference')
    protected.update(launch['protected'])
    paths = [folder/name for name in ('launch.json','completed.json','grail-native-operators.pt',
        'grail-receipt.json','grail-development.json','grail-hidden-maps.pt')]
    b_done = json.loads((args.b_run/'completed.json').read_text())
    b_launch = json.loads((args.b_run/'launch.json').read_text())
    b_journal = [json.loads(l) for l in (args.b_run/'train.jsonl').read_text().splitlines()]
    if (b_done.get('status') != 'awaiting_review' or b_done.get('step') != 2000
            or b_done.get('source_count') != 24000 or b_done.get('frozen_state_preserved') is not True
            or b_done.get('original_files_preserved') is not True
            or [r['step'] for r in b_journal] != list(range(1,2001))
            or [s for r in b_journal for s in r['source_ids']] != list(source_ids)
            or b_launch.get('source_ids') != list(source_ids)):
        raise ValueError('Matched B reference source exposure/completion differs')
    b_checkpoint = args.b_run/'checkpoint-step2000.pt'
    b_receipt = json.loads(b_checkpoint.with_suffix('.json').read_text())
    if (base.sha(b_checkpoint) != '43ef31f30cca751167cf91152f3ab039592b06860f0ffbd60531c8862c783954'
            or b_done['last_checkpoint_sha256'] != b_receipt['checkpoint_sha256']
            or base.sha(b_checkpoint) != b_receipt['checkpoint_sha256']):
        raise ValueError('Matched B checkpoint identity differs')
    b_references = {step:json.loads((args.b_run/f'development-step{step}.json').read_text()) for step in REVIEW_STEPS}
    if b_references[2000]['aggregate'] != b_receipt['quality']:
        raise ValueError('Matched B endpoint metrics differ from its checkpoint receipt')
    paths += [args.b_run/name for name in ('launch.json','completed.json','train.jsonl','checkpoint-step2000.pt','checkpoint-step2000.json')]
    paths += [args.b_run/f'development-step{step}.json' for step in REVIEW_STEPS]
    protected.update({str(p.resolve()):base.sha(p) for p in paths})
    if any(base.sha(p) != h for p,h in protected.items()):
        raise ValueError('A preserved original or initializer input changed')
    return zero,artifact,receipt,source_ids,recipe,reference,references,b_references,protected


def save_checkpoint(path,model,optimizer,step,source_ids,identity,selection,scored_samples):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError('Only step0/1000/2000 checkpoint states are authorized')
    expected = identity['source_ids'][:step*ACCUMULATION]
    if (len(identity['source_ids']) != 24000 or len(set(identity['source_ids'])) != 24000
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']
            or list(source_ids) != expected or len(source_ids) != step*ACCUMULATION
            or selection != identity['selection']):
        raise ValueError('GRAIL checkpoint source ledger or selection differs')
    params = base.parameters(model)
    if (not params or (set(optimizer.state) != set(params) if step else bool(optimizer.state))
            or any(float(s['step']) != step for s in optimizer.state.values())):
        raise ValueError('GRAIL checkpoint Adam counters/scope differ from uninterrupted training')
    for row in optimizer.param_groups:
        if (row['lr'] != 3e-5 or row['betas'] != (.9,.99) or row['eps'] != 1e-8 or row['weight_decay'] != 0
                or set(row['params']) != set(params)):
            raise ValueError('GRAIL checkpoint optimizer settings changed')
    if type(scored_samples) is not int or (scored_samples <= 0 if step else scored_samples != 0):
        raise ValueError('Missing exact scored audio accounting')
    payload = {'format':VERSION,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'step':step,'cut_index':1,'cut_updates':step,
        'global_updates':step,'selection':selection,'schedule_sha256':identity['schedule_sha256'],
        'sources_seen':list(source_ids),'scored_samples':scored_samples,'coefficients':dict(prior.COEFFICIENTS),
        'accumulation':ACCUMULATION,'teacher_source_sha256':base.SOURCE_SHA256,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'identity':identity,
        'optimizer_reset_after_start':False,'automatic_next_cut':False,'automatic_promotion':False}
    # Group+two Adam moments, without a full decoder copy. Allow serialization overhead.
    estimate = sum(t.numel()*t.element_size() for t in payload['group'].values())*(3 if step else 1)+8*1024**2
    if shutil.disk_usage(path.parent).free < estimate+32*1024**2:
        raise RuntimeError('Insufficient disk headroom for the bounded group/optimizer checkpoint')
    tmp = path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
    return {'checkpoint_sha256':base.sha(path),'step':step,'sources_seen':len(source_ids),
            'optimizer_and_rng_saved':True,'optimizer_reset_after_start':False,'bytes':path.stat().st_size}



def main():
    import grail_candidate_monitor as monitor_module
    import group_model
    import fresh_training_data
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0','original2000','preserved5000','b-dir','b-run','candidate-dir','manifest','source-plan',
                 'shards','assets','recipe','schedule','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--candidate-artifact-sha256',required=True)
    parser.add_argument('--candidate-state-sha256',required=True)
    parser.add_argument('--tensorboard',type=Path)
    parser.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = parser.parse_args()
    logdir = (args.tensorboard or args.out/'tensorboard').resolve()
    if args.out.exists() or logdir.exists(): raise FileExistsError('Use new fresh candidate recovery output and event directories')
    retained = (args.step0.parent,args.original2000.parent,args.preserved5000.parent,args.b_dir,args.b_run,args.candidate_dir,args.recipe.parent)
    if any(new.is_relative_to(old.resolve()) or old.resolve().is_relative_to(new)
           for new in (args.out.resolve(),logdir) for old in retained):
        raise ValueError('Fresh candidate output would overlap a preserved run')
    base.policy()
    manifest,pools,_ = base.load_data(args.manifest)
    if len(pools['calibration']) != 72 or len(pools['development']) != 96:
        raise ValueError('Use the original72 calibration and96 development sources')
    fresh = fresh_training_data.FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    data = prior.SourceStream(pools,fresh)
    zero,artifact,receipt,source_ids,recipe,reference,references,b_references,protected = authenticate_inputs(args,data,pools)
    modules = (b,prior,progressive,control,base,screen,replay,joint,group_model,fresh_training_data,monitor_module)
    protected.update({str(Path(m.__file__).resolve()):base.sha(m.__file__) for m in modules})
    protected[str(Path(__file__).resolve())] = base.sha(__file__)
    process = control.gpu_idle_snapshot()
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise ValueError('Use the original qualified Torch/cuDNN runtime')
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    selection = artifact['selection']
    model = progressive.initialize_from_teacher(teacher.model.decoder,selection)
    optimizer,installation = restore_candidate_start(model,teacher,zero,artifact,receipt,
        [c['source_id'] for c in pools['calibration']],artifact_sha256=args.candidate_artifact_sha256)
    if len(base.parameters(model)) != 90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Expected all90 joint group parameters and the original frozen teacher')
    frozen = screen.frozen_versions(model); teacher_frozen = resume.continuation.teacher_versions(teacher)
    teacher_hash = control.state_hash(teacher.model)
    common = base.objective(); metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    evaluator = UnifiedMonitor(None,{},metadata)
    boundary_panel,boundary_selection = base.select_boundary_panel(pools['development'],metadata)
    parameter_bytes = sum(v.numel()*v.element_size() for v in model.group_state_dict().values())
    required = parameter_bytes*7+64*1024**2
    ancestor = args.out.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < required: raise RuntimeError('Insufficient disk for group checkpoints0/1000/2000 and compact logs')
    args.out.mkdir(parents=True)
    monitor = monitor_module.RecoveryMonitor(logdir,source_metadata=metadata)
    screen.restore_rng(zero['rng'])
    identity = {'version':VERSION,'initialization':'Original teacher -> original pivoted basis -> four sealed zero-intercept shared hidden maps folded into native weights',
        'original_step0_sha256':b.STEP0_SHA,'reference_b_operators_sha256':b.B_SHA,'candidate_operators_sha256':args.candidate_artifact_sha256,
        'initial_decoder_state_sha256':args.candidate_state_sha256,'original2000_sha256':b.ORIGINAL2000_SHA,
        'preserved5000_sha256':b.STEP5000_SHA,'b_recovery_checkpoint_sha256':base.sha(args.b_run/'checkpoint-step2000.pt'),
        'selection':selection,'schedule_sha256':base.sha(args.schedule),'coefficients':dict(prior.COEFFICIENTS),
        'learning_rate':3e-5,'optimizer_betas':[.9,.99],'optimizer_eps':1e-8,'weight_decay':0,
        'execution_batch_size':1,'gradient_accumulation':12,'trainable_stages':[2,3,4],
        'source_interval':[0,24000],'source_ids':list(source_ids),'source_ids_sha256':screen.digest(list(source_ids)),
        'source_plan_identity_sha256':fresh.identity,'protected':protected,'optimizer_reset_after_start':False,
        'original_teacher_forever':True,'rng_origin':'Exact original step0 RNG restored after installation/monitor setup',
        'original_policy_seed':20260910,'backend':replay.backend_state(),'torch':str(torch.__version__),
        'cudnn':torch.backends.cudnn.version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'precision':'FP32 TF32 disabled','process_snapshot':process,'installation':installation,
        'review_steps':list(REVIEW_STEPS),'checkpoint_steps':list(CHECKPOINT_STEPS),'target_updates':UPDATES,
        'tensorboard':str(logdir),'disk_free_before':free,'estimated_checkpoints_and_logs_bytes':required,
        'boundary_panel':boundary_selection,'automatic_next_cut':False,'automatic_promotion':False,
        'new_startup_training_loss':False,'no_quality_based_early_stopping':True,'historical_bitwise_trajectory_claim':False,
        'original_pivoted_support_preserved':True,'hidden_map_intercept':False,'original_native_bias_preserved':True,
        'hidden_maps_sha256':artifact['hidden_maps_sha256'],'hidden_maps_deployed':False,'extra_inference_modules':0}
    base.write_json(args.out/'launch.json',identity)
    seen = []; samples = 0; warmed = set(); reports = {}; last_receipt = None
    started = time.monotonic(); update_seconds = validation_seconds = waiting_seconds = 0.
    failure = None; status = 'running'
    def assert_frozen():
        if screen.frozen_versions(model) != frozen or resume.continuation.teacher_versions(teacher) != teacher_frozen:
            raise RuntimeError('A frozen teacher or outer student tensor changed')
    def evaluate(step):
        nonlocal validation_seconds
        before = time.monotonic()
        def capture(rows):
            control._write_rows(args.out/f'quiet-windows-step{step}.jsonl.gz',rows)
            return summarize_regions(rows)
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            report = joint.evaluate_review(evaluator,model,teacher,pools['development'],common,summarize=capture)
            boundary = base.evaluate_boundaries(teacher,model,boundary_panel,selection,base.batch,base.teacher_forward)
        validation_seconds += time.monotonic()-before; assert_frozen()
        base.write_json(args.out/f'development-step{step}.json',report)
        base.write_json(args.out/f'boundaries-step{step}.json',boundary)
        return report
    try:
        prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
        initial = evaluate(0)
        comparison = {'quality':numeric_reference_check(initial,reference),
            'rng':replay.compare_tree(screen.rng_state(),zero['rng']),
            'optimizer':replay.compare_tree(optimizer.state_dict(),zero['optimizer']),
            'candidate_state_exact':control.state_hash(model.decoder) == args.candidate_state_sha256}
        base.write_json(args.out/'initial-parity.json',comparison)
        if (not comparison['quality']['passed'] or not comparison['rng']['equal']
                or not comparison['optimizer']['equal'] or not comparison['candidate_state_exact']):
            raise RuntimeError('Fresh candidate full baseline, native state, original RNG or fresh Adam failed to reproduce')
        reports[0] = initial; monitor.log_validation(initial,0,references[0],b_references[0])
        last_receipt = save_checkpoint(args.out/'checkpoint-step0.pt',model,optimizer,0,seen,identity,selection,samples)
        last_receipt.update({'frozen_state_preserved':True,'quality':initial['aggregate']})
        base.write_json(args.out/'checkpoint-step0.json',last_receipt)
        for step in range(1,UPDATES+1):
            before = time.monotonic()
            crops = resume.take_when_ready(data,(step-1)*12,12,step=step,monitor=monitor,wait_seconds=args.shard_wait_seconds)
            waiting_seconds += time.monotonic()-before
            ids = [c['source_id'] for c in crops]
            if ids != list(source_ids[(step-1)*12:step*12]): raise RuntimeError('Original ordered crop prefix changed')
            prior.warm_student(model,teacher,crops,warmed,optimizer)
            before = time.monotonic()
            values,checks = perform_update(model,teacher,crops,common,optimizer,diagnostics=step in (1,1001) or step%25 == 0)
            duration = time.monotonic()-before; update_seconds += duration
            seen.extend(ids); samples += sum(c['valid_scored_samples'] for c in crops)
            record = {'step':step,'updates':step,'global_step':step,**values,'source_ids':ids,'unique_sources':len(seen),
                'audio_hours':samples/48000/3600,'step_seconds':duration,'elapsed_seconds':time.monotonic()-started,
                'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,step)
            if step == 1 or step%25 == 0:
                base.event('grail_candidate_progress',step=step,target=UPDATES,total=values['total'],sources=len(seen),
                    training_update_seconds=update_seconds,elapsed_seconds=record['elapsed_seconds'])
            if step in REVIEW_STEPS:
                report = evaluate(step); reports[step] = report
                monitor.log_validation(report,step,references.get(step),b_references.get(step)); data.assert_unchanged()
                if step in CHECKPOINT_STEPS:
                    last_receipt = save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,seen,identity,selection,samples)
                    last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                    base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                base.write_json(args.out/f'review-step{step}.json',{'step':step,'source_count':len(seen),
                    'aggregate':report['aggregate'],'quiet_regions':report['quiet_regions'],
                    'original_reference_aggregate':references[step]['aggregate'] if step in references else None,
                    'b_reference_aggregate':b_references[step]['aggregate'] if step in b_references else None,
                    'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                    'elapsed_seconds':time.monotonic()-started,'automatic_next_cut':False})
                base.event('grail_candidate_review_ready',step=step,quality=report['aggregate'],automatic_next_cut=False)
        status = 'awaiting_review'
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        monitor.close()
        preserved = screen.frozen_versions(model) == frozen and resume.continuation.teacher_versions(teacher) == teacher_frozen
        preserved = preserved and control.state_hash(teacher.model) == teacher_hash
        data.assert_unchanged(); files = all(base.sha(path) == checksum for path,checksum in protected.items())
        base.write_json(args.out/'completed.json',{'version':VERSION,'status':status,'failure':failure,
            'step':len(seen)//12,'source_count':len(seen),'scored_samples':samples,
            'last_checkpoint_sha256':last_receipt['checkpoint_sha256'] if last_receipt else None,
            'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
            'elapsed_seconds':time.monotonic()-started,'frozen_state_preserved':preserved,'original_files_preserved':files,
            'review_steps_completed':list(reports),'optimizer_reset_after_start':False,'automatic_next_cut':False,
            'automatic_promotion':False,'final':reports[max(reports)]['aggregate'] if reports else None})
        if not preserved or not files: raise RuntimeError('Fresh candidate recovery changed protected original/B state')


if __name__ == '__main__': main()
