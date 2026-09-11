"""Recover a sealed combined B/startup initialization for exactly2,500 updates.

Original teacher, widths, reconstruction objective and first30,000 executed
sources stay fixed. Startup feasibility is observed, never a new training loss.
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
VERSION = 'audiovae2_combined_recovery_v1'
INIT_VERSION = 'audiovae2_combined_startup_initialization_v1'
ACCUMULATION,UPDATES = 12,2500
REVIEW_STEPS = (0,250,500,1000,1500,2000,2500)
CHECKPOINT_STEPS = (0,1000,2000,2500)
OPERATOR_PATHS = b.OPERATOR_PATHS
B2000_SHA = '43ef31f30cca751167cf91152f3ab039592b06860f0ffbd60531c8862c783954'
perform_update = prior.perform_update


def install_combined_operators(model,artifact,receipt,b_artifact,*,artifact_sha256,b_operators_sha256,calibration_ids):
    """Install only on authenticated, untrained B; validate every tensor first."""
    if (artifact.get('format') != INIT_VERSION or artifact.get('variant') != 'combined_B_startup'
            or artifact.get('base_step0_sha256') != b.STEP0_SHA
            or artifact.get('b_operators_sha256') != b_operators_sha256
            or artifact.get('ridge') != 1e-6 or artifact.get('interface_atol') != 1e-5
            or artifact.get('interface_rtol') != 1e-4
            or artifact.get('selection') != model.selections
            or artifact.get('fit_source_ids') != list(calibration_ids)
            or len(calibration_ids) != 72 or len(set(calibration_ids)) != 72
            or artifact.get('automatic_promotion') is not False
            or receipt.get('operators_sha256') != artifact_sha256
            or receipt.get('base_b_state_sha256') != control.state_hash(model.decoder)
            or receipt.get('native_fp32_constraints_passed') is not True
            or receipt.get('stage2_b_operators_preserved') is not True
            or receipt.get('unchanged_frozen_and_unselected_tensors') is not True
            or receipt.get('changed_native_paths') != list(OPERATOR_PATHS)
            or receipt.get('all9_residual_units_preserved') is not True
            or receipt.get('neural_training_updates') != 0 or receipt.get('extra_inference_modules') != 0
            or receipt.get('automatic_promotion') is not False
            or set(artifact.get('operators',{})) != set(OPERATOR_PATHS)):
        raise ValueError('Combined artifact, B origin, native scope or calibration constraints differ')
    state = model.decoder.state_dict(); proposed = state.copy(); modules = {}
    for path in OPERATOR_PATHS:
        module = model.decoder.get_submodule(path); supplied = artifact['operators'][path]
        expected = module.state_dict()
        if set(supplied) != set(expected): raise ValueError('Combined native state keys differ: '+path)
        for key,value in supplied.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
                raise ValueError('Combined native tensor geometry/dtype/finiteness differs: '+path+'.'+key)
            if path != OPERATOR_PATHS[-1] and not torch.equal(value.cpu(),b_artifact['operators'][path][key].cpu()):
                raise ValueError('Combined initialization changed a preserved B stage2 operator')
            proposed[path+'.'+key] = value
        modules[path] = module
    expected_hash = b._state_hash(proposed)
    if receipt.get('candidate_state_sha256') != expected_hash:
        raise ValueError('Combined coefficients do not reconstruct the sealed full candidate')
    objects = [(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
    for path,module in modules.items(): module.load_state_dict(artifact['operators'][path],strict=True)
    if (objects != [(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
            or control.state_hash(model.decoder) != expected_hash):
        raise RuntimeError('Combined installation changed topology or failed native state identity')
    return {'candidate_state_sha256':expected_hash,'parameter_objects_preserved':True,
            'stage2_b_operators_preserved':True,'only_stage3_up_changed_relative_to_b':True,
            'fits_repeated':False,'optimizer_updates':0}


def restore_combined_start(model,teacher,zero,b_artifact,b_receipt,artifact,receipt,calibration_ids,
                           *,artifact_sha256,b_operators_sha256=b.B_SHA):
    optimizer,b_restore = b.restore_b_start(model,teacher,zero,b_artifact,b_receipt,calibration_ids,
                                          artifact_sha256=b_operators_sha256)
    installed = install_combined_operators(model,artifact,receipt,b_artifact,artifact_sha256=artifact_sha256,
        b_operators_sha256=b_operators_sha256,calibration_ids=calibration_ids)
    screen.restore_rng(zero['rng'])
    if (optimizer.state or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
            or not replay.compare_tree(screen.rng_state(),zero['rng'])['equal']):
        raise RuntimeError('Combined start changed original fresh Adam or step0 RNG')
    return optimizer,{'b_restoration':b_restore,'combined_installation':installed,
                      'original_rng_and_fresh_adam_exact':True}


def validate_source_schedule(zero,original2000,original5000,b2000,source_ids):
    first24k = b.validate_source_schedule(zero,original2000,source_ids)
    chosen = tuple(source_ids[:UPDATES*ACCUMULATION])
    if (len(chosen) != 30000 or len(set(chosen)) != 30000
            or original5000.get('format') != 'audiovae2_progressive_same_cut_5000_v1'
            or original5000.get('cut_index') != 1 or original5000.get('cut_updates') != 5000
            or original5000.get('global_updates') != 5000
            or len(original5000.get('sources_seen',[])) != 60000
            or len(set(original5000['sources_seen'])) != 60000
            or original5000['sources_seen'][:30000] != list(chosen)
            or original5000.get('original_cut_identity') != zero.get('identity')
            or original5000.get('selection') != zero['selection']
            or original5000.get('schedule_sha256') != zero['schedule_sha256']
            or original5000.get('accumulation') != ACCUMULATION
            or original5000.get('coefficients') != prior.COEFFICIENTS
            or b2000.get('format') != b.VERSION or b2000.get('step') != 2000
            or b2000.get('sources_seen') != list(first24k) or b2000.get('selection') != zero['selection']
            or b2000.get('schedule_sha256') != zero['schedule_sha256']
            or b2000.get('accumulation') != ACCUMULATION or b2000.get('coefficients') != prior.COEFFICIENTS
            or b2000.get('optimizer_reset_after_start') is not False):
        raise ValueError('Combined first30,000 sources do not match original5000 and B2000 executed ledgers')
    for payload in (original5000,b2000):
        if (payload.get('teacher_source_sha256') != base.SOURCE_SHA256
                or payload.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256):
            raise ValueError('Original/B source ledger used a different teacher')
    return chosen


def save_checkpoint(path,model,optimizer,step,source_ids,identity,selection,scored_samples):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError('Only step0/1000/2000/2500 checkpoint states are authorized')
    if (len(identity['source_ids']) != 30000 or len(set(identity['source_ids'])) != 30000
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']
            or list(source_ids) != identity['source_ids'][:step*12] or len(source_ids) != step*12
            or selection != identity['selection']):
        raise ValueError('Combined checkpoint source ledger or selection differs')
    params = base.parameters(model)
    if (not params or (set(optimizer.state) != set(params) if step else bool(optimizer.state))
            or any(float(s['step']) != step for s in optimizer.state.values())):
        raise ValueError('Combined checkpoint Adam counters/scope differ from uninterrupted training')
    for row in optimizer.param_groups:
        if (row['lr'] != 3e-5 or row['betas'] != (.9,.99) or row['eps'] != 1e-8 or row['weight_decay'] != 0
                or set(row['params']) != set(params)):
            raise ValueError('Combined checkpoint optimizer recipe changed')
    if type(scored_samples) is not int or (scored_samples <= 0 if step else scored_samples != 0):
        raise ValueError('Missing scored audio accounting')
    payload = {'format':VERSION,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'step':step,'cut_index':1,'cut_updates':step,
        'global_updates':step,'selection':selection,'schedule_sha256':identity['schedule_sha256'],
        'sources_seen':list(source_ids),'scored_samples':scored_samples,'coefficients':dict(prior.COEFFICIENTS),
        'accumulation':12,'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'identity':identity,'optimizer_reset_after_start':False,'automatic_next_cut':False,'automatic_promotion':False}
    estimate = sum(v.numel()*v.element_size() for v in payload['group'].values())*(3 if step else 1)+8*1024**2
    if shutil.disk_usage(path.parent).free < estimate+32*1024**2:
        raise RuntimeError('Insufficient disk for group and continuous optimizer checkpoint')
    tmp = path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
    return {'checkpoint_sha256':base.sha(path),'step':step,'sources_seen':len(source_ids),
            'optimizer_and_rng_saved':True,'optimizer_reset_after_start':False,'bytes':path.stat().st_size}


def authenticate_inputs(args,data,pools):
    zero,b_artifact,b_receipt,_,recipe,_,references,protected = b.authenticate_inputs(args,data,pools)
    original2000 = torch.load(args.original2000,map_location='cpu',weights_only=True,mmap=True)
    original5000 = torch.load(args.preserved5000,map_location='cpu',weights_only=True,mmap=True)
    b_path = args.b_run/'checkpoint-step2000.pt'
    if base.sha(b_path) != B2000_SHA: raise ValueError('Use the preserved completed B2000 reference')
    b_complete = json.loads((args.b_run/'completed.json').read_text())
    b_checkpoint_receipt = json.loads(b_path.with_suffix('.json').read_text())
    b_launch = json.loads((args.b_run/'launch.json').read_text())
    b2000 = torch.load(b_path,map_location='cpu',weights_only=True,mmap=True)
    if (b_complete.get('status') != 'awaiting_review' or b_complete.get('step') != 2000
            or b_complete.get('source_count') != 24000 or b_complete.get('frozen_state_preserved') is not True
            or b_complete.get('original_files_preserved') is not True
            or b_complete.get('last_checkpoint_sha256') != B2000_SHA
            or b_checkpoint_receipt.get('checkpoint_sha256') != B2000_SHA
            or b2000.get('identity') != b_launch or b_launch.get('b_operators_sha256') != b.B_SHA
            or b_launch.get('initial_decoder_state_sha256') != b.B_STATE_SHA):
        raise ValueError('B reference recovery is incomplete or has different initialization lineage')
    prior.validate_recipe(b_launch)
    chosen = validate_source_schedule(zero,original2000,original5000,b2000,data.source_ids)
    if (original5000['identity'].get('source_plan_identity_sha256') != data.fresh.identity
            or original5000['identity'].get('source_ids',[])[:6000] != list(chosen[24000:])):
        raise ValueError('Original2500 used a different source plan or source prefix')
    final_complete = json.loads((args.preserved5000.parent/'completed.json').read_text())
    if (final_complete.get('status') != 'awaiting_review' or final_complete.get('step') != 5000
            or final_complete.get('last_checkpoint_sha256') != b.STEP5000_SHA
            or final_complete.get('frozen_state_preserved') is not True
            or final_complete.get('original_files_preserved') is not True):
        raise ValueError('Original5000 is not an authenticated completed reference')
    original_rows = [json.loads(line) for line in (args.preserved5000.parent/'train.jsonl').read_text().splitlines() if line.strip()]
    b_rows = [json.loads(line) for line in (args.b_run/'train.jsonl').read_text().splitlines() if line.strip()]
    if [r.get('step') for r in original_rows] != list(range(2001,5001)) or [r.get('step') for r in b_rows] != list(range(1,2001)):
        raise ValueError('Original/B executed training journals are incomplete or duplicate steps')
    for row in original_rows[:500]+b_rows:
        if row.get('source_ids') != list(chosen[(row['step']-1)*12:row['step']*12]):
            raise ValueError('Original/B source order differs at an executed update')
    artifact_path = args.combined_dir/'combined-native-operators.pt'
    if base.sha(artifact_path) != args.combined_artifact_sha256:
        raise ValueError('Combined native artifact differs from the explicitly approved SHA')
    artifact = torch.load(artifact_path,map_location='cpu',weights_only=True)
    receipt = json.loads((args.combined_dir/'combined-receipt.json').read_text())
    complete = json.loads((args.combined_dir/'completed.json').read_text())
    launch = json.loads((args.combined_dir/'launch.json').read_text())
    calibration_ids = [c['source_id'] for c in pools['calibration']]
    development_ids = [c['source_id'] for c in pools['development']]
    if (complete.get('version') != INIT_VERSION or complete.get('complete') is not True
            or complete.get('status') != 'evaluated' or complete.get('failure') is not None
            or complete.get('files_preserved') is not True or not complete.get('states_preserved')
            or not all(complete['states_preserved'].values())
            or complete.get('native_fp32_constraints_passed') is not True
            or complete.get('calibration_sources') != 72 or complete.get('development_sources') != 96
            or complete.get('neural_training_updates') != 0 or complete.get('candidate_receipt') != receipt
            or receipt.get('candidate_state_sha256') != args.combined_state_sha256
            or receipt.get('base_b_state_sha256') != b.B_STATE_SHA
            or receipt.get('full_group_widths') != [384,256,128]
            or launch.get('fit_source_ids') != calibration_ids or launch.get('development_source_ids') != development_ids
            or launch.get('selection') != zero['selection'] or launch.get('b_operators_sha256') != b.B_SHA):
        raise ValueError('Combined startup fit is infeasible, incomplete, or belongs to a different candidate/partition')
    reference = json.loads((args.combined_dir/'combined-development.json').read_text())
    if complete['results']['combined']['aggregate'] != reference['aggregate']:
        raise ValueError('Combined baseline differs from completed initializer')
    references[2500] = json.loads((args.preserved5000.parent/'development-step2500.json').read_text())
    b_references = {step:json.loads((args.b_run/f'development-step{step}.json').read_text()) for step in b.REVIEW_STEPS}
    if b_complete['final'] != b_references[2000]['aggregate']:
        raise ValueError('B matched2000 reference differs from its completed run')
    protected = {**protected,**original5000['identity']['protected'],**b_launch['protected'],**launch['protected']}
    # Completed B outputs, including its own group checkpoints, remain read-only.
    paths = [p for p in args.b_run.rglob('*') if p.is_file()]
    paths += [args.preserved5000.parent/name for name in ('completed.json','launch.json','train.jsonl','development-step2500.json')]
    paths += [args.combined_dir/name for name in ('combined-native-operators.pt','combined-receipt.json',
                                                'combined-development.json','launch.json','completed.json')]
    protected.update({str(path.resolve()):base.sha(path) for path in paths})
    for path,checksum in protected.items():
        if base.sha(path) != checksum: raise ValueError('Protected original/B/combined input changed: '+path)
    return zero,b_artifact,b_receipt,artifact,receipt,chosen,recipe,reference,references,b_references,protected


def main():
    import combined_recovery_monitor as monitor_module
    import group_model
    import fresh_training_data
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0','original2000','preserved5000','b-dir','b-run','combined-dir','manifest','source-plan',
                 'shards','assets','recipe','schedule','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--combined-artifact-sha256',required=True)
    parser.add_argument('--combined-state-sha256',required=True)
    parser.add_argument('--tensorboard',type=Path)
    parser.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = parser.parse_args()
    logdir = (args.tensorboard or args.out/'tensorboard').resolve()
    if args.out.exists() or logdir.exists(): raise FileExistsError('Use new combined recovery output and event directories')
    retained = (args.step0.parent,args.original2000.parent,args.preserved5000.parent,args.b_dir,args.b_run,args.combined_dir,args.recipe.parent)
    if any(new.is_relative_to(old.resolve()) or old.resolve().is_relative_to(new)
           for new in (args.out.resolve(),logdir) for old in retained):
        raise ValueError('Combined output would overlap a preserved run')
    base.policy()
    manifest,pools,_ = base.load_data(args.manifest)
    if len(pools['calibration']) != 72 or len(pools['development']) != 96:
        raise ValueError('Use the original72 calibration and96 development sources')
    fresh = fresh_training_data.FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    data = prior.SourceStream(pools,fresh)
    zero,b_artifact,b_receipt,artifact,receipt,source_ids,recipe,reference,references,b_references,protected = authenticate_inputs(args,data,pools)
    modules = (b,prior,progressive,control,base,screen,replay,joint,group_model,fresh_training_data,monitor_module)
    protected.update({str(Path(m.__file__).resolve()):base.sha(m.__file__) for m in modules})
    protected[str(Path(__file__).resolve())] = base.sha(__file__)
    process = control.gpu_idle_snapshot()
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise ValueError('Use the original qualified Torch/cuDNN runtime')
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model = progressive.initialize_from_teacher(teacher.model.decoder,zero['selection'])
    optimizer,installation = restore_combined_start(model,teacher,zero,b_artifact,b_receipt,artifact,receipt,
        [c['source_id'] for c in pools['calibration']],artifact_sha256=args.combined_artifact_sha256)
    if len(base.parameters(model)) != 90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Expected all90 joint group parameters and the original frozen teacher')
    frozen = screen.frozen_versions(model); teacher_frozen = resume.continuation.teacher_versions(teacher)
    teacher_hash = control.state_hash(teacher.model)
    common = base.objective(); metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    evaluator = UnifiedMonitor(None,{},metadata)
    boundary_panel,boundary_selection = base.select_boundary_panel(pools['development'],metadata)
    parameter_bytes = sum(v.numel()*v.element_size() for v in model.group_state_dict().values())
    required = parameter_bytes*10+64*1024**2
    ancestor = args.out.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < required: raise RuntimeError('Insufficient disk for group checkpoints0/1000/2000/2500 and compact logs')
    args.out.mkdir(parents=True)
    monitor = monitor_module.RecoveryMonitor(logdir,source_metadata=metadata)
    screen.restore_rng(zero['rng'])
    identity = {'version':VERSION,'initialization':'Original step0 -> sealed B -> sealed startup refit on actual B inputs',
        'original_step0_sha256':b.STEP0_SHA,'b_operators_sha256':b.B_SHA,'combined_operators_sha256':args.combined_artifact_sha256,
        'initial_decoder_state_sha256':args.combined_state_sha256,'original2000_sha256':b.ORIGINAL2000_SHA,
        'preserved5000_sha256':b.STEP5000_SHA,'b_recovery_checkpoint_sha256':base.sha(args.b_run/'checkpoint-step2000.pt'),
        'selection':zero['selection'],'schedule_sha256':base.sha(args.schedule),'coefficients':dict(prior.COEFFICIENTS),
        'learning_rate':3e-5,'optimizer_betas':[.9,.99],'optimizer_eps':1e-8,'weight_decay':0,
        'execution_batch_size':1,'gradient_accumulation':12,'trainable_stages':[2,3,4],
        'source_interval':[0,30000],'source_ids':list(source_ids),'source_ids_sha256':screen.digest(list(source_ids)),
        'source_plan_identity_sha256':fresh.identity,'protected':protected,'optimizer_reset_after_start':False,
        'original_teacher_forever':True,'rng_origin':'Exact original step0 RNG restored after installation/monitor setup',
        'original_policy_seed':20260910,'backend':replay.backend_state(),'torch':str(torch.__version__),
        'cudnn':torch.backends.cudnn.version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'precision':'FP32 TF32 disabled','process_snapshot':process,'installation':installation,
        'review_steps':list(REVIEW_STEPS),'checkpoint_steps':list(CHECKPOINT_STEPS),'target_updates':UPDATES,
        'tensorboard':str(logdir),'disk_free_before':free,'estimated_checkpoints_and_logs_bytes':required,
        'boundary_panel':boundary_selection,'automatic_next_cut':False,'automatic_promotion':False,
        'new_startup_training_loss':False,'no_quality_based_early_stopping':True,'historical_bitwise_trajectory_claim':False}
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
            boundary = base.evaluate_boundaries(teacher,model,boundary_panel,zero['selection'],base.batch,base.teacher_forward)
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
            'candidate_state_exact':control.state_hash(model.decoder) == args.combined_state_sha256}
        base.write_json(args.out/'initial-parity.json',comparison)
        if (not comparison['quality']['passed'] or not comparison['rng']['equal']
                or not comparison['optimizer']['equal'] or not comparison['candidate_state_exact']):
            raise RuntimeError('Combined full baseline, native state, original RNG or fresh Adam failed to reproduce')
        reports[0] = initial; monitor.log_validation(initial,0,references[0],b_references[0])
        last_receipt = save_checkpoint(args.out/'checkpoint-step0.pt',model,optimizer,0,seen,identity,zero['selection'],samples)
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
            values,checks = perform_update(model,teacher,crops,common,optimizer,diagnostics=step in (1,1001,2001) or step%25 == 0)
            duration = time.monotonic()-before; update_seconds += duration
            seen.extend(ids); samples += sum(c['valid_scored_samples'] for c in crops)
            record = {'step':step,'updates':step,'global_step':step,**values,'source_ids':ids,'unique_sources':len(seen),
                'audio_hours':samples/48000/3600,'step_seconds':duration,'elapsed_seconds':time.monotonic()-started,
                'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,step)
            if step == 1 or step%25 == 0:
                base.event('combined_recovery_progress',step=step,target=UPDATES,total=values['total'],sources=len(seen),
                    training_update_seconds=update_seconds,elapsed_seconds=record['elapsed_seconds'])
            if step in REVIEW_STEPS:
                report = evaluate(step); reports[step] = report
                monitor.log_validation(report,step,references.get(step),b_references.get(step)); data.assert_unchanged()
                if step in CHECKPOINT_STEPS:
                    last_receipt = save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,seen,identity,zero['selection'],samples)
                    last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                    base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                base.write_json(args.out/f'review-step{step}.json',{'step':step,'source_count':len(seen),
                    'aggregate':report['aggregate'],'quiet_regions':report['quiet_regions'],
                    'original_reference_aggregate':references[step]['aggregate'] if step in references else None,
                    'b_reference_aggregate':b_references[step]['aggregate'] if step in b_references else None,
                    'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                    'elapsed_seconds':time.monotonic()-started,'automatic_next_cut':False})
                base.event('combined_recovery_review_ready',step=step,quality=report['aggregate'],automatic_next_cut=False)
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
        if not preserved or not files: raise RuntimeError('Combined recovery changed protected original/B state')


if __name__ == '__main__': main()
