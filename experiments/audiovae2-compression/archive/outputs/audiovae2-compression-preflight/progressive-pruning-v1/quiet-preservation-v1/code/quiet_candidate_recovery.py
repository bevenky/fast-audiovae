"""Independent 2,000-update quiet-constrained recovery from the fresh combined initializer.

Historical checkpoints authenticate RNG, empty Adam settings and source order.
Their learned or sliced group tensors are never installed into this candidate.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import torch
import reconstruction_b_recovery as b
import combined_recovery as combined
import quiet_projected_update as quiet

base,screen,replay,resume,joint = b.base,b.screen,b.replay,b.resume,b.joint
prior,progressive,control = b.prior,b.progressive,b.control
VERSION = 'audiovae2_quiet_candidate_recovery_v1'
COMBINED2500_SHA = '93c1b49bed08a8626f0289b4ca096f94df5cac588fdde7f7488a365f065ad5e0'
OPERATOR_PATHS = b.OPERATOR_PATHS
ACCUMULATION,UPDATES = 12,2000
REVIEW_STEPS = (0,250,500,1000,1500,2000)
CHECKPOINT_STEPS = (0,1000,2000)
perform_update = quiet.perform_update


def accumulate_update_totals(totals,values):
    """Count actual optimizer batches separately from accepted weight movement."""
    def number(value,name):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonnumeric update accounting: '+name)
        return value
    flags={name:number(values.get('q_'+name),name) for name in ('projected','zero_displacement')}
    fraction=number(values.get('q_accepted_fraction'),'accepted_fraction')
    times={name:number(values.get('q_'+name),name) for name in ('auxiliary_seconds','ordinary_update_seconds')}
    if any(value not in (0,1) for value in flags.values()) or fraction not in quiet.FRACTIONS or any(v<0 for v in times.values()):
        raise ValueError('Invalid projection flag, fixed backtracking fraction or runtime')
    if fraction==0 and flags['zero_displacement']!=1:
        raise ValueError('An exact zero fraction must report unchanged parameter values')
    defaults={'updates':0,'projected_updates':0,'zero_displacements':0,'backtracking_updates':0,
              'auxiliary_seconds':0.,'ordinary_seconds':0.}
    if totals and set(totals)!=set(defaults):raise ValueError('Update totals schema changed')
    current={**defaults,**totals}
    for name,value in current.items():
        number(value,name)
        if value<0 or (name not in ('auxiliary_seconds','ordinary_seconds') and (type(value)is not int or value>current['updates'])):
            raise ValueError('Invalid previous update accounting: '+name)
    updated={**current,'updates':current['updates']+1,
        'projected_updates':current['projected_updates']+int(flags['projected']),
        'zero_displacements':current['zero_displacements']+int(flags['zero_displacement']),
        'backtracking_updates':current['backtracking_updates']+int(fraction<1),
        'auxiliary_seconds':current['auxiliary_seconds']+times['auxiliary_seconds'],
        'ordinary_seconds':current['ordinary_seconds']+times['ordinary_update_seconds']}
    if any(not math.isfinite(v) for v in updated.values()):raise ValueError('Update accounting overflow')
    totals.clear();totals.update(updated)
    return totals


def restore_quiet_start(model,teacher,zero,b_artifact,b_receipt,artifact,receipt,calibration_ids,*,artifact_sha256):
    """Fresh teacher factory -> B native fits -> C native fits; never load a group checkpoint."""
    if (zero.get('cut_updates') != 0 or zero.get('global_updates') != 0
            or zero.get('sources_seen') != [] or zero.get('optimizer',{}).get('state') != {}
            or model.selections != zero.get('selection')):
        raise ValueError('Quiet recovery requires the original selection, zero updates and empty Adam')
    teacher_hash=control.state_hash(teacher.model.decoder)
    if b_receipt.get('original_step0_pristine',{}).get('teacher_state_sha256') != teacher_hash:
        raise ValueError('Quiet initialization used a different original frozen teacher')
    b_install=b.install_b_operators(model,b_artifact,b_receipt,artifact_sha256=b.B_SHA,
        step0_sha256=b.STEP0_SHA,calibration_ids=calibration_ids)
    c_install=combined.install_combined_operators(model,artifact,receipt,b_artifact,
        artifact_sha256=artifact_sha256,b_operators_sha256=b.B_SHA,calibration_ids=calibration_ids)
    optimizer=progressive.fresh_optimizer(model,lr=3e-5)
    screen.restore_rng(zero['rng'])
    if (optimizer.state or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
            or not replay.compare_tree(screen.rng_state(),zero['rng'])['equal']
            or control.state_hash(teacher.model.decoder) != teacher_hash):
        raise RuntimeError('Quiet recovery changed fresh Adam, original RNG or frozen teacher')
    return optimizer,{'b_installation':b_install,'combined_installation':c_install,
        'old_group_state_loaded':False,'original_step0_rng_restored_exactly':True,'fresh_adam_matches_original':True}


def authenticate_inputs(args,data,pools):
    values=combined.authenticate_inputs(args,data,pools)
    zero,b_artifact,b_receipt,artifact,receipt,all_sources,recipe,reference,references,_,protected=values
    source_ids=tuple(all_sources[:UPDATES*ACCUMULATION])
    folder=args.combined_run
    done=json.loads((folder/'completed.json').read_text())
    launch=json.loads((folder/'launch.json').read_text())
    journal=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
    endpoint=folder/'checkpoint-step2500.pt'
    endpoint_receipt=json.loads(endpoint.with_suffix('.json').read_text())
    if (done.get('version') != combined.VERSION or done.get('status') != 'awaiting_review'
            or done.get('step') != 2500 or done.get('source_count') != 30000
            or done.get('frozen_state_preserved') is not True or done.get('original_files_preserved') is not True
            or base.sha(endpoint) != COMBINED2500_SHA or done.get('last_checkpoint_sha256') != COMBINED2500_SHA
            or endpoint_receipt.get('checkpoint_sha256') != COMBINED2500_SHA
            or launch.get('initial_decoder_state_sha256') != args.combined_state_sha256
            or launch.get('combined_operators_sha256') != args.combined_artifact_sha256
            or launch.get('selection') != zero['selection']
            or [r['step'] for r in journal] != list(range(1,2501))
            or [sid for r in journal for sid in r['source_ids']] != list(all_sources)
            or launch.get('source_ids') != list(all_sources)
            or len(source_ids) != 24000 or len(set(source_ids)) != 24000):
        raise ValueError('Combined comparator is incomplete or differs in initializer/source exposure')
    prior.validate_recipe(launch)
    c_references={step:json.loads((folder/f'development-step{step}.json').read_text()) for step in REVIEW_STEPS}
    if c_references[0]['aggregate'] != reference['aggregate']:
        raise ValueError('Combined recovery baseline differs from the sealed initializer')
    c2000=folder/'checkpoint-step2000.pt';c2000_receipt=json.loads(c2000.with_suffix('.json').read_text())
    if (base.sha(c2000) != c2000_receipt['checkpoint_sha256']
            or c2000_receipt.get('quality') != c_references[2000]['aggregate']):
        raise ValueError('Matched combined2000 quality differs from its checkpoint receipt')
    paths=[folder/name for name in ('launch.json','completed.json','train.jsonl','checkpoint-step2500.pt',
        'checkpoint-step2500.json','checkpoint-step2000.pt','checkpoint-step2000.json')]
    paths += [folder/f'development-step{step}.json' for step in REVIEW_STEPS]
    protected.update(launch['protected'])
    protected.update({str(path.resolve()):base.sha(path) for path in paths})
    if any(base.sha(path) != checksum for path,checksum in protected.items()):
        raise ValueError('Protected original/B/combined input changed')
    return zero,b_artifact,b_receipt,artifact,receipt,source_ids,recipe,reference,references,c_references,protected


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
        raise ValueError('B checkpoint source ledger or selection differs')
    params = base.parameters(model)
    if (not params or (set(optimizer.state) != set(params) if step else bool(optimizer.state))
            or any(float(s['step']) != step for s in optimizer.state.values())):
        raise ValueError('B checkpoint Adam counters/scope differ from uninterrupted training')
    for row in optimizer.param_groups:
        if (row['lr'] != 3e-5 or row['betas'] != (.9,.99) or row['eps'] != 1e-8 or row['weight_decay'] != 0
                or set(row['params']) != set(params)):
            raise ValueError('B checkpoint optimizer settings changed')
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
    import quiet_candidate_monitor as monitor_module
    import group_model
    import fresh_training_data
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0','original2000','preserved5000','b-dir','b-run','combined-dir','combined-run','manifest','source-plan',
                 'shards','assets','recipe','schedule','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--combined-artifact-sha256',required=True)
    parser.add_argument('--combined-state-sha256',required=True)
    parser.add_argument('--tensorboard',type=Path)
    parser.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = parser.parse_args()
    logdir = (args.tensorboard or args.out/'tensorboard').resolve()
    if args.out.exists() or logdir.exists(): raise FileExistsError('Use new fresh candidate recovery output and event directories')
    retained = (args.step0.parent,args.original2000.parent,args.preserved5000.parent,args.b_dir,args.b_run,args.combined_dir,args.combined_run,args.recipe.parent)
    if any(new.is_relative_to(old.resolve()) or old.resolve().is_relative_to(new)
           for new in (args.out.resolve(),logdir) for old in retained):
        raise ValueError('Fresh candidate output would overlap a preserved run')
    base.policy()
    manifest,pools,_ = base.load_data(args.manifest)
    if len(pools['calibration']) != 72 or len(pools['development']) != 96:
        raise ValueError('Use the original72 calibration and96 development sources')
    fresh = fresh_training_data.FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    data = prior.SourceStream(pools,fresh)
    zero,b_artifact,b_receipt,artifact,receipt,source_ids,recipe,reference,references,c_references,protected = authenticate_inputs(args,data,pools)
    modules = (b,combined,quiet,prior,progressive,control,base,screen,replay,joint,group_model,fresh_training_data,monitor_module)
    protected.update({str(Path(m.__file__).resolve()):base.sha(m.__file__) for m in modules})
    protected[str(Path(__file__).resolve())] = base.sha(__file__)
    process = control.gpu_idle_snapshot()
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise ValueError('Use the original qualified Torch/cuDNN runtime')
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    selection = artifact['selection']
    model = progressive.initialize_from_teacher(teacher.model.decoder,selection)
    optimizer,installation = restore_quiet_start(model,teacher,zero,b_artifact,b_receipt,artifact,receipt,
        [c['source_id'] for c in pools['calibration']],artifact_sha256=args.combined_artifact_sha256)
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
    identity = {'version':VERSION,'initialization':'Original teacher -> original selected factory -> B native fits -> combined native startup fit',
        'original_step0_sha256':b.STEP0_SHA,'reference_b_operators_sha256':b.B_SHA,'candidate_operators_sha256':args.combined_artifact_sha256,
        'initial_decoder_state_sha256':args.combined_state_sha256,'original2000_sha256':b.ORIGINAL2000_SHA,
        'preserved5000_sha256':b.STEP5000_SHA,'combined_recovery_checkpoint_sha256':COMBINED2500_SHA,'projected_update_method':quiet.VERSION,
        'constraint_scope':'Current batch only; no global startup retention guarantee','adam_moments':'Advance once per original ordinary gradient batch',
        'zero_displacement_allowed':True,'new_inference_operations':0,
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
        'new_startup_training_loss':False,'no_quality_based_early_stopping':True,'historical_bitwise_trajectory_claim':False}
    base.write_json(args.out/'launch.json',identity)
    seen = []; samples = 0; warmed = set(); reports = {}; last_receipt = None
    started = time.monotonic(); update_seconds = validation_seconds = waiting_seconds = 0.
    failure = None; status = 'running'
    update_totals={'updates':0,'projected_updates':0,'zero_displacements':0,'backtracking_updates':0,
                   'auxiliary_seconds':0.,'ordinary_seconds':0.}
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
            'candidate_state_exact':control.state_hash(model.decoder) == args.combined_state_sha256}
        base.write_json(args.out/'initial-parity.json',comparison)
        if (not comparison['quality']['passed'] or not comparison['rng']['equal']
                or not comparison['optimizer']['equal'] or not comparison['candidate_state_exact']):
            raise RuntimeError('Fresh candidate full baseline, native state, original RNG or fresh Adam failed to reproduce')
        reports[0] = initial; monitor.log_validation(initial,0,references[0],c_references[0])
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
            accumulate_update_totals(update_totals,values)
            if update_totals['updates']!=step:raise RuntimeError('Optimizer and displacement accounting steps differ')
            record = {'step':step,'updates':step,'global_step':step,**values,'source_ids':ids,'unique_sources':len(seen),
                'audio_hours':samples/48000/3600,'step_seconds':duration,'elapsed_seconds':time.monotonic()-started,
                'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,step)
            if step == 1 or step%25 == 0:
                base.event('quiet_candidate_progress',step=step,target=UPDATES,total=values['total'],sources=len(seen),
                    training_update_seconds=update_seconds,elapsed_seconds=record['elapsed_seconds'])
            if step in REVIEW_STEPS:
                report = evaluate(step); reports[step] = report
                monitor.log_validation(report,step,references.get(step),c_references.get(step)); data.assert_unchanged()
                if step in CHECKPOINT_STEPS:
                    last_receipt = save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,seen,identity,selection,samples)
                    last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                    base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                base.write_json(args.out/f'review-step{step}.json',{'step':step,'source_count':len(seen),
                    'aggregate':report['aggregate'],'quiet_regions':report['quiet_regions'],
                    'original_reference_aggregate':references[step]['aggregate'] if step in references else None,
                    'combined_reference_aggregate':c_references[step]['aggregate'] if step in c_references else None,
                    'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
                    'elapsed_seconds':time.monotonic()-started,'automatic_next_cut':False})
                base.event('quiet_candidate_review_ready',step=step,quality=report['aggregate'],automatic_next_cut=False)
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
            'automatic_promotion':False,'projected_updates':update_totals['projected_updates'],'zero_displacements':update_totals['zero_displacements'],
            'accepted_nonzero_displacements':update_totals['updates']-update_totals['zero_displacements'],
            'backtracking_updates':update_totals['backtracking_updates'],
            'constraint_auxiliary_seconds':update_totals['auxiliary_seconds'],'ordinary_update_seconds':update_totals['ordinary_seconds'],
            'final':reports[max(reports)]['aggregate'] if reports else None})
        if not preserved or not files: raise RuntimeError('Fresh candidate recovery changed protected original/B state')


if __name__ == '__main__': main()
