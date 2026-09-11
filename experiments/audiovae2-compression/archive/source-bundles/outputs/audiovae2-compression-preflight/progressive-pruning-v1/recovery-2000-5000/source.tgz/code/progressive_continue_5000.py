"""Continue the same384/256 cut from2000 to5000 with original AdamW state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import progressive_train as prior
import progressive_continue as previous
import progressive_model as model_api
import fresh_training_data as fresh_module
from compare_accumulation_v1 import numeric_reference_check
from unified_monitor import UnifiedMonitor

base,screen,replay,resume,joint = prior.base,prior.screen,prior.replay,prior.resume,prior.joint
VERSION = 'audiovae2_progressive_same_cut_5000_v1'
PARENT_SHA256 = 'a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f'
START,STOP,EVERY,ACCUMULATION = 2000,5000,500,12
START_CURSOR,STOP_CURSOR = START*ACCUMULATION,STOP*ACCUMULATION


def source_window(parent, source_ids):
    if (parent.get('cut_index') != 1 or parent.get('cut_updates') != START
            or parent.get('global_updates') != START or parent.get('accumulation') != ACCUMULATION
            or parent.get('format') != previous.VERSION or parent.get('optimizer_reset') is not False
            or parent.get('sources_seen') != list(source_ids[:START_CURSOR])
            or len(parent.get('sources_seen',[])) != START_CURSOR
            or len(set(parent.get('sources_seen',[]))) != START_CURSOR):
        raise ValueError('Expected the complete24000-source step2000 continuation checkpoint')
    chosen = list(source_ids[START_CURSOR:STOP_CURSOR])
    if (len(chosen) != STOP_CURSOR-START_CURSOR or len(set(chosen)) != len(chosen)
            or set(chosen)&set(parent['sources_seen'])):
        raise ValueError('Continuation requires36000 new distinct sources')
    return {'start':START_CURSOR,'stop':STOP_CURSOR,'historical':list(parent['sources_seen']),'source_ids':chosen}


def authenticate(args, data, schedule):
    checksum = base.sha(args.parent_checkpoint)
    if checksum != PARENT_SHA256: raise ValueError('Select the preserved cut1 step2000 continuation checkpoint')
    parent = torch.load(args.parent_checkpoint,map_location='cpu',weights_only=True,mmap=True)
    folder = args.parent_checkpoint.parent
    launch = json.loads((folder/'launch.json').read_text())
    completed = json.loads((folder/'completed.json').read_text())
    receipt = json.loads(args.parent_checkpoint.with_suffix('.json').read_text())
    report = json.loads((folder/'development-step2000.json').read_text())
    baseline = json.loads((args.original_dir/'development-step0.json').read_text())
    original_launch = json.loads((args.original_dir/'launch.json').read_text())
    last_training = json.loads((folder/'train.jsonl').read_text().splitlines()[-1])
    if (parent.get('format') != previous.VERSION or parent['identity'] != launch
            or parent['selection'] != schedule[1] or parent['original_cut_identity'] != original_launch
            or tuple(len(parent['selection'][k]) for k in ('stage2_indices','stage3_indices')) != (384,256)
            or original_launch.get('recipe_sha256') != base.sha(args.recipe)
            or original_launch.get('source_ids') != list(data.source_ids[:12000])
            or original_launch.get('cut_index') != 1
            or launch.get('source_plan_identity_sha256') != data.fresh.identity
            or launch.get('source_interval') != [12000,24000]
            or launch.get('source_ids') != list(data.source_ids[12000:24000])
            or launch.get('source_ids_sha256') != screen.digest(list(data.source_ids[12000:24000]))
            or launch.get('parent_checkpoint_sha256') != previous.PARENT_SHA256
            or parent.get('schedule_sha256') != base.sha(args.schedule)
            or launch.get('schedule_sha256') != base.sha(args.schedule)
            or launch.get('precision') != 'FP32 TF32 disabled'
            or launch.get('optimizer_reset') is not False or launch.get('width_changed') is not False
            or parent.get('coefficients') != prior.COEFFICIENTS
            or parent.get('teacher_source_sha256') != base.SOURCE_SHA256
            or parent.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256
            or completed.get('status') != 'awaiting_review' or completed.get('step') != START
            or completed.get('cut_updates') != START or completed.get('sources_seen') != START_CURSOR
            or completed.get('last_checkpoint_sha256') != checksum
            or completed.get('original_files_preserved') is not True
            or completed.get('frozen_state_preserved') is not True
            or completed.get('optimizer_reset') is not False
            or receipt.get('checkpoint_sha256') != checksum or receipt.get('cut_updates') != START
            or receipt.get('cut_index') != 1 or receipt.get('global_updates') != START
            or receipt.get('sources_seen') != START_CURSOR or receipt.get('optimizer_reset') is not False
            or receipt.get('optimizer_and_rng_saved') is not True or receipt.get('frozen_state_preserved') is not True
            or receipt.get('quality') != report['aggregate'] or completed.get('final') != report['aggregate']
            or last_training.get('step') != START or last_training.get('unique_sources') != START_CURSOR
            or last_training.get('source_ids') != parent['sources_seen'][-12:]
            or not math.isclose(last_training['audio_hours'],parent['scored_samples_this_cut']/48000/3600,abs_tol=1e-8)
            or not math.isfinite(last_training['elapsed_seconds']) or last_training['elapsed_seconds'] < 0):
        raise ValueError('Parent launch, source lineage, width, or quality receipt differs')
    prior.validate_recipe(launch); prior.validate_recipe(original_launch)
    for path,expected in launch['protected'].items():
        if base.sha(path) != expected: raise ValueError('Protected parent input changed: '+path)
    source_window(parent,data.source_ids)
    return parent,report,baseline,launch,last_training


def progress_accounting(parent,last_training,samples,elapsed):
    return {'audio_hours':(parent['scored_samples_this_cut']+samples)/48000/3600,
            'elapsed_seconds':last_training['elapsed_seconds']+elapsed,
            'continuation_audio_hours':samples/48000/3600,'continuation_elapsed_seconds':elapsed}


def restore_state(model, parent):
    if parent.get('cut_updates') != START or parent.get('accumulation') != ACCUMULATION:
        raise ValueError('Restore requires the existing step2000 AdamW state')
    payload = {'group':parent['group'],'optimizer':parent['optimizer'],'rng':parent['rng'],
               'step':START,'identity':{'learning_rate':3e-5}}
    optimizer = resume.continuation.restore_training_state(model,payload)
    comparisons = {key:replay.compare_tree(actual,expected) for key,actual,expected in (
        ('group',dict(model.group_state_dict()),dict(parent['group'])),
        ('optimizer',optimizer.state_dict(),parent['optimizer']),('rng',screen.rng_state(),parent['rng']))}
    if not all(row['equal'] for row in comparisons.values()):
        raise RuntimeError('Group weights, AdamW moments or RNG did not restore exactly')
    return optimizer,comparisons


def save_state(path,model,optimizer,parent,identity,seen,samples,step):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    if step not in range(START+EVERY,STOP+1,EVERY): raise ValueError('Only the approved recovery reviews can be checkpointed')
    expected = parent['sources_seen']+identity['source_ids'][:(step-START)*ACCUMULATION]
    if (list(seen) != expected or len(seen) != step*ACCUMULATION or len(set(seen)) != len(seen)
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']
            or identity.get('parent_checkpoint_sha256') != PARENT_SHA256
            or identity.get('selection') != parent['selection']):
        raise ValueError('Continuation source cursor or unchanged width identity differs')
    params = base.parameters(model)
    if not params or set(optimizer.state) != set(params): raise ValueError('Incomplete joint optimizer state')
    if any(float(state['step']) != step for state in optimizer.state.values()):
        raise ValueError('AdamW counter was reset or differs from completed updates')
    for group in optimizer.param_groups:
        if (group['lr'] != 3e-5 or group['betas'] != (.9,.99)
                or group['eps'] != 1e-8 or group['weight_decay'] != 0):
            raise ValueError('Unchanged optimizer policy required')
    if type(samples) is not int or samples <= 0: raise ValueError('Invalid continuation audio accounting')
    payload = {'format':VERSION,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'cut_index':1,'cut_updates':step,
        'global_updates':step,'selection':parent['selection'],'schedule_sha256':parent['schedule_sha256'],
        'sources_seen':list(seen),'continuation_scored_samples':samples,
        'scored_samples_this_cut':parent['scored_samples_this_cut']+samples,
        'coefficients':dict(prior.COEFFICIENTS),'accumulation':ACCUMULATION,
        'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'identity':identity,'original_cut_identity':parent['original_cut_identity'],'optimizer_reset':False,'automatic_next_cut':False}
    temporary = path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
    return {'checkpoint_sha256':base.sha(path),'cut_index':1,'cut_updates':step,'global_updates':step,
            'sources_seen':len(seen),'optimizer_and_rng_saved':True,'optimizer_reset':False}


def main():
    from progressive_continue_5000_monitor import ContinuationMonitor
    import progressive_continue_5000_monitor as monitor_module
    import progressive_extended_data as data_module
    p=argparse.ArgumentParser()
    for name in ('parent-checkpoint','original-dir','manifest','source-plan','shards','extension-plan','extension-shards','assets','recipe','schedule','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tensorboard',type=Path)
    p.add_argument('--shard-wait-seconds',type=int,default=3600)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a new isolated continuation directory')
    logdir=(args.tensorboard or args.out/'tensorboard').resolve()
    if any(logdir.is_relative_to(path.resolve().parent) for path in (args.parent_checkpoint,args.recipe)):
        raise ValueError('Dashboard would overlap a preserved run')
    base.policy()
    manifest,pools,cache_receipt=base.load_data(args.manifest)
    if len(pools['development'])!=96 or len(pools['calibration'])!=72:
        raise ValueError('Original held-out panels changed')
    fresh=fresh_module.FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    original_data=prior.SourceStream(pools,fresh)
    data=data_module.ExtendedProgressiveData(original_data,args.extension_plan,args.extension_shards)
    if len(data.source_ids)!=STOP_CURSOR or tuple(data.source_ids[:30000])!=original_data.source_ids:
        raise ValueError('Extension must preserve the exact original30000-source stream')
    schedule=prior.authenticate_schedule(json.loads(args.schedule.read_text()),manifest_sha=base.sha(args.manifest),
        source_plan_sha=base.sha(args.source_plan),calibration_ids=[c['source_id'] for c in pools['calibration']])
    prior.validate_recipe(json.loads(args.recipe.read_text()))
    parent,reference,baseline,parent_launch,last_training=authenticate(args,data,schedule)
    window=source_window(parent,data.source_ids)
    protected=dict(parent_launch['protected'])
    paths=[args.parent_checkpoint,args.parent_checkpoint.with_suffix('.json'),args.parent_checkpoint.parent/'launch.json',
        args.parent_checkpoint.parent/'completed.json',args.parent_checkpoint.parent/'development-step2000.json',
        args.original_dir/'development-step0.json',args.original_dir/'launch.json',args.extension_plan,Path(data_module.__file__),Path(previous.__file__),Path(__file__),Path(prior.__file__),Path(model_api.__file__),
        Path(monitor_module.__file__),Path(fresh_module.__file__),args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth']
    protected.update({str(path.resolve()):base.sha(path) for path in paths})
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    if str(torch.__version__)!=parent_launch['torch'] or torch.backends.cudnn.version()!=parent_launch['cudnn']:
        raise ValueError('Original training runtime changed')
    model=model_api.initialize_from_teacher(teacher.model.decoder,parent['selection'])
    optimizer,restoration=restore_state(model,parent)
    if len(base.parameters(model))!=90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Expected all90 group parameters and the frozen original teacher')
    frozen=screen.frozen_versions(model);teacher_frozen=resume.continuation.teacher_versions(teacher)
    common=base.objective();metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
    boundary_panel,_=base.select_boundary_panel(pools['development'],metadata)
    evaluator=UnifiedMonitor(None,{},metadata)
    args.out.mkdir(parents=True)
    identity={'version':VERSION,'parent_checkpoint_sha256':PARENT_SHA256,'selection':parent['selection'],
        'schedule_sha256':base.sha(args.schedule),'coefficients':prior.COEFFICIENTS,'learning_rate':3e-5,
        'execution_batch_size':1,'gradient_accumulation':12,'trainable_stages':[2,3,4],
        'starting_step':START,'target_step':STOP,'source_interval':[START_CURSOR,STOP_CURSOR],
        'source_ids':window['source_ids'],'source_ids_sha256':screen.digest(window['source_ids']),
        'source_plan_identity_sha256':fresh.identity,'extension_plan_identity_sha256':data.identity,
        'extension_plan_sha256':base.sha(args.extension_plan),'protected':protected,
        'optimizer_reset':False,'width_changed':False,'original_teacher_forever':True,'automatic_next_cut':False,
        'previous_training_elapsed_seconds':last_training['elapsed_seconds'],
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'precision':'FP32 TF32 disabled',
        'tensorboard':str(logdir)}
    monitor=ContinuationMonitor(logdir,parent_dir=args.parent_checkpoint.parent,original_dir=args.original_dir,source_metadata=metadata)
    identity['monitor_history']=monitor.history_identity
    base.write_json(args.out/'launch.json',identity)
    screen.restore_rng(parent['rng'])
    seen=list(parent['sources_seen']);samples=0;warmed=set();last_receipt=None;step=START
    started=time.monotonic();failure=None;status='running';reports=[]
    try:
        prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            initial=joint.evaluate_review(evaluator,model,teacher,pools['development'],common)
        comparison=numeric_reference_check(initial,reference)
        after_validation={key:replay.compare_tree(actual,expected) for key,actual,expected in (
            ('group',dict(model.group_state_dict()),dict(parent['group'])),
            ('optimizer',optimizer.state_dict(),parent['optimizer']),('rng',screen.rng_state(),parent['rng']))}
        base.write_json(args.out/'restore-check.json',{'state':restoration,'quality':comparison,'after_validation':after_validation})
        if not comparison['passed']:raise RuntimeError('Restored step2000 quality differs from its saved report')
        if not all(r['equal'] for r in after_validation.values()):raise RuntimeError('Starting validation altered the restored state')
        base.write_json(args.out/'development-step2000.json',initial);reports.append(initial)
        monitor.log_validation(initial,START)
        base.event('progressive_same_cut_started',step=START,target=STOP,optimizer_reset=False)
        for step in range(START+1,STOP+1):
            cursor=(step-1)*ACCUMULATION
            crops=resume.take_when_ready(data,cursor,ACCUMULATION,step=step,monitor=monitor,
                wait_seconds=args.shard_wait_seconds)
            ids=[c['source_id'] for c in crops]
            if ids!=window['source_ids'][(step-START-1)*ACCUMULATION:(step-START)*ACCUMULATION]:
                raise RuntimeError('Continuation source order differs')
            prior.warm_student(model,teacher,crops,warmed,optimizer);before=time.monotonic()
            values,checks=prior.perform_update(model,teacher,crops,common,optimizer,
                diagnostics=step==START+1 or step%25==0)
            seen.extend(ids);samples+=sum(c['valid_scored_samples'] for c in crops)
            record={'step':step,'updates':step-START,'global_step':step,**values,'source_ids':ids,'unique_sources':len(seen),
                **progress_accounting(parent,last_training,samples,time.monotonic()-started),
                'step_seconds':time.monotonic()-before,'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as h:h.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,step)
            if step==START+1 or step%25==0:
                base.event('progressive_same_cut_progress',step=step,total=values['total'],elapsed_seconds=record['elapsed_seconds'])
            if step%EVERY==0:
                with replay.diagnostic_state_guard(model,teacher,optimizer):
                    report=joint.evaluate_review(evaluator,model,teacher,pools['development'],common)
                    boundary=base.evaluate_boundaries(teacher,model,boundary_panel,parent['selection'],base.batch,base.teacher_forward)
                if report['quiet_regions']['window_identity_sha256']!=baseline['quiet_regions']['window_identity_sha256']:
                    raise RuntimeError('Original quiet windows changed')
                reports.append(report);monitor.log_validation(report,step)
                base.write_json(args.out/f'development-step{step}.json',report)
                base.write_json(args.out/f'boundaries-step{step}.json',boundary)
                if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:
                    raise RuntimeError('Frozen model component changed')
                data.assert_unchanged()
                last_receipt=save_state(args.out/f'checkpoint-step{step}.pt',model,optimizer,parent,identity,seen,samples,step)
                last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                base.event('progressive_same_cut_review_ready',step=step,quality=report['aggregate'],automatic_next_cut=False)
        status='awaiting_review'
    except BaseException as exc:
        failure=repr(exc);status='failed';raise
    finally:
        monitor.close();monitor.assert_history_unchanged();data.assert_unchanged()
        preserved=screen.frozen_versions(model)==frozen and resume.continuation.teacher_versions(teacher)==teacher_frozen
        files=all(base.sha(path)==value for path,value in protected.items())
        base.write_json(args.out/'completed.json',{'version':VERSION,'status':status,'failure':failure,
            'step':START+(len(seen)-START_CURSOR)//12,'cut_updates':START+(len(seen)-START_CURSOR)//12,'sources_seen':len(seen),
            'new_sources':len(seen)-START_CURSOR,'scored_samples':samples,'elapsed_seconds':time.monotonic()-started,
            'last_checkpoint_sha256':last_receipt['checkpoint_sha256'] if last_receipt else None,
            'frozen_state_preserved':preserved,'original_files_preserved':files,'optimizer_reset':False,
            'automatic_next_cut':False,'final':reports[-1]['aggregate'] if reports else None})
        if not preserved or not files:raise RuntimeError('Protected progressive state changed')


if __name__=='__main__':main()
