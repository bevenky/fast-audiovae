"""One externally reviewed, source-unique recovery segment after a width cut.

The original AudioVAE2 remains the teacher. Only the student stage2–4 widths
change; reconstruction losses and the singleton accumulation update are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import joint_recovery_v2 as joint
import fresh_training_data as fresh_module
from fresh_training_data import FreshTrainingData
from unified_monitor import UnifiedMonitor

base, screen, replay, resume = joint.base, joint.screen, joint.replay, joint.resume
VERSION = 'audiovae2_progressive_train_v1'
ACCUMULATION, UPDATES, EVERY = 12, 1000, 500
COEFFICIENTS = {'waveform':1.0, 'mel':0.0006674012905982311, 'feature':0.009304078923434964}


class SourceStream:
    """The original fitting cache followed by the immutable fresh-source plan."""
    def __init__(self, pools, fresh):
        self.fit, self.fresh = pools['fit'], fresh
        if len(self.fit) != 3000:
            raise ValueError('Expected the original3000-source fitting panel')
        self.source_ids = tuple(c['source_id'] for c in self.fit)+tuple(fresh.source_ids)
        forbidden = {c['source_id'] for c in pools['calibration']+pools['development']}
        if len(set(self.source_ids)) != len(self.source_ids) or set(self.source_ids)&forbidden:
            raise ValueError('Progressive fitting sources repeat or intersect held-out data')

    def take(self, start, count):
        if type(start) is not int or type(count) is not int or start < 0 or count < 1 or start+count > len(self.source_ids):
            raise ValueError('Invalid progressive source interval')
        stop = start+count
        result = list(self.fit[start:min(stop,len(self.fit))]) if start < len(self.fit) else []
        if stop > len(self.fit):
            a = max(start,len(self.fit))-len(self.fit)
            result += self.fresh.take(a,stop-max(start,len(self.fit)))
        if [c['source_id'] for c in result] != list(self.source_ids[start:stop]):
            raise RuntimeError('Progressive source order changed')
        return result

    def assert_unchanged(self):
        self.fresh.assert_unchanged()


def source_window(source_ids, parent=None, updates=UPDATES):
    if type(updates) is not int or updates != UPDATES:
        raise ValueError('This launch is bounded to exactly1000 recovery updates')
    historical = [] if parent is None else list(parent['sources_seen'])
    if historical != list(source_ids[:len(historical)]) or len(set(historical)) != len(historical):
        raise ValueError('Parent source ledger is not the complete progressive prefix')
    start = len(historical)
    chosen = list(source_ids[start:start+updates*ACCUMULATION])
    if len(chosen) != updates*ACCUMULATION or len(set(chosen)) != len(chosen) or set(chosen)&set(historical):
        raise ValueError('Insufficient new distinct sources for this width cut')
    return {'start':start,'stop':start+len(chosen),'historical':historical,'source_ids':chosen}


def validate_recipe(recipe):
    if (recipe.get('coefficients') != COEFFICIENTS or recipe.get('learning_rate') != 3e-5
            or recipe.get('gradient_accumulation') != ACCUMULATION or recipe.get('execution_batch_size') != 1
            or recipe.get('trainable_stages') != [2,3,4]):
        raise ValueError('Progressive recovery must retain the approved reconstruction recipe')
    return dict(COEFFICIENTS)


def validate_schedule(schedule):
    expected = [(512,256),(384,256),(384,192),(256,192),(256,128)]
    steps = schedule.get('steps')
    if not isinstance(steps,list) or len(steps) != len(expected):
        raise ValueError('Expected the approved four nested width cuts')
    previous = None
    for row,widths in zip(steps,expected):
        for key,n,maximum in [('stage2_indices',widths[0],512),('stage3_indices',widths[1],256)]:
            ids = row.get(key)
            if (not isinstance(ids,list) or len(ids) != n or ids != sorted(set(ids))
                    or any(type(i) is not int or not 0 <= i < maximum for i in ids)):
                raise ValueError('Invalid original-coordinate width selection')
            if previous is not None and not set(ids) <= set(previous[key]):
                raise ValueError('A later cut cannot reintroduce removed coordinates')
        previous = row
    return steps


def authenticate_schedule(schedule, *, manifest_sha, source_plan_sha, calibration_ids):
    if (schedule.get('version') != 'audiovae2_progressive_schedule_v1'
            or schedule.get('identity_sha256') != screen.digest({k:v for k,v in schedule.items() if k!='identity_sha256'})
            or schedule.get('teacher_source_sha256') != base.SOURCE_SHA256
            or schedule.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256
            or schedule.get('manifest_sha256') != manifest_sha
            or schedule.get('source_plan_sha256') != source_plan_sha
            or schedule.get('calibration_source_ids') != list(calibration_ids)
            or len(set(calibration_ids)) != 72
            or schedule.get('original_endpoint_reproduced') is not True
            or schedule.get('teacher_frozen') is not True):
        raise ValueError('Schedule identity, fitting calibration, or frozen teacher provenance differs')
    return validate_schedule(schedule)


def authenticate_parent(path, schedule_sha, cut_index, source_ids):
    if path is None:
        if cut_index != 1: raise ValueError('Only the first cut starts from original teacher weights')
        return None, None
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    completion = json.loads((path.parent/'completed.json').read_text())
    checksum = base.sha(path)
    payload = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (payload.get('format') != VERSION or payload.get('cut_index') != cut_index-1
            or payload.get('cut_updates') != UPDATES or payload.get('schedule_sha256') != schedule_sha
            or receipt.get('checkpoint_sha256') != checksum or receipt.get('frozen_state_preserved') is not True
            or completion.get('last_checkpoint_sha256') != checksum or completion.get('status') != 'awaiting_review'
            or completion.get('cut_updates') != UPDATES or completion.get('original_files_preserved') is not True
            or payload.get('coefficients') != COEFFICIENTS or payload.get('accumulation') != ACCUMULATION
            or payload.get('teacher_source_sha256') != base.SOURCE_SHA256
            or payload.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256
            or payload.get('sources_seen') != list(source_ids[:len(payload.get('sources_seen',[]))])):
        raise ValueError('Previous cut is not an authenticated, completed progressive checkpoint')
    if len(payload['sources_seen']) != (cut_index-1)*UPDATES*ACCUMULATION:
        raise ValueError('Previous cut source cursor differs from its completed updates')
    return payload, checksum


def save_state(path, model, optimizer, *, cut_index, cut_updates, selection, identity, seen, samples):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    if cut_updates not in (0,EVERY,UPDATES) or cut_index != identity['cut_index']:
        raise ValueError('Checkpoint is not on the approved cut review boundary')
    expected = identity['historical_sources']+identity['source_ids'][:cut_updates*ACCUMULATION]
    if (list(seen) != expected or len(set(seen)) != len(seen)
            or len(seen) != ((cut_index-1)*UPDATES+cut_updates)*ACCUMULATION
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']):
        raise ValueError('Checkpoint source history is incomplete or reordered')
    params = base.parameters(model)
    if not params or (set(optimizer.state) != set(params) if cut_updates else bool(optimizer.state)):
        raise ValueError('Checkpoint optimizer scope differs from the joint group')
    if any(float(s['step']) != cut_updates for s in optimizer.state.values()):
        raise ValueError('Fresh per-cut AdamW counter differs from completed updates')
    for group in optimizer.param_groups:
        if group['lr'] != 3e-5 or group['betas'] != (.9,.99) or group['eps'] != 1e-8 or group['weight_decay'] != 0:
            raise ValueError('Optimizer recipe changed')
    if type(samples) is not int or (samples <= 0 if cut_updates else samples != 0):
        raise ValueError('Missing scored sample accounting')
    artifact = {'format':VERSION,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'cut_index':cut_index,'cut_updates':cut_updates,
        'global_updates':(cut_index-1)*UPDATES+cut_updates,'selection':selection,'schedule_sha256':identity['schedule_sha256'],
        'sources_seen':list(seen),'scored_samples_this_cut':samples,'coefficients':dict(COEFFICIENTS),
        'accumulation':ACCUMULATION,'teacher_source_sha256':base.SOURCE_SHA256,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'identity':identity,
        'automatic_promotion':False,'optimizer_restart_at_each_cut':True}
    temporary = path.with_suffix('.tmp'); torch.save(artifact,temporary); temporary.replace(path)
    return {'checkpoint_sha256':base.sha(path),'cut_index':cut_index,'cut_updates':cut_updates,
            'global_updates':artifact['global_updates'],'sources_seen':len(seen),'optimizer_and_rng_saved':True}


def warm_student(model, teacher, crops, warmed, optimizer=None):
    for crop in crops:
        shape = tuple(crop['latents'].shape)
        if shape not in warmed:
            with replay.diagnostic_state_guard(model,teacher,optimizer), torch.no_grad():
                z,_,_,_ = base.batch([crop]); trace = base.teacher_forward(teacher,z)
                for _ in range(3): model.suffix_from_group(model.group_from_input(trace['group_input']))
            warmed.add(shape)


def perform_update(model, teacher, crops, common, optimizer, *, diagnostics=False):
    if len(crops) != ACCUMULATION or len({c['source_id'] for c in crops}) != ACCUMULATION:
        raise ValueError('A progressive update requires12 distinct source crops')
    with replay.observe_teacher_cache(crops) as checks:
        values = screen.training_update(model,teacher,crops,'current',common,COEFFICIENTS,optimizer,
                                        record_diagnostics=diagnostics)
    if len(checks) != ACCUMULATION or not all(c['allclose_original_tolerance'] for c in checks):
        raise RuntimeError('Original teacher output differs from the sealed target')
    return values,checks


def main():
    import progressive_model as model_api
    from progressive_monitor import ProgressiveMonitor
    parser = argparse.ArgumentParser()
    for name in ('manifest','source-plan','shards','assets','recipe','schedule','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-checkpoint',type=Path)
    parser.add_argument('--cut-index',type=int,default=1)
    parser.add_argument('--tensorboard',type=Path)
    parser.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new progressive cut directory')
    if not 1 <= args.cut_index <= 4: raise ValueError('Invalid progressive cut')
    base.policy()
    manifest,pools,cache_receipt = base.load_data(args.manifest)
    if len(pools['development']) != 96 or len(pools['calibration']) != 72:
        raise ValueError('The fixed held-out panels changed')
    fresh = FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    data = SourceStream(pools,fresh)
    recipe = json.loads(args.recipe.read_text()); coefficients = validate_recipe(recipe)
    schedule = json.loads(args.schedule.read_text())
    steps = authenticate_schedule(schedule,manifest_sha=base.sha(args.manifest),source_plan_sha=base.sha(args.source_plan),
                                  calibration_ids=[c['source_id'] for c in pools['calibration']])
    parent,parent_sha = authenticate_parent(args.parent_checkpoint,base.sha(args.schedule),args.cut_index,data.source_ids)
    window = source_window(data.source_ids,parent)
    selection = steps[args.cut_index]
    if parent is not None and parent['selection'] != steps[args.cut_index-1]:
        raise ValueError('Parent original-channel coordinate selection changed')
    protected_paths = [args.manifest,args.source_plan,args.recipe,args.schedule,args.assets/'audio_vae_v2.py',
        args.assets/'audiovae.pth',Path(cache_receipt['path']),Path(__file__),Path(model_api.__file__),
        Path(base.__file__),Path(screen.__file__),Path(joint.__file__),Path(replay.__file__),
        Path(__file__).with_name('progressive_monitor.py'),Path(fresh_module.__file__)]
    if args.parent_checkpoint:
        protected_paths += [args.parent_checkpoint,args.parent_checkpoint.with_suffix('.json'),args.parent_checkpoint.parent/'completed.json']
    protected = {str(p.resolve()):base.sha(p) for p in protected_paths}
    logdir = (args.tensorboard or args.out/'tensorboard').resolve()
    if any(logdir.is_relative_to(p.resolve().parent) for p in (args.recipe,args.parent_checkpoint) if p):
        raise ValueError('Dashboard would overlap a retained original run')
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise ValueError('Use the qualified teacher and training runtime')
    teacher_frozen = resume.continuation.teacher_versions(teacher)
    if parent is None:
        model = model_api.initialize_from_teacher(teacher.model.decoder,selection)
    else:
        old = model_api.initialize_from_teacher(teacher.model.decoder,parent['selection'])
        old.load_group_state_dict(parent['group'])
        model = model_api.migrate(old,selection); del old
        screen.restore_rng(parent['rng'])
    optimizer = model_api.fresh_optimizer(model,lr=3e-5)
    if optimizer.state: raise RuntimeError('Each width cut must begin with fresh AdamW moments')
    if len(base.parameters(model)) != 90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Expected all90 joint group parameters and a fully frozen teacher')
    frozen = screen.frozen_versions(model)
    common = base.objective()
    source_metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    boundary_panel,boundary_selection = base.select_boundary_panel(pools['development'],source_metadata)
    evaluator = UnifiedMonitor(None,{},source_metadata)
    args.out.mkdir(parents=True)
    identity = {'version':VERSION,'cut_index':args.cut_index,'selection':selection,'schedule_sha256':base.sha(args.schedule),
        'parent_checkpoint_sha256':parent_sha,'original_teacher_forever':True,'optimizer_restart_at_each_cut':True,
        'coefficients':coefficients,'learning_rate':3e-5,'optimizer_betas':[.9,.99],'optimizer_eps':1e-8,'weight_decay':0,
        'execution_batch_size':1,'gradient_accumulation':ACCUMULATION,'trainable_stages':[2,3,4],
        'source_interval':[window['start'],window['stop']],'source_ids':window['source_ids'],
        'source_ids_sha256':screen.digest(window['source_ids']),'historical_sources':window['historical'],
        'source_plan_identity_sha256':fresh.identity,'initial_group_sha256':screen.group_digest(model),
        'currently_available_source_capacity':len(data.source_ids),'later_cuts_may_require_additional_disjoint_data':True,
        'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'protected':protected,'recipe_sha256':base.sha(args.recipe),'review_every_updates':EVERY,
        'target_cut_updates':UPDATES,'automatic_next_cut':False,'precision':'FP32 TF32 disabled',
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'boundary_panel':boundary_selection,
        'initial_optimizer_state_entries':len(optimizer.state),'tensorboard':str(logdir)}
    base.write_json(args.out/'launch.json',identity)
    precut = {'format':'audiovae2_progressive_pre_cut_v1','selection':steps[args.cut_index-1],
        'parent_checkpoint_sha256':parent_sha,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'group':{k:v.detach().cpu() for k,v in (parent['group'] if parent else
            {k:teacher.model.decoder.state_dict()[k] for k in model.group_state_dict()}).items()},
        'rng':screen.rng_state(),'original_teacher_forever':True}
    torch.save(precut,args.out/'pre-cut.pt')
    base.write_json(args.out/'pre-cut.json',{'sha256':base.sha(args.out/'pre-cut.pt'),
        'selection':precut['selection'],'parent_checkpoint_sha256':parent_sha})
    del precut
    monitor = ProgressiveMonitor(logdir,cut_name=f"stage2_{len(selection['stage2_indices'])}_stage3_{len(selection['stage3_indices'])}",
        widths=(len(selection['stage2_indices']),len(selection['stage3_indices'])),total_updates=UPDATES)
    seen = list(window['historical']); samples = 0; warmed = set(); reports = []; last_receipt = None
    started = time.monotonic(); failure = None; status = 'running'; updates = 0
    try:
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            copy_records = base.numerical_preflight(teacher,boundary_panel)
        base.write_json(args.out/'full-width-copy.json',{'passed':True,'records':copy_records})
        warm_student(model,teacher,pools['development'],warmed,optimizer)
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            baseline = joint.evaluate_review(evaluator,model,teacher,pools['development'],common)
            boundaries = base.evaluate_boundaries(teacher,model,boundary_panel,selection,base.batch,base.teacher_forward)
        base.write_json(args.out/'development-step0.json',baseline); reports.append(baseline)
        base.write_json(args.out/'boundaries-step0.json',boundaries)
        if screen.frozen_versions(model) != frozen or resume.continuation.teacher_versions(teacher) != teacher_frozen:
            raise RuntimeError('Preflight changed a frozen component')
        initial_receipt = save_state(args.out/'checkpoint-step0.pt',model,optimizer,cut_index=args.cut_index,
            cut_updates=0,selection=selection,identity=identity,seen=seen,samples=0)
        initial_receipt.update({'frozen_state_preserved':True,'quality':baseline['aggregate']})
        base.write_json(args.out/'checkpoint-step0.json',initial_receipt)
        monitor.log_validation(baseline,0)
        base.event('progressive_cut_started',cut_index=args.cut_index,widths=identity['selection'],sources=12000)
        for update in range(1,UPDATES+1):
            cursor = window['start']+(update-1)*ACCUMULATION
            crops = resume.take_when_ready(data,cursor,ACCUMULATION,step=update,writer=None,monitor=None,
                                           wait_seconds=args.shard_wait_seconds)
            ids = [c['source_id'] for c in crops]
            if ids != window['source_ids'][(update-1)*ACCUMULATION:update*ACCUMULATION]:
                raise RuntimeError('Loaded fitting sources differ from the planned prefix')
            warm_student(model,teacher,crops,warmed,optimizer)
            before = time.monotonic()
            values,checks = perform_update(model,teacher,crops,common,optimizer,
                                          diagnostics=update==1 or update%25==0)
            seen.extend(ids); samples += sum(c['valid_scored_samples'] for c in crops); updates = update
            record = {'step':update,'updates':update,'global_step':(args.cut_index-1)*UPDATES+update,
                **values,'source_ids':ids,'unique_sources':len(seen),'audio_hours':samples/48000/3600,
                'step_seconds':time.monotonic()-before,'elapsed_seconds':time.monotonic()-started,
                'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,update)
            if update==1 or update%25==0:
                base.event('progressive_training',**{k:v for k,v in record.items() if k not in ('teacher_cache_checks','source_ids')})
            if update%EVERY == 0:
                with replay.diagnostic_state_guard(model,teacher,optimizer):
                    report = joint.evaluate_review(evaluator,model,teacher,pools['development'],common)
                    boundary = base.evaluate_boundaries(teacher,model,boundary_panel,selection,base.batch,base.teacher_forward)
                if report['quiet_regions']['window_identity_sha256'] != baseline['quiet_regions']['window_identity_sha256']:
                    raise RuntimeError('Quiet validation window identities changed')
                reports.append(report); monitor.log_validation(report,update)
                base.write_json(args.out/f'development-step{update}.json',report)
                base.write_json(args.out/f'boundaries-step{update}.json',boundary)
                if screen.frozen_versions(model) != frozen or resume.continuation.teacher_versions(teacher) != teacher_frozen:
                    raise RuntimeError('A frozen model component changed')
                data.assert_unchanged()
                last_receipt = save_state(args.out/f'checkpoint-step{update}.pt',model,optimizer,cut_index=args.cut_index,
                    cut_updates=update,selection=selection,identity=identity,seen=seen,samples=samples)
                last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                base.write_json(args.out/f'checkpoint-step{update}.json',last_receipt)
                base.event('progressive_review_ready',cut_updates=update,quality=report['aggregate'],automatic_next_cut=False)
        status = 'awaiting_review'
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        monitor.close()
        preserved = screen.frozen_versions(model) == frozen and resume.continuation.teacher_versions(teacher) == teacher_frozen
        data.assert_unchanged()
        original_files = all(base.sha(p) == checksum for p,checksum in protected.items())
        receipt = {'version':VERSION,'status':status,'failure':failure,'cut_index':args.cut_index,'cut_updates':updates,
            'source_count':len(seen),'scored_samples':samples,'elapsed_seconds':time.monotonic()-started,
            'last_checkpoint_sha256':last_receipt['checkpoint_sha256'] if last_receipt else None,
            'frozen_state_preserved':preserved,'original_files_preserved':original_files,'automatic_next_cut':False,
            'final':reports[-1]['aggregate'] if reports else None}
        base.write_json(args.out/'completed.json',receipt)
        if not preserved or not original_files: raise RuntimeError('Progressive experiment changed protected state')


if __name__ == '__main__': main()
