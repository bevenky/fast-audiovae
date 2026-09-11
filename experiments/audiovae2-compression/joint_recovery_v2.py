"""Reviewed, source-unique continuation of the unchanged joint stage 2–4 model.

One invocation advances at most 1000 updates and then exits for external review.
The observation-only quiet split shares the existing validation forward pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import joint_recovery_v1 as previous
import audit_quiet_windows_v1 as quiet_audit
from unified_monitor import UnifiedMonitor

base, screen, replay, resume = previous.base, previous.screen, previous.replay, previous.resume
VERSION = 'audiovae2_joint_recovery_v2'
DECISION_VERSION = 'audiovae2_joint_recovery_decision_v2'
ROOT_STEP, ROOT_CURSOR, END_STEP, ACCUMULATION, MAX_UPDATES, EVERY = 5625, 24000, 10000, 12, 1000, 500
ROOT_SHA256 = '0da2f6e98a29b025b41df84dbb674e1a35fbea2629f698ef33fe62f687c18b1b'


def tensorboard_path(args):
    path = (args.tensorboard_logdir or args.out/'tensorboard'/'raw').resolve()
    for original in (args.joint_run,args.candidate.parent,args.checkpoint.parent,
                     args.anchor_checkpoint.parent,args.screen_out,args.base_out):
        if path.is_relative_to(original.resolve()):
            raise ValueError('TensorBoard destination overlaps a preserved original run')
    return path


def segment_window(parent, original_ids, fresh_ids, updates=None):
    step = parent.get('optimizer_step')
    if (type(step) is not int or step < ROOT_STEP or step >= END_STEP
            or (step-ROOT_STEP) % EVERY or parent.get('accumulation') != ACCUMULATION
            or parent.get('format') not in (previous.VERSION, VERSION)):
        raise ValueError('Expected a preserved joint checkpoint on the 500-update review grid')
    if parent['format'] == previous.VERSION and step != ROOT_STEP:
        raise ValueError('Only the authenticated step5625 v1 checkpoint can enter continuation')
    if updates is None: updates = min(MAX_UPDATES, END_STEP-step)
    if (type(updates) is not int or updates <= 0 or updates > MAX_UPDATES or step+updates > END_STEP
            or ((step+updates-ROOT_STEP) % EVERY and step+updates != END_STEP)):
        raise ValueError('Segment must end on a 500-update review or final step10000, within 1000 updates')
    cursor = ROOT_CURSOR+(step-ROOT_STEP)*ACCUMULATION
    historical = list(parent['historical_sources_seen'])+list(parent['additional_sources_seen'])
    expected = list(original_ids)+list(fresh_ids[:cursor])
    if (len(original_ids) != 3000 or historical != expected or len(set(historical)) != len(historical)
            or len(expected) != 3000+cursor):
        raise ValueError('Checkpoint history differs from the complete unique source prefix')
    if parent['format'] == VERSION and (parent.get('fresh_cursor') != cursor
            or parent.get('total_source_count') != len(historical)
            or parent['identity'].get('root_checkpoint_sha256') != ROOT_SHA256):
        raise ValueError('Continuation cursor or preserved root identity differs')
    chosen = list(fresh_ids[cursor:cursor+updates*ACCUMULATION])
    if (len(chosen) != updates*ACCUMULATION or len(set(chosen)) != len(chosen)
            or set(chosen)&set(historical)):
        raise ValueError('Segment sources are missing, repeated, or already consumed')
    return {'start_step':step, 'updates':updates, 'target_step':step+updates,
            'start_cursor':cursor, 'stop_cursor':cursor+len(chosen),
            'historical':historical, 'source_ids':chosen}


def validate_decision(decision, *, parent_sha256, window, plan_sha256, plan_identity,
                      runner_sha256, previous_action):
    expected = {'version':DECISION_VERSION, 'action':'continue',
        'parent_checkpoint_sha256':parent_sha256, 'start_step':window['start_step'],
        'fresh_cursor':window['start_cursor'], 'updates':window['updates'],
        'target_step':window['target_step'], 'source_plan_sha256':plan_sha256,
        'source_plan_identity_sha256':plan_identity, 'runner_sha256':runner_sha256,
        'previous_gate_action':'continue', 'architecture_changed':False, 'loss_changed':False,
        'optimizer_changed':False}
    if previous_action != 'continue' or not isinstance(decision, dict):
        raise ValueError('A completed favorable external review is required before the next segment')
    for key, value in expected.items():
        if type(decision.get(key)) is not type(value) or decision.get(key) != value:
            raise ValueError('Review decision differs: '+key)
    if not isinstance(decision.get('reviewed_by'), str) or not decision['reviewed_by'].strip():
        raise ValueError('Review decision must name its explicit review authority')
    return decision


def restart_checkpoint(model, parent):
    step = parent.get('optimizer_step')
    if (type(step) is not int or not ROOT_STEP <= step < END_STEP
            or parent.get('format') not in (previous.VERSION, VERSION)
            or parent.get('accumulation') != ACCUMULATION):
        raise ValueError('Unexpected joint continuation state')
    payload = {'group':parent['group'], 'optimizer':parent['optimizer'], 'rng':parent['rng'],
               'step':step, 'identity':parent['original_training_identity']}
    optimizer = resume.continuation.restore_training_state(model, payload)
    comparisons = {k:replay.compare_tree(a,b) for k,a,b in (
        ('group',dict(model.group_state_dict()),dict(parent['group'])),
        ('optimizer',optimizer.state_dict(),parent['optimizer']),('rng',screen.rng_state(),parent['rng']))}
    if not all(c['equal'] for c in comparisons.values()):
        raise RuntimeError('Checkpoint group, AdamW moments, or RNG did not restore exactly')
    return optimizer, comparisons


def read_history(entries, expected_last):
    if not isinstance(entries, list) or not entries: raise ValueError('Missing complete review history')
    reports, steps = [], []
    for entry in entries:
        if type(entry.get('step')) is not int: raise ValueError('Review steps must be integers')
        path = Path(entry['path'])
        if base.sha(path) != entry['sha256']: raise ValueError('Earlier review report changed')
        steps.append(entry['step']); reports.append(json.loads(path.read_text()))
    expected = list(range(ROOT_STEP, expected_last+1, EVERY))
    if expected_last == END_STEP and expected[-1] != END_STEP: expected.append(END_STEP)
    if steps != expected: raise ValueError('Review history is missing or reordered')
    return reports, steps


def report_entry(path, step):
    return {'path':str(path.resolve()), 'sha256':base.sha(path), 'step':step}


def save_state(path, model, optimizer, parent, seen, samples, identity, history, review):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    historical = list(parent['historical_sources_seen'])+list(parent['additional_sources_seen'])
    start = identity['starting_optimizer_step']; cursor = identity['source_interval'][0]
    if (not seen or len(seen) % ACCUMULATION or len(seen) > MAX_UPDATES*ACCUMULATION
            or len(set(seen)) != len(seen) or set(seen)&set(historical)
            or list(seen) != identity['source_ids'][:len(seen)]
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']
            or cursor != ROOT_CURSOR+(start-ROOT_STEP)*ACCUMULATION
            or identity['source_interval'][1] != cursor+len(identity['source_ids'])):
        raise ValueError('Saved source lineage or planned order differs')
    updates = len(seen)//ACCUMULATION; step = start+updates
    if (start != parent['optimizer_step'] or step > identity['target_optimizer_step'] or step > END_STEP
            or (step-ROOT_STEP) % EVERY and step != END_STEP
            or identity.get('root_checkpoint_sha256') != ROOT_SHA256
            or identity.get('gradient_accumulation') != ACCUMULATION
            or identity.get('execution_batch_size') != 1
            or identity.get('learning_rate') != 3e-5
            or identity.get('coefficients') != parent['original_training_identity']['coefficients']):
        raise ValueError('Saved recipe, counter, or review boundary differs')
    if type(samples) is not int or samples <= 0: raise ValueError('Invalid segment sample accounting')
    read_history(history, step)
    if review.get('step') != step or review.get('action') not in ('continue','pause_for_diagnosis'):
        raise ValueError('Checkpoint lacks its completed quality review')
    params = base.parameters(model)
    if not params or set(optimizer.state) != set(params): raise ValueError('Missing or additional optimizer state')
    if any(float(state['step']) != step for state in optimizer.state.values()):
        raise ValueError('AdamW counters differ from completed updates')
    artifact = {'format':VERSION, 'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(), 'rng':screen.rng_state(), 'optimizer_step':step,
        'updates':updates, 'continuation_updates':step-ROOT_STEP, 'accumulation':ACCUMULATION,
        'scored_samples':samples, 'continued_scored_samples':parent.get('continued_scored_samples',0)+samples,
        'fresh_cursor':cursor+len(seen), 'total_source_count':len(historical)+len(seen),
        'historical_sources_seen':historical, 'additional_sources_seen':list(seen),
        'original_training_identity':parent['original_training_identity'], 'identity':identity,
        'review_history':history, 'latest_review':review, 'automatic_promotion':False}
    temporary = path.with_suffix('.tmp'); torch.save(artifact,temporary); temporary.replace(path)
    return {'checkpoint_sha256':base.sha(path), 'optimizer_step':step, 'updates':updates,
        'fresh_cursor':artifact['fresh_cursor'], 'total_source_count':artifact['total_source_count'],
        'sources_consumed':len(seen), 'optimizer_and_rng_saved':True}


def evaluate_review(monitor, model, teacher, crops, common, summarize=None):
    if summarize is None:
        from joint_recovery_gates_v2 import summarize_regions
        summarize = summarize_regions
    original = base.quiet_window_metrics; captured = []; cursor = 0
    def observe(prediction, target, valid=None, **kwargs):
        nonlocal cursor
        if kwargs or (valid is not None and (valid.shape != prediction.shape
                or valid.dtype != torch.bool or not bool(valid.all()))):
            raise ValueError('Split quiet metrics require unchanged default limits and contiguous scored samples')
        if cursor >= len(crops): raise RuntimeError('Extra quiet observation')
        crop = crops[cursor]
        if prediction.numel() != crop['valid_scored_samples']: raise ValueError('Quiet source extent differs')
        raw = original(prediction,target,valid,**kwargs)
        captured.extend(quiet_audit.describe_quiet_windows(prediction,target,crop,raw)); cursor += 1
        return raw
    base.quiet_window_metrics = observe
    try: report = previous.evaluate_review(monitor,model,teacher,crops,common)
    finally: base.quiet_window_metrics = original
    if cursor != len(crops) or [r['source_id'] for r in report['rows']] != [c['source_id'] for c in crops]:
        raise RuntimeError('Quiet observation source order differs')
    if (len(captured) != report['aggregate']['quiet_windows']
            or sum(not r['passed'] for r in captured) != report['aggregate']['quiet_failed_windows']):
        raise RuntimeError('Split quiet counts differ from unchanged evaluation')
    report['quiet_regions'] = summarize(captured)
    return report


def authenticate_parent(path, root, root_reference, data, pools):
    checksum = base.sha(path)
    if checksum == ROOT_SHA256:
        return root, root_reference, checksum, [], [], []
    payload = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    receipt = json.loads(path.with_suffix('.json').read_text())
    launch = json.loads((path.parent/'launch.json').read_text())
    completion = json.loads((path.parent/'completed.json').read_text())
    if (payload.get('format') != VERSION or payload.get('identity') != launch
            or launch.get('root_checkpoint_sha256') != ROOT_SHA256
            or launch.get('runner_sha256') != base.sha(Path(__file__))
            or launch.get('source_plan_identity_sha256') != data.identity
            or launch.get('source_plan_sha256') != base.sha(data.plan_path)
            or payload.get('original_training_identity') != root['original_training_identity']
            or receipt.get('checkpoint_sha256') != checksum or receipt.get('frozen_state_preserved') is not True
            or receipt.get('optimizer_step') != payload.get('optimizer_step')
            or receipt.get('fresh_cursor') != payload.get('fresh_cursor')
            or completion.get('status') != 'awaiting_review'
            or completion.get('last_checkpoint_sha256') != checksum
            or completion.get('step') != payload.get('optimizer_step')
            or completion.get('step') != launch.get('target_optimizer_step')
            or completion.get('fresh_cursor') != payload.get('fresh_cursor')
            or completion.get('total_source_count') != payload.get('total_source_count')
            or completion.get('updates') != payload.get('updates')
            or completion.get('sources') != len(payload.get('additional_sources_seen',[]))
            or completion.get('original_files_preserved') is not True):
        raise ValueError('Continuation checkpoint, completion, or source identity differs')
    if any(base.sha(p) != s for p,s in launch['protected'].items()):
        raise ValueError('Protected parent lineage files changed')
    reports, steps = read_history(payload['review_history'],payload['optimizer_step'])
    if reports[-1]['aggregate'] != receipt['quality'] or reports[-1]['aggregate'] != completion['final']:
        raise ValueError('Parent quality and checkpoint receipts differ')
    if payload['latest_review'] != completion['reviews'][-1]: raise ValueError('Parent review receipt differs')
    if set(payload['historical_sources_seen']+payload['additional_sources_seen']) & {
            c['source_id'] for c in pools['calibration']+pools['development']}:
        raise ValueError('Parent sources leaked calibration/development')
    return payload,reports[-1],checksum,list(payload['review_history']),reports,steps


def main():
    from continuation_data_v2 import ContinuationData
    import continuation_data_v2 as data_module
    import joint_recovery_gates_v2 as gates
    p = argparse.ArgumentParser()
    for name in ('checkpoint','candidate','anchor-checkpoint','screen-out','base-out','manifest',
                 'fresh-manifest','shards','extended-manifest','extended-shards','assets','joint-run','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--decision',type=Path)
    p.add_argument('--updates',type=int)
    p.add_argument('--tensorboard-logdir',type=Path)
    p.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new segment directory')
    logdir = tensorboard_path(args)
    base.policy()
    _,selection,_,manifest,pools,original,_,original_sha,_ = resume.authenticate(args)
    old_data = replay.FreshTrainingData(args.fresh_manifest,args.manifest,pools,args.shards)
    candidate,_,candidate_sha = previous.prior.authenticate_candidate(args.candidate,original,list(old_data.source_ids[10500:12000]))
    if (candidate['identity']['start_checkpoint_sha256'] != original_sha
            or candidate['identity']['runner_sha256'] != base.sha(previous.prior.previous.__file__)
            or candidate['identity']['resume_identity'] != original['resume_identity']):
        raise ValueError('Original 4500-to-4625 candidate lineage differs')
    root,root_reference,root_sha = quiet_audit.authenticate_final(args,candidate,old_data,pools)
    data = ContinuationData(args.extended_manifest,args.manifest,pools,args.extended_shards)
    if list(data.source_ids[:27000]) != list(old_data.source_ids): raise ValueError('Extended plan changed original prefix')
    parent_path = args.resume_from or args.joint_run/'checkpoint-step5625.pt'
    parent,reference,parent_sha,history,reports,steps = authenticate_parent(parent_path,root,root_reference,data,pools)
    window = segment_window(parent,[c['source_id'] for c in pools['fit']],data.source_ids,args.updates)
    if data.committed_cursor != window['start_cursor']:
        raise ValueError('Rolling-cache commitment differs from the selected training checkpoint')
    development = pools['development']; source_metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    if len(development) != 96 or set(window['source_ids']) & {c['source_id'] for c in pools['calibration']+development}:
        raise ValueError('Expected fixed held-out96 sources without training leakage')
    if parent_sha != ROOT_SHA256 and args.decision is None:
        raise ValueError('Later segments require an explicit externally reviewed continuation decision')
    decision = None
    if args.decision:
        decision = validate_decision(json.loads(args.decision.read_text()),parent_sha256=parent_sha,window=window,
            plan_sha256=base.sha(args.extended_manifest),plan_identity=data.identity,runner_sha256=base.sha(Path(__file__)),
            previous_action=parent.get('latest_review',{'action':'continue'})['action'])
    protected_paths = [args.checkpoint,args.candidate,args.anchor_checkpoint,args.manifest,args.fresh_manifest,
        args.extended_manifest,parent_path,args.joint_run/'checkpoint-step5625.pt',args.joint_run/'checkpoint-step5625.json',
        args.joint_run/'launch.json',args.joint_run/'completed.json',args.joint_run/'development-step5625.json',
        args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',Path(__file__),Path(previous.__file__),
        Path(quiet_audit.__file__),Path(base.__file__),Path(screen.__file__),Path(replay.__file__),Path(resume.__file__),
        Path(data_module.__file__),Path(gates.__file__),Path(__file__).with_name('unified_monitor.py')]
    protected_paths += [Path(e['path']) for e in history]
    if parent_sha != ROOT_SHA256:
        protected_paths += [parent_path.with_suffix('.json'),parent_path.parent/'launch.json',parent_path.parent/'completed.json']
    if args.decision: protected_paths.append(args.decision)
    protected = {str(path.resolve()):base.sha(path) for path in protected_paths}
    args.out.mkdir(parents=True)
    identity = {'version':VERSION,'root_checkpoint_sha256':root_sha,'parent_checkpoint_sha256':parent_sha,
        'parent_checkpoint_path':str(parent_path.resolve()),'starting_optimizer_step':window['start_step'],
        'target_optimizer_step':window['target_step'],'source_interval':[window['start_cursor'],window['stop_cursor']],
        'source_ids':window['source_ids'],'source_ids_sha256':screen.digest(window['source_ids']),
        'source_plan_sha256':base.sha(args.extended_manifest),'source_plan_identity_sha256':data.identity,
        'runner_sha256':base.sha(Path(__file__)),'gradient_accumulation':ACCUMULATION,'execution_batch_size':1,
        'coefficients':parent['original_training_identity']['coefficients'],'learning_rate':3e-5,
        'protected':protected,'review_every_updates':EVERY,'trainable_stages':[2,3,4],
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'backend':replay.backend_state(),
        'decision':decision,'automatic_promotion':False,'automatic_freezing':False,'requires_external_segment_review':True,
        'tensorboard_logdir':str(logdir)}
    base.write_json(args.out/'launch.json',identity)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(logdir),flush_secs=10)
    baseline = json.loads((args.screen_out/'current_lr3e-5'/'development-step0.json').read_text())['aggregate']
    monitor = UnifiedMonitor(writer,baseline,source_metadata); monitor.initialize_layout()
    # The first sealed shard may still be preparing: wait before allocating models.
    resume.take_when_ready(data,window['start_cursor'],ACCUMULATION,step=window['start_step'],writer=writer,
                           monitor=monitor,wait_seconds=args.shard_wait_seconds)
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model = base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    common = base.objective(); frozen = screen.frozen_versions(model); teacher_frozen = resume.continuation.teacher_versions(teacher)
    optimizer,restoration = restart_checkpoint(model,parent)
    boundary_panel,_ = base.select_boundary_panel(development,source_metadata)
    seen,seen_set,samples,cache_count,warmed,reviews = [],set(window['historical']),0,0,set(),[]
    started = time.monotonic(); failure = None; status = 'running'; last_receipt = None
    def log_report(report,step):
        base.log_validation(writer,report,step,source_metadata); monitor.log_validation(report,step)
        if hasattr(gates,'log_quiet_regions'): gates.log_quiet_regions(writer,report['quiet_regions'],step)
        writer.add_scalar('overview/Training progress to 10000 steps (%)',100*step/END_STEP,step)
        writer.flush()
    try:
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            initial = evaluate_review(monitor,model,teacher,development,common)
        actual = {k:v for k,v in initial.items() if k != 'quiet_regions'} if 'quiet_regions' not in reference else initial
        check = previous.prior.previous.numeric_reference_check(actual,reference)
        base.write_json(args.out/'restore-check.json',{'state':restoration,'saved_quality':check})
        if not check['passed']: raise RuntimeError('Restored saved96-source quality differs')
        initial_path = args.out/f'development-step{window["start_step"]}.json'; base.write_json(initial_path,initial)
        if not history:
            history.append(report_entry(initial_path,ROOT_STEP)); reports.append(initial); steps.append(ROOT_STEP)
        log_report(initial,window['start_step'])
        base.event('joint_continuation_started',step=window['start_step'],fresh_cursor=window['start_cursor'],target=window['target_step'])
        for update in range(1,window['updates']+1):
            step = window['start_step']+update; cursor = window['start_cursor']+len(seen)
            crops = resume.take_when_ready(data,cursor,ACCUMULATION,step=step,writer=writer,monitor=monitor,
                                           wait_seconds=args.shard_wait_seconds)
            chosen = [c['source_id'] for c in crops]
            if chosen != window['source_ids'][len(seen):len(seen)+ACCUMULATION] or set(chosen)&seen_set:
                raise RuntimeError('Continuation would repeat or reorder a source')
            for crop in crops:
                shape = tuple(crop['latents'].shape)
                if shape not in warmed:
                    with replay.diagnostic_state_guard(model,teacher,optimizer), torch.no_grad():
                        z,_,_,_ = base.batch([crop]); trace = base.teacher_forward(teacher,z)
                        for _ in range(3): model.suffix_from_group(model.group_from_input(trace['group_input']))
                    warmed.add(shape)
            before = time.monotonic()
            with replay.observe_teacher_cache(crops) as checks:
                values = screen.training_update(model,teacher,crops,'current',common,identity['coefficients'],optimizer,
                                                record_diagnostics=update==1 or update%25==0)
            if len(checks) != ACCUMULATION or not all(c['allclose_original_tolerance'] for c in checks):
                raise RuntimeError('Teacher differs from authenticated cached targets')
            cache_count += len(checks); seen.extend(chosen); seen_set.update(chosen)
            samples += sum(c['valid_scored_samples'] for c in crops)
            record = {'step':step,'updates':update,**values,'source_ids':chosen,'unique_sources':len(seen_set),
                'additional_sources':len(seen),'fresh_cursor':cursor+ACCUMULATION,'scored_samples':samples,
                'audio_hours':samples/48000/3600,'elapsed_seconds':time.monotonic()-started,
                'step_seconds':time.monotonic()-before,'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            base.log_training(writer,record,identity['coefficients'],identity['learning_rate'],step)
            if update==1 or update%25==0:
                monitor.log_training(record,step)
                writer.add_scalar('overview/Training progress to 10000 steps (%)',100*step/END_STEP,step)
                writer.flush()
                base.event('joint_continuation_progress',step=step,updates=update,fresh_cursor=cursor+ACCUMULATION,
                           total=values['total'],elapsed_seconds=record['elapsed_seconds'])
            if (step-ROOT_STEP)%EVERY == 0 or step == END_STEP:
                with replay.diagnostic_state_guard(model,teacher,optimizer):
                    report = evaluate_review(monitor,model,teacher,development,common)
                    boundaries = base.evaluate_boundaries(teacher,model,boundary_panel,selection,base.batch,base.teacher_forward)
                reports.append(report); steps.append(step)
                path = args.out/f'development-step{step}.json'; base.write_json(path,report); history.append(report_entry(path,step))
                base.write_json(args.out/f'boundaries-step{step}.json',boundaries)
                log_report(report,step); base.log_boundaries(writer,boundaries,step)
                if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:
                    raise RuntimeError('Frozen teacher or decoder state changed')
                data.assert_unchanged()
                review = gates.classify_review(reports,steps); review.update(step=step,updates=update)
                if review.get('action') not in ('continue','pause_for_diagnosis'): raise ValueError('Unknown review action')
                reviews.append(review); base.write_json(args.out/'reviews.json',reviews)
                checkpoint_path = args.out/f'checkpoint-step{step}.pt'
                last_receipt = save_state(checkpoint_path,model,optimizer,parent,seen,samples,identity,history,review)
                last_receipt.update(frozen_state_preserved=True,teacher_cache_checks=cache_count,quality=report['aggregate'])
                base.write_json(checkpoint_path.with_suffix('.json'),last_receipt)
                # The provider may now retire only its own consumed rolling-cache files.
                data.checkpoint_committed(checkpoint_path,last_receipt['checkpoint_sha256'],cursor+ACCUMULATION)
                base.event('joint_continuation_review',**review); writer.add_text('Monitor/Recovery review',json.dumps(review,indent=2),step); writer.flush()
                if review['action'] != 'continue': status='paused_for_diagnosis'; break
        else: status = 'completed_target' if window['target_step']==END_STEP else 'awaiting_review'
        if any(base.sha(p)!=s for p,s in protected.items()): raise RuntimeError('Protected input/source files changed')
        if last_receipt is None: raise RuntimeError('No durable reviewed checkpoint was saved')
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_updates':len(seen)//ACCUMULATION}
        base.write_json(args.out/'failure.json',failure); raise
    finally:
        writer.flush(); writer.close()
        preservation={'failure':failure,'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_frozen,
            'frozen_student_preserved':screen.frozen_versions(model)==frozen,
            'original_files_preserved':all(base.sha(p)==s for p,s in protected.items())}
        base.write_json(args.out/'preservation.json',preservation)
    if not all(preservation[k] for k in ('teacher_preserved','frozen_student_preserved','original_files_preserved')):
        raise RuntimeError('Preservation failed; completion receipt withheld')
    base.write_json(args.out/'completed.json',{'version':VERSION,'status':status,'step':window['start_step']+len(seen)//ACCUMULATION,
        'updates':len(seen)//ACCUMULATION,'sources':len(seen),'fresh_cursor':window['start_cursor']+len(seen),
        'total_source_count':len(seen_set),'scored_samples':samples,'elapsed_seconds':time.monotonic()-started,
        'initial':initial['aggregate'],'final':reports[-1]['aggregate'],'reviews':reviews,
        'last_checkpoint_sha256':last_receipt['checkpoint_sha256'],'teacher_cache_checks':cache_count,
        'frozen_state_preserved':True,'original_files_preserved':True,'automatic_promotion':False,
        'external_review_required':status=='awaiting_review'})


if __name__ == '__main__': main()
