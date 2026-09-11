"""Bounded joint recovery from the preserved accumulation12 decoder.

No architecture, objective, optimizer, or inference changes. Every 250 updates
save a complete state and evaluate the same development panel before deciding
whether to continue. New sources extend the candidate's exact data lineage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import compare_projected_hints_v1 as prior
from unified_monitor import UnifiedMonitor

base, screen, replay, resume = prior.base, prior.screen, prior.replay, prior.resume
VERSION = 'audiovae2_joint_recovery_v1'
START_STEP, START_SOURCE, ACCUMULATION, MAX_UPDATES, EVERY = 4625, 12000, 12, 1000, 250


def validate_source_window(candidate, original_ids, fresh_ids, start=START_SOURCE,
                           updates=MAX_UPDATES, accumulation=ACCUMULATION):
    if (type(updates) is not int or updates <= 0 or updates > MAX_UPDATES or updates % EVERY
            or type(start) is not int or start != START_SOURCE
            or type(accumulation) is not int or accumulation != ACCUMULATION):
        raise ValueError('Expected 250..1000 updates, accumulation12, starting at fresh12000')
    historical = list(candidate['historical_sources_seen']) + list(candidate['additional_sources_seen'])
    expected = list(original_ids) + list(fresh_ids[:START_SOURCE])
    if len(original_ids) != 3000 or historical != expected or len(set(historical)) != len(historical):
        raise ValueError('Candidate history is not the exact 15000-source lineage')
    chosen = list(fresh_ids[start:start+updates*accumulation])
    if (len(chosen) != updates*accumulation or len(set(chosen)) != len(chosen)
            or set(chosen) & set(historical)):
        raise ValueError('Recovery sources are missing, repeated, or already seen')
    return historical, chosen


def save_state(path, model, optimizer, candidate, seen, samples, identity):
    if path.exists(): raise FileExistsError(path)
    historical = candidate['historical_sources_seen']+candidate['additional_sources_seen']
    if (not seen or len(seen)>MAX_UPDATES*ACCUMULATION or len(set(seen))!=len(seen)
            or set(seen)&set(historical)):
        raise ValueError('Invalid saved source history')
    if (identity.get('source_interval',[None,None])[0]!=START_SOURCE
            or identity['source_interval'][1]<START_SOURCE+len(seen)
            or identity.get('source_ids_sha256') is None):
        raise ValueError('Saved source plan differs')
    if START_SOURCE+len(seen)==identity['source_interval'][1] and screen.digest(seen)!=identity['source_ids_sha256']:
        raise ValueError('Final source digest differs')
    params = base.parameters(model)
    if not params or set(optimizer.state)!=set(params):
        raise ValueError('Missing or extra optimizer parameter state')
    updates = len(seen)//ACCUMULATION
    step = START_STEP+updates
    if len(seen) % ACCUMULATION or any(float(s['step']) != step for s in optimizer.state.values()):
        raise RuntimeError('Saved AdamW counters do not match complete updates')
    artifact = {'format': VERSION, 'group': {k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer': optimizer.state_dict(), 'rng': screen.rng_state(), 'optimizer_step': step,
        'updates': updates, 'accumulation': ACCUMULATION, 'scored_samples': samples,
        'historical_sources_seen': historical,
        'additional_sources_seen': list(seen), 'original_training_identity': candidate['original_training_identity'],
        'identity': identity, 'automatic_promotion': False}
    temporary = path.with_suffix('.tmp')
    torch.save(artifact, temporary)
    temporary.replace(path)
    return {'checkpoint_sha256': base.sha(path), 'optimizer_step': step, 'updates': updates,
            'sources_consumed': len(seen), 'optimizer_and_rng_saved': True}


def evaluate_review(monitor, model, teacher, crops, common):
    """Observe continuous near-silence error in the existing scoring pass."""
    original = base.quiet_window_metrics
    captured = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        near = [w for w in result['windows'] if w['is_quiet'] and w['teacher_rms']<=1e-5]
        count = sum(w['valid_samples'] for w in near)
        power = sum(w['valid_samples']*w['residual_rms']**2 for w in near)
        captured.append({'near_samples':count,'near_error_sum':power,
                         'near_residual_rms':math.sqrt(power/count) if count else None})
        return result
    base.quiet_window_metrics = observe
    try: report = monitor.evaluate(model,teacher,crops,common)
    finally: base.quiet_window_metrics = original
    if len(captured)!=len(report['rows']): raise RuntimeError('Near-silence observation count differs')
    total = sum(r['near_samples'] for r in captured)
    power = sum(r['near_error_sum'] for r in captured)
    report['recovery_window_metrics'] = {'near_samples':total,'near_error_sum':power,
        'near_residual_rms':math.sqrt(power/total) if total else None,
        'by_source':{r['source_id']:v for r,v in zip(report['rows'],captured)}}
    return report


def main():
    from joint_recovery_gates_v1 import classify_review
    p = argparse.ArgumentParser()
    for name in ('checkpoint','candidate','anchor-checkpoint','screen-out','base-out','manifest',
                 'fresh-manifest','shards','assets','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--updates', type=int, default=MAX_UPDATES)
    p.add_argument('--shard-wait-seconds', type=int, default=1800)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new recovery directory')
    base.policy()
    _, selection, _, manifest, pools, original, _, original_sha, _ = resume.authenticate(args)
    data = replay.FreshTrainingData(args.fresh_manifest, args.manifest, pools, args.shards)
    old_ids = list(data.source_ids[10500:12000])
    candidate, reference, candidate_sha = prior.authenticate_candidate(args.candidate, original, old_ids)
    if (candidate['identity']['start_checkpoint_sha256'] != original_sha
            or candidate['identity']['runner_sha256'] != base.sha(prior.previous.__file__)
            or candidate['identity']['resume_identity'] != original['resume_identity']
            or candidate['identity']['source_ids'] != old_ids):
        raise ValueError('Candidate parent checkpoint, source, or recipe identity changed')
    historical, ids = validate_source_window(candidate, [c['source_id'] for c in pools['fit']],
                                            data.source_ids, updates=args.updates)
    development = pools['development']
    if len(development) != 96: raise ValueError('Expected the fixed96-source panel')
    if set(ids) & {c['source_id'] for c in pools['calibration']+development}:
        raise ValueError('Calibration or development leakage')
    protected_paths = [args.checkpoint,args.candidate,args.anchor_checkpoint,args.manifest,args.fresh_manifest,
        args.candidate.parent/'completed.json',args.candidate.parent.parent/'completed.json',
        args.candidate.parent/'development-source1500.json',Path(__file__),Path(base.__file__),
        Path(screen.__file__),Path(replay.__file__),Path(prior.__file__),Path(resume.__file__),
        Path(__file__).with_name('joint_recovery_gates_v1.py')]
    protected = {str(path):base.sha(path) for path in protected_paths}
    args.out.mkdir(parents=True)
    identity = {'version': VERSION, 'candidate_sha256': candidate_sha, 'starting_optimizer_step': START_STEP,
        'target_optimizer_step': START_STEP+args.updates, 'source_interval': [START_SOURCE,START_SOURCE+len(ids)],
        'source_ids_sha256': screen.digest(ids), 'gradient_accumulation': ACCUMULATION, 'execution_batch_size': 1,
        'coefficients': candidate['original_training_identity']['coefficients'], 'learning_rate': 3e-5,
        'protected': protected, 'review_every_updates': EVERY, 'trainable_stages': [2,3,4],
        'torch': str(torch.__version__), 'cudnn': torch.backends.cudnn.version(), 'backend': replay.backend_state(),
        'conditional_reserve_authorized_for_this_continuation': [START_SOURCE,START_SOURCE+len(ids)],
        'automatic_promotion': False, 'automatic_freezing': False}
    base.write_json(args.out/'launch.json', identity)
    # Wait without holding teacher/student GPU memory while the first shard is prepared.
    wait_start = time.monotonic()
    while not (args.shards/f'{START_SOURCE:06d}-{START_SOURCE+300:06d}'/'receipt.json').is_file():
        if time.monotonic()-wait_start > args.shard_wait_seconds: raise TimeoutError('First teacher shard not sealed')
        base.event('waiting_for_teacher_shard', source=START_SOURCE)
        time.sleep(15)
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth', device='cuda')
    model = base.build_student(teacher.model.decoder, selection['stage2_indices'], selection['stage3_indices'])
    common = base.objective()
    frozen, teacher_frozen = screen.frozen_versions(model), resume.continuation.teacher_versions(teacher)
    optimizer, restoration = prior.restart_candidate(model, candidate)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tensorboard'/'raw'), flush_secs=10)
    source_metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    baseline = json.loads((args.screen_out/'current_lr3e-5'/'development-step0.json').read_text())['aggregate']
    monitor = UnifiedMonitor(writer, baseline, source_metadata)
    monitor.initialize_layout()
    screen.restore_rng(candidate['rng'])
    boundary_panel, _ = base.select_boundary_panel(development, source_metadata)
    seen, samples, reports, reviews = [], 0, [], []
    seen_set = set(historical)
    warmed = set()
    started = time.monotonic()
    failure = None
    cache_count = 0
    status = 'running'
    try:
        with replay.diagnostic_state_guard(model, teacher, optimizer):
            initial = evaluate_review(monitor, model, teacher, development, common)
        check = prior.previous.numeric_reference_check({k:v for k,v in initial.items() if k!='recovery_window_metrics'}, reference)
        base.write_json(args.out/'restore-check.json', {'state':restoration,'saved_quality':check})
        if not check['passed']: raise RuntimeError('Restored96-source quality differs from preserved checkpoint')
        reports.append(initial)
        base.write_json(args.out/f'development-step{START_STEP}.json', initial)
        base.log_validation(writer, initial, START_STEP, source_metadata)
        monitor.log_validation(initial, START_STEP)
        writer.add_scalar('overview/Recovery progress to 1000 updates (%)', 0, START_STEP)
        writer.flush()
        base.event('joint_recovery_started', step=START_STEP, candidate_sha256=candidate_sha)
        for update in range(1, args.updates+1):
            step = START_STEP+update
            cursor = START_SOURCE+len(seen)
            crops = resume.take_when_ready(data, cursor, ACCUMULATION, step=step, writer=writer, monitor=monitor,
                                           wait_seconds=args.shard_wait_seconds)
            chosen = [c['source_id'] for c in crops]
            if chosen != ids[len(seen):len(seen)+ACCUMULATION] or set(chosen)&seen_set:
                raise RuntimeError('Recovery would repeat or reorder a source')
            for crop in crops:
                shape = tuple(crop['latents'].shape)
                if shape not in warmed:
                    with replay.diagnostic_state_guard(model, teacher, optimizer), torch.no_grad():
                        z,_,_,_ = base.batch([crop]); trace = base.teacher_forward(teacher,z)
                        for _ in range(3): model.suffix_from_group(model.group_from_input(trace['group_input']))
                    warmed.add(shape)
            before = time.monotonic()
            with replay.observe_teacher_cache(crops) as checks:
                values = screen.training_update(model, teacher, crops, 'current', common, identity['coefficients'],
                                                optimizer, record_diagnostics=update==1 or update%25==0)
            if not all(c['allclose_original_tolerance'] for c in checks):
                raise RuntimeError('Teacher output differs from authenticated cached targets')
            cache_count += len(checks)
            seen.extend(chosen); seen_set.update(chosen)
            samples += sum(c['valid_scored_samples'] for c in crops)
            record = {'step':step,'updates':update,**values,'source_ids':chosen,'unique_sources':len(seen_set),
                'additional_sources':len(seen),'audio_hours':samples/48000/3600,'scored_samples':samples,
                'elapsed_seconds':time.monotonic()-started,'step_seconds':time.monotonic()-before,
                'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            base.log_training(writer, record, identity['coefficients'], identity['learning_rate'], step)
            if update==1 or update%25==0:
                monitor.log_training(record, step)
                writer.add_scalar('overview/Recovery progress to 1000 updates (%)', update/10, step)
                writer.flush()
                base.event('joint_recovery_progress',step=step,updates=update,sources=len(seen),
                           total=values['total'],elapsed_seconds=record['elapsed_seconds'])
            if update%EVERY==0:
                with replay.diagnostic_state_guard(model, teacher, optimizer):
                    report = evaluate_review(monitor, model, teacher, development, common)
                    boundaries = base.evaluate_boundaries(teacher, model, boundary_panel, selection, base.batch, base.teacher_forward)
                reports.append(report)
                base.write_json(args.out/f'development-step{step}.json',report)
                base.write_json(args.out/f'boundaries-step{step}.json',boundaries)
                base.log_validation(writer, report, step, source_metadata)
                base.log_boundaries(writer, boundaries, step)
                monitor.log_validation(report, step)
                if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:
                    raise RuntimeError('Frozen teacher or decoder modules changed')
                data.assert_unchanged()
                receipt = save_state(args.out/f'checkpoint-step{step}.pt',model,optimizer,candidate,seen,samples,identity)
                receipt.update({'frozen_state_preserved':True,'teacher_cache_checks':cache_count,'quality':report['aggregate']})
                base.write_json(args.out/f'checkpoint-step{step}.json',receipt)
                review = classify_review(reports)
                review.update({'step':step,'updates':update})
                reviews.append(review)
                base.write_json(args.out/'reviews.json',reviews)
                base.event('joint_recovery_review',**review)
                writer.add_text('Monitor/Recovery review',json.dumps(review,indent=2),step)
                writer.flush()
                if review['action'] != 'continue':
                    status = 'paused_for_diagnosis'; break
        else: status = 'completed_budget'
        if any(base.sha(p)!=s for p,s in protected.items()): raise RuntimeError('Protected original files changed')
        base.write_json(args.out/'completed.json', {'version':VERSION,'status':status,'step':START_STEP+len(seen)//ACCUMULATION,
            'updates':len(seen)//ACCUMULATION,'sources':len(seen),'scored_samples':samples,'elapsed_seconds':time.monotonic()-started,
            'initial':initial['aggregate'],'final':reports[-1]['aggregate'],'reviews':reviews,
            'teacher_cache_checks':cache_count,'frozen_state_preserved':True,'original_files_preserved':True,'automatic_promotion':False})
    except BaseException as exc:
        failure = {'type':type(exc).__name__,'message':str(exc),'completed_updates':len(seen)//ACCUMULATION}
        base.write_json(args.out/'failure.json',failure)
        raise
    finally:
        writer.flush(); writer.close()
        base.write_json(args.out/'preservation.json',{'failure':failure,'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_frozen,
            'frozen_student_preserved':screen.frozen_versions(model)==frozen,
            'original_files_preserved':all(base.sha(p)==s for p,s in protected.items())})


if __name__ == '__main__': main()
