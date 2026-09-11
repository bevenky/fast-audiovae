"""Matched-source accumulation3/12 diagnostic, starting from original step4500.

Both arms call the unchanged singleton, sample/element-pooled training update.
The main trainer and its checkpoints are never changed or automatically resumed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch
import replay_late_segment as replay
from unified_monitor import UnifiedMonitor

base, screen, resume = replay.base, replay.screen, replay.resume
VERSION = 'audiovae2_accumulation_comparison_v1'
START_STEP, FRESH_START, SOURCES, EVERY = 4500, 10500, 1500, 60
ARMS = (3, 12)


def comparison_schedule(accumulation, total_sources=SOURCES, every_sources=EVERY, start_step=START_STEP):
    if (type(accumulation) is not int or accumulation not in ARMS
            or type(total_sources) is not int or total_sources <= 0
            or type(every_sources) is not int or every_sources <= 0
            or total_sources % accumulation or every_sources % accumulation
            or total_sources % every_sources or type(start_step) is not int or start_step < 0):
        raise ValueError('Source checkpoints must align with complete accumulation groups')
    return [{'sources_consumed': n, 'optimizer_step': start_step+n//accumulation,
             'updates': n//accumulation} for n in range(0, total_sources+1, every_sources)]


def source_batch(fresh_ids, consumed, accumulation):
    if (type(consumed) is not int or consumed < 0 or consumed >= SOURCES
            or type(accumulation) is not int or accumulation not in ARMS
            or consumed % accumulation or consumed+accumulation > SOURCES):
        raise ValueError('Invalid matched-source cursor')
    a, b = FRESH_START+consumed, FRESH_START+consumed+accumulation
    ids = list(fresh_ids[a:b])
    if len(ids) != accumulation or len(set(ids)) != accumulation:
        raise ValueError('Missing or repeated fresh sources')
    return (a, b), ids


def restart_arm(model, payload):
    if payload.get('step') != START_STEP: raise ValueError('Both arms start at original step4500')
    optimizer = resume.continuation.restore_training_state(model, payload)
    comparisons = {
        'group': replay.compare_tree(dict(model.group_state_dict()), dict(payload['group'])),
        'optimizer': replay.compare_tree(optimizer.state_dict(), payload['optimizer']),
        'rng': replay.compare_tree(screen.rng_state(), payload['rng']),
    }
    if not all(c['equal'] for c in comparisons.values()):
        raise RuntimeError('Arm did not restore the exact common state: '+str(comparisons))
    return optimizer, comparisons


def run_update(model, teacher, crops, common, coefficients, optimizer):
    if len(crops) not in ARMS: raise ValueError('Only accumulation3 or12 is allowed')
    # The original update already divides each singleton by the GLOBAL pooled
    # sample/element denominator. Do not divide the summed objective again.
    return replay.replay_step(model, teacher, crops, common, coefficients, optimizer,
                              record_diagnostics=True)


def numeric_reference_check(actual, expected, path='root'):
    """The existing saved-metric tolerance, with exact integer/count semantics."""
    failures = []
    def visit(a, b, where):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b): failures.append(where+'/keys'); return
            for key in b: visit(a[key], b[key], where+'/'+str(key))
        elif isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b): failures.append(where+'/sequence'); return
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, where+'/'+str(i))
        elif type(b) is float:
            if (not isinstance(a, (float, int)) or isinstance(a, bool) or not math.isfinite(a)
                    or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)):
                failures.append(where)
        elif type(a) is not type(b) or a != b: failures.append(where)
    visit(actual, expected, path)
    return {'passed': not failures, 'failures': failures, 'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-6}


def trajectory_summary(snapshots):
    """Descriptive equal-exposure measurements, never best-checkpoint selection."""
    result = {}
    for source_id in replay.CASE_IDS:
        rows = [next(r for r in s['cases'] if r['source_id'] == source_id) for s in snapshots]
        gains = [r['active_rms_gain'] for r in rows]
        if any(g is None or g <= 0 or not math.isfinite(g) for g in gains):
            raise ValueError('Fixed active case has invalid RMS gain')
        logs = [abs(math.log(g)) for g in gains]
        mean = sum(gains)/len(gains)
        differences = [b-a for a,b in zip(gains,gains[1:])]
        signs = [1 if x > 0 else -1 for x in differences if x != 0]
        result[source_id] = {'observations': len(rows), 'initial_gain': gains[0], 'final_gain': gains[-1],
            'gain_min': min(gains), 'gain_max': max(gains), 'gain_mean': mean,
            'gain_population_std': math.sqrt(sum((g-mean)**2 for g in gains)/len(gains)),
            'gain_total_variation': sum(abs(x) for x in differences),
            'gain_direction_reversals': sum(a != b for a,b in zip(signs,signs[1:])),
            'mean_absolute_log_gain': sum(logs)/len(logs),
            'max_absolute_log_gain': max(logs),
            'mean_waveform_mae': sum(r['mae'] for r in rows)/len(rows),
            'initial_waveform_mae': rows[0]['mae'], 'final_waveform_mae': rows[-1]['mae'],
            'mean_active_cosine': sum(r['active_cosine'] for r in rows)/len(rows)}
    return result


def weighted_quality(report, coefficients):
    a = report['aggregate']
    return coefficients['waveform']*a['mae']+coefficients['mel']*a['mel']+coefficients['feature']*a['group_mse']


def save_arm(path, model, optimizer, payload, accumulation, ids, samples, identity):
    if path.exists(): raise FileExistsError(path)
    step = START_STEP+len(ids)//accumulation
    counters = [float(s['step']) for s in optimizer.state.values()]
    if not counters or any(n != step for n in counters): raise RuntimeError('AdamW counters differ from actual updates')
    artifact = {'format': VERSION, 'group': {k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer': optimizer.state_dict(), 'rng': screen.rng_state(), 'optimizer_step': step,
        'accumulation': accumulation, 'updates': len(ids)//accumulation, 'sources_consumed': len(ids),
        'scored_samples': samples, 'additional_sources_seen': list(ids),
        'historical_sources_seen': list(payload['sources_seen']), 'identity': identity,
        'original_training_identity': payload['identity'], 'automatic_promotion': False}
    torch.save(artifact, path)
    return {'checkpoint_sha256': base.sha(path), 'optimizer_step': step, 'optimizer_parameter_counters': counters}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint','expected','anchor-checkpoint','screen-out','base-out','manifest','fresh-manifest','shards','assets','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new isolated comparison directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < 600<<20: raise OSError('Comparison needs600MiB free temporary storage')
    base.policy()
    _, selection, _, manifest, pools, payload, reference, start_sha, _ = resume.authenticate(args)
    expected_sha = base.sha(args.expected)
    expected = torch.load(args.expected, map_location='cpu', weights_only=True, mmap=True)
    receipt = json.loads(args.expected.with_suffix('.json').read_text())
    if (receipt['checkpoint_sha256'] != expected_sha or receipt['step'] != 5000
            or receipt['resume_identity_sha256'] != screen.digest(expected['resume_identity'])
            or receipt['frozen_state_preserved'] is not True): raise ValueError('Original endpoint receipt changed')
    data = replay.FreshTrainingData(args.fresh_manifest, args.manifest, pools, args.shards)
    replay.replay_window(payload, expected, [r['source_id'] for r in pools['fit']], data.source_ids)
    execution = payload['resume_identity']
    required = {'precision':'float32','tf32':False,'learning_rate':3e-5,'approved_source_range':[0,12000],
        'fresh_plan_identity_sha256':data.identity,'original_manifest_sha256':base.sha(args.manifest),
        'fresh_manifest_sha256':base.sha(args.fresh_manifest),'resume_sha256':base.sha(resume.__file__),
        'fresh_loader_sha256':base.sha(Path(__file__).with_name('fresh_training_data.py')),
        'monitor_sha256':base.sha(Path(__file__).with_name('unified_monitor.py'))}
    if any(execution.get(k) != v for k,v in required.items()): raise ValueError('Original execution contract changed')
    if payload['identity']['definition'] != 'current': raise ValueError('This comparison preserves the current recipe')
    chosen_ids = list(data.source_ids[FRESH_START:FRESH_START+SOURCES])
    if len(chosen_ids) != SOURCES or len(set(chosen_ids)) != SOURCES: raise ValueError('Comparison requires1500 unique sources')
    # Validate all required sealed shards before the first optimizer update.
    for cursor in range(FRESH_START, FRESH_START+SOURCES, 300): data.take(cursor,300)
    development = {c['source_id']:c for c in pools['development']}
    if len(development) != 96 or len(pools['development']) != 96: raise ValueError('The fixed96-source panel changed')
    cases = [development[s] for s in replay.CASE_IDS]
    row_metadata = {r['source_id']:r for r in data.plan['rows']}
    protected_paths = (args.checkpoint,args.expected,args.anchor_checkpoint,args.manifest,args.fresh_manifest,Path(__file__),
        args.checkpoint.with_suffix('.json'),args.expected.with_suffix('.json'),args.checkpoint.parent/'train.jsonl',
        args.checkpoint.parent/'development-step4500.json',Path(replay.__file__),Path(base.__file__),Path(screen.__file__))
    protected = {str(p):base.sha(p) for p in protected_paths}
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model = base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    common = base.objective()
    monitor = UnifiedMonitor(None, {}, {})
    frozen, teacher_frozen = screen.frozen_versions(model), resume.continuation.teacher_versions(teacher)
    coefficients = payload['identity']['coefficients']
    identity = {'version':VERSION,'runner_sha256':base.sha(__file__),'start_checkpoint_sha256':start_sha,
        'source_interval':[FRESH_START,FRESH_START+SOURCES],'source_ids_sha256':screen.digest(chosen_ids),
        'source_ids':chosen_ids,'arms':list(ARMS),'execution_batch_size':1,'learning_rate':3e-5,
        'coefficients':coefficients,'starting_optimizer_step':START_STEP,'case_ids':list(replay.CASE_IDS),
        'development_ids':list(development),'checkpoints_sources':list(range(0,SOURCES+1,EVERY)),
        'backend':replay.backend_state(),'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),
        'protected_files_sha256':protected,'original_training_identity':payload['identity'],
        'resume_identity':execution,'automatic_promotion':False,
        'interpretation':'Equal source exposure; accumulation also changes AdamW update count and moment history per source.'}
    args.out.mkdir()
    base.write_json(args.out/'launch.json',identity)
    arm_results = {}; baseline_first = None; started = time.monotonic(); failure = None
    try:
        for accumulation in ARMS:
            arm_name = f'accumulation{accumulation}'; folder = args.out/arm_name; folder.mkdir()
            optimizer, restoration = restart_arm(model,payload)
            model._replay_case_warmed_shapes = set()
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                initial = monitor.evaluate(model,teacher,pools['development'],common)
                initial_cases = replay.evaluate_cases(model,teacher,cases)
            original_check = numeric_reference_check(initial,reference)
            paired_check = original_check if baseline_first is None else numeric_reference_check(initial,baseline_first)
            base.write_json(folder/'development-source0.json',initial)
            base.write_json(folder/'restoration.json',{'state':restoration,'original_quality':original_check,'paired_quality':paired_check})
            if not original_check['passed'] or not paired_check['passed']:
                raise RuntimeError('Common initial96-source quality does not match the saved tolerance')
            if baseline_first is None: baseline_first = initial
            schedule = comparison_schedule(accumulation); by_exposure={r['sources_consumed']:r for r in schedule}
            snapshots = [{**schedule[0],'scored_samples':0,'cases':initial_cases}]
            base.write_json(folder/'case-trajectories.json',snapshots)
            seen=[]; samples=0; warmed=set(); cache_checks=[]; update_count=0; arm_start=time.monotonic()
            for consumed in range(0,SOURCES,accumulation):
                (a,b),ids=source_batch(data.source_ids,consumed,accumulation); crops=data.take(a,b-a)
                if [c['source_id'] for c in crops] != ids: raise ValueError('Loaded crops differ from sealed source order')
                for crop in crops:
                    shape=tuple(crop['latents'].shape)
                    if shape not in warmed:
                        with replay.diagnostic_state_guard(model,teacher,optimizer),torch.no_grad():
                            z,_,_,_=base.batch([crop]);trace=base.teacher_forward(teacher,z)
                            for _ in range(3):model.suffix_from_group(model.group_from_input(trace['group_input']))
                        warmed.add(shape)
                levels=replay.cached_levels(crops,row_metadata)
                with replay.observe_teacher_cache(crops) as checks:
                    values,direction=run_update(model,teacher,crops,common,coefficients,optimizer)
                seen.extend(ids);samples+=sum(c['valid_scored_samples'] for c in crops);update_count+=1
                cache_checks.extend(checks)
                record={'sources_consumed':len(seen),'optimizer_step':START_STEP+update_count,'updates':update_count,
                    'accumulation':accumulation,'scored_samples':samples,'source_ids':ids,'values':values,'direction':direction,
                    'sources':levels,'teacher_cache_checks':checks,'elapsed_seconds':time.monotonic()-arm_start}
                with (folder/'train.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
                if len(seen) in by_exposure:
                    with replay.diagnostic_state_guard(model,teacher,optimizer):
                        measured=replay.evaluate_cases(model,teacher,cases)
                    snapshots.append({**by_exposure[len(seen)],'scored_samples':samples,'cases':measured})
                    base.write_json(folder/'case-trajectories.json',snapshots)
                    base.event('accumulation_progress',arm=arm_name,sources=len(seen),optimizer_step=START_STEP+update_count,
                               elapsed_seconds=time.monotonic()-arm_start)
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                final=monitor.evaluate(model,teacher,pools['development'],common)
            base.write_json(folder/'development-source1500.json',final)
            if seen != chosen_ids or len(snapshots) != 26: raise RuntimeError('Incomplete matched exposure or diagnostic schedule')
            saved=save_arm(folder/'final.pt',model,optimizer,payload,accumulation,seen,samples,identity)
            if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:
                raise RuntimeError('Frozen modules changed')
            data.assert_unchanged()
            if any(base.sha(path)!=value for path,value in protected.items()): raise RuntimeError('Protected original file changed')
            result={'arm':arm_name,**saved,'updates':update_count,'sources_consumed':len(seen),'scored_samples':samples,
                'scored_audio_seconds':samples/48000,'initial':initial['aggregate'],'final':final['aggregate'],
                'initial_near':{k:v for k,v in initial['overview_window_metrics'].items() if k!='by_source'},
                'final_near':{k:v for k,v in final['overview_window_metrics'].items() if k!='by_source'},
                'initial_weighted_objective':weighted_quality(initial,coefficients),'final_weighted_objective':weighted_quality(final,coefficients),
                'trajectory':trajectory_summary(snapshots),'teacher_cache_sources':len(cache_checks),
                'teacher_cache_all_bitwise':all(c['bitwise_equal'] for c in cache_checks),
                'teacher_cache_all_original_tolerance':all(c['allclose_original_tolerance'] for c in cache_checks),
                'teacher_cache_max_abs':max(c['max_abs'] for c in cache_checks),'frozen_state_preserved':True,
                'starting_state_exact':True,'original_initial_quality_passed':original_check['passed'],
                'paired_initial_quality_passed':paired_check['passed'],
                'elapsed_seconds':time.monotonic()-arm_start,'automatic_promotion':False}
            base.write_json(folder/'completed.json',result);arm_results[arm_name]=result
            base.event('accumulation_arm_complete',arm=arm_name,optimizer_step=saved['optimizer_step'],quality=final['aggregate'])
        if arm_results['accumulation3']['scored_samples']!=arm_results['accumulation12']['scored_samples']:
            raise RuntimeError('Arms consumed different audio exposure')
        base.write_json(args.out/'completed.json',{'version':VERSION,'arms':arm_results,'matched_sources':True,
            'matched_scored_samples':True,'original_files_preserved':True,'frozen_state_preserved':True,
            'comparison_valid':all(r['teacher_cache_all_original_tolerance'] and r['starting_state_exact']
                and r['original_initial_quality_passed'] and r['paired_initial_quality_passed'] for r in arm_results.values()),
            'elapsed_seconds':time.monotonic()-started,'automatic_promotion':False,
            'limits':['One fixed1500-source segment, not generalization across seeds or training periods.',
                'Accumulation changes both update variance and the number of AdamW updates per source.',
                'No historical bitwise replay claim; the two arms use the same new instrumented path.',
                'Intermediate four-case data are diagnostic; no best checkpoint is selected.']})
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_arms':list(arm_results)}
        base.write_json(args.out/'failure.json',failure);raise
    finally:
        base.write_json(args.out/'preservation.json',{'frozen_state_preserved':screen.frozen_versions(model)==frozen,
            'teacher_state_preserved':resume.continuation.teacher_versions(teacher)==teacher_frozen,
            'original_files_preserved':all(base.sha(path)==value for path,value in protected.items()),'failure':failure})


if __name__=='__main__':main()
