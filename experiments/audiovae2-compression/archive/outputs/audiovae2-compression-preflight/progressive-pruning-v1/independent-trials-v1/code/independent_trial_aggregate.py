"""CPU-only completion audit for D/G/Q; emit only allowlisted aggregates.

Usage: python independent_trial_aggregate.py --run RESULTS --reference RESULTS
No model construction, inference, optimization, audio or latent loading occurs.
Checkpoint tensors and source/window identities remain inside this process.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
from joint_recovery_gates_v2 import REGIONS, summarize_regions, _validate_regions

VERSION='audiovae2_independent_trial_aggregate_v1'
REVIEWS=(0,250,500,1000,1500,2000)
COEFFICIENTS={'waveform':1.,'mel':0.0006674012905982311,'feature':0.009304078923434964}
B_VERSION='audiovae2_reconstruction_b_recovery_v1'
C_VERSION='audiovae2_combined_recovery_v1'
RUNS={'audiovae2_fresh_candidate_recovery_v1':('D',B_VERSION),
      'audiovae2_grail_candidate_recovery_v1':('G',B_VERSION),
      'audiovae2_quiet_candidate_recovery_v1':('Q',C_VERSION)}
METRICS=('sources','samples','nonquiet_samples','mae','mse','waveform_nrmse','mel','group_mse','group_nrmse',
         'nonquiet_cosine_mean','quiet_residual_rms_mean','quiet_windows','quiet_failed_windows','peak_abs_max','overshoot_samples')
REGION_METRICS=('windows','samples','failed','amplitude_failed','residual_rms','centered_residual_rms',
                'teacher_rms','output_rms','output_limit_excess_rms')
CATEGORIES=('passed','residual_only','amplitude_only','both')
TIMES=('training_update_seconds','validation_seconds','waiting_seconds','elapsed_seconds')
FRACTIONS=(1.,.5,.25,.125,.0625,0.)


class AuditError(ValueError):pass


def require(condition,message):
    if not condition:raise AuditError(message)


def finite(value,*,nullable=False):
    require((value is None and nullable) or (type(value) in (int,float) and math.isfinite(value)),
            'Expected a finite scalar aggregate')
    return value


def read(path):return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20),b''):result.update(chunk)
    return result.hexdigest()


def audit_optimizer(saved,initial,identity,step=2000):
    require(saved['identity']==initial['identity']==identity,'Checkpoint identities differ from launch')
    require(initial['step']==0 and initial['sources_seen']==[] and initial['optimizer']['state']=={},'Initial Adam is not fresh')
    require(saved['optimizer']['param_groups']==initial['optimizer']['param_groups'],'Adam options or parameter order changed')
    groups=saved['optimizer']['param_groups']
    expected={'lr':3e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0.,'amsgrad':False,
              'maximize':False,'capturable':False,'differentiable':False}
    require(len(groups)==1 and all(groups[0].get(k)==v for k,v in expected.items()),'Adam recipe changed')
    require(identity['learning_rate']==3e-5 and identity['optimizer_betas']==[.9,.99]
        and identity['optimizer_eps']==1e-8 and identity['weight_decay']==0
        and identity['execution_batch_size']==1 and identity['gradient_accumulation']==12
        and identity['trainable_stages']==[2,3,4] and identity['coefficients']==COEFFICIENTS,'Launch objective or execution policy changed')
    require(saved['accumulation']==initial['accumulation']==12
        and saved['coefficients']==initial['coefficients']==COEFFICIENTS
        and saved['optimizer_reset_after_start'] is False,'Checkpoint objective or optimizer continuity changed')
    require(saved['selection']==initial['selection']==identity['selection'],'Checkpoint channel selection changed')
    require(saved['cut_updates']==saved['global_updates']==step and initial['cut_updates']==initial['global_updates']==0,
            'Checkpoint optimizer counters differ')
    for key in ('teacher_source_sha256','teacher_checkpoint_sha256'):
        require(saved.get(key) and saved[key]==initial.get(key),'Teacher identity changed within recovery')
    require(list(saved['group'])==list(initial['group']) and len(saved['group'])==90,'Group tensor topology differs')
    states=saved['optimizer']['state'];ids=groups[0]['params']
    require(len(ids)==len(set(ids))==len(states)==90 and set(ids)==set(states),'Expected all90 unique Adam states')
    for pid,(name,param) in zip(ids,saved['group'].items()):
        original=initial['group'][name]
        require(torch.is_tensor(param) and param.device.type=='cpu' and param.dtype==torch.float32
            and param.shape==original.shape and original.dtype==torch.float32
            and bool(torch.isfinite(param).all()) and bool(torch.isfinite(original).all()),'Nonfinite or incompatible group tensor')
        state=states[pid]
        require(set(state)=={'step','exp_avg','exp_avg_sq'},'Unexpected Adam state keys')
        require(all(torch.is_tensor(t) and t.device.type=='cpu' and bool(torch.isfinite(t).all()) for t in state.values()),'Nonfinite Adam state')
        require(state['step'].numel()==1 and float(state['step'])==step,'Adam counter differs from completed batches')
        require(state['exp_avg'].shape==state['exp_avg_sq'].shape==param.shape
            and state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
            and bool((state['exp_avg_sq']>=0).all()),'Adam moments do not match their parameter')
    return {'adam_parameter_states':90,'adam_counter':step,'adam_finite':True,'adam_recipe_and_order_unchanged':True,
            'group_tensors_finite':True,'initial_adam_fresh':True,'checkpoint_identities_match_launch':True}


def audit_journal(folder,identity,done):
    expected=done['step'];rows=[];sources=[];samples=nonexact=0
    with (folder/'train.jsonl').open() as handle:
        for step,line in enumerate(handle,1):
            row=json.loads(line);chosen=row['source_ids'];checks=row['teacher_cache_checks']
            require(row['step']==step and len(chosen)==len(checks)==12,'Training journal step or batch size differs')
            require(row['unique_sources']==step*12 and [c['source_id'] for c in checks]==chosen,'Training/cache source accounting differs')
            require(all(c['allclose_original_tolerance'] is True and type(c['samples']) is int and c['samples']>0 for c in checks),
                    'Teacher cache validation failed')
            for key in ('total','waveform','mel','feature','step_seconds'):finite(row[key])
            require(row['step_seconds']>=0,'Negative training duration')
            sources.extend(chosen);samples+=sum(c['samples'] for c in checks)
            nonexact+=sum(c['bitwise_equal'] is not True for c in checks);rows.append(row)
    require(len(rows)==expected and len(sources)==len(set(sources))==expected*12,'Incomplete or repeated training exposure')
    require(sources==identity['source_ids'] and digest(sources)==identity['source_ids_sha256'],'Journal differs from the sealed ordered source ledger')
    require(identity['source_interval']==[0,expected*12] and samples==done['scored_samples'],'Source interval or audio exposure differs')
    require(rows[-1]['training_update_seconds']==done['training_update_seconds'],'Cumulative training timing differs')
    return rows,sources,{'scored_samples':samples,'scored_audio_hours':samples/48000/3600,
                        'teacher_cache_comparisons':len(sources),'teacher_cache_nonexact':nonexact,'teacher_cache_all_pass':True}


def quality(report):
    require(report['aggregate']['sources']==96 and len(report['rows'])==96
        and len({r['source_id'] for r in report['rows']})==96,'Development panel differs from96 unique sources')
    require(set(report['quiet_regions']['regions'])==set(REGIONS),'Expected all seven quiet regions')
    _validate_regions([report])
    regions={}
    for name in REGIONS:
        row=report['quiet_regions']['regions'][name]
        regions[name]={key:finite(row[key],nullable=True) for key in REGION_METRICS}
        regions[name]['failure_categories']={key:finite(row['failure_categories'][key]) for key in CATEGORIES}
    by_source=report['overview_window_metrics']['by_source']
    require(set(by_source)=={r['source_id'] for r in report['rows']},'Active energy source support changed')
    require(all(finite(r[k])>=0 for r in by_source.values() for k in ('active_teacher_energy','active_student_energy')),
            'Negative active waveform energy')
    teacher=sum(r['active_teacher_energy'] for r in by_source.values())
    student=sum(r['active_student_energy'] for r in by_source.values())
    require(teacher>=0 and student>=0,'Negative active waveform energy')
    return {'aggregate':{key:finite(report['aggregate'][key],nullable=True) for key in METRICS},
            'regions':regions,'active_rms_ratio':math.sqrt(student/teacher) if teacher else None}


def paired(now,reference):
    require([r['source_id'] for r in now['rows']]==[r['source_id'] for r in reference['rows']], 'Matched development source ordering changed')
    _validate_regions([reference,now])
    require(now['aggregate']['samples']==reference['aggregate']['samples']
        and now['aggregate']['nonquiet_samples']==reference['aggregate']['nonquiet_samples'],'Matched scored support changed')
    counts={}
    for key,higher in (('mae',False),('mel',False),('group_mse',False),('cosine',True)):
        pairs=[]
        for a,b in zip(now['rows'],reference['rows']):
            require(a['samples']==b['samples'],'A matched source scored span changed')
            x,y=finite(a[key],nullable=True),finite(b[key],nullable=True)
            if x is not None and y is not None:pairs.append((x,y))
        counts[key]={'compared':len(pairs),'improved':sum(x>y if higher else x<y for x,y in pairs),
                     'worsened':sum(x<y if higher else x>y for x,y in pairs),'equal':sum(x==y for x,y in pairs)}
    return counts


def validate_quiet_capture(folder,step,report):
    with gzip.open(folder/f'quiet-windows-step{step}.jsonl.gz','rt') as handle:windows=[json.loads(line) for line in handle]
    require(summarize_regions(windows)==report['quiet_regions'],'Captured quiet rows differ from the saved seven-region summary')
    require(len(windows)==report['aggregate']['quiet_windows']
        and sum(not w['passed'] for w in windows)==report['aggregate']['quiet_failed_windows']
        and all(w['passed']==(w['failure_category']=='passed') for w in windows),'Quiet failure counts changed')
    require(sum(w['valid_samples'] for w in windows)==sum(r['quiet_samples'] for r in report['rows']),'Quiet scored support changed')


def q_totals(rows,done):
    projected=zero=backtracking=0;fractions={str(f):0 for f in FRACTIONS};aux=ordinary=0.
    for row in rows:
        p,z,f=(finite(row[k]) for k in ('q_projected','q_zero_displacement','q_accepted_fraction'))
        require(p in (0,1) and z in (0,1) and f in FRACTIONS and (f!=0 or z==1),'Invalid quiet intervention accounting')
        require(row['q_caps_passed']==1,'A current-batch nonlinear cap failed')
        a,o=finite(row['q_auxiliary_seconds']),finite(row['q_ordinary_update_seconds'])
        require(a>=0 and o>=0,'Invalid quiet timing')
        projected+=int(p);zero+=int(z);backtracking+=int(f<1);fractions[str(float(f))]+=1;aux+=a;ordinary+=o
    expected={'projected_updates':projected,'zero_displacements':zero,'backtracking_updates':backtracking,
              'accepted_nonzero_displacements':len(rows)-zero,'constraint_auxiliary_seconds':aux,'ordinary_update_seconds':ordinary}
    require(all(done.get(k)==v for k,v in expected.items()),'Quiet intervention totals disagree with actual update records')
    return {**expected,'accepted_fraction_counts':fractions}


def audit_run(folder,*,expected_version,endpoint,hash_cache):
    identity,done=read(folder/'launch.json'),read(folder/'completed.json')
    reviews=list(REVIEWS)+( [2500] if endpoint==2500 else [] )
    require(identity['version']==done['version']==expected_version and done['status']=='awaiting_review'
        and done.get('failure') is None and done['step']==endpoint and done['source_count']==endpoint*12
        and done['frozen_state_preserved'] is True and done['original_files_preserved'] is True
        and done['review_steps_completed']==reviews and identity['review_steps']==reviews
        and identity['target_updates']==endpoint,'Trial is not a completed, preserved fixed-budget run')
    for key in TIMES:require(finite(done[key])>=0,'Negative completion duration')
    def file_sha(path):
        path=str(Path(path).resolve())
        if path not in hash_cache:hash_cache[path]=sha(path)
        return hash_cache[path]
    require(identity['protected'] and all(file_sha(path)==value for path,value in identity['protected'].items()),'A protected input changed')
    rows,sources,exposure=audit_journal(folder,identity,done)
    parity=read(folder/'initial-parity.json')
    require(parity['quality']['passed'] is True and parity['rng']['equal'] is True
        and parity['optimizer']['equal'] is True and parity['candidate_state_exact'] is True,'Initial quality/state/RNG/Adam parity failed')
    endpoint_path=folder/f'checkpoint-step{endpoint}.pt';endpoint_receipt=read(endpoint_path.with_suffix('.json'))
    require(file_sha(endpoint_path)==endpoint_receipt['checkpoint_sha256']==done['last_checkpoint_sha256'],'Completed checkpoint hash differs')
    require(endpoint_receipt['step']==endpoint and endpoint_receipt['sources_seen']==endpoint*12
        and endpoint_receipt['frozen_state_preserved'] is True,'Endpoint receipt counters or frozen-state flag differ')
    saved_path=folder/'checkpoint-step2000.pt';receipt=read(saved_path.with_suffix('.json'))
    initial_path=folder/'checkpoint-step0.pt';initial_receipt=read(initial_path.with_suffix('.json'))
    require(file_sha(saved_path)==receipt['checkpoint_sha256'] and file_sha(initial_path)==initial_receipt['checkpoint_sha256'],
            'Initial or matched2000 checkpoint hash differs')
    require(receipt['step']==2000 and receipt['sources_seen']==24000 and receipt['frozen_state_preserved'] is True
        and initial_receipt['step']==0 and initial_receipt['sources_seen']==0 and initial_receipt['frozen_state_preserved'] is True,
        'Checkpoint receipt steps or source exposure differ')
    saved=torch.load(saved_path,map_location='cpu',weights_only=True,mmap=True)
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    require(saved['format']==initial['format']==expected_version and saved['step']==2000
        and saved['sources_seen']==sources[:24000] and saved['scored_samples']==sum(c['samples'] for r in rows[:2000] for c in r['teacher_cache_checks']),
        'Saved2000 checkpoint exposure differs')
    integrity=audit_optimizer(saved,initial,identity)
    teacher_identity=(saved['teacher_source_sha256'],saved['teacher_checkpoint_sha256'])
    reports={step:read(folder/f'development-step{step}.json') for step in REVIEWS}
    require(reports[2000]['aggregate']==receipt['quality'] and reports[0]['aggregate']==initial_receipt['quality'],
            'Saved development metrics differ from checkpoint receipts')
    final=reports[2000] if endpoint==2000 else read(folder/'development-step2500.json')
    require(final['aggregate']==endpoint_receipt['quality']==done['final'],'Final quality differs from completion or endpoint receipt')
    _validate_regions(list(reports.values()))
    clean={}
    for step,report in reports.items():
        clean[str(step)]=quality(report);validate_quiet_capture(folder,step,report)
    del saved,initial
    return {'identity':identity,'done':done,'rows':rows,'sources':sources,'reports':reports,'quality':clean,
            'teacher_identity':teacher_identity,
            'integrity':{**integrity,'checkpoint_step2000_sha256':file_sha(saved_path),'checkpoint_step0_sha256':file_sha(initial_path),
                         'initial_parity_passed':True,'checkpoint_hashes_exact':True,'protected_files_unchanged':True,
                         'quiet_capture_summaries_verified':True},'exposure':exposure}


def collect(run,reference):
    run,reference=Path(run).resolve(),Path(reference).resolve()
    require(run!=reference,'Run and comparator must be distinct')
    version=read(run/'launch.json')['version'];require(version in RUNS,'Unsupported independent trial version')
    arm,reference_version=RUNS[version];hash_cache={}
    current=audit_run(run,expected_version=version,endpoint=2000,hash_cache=hash_cache)
    previous=audit_run(reference,expected_version=reference_version,endpoint=2500 if arm=='Q' else 2000,hash_cache=hash_cache)
    require(current['sources']==previous['sources'][:24000],'Trial source order differs from matched reference prefix')
    require(current['teacher_identity']==previous['teacher_identity'],'Trial and comparator teacher identities differ')
    protected={str(Path(path).resolve()):value for path,value in current['identity']['protected'].items()}
    for name in ('launch.json','completed.json','train.jsonl','checkpoint-step2000.pt','checkpoint-step2000.json'):
        path=str((reference/name).resolve())
        require(path in protected and hash_cache[path]==protected[path],'Comparator evidence is not bound to trial launch')
    for step in REVIEWS:
        path=str((reference/f'development-step{step}.json').resolve())
        require(path in protected and sha(path)==protected[path],'Matched report is not bound to trial launch')
    if arm=='Q':
        require(current['identity']['initial_decoder_state_sha256']==previous['identity']['initial_decoder_state_sha256'],
                'Quiet arm did not start at the same combined initializer')
    intervention=q_totals(current['rows'],current['done']) if arm=='Q' else None
    prefix_row=previous['rows'][1999]
    prefix_samples=sum(c['samples'] for r in previous['rows'][:2000] for c in r['teacher_cache_checks'])
    prefix_times={k:finite(prefix_row[k]) for k in TIMES}
    require(all(v>=0 for v in prefix_times.values()),'Negative matched reference prefix duration')
    return {'version':VERSION,'complete':True,'arm':arm,'reference_arm':'C' if arm=='Q' else 'B',
        'optimizer_updates':2000,'unique_training_sources':24000,'development_sources':96,
        'integrity':{**current['integrity'],'source_order_matches_reference_prefix':True,'reference_integrity_passed':True},
        'training':{**current['exposure'],**{k:current['done'][k] for k in TIMES}},
        'reference_completed_run_updates':previous['done']['step'],
        'reference_matched2000_training':{'optimizer_updates':2000,'unique_training_sources':24000,
            'scored_samples':prefix_samples,'scored_audio_hours':prefix_samples/48000/3600,**prefix_times,
            'timing_scope':'Original update2000 journal, before its final validation; core training time matches exposure'},
        'quality':current['quality'],'reference_quality':previous['quality'],
        'paired_improvement_counts':{str(step):paired(current['reports'][step],previous['reports'][step]) for step in REVIEWS},
        'quiet_interventions':intervention,'automatic_promotion':False,
        'scope':'Saved reports and CPU checkpoints only; no inference or optimization; aggregate-only output'}


def save_report(report,path):
    path=Path(path);text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if path.exists():
        require(path.read_text()==text,'Output already exists with different contents')
        return
    require(path.parent.is_dir(),'Output parent must already exist')
    with path.open('x') as handle:handle.write(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    try:
        report=collect(args.run,args.reference)
        save_report(report,args.out or args.run/'independent-trial-aggregate.json')
    except Exception as exc:
        print(json.dumps({'complete':False,'failure':str(exc) if isinstance(exc,AuditError) else 'Audit could not complete: '+type(exc).__name__}))
        raise SystemExit(1)
    print(json.dumps(report,allow_nan=False))


if __name__=='__main__':main()
