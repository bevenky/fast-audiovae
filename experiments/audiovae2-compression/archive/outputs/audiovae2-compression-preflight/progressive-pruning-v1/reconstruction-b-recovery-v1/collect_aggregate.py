"""Read-only completed-run audit; emit aggregates, never source IDs or tensors."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/tmp/fast-audiovae-progressive-pruning-v1')
RUN = ROOT/'reconstruction-b-recovery-v1'
OUT = RUN/'results'
KEYS = ('mae','mel','group_mse','nonquiet_cosine_mean','quiet_residual_rms_mean',
        'quiet_windows','quiet_failed_windows','peak_abs_max','overshoot_samples')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def nums(d):
    return {k:v for k,v in d.items() if v is None or isinstance(v,(int,float,bool))}

def quality(report):
    return {'aggregate':{k:report['aggregate'][k] for k in KEYS},
            'regions':{k:{**nums(v),'failure_categories':v['failure_categories']}
                       for k,v in report['quiet_regions']['regions'].items()}}

def paired(now,ref):
    assert len(now['rows'])==len(ref['rows'])==96
    nr={r['source_id']:r for r in now['rows']}
    rr={r['source_id']:r for r in ref['rows']}
    assert nr.keys()==rr.keys() and len(nr)==96
    assert now['quiet_regions']['window_identity_sha256']==ref['quiet_regions']['window_identity_sha256']
    assert now['aggregate']['samples']==ref['aggregate']['samples']
    result={}
    for key,higher in (('mae',False),('mel',False),('group_mse',False),('cosine',True)):
        pairs=[(nr[k][key],rr[k][key]) for k in nr if nr[k][key] is not None and rr[k][key] is not None]
        result[key]={'sources':len(pairs),
            'improved':sum((a>b if higher else a<b) for a,b in pairs),
            'worsened':sum((a<b if higher else a>b) for a,b in pairs),
            'equal':sum(a==b for a,b in pairs)}
    return result

def fixed_quiet_identity(now, reference):
    assert now['window_identity_sha256']==reference['window_identity_sha256']
    assert now['version']==reference['version'] and now['regions'].keys()==reference['regions'].keys()
    for name, region in now['regions'].items():
        for key in ('windows','samples','window_ids_sha256','teacher_rms'):
            assert region[key]==reference['regions'][name][key], 'Fixed quiet-region support changed'

def audit_optimizer(saved, initial, identity, torch):
    assert saved['identity']==identity==initial['identity'], 'Saved identity differs from launch'
    assert initial['step']==0 and initial['sources_seen']==[] and initial['optimizer']['state']=={}
    assert saved['optimizer']['param_groups']==initial['optimizer']['param_groups'], 'Adam options or parameter order changed'
    groups=saved['optimizer']['param_groups']
    assert len(groups)==1
    expected={'lr':3e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0,
              'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False}
    assert all(groups[0][key]==value for key,value in expected.items()), 'Adam recipe changed'
    assert identity['learning_rate']==3e-5 and identity['optimizer_betas']==[.9,.99]
    assert identity['optimizer_eps']==1e-8 and identity['weight_decay']==0
    assert saved['accumulation']==identity['gradient_accumulation']==12
    assert identity['execution_batch_size']==1 and saved['coefficients']==identity['coefficients']
    states=saved['optimizer']['state']; parameters=groups[0]['params']
    assert len(parameters)==len(set(parameters))==len(states)==90 and set(parameters)==set(states)
    for state in states.values():
        assert set(state)=={'step','exp_avg','exp_avg_sq'}, 'Unexpected Adam moment state'
        assert all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in state.values()), 'Nonfinite Adam state'
        assert state['step'].numel()==1 and float(state['step'])==2000
        assert state['exp_avg'].shape==state['exp_avg_sq'].shape
        assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
        assert bool((state['exp_avg_sq']>=0).all()), 'Negative Adam second moment'

def main():
    from joint_recovery_gates_v2 import summarize_regions
    done=read(OUT/'completed.json')
    assert done['status']=='awaiting_review' and done['step']==2000 and done['source_count']==24000
    assert done['frozen_state_preserved'] and done['original_files_preserved']
    assert done['review_steps_completed']==[0,250,500,1000,1500,2000]
    identity=read(OUT/'launch.json')
    journal=[json.loads(l) for l in (OUT/'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in journal]==list(range(1,2001))
    ids=[s for r in journal for s in r['source_ids']]
    assert ids==identity['source_ids'] and len(ids)==len(set(ids))==24000
    cache=[x for r in journal for x in r['teacher_cache_checks']]
    assert len(cache)==24000 and all(x['allclose_original_tolerance'] for x in cache)
    assert all(sha(p)==h for p,h in identity['protected'].items())
    ckpt=OUT/'checkpoint-step2000.pt'
    checkpoint_receipt=read(ckpt.with_suffix('.json'))
    assert sha(ckpt)==done['last_checkpoint_sha256']==checkpoint_receipt['checkpoint_sha256']
    import torch
    saved=torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True)
    assert saved['step']==2000 and saved['sources_seen']==ids
    initial_path=OUT/'checkpoint-step0.pt'
    assert sha(initial_path)==read(initial_path.with_suffix('.json'))['checkpoint_sha256']
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    audit_optimizer(saved,initial,identity,torch)
    assert all(bool(torch.isfinite(v).all()) for v in saved['group'].values())
    assert saved['scored_samples']==done['scored_samples']
    report={'complete':True,'step':2000,'unique_training_sources':24000,'development_sources':96,
            'optimizer_tensors_with_counter2000':90,'teacher_cache_comparisons':len(cache),
            'adam_moments_finite':True,'adam_settings_and_parameter_order_unchanged':True,
            'checkpoint_identities_match_launch':True,
            'teacher_cache_all_pass':True,'source_order_matches_original':True,
            'protected_files_unchanged':True,'trained_group_finite':True,
            'checkpoint_step2000_sha256':done['last_checkpoint_sha256'],
            'training_update_seconds':done['training_update_seconds'],
            'validation_seconds':done['validation_seconds'],'waiting_seconds':done['waiting_seconds'],
            'elapsed_seconds':done['elapsed_seconds'],'scored_audio_hours':done['scored_samples']/48000/3600,
            'quality':{},'paired_against_same_step':{},'paired_against_old5000':{},
            'quiet_disjoint':{},'original_quality':{}}
    old={}
    for step in (0,500,1000,1500,2000,5000):
        folder=ROOT/'cut1-384-256' if step<=1000 else ROOT/'recovery-1000-2000/segment-1000-2000' if step<=2000 else ROOT/'recovery-2000-5000/segment-2000-5000'
        old[step]=read(folder/f'development-step{step}.json')
        report['original_quality'][step]=quality(old[step])
    old_folder=ROOT/'recovery-2000-5000/segment-2000-5000'
    old_checkpoint=old_folder/'checkpoint-step5000.pt'
    old_receipt_path=old_checkpoint.with_suffix('.json')
    protected={str(Path(p).resolve()):h for p,h in identity['protected'].items()}
    assert protected[str(old_checkpoint.resolve())]==identity['preserved5000_sha256']
    assert sha(old_receipt_path)==protected[str(old_receipt_path.resolve())]
    old_receipt=read(old_receipt_path)
    assert old_receipt['checkpoint_sha256']==identity['preserved5000_sha256']
    old_report_bound='quality' in old_receipt
    if old_report_bound:
        assert old_receipt['quality']==old[5000]['aggregate'], 'Old5000 report differs from its protected checkpoint receipt'
    report['old5000_aggregate_bound_to_protected_checkpoint_receipt']=old_report_bound
    for previous in old.values(): fixed_quiet_identity(previous['quiet_regions'],old[0]['quiet_regions'])
    verified_steps=[]
    for step in (0,250,500,1000,1500,2000):
        new=read(OUT/f'development-step{step}.json')
        fixed_quiet_identity(new['quiet_regions'],old[0]['quiet_regions'])
        assert [r['source_id'] for r in new['rows']]==[r['source_id'] for r in old[0]['rows']]
        assert new['aggregate']['sources']==96 and new['aggregate']['samples']==old[0]['aggregate']['samples']
        assert new['aggregate']['nonquiet_samples']==old[0]['aggregate']['nonquiet_samples']
        if step==2000: assert checkpoint_receipt['quality']==new['aggregate']
        report['quality'][step]=quality(new)
        if step in old: report['paired_against_same_step'][step]=paired(new,old[step])
        if step in (1000,2000):report['paired_against_old5000'][step]=paired(new,old[5000])
        with gzip.open(OUT/f'quiet-windows-step{step}.jsonl.gz','rt') as f:
            windows=[json.loads(l) for l in f]
        recomputed=summarize_regions(windows)
        assert recomputed==new['quiet_regions'], 'Captured quiet windows do not reproduce the saved complete summary'
        fixed_quiet_identity(recomputed,old[0]['quiet_regions'])
        assert len(windows)==new['aggregate']['quiet_windows']
        assert sum(not w['passed'] for w in windows)==new['aggregate']['quiet_failed_windows']
        assert all(w['passed']==(w['failure_category']=='passed') for w in windows)
        assert sum(w['valid_samples'] for w in windows)==sum(r['quiet_samples'] for r in new['rows'])
        verified_steps.append(step)
        parts={}
        for label,select in (
            ('startup',lambda w:w['teacher_rms']<=1e-5 and w['source_start_sample']<960),
            ('other_near',lambda w:w['teacher_rms']<=1e-5 and w['source_start_sample']>=960),
            ('remaining_quiet',lambda w:w['teacher_rms']>1e-5)):
            rows=[w for w in windows if select(w)];n=sum(w['valid_samples'] for w in rows)
            parts[label]={'windows':len(rows),'passed':sum(w['passed'] for w in rows),
                'failed':sum(not w['passed'] for w in rows),
                'residual_rms':math.sqrt(sum(w['residual_square_sum'] for w in rows)/n),
                'centered_residual_rms':math.sqrt(sum(w['centered_residual_square_sum'] for w in rows)/n),
                'output_rms':math.sqrt(sum(w['student_rms']**2*w['valid_samples'] for w in rows)/n),
                'teacher_rms':math.sqrt(sum(w['teacher_rms']**2*w['valid_samples'] for w in rows)/n),
                'output_limit_excess_rms':math.sqrt(sum(max(w['student_rms']-w['output_rms_limit'],0.)**2*w['valid_samples'] for w in rows)/n),
                'failure_categories':{c:sum(w['failure_category']==c for w in rows)
                    for c in ('passed','residual_only','amplitude_only','both')}}
        assert sum(v['windows'] for v in parts.values())==2544
        assert sum(v['failed'] for v in parts.values())==new['aggregate']['quiet_failed_windows']
        report['quiet_disjoint'][step]=parts
    report['quiet_capture_summary_verified_steps']=verified_steps
    report['all_quiet_window_and_region_identities_unchanged']=True
    report['scope']='Existing reports and CPU checkpoint only; no new inference or optimization; aggregate-only output'
    (RUN/'aggregate-completed-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))

if __name__=='__main__':
    main()
