"""Saved-artifact audit tests; synthetic CPU tensors only, no neural execution."""
from copy import deepcopy
from pathlib import Path
import gzip
import json
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import independent_trial_aggregate as audit


def write(path,value):path.write_text(json.dumps(value,allow_nan=True))


def quality(scale):
    windows=[];rows=[];energy={}
    for i in range(96):
        sid=f'PRIVATE_SOURCE_{i}'
        start,level,zero=((0,9e-6,True),(960,5e-4,True),(1920,9e-6,True),(38400,9e-6,False),(48960,4e-4,False))[i%5]
        windows.append({'window_id':f'PRIVATE_WINDOW_{i}','source_id':sid,'source_start_sample':start,
            'source_stop_sample':start+960,'valid_samples':960,'is_quiet':True,'teacher_rms':level,
            'student_rms':level,'residual_limit':max(.02**.5*level,1e-5),'output_rms_limit':max(10**.05*level,1e-5),
            'source_reference_exact_zero':zero,'failure_category':'passed','passed':True,
            'residual_square_sum':960e-12,'centered_residual_square_sum':480e-12})
        rows.append({'source_id':sid,'samples':1920,'quiet_samples':960,'mae':.02*scale,'mel':.5*scale,
                     'group_mse':.1*scale,'cosine':None if i==0 else 1-.1*scale,'private_detail':'DO_NOT_EXPORT'})
        energy[sid]={'active_teacher_energy':2.,'active_student_energy':2.*scale**2}
    aggregate={'sources':96,'samples':96*1920,'nonquiet_samples':96*960,'mae':.02*scale,'mse':.001*scale,
        'waveform_nrmse':.1*scale,'mel':.5*scale,'group_mse':.1*scale,'group_nrmse':.1*scale,
        'nonquiet_cosine_mean':1-.1*scale,'quiet_residual_rms_mean':1e-6,'quiet_windows':96,
        'quiet_failed_windows':0,'peak_abs_max':.8,'overshoot_samples':0,'private_number':987654321}
    return {'aggregate':aggregate,'quiet_regions':audit.summarize_regions(windows),'rows':rows,
            'overview_window_metrics':{'by_source':energy}},windows


def optimizer_pair(identity):
    params=[torch.nn.Parameter(torch.zeros(2+i%3)) for i in range(90)]
    opt=torch.optim.AdamW(params,lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    group={f'parameter_{i}':p.detach().clone() for i,p in enumerate(params)}
    common={'identity':identity,'format':identity['version'],'group':group,'selection':identity['selection'],
        'accumulation':12,'coefficients':audit.COEFFICIENTS,'optimizer_reset_after_start':False,
        'teacher_source_sha256':'teacher-source','teacher_checkpoint_sha256':'teacher-weights'}
    initial={**deepcopy(common),'step':0,'cut_updates':0,'global_updates':0,'sources_seen':[],
             'scored_samples':0,'optimizer':deepcopy(opt.state_dict()),'rng':{'torch':torch.get_rng_state()}}
    for p in params:opt.state[p]={'step':torch.tensor(2000.),'exp_avg':torch.full_like(p,.1),'exp_avg_sq':torch.full_like(p,.2)}
    saved={**deepcopy(common),'step':2000,'cut_updates':2000,'global_updates':2000,
        'sources_seen':identity['source_ids'][:24000],'scored_samples':24000*1920,
        'optimizer':deepcopy(opt.state_dict()),'rng':{'torch':torch.get_rng_state()}}
    return initial,saved


def make_run(folder,version,endpoint,protected,scale=1.,quiet=False):
    folder.mkdir();ids=[f'PRIVATE_TRAIN_{i}' for i in range(endpoint*12)]
    reviews=list(audit.REVIEWS)+([2500] if endpoint==2500 else [])
    identity={'version':version,'selection':{'stage2_indices':list(range(384)),'stage3_indices':list(range(256))},
        'source_ids':ids,'source_ids_sha256':audit.digest(ids),'source_interval':[0,endpoint*12],
        'protected':protected,'target_updates':endpoint,'review_steps':reviews,
        'learning_rate':3e-5,'optimizer_betas':[.9,.99],'optimizer_eps':1e-8,'weight_decay':0.,
        'execution_batch_size':1,'gradient_accumulation':12,'trainable_stages':[2,3,4],
        'coefficients':audit.COEFFICIENTS,'initial_decoder_state_sha256':'same-initial-state'}
    write(folder/'launch.json',identity)
    reports={}
    for step in reviews:
        report,windows=quality(scale);reports[step]=report
        write(folder/f'development-step{step}.json',report)
        with gzip.open(folder/f'quiet-windows-step{step}.jsonl.gz','wt') as handle:
            handle.write(''.join(json.dumps(w)+'\n' for w in windows))
    initial,saved=optimizer_pair(identity)
    for step,payload in ((0,initial),(2000,saved)):
        path=folder/f'checkpoint-step{step}.pt';torch.save(payload,path)
        write(path.with_suffix('.json'),{'checkpoint_sha256':audit.sha(path),'step':step,'sources_seen':step*12,
            'frozen_state_preserved':True,'quality':reports[step]['aggregate']})
    if endpoint==2500:
        path=folder/'checkpoint-step2500.pt';torch.save({'preserved':'only its pinned receipt needed for comparison'},path)
        write(path.with_suffix('.json'),{'checkpoint_sha256':audit.sha(path),'step':2500,'sources_seen':30000,
            'frozen_state_preserved':True,'quality':reports[2500]['aggregate']})
    parity={'quality':{'passed':True},'rng':{'equal':True},'optimizer':{'equal':True},'candidate_state_exact':True}
    write(folder/'initial-parity.json',parity)
    records=[]
    for step in range(1,endpoint+1):
        chosen=ids[(step-1)*12:step*12]
        row={'step':step,'source_ids':chosen,'unique_sources':step*12,
            'teacher_cache_checks':[{'source_id':sid,'samples':1920,'allclose_original_tolerance':True,'bitwise_equal':True} for sid in chosen],
            'total':.1,'waveform':.2,'mel':.3,'feature':.4,'step_seconds':.25,
            'training_update_seconds':step*.25,'validation_seconds':4.,'waiting_seconds':1.,'elapsed_seconds':step*.25+5}
        if quiet:row.update(q_projected=int(step%2==0),q_zero_displacement=int(step%3==0),
            q_accepted_fraction=0. if step%3==0 else 1.,q_caps_passed=1,q_auxiliary_seconds=.125,q_ordinary_update_seconds=.0625)
        records.append(row)
    (folder/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    done={'version':version,'status':'awaiting_review','failure':None,'step':endpoint,'source_count':endpoint*12,
        'frozen_state_preserved':True,'original_files_preserved':True,'review_steps_completed':reviews,
        'last_checkpoint_sha256':audit.sha(folder/f'checkpoint-step{endpoint}.pt'),'scored_samples':endpoint*12*1920,
        'training_update_seconds':endpoint*.25,'validation_seconds':5.,'waiting_seconds':1.,'elapsed_seconds':endpoint*.25+6,
        'final':reports[endpoint]['aggregate']}
    if quiet:done.update(projected_updates=endpoint//2,zero_displacements=endpoint//3,
        accepted_nonzero_displacements=endpoint-endpoint//3,backtracking_updates=endpoint//3,
        constraint_auxiliary_seconds=endpoint*.125,ordinary_update_seconds=endpoint*.0625)
    write(folder/'completed.json',done)
    return identity,initial,saved


def fixture(tmp_path,arm='D'):
    protected=tmp_path/'sealed.txt';protected.write_text('immutable source fixture')
    reference=tmp_path/'reference';version=next(k for k,(letter,_) in audit.RUNS.items() if letter==arm)
    ref_version=audit.C_VERSION if arm=='Q' else audit.B_VERSION
    make_run(reference,ref_version,2500 if arm=='Q' else 2000,{str(protected):audit.sha(protected)})
    pinned={str(path):audit.sha(path) for path in reference.iterdir() if path.is_file()}
    run=tmp_path/'run';make_run(run,version,2000,pinned,scale=.8,quiet=arm=='Q')
    return run,reference


@pytest.mark.parametrize('arm',['D','G','Q'])
def test_full_saved_artifact_audit_all_arms_and_no_private_output(tmp_path,arm):
    run,reference=fixture(tmp_path,arm);report=audit.collect(run,reference)
    assert report['complete'] and report['arm']==arm and report['optimizer_updates']==2000
    assert report['unique_training_sources']==24000 and report['integrity']['adam_parameter_states']==90
    assert report['reference_completed_run_updates']==(2500 if arm=='Q' else 2000)
    assert report['reference_matched2000_training']['scored_samples']==24000*1920
    assert report['reference_matched2000_training']['training_update_seconds']==500.
    assert report['paired_improvement_counts']['2000']['mae']=={'compared':96,'improved':96,'worsened':0,'equal':0}
    assert report['paired_improvement_counts']['2000']['cosine']['compared']==95
    assert set(report['quality']['0']['regions'])==set(audit.REGIONS)
    text=json.dumps(report)
    for forbidden in ('PRIVATE_SOURCE','PRIVATE_WINDOW','PRIVATE_TRAIN','DO_NOT_EXPORT','987654321','by_source','window_ids',str(tmp_path)):
        assert forbidden not in text
    if arm=='Q':assert report['quiet_interventions']['zero_displacements']==666
    else:assert report['quiet_interventions'] is None
    output=run/'aggregate.json';audit.save_report(report,output);stamp=output.stat().st_mtime_ns
    audit.save_report(report,output);assert output.stat().st_mtime_ns==stamp
    with pytest.raises(audit.AuditError):audit.save_report({**report,'complete':False},output)


@pytest.mark.parametrize('damage',['incomplete','source_order','cache','parity','checkpoint','quiet_capture','matched_report'])
def test_full_audit_rejects_corrupt_or_incomplete_evidence(tmp_path,damage):
    run,reference=fixture(tmp_path)
    if damage=='incomplete':
        value=audit.read(run/'completed.json');value['step']=1999;write(run/'completed.json',value)
    elif damage in ('source_order','cache'):
        path=run/'train.jsonl';records=[json.loads(line) for line in path.read_text().splitlines()]
        if damage=='source_order':records[0]['source_ids'].reverse()
        else:records[0]['teacher_cache_checks'][0]['allclose_original_tolerance']=False
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    elif damage=='parity':
        value=audit.read(run/'initial-parity.json');value['candidate_state_exact']=False;write(run/'initial-parity.json',value)
    elif damage=='checkpoint':
        with (run/'checkpoint-step2000.pt').open('ab') as handle:handle.write(b'corruption')
    elif damage=='quiet_capture':
        path=run/'quiet-windows-step500.jsonl.gz'
        with gzip.open(path,'rt') as handle:windows=[json.loads(line) for line in handle]
        windows[0]['residual_square_sum']*=2
        with gzip.open(path,'wt') as handle:handle.write(''.join(json.dumps(w)+'\n' for w in windows))
    else:
        value=audit.read(reference/'development-step2000.json');value['rows'][0]['mae']=9
        write(reference/'development-step2000.json',value)
    with pytest.raises((audit.AuditError,ValueError)):audit.collect(run,reference)
    assert not (run/'independent-trial-aggregate.json').exists()


@pytest.mark.parametrize('damage',['counter','nan_moment','wrong_shape','negative_second','missing_parameter','recipe','order'])
def test_optimizer_audit_checks_every_state_against_parameter_shape(damage):
    identity={'version':'fixture','selection':{},'source_ids':['unused']*24000,'learning_rate':3e-5,
        'optimizer_betas':[.9,.99],'optimizer_eps':1e-8,'weight_decay':0.,'execution_batch_size':1,
        'gradient_accumulation':12,'trainable_stages':[2,3,4],'coefficients':audit.COEFFICIENTS}
    initial,saved=optimizer_pair(identity);states=saved['optimizer']['state'];last=states[89]
    if damage=='counter':last['step'].fill_(1999)
    elif damage=='nan_moment':last['exp_avg'][0]=float('nan')
    elif damage=='wrong_shape':last['exp_avg']=last['exp_avg'][:-1]
    elif damage=='negative_second':last['exp_avg_sq'][0]=-1
    elif damage=='missing_parameter':states.pop(89)
    elif damage=='recipe':saved['optimizer']['param_groups'][0]['lr']=1e-3
    else:saved['optimizer']['param_groups'][0]['params'].reverse()
    with pytest.raises(audit.AuditError):audit.audit_optimizer(saved,initial,identity)


def test_paired_counts_require_exact_fixed_source_and_window_support():
    now,_=quality(.9);reference,_=quality(1.)
    now['rows'][0]['source_id']='different'
    with pytest.raises(audit.AuditError):audit.paired(now,reference)
    now,_=quality(.9);now['quiet_regions']['window_identity_sha256']='changed'
    with pytest.raises(ValueError):audit.paired(now,reference)


def test_q_aggregate_rejects_fabricated_rejection_totals(tmp_path):
    run,reference=fixture(tmp_path,'Q');value=audit.read(run/'completed.json')
    value['zero_displacements']+=1;write(run/'completed.json',value)
    with pytest.raises(audit.AuditError,match='intervention totals'):audit.collect(run,reference)
