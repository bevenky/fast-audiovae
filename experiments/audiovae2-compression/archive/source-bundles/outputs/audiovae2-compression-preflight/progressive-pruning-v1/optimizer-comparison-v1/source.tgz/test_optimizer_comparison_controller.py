"""Pure receipts/selection/process parsing tests. No subprocess or model runs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parent))
import optimizer_comparison_controller as run


def put(path,value):
    path.write_text(json.dumps(value,allow_nan=False));return path


def candidates():
    return [{'name':f'{m}-{i}','method':m,'optimizer_kwargs':{'matrix_lr':rate}}
            for m in run.METHODS for i,rate in enumerate(run.GRIDS[m])]


@pytest.fixture
def plan(tmp_path):
    config=put(tmp_path/'config.json',{'fresh':True})
    reference=put(tmp_path/'reference.json',{'complete':True})
    code=tmp_path/'code';code.mkdir()
    names=('optimizer_comparison_runner.py','recovery_optimizers.py','startup_mixed_optimizer_update.py')
    for name in names:(code/name).write_text('# sealed fixture\n')
    (code/'optimizer_comparison_controller.py').write_bytes(Path(run.__file__).read_bytes())
    manifest=put(tmp_path/'manifest.json',{name:run.sha(code/name) for name in (*names,'optimizer_comparison_controller.py')})
    return {'version':run.PLAN_VERSION,'config':str(config),'config_sha256':run.sha(config),
            'parity_reference':str(reference),'parity_reference_sha256':run.sha(reference),
            'source_manifest':str(manifest),'source_manifest_sha256':run.sha(manifest),
            'python':sys.executable,'pythonpath':str(code),'root':str(tmp_path/'outputs'),
            'candidates':candidates(),'group_parameter_bytes':26_500_000}


def test_plan_control_first_and_ordered_pythonpath(plan,tmp_path):
    earlier=tmp_path/'empty';earlier.mkdir();plan['pythonpath']=str(earlier)+':'+plan['pythonpath']
    ordered=run.validate_plan(plan)
    assert ordered[0]['method']=='adamw' and ordered[0]['optimizer_kwargs']['matrix_lr']==3e-5
    assert len(ordered)==12 and len({c['name'] for c in ordered})==12


@pytest.mark.parametrize('damage',['rate','duplicate','adam','traversal','source','config'])
def test_plan_rejects_grid_hash_and_path_drift(plan,damage):
    if damage=='rate':plan['candidates'][0]['optimizer_kwargs']['matrix_lr']=.1
    elif damage=='duplicate':plan['candidates'][0]=deepcopy(plan['candidates'][1])
    elif damage=='adam':plan['candidates'][0]['optimizer_kwargs']['adam_lr']=.01
    elif damage=='traversal':plan['candidates'][0]['name']='../escaped'
    elif damage=='source':(Path(plan['pythonpath'])/'recovery_optimizers.py').write_text('changed')
    else:Path(plan['config']).write_text('changed')
    with pytest.raises((ValueError,run.TechnicalFailure)):run.validate_plan(plan)


@pytest.fixture
def completed(tmp_path,plan):
    out=tmp_path/'trial';out.mkdir();c=candidates()[1];trial=run.trial_spec(plan,c,64)
    sources=[f'source-{i}' for i in range(768)]
    identity={k:run.digest(k) for k in ('initial_state_sha256','initial_rng_sha256','initializer_artifact_sha256',
                                     'source_plan_identity_sha256','training_probe_identity_sha256')}
    optimizer={'method':trial['method'],'adam':{'lr':3e-5,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.},
               'matrix':{'lr':trial['optimizer_kwargs']['matrix_lr']},'tensor_count':90}
    launch={'version':run.RUNNER_VERSION,'trial':trial,'source_ids':sources,'source_ids_sha256':run.digest(sources),
            'protected':{plan['source_manifest']:plan['source_manifest_sha256']},'optimizer':optimizer,**identity}
    done={'version':run.RUNNER_VERSION,'trial_name':trial['name'],'method':trial['method'],'optimizer':optimizer,
        'neural_training_target':64,'complete':True,'updates':64,'all_preservation_checks_passed':True,
        'preserved':{'teacher':True,'files':True,'frozen_outer':True},'ordinary_unique_sources':768,
        'ordinary_source_prefix_sha256':run.digest(sources),'training_probe_before':{'total':1.},
        'training_probe_after':{'total':.8},'parity':{'passed':True,'updates':64},'failure_category':None,
        'optimizer_activity':{'matrix_parameters':9,'matrix_steps':[64]*9},**identity}
    rows=[]
    for step in range(1,65):
        rows.append({'step':step,'unique_sources':step*12,'source_ids':sources[(step-1)*12:step*12],
            'teacher_cache_checks':[{'allclose_original_tolerance':True}]*12,
            'q_caps_passed':1,'q_full_primal_verified':1,'q_kkt_passed':1,
            'startup_anchor_after_passed':6,'startup_anchor_updates':step,
            'total':1.,'waveform':1.,'mel':1.,'feature':1.,
            'q_accepted_displacement_norm':.1,'q_accepted_fraction':1.,'q_zero_displacement':0})
    put(out/'launch.json',launch);put(out/'completed.json',done)
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return out,trial,done,launch,rows


def test_verified_nonzero64_and_training_only_score(completed):
    out,trial,done,_,_=completed
    done['milestone_quality']={'64':{'aggregate':{'arbitrary_dev_score':-99999}}}
    put(out/'completed.json',done)
    result=run.inspect_trial(out,trial,control=True)
    assert result['eligible'] and result['updates']==64 and result['nonzero_updates']==64
    assert result['training_probe_total']==.8 and 'milestone_quality' not in result


@pytest.mark.parametrize('damage',['preservation','identity','source','cache','anchor','counter','activity','parity','recipe'])
def test_completion_and_journal_contract_fail_closed(completed,damage):
    out,trial,done,launch,rows=completed
    if damage=='preservation':done['preserved']['files']=False
    elif damage=='identity':done['initial_rng_sha256']='a'*64
    elif damage=='source':rows[5]['source_ids'][0]='different'
    elif damage=='cache':rows[5]['teacher_cache_checks']=[{'allclose_original_tolerance':False}]*12
    elif damage=='anchor':rows[5]['startup_anchor_after_passed']=5
    elif damage=='counter':done['updates']=63
    elif damage=='activity':done['optimizer_activity']['matrix_steps'][0]=63
    elif damage=='recipe':done['optimizer']['matrix']['lr']=.7
    else:done['parity']['passed']=False
    put(out/'completed.json',done)
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(run.TechnicalFailure):run.inspect_trial(out,trial,control=True)


def test_zero_motion_is_ineligible_not_technical(completed):
    out,trial,_,_,rows=completed
    rows[-1].update(q_accepted_displacement_norm=0.,q_zero_displacement=1,q_accepted_fraction=0.)
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=run.inspect_trial(out,trial,control=True)
    assert not result['eligible'] and result['nonzero_updates']==63 and result['reason']=='zero_parameter_updates'


def test_only_certified_numeric_failure_can_continue(completed):
    out,trial,done,_,_=completed
    done.update(complete=False,parity=None,failure_category='RuntimeError',failure_kind='model_numeric')
    put(out/'completed.json',done)
    result=run.inspect_trial(out,trial,exit_code=1)
    assert not result['eligible'] and result['reason']=='model_numeric_failure'
    done['failure_kind']='technical';put(out/'completed.json',done)
    with pytest.raises(run.TechnicalFailure):run.inspect_trial(out,trial,exit_code=1)


@pytest.mark.parametrize('method,key,valid',[
    ('normuon','nor_neuron_state_shapes',[[384,1]]*3+[[256,1]]*3+[[128,1]]*3),
    ('shampoo','shampoo_last_refresh_steps',[60]*9)])
def test_extra_algorithm_state_must_be_engaged(completed,method,key,valid):
    out,trial,done,launch,_=completed
    trial['method']=done['method']=launch['optimizer']['method']=done['optimizer']['method']=method
    launch['trial']=trial;put(out/'launch.json',launch);put(out/'completed.json',done)
    with pytest.raises(run.TechnicalFailure):run.inspect_trial(out,trial)
    done['optimizer_activity'][key]=valid;put(out/'completed.json',done)
    assert run.inspect_trial(out,trial)['eligible']


def selection_records(plan):
    return [{**c,'eligible':True,'training_probe_total':float(i+1),'identity':{'fixed':'identity'},
             'completed_path':f'/reports/{c["name"]}/completed.json','completed_sha256':'a'*64,
             'parity_passed':c['method']=='adamw' and c['optimizer_kwargs']['matrix_lr']==3e-5}
            for i,c in enumerate(plan['candidates'])]


def test_select_training_minimum_tie_break_and_no_dev(plan):
    records=selection_records(plan)
    for row in records:row['training_probe_total']=1.
    decision=run.select_candidates(plan,records,'b'*64)
    assert decision['development_used_for_selection'] is False
    assert all(decision['selected'][m]['optimizer_kwargs']['matrix_lr']==min(run.GRIDS[m]) for m in run.METHODS)
    records[0]['eligible']=False
    decision=run.select_candidates(plan,records,'b'*64)
    assert decision['selected']['adamw']['optimizer_kwargs']['matrix_lr']==3e-5


@pytest.mark.parametrize('damage',['missing','identity','parity'])
def test_selection_requires_all_candidates_shared_identity_and_control(plan,damage):
    rows=selection_records(plan)
    if damage=='missing':rows.pop()
    elif damage=='identity':rows[-1]['identity']={'different':True}
    else:
        for row in rows:row['parity_passed']=False
    with pytest.raises(run.TechnicalFailure):run.select_candidates(plan,rows,'b'*64)


def test_storage_reserve_and_no_overwrite(plan,tmp_path):
    assert run.required_free_bytes(plan,'adamw',64)==96*run.MIB
    assert run.required_free_bytes(plan,'shampoo',2000)>run.required_free_bytes(plan,'adamw',2000)
    del plan['group_parameter_bytes']
    with pytest.raises(ValueError):run.required_free_bytes(plan,'adamw',2000)
    path=tmp_path/'retained.json';run.write_new(path,{'old':True})
    with pytest.raises(FileExistsError):run.write_new(path,{'new':True})
    assert run.read(path)=={'old':True}


def test_proc_stat_handles_parentheses_and_start_time():
    fields=['R']+['0']*18+['123456']+['0']*4
    assert run.parse_proc_stat('77 (python (worker)) '+' '.join(fields))=={'state':'R','start_ticks':'123456'}
