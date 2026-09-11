"""Analysis-only validation of saved corrected-screen artifacts."""
from copy import deepcopy
import json

import pytest
import torch

from summarize_corrected_screen import (gradient_summary,validate_streams,validate_gradient_audit,
    validate_calibration,state_fingerprint,validate_report,json_metadata)
from audiovae_student.objective_comparison import state_fingerprint as training_fingerprint


def metrics(path,steps):
    rows=[{'step':step,'examples':32,'scored_samples':10000,'discriminator_updates':i+8,
        **{name+'/scaled_norm':value for name,value in
           [('teacher_waveform',.3),('teacher_mel',.4),('feature_matching',.2),('adversarial',.1)]}}
        for i,step in enumerate(steps)]
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    return rows


def test_metrics_require_every_update_not_only_count_and_endpoints(tmp_path):
    path=tmp_path/'metrics.jsonl'
    metrics(path,[8091,8092,8092,8094])
    with pytest.raises(ValueError,match='duplicated'):
        gradient_summary(path,expected_steps=range(8091,8095))
    metrics(path,[8091,8092,8093,8094])
    result=gradient_summary(path,expected_steps=range(8091,8095),expected_examples=32,
        expected_sample_counts=[10000]*4,parent_discriminator_updates=7)
    assert result['all']==pytest.approx(dict(teacher_waveform=.3,teacher_mel=.4,feature_matching=.2,adversarial=.1))
    with pytest.raises(ValueError,match='valid samples'):
        gradient_summary(path,expected_sample_counts=[10000,9999,10000,10000])
    with pytest.raises(ValueError,match='counter'):
        gradient_summary(path,parent_discriminator_updates=8)


def streams():
    result={}
    for case in range(2):
        for frames in (2,4):
            counts=[frames]*(7//frames)+([7%frames] if 7%frames else [])
            result[f'{case}_{frames}']={'passed':True,'sample_count_passed':True,'frames_per_chunk':frames,
                'numerical_tolerance':2e-6,'rtf_measured':False,'input_frames':7,'expected_samples':7*1920,
                'batch_samples':7*1920,'stream_samples':7*1920,'flush_samples':0,'max_abs_error':1e-7,
                'chunks':[{'input_frames':n,'expected_samples':n*1920,'output_samples':n*1920} for n in counts]}
    return result


@pytest.mark.parametrize('kind',['empty','missing','lost_chunk','wrong_total','nonfinite','wrong_duration'])
def test_streaming_flags_alone_cannot_hide_missing_samples(kind):
    report=streams()
    validate_streams(report)
    if kind=='empty':report={}
    elif kind=='missing':report.pop('1_4')
    elif kind=='lost_chunk':report['0_2']['chunks'].pop()
    elif kind=='wrong_total':report['0_2']['stream_samples']-=1
    elif kind=='nonfinite':report['0_2']['max_abs_error']=float('nan')
    else:report['0_2']['frames_per_chunk']=4
    with pytest.raises(ValueError):validate_streams(report)


def test_gradient_audit_is_bound_to_model_optimizer_and_step():
    audit={'method':'fixed_weight_output_gradient_audit','parameter_updates':0,'optimizer_updates':0,
           'fixed_state':{'model':'model','optimizer':'optimizer','step':8090}}
    validate_gradient_audit(audit,step=8090,model_sha='model',optimizer_sha='optimizer')
    with pytest.raises(ValueError):validate_gradient_audit(audit,step=8090,model_sha='other',optimizer_sha='optimizer')
    with pytest.raises(ValueError):validate_gradient_audit(audit,step=8091,model_sha='model',optimizer_sha='optimizer')


def calibration():
    state={'format_version':1,'config':{'ema_decay':.999},'weights':{'teacher_waveform':.3,'teacher_mel':.4,
        'feature_matching':.2,'adversarial':.1},'updates':8090,
        'ema':{name:{'total':900.,'weight':900.} for name in
               ('teacher_waveform','teacher_mel','feature_matching','adversarial')}}
    parent={'balancer':state,'model':{'w':torch.zeros(2)},'optimizer':{},'calibration':{'valid':True},
            'step':8090,'discriminator_updates':7590}
    after=deepcopy(state)
    for name in ('feature_matching','adversarial'):after['ema'][name]={'total':.003,'weight':1.}
    report={'method':'fixed_weight_selected_loss_ema_reset','calibrated_loss_names':['feature_matching','adversarial'],
        'parameter_updates':0,'optimizer_updates':0,'batches':1,'examples':32,
        'balancer_training_updates_preserved':True,
        'provenance':{'split':'train','pool':'gradient_calibration','data_plan_sha256':'data',
                      'fresh_discriminator_warmup_updates':64},
        'balancer_before':deepcopy(state),'balancer_after':after,
        'fixed_state':{**{key:state_fingerprint(parent[key]) for key in ('model','optimizer','calibration')},
                       'step':8090,'discriminator_updates':7590},
        'loss_statistics':{name:{'positive_observations':1} for name in ('feature_matching','adversarial')},
        'before':[{}],'after':[{}]}
    return parent,report


@pytest.mark.parametrize('kind',['unselected_ema','optimizer','zero_observations','updates','validation'])
def test_calibration_receipt_cannot_hide_other_resets(kind):
    parent,report=calibration()
    kwargs=dict(data_sha='data',batches=1,examples=32,warmup=64)
    validate_calibration(report,parent,**kwargs)
    if kind=='unselected_ema':report['balancer_after']['ema']['teacher_waveform']['total']=1
    elif kind=='optimizer':report['fixed_state']['optimizer']='different'
    elif kind=='zero_observations':report['loss_statistics']['adversarial']['positive_observations']=0
    elif kind=='updates':report['balancer_after']['updates']=8091
    else:report['provenance']['split']='validation'
    with pytest.raises(ValueError):validate_calibration(report,parent,**kwargs)


def test_report_requires_addon_membership_and_full_model_identity():
    migration={'variant':'control','architecture':{'terminal_tanh':False}}
    config={'example':(1,2)}
    full=json_metadata({'class':'audiovae_student.model.StudentDecoder','base_config':config,
        'architecture_config':None,'fusion_migration':migration,'parameter_state_sha256':'parameters'})
    full_sha=state_fingerprint(full)
    report={'evaluated_step':8090,'model_identity':full,'model_state_sha256':full_sha,
        'model_tensors_preserved':True,'module_modes_preserved':True,'global_rng_preserved':True,
        'optimizer_updates':0,'rows':[{'source_id':'addon','start_frame':0,'samples':343158,
            'original_scored_samples':343158,'context_excluded_samples':0,'context_start_frame':0,
            'model_state_sha256':full_sha,'evaluated_step':8090}]}
    expected={('addon',0):{'start_frame':0,'context_frames':0,'valid_scored_samples':343158}}
    kwargs=dict(variant='control',base_config=config,migration=migration)
    validate_report(report,expected,8090,'parameters',**kwargs)
    missing={**expected,('speech',0):{'start_frame':0,'context_frames':0,'valid_scored_samples':48000}}
    with pytest.raises(ValueError,match='membership'):validate_report(report,missing,8090,'parameters',**kwargs)
    report['model_state_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):validate_report(report,expected,8090,'parameters',**kwargs)


def test_analysis_fingerprint_matches_training_for_nested_tensor_state():
    state={'optimizer':{1:{'step':torch.tensor(5.),'moment':torch.arange(12).reshape(3,4)}},
           'config':{'tuple':(1,2),'list':[1,2]},'flag':True,'absent':None}
    assert state_fingerprint(state)==training_fingerprint(state)
