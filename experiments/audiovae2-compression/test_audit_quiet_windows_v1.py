from pathlib import Path
import sys
import pytest
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import audit_quiet_windows_v1 as audit


def crop(reference,context_frames=0,start_frame=0):
    return {'source_id':'sample','reference16k':reference.reshape(1,1,-1),
        'context_frames':context_frames,'start_frame':start_frame,'valid_scored_samples':reference.numel()*3-context_frames*1920}


def describe(p,t):
    p=p.reshape(1,1,-1);t=t.reshape(1,1,-1)
    c=crop(torch.zeros((p.numel()+2)//3));c['valid_scored_samples']=p.numel()
    raw=audit.base.quiet_window_metrics(p,t)
    return audit.describe_quiet_windows(p,t,c,raw),raw,c


def test_reference_uses_context_and_exact_partial_three_to_one_overlap():
    ref=torch.zeros(644,dtype=torch.float64);ref[640:]=torch.tensor([1.,2.,3.,4.])
    result=audit.source_reference_window(crop(ref,1,30),1,8)
    # Global crop positions1921..1928:2 slots of1,3 slots of2,2 slots of3.
    assert result['source_reference_rms']==pytest.approx((32/7)**.5)
    assert result['source_reference_window_16k']==[640,643]
    assert result['source_reference_weighted_48k_samples']==7
    assert result['source_reference_exact_zero'] is False
    assert audit.source_reference_window(crop(torch.zeros(10)),0,29)['source_reference_exact_zero'] is True
    with pytest.raises(ValueError):audit.source_reference_window(crop(torch.zeros(2)),0,7)


def test_residual_only_can_be_attenuation_without_added_rms():
    t=torch.full((960,),1e-4);p=torch.zeros_like(t)
    rows,_,_=describe(p,t);r=rows[0]
    assert r['failure_category']=='residual_only'
    assert r['least_squares_gain']==0
    assert r['output_rms_over_limit']==0
    assert r['source_reference_exact_zero'] is True
    assert r['teacher_level_bin'] in ('nonzero_to_1e-5','1e-5_to_1e-4','1e-4_to_1e-3')
    assert r['dc_energy_fraction']==pytest.approx(1)


def test_amplitude_only_is_separate_from_residual_and_preserves_native_thresholds():
    t=torch.full((960,),2e-4);p=t*1.13
    rows,raw,_=describe(p,t)
    assert rows[0]['failure_category']=='amplitude_only'
    assert rows[0]['residual_limit']==raw['windows'][0]['residual_limit']
    assert rows[0]['output_rms_limit']==raw['windows'][0]['output_rms_limit']
    assert rows[0]['least_squares_gain']==pytest.approx(1.13,rel=1e-6)


def test_zero_teacher_has_no_invented_gain_and_dc_ac_are_separate():
    t=torch.zeros(1920);p=torch.cat((torch.full((960,),2e-5),torch.tensor([2e-5,-2e-5]).repeat(480)))
    rows,_,_=describe(p,t)
    assert all(r['failure_category']=='both' for r in rows)
    assert all(r['least_squares_gain'] is None and r['cosine'] is None for r in rows)
    assert rows[0]['dc_energy_fraction']==pytest.approx(1)
    assert rows[1]['dc_energy_fraction']==pytest.approx(0)
    summary=audit.summarize_windows(rows)
    assert summary['window_dc_energy_fraction']==pytest.approx(.5)
    assert summary['centered_residual_rms']==pytest.approx(2e-5/(2**.5),rel=1e-6)


def test_grid_time_partial_tail_and_quiet_neighbor_labels():
    t=torch.cat((torch.full((960,),.03),torch.zeros(966)));p=t.clone()
    rows,raw,c=describe(p,t);c['start_frame']=21;c['context_frames']=30
    result=audit.describe_quiet_windows(p.reshape(1,1,-1),t.reshape(1,1,-1),{**c,'reference16k':torch.zeros(1,1,30000)},raw)
    assert len(result)==2 and result[0]['adjacent_active_window'] is True
    assert result[1]['partial_window'] is True and result[1]['valid_samples']==6
    assert result[0]['source_start_sample']==21*1920+960
    assert result[0]['temporal_bin']=='source_after800ms' and result[0]['actual_startup'] is False


def test_phase_templates_align_source_grid_and_detect_periodic_ac():
    p=torch.tensor([1.,-1.,2.,-2.],dtype=torch.float64).repeat(20).reshape(1,1,-1);t=torch.zeros_like(p)
    c=crop(torch.zeros(27));c['start_frame']=1
    raw={'windows':[{'is_quiet':True,'start_sample':1,'stop_sample':80}]}
    result=audit.phase_templates(p,t,c,raw,periods=(4,),min_cycles=8)[0]
    assert result['eligible'] and result['complete_cycles']==19
    assert result['aligned_source_samples']==[1924,2000]
    assert result['phase_template_ac_fraction_of_centered_power']==pytest.approx(1)
    assert result['phase_template_means']==[1.,-1.,2.,-2.]


def test_phase_analysis_never_concatenates_quiet_regions_or_accepts_short_runs():
    p=torch.randn(1,1,200);t=torch.zeros_like(p);c=crop(torch.zeros(67))
    raw={'windows':[{'is_quiet':True,'start_sample':0,'stop_sample':28},
        {'is_quiet':False,'start_sample':28,'stop_sample':60},{'is_quiet':True,'start_sample':60,'stop_sample':88}]}
    rows=audit.phase_templates(p,t,c,raw,periods=(4,),min_cycles=8)
    assert len(rows)==2 and all(not r['eligible'] for r in rows)
    assert [r['complete_cycles'] for r in rows]==[7,7]
    with pytest.raises(ValueError):audit.phase_templates(p,t,c,raw,min_cycles=2)


def test_detailed_observer_preserves_one_scored_teacher_call_and_restores_on_error(monkeypatch):
    c=crop(torch.zeros(320));c.update(teacher_audio=torch.zeros(1,1,960),latents=torch.zeros(1,64,1))
    original_q=audit.base.quiet_window_metrics;calls=[]
    def teacher_forward(_teacher,z):
        calls.append('teacher');return {'waveform':c['teacher_audio'],'group_input':torch.zeros(1,1,1)}
    monkeypatch.setattr(audit.base,'teacher_forward',teacher_forward)
    monkeypatch.setattr(audit.common,'warm_student',lambda *_args:None)
    def evaluate(_model,teacher,crops,spectral,batch_size=1):
        audit.base.teacher_forward(teacher,crops[0]['latents'])
        p=torch.zeros(1,1,960);q=audit.base.quiet_window_metrics(p,p)
        return {'rows':[{'source_id':'sample','samples':960,'quiet_windows':1,'quiet_failed':0}],
            'aggregate':{'quiet_windows':1,'quiet_failed_windows':0}}
    monkeypatch.setattr(audit.base,'evaluate',evaluate)
    result=audit.evaluate_detailed(None,None,[c],None)
    assert calls==['teacher'] and result['summary']['all']['windows']==1
    assert audit.base.quiet_window_metrics is original_q and audit.base.teacher_forward is teacher_forward
    def fail(*args,**kwargs):raise RuntimeError('sentinel')
    monkeypatch.setattr(audit.base,'evaluate',fail)
    with pytest.raises(RuntimeError,match='sentinel'):audit.evaluate_detailed(None,None,[c],None)
    assert audit.base.quiet_window_metrics is original_q and audit.base.teacher_forward is teacher_forward


def test_startup_flag_is_window_local_not_entire_source_starting_crop():
    p=torch.zeros(1,1,40000);c=crop(torch.zeros(13334));c['valid_scored_samples']=40000
    raw=audit.base.quiet_window_metrics(p,p)
    rows=audit.describe_quiet_windows(p,p,c,raw)
    assert all(r['source_starting_crop'] for r in rows)
    assert sum(r['actual_startup'] for r in rows)==2
    assert not rows[-1]['actual_startup'] and rows[-1]['temporal_bin']=='source_after800ms'


def test_pairing_rejects_threshold_and_source_reference_drift():
    rows,_,_=describe(torch.zeros(960),torch.full((960,),1e-4))
    result=audit.pair_windows(rows,rows)
    assert result['category_transitions']=={'residual_only -> residual_only':1}
    for key in ('residual_limit','source_reference_rms'):
        changed=[{**rows[0],key:rows[0][key]+1e-8}]
        with pytest.raises(RuntimeError,match='thresholds'):audit.pair_windows(rows,changed)
    with pytest.raises(RuntimeError,match='identities'):audit.pair_windows(rows,rows+rows)
