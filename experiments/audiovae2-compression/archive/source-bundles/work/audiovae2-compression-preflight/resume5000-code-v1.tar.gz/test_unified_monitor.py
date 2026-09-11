import pytest
import unified_monitor as module


class Writer:
    def __init__(self): self.scalars={}; self.text=[]
    def add_scalar(self,tag,value,step): self.scalars[(tag,step)]=value
    def add_text(self,*args): self.text.append(args)
    def flush(self): pass
    def add_custom_scalars(self,value): self.layout=value


def test_percent_units_target_fixed_baseline_and_no_clipping():
    writer=Writer()
    monitor=module.UnifiedMonitor(writer,{'mae':2.,'mel':0.,'group_mse':4.,'quiet_residual_rms_mean':1.}, {})
    monitor.initialize_layout()
    monitor.log_validation({'aggregate':{'nonquiet_cosine_mean':.944676,'mae':3.,'mel':1.,'group_mse':1.,
        'quiet_residual_rms_mean':.2,'quiet_windows':100,'quiet_failed_windows':25,'peak_abs_max':.9},
        'overview_window_metrics':{'near_silence_windows':20,'near_silence_failed_windows':5}},1000)
    values={k[0]:v for k,v in writer.scalars.items()}
    assert values['overview/Active waveform correlation (%)']==pytest.approx(94.4676)
    assert values['overview/Correlation target only (99%)']==99
    assert values['overview/Waveform MAE reduction vs step 0 (%)']==-50
    assert values['overview/Quiet windows passing (%)']==75
    assert values['overview/Near-silence windows passing (%)']==75
    assert 'overview/Mel error reduction vs step 0 (%)' not in values
    assert writer.layout['Progress']['All quality metrics'][0]=='Multiline'


def test_missing_near_silence_is_not_invented():
    writer=Writer(); monitor=module.UnifiedMonitor(writer,{}, {})
    monitor.log_validation({'aggregate':{'nonquiet_cosine_mean':None}},50)
    assert list(writer.scalars)==[('overview/Correlation target only (99%)',50)]
    assert 'unmeasured at this step' in writer.text[-1][1]


def window():
    return {'is_quiet':True,'teacher_rms':1e-6,'student_rms':1e-5,'valid_samples':960,'passed':False}


def test_observer_preserves_scores_and_restores_callback(monkeypatch):
    original=lambda *a,**k:{'windows':[window()]}
    monkeypatch.setattr(module.base,'quiet_window_metrics',original)
    aggregate={'mae':.01}
    def evaluate(*args,**kwargs):
        assert module.base.quiet_window_metrics(None)=={'windows':[window()]}
        return {'aggregate':aggregate,'rows':[{'source_id':'x','samples':960,'quiet_windows':1,'quiet_failed':1}]}
    monkeypatch.setattr(module.base,'evaluate',evaluate)
    result=module.UnifiedMonitor(Writer(),{},{}).evaluate(None,None,[{}],None)
    assert result['aggregate'] is aggregate
    assert result['overview_window_metrics']['near_silence_windows']==1
    assert result['overview_window_metrics']['near_silence_failed_windows']==1
    assert module.base.quiet_window_metrics is original


def test_observer_restored_after_exception(monkeypatch):
    original=module.base.quiet_window_metrics
    def evaluate(*args,**kwargs): raise RuntimeError('failed')
    monkeypatch.setattr(module.base,'evaluate',evaluate)
    with pytest.raises(RuntimeError,match='failed'):
        module.UnifiedMonitor(Writer(),{},{}).evaluate(None,None,[{}],None)
    assert module.base.quiet_window_metrics is original


def test_whistling_ratio_uses_active_energy_and_is_labeled():
    writer=Writer()
    monitor=module.UnifiedMonitor(writer,{}, {'w':{'verified_source_labels':['human_whistling_source_description']}})
    monitor.log_validation({'aggregate':{},'overview_window_metrics':{'by_source':{
        'w':{'active_teacher_energy':4.,'active_student_energy':9.}}}},1000)
    assert writer.scalars[('overview/Whistling level (% of teacher, 100 is matched)',1000)]==150
