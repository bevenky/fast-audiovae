import torch
from audiovae_student.cache import TrainingCrop
from diagnostic_common import pair_metrics,aggregate_metrics,model_metrics

def crop():
    t=torch.full((1,1,3840),1e-5)
    return TrainingCrop(torch.zeros(1,64,2),t,torch.zeros(1,1,1280),'hash','source',6,5,1,1,999)

def test_teacher_self_score_and_invalid_samples_excluded():
    c=crop();p=c.teacher_audio.clone();p[...,:1926]=float('nan');p[...,2919:]=100
    r=pair_metrics(c,p)
    assert r['samples']==993 and r['mse']==0 and r['quiet_failed_windows']==0 and r['overshoot_samples']==0

def test_pooled_quiet_metric():
    c=crop();a=pair_metrics(c,c.teacher_audio);b=pair_metrics(c,2*c.teacher_audio)
    r=aggregate_metrics([a,b])
    assert abs(r['quiet_residual_rms']-1e-5/(2**.5))<1e-11

def test_model_mode_preserved():
    c=crop()
    class Fixed(torch.nn.Module):
        def forward(self,z):return c.teacher_audio
    m=Fixed().train();r=model_metrics(m,[c],'cpu')
    assert m.training and r['aggregate']['mse']==0
