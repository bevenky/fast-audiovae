"""Analytic gradient directions and saved-state AdamW arithmetic, CPU only."""
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import diagnose_group_gradients as probe


def test_hypothetical_adam_matches_toy_optimizer_without_changing_saved_moments():
    p=torch.nn.Parameter(torch.tensor([.4,-.7,1.2],dtype=torch.float64))
    optimizer=torch.optim.AdamW([p],lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    for g in ([.2,-.3,.1],[-.1,.4,.3]):
        p.grad=torch.tensor(g,dtype=p.dtype);optimizer.step()
    state=optimizer.state[p];m=state['exp_avg'].clone();v=state['exp_avg_sq'].clone();step=int(state['step'])
    gradient=torch.tensor([.7,-.2,-.4],dtype=p.dtype)
    prediction=probe.adam_displacement(gradient,m,v,step,3e-5)
    torch.testing.assert_close(m,state['exp_avg'],atol=0,rtol=0);torch.testing.assert_close(v,state['exp_avg_sq'],atol=0,rtol=0)
    before=p.detach().clone();p.grad=gradient;optimizer.step()
    torch.testing.assert_close(p.detach()-before,prediction,rtol=1e-10,atol=1e-16)


def test_conflicting_branch_and_actual_delta_have_correct_directional_signs():
    grads={'waveform':torch.tensor([1.,0.]),'mel_linear':torch.tensor([0.,0.]),
           'mel_log':torch.tensor([-3.,0.]),'feature':torch.tensor([0.,1.]),
           'near_waveform':torch.tensor([1.,0.]),'near_amplitude_violation':torch.tensor([0.,0.])}
    coefficient={k:1. for k in probe.CORE}
    result=probe.vector_relations(grads,coefficient,torch.tensor([.2,-.1]),torch.tensor([.3,-.4]),{'stage2':[(0,1)],'stage3':[(1,2)]})
    assert result['pairwise_parameter_gradient_cosine']['waveform']['mel_log']==-1
    assert result['pairwise_parameter_gradient_cosine']['near_amplitude_violation']['waveform'] is None
    assert result['raw_gradient_descent_directional_derivatives']['near_waveform']==2
    assert result['hypothetical_adam_directional_derivatives']['near_waveform']>0
    assert result['actual_delta_local_derivatives_by_parameter_group']['stage2']['near_waveform']>0
    assert result['actual_delta_local_derivatives_by_parameter_group']['stage3']['near_waveform']==0
