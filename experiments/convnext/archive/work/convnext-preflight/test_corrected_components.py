import math
import pytest
import torch
from corrected_component_diagnostics import phase_decomposition


def test_orthogonal_dc_480_and_1920_components():
    x=torch.arange(1920*8,dtype=torch.float64)
    p=0.001+0.002*torch.sin(2*math.pi*x/480)+0.003*torch.cos(2*math.pi*x/1920)
    r=phase_decomposition(p,torch.zeros_like(p))
    assert r['quiet_cycles_40ms']==8
    assert r['power_components']['dc']==pytest.approx(1e-6)
    assert r['power_components']['phase480_ac']==pytest.approx(2e-6)
    assert r['power_components']['phase1920_additional']==pytest.approx(4.5e-6)
    assert r['power_components']['remaining']<1e-29
    assert r['decomposition_absolute_error']<1e-18


def test_teacher_quiet_definition_and_short_tail():
    t=torch.ones(1920*8+17)
    r=phase_decomposition(t,t)
    assert r['quiet_cycles_40ms']==0 and r['discarded_tail_samples']==17
    assert 'residual_rms' not in r
    r=phase_decomposition(torch.zeros(1920*8),torch.zeros(1920*8))
    assert r['residual_rms']==0
    assert all(v==0 for v in r['power_fractions'].values())


def test_invalid_input_rejected():
    with pytest.raises(ValueError):phase_decomposition(torch.zeros(5),torch.zeros(6))
    with pytest.raises(ValueError):phase_decomposition(torch.tensor([float('nan')]),torch.zeros(1))
