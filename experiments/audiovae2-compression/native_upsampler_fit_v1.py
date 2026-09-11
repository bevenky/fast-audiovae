"""Native stride-two, kernel-four affine reconstruction with a shared bias.

Read-only diagnostic helpers. Fitted weights replace an existing native operator;
no new deployed layer, phase-specific bias, gradient update, or teacher mutation.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
import diagnose_channel_contributions_v2 as common

group=common.group
PATH='model.5.block.1'
VERSION='audiovae2_native_upsampler_fit_v1'


def native_design(x):
    """Phase rows map current taps0/1 and previous taps2/3, with causal zeros."""
    if x.ndim!=3 or not x.is_floating_point() or x.shape[-1]<1:
        raise ValueError('Expected floating B,C,T native input')
    previous=torch.cat((torch.zeros_like(x[...,:1]),x[...,:-1]),-1)
    pair=torch.cat((x,previous),1);zero=torch.zeros_like(pair)
    return torch.stack((torch.cat((pair,zero),1),torch.cat((zero,pair),1)),-1).flatten(-2)


def native_weight_from_affine(a,cin):
    if cin<=0 or a.ndim!=2 or a.shape[1]!=4*cin or not torch.isfinite(a).all():
        raise ValueError('Affine map must have four native input blocks')
    w=torch.empty((cin,a.shape[0],4),device=a.device,dtype=a.dtype)
    for phase in range(2):
        w[:,:,phase]=a[:,phase*2*cin:phase*2*cin+cin].T
        w[:,:,phase+2]=a[:,phase*2*cin+cin:(phase+1)*2*cin].T
    return w


def native_affine_from_weight(w):
    if w.ndim!=3 or w.shape[2]!=4:raise ValueError('Expected Cin,Cout,4 native weights')
    return torch.cat([w[:,:,p].T for p in (0,2,1,3)],1)


def accumulate_native(stats,x,target,valid):
    design=native_design(x)
    if target.shape[0]!=x.shape[0] or target.shape[-1]!=design.shape[-1]:
        raise ValueError('Target must have twice the native input rate')
    weights=common.cell_weights(valid,target.shape[-1])
    if valid.shape[-1]!=target.shape[-1]*4:raise ValueError('Expected stage4 12kHz cells at48kHz')
    return common.accumulate_affine(stats,design,target,weights)


def fit_native(stats,cin,ridge=1e-6):
    a,c,report=common.fit_affine(stats,ridge)
    w=native_weight_from_affine(a,cin)
    report.update(native_input_channels=cin,native_output_channels=w.shape[1],stride=2,kernel_size=4,
        bias='one shared output-channel intercept across both phases',
        temporal_design='phase0: current tap0 + previous tap2; phase1: current tap1 + previous tap3',
        discarded_right_overlap=2,samples_per_output_cell=4)
    return w,c,report


def validate_native(module):
    if (not isinstance(module,nn.ConvTranspose1d) or module.stride!=(2,) or module.kernel_size!=(4,)
            or module.padding!=(0,) or module.output_padding!=(0,) or module.dilation!=(1,)
            or module.groups!=1 or module.bias is None):
        raise ValueError('Expected biased native causal ConvTranspose1d stride2 kernel4')


@torch.no_grad()
def apply_native_fit(module,weight,bias):
    validate_native(module)
    expected=group.effective_weight(module)
    if weight.shape!=expected.shape or bias.shape!=module.bias.shape or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('Native fit shape or finite values invalid')
    group.assign_effective_weight(module,weight.to(expected),bias.to(module.bias))


@contextmanager
def temporary_native_fit(module,weight,bias):
    saved={k:v.detach().clone() for k,v in module.state_dict().items()}
    try:
        apply_native_fit(module,weight,bias)
        yield
    finally:module.load_state_dict(saved,strict=True)


@contextmanager
def temporary_original_residuals(student_decoder,teacher_decoder):
    """Restore only the three original full-width stage4 residual units temporarily."""
    modules=[student_decoder.get_submodule(f'model.5.block.{i}') for i in (2,3,4)]
    targets=[teacher_decoder.get_submodule(f'model.5.block.{i}') for i in (2,3,4)]
    saved=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in modules]
    source=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in targets]
    # Validate all state shapes before the first copy.
    for before,after in zip(saved,source):
        if set(before)!=set(after) or any(before[k].shape!=after[k].shape or before[k].dtype!=after[k].dtype for k in before):
            raise ValueError('Original stage4 residual contract differs')
    try:
        for module,state in zip(modules,source):module.load_state_dict(state,strict=True)
        yield
    finally:
        for module,state in zip(modules,saved):module.load_state_dict(state,strict=True)
