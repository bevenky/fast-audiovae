"""Isolated contribution accounting and first-mixer affine reconstruction.

Uses a fresh teacher-derived sliced initialization, never an adapted checkpoint.
No optimizer, neural training or automatic promotion is permitted here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
import replay_late_segment as replay
import group_model as group
from unified_monitor import UnifiedMonitor

base, screen, resume = replay.base, replay.screen, replay.resume
diagnostic_quiet_metrics = base.quiet_window_metrics
VERSION = 'audiovae2_channel_contribution_v1'
PRIMARY = ('stage2_ru1','stage3_up','stage4_up')


def linear_response(module, x, weight=None, include_bias=False):
    """Native affine arithmetic, including causal transpose routing and trim."""
    weight = group.effective_weight(module) if weight is None else weight
    bias = module.bias if include_bias else None
    if isinstance(module, nn.ConvTranspose1d):
        s=module.stride[0]
        if (module.kernel_size!=(2*s,) or module.padding!=(0,) or module.output_padding!=(0,)
                or module.dilation!=(1,) or module.groups!=1):
            raise ValueError('Only the pinned causal transpose geometry is supported')
        result=F.conv_transpose1d(x,weight,bias,stride=s)
        return result[...,:x.shape[-1]*s]
    if (not isinstance(module,nn.Conv1d) or module.kernel_size!=(1,) or module.stride!=(1,)
            or module.padding!=(0,) or module.groups!=1): raise ValueError('Expected a pointwise mixer')
    return F.conv1d(x,weight,bias)


def operation_decomposition(teacher_module,student_module,teacher_input,student_input,
                            teacher_output,student_output,kept_inputs,kept_outputs):
    w=group.effective_weight(teacher_module).detach()
    ins=torch.as_tensor(kept_inputs,device=w.device,dtype=torch.long)
    outs=torch.as_tensor(kept_outputs,device=w.device,dtype=torch.long)
    allins=torch.arange(teacher_input.shape[1],device=w.device)
    dropped=allins[~torch.isin(allins,ins)]
    if (ins.unique().numel()!=ins.numel() or outs.unique().numel()!=outs.numel()
            or student_input.shape[1]!=len(ins) or student_output.shape[1]!=len(outs)):
        raise ValueError('Invalid selected channel geometry')
    if isinstance(teacher_module,nn.ConvTranspose1d):
        retained_w=w.index_select(0,ins).index_select(1,outs)
        dropped_w=w.index_select(0,dropped).index_select(1,outs)
    else:
        retained_w=w.index_select(0,outs).index_select(1,ins)
        dropped_w=w.index_select(0,outs).index_select(1,dropped)
    actual_w=group.effective_weight(student_module).detach()
    if not torch.allclose(actual_w,retained_w,rtol=1e-6,atol=1e-7):
        raise ValueError('Contribution identity applies only to the original sliced weights')
    xt=teacher_input.index_select(1,ins)
    kept=linear_response(student_module,xt,retained_w)
    removed=linear_response(student_module,teacher_input.index_select(1,dropped),dropped_w) if len(dropped) else torch.zeros_like(kept)
    bias=torch.zeros_like(kept)
    if teacher_module.bias is not None: bias=teacher_module.bias.detach().index_select(0,outs)[None,:,None].expand_as(kept)
    if (teacher_module.bias is None)!=(student_module.bias is None): raise ValueError('Bias presence changed')
    if student_module.bias is not None and not torch.equal(student_module.bias.detach(),teacher_module.bias.detach().index_select(0,outs)):
        raise ValueError('Contribution identity requires original selected bias')
    drift=linear_response(student_module,xt-student_input,retained_w)
    selected=teacher_output.index_select(1,outs)
    result={'kept':kept,'dropped':removed,'bias':bias,'teacher_selected':selected,
        'student':student_output,'retained_drift':drift,'gap':selected-student_output,
        'reconstructed_gap':removed+drift,'retained_input_error':xt-student_input}
    if isinstance(teacher_module,nn.ConvTranspose1d):
        s=teacher_module.stride[0];current=dropped_w.clone();current[...,s:]=0
        previous=dropped_w.clone();previous[...,:s]=0
        result['dropped_current']=linear_response(student_module,teacher_input.index_select(1,dropped),current)
        result['dropped_previous']=linear_response(student_module,teacher_input.index_select(1,dropped),previous)
    return result


def cell_weights(valid, output_length):
    """Count valid waveform samples in each feature cell, including short tails."""
    if valid.dtype!=torch.bool or valid.ndim!=3 or valid.shape[1]!=1 or output_length<=0 or valid.shape[-1]%output_length:
        raise ValueError('Invalid waveform-to-feature mask geometry')
    return valid.reshape(valid.shape[0],1,output_length,-1).sum(-1)


def accumulate_affine(stats,x,d,weights):
    if (x.ndim!=3 or d.ndim!=3 or x.shape[0]!=d.shape[0] or x.shape[-1]!=d.shape[-1]
            or weights.shape!=(x.shape[0],1,x.shape[-1]) or not torch.isfinite(weights).all() or (weights<0).any()):
        raise ValueError('Invalid weighted affine observations')
    xx=x.detach().movedim(1,-1).reshape(-1,x.shape[1]).double()
    dd=d.detach().movedim(1,-1).reshape(-1,d.shape[1]).double()
    ww=weights.detach().reshape(-1).double()
    keep=ww>0;xx,dd,ww=xx[keep],dd[keep],ww[keep]
    if not torch.isfinite(xx).all() or not torch.isfinite(dd).all(): raise ValueError('Nonfinite valid affine observations')
    if stats is None:
        stats={'n':torch.zeros((),device=x.device,dtype=torch.float64),
            'x_sum':torch.zeros(x.shape[1],device=x.device,dtype=torch.float64),
            'xx_sum':torch.zeros(x.shape[1],x.shape[1],device=x.device,dtype=torch.float64),
            'd_sum':torch.zeros(d.shape[1],device=x.device,dtype=torch.float64),
            'xd_sum':torch.zeros(x.shape[1],d.shape[1],device=x.device,dtype=torch.float64)}
    if ww.numel():
        chunk_n=ww.sum();mx=(xx*ww[:,None]).sum(0)/chunk_n;md=(dd*ww[:,None]).sum(0)/chunk_n
        xc,dc=xx-mx,dd-md;cx=xc.T@(xc*ww[:,None]);cd=xc.T@(dc*ww[:,None])
        if 'centered_xx' not in stats:
            stats.update(mean_x=mx,mean_d=md,centered_xx=cx,centered_xd=cd)
        else:
            old=stats['n'];total=old+chunk_n;dx=mx-stats['mean_x'];dy=md-stats['mean_d'];factor=old*chunk_n/total
            stats['centered_xx']+=cx+factor*dx[:,None]*dx[None,:]
            stats['centered_xd']+=cd+factor*dx[:,None]*dy[None,:]
            stats['mean_x']+=dx*(chunk_n/total);stats['mean_d']+=dy*(chunk_n/total)
    stats['n']+=ww.sum();stats['x_sum']+=(xx*ww[:,None]).sum(0)
    stats['xx_sum']+=xx.T@(xx*ww[:,None]);stats['d_sum']+=(dd*ww[:,None]).sum(0)
    stats['xd_sum']+=xx.T@(dd*ww[:,None])
    return stats


def fit_affine(stats,ridge=1e-6):
    if not math.isfinite(ridge) or ridge<=0: raise ValueError('A fixed positive ridge is required')
    n=torch.as_tensor(stats['n'],dtype=torch.float64,device=stats['xx_sum'].device)
    if not torch.isfinite(n) or n<=0: raise ValueError('No valid fitting observations')
    if 'centered_xx' in stats:
        mx,md=stats['mean_x'],stats['mean_d'];cov=stats['centered_xx']/n;cross=stats['centered_xd']/n
    else:
        mx=stats['x_sum'].double()/n;md=stats['d_sum'].double()/n
        cov=stats['xx_sum'].double()/n-mx[:,None]*mx[None,:]
        cross=stats['xd_sum'].double()/n-mx[:,None]*md[None,:]
    cov=(cov+cov.T)/2
    if not torch.isfinite(cov).all() or not torch.isfinite(cross).all(): raise ValueError('Nonfinite affine statistics')
    eigen=torch.linalg.eigvalsh(cov);largest=float(eigen[-1]);trace=float(cov.trace())
    tolerance=max(abs(largest),1.)*cov.shape[0]*torch.finfo(torch.float64).eps
    if float(eigen[0]) < -tolerance: raise ValueError('Centered covariance is not positive semidefinite within roundoff')
    if trace<=tolerance:
        a=torch.zeros((md.numel(),mx.numel()),device=cov.device,dtype=torch.float64);lam=0.;rank=0;condition=None;residual=0.
    else:
        lam=ridge*trace/cov.shape[0]
        if float(eigen[0])+lam<=0: raise RuntimeError('Regularized covariance is not positive definite')
        regular=cov+lam*torch.eye(cov.shape[0],device=cov.device,dtype=torch.float64)
        a=torch.linalg.solve(regular,cross).T
        rank=int((eigen>max(largest,0.)*cov.shape[0]*torch.finfo(torch.float64).eps).sum())
        condition=(largest+lam)/(float(eigen[0])+lam)
        residual=float((regular@a.T-cross).norm()/cross.norm().clamp_min(torch.finfo(torch.float64).tiny))
    c=md-a@mx
    if not torch.isfinite(a).all() or not torch.isfinite(c).all() or residual>1e-7:
        raise RuntimeError('Affine solution is nonfinite or numerically inaccurate')
    report={'weighted_sample_count':float(n),'input_channels':mx.numel(),'output_channels':md.numel(),
        'ridge_factor':ridge,'ridge_lambda':lam,'effective_rank':rank,'eigenvalue_min':float(eigen[0]),
        'eigenvalue_max':largest,'ridge_condition':condition,'raw_condition':largest/float(eigen[0]) if float(eigen[0])>tolerance else None,
        'normal_equation_relative_residual':residual,'correction_frobenius':float(a.norm()),'intercept_l2':float(c.norm()),
        'intercept_only':rank==0,'centering':'weighted mean, intercept unpenalized; no feature/RMS normalization',
        'statistics':'weighted centered Chan merge' if 'centered_xx' in stats else 'supplied raw moments',
        'solver':'float64 centered covariance; fixed lambda=1e-6*trace(Cxx)/input_channels; no held-out tuning'}
    return a,c,report


def affine_response(a,c,x):
    return F.conv1d(x,a.to(x).unsqueeze(-1),c.to(x))


@torch.no_grad()
def fold_affine_correction(module,a,c):
    if not isinstance(module,nn.Conv1d) or module.kernel_size!=(1,) or module.groups!=1 or module.bias is None:
        raise ValueError('Correction requires an existing biased pointwise convolution')
    w=group.effective_weight(module).detach()
    if a.shape!=w.shape[:2] or c.shape!=module.bias.shape: raise ValueError('Affine correction shape differs')
    group.assign_effective_weight(module,w+a.to(w).unsqueeze(-1),module.bias.detach()+c.to(module.bias))


def validate_fit_sources(calibration,development):
    left=[c['source_id'] for c in calibration];right=[c['source_id'] for c in development]
    if not left or not right or len(left)!=len(set(left)) or len(right)!=len(set(right)) or set(left)&set(right):
        raise ValueError('Fitting and held-out sources must be nonempty, unique and disjoint')
    return {'fit_source_ids':left,'development_source_ids':right,'fit_sha256':screen.digest(left),'development_sha256':screen.digest(right)}


@dataclass
class Operation:
    name: str
    path: str
    kept_inputs: list[int]
    kept_outputs: list[int]
    rate: int
    stride: int = 1


def operations(selection):
    k2,k3=selection['stage2_indices'],selection['stage3_indices']
    result=[]
    for stage,indices,rate in ((2,k2,1200),(3,k3,6000)):
        if stage==3:result.append(Operation('stage3_up','model.4.block.1',k2,k3,6000,5))
        for unit in range(1,4):
            result.append(Operation(f'stage{stage}_ru{unit}',f'model.{stage+1}.block.{unit+1}.block.3',indices,indices,rate))
    result.append(Operation('stage4_up','model.5.block.1',k3,list(range(128)),12000,2))
    return result


@contextmanager
def capture_operations(decoder,specs):
    captured={};handles=[]
    try:
        for spec in specs:
            def hook(module,args,output,name=spec.name):
                captured[name]={'input':args[0].detach(),'output':output.detach()}
            handles.append(decoder.get_submodule(spec.path).register_forward_hook(hook))
        yield captured
        if set(captured)!={s.name for s in specs}:raise RuntimeError('Missing operation capture')
    finally:
        for handle in handles:handle.remove()


@contextmanager
def temporary_output_change(module,change):
    handle=module.register_forward_hook(lambda _module,_args,output:change(output))
    try:yield
    finally:handle.remove()


def ablate_kept_output(output,missing,kept_outputs):
    indices=torch.as_tensor(kept_outputs,device=output.device,dtype=torch.long)
    if (missing.shape!=(output.shape[0],len(indices),output.shape[-1])
            or len(indices.unique())!=len(indices)):raise ValueError('Ablation output geometry differs')
    result=output.clone()
    result[:,indices,:]=output.index_select(1,indices)-missing
    return result


def dropped_response(module,teacher_input,spec):
    w=group.effective_weight(module).detach();keep=torch.as_tensor(spec.kept_inputs,device=w.device)
    allins=torch.arange(teacher_input.shape[1],device=w.device);drop=allins[~torch.isin(allins,keep)]
    outs=torch.as_tensor(spec.kept_outputs,device=w.device)
    selected=w.index_select(0,drop).index_select(1,outs) if isinstance(module,nn.ConvTranspose1d) else w.index_select(0,outs).index_select(1,drop)
    return linear_response(module,teacher_input.index_select(1,drop),selected)


def region_masks(target,valid,crop):
    a=crop['context_frames']*1920;b=a+crop['valid_scored_samples']
    scored=target[...,a:b]
    quiet=diagnostic_quiet_metrics(scored,scored,torch.ones_like(scored,dtype=torch.bool))
    masks={k:torch.zeros_like(valid) for k in ('quiet','near_silence','active','startup_40ms')};masks['all']=valid
    for window in quiet['windows']:
        lo,hi=a+window['start_sample'],a+window['stop_sample']
        masks['quiet' if window['is_quiet'] else 'active'][...,lo:hi]=True
        if window['teacher_rms']<=1e-5:masks['near_silence'][...,lo:hi]=True
    absolute=torch.arange(valid.shape[-1],device=valid.device)+crop['context_start_frame']*1920
    masks['startup_40ms']=valid&(absolute[None,None,:]<1920)
    return masks


def weighted_tensor_summary(tensor,weights):
    x=tensor.detach().double();w=weights.to(x);mass=float(w.sum())
    if mass==0:return {'weighted_samples':0,'mean':None,'rms':None,'channel_centered_rms':None}
    selected=w>0;x=x.masked_fill(~selected.expand_as(x),0)
    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite scored contribution')
    means=(x*w).sum((0,2))/mass;n=mass*x.shape[1]
    energy=float((x.square()*w).sum());centered=(x-means[None,:,None]).square()*w
    return {'weighted_samples':mass,'channels':x.shape[1],'mean':float(means.mean()),
        'rms':math.sqrt(energy/n),'channel_centered_rms':math.sqrt(float(centered.sum())/n),
        'channel_mean_rms':float(means.square().mean().sqrt()),'channel_mean_min':float(means.min()),
        'channel_mean_max':float(means.max())}


def additive_energy(terms,weights):
    w=weights.double();n=float(w.sum())*terms['kept'].shape[1]
    if not n:return {'weighted_elements':0}
    mask=weights>0
    raw_kept=terms['kept'].double().masked_fill(~mask,0)
    bias=terms['bias'].double().masked_fill(~mask,0)
    kept=raw_kept+bias;dropped=terms['dropped'].double().masked_fill(~mask,0)
    ke=float((kept.square()*w).sum())/n;de=float((dropped.square()*w).sum())/n
    cross=float((kept*dropped*w).sum())/n
    mass=float(w.sum());mean_kept=(kept*w).sum((0,2))/mass;mean_dropped=(dropped*w).sum((0,2))/mass
    dc_cross=float((mean_kept*mean_dropped).mean())
    teacher=float((terms['teacher_selected'].double().masked_fill(~mask,0).square()*w).sum())/n
    return {'weighted_elements':n,'kept_plus_bias_mean_square':ke,'dropped_mean_square':de,
        'cross_mean_product':cross,'reconstructed_mean_square':ke+de+2*cross,
        'kept_mean_square':float((raw_kept.square()*w).sum())/n,'bias_mean_square':float((bias.square()*w).sum())/n,
        'twice_kept_dropped_cross':2*float((raw_kept*dropped*w).sum())/n,
        'twice_kept_bias_cross':2*float((raw_kept*bias*w).sum())/n,
        'twice_dropped_bias_cross':2*float((dropped*bias*w).sum())/n,
        'twice_kept_plus_bias_dropped_dc_cross':2*dc_cross,
        'twice_kept_plus_bias_dropped_ac_cross':2*(cross-dc_cross),
        'teacher_mean_square':teacher,'energy_identity_absolute_error':abs(ke+de+2*cross-teacher),
        'scope':'Cross product is measured within this linear selected output, not an additive final-waveform attribution.'}


def summarize_terms(terms,masks,stride=1):
    tensors={k:v for k,v in terms.items() if k!='retained_input_error'};length=terms['dropped'].shape[-1]
    result={}
    for region,mask in masks.items():
        weights=cell_weights(mask,length)
        result[region]={name:weighted_tensor_summary(value,weights) for name,value in tensors.items()}
        result[region]['additive_energy']=additive_energy(terms,weights)
        if stride>1:
            result[region]['phases']={str(p):{name:weighted_tensor_summary(value[...,p::stride],weights[...,p::stride])
                for name,value in tensors.items() if name in ('dropped','dropped_current','dropped_previous','retained_drift')}
                for p in range(stride)}
    return result


def assert_decomposition(terms,label):
    checks={}
    for name,actual,expected in (
        ('teacher_sum',terms['kept']+terms['dropped']+terms['bias'],terms['teacher_selected']),
        ('gap_sum',terms['reconstructed_gap'],terms['gap'])):
        error=float((actual-expected).abs().max());passed=torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)
        checks[name]={'max_abs':error,'residual_rms':float((actual.double()-expected.double()).square().mean().sqrt()),'passed':passed}
        if not passed:raise RuntimeError(label+': contribution identity failed '+name)
    return checks


def trace_intervals(target,valid,crop):
    a=crop['context_frames']*1920;b=a+crop['valid_scored_samples'];n=b-a
    y=target[0,0,a:b];regions=region_masks(target,valid,crop)
    intervals={'scored_start':[a,min(b,a+4800)]}
    q=torch.nonzero(regions['near_silence'][0,0,a:b],as_tuple=False).flatten()
    if q.numel():
        point=a+int(q[0]);intervals['first_near_silence']=[max(a,point-480),min(b,point+1920)]
    peak=a+int(y.abs().argmax());intervals['teacher_peak']=[max(a,peak-1920),min(b,peak+1920)]
    return intervals


@torch.no_grad()
def warm_student(model,group_input):
    key=(tuple(group_input.shape),str(group_input.dtype),str(group_input.device))
    seen=getattr(model,'_contribution_warmed_shapes',set())
    if key not in seen:
        for _ in range(3):model.suffix_from_group(model.group_from_input(group_input))
        seen.add(key);model._contribution_warmed_shapes=seen


class MeasurementView:
    def __init__(self,model,teacher,specs,crops,variant,observations,trace_store,a=None,c=None):
        self.model,self.teacher,self.specs,self.crops=model,teacher,specs,crops
        self.variant,self.observations,self.trace_store=variant,observations,trace_store
        self.a,self.c=a,c;self.cursor=0;self.teacher_capture=None

    def group_from_input(self,x):
        crop=self.crops[self.cursor];z,target,valid,_=base.batch([crop]);self.current=(crop,target,valid)
        warm_student(self.model,x)
        first=self.specs[0]
        needed=self.specs if self.variant=='oracle_all8' else [first] if self.variant in ('fitted','oracle_first') else [
            s for s in self.specs if self.variant=='teacher_ablate_'+s.name]
        dropped={s.name:dropped_response(self.teacher.model.decoder.get_submodule(s.path),self.teacher_capture[s.name]['input'],s)
                 for s in needed}
        from contextlib import ExitStack
        with ExitStack() as stack:
            if self.variant.startswith('teacher_ablate_'):
                name=self.variant.removeprefix('teacher_ablate_');s=next(s for s in self.specs if s.name==name)
                stack.enter_context(temporary_output_change(self.teacher.model.decoder.get_submodule(s.path),
                    lambda output:ablate_kept_output(output,dropped[name],s.kept_outputs)))
                idx=group._sample_rate_index(self.teacher.model.decoder,x,None,48000)
                h=group._run(self.teacher.model.decoder,x,3,6,idx)
            else:
                restore=self.specs if self.variant=='oracle_all8' else [first] if self.variant=='oracle_first' else []
                for s in restore:
                    stack.enter_context(temporary_output_change(self.model.decoder.get_submodule(s.path),
                        lambda output,name=s.name:output+dropped[name]))
                with capture_operations(self.model.decoder,self.specs) as captured:h=self.model.group_from_input(x)
                masks=region_masks(target,valid,crop)
                if self.variant=='plain':
                    observed={}
                    for s in self.specs:
                        t,sm=self.teacher_capture[s.name],captured[s.name]
                        terms=operation_decomposition(self.teacher.model.decoder.get_submodule(s.path),self.model.decoder.get_submodule(s.path),
                            t['input'],sm['input'],t['output'],sm['output'],s.kept_inputs,s.kept_outputs)
                        checks=assert_decomposition(terms,s.name)
                        if s.name=='stage2_ru1':
                            if not torch.allclose(t['input'][:,s.kept_inputs,:],sm['input'],atol=1e-5,rtol=1e-4):
                                raise RuntimeError('First mixer no longer has matching retained inputs')
                            checks['retained_input_max_abs']=float(terms['retained_input_error'].abs().max())
                        if s.name in PRIMARY:observed[s.name]={'checks':checks,'regions':summarize_terms(terms,masks,s.stride)}
                    self.observations.append({'source_id':crop['source_id'],'operations':observed})
                elif self.variant=='fitted':
                    xp=captured[first.name]['input'];prediction=affine_response(self.a,self.c,xp);d=dropped[first.name]
                    regions={k:{'target':weighted_tensor_summary(d,cell_weights(mask,d.shape[-1])),
                        'prediction':weighted_tensor_summary(prediction,cell_weights(mask,d.shape[-1])),
                        'error':weighted_tensor_summary(prediction-d,cell_weights(mask,d.shape[-1]))} for k,mask in masks.items()}
                    self.observations.append({'source_id':crop['source_id'],'first_mixer_prediction':regions})
                elif self.variant in ('oracle_first','oracle_all8'):
                    names=[first.name] if self.variant=='oracle_first' else [s.name for s in self.specs]
                    checks={}
                    for s in self.specs:
                        if s.name not in names:continue
                        expected=self.teacher_capture[s.name]['output'][:,s.kept_outputs,:]
                        actual=captured[s.name]['output']
                        checks[s.name]={'max_abs':float((actual-expected).abs().max()),
                            'passed':torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)}
                    self.observations.append({'source_id':crop['source_id'],'restored_mixer_checks':checks})
                    if not all(v['passed'] for v in checks.values()):raise RuntimeError('Oracle restored mixer does not match original teacher coordinates')
        self.current_h=h
        return h

    def suffix_from_group(self,h):
        p=self.model.suffix_from_group(h);crop,target,valid=self.current
        if self.variant=='oracle_all8':
            if not torch.allclose(p[valid],target[valid],atol=1e-5,rtol=1e-4):
                raise RuntimeError('All-eight contribution oracle does not recover cached teacher audio')
            self.observations[-1]['waveform_teacher_max_abs']=float((p[valid]-target[valid]).abs().max())
        if self.trace_store is not None:
            self.trace_store(crop,target,p,valid,self.variant)
        self.cursor+=1
        return p


@torch.no_grad()
def evaluate_variant(model,teacher,specs,crops,spectral,variant,trace_store=None,a=None,c=None):
    observations=[];view=MeasurementView(model,teacher,specs,crops,variant,observations,trace_store,a,c)
    original=base.teacher_forward;seen=[]
    def observed(t,z):
        with capture_operations(t.model.decoder,specs) as captured:result=original(t,z)
        view.teacher_capture=captured
        crop=crops[len(seen)];start=crop['context_frames']*1920;stop=start+crop['valid_scored_samples']
        cached=crop['teacher_audio'][...,start:stop].to(z.device)
        actual=result['waveform'][...,start:stop]
        passed=torch.allclose(actual,cached,atol=1e-5,rtol=1e-4)
        seen.append({'source_id':crop['source_id'],'bitwise':torch.equal(actual,cached),'passed':passed,'max_abs':float((actual-cached).abs().max())})
        if not passed:raise RuntimeError('Pristine teacher differs from cached target')
        return result
    base.teacher_forward=observed
    try:report=UnifiedMonitor(None,{},{}).evaluate(view,teacher,crops,spectral)
    finally:base.teacher_forward=original
    if view.cursor!=len(crops) or len(seen)!=len(crops):raise RuntimeError('Incomplete synchronous source accounting')
    return {'quality':report,'observations':observations,'teacher_target_checks':seen}


@torch.no_grad()
def full_width_control(teacher,crops):
    control=base.build_student(teacher.model.decoder,list(range(512)),list(range(256)))
    rows=[];warmed=set()
    for crop in crops:
        z,target,valid,_=base.batch([crop]);expected=base.teacher_forward(teacher,z)
        shape=tuple(z.shape)
        if shape not in warmed:
            for _ in range(3):control.forward_from_latents(z)
            warmed.add(shape)
        actual=control.forward_from_latents(z);row={'source_id':crop['source_id']}
        for key in ('group_output','waveform'):
            mask=valid if key=='waveform' else cell_weights(valid,actual[key].shape[-1])>0
            x,y=actual[key].masked_select(mask),expected[key].masked_select(mask);error=x.double()-y.double()
            row[key]={'max_abs':float(error.abs().max()),'residual_rms':float(error.square().mean().sqrt()),
                'mean_residual':float(error.mean()),'bitwise_equal':torch.equal(x,y),'passed':torch.allclose(x,y,atol=1e-5,rtol=1e-4)}
            if not row[key]['passed']:raise RuntimeError('Full-width copied decoder does not match teacher')
        rows.append(row)
    return {'rows':rows,'scope':'Same-width copied full decoder numerical control, not a bound on every narrower contraction.',
        'relative_tolerance':1e-4,'absolute_tolerance':1e-5}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    for name in ('assets','base-out','manifest','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh isolated diagnostic directory')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.out.parent).free<300<<20:raise OSError('Diagnostic needs300MiB free temporary storage')
    base.policy();metadata,selection,initial,preflight,manifest,pools=screen.authenticate_inputs(args)
    split=validate_fit_sources(pools['calibration'],pools['development'])
    source_meta={r['source_id']:r for r in manifest['splits']['development']['rows']}
    panel,panel_selection=base.select_boundary_panel(pools['development'],source_meta)
    ids=list(dict.fromkeys(list(replay.CASE_IDS)+[r['source_id'] for r in panel]))
    development={c['source_id']:c for c in pools['development']};targeted=[development[s] for s in ids]
    paths=[args.base_out/'initial.pt',args.base_out/'preflight.json',args.base_out/'channel-selection.json',args.manifest,Path(__file__),
        Path(group.__file__),Path(base.__file__),Path(screen.__file__),args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth']
    protected={str(p):base.sha(p) for p in paths}
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    if not replay.compare_tree(dict(model.group_state_dict()),dict(initial['group']))['equal']:
        raise RuntimeError('Fresh sliced initialization differs from authenticated original initial group')
    common=base.objective();specs=operations(selection);first=specs[0]
    teacher_state=resume.continuation.teacher_versions(teacher);model_state=screen.frozen_versions(model)
    original_group={k:v.detach().cpu().clone() for k,v in model.group_state_dict().items()}
    args.out.mkdir();started=time.monotonic();failure=None
    identity={'version':VERSION,**split,'targeted_ids':ids,'primary_operations':list(PRIMARY),'all_operations':[s.name for s in specs],
        'initial_sha256':metadata['initial_sha256'],'selection':selection,'source_sha256':base.sha(__file__),'protected':protected,
        'fit_policy':'First stage2 RU1 only. All valid calibration cells weighted by0..40 scored waveform samples. Centered FP64 ridge.',
        'no_neural_training':True,'no_adapted_checkpoint_loaded':True,'automatic_promotion':False,'backend':replay.backend_state(),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'ridge_factor':1e-6}
    base.write_json(args.out/'launch.json',identity)
    traces={};trace_index=[]
    def save_trace(crop,target,prediction,valid,variant):
        if crop['source_id'] not in ids:return
        source_index=ids.index(crop['source_id'])
        for name,(lo,hi) in trace_intervals(target,valid,crop).items():
            key=f'source{source_index}_{name}';teacher_key=key+'_teacher'
            if teacher_key not in traces:
                traces[teacher_key]=target[0,0,lo:hi].cpu().numpy().copy()
                trace_index.append({'key':key,'source_id':crop['source_id'],'interval':name,'crop_samples':[lo,hi],
                    'absolute_source_samples':[crop['context_start_frame']*1920+lo,crop['context_start_frame']*1920+hi],
                    'sample_rate':48000,'selection':'Teacher-only scored start, first near-silence, or teacher peak; never candidate ranking'})
            traces[key+'_'+variant]=prediction[0,0,lo:hi].cpu().numpy().copy()
    try:
        base.write_json(args.out/'full-width-copy-control.json',full_width_control(teacher,targeted))
        stats=None;fit_checks=[]
        for index,crop in enumerate(pools['calibration']):
            z,t,valid,_=base.batch([crop])
            with capture_operations(teacher.model.decoder,[first]) as tc:tr=base.teacher_forward(teacher,z)
            warm_student(model,tr['group_input'])
            with capture_operations(model.decoder,[first]) as sc:model.group_from_input(tr['group_input'])
            terms=operation_decomposition(teacher.model.decoder.get_submodule(first.path),model.decoder.get_submodule(first.path),
                tc[first.name]['input'],sc[first.name]['input'],tc[first.name]['output'],sc[first.name]['output'],first.kept_inputs,first.kept_outputs)
            checks=assert_decomposition(terms,first.name)
            if not torch.allclose(terms['retained_input_error'],torch.zeros_like(terms['retained_input_error']),atol=1e-5,rtol=0):
                raise RuntimeError('Fitting first-mixer input alignment differs')
            weights=cell_weights(valid,terms['dropped'].shape[-1]);stats=accumulate_affine(stats,sc[first.name]['input'],terms['dropped'],weights)
            fit_checks.append({'source_id':crop['source_id'],'weighted_samples':int(weights.sum()),'checks':checks})
        a,c,fit=fit_affine(stats,ridge=1e-6)
        fitted=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
        fitted_frozen=screen.frozen_versions(fitted)
        fold_affine_correction(fitted.decoder.get_submodule(first.path),a,c)
        for name,value in fitted.group_state_dict().items():
            if not name.startswith(first.path+'.') and not torch.equal(value.cpu(),original_group[name]):
                raise RuntimeError('Affine fit changed a different group operation')
        # Readout-vs-fold parity on a held fitting crop, without changing its scope.
        crop=pools['calibration'][0];z,_,_,_=base.batch([crop]);tr=base.teacher_forward(teacher,z)
        with capture_operations(model.decoder,[first]) as cap:model.group_from_input(tr['group_input'])
        x=cap[first.name]['input'];original=cap[first.name]['output'];actual=fitted.decoder.get_submodule(first.path)(x)
        expected=original+affine_response(a,c,x)
        fit['fold_max_abs']=float((actual-expected).abs().max());fit['fold_passed']=torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)
        if not fit['fold_passed']:raise RuntimeError('Existing pointwise fold changes affine correction')
        fit.update(fit_sources=split['fit_source_ids'],fit_checks=fit_checks)
        base.write_json(args.out/'affine-fit.json',fit)
        torch.save({'format':VERSION,'A':a.cpu(),'c':c.cpu(),
            'folded_pointwise_state':{k:v.detach().cpu() for k,v in fitted.decoder.get_submodule(first.path).state_dict().items()},
            'identity':identity},args.out/'affine-diagnostic.pt')
        # Fit and held-out residuals are scored separately with the frozen map.
        for name,crops in (('fit',pools['calibration']),('development',pools['development'])):
            report=evaluate_variant(fitted,teacher,specs,crops,common,'fitted',save_trace if name=='development' else None,a,c)
            base.write_json(args.out/f'{name}-fitted.json',report)
            base.event('contribution_progress',completed=f'{name}-fitted',seconds=time.monotonic()-started)
        for variant,crops in [('plain',pools['development']),('oracle_first',pools['development'])]+[
                ('teacher_ablate_'+op,targeted) for op in PRIMARY]+[('oracle_all8',targeted)]:
            with replay.diagnostic_state_guard(model,teacher):
                report=evaluate_variant(model,teacher,specs,crops,common,variant,save_trace)
            base.write_json(args.out/(variant+'.json'),report)
            base.event('contribution_progress',completed=variant,seconds=time.monotonic()-started)
        # A pristine post-counterfactual pass protects against target contamination.
        with replay.diagnostic_state_guard(model,teacher):
            post=evaluate_variant(model,teacher,specs,[targeted[0]],common,'plain')
        base.write_json(args.out/'post-counterfactual-target-check.json',post['teacher_target_checks'])
        if not replay.compare_tree(dict(model.group_state_dict()),original_group)['equal']:raise RuntimeError('Pristine sliced model changed')
        if screen.frozen_versions(model)!=model_state or resume.continuation.teacher_versions(teacher)!=teacher_state:
            raise RuntimeError('Frozen model or teacher changed')
        if screen.frozen_versions(fitted)!=fitted_frozen:raise RuntimeError('Affine diagnostic changed a frozen module')
        if any(base.sha(p)!=v for p,v in protected.items()):raise RuntimeError('Original protected file changed')
        import numpy as np
        np.savez_compressed(args.out/'waveform-snippets.npz',**traces);base.write_json(args.out/'waveform-snippets.json',trace_index)
        base.write_json(args.out/'completed.json',{'version':VERSION,'completed':True,'no_neural_training':True,
            'original_files_preserved':True,'teacher_preserved':True,'initial_student_preserved':True,'automatic_promotion':False,
            'affine_fit':fit,'fit_sources':len(pools['calibration']),'development_sources':len(pools['development']),
            'targeted_sources':len(targeted),'elapsed_seconds':time.monotonic()-started,
            'limits':['Initial-coordinate interventions do not uniquely explain adapted5000 failures.',
                'Teacher ablation retains other full teacher channels downstream; it is not the sliced model.',
                'First-mixer restoration leaves later omissions intact; all8 oracle requires unavailable teacher features at inference.',
                'A one-mixer affine fit does not establish the best achievable complete nonlinear group reconstruction.']})
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc)};base.write_json(args.out/'failure.json',failure);raise
    finally:
        if traces:
            import numpy as np
            np.savez_compressed(args.out/'waveform-snippets.npz',**traces);base.write_json(args.out/'waveform-snippets.json',trace_index)
        base.write_json(args.out/'preservation.json',{'failure':failure,'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_state,
            'frozen_student_preserved':screen.frozen_versions(model)==model_state,
            'initial_group_preserved':replay.compare_tree(dict(model.group_state_dict()),original_group)['equal'],
            'original_files_preserved':all(base.sha(p)==v for p,v in protected.items())})


if __name__=='__main__':main()
