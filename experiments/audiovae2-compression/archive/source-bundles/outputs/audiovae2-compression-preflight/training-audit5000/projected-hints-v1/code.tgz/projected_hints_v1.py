"""Training-only, per-frame teacher hints at the two narrowed upsamplers.

Readouts remain separate from the decoder, so model export/state never contains
an auxiliary layer. Their teacher targets are always detached.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
import diagnose_channel_contributions_v2 as contribution

base=contribution.base
HINT_PATHS={'stage3_up':'model.4.block.1','stage4_up':'model.5.block.1'}
SAMPLES_PER_CELL={'stage3_up':8,'stage4_up':4}


@contextmanager
def capture_hints(decoder,*,detach=False):
    values={};handles=[]
    try:
        for name,path in HINT_PATHS.items():
            def hook(_module,_args,output,key=name):values[key]=output.detach() if detach else output
            handles.append(decoder.get_submodule(path).register_forward_hook(hook))
        yield values
        if set(values)!=set(HINT_PATHS):raise RuntimeError('Missing upsampler hint capture')
    finally:
        for h in handles:h.remove()


class LinearHint(nn.Module):
    def __init__(self,weight,bias,*,trainable=False):
        super().__init__()
        if weight.ndim!=2 or bias.shape!=(weight.shape[0],):raise ValueError('Invalid affine readout dimensions')
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():raise ValueError('Nonfinite affine readout')
        self.weight=nn.Parameter(weight.detach().clone(),requires_grad=trainable)
        self.bias=nn.Parameter(bias.detach().clone(),requires_grad=trainable)
    def forward(self,x):return F.conv1d(x,self.weight.to(x)[...,None],self.bias.to(x))


def feature_weights(valid,feature_length,samples_per_cell):
    if valid.shape[-1]!=feature_length*samples_per_cell:raise ValueError('Hint time grid does not align with waveform')
    return contribution.cell_weights(valid,feature_length)


def weighted_hint_loss(projector,student,teacher,valid,*,total_valid_samples,samples_per_cell):
    if (student.ndim!=3 or teacher.ndim!=3 or student.shape[0]!=teacher.shape[0]
            or student.shape[-1]!=teacher.shape[-1] or total_valid_samples<=0):raise ValueError('Invalid hint pair or denominator')
    weights=feature_weights(valid,student.shape[-1],samples_per_cell).to(student)
    prediction=projector(student);target=teacher.detach()
    if prediction.shape!=target.shape:raise ValueError('Projected teacher feature shape differs')
    difference=torch.where(weights>0,prediction-target,torch.zeros_like(prediction))
    return (difference.square()*weights).sum()/(total_valid_samples*target.shape[1])


def hint_losses(projectors,student,teacher,valid,total_valid_samples):
    if set(projectors)!=set(HINT_PATHS):raise ValueError('Expected exactly two external hint projectors')
    return {key:weighted_hint_loss(projectors[key],student[key],teacher[key],valid,
        total_valid_samples=total_valid_samples,samples_per_cell=SAMPLES_PER_CELL[key]) for key in HINT_PATHS}


def upstream_parameter_names(model,hint):
    if hint not in HINT_PATHS:raise ValueError('Unknown hint')
    boundary=4 if hint=='stage3_up' else 5
    prefixes=[]
    for stage in range(3,boundary):prefixes.extend((f'model.{stage}.',f'sr_cond_model.{stage}.'))
    prefixes.extend((f'sr_cond_model.{boundary}.',f'model.{boundary}.block.0.',f'model.{boundary}.block.1.'))
    return [n for n,_ in model.group_named_parameters() if n.startswith(tuple(prefixes))]


def forward_branches(model,teacher,crop,spectral,coefficients,projectors,denominators):
    z,target,valid,spans=base.batch([crop])
    with torch.no_grad(),capture_hints(teacher.model.decoder,detach=True) as teacher_features:
        trace=base.teacher_forward(teacher,z)
    with capture_hints(model.decoder) as student_features:
        h=model.group_from_input(trace['group_input']);prediction=model.suffix_from_group(h)
    existing=base.losses(prediction,target,h,trace['group_output'],valid,spans,spectral,denominators)
    hints=hint_losses(projectors,student_features,teacher_features,valid,denominators['samples'])
    return existing,hints


def training_update(model,teacher,crops,spectral,coefficients,optimizer,projectors,hint_coefficients,
                    *,projector_optimizer=None,record_diagnostics=False,record_components=False):
    denominators=base.reconstruction_denominators(crops,spectral)
    accumulated={key:0. for key in ('waveform','mel','feature',*HINT_PATHS)}
    params=base.parameters(model)
    old=[p.detach().clone() for p in params] if record_components else None
    components={k:[torch.zeros_like(p) for p in params] for k in ('existing',*HINT_PATHS)} if record_components else None
    optimizer.zero_grad(set_to_none=True)
    if projector_optimizer is not None:projector_optimizer.zero_grad(set_to_none=True)
    for crop in crops:
        existing,hints=forward_branches(model,teacher,crop,spectral,coefficients,projectors,denominators)
        total=sum(coefficients[key]*value for key,value in existing.items())
        if record_components:
            for name,branch in {'existing':total,**hints}.items():
                grads=torch.autograd.grad(branch,params,retain_graph=True,allow_unused=True)
                for dest,grad in zip(components[name],grads):
                    if grad is not None:dest.add_(grad.detach())
        # Do not add zero-valued branches to the baseline backward arithmetic.
        for key,value in hints.items():
            if hint_coefficients[key]:total=total+hint_coefficients[key]*value
        if not torch.isfinite(total):raise RuntimeError('Nonfinite projected-hint objective')
        total.backward()
        for key,value in {**existing,**hints}.items():accumulated[key]+=float(value.detach())
    params=base.parameters(model)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise RuntimeError('Missing or nonfinite student gradient')
    diagnostics={}
    if record_diagnostics:diagnostics['gradient_norm']=float(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in params])))
    optimizer.step()
    if record_components:
        delta=[p.detach()-before for p,before in zip(params,old)]
        diagnostics['student_actual_update']={'norm':_norm(delta),'component_gradient_dot_delta':{k:_dot(v,delta) for k,v in components.items()},
            'component_gradient_norm':{k:_norm(v) for k,v in components.items()},'hint_coefficients':dict(hint_coefficients)}
    if projector_optimizer is not None:
        pparams=[p for q in projectors.values() for p in q.parameters()]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in pparams):raise RuntimeError('Missing or nonfinite projector gradient')
        projector_optimizer.step()
    return {'total':sum(coefficients[k]*accumulated[k] for k in coefficients)+sum(hint_coefficients[k]*accumulated[k] for k in HINT_PATHS),**accumulated,**diagnostics}


@torch.no_grad()
def fit_projectors(model,teacher,crops,*,trainable=False,ridge=1e-6):
    ids=[c['source_id'] for c in crops]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Calibration sources must be nonempty and unique')
    stats={key:None for key in HINT_PATHS};rows=[]
    for crop in crops:
        z,_,valid,_=base.batch([crop])
        with capture_hints(teacher.model.decoder,detach=True) as tf:trace=base.teacher_forward(teacher,z)
        contribution.warm_student(model,trace['group_input'])
        with capture_hints(model.decoder,detach=True) as sf:model.group_from_input(trace['group_input'])
        for key in HINT_PATHS:
            w=feature_weights(valid,sf[key].shape[-1],SAMPLES_PER_CELL[key])
            stats[key]=contribution.accumulate_affine(stats[key],sf[key],tf[key],w)
        rows.append({'source_id':crop['source_id'],'valid_samples':int(valid.sum())})
    projectors={};reports={}
    for key in HINT_PATHS:
        a,c,report=contribution.fit_affine(stats[key],ridge=ridge)
        projectors[key]=LinearHint(a.float(),c.float(),trainable=trainable)
        reports[key]=report
    return projectors,{'source_ids':ids,'rows':rows,'projectors':reports,'ridge':ridge,'trainable_after_fit':trainable}


def _dot(left,right):return sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right))
def _norm(values):return math.sqrt(_dot(values,values))


def calibrate_hint_coefficients(model,teacher,crops,spectral,coefficients,projectors,*,fraction=.05):
    if not math.isfinite(fraction) or fraction<=0:raise ValueError('Positive finite hint gradient fraction required')
    names,params=zip(*model.group_named_parameters());denom=base.reconstruction_denominators(crops,spectral)
    gradients={k:[torch.zeros_like(p) for p in params] for k in ('existing',*HINT_PATHS)}
    losses={k:0. for k in gradients}
    with contribution.replay.diagnostic_state_guard(model,teacher):
        for crop in crops:
            existing,hints=forward_branches(model,teacher,crop,spectral,coefficients,projectors,denom)
            branches={'existing':sum(coefficients[k]*v for k,v in existing.items()),**hints}
            for i,(key,value) in enumerate(branches.items()):
                grads=torch.autograd.grad(value,params,retain_graph=i<len(branches)-1,allow_unused=True)
                if not torch.isfinite(value) or any(g is not None and not torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite hint calibration gradient')
                for dest,grad in zip(gradients[key],grads):
                    if grad is not None:dest.add_(grad.detach())
                losses[key]+=float(value.detach())
    output={};rows={}
    for key in HINT_PATHS:
        connected=set(upstream_parameter_names(model,key));indices=[i for i,n in enumerate(names) if n in connected]
        old=[gradients['existing'][i] for i in indices];new=[gradients[key][i] for i in indices]
        a,b=_norm(old),_norm(new)
        if not math.isfinite(a+b) or a<=0 or b<=0:raise RuntimeError('Zero or nonfinite connected student calibration gradient')
        output[key]=fraction*a/b
        rows[key]={'connected_student_parameters':[names[i] for i in indices],
            'existing_gradient_norm':a,'hint_gradient_norm':b,'coefficient':output[key],
            'cosine_with_existing':_dot(old,new)/(a*b),'fraction':fraction}
    combined=[sum(output[k]*gradients[k][i] for k in HINT_PATHS) for i in range(len(params))]
    old=gradients['existing'];a,b=_norm(old),_norm(combined)
    return output,{'source_ids':[c['source_id'] for c in crops],'denominators':denom,'losses':losses,'hints':rows,
        'existing_all_student_gradient_norm':a,'combined_weighted_hint_all_student_gradient_norm':b,
        'combined_ratio_all_student':b/a,'combined_cosine_with_existing':_dot(old,combined)/(a*b) if a*b else None,
        'projector_gradients_excluded':True,'fraction_per_hint':fraction,'state_restored':True}
