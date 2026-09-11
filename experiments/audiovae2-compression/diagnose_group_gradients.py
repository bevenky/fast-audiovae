"""Fixed-case, no-update loss-gradient audit of the retained decoder group."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics

import torch

import resume_settings as resume

screen,base = resume.screen,resume.base
SOURCES = (
    "es_419:train:13461728374156750135.wav",
    "kn_in:train:15096009457771558023.wav",
    "kashmiri:1970324837177077_chunk_1.flac",
    "freesound:428921",
)
CORE = ("waveform","mel_linear","mel_log","feature")


def adam_displacement(gradient, first_moment, second_moment, step, learning_rate,
                      betas=(.9,.99), eps=1e-8):
    """Hypothetical AdamW direction, without changing weights or saved moments."""
    if not (gradient.shape==first_moment.shape==second_moment.shape):
        raise ValueError("Adam vectors must have equal shapes")
    if type(step) is not int or step<0 or (second_moment<0).any():
        raise ValueError("Invalid saved optimizer state")
    b1,b2=betas
    m=b1*first_moment+(1-b1)*gradient
    v=b2*second_moment+(1-b2)*gradient.square()
    return -learning_rate*(m/(1-b1**(step+1)))/(v.sqrt()/math.sqrt(1-b2**(step+1))+eps)


def dot(a,b): return float(torch.dot(a.double(),b.double()))


def vector_relations(grads,coefficient,adam=None,actual_delta=None,segments=None,zero_momentum_adam=None):
    total=sum(coefficient[k]*grads[k] for k in CORE)
    vectors={**grads,"weighted_total":total}
    norms={k:math.sqrt(dot(v,v)) for k,v in vectors.items()}
    cosines={a:{b:dot(x,y)/(norms[a]*norms[b]) if norms[a]*norms[b] else None
                for b,y in vectors.items()} for a,x in vectors.items()}
    result={"gradient_norms":norms,"pairwise_parameter_gradient_cosine":cosines,
        "weighted_core_gradient_norms":{k:coefficient[k]*norms[k] for k in CORE},
        "raw_gradient_descent_directional_derivatives":{k:-dot(v,total) for k,v in grads.items()}}
    if adam is not None:
        result["hypothetical_adam_directional_derivatives"]={k:dot(v,adam) for k,v in grads.items()}
        result["hypothetical_adam_displacement_norm"]=math.sqrt(dot(adam,adam))
    if zero_momentum_adam is not None:
        result["counterfactual_zero_first_moment_adam_directional_derivatives"]={k:dot(v,zero_momentum_adam) for k,v in grads.items()}
    if actual_delta is not None:
        result["actual_4500_to_5000_local_directional_derivatives"]={k:dot(v,actual_delta) for k,v in grads.items()}
        groups={}
        for label,intervals in (segments or {}).items():
            groups[label]={k:sum(dot(v[a:b],actual_delta[a:b]) for a,b in intervals) for k,v in grads.items()}
        result["actual_delta_local_derivatives_by_parameter_group"]=groups
    return result


class SpectralTap:
    def __init__(self,objective): self.objective=objective; self.config=objective.config; self.terms=[]
    def group_terms(self,*args):
        term=self.objective.group_terms(*args);self.terms.append(term);return term
    def aggregate(self,*args): return self.objective.aggregate(*args)


def source_losses(model,teacher,crop,spectral,denominators):
    z,target,valid,spans=base.batch([crop]);trace=base.teacher_forward(teacher,z)
    with torch.no_grad():
        for _ in range(3):
            model.suffix_from_group(model.group_from_input(trace["group_input"]))
    base.assert_close(trace["waveform"][valid],target[valid],"gradient audit sealed teacher target")
    h=model.group_from_input(trace["group_input"]);prediction=model.suffix_from_group(h)
    tap=SpectralTap(spectral)
    original=base.losses(prediction,target,h,trace["group_output"],valid,spans,tap,denominators)
    rec=base._reconstruction(tap.terms,spectral,denominators)
    branches={"waveform":original["waveform"],"mel_linear":spectral.config.mel_linear_weight*rec["teacher_mel_linear"],
              "mel_log":spectral.config.mel_log_weight*rec["teacher_mel_log"],"feature":original["feature"]}
    if not torch.equal(branches["mel_linear"]+branches["mel_log"],original["mel"]):
        raise RuntimeError("Split mel terms do not reproduce the unchanged objective")
    a,b=spans[0];p,t=prediction[...,a:b],target[...,a:b]
    quality=base.quiet_window_metrics(p,t,torch.ones_like(p,dtype=torch.bool))
    near=torch.zeros_like(p,dtype=torch.bool);active=torch.zeros_like(p,dtype=torch.bool)
    near_windows=[];amplitude=prediction.sum()*0
    for w in quality["windows"]:
        start,stop=w["start_sample"],w["stop_sample"]
        if not w["is_quiet"]:active[...,start:stop]=True
        if w["is_quiet"] and w["teacher_rms"]<=1e-5:
            near[...,start:stop]=True
            pw,tw=p[...,start:stop],t[...,start:stop]
            rms=pw.square().mean().sqrt()
            amplitude=amplitude+w["valid_samples"]*torch.relu(rms-w["output_rms_limit"]).square()/denominators["samples"]
            pd,td=pw.detach().double(),tw.detach().double();e=pd-td
            near_windows.append({"start_sample":start,"valid_samples":w["valid_samples"],
                "student_signed_mean":float(pd.mean()),"teacher_signed_mean":float(td.mean()),
                "residual_signed_mean":float(e.mean()),"student_rms":float(pd.square().mean().sqrt()),
                "teacher_rms":float(td.square().mean().sqrt()),"residual_rms":float(e.square().mean().sqrt()),
                "centered_residual_rms":float(((e-e.mean()).square().mean()).sqrt()),
                "output_rms_limit":w["output_rms_limit"],"passed":w["passed"]})
    branches["near_waveform"]=(p-t).abs().masked_fill(~near,0).sum()/denominators["samples"]
    branches["near_amplitude_violation"]=amplitude
    if active.any():
        pe=p.masked_select(active).square().sum();te=t.masked_select(active).square().sum()
        branches["active_log_gain"]=.5*(.5*torch.log(pe/te)).square()
        rms_gain=float((pe.detach()/te).sqrt())
    else:
        branches["active_log_gain"]=prediction.sum()*0;rms_gain=None
    count=sum(w["valid_samples"] for w in near_windows)
    near_summary={"samples":count,"windows":len(near_windows),"passes":sum(w["passed"] for w in near_windows)}
    for key in ("student_signed_mean","teacher_signed_mean","residual_signed_mean"):
        near_summary[key+"_sample_pooled"]=sum(w[key]*w["valid_samples"] for w in near_windows)/count if count else None
        near_summary[key+"_window_median"]=statistics.median(w[key] for w in near_windows) if count else None
    details={"source_id":crop["source_id"],"valid_samples":crop["valid_scored_samples"],
             "active_samples":int(active.sum()),"active_rms_gain":rms_gain,"near_summary":near_summary,
             "near_windows":near_windows,"mae":float((p.detach().double()-t.detach().double()).abs().mean()),
             "losses":{k:float(v.detach()) for k,v in branches.items()}}
    return branches,details


def flatten_state(state,names,device):
    return torch.cat([state[name].detach().reshape(-1).to(device) for name in names])


def optimizer_vectors(payload,named,device):
    group=payload["optimizer"]["param_groups"]
    if len(group)!=1:raise ValueError("Expected one AdamW parameter group")
    group=group[0]
    if (tuple(group["betas"])!=(.9,.99) or group["eps"]!=1e-8 or group["weight_decay"]!=0
            or group["lr"]!=payload["identity"]["learning_rate"] or len(group["params"])!=len(named)):
        raise ValueError("Saved optimizer recipe differs")
    first=[];second=[]
    for identity,(name,p) in zip(group["params"],named):
        state=payload["optimizer"]["state"][identity]
        if float(state["step"])!=payload["step"]:raise ValueError("Saved moment step differs")
        for key,dest in (("exp_avg",first),("exp_avg_sq",second)):
            if state[key].shape!=p.shape or not torch.isfinite(state[key]).all():raise ValueError("Invalid optimizer moments")
            dest.append(state[key].reshape(-1).to(device))
    return torch.cat(first),torch.cat(second)


def parameter_segments(named):
    result={};offset=0
    for name,p in named:
        stage="stage"+str(int(name.split('.')[1])-1)
        component="conditioning" if name.startswith("sr_cond_model.") else "snake" if name.endswith(".alpha") else "convolution"
        for label in (stage,component,stage+"/"+component):result.setdefault(label,[]).append((offset,offset+p.numel()))
        offset+=p.numel()
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ("checkpoint","anchor-checkpoint","screen-out","base-out","manifest","assets","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("Use a new diagnostic report path")
    base.policy()
    _,selection,identity,_,pools,latest,_,_,_=resume.authenticate(args)
    if latest["step"]!=5000 or latest["resume_identity"]["resume_sha256"]!=base.sha(resume.__file__):
        raise ValueError("Expected the unchanged completed5000-step candidate")
    paths={1000:args.anchor_checkpoint,4500:args.checkpoint.with_name("checkpoint-step4500.pt"),5000:args.checkpoint}
    hashes={s:base.sha(path) for s,path in paths.items()}
    states={s:torch.load(path,map_location="cpu",weights_only=True,mmap=True) for s,path in paths.items()}
    for step,state in states.items():
        receipt=json.loads(paths[step].with_suffix('.json').read_text())
        if (state["step"]!=step or state["identity"]!=latest["identity"] or receipt['checkpoint_sha256']!=hashes[step]):
            raise ValueError("Checkpoint identity or receipt changed")
    saved_quality={s:{r['source_id']:r for r in json.loads((path.parent/f'development-step{s}.json').read_text())['rows']} for s,path in paths.items()}
    by_id={c['source_id']:c for c in pools['development']};crops=[by_id[s] for s in SOURCES]
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    spectral=base.objective();denominators=base.reconstruction_denominators(crops,spectral)
    named=model.group_named_parameters();params=[p for _,p in named];names=[n for n,_ in named]
    coefficient={'waveform':1.,'mel_linear':latest['identity']['coefficients']['mel'],
                 'mel_log':latest['identity']['coefficients']['mel'],'feature':latest['identity']['coefficients']['feature']}
    if latest['identity']['coefficients']['waveform']!=1.:raise ValueError('Waveform coefficient changed')
    segments=parameter_segments(named)
    delta=flatten_state(states[5000]['group'],names,'cuda')-flatten_state(states[4500]['group'],names,'cuda')
    frozen=screen.frozen_versions(model);teacher_frozen=resume.continuation.teacher_versions(teacher)
    rng=screen.rng_state();original_grads=[p.grad for p in params];original_modes=[m.training for m in model.modules()]
    report={'source_ids':list(SOURCES),'denominators':denominators,'coefficients':coefficient,'checkpoint_sha256':hashes,
            'parameter_updates':0,'checkpoints':{},'implementation_sha256':base.sha(__file__),
            'interpretation':'Negative directional derivatives reduce the named loss locally; positive ones increase it. Raw gradient descent is not AdamW. Hypothetical AdamW uses this diagnostic panel plus saved moments, not an actual fitting batch. Per-source hypothetical updates use the single-source contribution with the common four-source denominator. The zero-first-moment control preserves the saved second moment but discards momentum. Actual4500to5000 delta dots are endpoint-local first-order projections, not causal attribution of the finite interval.',
            'diagnostic_only_losses':['near_waveform','near_amplitude_violation','active_log_gain'],
            'selection':'Fixed observed near-silence failures in two languages, the largest late amplitude/MAE contributor, and the only held-out whistle recording. No population generalization is claimed.'}
    try:
        for step,state in states.items():
            model.load_group_state_dict(state['group']);versions=resume.continuation.teacher_versions(model)
            m,v=optimizer_vectors(state,named,'cuda');pooled={};values={};details=[]
            for crop in crops:
                branches,detail=source_losses(model,teacher,crop,spectral,denominators)
                expected=saved_quality[step][crop['source_id']]['mae']
                if not math.isclose(detail['mae'],expected,rel_tol=1e-5,abs_tol=1e-6):raise RuntimeError('Gradient probe does not reproduce saved source MAE')
                detail['saved_mae']=expected;detail['saved_mae_reproduced']=True
                grads={}
                for index,(key,loss) in enumerate(branches.items()):
                    gradient=torch.autograd.grad(loss,params,retain_graph=index<len(branches)-1)
                    grads[key]=torch.cat([x.detach().reshape(-1) for x in gradient])
                    if not torch.isfinite(grads[key]).all():raise RuntimeError('Nonfinite gradient')
                    if key not in pooled:pooled[key]=torch.zeros_like(grads[key]);values[key]=0.
                    pooled[key].add_(grads[key]);values[key]+=detail['losses'][key]
                total=sum(coefficient[k]*grads[k] for k in CORE)
                displacement=adam_displacement(total,m,v,step,state['identity']['learning_rate'])
                zero_momentum=adam_displacement(total,torch.zeros_like(m),v,step,state['identity']['learning_rate'])
                detail['gradient_relations']=vector_relations(grads,coefficient,displacement,delta,segments,zero_momentum)
                details.append(detail);base.event('source_gradient_audit',step=step,source_id=crop['source_id'])
                del grads,branches,total,displacement,zero_momentum
            total=sum(coefficient[k]*pooled[k] for k in CORE)
            displacement=adam_displacement(total,m,v,step,state['identity']['learning_rate'])
            zero_momentum=adam_displacement(total,torch.zeros_like(m),v,step,state['identity']['learning_rate'])
            report['checkpoints'][step]={'per_source':details,'losses':values,
                'gradient_relations':vector_relations(pooled,coefficient,displacement,delta,segments,zero_momentum)}
            if versions!=resume.continuation.teacher_versions(model):raise RuntimeError('No-update probe changed model tensors')
            del pooled,m,v,total,displacement,zero_momentum
        if any(base.sha(path)!=hashes[s] for s,path in paths.items()):raise RuntimeError('Saved checkpoint bytes changed')
        if frozen!=screen.frozen_versions(model) or teacher_frozen!=resume.continuation.teacher_versions(teacher):
            raise RuntimeError('Frozen decoder or teacher changed')
        if any(p.grad is not g for p,g in zip(params,original_grads)) or [m.training for m in model.modules()]!=original_modes:
            raise RuntimeError('Parameter gradients or module modes changed')
        report['preservation']={'checkpoint_bytes':True,'frozen_decoder':True,'teacher':True,'parameter_grad_slots':True,'module_modes':True,'optimizer_state_read_only':True,'rng_restored':True}
    finally:
        screen.restore_rng(rng)
    base.write_json(args.out,report)
    base.event('gradient_audit_complete',path=str(args.out),parameter_updates=0)


if __name__=='__main__':main()
