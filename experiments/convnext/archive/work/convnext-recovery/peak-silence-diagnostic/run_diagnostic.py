"""Frozen current-checkpoint peak and silence diagnosis. No training steps.

Preserves the sealed canonical panel, excludes the historical six context
samples, and deduplicates waveform events by original source sample index.
Counterfactual template subtraction is diagnostic only, never a model edit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.fusion_architecture import FusionStudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha
from canonical_evaluation import load_canonical_cache

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
ROOT = Path('/tmp/fast-audiovae-recovery-20260909')
CHECKPOINTS = {
    'baseline': (BASE/'corrected-screen/targeted/final.pt', '0d39c801ad9c25af579460c46dd189d12a2a03bb8676283aa23ea145dd95f8e4'),
    'quarter': (BASE/'corrected-update-400-reference-v1/quarter_rate/final.pt', 'f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948'),
}


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def status(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw)), flush=True)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None


def phase_components(cycles):
    """Orthogonal DC, 480, additional1920, and nonperiodic decomposition."""
    x = np.asarray(cycles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1920 or not len(x):
        raise ValueError('Expected nonempty complete1920-sample cycles')
    template = x.mean(0)
    dc = float(template.mean())
    p480 = template.reshape(4, 480).mean(0)-dc
    p1920 = template-dc-np.tile(p480, 4)
    terms = {'dc': dc*dc, 'period480': float(np.mean(p480*p480)),
             'additional_period1920': float(np.mean(p1920*p1920)),
             'remaining': float(np.mean((x-template)**2))}
    total = float(np.mean(x*x))
    if not math.isclose(sum(terms.values()), total, rel_tol=1e-9, abs_tol=1e-22):
        raise ValueError('Phase energy decomposition failed')
    return {'cycles': len(x), 'samples': x.size, 'rms': math.sqrt(total),
            'power': total, 'component_power': terms,
            'component_share': {k: v/total if total else 0 for k, v in terms.items()}}


def contiguous(indices):
    return [g for g in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1) if len(g)]


def event_rows(source, observations):
    rows = []
    keys = sorted(observations)
    for group in contiguous(np.array(keys, dtype=np.int64)):
        peak = max(group, key=lambda i: abs(observations[int(i)]['quarter']))
        value = dict(observations[int(peak)])
        value['baseline_abs_max_within_quarter_event']=max(abs(observations[int(i)]['baseline_at_quarter_peak']) for i in group)
        rows.append(dict(source_id=source, start_sample=int(group[0]),
            end_sample_exclusive=int(group[-1]+1), samples=len(group),
            duration_ms=len(group)/48, peak_sample=int(peak), time_seconds=int(peak)/48000,
            phase480=int(peak%480), distance_to_480_boundary=int(min(peak%480, 480-peak%480)),
            **value))
    return rows


def local_event_info(p, t, index):
    """Teacher neighborhood and lag check, not automatic signal alignment."""
    lo, hi = max(0,index-240), min(len(p),index+241)
    a, b = p[lo:hi], t[lo:hi]
    best = (cosine(a,b), 0)
    for lag in range(-48,49):
        if lo+lag<0 or hi+lag>len(t):
            continue
        score = cosine(a,t[lo+lag:hi+lag])
        if score is not None and (best[0] is None or score>best[0]):
            best=(score,lag)
    return {'teacher_at_sample':float(t[index]),
            'teacher_abs_max_within1ms':float(np.max(np.abs(t[max(0,index-48):index+49]))),
            'teacher_abs_max_within5ms':float(np.max(np.abs(b))),
            'local_student_rms':rms(a), 'local_teacher_rms':rms(b),
            'local_cosine':cosine(a,b), 'best_local_cosine_within1ms_lag':best[0],
            'best_lag_samples':best[1]}


def quiet_sample_mask(t, start, stop):
    mask=np.zeros(len(t),dtype=bool)
    for first in range(0,len(t),960):
        lo,hi=max(start,first),min(stop,first+960)
        if hi>lo and rms(t[lo:hi])<=.001:
            mask[lo:hi]=True
    return mask


def selftest():
    t=np.arange(1920)
    cycle=.3+.2*np.sin(2*np.pi*t/480)+.1*np.cos(2*np.pi*t/1920)
    result=phase_components(np.tile(cycle,(8,1)))
    assert abs(result['component_power']['dc']-.09)<1e-12
    assert abs(result['component_power']['period480']-.02)<1e-12
    assert abs(result['component_power']['additional_period1920']-.005)<1e-12
    assert [len(g) for g in contiguous(np.array([2,3,7,9,10]))]==[2,1,2]
    t=np.zeros(2000);t[1000:]=.02
    m=quiet_sample_mask(t,6,1900)
    assert m.sum()==954 and not m[:6].any() and not m[960:].any()
    print('phase, contiguous-event, and masked-window fixtures passed')


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    receipt_path=ROOT/'canonical-panel-v1/receipt.json'
    crops,metadata,receipt=load_canonical_cache(receipt_path)
    models,states,hashes={},{},{}
    for name,(path,expected) in CHECKPOINTS.items():
        if file_sha(path)!=expected:raise ValueError('Checkpoint mismatch '+name)
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['engine']
        model=FusionStudentDecoder.from_decoder(StudentDecoder(StudentConfig(**state['model_config'])))
        model.load_state_dict(state['model'],strict=True)
        model=model.cuda().eval().requires_grad_(False)
        if state_fingerprint(model.state_dict())!=state_fingerprint(state['model']):
            raise ValueError('Model restore differs')
        models[name],states[name],hashes[name]=model,state,state_fingerprint(model.state_dict())
    identity={'scope':'Frozen canonical-panel inference and disposable waveform gradients; zero optimizer steps',
              'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),
              'checkpoint_sha256':{k:v[1] for k,v in CHECKPOINTS.items()},
              'source_sha256':file_sha(__file__),'canonical_cache_sha256':receipt['sha256'],
              'crops':len(crops),'model_mode':'eval FP32, TF32 off; GPU diagnostic, not RTF',
              'mask':'Original scored slice with six samples excluded only for interior cached context',
              'phase_scope':'Complete teacher-quiet40ms cycles, deduplicated; distinct from20ms acceptance panel',
              'no_new_training':True}
    save(out/'identity.json',identity)
    by_source=defaultdict(dict);cycle_bank=defaultdict(dict);fixtures={};rows=[]
    baseline_bad=defaultdict(set)
    selected_grad={};observation_count=0;overlap_diff=0.;total_samples=0;quiet_windows=0
    metrics={n:{'quiet_error_sum':0.,'quiet_student_energy':0.,'quiet_teacher_energy':0.,
                'quiet_cross_product':0.,'quiet_samples':0,'max_peak':0.,'overshoot_observations':0} for n in models}
    pattern_counter={n:{'quiet_before':0.,'quiet_after':0.,'quiet_samples':0,
                        'active_before':0.,'active_after':0.,'active_samples':0} for n in models}
    # Compute a fixed zero-input residual template before natural inference.
    zero=next(c for c in crops if c.source_id=='encoded_zero')
    templates={}
    with torch.inference_mode():
        for name,model in models.items():
            p=model(zero.latents.cuda()).cpu().flatten().numpy()
            t=zero.teacher_audio.flatten().numpy()
            cycles=(p-t)[96000:288000].reshape(-1,1920)
            templates[name]=cycles.mean(0)
            fixtures[name]={'encoded_zero_steady':phase_components(cycles),
                'teacher_rms':rms(t[96000:]),'student_rms':rms(p[96000:]),
                'latent_rms':float(zero.latents.square().mean().sqrt()),
                'latent_steady_temporal_variation_rms':float((zero.latents[...,50:]-zero.latents[...,50:].mean(-1,keepdim=True)).square().mean().sqrt())}
    for number,c in enumerate(crops):
        with torch.inference_mode():
            predictions={n:m(c.latents.cuda()).cpu().flatten().numpy() for n,m in models.items()}
        t=c.teacher_audio.flatten().numpy()
        start=c.scored_slice.start+(6 if c.context_start_frame>0 else 0)
        stop=c.scored_slice.stop
        offset=c.context_start_frame*1920
        qmask=quiet_sample_mask(t,start,stop)
        natural=not c.source_id.startswith('encoded_')
        p=predictions['quarter']
        bad=np.flatnonzero(np.abs(p[start:stop])>1)+start
        if natural:
            total_samples+=stop-start;observation_count+=len(bad)
            quiet_windows+=sum(bool(qmask[first:first+960].any()) for first in range(0,len(t),960))
            baseline_bad[c.source_id].update(int(i+offset) for i in np.flatnonzero(np.abs(predictions['baseline'][start:stop])>1)+start)
        for n,v in predictions.items():
            if not np.isfinite(v).all() or len(v)!=len(t):raise ValueError('Invalid prediction')
            if natural:
                r=v.astype(np.float64)-t
                metrics[n]['quiet_error_sum']+=float(np.sum(r[qmask]**2))
                metrics[n]['quiet_student_energy']+=float(np.sum(v[qmask].astype(np.float64)**2))
                metrics[n]['quiet_teacher_energy']+=float(np.sum(t[qmask].astype(np.float64)**2))
                metrics[n]['quiet_cross_product']+=float(np.sum(v[qmask].astype(np.float64)*t[qmask]))
                metrics[n]['quiet_samples']+=int(qmask.sum())
                metrics[n]['max_peak']=max(metrics[n]['max_peak'],float(abs(v[start:stop]).max()))
                metrics[n]['overshoot_observations']+=int((abs(v[start:stop])>1).sum())
                fixed=templates[n][(np.arange(len(v))+offset)%1920]
                corrected=r-fixed
                for region,mask in [('quiet',qmask),('active',~qmask & (np.arange(len(v))>=start)&(np.arange(len(v))<stop))]:
                    pattern_counter[n][region+'_before']+=float(np.sum(r[mask]**2))
                    pattern_counter[n][region+'_after']+=float(np.sum(corrected[mask]**2))
                    pattern_counter[n][region+'_samples']+=int(mask.sum())
        for i in bad:
            if not natural:continue
            absolute=int(i+offset)
            value={'quarter':float(p[i]),'baseline_at_quarter_peak':float(predictions['baseline'][i]),
                   'crop_start_frame':c.start_frame, **local_event_info(p,t,int(i))}
            existing=by_source[c.source_id].get(absolute)
            if existing is not None:
                overlap_diff=max(overlap_diff,abs(existing['quarter']-value['quarter']))
            else:
                by_source[c.source_id][absolute]=value
        if natural:
            for first in range(((start+1919)//1920)*1920,stop-1919,1920):
                teacher=t[first:first+1920]
                # Require each of the two20ms halves to meet the sealed quiet threshold.
                if max(rms(teacher[:960]),rms(teacher[960:]))>.001:continue
                index=(first+offset)//1920
                if index not in cycle_bank[c.source_id]:
                    cycle_bank[c.source_id][index]={n:(v[first:first+1920]-teacher).astype(np.float64) for n,v in predictions.items()}
        row={'source_id':c.source_id,'start_frame':c.start_frame,'samples':stop-start,
             'quarter_peak':float(abs(p[start:stop]).max()),'teacher_peak':float(abs(t[start:stop]).max()),
             'overshoot_observations':len(bad),'quiet_samples':int(qmask.sum()),
             'student_rms':rms(p[start:stop]),'teacher_rms':rms(t[start:stop])}
        rows.append(row)
        # One peak crop/source, encoded-zero, and the three known quiet regressions.
        regression=any(s in c.source_id for s in ['b82da8bb-ec78-4b4e-b953-4cb86642f27a','10172384069529710044','freesound:220649'])
        if len(bad) or c.source_id=='encoded_zero' or regression:
            score=float(abs(p[start:stop]).max()) if len(bad) else rms((p-t)[qmask]) if qmask.any() else 0.
            if c.source_id not in selected_grad or score>selected_grad[c.source_id][0]:
                selected_grad[c.source_id]=(score,c,torch.from_numpy(p.copy()).reshape(1,1,-1))
        if number%50==0:status('canonical_frozen_inference',completed=number+1,total=len(crops))
    if total_samples!=25342830 or observation_count!=588:raise ValueError('Current checkpoint panel did not reproduce')
    if quiet_windows!=3434 or metrics['quarter']['quiet_samples']!=3291645:
        raise ValueError('Teacher-quiet mask differs from sealed canonical counts')
    if overlap_diff>2e-6:raise ValueError('Overlapping predictions disagree above parity tolerance')
    events=[]
    for sid,obs in by_source.items():events.extend(event_rows(sid,obs))
    phase={n:[] for n in models}
    for sid,items in cycle_bank.items():
        if len(items)<8:continue
        for name in models:
            r=phase_components([items[i][name] for i in sorted(items)])
            r['source_id']=sid;phase[name].append(r)
    pooled={}
    for name,parts in phase.items():
        samples=sum(v['samples'] for v in parts)
        powers={k:sum(v['samples']*v['component_power'][k] for v in parts)/samples
                for k in ['dc','period480','additional_period1920','remaining']}
        power=sum(powers.values())
        pooled[name]={'sources':len(parts),'samples':samples,'rms':math.sqrt(power),
                      'component_power':powers,'component_share':{k:v/power for k,v in powers.items()}}
    for value in metrics.values():
        for kind in ['student','teacher']:
            value['quiet_'+kind+'_rms']=math.sqrt(value['quiet_'+kind+'_energy']/value['quiet_samples'])
        value['quiet_residual_rms']=math.sqrt(value['quiet_error_sum']/value['quiet_samples'])
        value['quiet_cosine']=value['quiet_cross_product']/math.sqrt(value['quiet_student_energy']*value['quiet_teacher_energy'])
    for values in pattern_counter.values():
        for region in ['quiet','active']:
            values[region+'_mse_change_percent']=(values[region+'_after']/values[region+'_before']-1)*100
    result={'identity':identity,'metrics':metrics,'fixtures':fixtures,'peak_events':events,
            'baseline_unique_overshoot_samples':sum(len(v) for v in baseline_bad.values()),
            'new_quarter_overshoot_samples':sum(len(set(v)-baseline_bad[s]) for s,v in by_source.items()),
            'resolved_baseline_overshoot_samples':sum(len(v-set(by_source[s])) for s,v in baseline_bad.items()),
            'unique_overshoot_samples':sum(len(v) for v in by_source.values()),'contiguous_events':len(events),
            'overshoot_observations':observation_count,'overlapping_prediction_max_difference':overlap_diff,
            'events_within6_samples_of_480_boundary':sum(e['distance_to_480_boundary']<=6 for e in events),
            'phase_decomposition':pooled,'phase_by_source':phase,
            'fixed_encoded_zero_template_counterfactual':pattern_counter,'rows':rows,
            'counterfactual_caution':'Template subtraction is a frozen diagnostic, not a proposed deployed correction or heldout-fitted model'}
    save(out/'waveform-diagnosis.json',result)
    status('waveform_diagnosis_complete',events=len(events),unique_samples=result['unique_overshoot_samples'])
    from gradient_probe import FrozenWaveformGradientProbe
    probe=FrozenWaveformGradientProbe(states['quarter'],device='cuda')
    gradients=[]
    for i,(_,c,p) in enumerate(selected_grad.values()):
        status('frozen_waveform_gradients',completed=i,total=len(selected_grad),source_id=c.source_id)
        gradients.append(probe.probe(c,p,include_first_view=True))
        save(out/'gradient-diagnosis.json',gradients)
    integrity={n:state_fingerprint(m.state_dict())==hashes[n] and file_sha(CHECKPOINTS[n][0])==CHECKPOINTS[n][1] for n,m in models.items()}
    if not all(integrity.values()):raise ValueError('Immutable state changed')
    save(out/'complete.json',{'no_training_steps':True,'preserved_checkpoints_and_models':integrity,
                             'gradient_probes':len(gradients),'source_sha256':file_sha(__file__)})
    status('complete',out=str(out))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--out')
    args=parser.parse_args()
    if args.selftest:selftest()
    else:
        if not args.out:parser.error('--out required')
        lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
        with lock.open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.out)
