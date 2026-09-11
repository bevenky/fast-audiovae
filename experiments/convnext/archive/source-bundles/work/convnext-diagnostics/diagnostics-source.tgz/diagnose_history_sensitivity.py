"""Matched teacher/student history and real-input amplitude diagnostics.

Read-only frozen-model evaluation. Raw waveform errors are primary. Bounded lag
and gain fits are secondary diagnostics and never alter targets, quality scores,
checkpoint tensors or audio. Longest available context is not claimed to be a
whole-recording reference. Amplitude probes re-encode the same valid input window.
"""
from __future__ import annotations
import argparse,hashlib,math
import numpy as np

HOP=1920
PREFIXES=(0,8,16,29,30,40)
GAINS=(1.,.5,.1,.01,0.)

def array(x):
    if hasattr(x,'detach'):x=x.detach().float().cpu().numpy()
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not x.size or not np.isfinite(x).all():raise ValueError('Nonempty finite audio required')
    return x

def raw_metrics(prediction,target):
    p,t=array(prediction),array(target)
    if p.shape!=t.shape:raise ValueError('Matched valid waveform lengths required')
    e=p-t;ep=float(np.mean(e*e));tp=float(np.mean(t*t));pp=float(np.mean(p*p))
    return {'samples':p.size,'mae':float(np.mean(np.abs(e))),'mse':ep,'residual_rms':math.sqrt(ep),
        'max_abs_error':float(np.max(np.abs(e))),'prediction_rms':math.sqrt(pp),'target_rms':math.sqrt(tp),
        'prediction_peak':float(np.max(np.abs(p))),'target_peak':float(np.max(np.abs(t))),
        'residual_dc':float(np.mean(e)),'cosine':float(np.sum(p*t)/math.sqrt(np.sum(p*p)*np.sum(t*t))) if pp and tp else None,
        'snr_db':10*math.log10(tp/ep) if tp>0 and ep>0 else None,
        'equal_values':bool(np.array_equal(p,t))}

def lag_gain_diagnostic(prediction,target,max_lag=48,gain_bounds=(.8,1.25)):
    """All candidates use one common interior target grid, no wrap or padding.

    Positive lag compares p[t+lag] against target[t]. Gain multiplies prediction.
    Exact ties prefer zero lag, then lower absolute lag. No corrected audio is
    exported and raw error on this same interior is reported alongside the fit.
    """
    p,t=array(prediction),array(target)
    if p.shape!=t.shape or type(max_lag) is not int or max_lag<0:raise ValueError('Invalid lag geometry')
    lo,hi=gain_bounds
    if not 0<lo<=1<=hi or not np.isfinite([lo,hi]).all():raise ValueError('Gain bounds must include1')
    margin=min(max_lag,max(0,(len(p)-8)//2));stop=len(p)-margin;ref=t[margin:stop]
    baseline=raw_metrics(p[margin:stop],ref)
    # One compiled valid correlation evaluates all97 bounded lags on one grid.
    dots=np.correlate(p,ref,mode='valid')
    energy=np.concatenate(([0.],np.cumsum(p*p)))
    starts=np.arange(2*margin+1);den=energy[starts+len(ref)]-energy[starts]
    gains=np.ones_like(den);np.divide(dots,den,out=gains,where=den>0);gains=np.clip(gains,lo,hi)
    estimated=(gains*gains*den-2*gains*dots+float(np.sum(ref*ref)))/len(ref)
    order=sorted(range(2*margin+1),key=lambda i:(float(estimated[i]),abs(i-margin),i-margin))
    # Recompute the selected fit directly, avoiding cancellation in the report.
    i=order[0];lag=i-margin;gain=float(gains[i]);values=p[i:i+len(ref)]
    mse=float(np.mean((gain*values-ref)**2))
    best={'lag_samples':lag,'lag_seconds':lag/48000.,'prediction_gain':gain,'mse':mse,'residual_rms':math.sqrt(mse)}
    return {'raw_common_grid':baseline,'fit':best,'common_samples':len(ref),'excluded_each_end_samples':margin,
        'gain_bounds':list(gain_bounds),'diagnostic_only':True,'lag_convention':'compare prediction[t+lag] to target[t]',
        'does_not_replace_raw_primary':True}

def phase_variance(prediction,target,period=480):
    p,t=array(prediction),array(target)
    if p.shape!=t.shape or type(period)is not int or period<=0:raise ValueError('Invalid phase geometry')
    cycles=len(p)//period
    if cycles<2:return {'period_samples':period,'cycles':cycles,'available':False}
    e=(p-t)[:cycles*period].reshape(cycles,period);template=e.mean(0);power=float(np.mean(e*e));dc=float(e.mean())
    ac=float(np.mean((template-dc)**2))
    return {'period_samples':period,'cycles':cycles,'available':True,'discarded_tail_samples':len(p)%period,
        'residual_dc':dc,'phase_mean_variance':ac,'phase_template_ac_fraction_of_residual_power':ac/power if power else 0.,
        'phase_rms_min':float(np.sqrt(np.mean(e*e,axis=0)).min()),'phase_rms_max':float(np.sqrt(np.mean(e*e,axis=0)).max()),
        'interpretation':'Descriptive grid dependence, not proof of a tone or architectural cause'}

def metrics(prediction,target,fit=True):
    out={'raw_primary':raw_metrics(prediction,target),'phase480':phase_variance(prediction,target,480),'phase1920':phase_variance(prediction,target,1920)}
    if fit:out['secondary_lag_gain']=lag_gain_diagnostic(prediction,target)
    return out

def slope(xs,ys,log_x=False,log_y=False):
    values=[(float(x),float(y)) for x,y in zip(xs,ys) if np.isfinite(x) and np.isfinite(y) and (not log_x or x>0) and (not log_y or y>0)]
    if len(values)<2:return None
    x,y=np.asarray(values).T
    if log_x:x=np.log(x)
    if log_y:y=np.log(y)
    x=x-x.mean();den=float(np.sum(x*x))
    return float(np.sum(x*(y-y.mean()))/den) if den else None

def category(info,crop):
    if crop.source_id.startswith('encoded_') or info.get('synthetic') is True:return 'encoded_control'
    text=str(info.get('condition','')).lower()
    if 'laugh' in text:return 'laughter'
    if 'whistl' in text:return 'whistle'
    if any(s in text for s in ('quiet','silence','whisper','breath')):return 'quiet_or_low_level'
    if text=='speech':return 'speech'
    return 'other'

def select_cases(ctx,maximum=8):
    if type(maximum)is not int or not 1<=maximum<=8:raise ValueError('Bounded maximum1..8 required')
    candidates={}
    for c in [*ctx.heldout,*ctx.probe_crops]:
        valid=c.context_frames*HOP+c.valid_scored_samples;frames=min(valid//HOP,c.latents.shape[-1])
        if frames<72 or c.reference16k is None:continue
        info=ctx.metadata.get(c.source_id,{})
        # Prefer a true utterance start, then the longest real available history.
        key=(c.context_start_frame!=0,-frames,c.start_frame)
        if c.source_id not in candidates or key<candidates[c.source_id][0]:candidates[c.source_id]=(key,c,info,frames)
    groups={}
    for _,c,info,frames in sorted(candidates.values(),key=lambda v:(v[0],v[1].source_id)):
        groups.setdefault(category(info,c),[]).append((c,info,frames))
    chosen=[];order=('speech','laughter','whistle','quiet_or_low_level','encoded_control','other')
    for cycle in range(2):
        for kind in order[:4]:
            if len(chosen)>=maximum:break
            if len(groups.get(kind,[]))>cycle:chosen.append((kind,*groups[kind][cycle]))
    for kind in order[4:]:
        for item in groups.get(kind,[]):
            if len(chosen)>=maximum:break
            chosen.append((kind,*item))
    return chosen,{'eligible_source_counts':{k:len(v) for k,v in groups.items()},'selected_categories':[x[0] for x in chosen],
        'missing_natural_categories':[k for k in order[:4] if k not in groups],
        'selection':'Source-diverse; prefer true-start windows; >=40 real latent history frames before the same last32 complete scored frames; encoded controls explicitly labeled'}

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def checked_decode(fn,z,device):
    import torch
    original=tensor_hash(z);value=z.to(device=device,dtype=torch.float32)
    before=tensor_hash(value);y=fn(value)
    if y.shape!=(1,1,z.shape[-1]*HOP) or y.dtype!=torch.float32 or not bool(torch.isfinite(y).all()):raise ValueError('Decoder shape/dtype/finite contract failed')
    if before!=tensor_hash(value) or original!=tensor_hash(z):raise RuntimeError('Diagnostic decoder mutated its input')
    return y.detach().cpu()

def future_probe(fn,z,device,anchor):
    import torch
    cut=anchor+8;altered=z.clone();delta=torch.sin(torch.arange(1,z.shape[1]+1,dtype=z.dtype)).reshape(1,-1,1)*.125
    altered[...,cut:]+=delta
    a=checked_decode(fn,z,device)[...,anchor*HOP:cut*HOP];b=checked_decode(fn,altered,device)[...,anchor*HOP:cut*HOP]
    m=raw_metrics(b,a)
    return {'changed_latent_start_frame':cut,'scored_prefix_start_frame':anchor,'prefix_samples':a.numel(),
        'raw_difference':m,'bitwise_equal':bool(torch.equal(a.view(torch.int32),b.view(torch.int32))),
        'within_1e_5_absolute_1e_4_relative':bool(torch.allclose(a,b,atol=1e-5,rtol=1e-4)),
        'meaning':'Empirical future independence on this fixed input, not an analytical receptive-field proof'}

def run(ctx,names=('parent','targeted','complex'),maximum=8):
    import torch
    from audiovae_student.fusion_evaluation import _preserved_evaluation
    from diagnostic_common import atomic_json,status
    selected,selection=select_cases(ctx,maximum)
    if not selected:raise ValueError('No valid source has40history+32score frames')
    result={'format_version':1,'scope':__doc__,'selection':selection,'prefix_frames':list(PREFIXES),'gain_factors':list(GAINS),
        'teacher':[],'checkpoints':{},'parameter_updates':0,'optimizer_updates':0,'primary_scores_or_audio_changed':False,
        'device_scope':'Frozen teacher/student diagnostics, not an inference-speed benchmark',
        'diagnostic_score_grid':'Last32 complete valid frames; includes the first samples of that interior score so boundary effects remain visible. Historical panel scores and their six-sample exclusion are untouched.'}
    cases=[];teacher=ctx.teacher()
    with _preserved_evaluation(teacher.model):
        for kind,c,info,frames in selected:
            status('history_teacher',source=c.source_id)
            z=c.latents[...,:frames].contiguous().clone();anchor=frames-32;score=slice(anchor*HOP,frames*HOP)
            y=checked_decode(teacher.decode,z,teacher.device);repeated=checked_decode(teacher.decode,z,teacher.device)
            target=y[...,score];history={}
            for prefix in PREFIXES:
                reset=checked_decode(teacher.decode,z[...,anchor-prefix:].contiguous(),teacher.device)
                history[prefix]=reset[...,prefix*HOP:(prefix+32)*HOP]
            audio,source=ctx.source_audio(c);audio=audio.float().contiguous();valid=source['valid_input_samples']
            if audio.shape!=(1,1,valid) or not bool(torch.isfinite(audio).all()):raise ValueError('Invalid supplied valid input waveform')
            amplitude=[]
            for factor in GAINS:
                raw=audio*factor;original=tensor_hash(raw)
                device_raw=raw.to(teacher.device);device_raw_sha=tensor_hash(device_raw)
                az=teacher.encode(device_raw).detach().cpu()
                if tensor_hash(device_raw)!=device_raw_sha:raise RuntimeError('Encoder mutated its input')
                ay=checked_decode(teacher.decode,az,teacher.device)
                if tensor_hash(raw)!=original:raise RuntimeError('Amplitude probe mutated source input')
                if frames*HOP>valid*3:raise ValueError('Amplitude score includes right padding')
                amplitude.append({'gain':factor,'z':az,'teacher':ay[...,score],'raw_input_sha256':original,
                    'encoded_z_sha256':tensor_hash(az),'input_rms':float(raw.square().mean().sqrt())})
            baseline=amplitude[0]['teacher'];baseline_again_z=teacher.encode(audio.to(teacher.device)).detach().cpu()
            baseline_again=checked_decode(teacher.decode,baseline_again_z,teacher.device)[...,score]
            tr={'source_id':c.source_id,'start_frame':c.start_frame,'category':kind,'metadata':info,'input_scope':source,
                'longest_reference_frames':frames,'available_history_frames':anchor,'absolute_score_start_frame':c.context_start_frame+anchor,
                'scored_frames':32,'latent_sha256':tensor_hash(z),'reference_is_complete_original_recording':False,
                'history_reference_kind':'true-start partial utterance' if source['true_start'] else 'truncated supplied context window',
                'teacher_repeat':raw_metrics(repeated,y),'teacher_repeat_bitwise':bool(torch.equal(repeated.view(torch.int32),y.view(torch.int32))),
                'teacher_future':future_probe(teacher.decode,z,teacher.device,anchor),
                'history':[{'prefix_frames':p,'teacher_reset_vs_longest_history':metrics(history[p],target,fit=False)} for p in PREFIXES],
                'amplitude':[{'gain':a['gain'],'input_rms':a['input_rms'],'raw_input_sha256':a['raw_input_sha256'],
                    'encoded_z_sha256':a['encoded_z_sha256'],'teacher_rms':float(a['teacher'].square().mean().sqrt()),
                    'teacher_homogeneity_error':raw_metrics(a['teacher'],baseline*a['gain'])} for a in amplitude],
                'amplitude_gain1_reencode_repeat':raw_metrics(baseline_again,baseline)}
            if source['true_start']:
                tr['cached_teacher_vs_redecoded_z']=raw_metrics(target,c.teacher_audio[...,score])
                tr['gain1_reencoded_teacher_vs_cached_target']=raw_metrics(baseline,c.teacher_audio[...,score])
            tr['teacher_history_log_error_slope_per_frame']=slope(PREFIXES,[r['teacher_reset_vs_longest_history']['raw_primary']['residual_rms'] for r in tr['history']],log_y=True)
            tr['teacher_log_rms_vs_log_gain_slope']=slope([a['gain'] for a in amplitude],[r['teacher_rms'] for r in tr['amplitude']],True,True)
            result['teacher'].append(tr);cases.append((c,z,anchor,target,history,amplitude))
    del teacher;torch.cuda.empty_cache()
    atomic_json(ctx.out/'history-sensitivity-teacher.json',result)
    for name in names:
        engine=ctx.engine(name,device='cuda');rows=[]
        with _preserved_evaluation(engine.model):
            for c,z,anchor,target,teacher_hist,amplitude in cases:
                status('history_student',checkpoint=name,source=c.source_id)
                full=checked_decode(engine.model,z,engine.device)[...,anchor*HOP:];hist=[]
                for prefix in PREFIXES:
                    pred=checked_decode(engine.model,z[...,anchor-prefix:].contiguous(),engine.device)[...,prefix*HOP:(prefix+32)*HOP]
                    hist.append({'prefix_frames':prefix,'student_reset_vs_longest_history':raw_metrics(pred,full),
                        'same_history_student_vs_teacher':metrics(pred,teacher_hist[prefix],fit=False),
                        'student_reset_vs_teacher_longest_history':raw_metrics(pred,target)})
                amplitude_predictions=[checked_decode(engine.model,a['z'],engine.device)[...,anchor*HOP:(anchor+32)*HOP] for a in amplitude]
                amp=[]
                for a,pred in zip(amplitude,amplitude_predictions):
                    amp.append({'gain':a['gain'],'input_rms':a['input_rms'],'student_vs_reencoded_teacher':metrics(pred,a['teacher'],fit=a['gain']==1.),
                        'student_homogeneity_error':raw_metrics(pred,amplitude_predictions[0]*a['gain'])})
                secants=[]
                for i in range(len(amplitude)-1):
                    da=amplitude[i]['gain']-amplitude[i+1]['gain']
                    ps=(amplitude_predictions[i]-amplitude_predictions[i+1])/da
                    ts=(amplitude[i]['teacher']-amplitude[i+1]['teacher'])/da
                    secants.append({'gains':[amplitude[i]['gain'],amplitude[i+1]['gain']],'student_vs_teacher_secant':raw_metrics(ps,ts),
                        'interpretation':'Finite amplitude secant through full re-encoding, not a local JVP'})
                rows.append({'source_id':c.source_id,'start_frame':c.start_frame,'longest_history_student_vs_teacher':metrics(full,target),
                    'history':hist,'history_log_error_slope_per_frame':slope(PREFIXES,[r['student_reset_vs_longest_history']['residual_rms'] for r in hist],log_y=True),
                    'future':future_probe(engine.model,z,engine.device,anchor),'amplitude':amp,'amplitude_secants':secants,
                    'student_log_rms_vs_log_gain_slope':slope(GAINS,[r['student_vs_reencoded_teacher']['raw_primary']['prediction_rms'] for r in amp],True,True),
                    'teacher_log_rms_vs_log_gain_slope':slope(GAINS,[r['student_vs_reencoded_teacher']['raw_primary']['target_rms'] for r in amp],True,True)})
        entry={'checkpoint_sha256':ctx.expected_hashes[name],'step':engine.step,'rows':rows,'model_state_unchanged':True,'parameter_updates':0,'optimizer_updates':0}
        result['checkpoints'][name]=entry;atomic_json(ctx.out/f'history-sensitivity-{name}.json',entry)
        del engine;torch.cuda.empty_cache()
    result['file_verification']=ctx.verify_files();atomic_json(ctx.out/'history-sensitivity.json',result);return result

if __name__=='__main__':
    from diagnostic_common import load_context
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',default=['parent','targeted','complex']);p.add_argument('--maximum',type=int,default=8)
    a=p.parse_args();run(load_context(),tuple(a.checkpoints),a.maximum)
