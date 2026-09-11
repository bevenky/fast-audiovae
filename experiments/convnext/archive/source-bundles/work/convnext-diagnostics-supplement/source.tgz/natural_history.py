"""Natural-audio supplement for frozen teacher/student history diagnostics.

This separate panel uses16 complete scored frames (640ms) after at least40
real latent history frames. It closes the earlier panel's natural-source
coverage gap without editing its source or results. No synthetic replacement,
latent padding, target rescaling or primary quality-mask changes are permitted.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,math
from pathlib import Path
import numpy as np

MAIN_SHA256='6253d3062fada4e9d29d94af6ab207c45347576fab81f18bf84efafc8285999e'
_spec=importlib.util.find_spec('diagnose_history_sensitivity')
if _spec is None or _spec.origin is None:raise ImportError('Place the frozen main diagnostics directory on PYTHONPATH')
MAIN_PATH=Path(_spec.origin)
if hashlib.sha256(MAIN_PATH.read_bytes()).hexdigest()!=MAIN_SHA256:raise RuntimeError('Frozen main helper source differs')
from diagnose_history_sensitivity import HOP,PREFIXES,GAINS,array,raw_metrics,metrics,slope,tensor_hash,checked_decode,future_probe

SCORE_FRAMES=16
NATURAL_CATEGORIES=('speech','laughter','whistle','quiet_or_low_level')

def is_synthetic(crop,info):
    text=' '.join(str(info.get(k,'')) for k in ('condition','category','kind','source_type')).lower()
    return crop.source_id.startswith(('encoded_','synthetic_','fixture_')) or info.get('synthetic') is True or any(v in text for v in ('synthetic','digital_zero','encoded_control','test_fixture'))

def natural_category(crop,info):
    if is_synthetic(crop,info):return None
    text=str(info.get('condition','')).lower()
    if 'laugh' in text:return 'laughter'
    if 'whistl' in text:return 'whistle'
    if any(v in text for v in ('quiet','silence','whisper','breath')):return 'quiet_or_low_level'
    if text=='speech':return 'speech'
    return 'other_natural'

def select_cases(ctx,maximum=8):
    if type(maximum)is not int or not 1<=maximum<=8:raise ValueError('Maximum must be1..8')
    by_source={};excluded={'synthetic':0,'missing_reference16k':0,'insufficient_real_history':0}
    seen=set()
    for c in [*ctx.heldout,*ctx.probe_crops]:
        key=(c.source_id,c.start_frame,c.context_start_frame)
        if key in seen:continue
        seen.add(key);info=ctx.metadata.get(c.source_id,{})
        if is_synthetic(c,info):excluded['synthetic']+=1;continue
        if c.reference16k is None:excluded['missing_reference16k']+=1;continue
        valid=c.context_frames*HOP+c.valid_scored_samples
        frames=min(valid//HOP,c.latents.shape[-1])
        if frames<40+SCORE_FRAMES:excluded['insufficient_real_history']+=1;continue
        kind=natural_category(c,info)
        if kind in ('speech','other_natural') and hasattr(c,'teacher_audio'):
            target=array(c.teacher_audio[...,(frames-SCORE_FRAMES)*HOP:frames*HOP])
            if float(np.sqrt(np.mean(target*target)))<=.001:kind='quiet_or_low_level'
        # Selection is source-diverse and source-order deterministic, not based
        # on student error or gain. Prefer true-start reference windows.
        priority=(c.context_start_frame!=0,-frames,c.start_frame)
        if c.source_id not in by_source or priority<by_source[c.source_id][0]:by_source[c.source_id]=(priority,kind,c,info,frames)
    groups={}
    for _,kind,c,info,frames in sorted(by_source.values(),key=lambda v:(v[0],v[2].source_id)):
        groups.setdefault(kind,[]).append((kind,c,info,frames))
    if 'quiet_or_low_level' in groups:
        def quiet_priority(item):
            _,c,_,frames=item
            rms=float(np.sqrt(np.mean(array(c.teacher_audio[...,(frames-SCORE_FRAMES)*HOP:frames*HOP])**2))) if hasattr(c,'teacher_audio') else float('inf')
            return (rms,c.context_start_frame!=0,c.source_id)
        groups['quiet_or_low_level'].sort(key=quiet_priority)
    chosen=[]
    for repeat in range(2):
        for kind in NATURAL_CATEGORIES:
            if len(chosen)>=maximum:break
            if len(groups.get(kind,[]))>repeat:chosen.append(groups[kind][repeat])
    if len(chosen)<maximum:
        used={v[1].source_id for v in chosen}
        for kind in (*NATURAL_CATEGORIES,'other_natural'):
            for item in groups.get(kind,[]):
                if len(chosen)>=maximum:break
                if item[1].source_id not in used:chosen.append(item);used.add(item[1].source_id)
    coverage={kind:sum(v[0]==kind for v in chosen) for kind in NATURAL_CATEGORIES}
    return chosen,{'eligible_source_counts':{k:len(v) for k,v in groups.items()},'selected_category_counts':coverage,
        'missing_requested_categories':[k for k,v in coverage.items() if not v],'excluded_crops':excluded,
        'selected_categories':[v[0] for v in chosen],'synthetic_replacement_allowed':False,
        'minimum_real_history_frames':40,'score_frames':SCORE_FRAMES,'score_seconds':.64,
        'selection':'Up to two per requested natural category first, then other eligible natural sources; one crop per source; prefer true-start windows; no student-error-based selection',
        'quiet_label':'Natural teacher RMS<=0.001 in the scored window, or quiet/silence/whisper/breath condition as explicit fallback. Prefer lowest teacher RMS within this group; actual RMS and strict quiet flag are reported, never assumed from condition.'}

def run(ctx,names=('parent','targeted','complex'),maximum=8):
    import torch
    from audiovae_student.fusion_evaluation import _preserved_evaluation
    from diagnostic_common import atomic_json,status
    selected,selection=select_cases(ctx,maximum)
    if not selected:raise ValueError('No valid source has40history+16score frames')
    out=ctx.out/'natural-history-supplement-v1'
    if out.exists():raise ValueError('Preserve existing supplement outputs')
    out.mkdir(parents=True)
    result={'format_version':1,'main_source_sha256':MAIN_SHA256,'supplement_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':__doc__,'selection':selection,'prefix_frames':list(PREFIXES),'gain_factors':list(GAINS),
        'teacher':[],'checkpoints':{},'parameter_updates':0,'optimizer_updates':0,'primary_scores_or_audio_changed':False,
        'device_scope':'Frozen teacher/student diagnostics, not an inference-speed benchmark',
        'diagnostic_score_grid':'Last16 complete valid frames; includes the first samples of that interior score so boundary effects remain visible. Historical panel scores and their six-sample exclusion are untouched.'}
    cases=[];teacher=ctx.teacher()
    with _preserved_evaluation(teacher.model):
        for kind,c,info,frames in selected:
            status('history_teacher',source=c.source_id)
            z=c.latents[...,:frames].contiguous().clone();anchor=frames-SCORE_FRAMES;score=slice(anchor*HOP,frames*HOP)
            y=checked_decode(teacher.decode,z,teacher.device);repeated=checked_decode(teacher.decode,z,teacher.device)
            target=y[...,score];history={}
            for prefix in PREFIXES:
                reset=checked_decode(teacher.decode,z[...,anchor-prefix:].contiguous(),teacher.device)
                history[prefix]=reset[...,prefix*HOP:(prefix+SCORE_FRAMES)*HOP]
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
                'scored_frames':SCORE_FRAMES,'latent_sha256':tensor_hash(z),'reference_is_complete_original_recording':False,
                'history_reference_kind':'true-start partial utterance' if source['true_start'] else 'truncated supplied context window',
                'teacher_repeat':raw_metrics(repeated,y),'teacher_repeat_bitwise':bool(torch.equal(repeated.view(torch.int32),y.view(torch.int32))),
                'teacher_future':future_probe(teacher.decode,z,teacher.device,anchor),
                'history':[{'prefix_frames':p,'teacher_reset_vs_longest_history':metrics(history[p],target,fit=False)} for p in PREFIXES],
                'amplitude':[{'gain':a['gain'],'input_rms':a['input_rms'],'raw_input_sha256':a['raw_input_sha256'],
                    'encoded_z_sha256':a['encoded_z_sha256'],'teacher_rms':float(a['teacher'].square().mean().sqrt()),
                    'teacher_homogeneity_error':raw_metrics(a['teacher'],baseline*a['gain'])} for a in amplitude],
                'amplitude_gain1_reencode_repeat':raw_metrics(baseline_again,baseline)}
            tr['teacher_scored_rms']=float(target.square().mean().sqrt())
            tr['teacher_scored_is_quiet_at_0_001']=tr['teacher_scored_rms']<=.001
            tr['natural_source_verified_by_selection']=True
            if source['true_start']:
                tr['cached_teacher_vs_redecoded_z']=raw_metrics(target,c.teacher_audio[...,score])
                tr['gain1_reencoded_teacher_vs_cached_target']=raw_metrics(baseline,c.teacher_audio[...,score])
            tr['teacher_history_log_error_slope_per_frame']=slope(PREFIXES,[r['teacher_reset_vs_longest_history']['raw_primary']['residual_rms'] for r in tr['history']],log_y=True)
            tr['teacher_log_rms_vs_log_gain_slope']=slope([a['gain'] for a in amplitude],[r['teacher_rms'] for r in tr['amplitude']],True,True)
            result['teacher'].append(tr);cases.append((c,z,anchor,target,history,amplitude))
    del teacher;torch.cuda.empty_cache()
    atomic_json(out/'history-sensitivity-teacher.json',result)
    for name in names:
        engine=ctx.engine(name,device='cuda');rows=[]
        with _preserved_evaluation(engine.model):
            for c,z,anchor,target,teacher_hist,amplitude in cases:
                status('history_student',checkpoint=name,source=c.source_id)
                full=checked_decode(engine.model,z,engine.device)[...,anchor*HOP:];hist=[]
                for prefix in PREFIXES:
                    pred=checked_decode(engine.model,z[...,anchor-prefix:].contiguous(),engine.device)[...,prefix*HOP:(prefix+SCORE_FRAMES)*HOP]
                    hist.append({'prefix_frames':prefix,'student_reset_vs_longest_history':raw_metrics(pred,full),
                        'same_history_student_vs_teacher':metrics(pred,teacher_hist[prefix],fit=False),
                        'student_reset_vs_teacher_longest_history':raw_metrics(pred,target)})
                amplitude_predictions=[checked_decode(engine.model,a['z'],engine.device)[...,anchor*HOP:(anchor+SCORE_FRAMES)*HOP] for a in amplitude]
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
        result['checkpoints'][name]=entry;atomic_json(out/f'history-sensitivity-{name}.json',entry)
        del engine;torch.cuda.empty_cache()
    if hashlib.sha256(MAIN_PATH.read_bytes()).hexdigest()!=MAIN_SHA256:raise RuntimeError('Frozen helper source changed')
    result['file_verification']=ctx.verify_files();atomic_json(out/'history-sensitivity.json',result);return result

if __name__=='__main__':
    from diagnostic_common import load_context
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',default=['parent','targeted','complex']);p.add_argument('--maximum',type=int,default=8)
    a=p.parse_args();run(load_context(),tuple(a.checkpoints),a.maximum)
