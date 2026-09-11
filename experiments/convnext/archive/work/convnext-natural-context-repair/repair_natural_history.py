"""Natural cached-history diagnostics with separately verified raw amplitude probes.

The cached latent and target panel does not require reference16k. Natural source
recordings are loaded independently for amplitude probes. All score windows are
16 real frames; no synthetic context or historical quality-score change occurs.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from diagnose_history_sensitivity import HOP,PREFIXES,GAINS,array,raw_metrics,metrics,slope,tensor_hash,checked_decode,future_probe
from natural_history import is_synthetic,natural_category,NATURAL_CATEGORIES,MAIN_SHA256,MAIN_PATH
SCORE=16

def choose(ctx,maximum=8):
    if type(maximum)is not int or not 4<=maximum<=8:raise ValueError('Need4..8 source-diverse natural cases')
    candidates=[];seen=set();excluded={'synthetic':0,'under56_real_frames':0}
    for c in [*ctx.heldout,*ctx.probe_crops]:
        key=(c.source_id,c.start_frame,c.context_start_frame)
        if key in seen:continue
        seen.add(key);info=ctx.metadata.get(c.source_id,{})
        if is_synthetic(c,info):excluded['synthetic']+=1;continue
        frames=min(c.latents.shape[-1],(c.context_frames*HOP+c.valid_scored_samples)//HOP)
        if frames<56:excluded['under56_real_frames']+=1;continue
        category=natural_category(c,info);end=frames
        y=array(c.teacher_audio[...,:frames*HOP]);power=np.r_[0.,np.cumsum(y*y)]
        def rms_at(end):return math.sqrt(max(0.,float(power[end*HOP]-power[(end-SCORE)*HOP]))/(SCORE*HOP))
        rms=rms_at(end)
        # Inspect only real teacher samples and require40 preceding real frames.
        # Quiet selection is independent of every student output.
        quiet_end=min(range(56,frames+1),key=rms_at);quiet_rms=rms_at(quiet_end)
        candidates.append({'crop':c,'metadata':info,'category':category,'frames':end,'teacher_selection_rms':rms,'quiet_search':False})
        if quiet_rms<=.001 or category=='quiet_or_low_level':
            candidates.append({'crop':c,'metadata':info,'category':'quiet_or_low_level','frames':quiet_end,
                'teacher_selection_rms':quiet_rms,'quiet_search':True})
    groups={}
    for item in candidates:
        if item['category'] not in NATURAL_CATEGORIES:continue
        groups.setdefault(item['category'],[]).append(item)
    def priority(v):
        c=v['crop'];lead=(v['teacher_selection_rms'],) if v['category']=='quiet_or_low_level' else ()
        return lead+(c.context_start_frame!=0,-v['frames'],c.source_id,c.start_frame)
    for group in groups.values():group.sort(key=priority)
    selected=[];used=set()
    # Reserve quiet first so one naturally quiet speech source cannot consume
    # both categories. All conditions still have their own required source.
    order=('quiet_or_low_level','speech','laughter','whistle')
    for _ in range(2):
        for kind in order:
            if len(selected)>=maximum:break
            item=next((v for v in groups.get(kind,[]) if v['crop'].source_id not in used),None)
            if item is not None:selected.append(item);used.add(item['crop'].source_id)
    counts={kind:sum(v['category']==kind for v in selected) for kind in NATURAL_CATEGORIES}
    missing=[k for k,v in counts.items() if not v]
    if missing:raise ValueError('Required natural categories absent: '+','.join(missing))
    return selected,{'selected_category_counts':counts,'missing_categories':missing,'excluded':excluded,
        'selection':'Distinct source identities, up to two per group; all four natural groups required; no reference16k gate; lowest teacher RMS for natural quiet, then true-start and longer available real context',
        'quiet_definition':'Teacher RMS<=0.001 in16complete real frames if available; otherwise explicitly labeled whisper/breath/quiet-condition fallback',
        'reference16k_required_for_history':False,'score_frames':SCORE,'score_seconds':.64,'minimum_history_frames':40}

def source_rows(ctx):
    rows={v['source_id']:v for v in ctx.parent['identity']['data']['heldout']['rows']};pins={}
    base=ctx.out.parent
    for name in ('validation-appendix','language-validation'):
        p=base/name/'manifest.jsonl'
        if not p.exists():continue
        raw=p.read_bytes();pins[str(p)]=hashlib.sha256(raw).hexdigest()
        ready=p.parent/'ready.json'
        if ready.exists():
            proof=json.loads(ready.read_text());expected=proof.get('files_sha256',{}).get('manifest.jsonl')
            if expected and expected!=pins[str(p)]:raise ValueError('Heldout source manifest differs from original readiness proof')
            pins[str(ready)]=hashlib.sha256(ready.read_bytes()).hexdigest()
        for line in raw.decode().splitlines():
            row=json.loads(line);sid=row['source_id']
            if sid in rows and rows[sid]!=row:raise ValueError('Conflicting authentic source row '+sid)
            rows[sid]=row
    return rows,pins

def authentic_audio(row):
    import torch
    from audiovae_student.data import ManifestRow
    from audiovae_student.source_corpus import read_native_16k
    from audiovae_student import source_corpus
    p=Path(row['audio_path']);raw=p.read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=row['audio_sha256']:raise ValueError('Raw source bytes differ')
    a=read_native_16k(raw,ManifestRow.from_dict(row))
    if a.shape[:2]!=(1,1) or not a.shape[-1] or not bool(torch.isfinite(a).all()):raise ValueError('Source is not valid mono16k')
    if abs(a.shape[-1]/16000-row['duration_seconds'])>1/16000+1e-10:raise ValueError('Source duration mismatch')
    return a,{'source_id':row['source_id'],'audio_sha256':actual,
        'prepared_samples16k':a.shape[-1],'source_split':row['split'],'sample_rate_hz':16000,
        'reader_source_sha256':hashlib.sha256(Path(source_corpus.__file__).read_bytes()).hexdigest(),
        'reader':'Original SourceCorpus.read_native_16k; prepared mono16k bytes, no new resampling, gain or channel change',
        'scope':'Complete supplied source-manifest recording segment; not necessarily the entire parent recording',
        'gain_policy':row['gain_policy'],'raw_source_verified':True}

def inventory(ctx,selected):
    rows,pins=source_rows(ctx);report=[];loaded={}
    for item in selected:
        c=item['crop'];row={'source_id':c.source_id,'category':item['category'],'frames':item['frames'],
            'context_start_frame':c.context_start_frame,'raw_reference_cached':c.reference16k is not None,
            'teacher_selection_rms':item['teacher_selection_rms'],'strict_quiet':item['teacher_selection_rms']<=.001}
        try:
            if c.source_id not in rows:raise FileNotFoundError('No heldout source-manifest row')
            audio,info=authentic_audio(rows[c.source_id]);end=(c.context_start_frame+item['frames'])*640
            if end>audio.shape[-1]:raise ValueError('Matching diagnostic interval exceeds real source samples')
            loaded[c.source_id]=(audio,info);row['amplitude_source']=info
        except (OSError,ValueError,KeyError) as e:row['amplitude_unavailable']=str(e)
        report.append(row)
    return loaded,{'source_manifest_sha256':pins,'cases':report,'amplitude_available_cases':len(loaded)}

def run(ctx,names=('parent','targeted','complex'),maximum=8,inventory_only=False):
    import torch
    from audiovae_student.fusion_evaluation import _preserved_evaluation
    from diagnostic_common import atomic_json,status
    selected,selection=choose(ctx,maximum);sources,raw_inventory=inventory(ctx,selected)
    out=ctx.out/'natural-history-repair-v1'
    if inventory_only:return {'selection':selection,'inventory':raw_inventory}
    if out.exists():raise ValueError('Refusing to overwrite existing repair')
    out.mkdir(parents=True)
    result={'format_version':1,'scope':__doc__,'main_source_sha256':MAIN_SHA256,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':selection,'source_inventory':raw_inventory,
        'teacher':[],'checkpoints':{},'prefix_frames':list(PREFIXES),'gain_factors':list(GAINS),
        'parameter_updates':0,'optimizer_updates':0,'historical_panel_changed':False,
        'diagnostic_score_grid':'16 complete real frames at the same absolute interval per source; includes boundary samples; separate from historical quality mask',
        'amplitude_policy':'Encode each scaled complete authentic source independently; gain1 uses the same full source as every other gain; cached decoder history is a separate experiment'}
    atomic_json(out/'selection.json',{'selection':selection,'source_inventory':raw_inventory})
    cases=[];teacher=ctx.teacher()
    with _preserved_evaluation(teacher.model):
        for item in selected:
            c=item['crop'];frames=item['frames'];anchor=frames-SCORE;score=slice(anchor*HOP,frames*HOP)
            status('repair_teacher',source=c.source_id,category=item['category'])
            z=c.latents[...,:frames].contiguous().clone();y=checked_decode(teacher.decode,z,teacher.device);target=y[...,score]
            repeated=checked_decode(teacher.decode,z,teacher.device);hist={}
            for p in PREFIXES:
                hist[p]=checked_decode(teacher.decode,z[...,anchor-p:].contiguous(),teacher.device)[...,p*HOP:(p+SCORE)*HOP]
            absolute_start=c.context_start_frame+anchor;abs_score=slice(absolute_start*HOP,(absolute_start+SCORE)*HOP)
            amplitudes=[]
            if c.source_id in sources:
                audio,source=sources[c.source_id]
                for factor in GAINS:
                    raw=(audio*factor).to(teacher.device);before=tensor_hash(raw)
                    az=teacher.encode(raw).detach().cpu()
                    if tensor_hash(raw)!=before:raise RuntimeError('Encoder mutated supplied input')
                    ay=checked_decode(teacher.decode,az,teacher.device)[...,abs_score]
                    if ay.shape[-1]!=SCORE*HOP:raise ValueError('Amplitude scoregrid mismatch')
                    amplitudes.append({'gain':factor,'z':az,'teacher':ay,'raw_sha256':before,
                        'input_rms':float((audio*factor).square().mean().sqrt())})
            tr={'source_id':c.source_id,'category':item['category'],'metadata':item['metadata'],
                'longest_reference_frames':frames,'available_history_frames':anchor,'absolute_score_start_frame':absolute_start,
                'history_reference_kind':'true-start partial source' if c.context_start_frame==0 else 'truncated supplied cached latent window',
                'latent_sha256':tensor_hash(z),'scored_frames':SCORE,'teacher_rms':float(target.square().mean().sqrt()),
                'strict_quiet_at_0_001':float(target.square().mean().sqrt())<=.001,
                'teacher_repeat':raw_metrics(repeated,y),'teacher_repeat_bitwise':bool(torch.equal(repeated.view(torch.int32),y.view(torch.int32))),
                'teacher_future':future_probe(teacher.decode,z,teacher.device,anchor),
                'history':[{'prefix_frames':p,'teacher_reset_vs_longest_history':metrics(hist[p],target,fit=False)} for p in PREFIXES],
                'cached_original_target_vs_redecoded_window':raw_metrics(c.teacher_audio[...,score],target),
                'amplitude_status':'available' if amplitudes else 'unavailable: authentic source missing or invalid','amplitude':[]}
            if amplitudes:
                baseline=amplitudes[0]['teacher'];audio=sources[c.source_id][0]
                bz=teacher.encode(audio.to(teacher.device)).detach().cpu();by=checked_decode(teacher.decode,bz,teacher.device)[...,abs_score]
                tr['amplitude_gain1_reencode_repeat']=raw_metrics(by,baseline)
                tr['full_source_gain1_teacher_vs_cached_target']=raw_metrics(baseline,c.teacher_audio[...,score])
                same_z=amplitudes[0]['z'][...,c.context_start_frame:c.context_start_frame+frames].contiguous()
                tr['full_source_gain1_latents_vs_cached_window']=raw_metrics(same_z,z)
                same_y=checked_decode(teacher.decode,same_z,teacher.device)[...,score]
                tr['gain1_same_cached_decoder_history_vs_cached_latent_decode']=raw_metrics(same_y,target)
                tr['gain1_full_source_vs_same_cached_decoder_history']=raw_metrics(baseline,same_y)
                tr['amplitude']=[{'gain':a['gain'],'input_rms':a['input_rms'],'raw_input_sha256':a['raw_sha256'],
                    'encoded_z_sha256':tensor_hash(a['z']),'teacher_rms':float(a['teacher'].square().mean().sqrt()),
                    'teacher_homogeneity_error':raw_metrics(a['teacher'],baseline*a['gain'])} for a in amplitudes]
                tr['teacher_log_rms_vs_log_gain_slope']=slope(GAINS,[v['teacher_rms'] for v in tr['amplitude']],True,True)
            tr['teacher_history_log_error_slope_per_frame']=slope(PREFIXES,[v['teacher_reset_vs_longest_history']['raw_primary']['residual_rms'] for v in tr['history']],log_y=True)
            result['teacher'].append(tr);cases.append((c,z,anchor,absolute_start,target,hist,amplitudes))
    del teacher;torch.cuda.empty_cache();atomic_json(out/'history-sensitivity-teacher.json',result)
    for name in names:
        engine=ctx.engine(name,device='cuda');rows=[]
        with _preserved_evaluation(engine.model):
            for c,z,anchor,absolute_start,target,teacher_hist,amplitudes in cases:
                status('repair_student',checkpoint=name,source=c.source_id)
                full=checked_decode(engine.model,z,engine.device)[...,anchor*HOP:];hist=[]
                for p in PREFIXES:
                    pred=checked_decode(engine.model,z[...,anchor-p:].contiguous(),engine.device)[...,p*HOP:(p+SCORE)*HOP]
                    hist.append({'prefix_frames':p,'student_reset_vs_longest_history':raw_metrics(pred,full),
                        'same_history_student_vs_teacher':metrics(pred,teacher_hist[p],fit=False),
                        'student_reset_vs_teacher_longest_history':raw_metrics(pred,target)})
                row={'source_id':c.source_id,'longest_history_student_vs_teacher':metrics(full,target),'history':hist,
                    'future':future_probe(engine.model,z,engine.device,anchor),
                    'history_log_error_slope_per_frame':slope(PREFIXES,[v['student_reset_vs_longest_history']['residual_rms'] for v in hist],log_y=True),
                    'amplitude':[],'amplitude_secants':[]}
                if amplitudes:
                    sc=slice(absolute_start*HOP,(absolute_start+SCORE)*HOP)
                    predictions=[checked_decode(engine.model,a['z'],engine.device)[...,sc] for a in amplitudes]
                    for a,pred in zip(amplitudes,predictions):
                        row['amplitude'].append({'gain':a['gain'],'input_rms':a['input_rms'],
                            'student_vs_reencoded_teacher':metrics(pred,a['teacher'],fit=a['gain']==1.),
                            'student_homogeneity_error':raw_metrics(pred,predictions[0]*a['gain'])})
                    for i in range(len(amplitudes)-1):
                        da=amplitudes[i]['gain']-amplitudes[i+1]['gain']
                        row['amplitude_secants'].append({'gains':[amplitudes[i]['gain'],amplitudes[i+1]['gain']],
                            'student_vs_teacher_secant':raw_metrics((predictions[i]-predictions[i+1])/da,(amplitudes[i]['teacher']-amplitudes[i+1]['teacher'])/da),
                            'interpretation':'Full-source re-encoding finite amplitude secant, not local JVP'})
                    row['student_log_rms_vs_log_gain_slope']=slope(GAINS,[v['student_vs_reencoded_teacher']['raw_primary']['prediction_rms'] for v in row['amplitude']],True,True)
                    row['teacher_log_rms_vs_log_gain_slope']=slope(GAINS,[v['student_vs_reencoded_teacher']['raw_primary']['target_rms'] for v in row['amplitude']],True,True)
                rows.append(row)
        result['checkpoints'][name]={'checkpoint_sha256':ctx.expected_hashes[name],'step':engine.step,'rows':rows,'model_state_unchanged':True,'parameter_updates':0,'optimizer_updates':0}
        atomic_json(out/f'history-sensitivity-{name}.json',result['checkpoints'][name]);del engine;torch.cuda.empty_cache()
    if hashlib.sha256(MAIN_PATH.read_bytes()).hexdigest()!=MAIN_SHA256:raise RuntimeError('Frozen helper source changed')
    result['file_verification']=ctx.verify_files();atomic_json(out/'history-sensitivity.json',result);return result

if __name__=='__main__':
    from diagnostic_common import load_context
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--maximum',type=int,default=8);p.add_argument('--checkpoints',nargs='+',choices=('parent','targeted','complex'),default=['parent','targeted','complex']);p.add_argument('--inventory-only',action='store_true')
    a=p.parse_args();r=run(load_context(),tuple(a.checkpoints),a.maximum,a.inventory_only)
    if a.inventory_only:print(json.dumps(r,indent=2,allow_nan=False))
