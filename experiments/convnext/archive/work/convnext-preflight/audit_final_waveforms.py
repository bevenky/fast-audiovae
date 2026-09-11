"""Bounded, no-update diagnostic of the completed r5 decoder on dev data.

Original training assets stay read-only. Audio remains on Runpod. Diagnostic
lag/gain/band decompositions never alter the unshifted acceptance measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r5')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')
PLAN = Path('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1')
OUT = Path(sys.argv[1])
sys.path.insert(0, str(BASE))

from audiovae_student.cache import load_cache, prepare_utterance_cache
from audiovae_student.data import load_manifest
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.source_corpus import read_native_16k
from audiovae_student.teacher import FrozenAudioVAE2
from run_representative_stage import load_dev_panel, panel_crop


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def cos(x, y):
    return float(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30))


def db_ratio(x, y):
    return float(20*np.log10(max(x, 1e-15)/max(y, 1e-15)))


def lag_audit(p, t):
    n = len(t)
    maximum = min(960, n//4)
    fft_size = 1 << (2*n-2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(p,fft_size)*np.fft.rfft(t[::-1],fft_size),fft_size)[:2*n-1]
    lags = np.arange(-maximum, maximum+1)
    pp = np.r_[0., np.cumsum(p*p)]
    tt = np.r_[0., np.cumsum(t*t)]
    scores = []
    for lag in lags:
        ps, pe = max(0, lag), n + min(0, lag)
        ts, te = max(0, -lag), n - max(0, lag)
        scores.append(corr[n-1+lag]/max(np.sqrt((pp[pe]-pp[ps])*(tt[te]-tt[ts])), 1e-30))
    best = int(np.argmax(scores))
    return {'search_samples_each_direction': maximum, 'best_lag_samples': int(lags[best]),
            'best_lag_ms': float(lags[best]/48), 'best_cosine': float(scores[best]),
            'interpretation': 'positive lag means student delayed relative to teacher; diagnostic only'}


def phase_template(audio, mask):
    # Mask consists of complete original-grid 20 ms windows, each 2 head blocks.
    windows = audio[:len(mask)*960].reshape(-1,960)[mask]
    blocks = windows.reshape(-1,480)
    if not len(blocks):
        return None, None
    dc = float(blocks.mean())
    centered = blocks-dc
    template = centered.mean(axis=0)
    power = float(np.mean(centered*centered))
    detail = {'head_blocks': len(blocks), 'rms': rms(blocks), 'dc': dc,
              'dc_power_fraction': dc*dc/max(float(np.mean(blocks*blocks)),1e-30),
              'phase480_template_rms': rms(template),
              'phase480_explained_ac_power_fraction': float(np.mean(template*template)/max(power,1e-30)),
              'residual_after_phase_template_rms': rms(centered-template)}
    return detail, template


def metrics(p, t):
    error = p-t
    teacher_energy = float(np.dot(t,t))
    gain = float(np.dot(p,t)/max(teacher_energy,1e-30))
    orthogonal = p-gain*t
    result = {'samples':len(t), 'teacher_rms':rms(t), 'student_rms':rms(p),
              'student_to_teacher_db':db_ratio(rms(p),rms(t)), 'cosine':cos(p,t),
              'l1':float(np.mean(np.abs(error))), 'error_rms':rms(error),
              'error_snr_db':db_ratio(rms(t),rms(error)),
              'student_dc':float(p.mean()), 'student_dc_power_fraction':float(p.mean()**2/max(np.mean(p*p),1e-30)),
              'error_after_dc_removal_rms':rms(error-error.mean()),
              'teacher_projection_gain':gain, 'orthogonal_error_rms':rms(orthogonal),
              'error_power_from_projection_gain_fraction':float((gain-1)**2*teacher_energy/max(np.dot(error,error),1e-30)),
              'lag_diagnostic':lag_audit(p,t)}
    fp,ft=np.fft.rfft(p),np.fft.rfft(t)
    hz=np.fft.rfftfreq(len(t),1/48000)
    bands=[]
    for lo,hi in ((0,300),(300,3000),(3000,8000),(8000,16000),(16000,24001)):
        mask=(hz>=lo)&(hz<hi)
        bp=np.fft.irfft(fp*mask,n=len(t));bt=np.fft.irfft(ft*mask,n=len(t))
        bands.append({'hz':[lo,min(hi,24000)],'teacher_rms':rms(bt),'student_rms':rms(bp),
                      'student_to_teacher_db':db_ratio(rms(bp),rms(bt)),
                      'cosine':cos(bp,bt),'residual_rms':rms(bp-bt)})
    result['bands']=bands
    n=len(t)//960
    tw=t[:n*960].reshape(n,960);pw=p[:n*960].reshape(n,960)
    tr=np.sqrt(np.mean(tw*tw,axis=1));pr=np.sqrt(np.mean(pw*pw,axis=1))
    quiet=tr<=.001
    result['quiet']={'windows':int(quiet.sum()),'complete_windows':n}
    template=None
    if quiet.any():
        qt=tw[quiet].reshape(-1);qp=pw[quiet].reshape(-1)
        sd,template=phase_template(p,quiet);td,_=phase_template(t,quiet);ed,_=phase_template(error,quiet)
        result['quiet'].update({'teacher_rms':rms(qt),'student_rms':rms(qp),'residual_rms':rms(qp-qt),
                               'student_to_teacher_db':db_ratio(rms(qp),rms(qt)),
                               'student_phase480':sd,'teacher_phase480':td,'residual_phase480':ed})
    return result,template


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    checkpoint=BASE/'training-runs/representative-reconstruction-v1/latest.pt'
    frozen_files=[checkpoint,BASE/'source.tgz',BASE/'teacher-cache-pilot-v1/index.sqlite3',
                  BASE/'training-runs/representative-reconstruction-v1/metrics.jsonl']
    before={str(path):sha(path) for path in frozen_files}
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    evaluation=json.loads((checkpoint.parent/'evaluation-step001009.json').read_text())
    rows,counts,metadata,windows,panel_identity=load_dev_panel(PANEL)
    by_id={r.source_id:r for r in rows}
    by_spec={(w['source_id'],w['start_frame']):w for w in windows}
    saved={(x['source_id'],x['start_frame']):x for x in payload['identity']['heldout']}
    all_e=evaluation['rows']
    selected=[]
    def add(candidates,label,mode='median'):
        candidates=[x for x in candidates if (x['source_id'],x['start_frame']) not in {(v['source_id'],v['start_frame']) for v in selected}]
        if not candidates:return
        candidates=sorted(candidates,key=lambda x:x['waveform_cosine'])
        x=dict(candidates[len(candidates)//2] if mode=='median' else candidates[0])
        x['selection_reason']=label
        selected.append(x)
    for x in sorted(all_e,key=lambda x:x['teacher_rms'])[:2]:
        add([x],'two lowest teacher-RMS saved dev crops')
    for lang in ('en','hi','es','fr'):
        add([x for x in all_e if metadata[x['source_id']]['condition']=='speech'
             and metadata[x['source_id']]['language'].split('_')[0]==lang], 'median speech cosine:'+lang)
    add([x for x in all_e if metadata[x['source_id']]['condition']=='Laughter'],'median laughter')
    add([x for x in all_e if metadata[x['source_id']]['condition']=='Laughter'],'lowest remaining laughter','lowest')
    add([x for x in all_e if metadata[x['source_id']]['condition']=='Screaming'],'median screaming')
    for source in ('freesound:457967','freesound:539498'):
        add([x for x in all_e if x['source_id']==source],'one crop per human whistle source')
    add([x for x in all_e if metadata[x['source_id']]['condition']=='generic_emotion_nonverbal'],'median generic expressive')
    assert len(selected)<=12
    cache=BASE/'teacher-cache-pilot-v1'
    con=sqlite3.connect('file:'+str(cache/'index.sqlite3')+'?mode=ro&immutable=1',uri=True)
    entries={info['row_id']:(key,info) for key,raw in con.execute('SELECT lookup_key,info FROM entries')
             for info in [json.loads(raw)]}
    con.close()
    teacher=None;teacher_fingerprint=None
    record_cache={}
    first=load_manifest(PLAN/'train-manifest.jsonl')[0]
    model=StudentDecoder(StudentConfig(**payload['engine']['model_config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['engine']['model'])
    student_before=state_fingerprint(model.state_dict())
    results=[];templates=[]
    for item in selected:
        source=item['source_id']; row=by_id[source]
        if source not in record_cache:
            if source in entries:
                key,info=entries[source];path=cache/'targets'/(key+'.pt')
                assert sha(path)==info['file_sha256']
                record=load_cache(path);origin='existing immutable teacher cache'
            else:
                if teacher is None:
                    teacher=FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cuda')
                    teacher=_target_teacher(teacher,first)
                    assert teacher.provenance==payload['identity']['data']['source_corpus']['teacher']
                    warm_bytes=Path(first.audio_path).read_bytes();assert hashlib.sha256(warm_bytes).hexdigest()==first.audio_sha256
                    warm=read_native_16k(warm_bytes,first).cuda()
                    for _ in range(2):teacher.decode(teacher.encode(warm))
                    teacher_fingerprint=state_fingerprint(teacher.model.state_dict())
                audio_bytes=Path(row.audio_path).read_bytes();assert hashlib.sha256(audio_bytes).hexdigest()==row.audio_sha256
                audio=read_native_16k(audio_bytes,row)
                assert audio.shape[-1]==counts[source]
                record=prepare_utterance_cache(audio,row,teacher)
                origin='regenerated serial whole-utterance frozen original FP32 teacher; old dev cache evicted'
            record_cache[source]=(record,origin)
        record,origin=record_cache[source]
        crop=panel_crop(record,by_spec[(source,item['start_frame'])])
        identity=_crop_identity([crop])[0];previous=saved[(source,item['start_frame'])]
        assert all(identity[k]==previous[k] for k in identity if k not in ('latents_sha256','teacher_sha256'))
        with torch.no_grad():
            prediction=model(crop.latents.cuda())[...,crop.scored_slice].cpu().numpy().reshape(-1).astype(np.float64)
        target=crop.teacher_audio[...,crop.scored_slice].numpy().reshape(-1).astype(np.float64)
        detail,template=metrics(prediction,target)
        detail.update({'source_id':source,'start_frame':item['start_frame'],'metadata':metadata[source],
                       'selection_reason':item['selection_reason'],'target_origin':origin,
                       'crop_identity':identity,'saved_crop_identity':previous,
                       'same_latent_bytes_as_saved':identity['latents_sha256']==previous['latents_sha256'],
                       'same_teacher_bytes_as_saved':identity['teacher_sha256']==previous['teacher_sha256'],
                       'saved_evaluation_cosine':item['waveform_cosine'],
                       'saved_evaluation_student_rms':item['student_rms'],
                       'saved_evaluation_teacher_rms':item['teacher_rms']})
        results.append(detail)
        if template is not None:templates.append((source,item['start_frame'],template))
        print(json.dumps({'source_id':source,'cosine':detail['cosine'],'target_hash_match':detail['same_teacher_bytes_as_saved'],
                          'quiet_windows':detail['quiet']['windows']}),flush=True)
    pairs=[{'a':[sa,fa],'b':[sb,fb],'phase480_template_cosine':cos(a,b)}
           for i,(sa,fa,a) in enumerate(templates) for sb,fb,b in templates[i+1:]]
    after={str(path):sha(path) for path in frozen_files}
    assert before==after
    assert state_fingerprint(model.state_dict())==student_before
    if teacher is not None:assert state_fingerprint(teacher.model.state_dict())==teacher_fingerprint
    output={'checkpoint':str(checkpoint),'checkpoint_sha256':before[str(checkpoint)],'step':payload['engine']['step'],
            'policy':'12-crop maximum no-update diagnostic; no RTF measurements; unshifted acceptance unchanged',
            'target_regeneration_note':'Byte mismatches are reported, never silently treated as original saved targets.',
            'frozen_files_unchanged':True,'model_and_teacher_unchanged':True,'clips':results,
            'cross_clip_quiet_phase480_templates':pairs}
    (OUT/'waveform-audit.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps({'output':str(OUT/'waveform-audit.json'),'clips':len(results),'unchanged':True}),flush=True)


if __name__=='__main__':main()
