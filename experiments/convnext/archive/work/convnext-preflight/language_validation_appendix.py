"""Freeze and evaluate a separate newly acquired language panel on the Runpod CPU."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.comparison_data import load_comparison_plan,assert_comparison_disjoint,unknown_session
from audiovae_student.data import load_manifest,validate_manifest
from audiovae_student.distillation_training import evaluate_crops,waveform_gate
from audiovae_student.model import StudentConfig,StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.recipe_v2 import RecipeV2Config,RecipeV2Engine
from audiovae_student.representative_pilot import summarize_evaluation,_verify_panel_targets
from audiovae_student.restart_data import file_sha,digest
from audiovae_student.source_corpus import SourceCorpus,read_native_16k
from audiovae_student.teacher import FrozenAudioVAE2
from validation_appendix import BASE,OLD,immutable


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--manifest',type=Path,required=True)
    args.add_argument('--acquisition-report',type=Path,required=True)
    options=args.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out=BASE/'remediation/language-validation'
    out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(options.manifest)
    if len(rows)>80 or not rows or len({r.source_id for r in rows})!=len(rows):
        raise ValueError('Language panel must have 1 to80 distinct files')
    if any(r.split not in ('dev','test') or r.license!='CC-BY-4.0' for r in rows):
        raise ValueError('Expected reviewed CC-BY-4.0 held-out files only')
    validate_manifest(rows)
    plans=[load_comparison_plan(BASE/'data/recipe-v2'/name) for name in ('optimization','calibration')]
    training={r.source_id:r for p in plans for r in p['rows']}
    assert_comparison_disjoint(training.values(),rows)
    state=json.loads((BASE/'remediation/monitoring/state.json').read_text())
    receipt=next(v for v in state['checkpoints'] if v['threshold']==3000 and v['state']=='retained')
    checkpoint=Path(receipt['path'])
    if file_sha(checkpoint)!=receipt['sha256']:
        raise ValueError('Retained 3000-step checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    old_sources={r['source_id'] for r in payload['identity']['data']['heldout']['rows']}
    if old_sources&{r.source_id for r in rows}:
        raise ValueError('New language panel duplicates original panel')
    old_target_sha=file_sha(BASE/'heldout-targets.pt')
    if old_target_sha!='3349f715c9ae4c4b7ac859cde935cdc949395bfc0e15915ad759c8e547a4dd2e':
        raise ValueError('Pinned original trend targets changed')
    counts,checks,metadata,windows={},{},{},[]
    for row in rows:
        raw=Path(row.audio_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row.audio_sha256:
            raise ValueError('Held-out source file hash changed')
        audio=read_native_16k(raw,row)
        count=audio.shape[-1]
        if not torch.isfinite(audio).all() or abs(count/16000-row.duration_seconds)>1/16000+1e-10:
            raise ValueError('Decoded source is invalid or its length differs')
        counts[row.source_id]=count
        checks[row.source_id]={'sha256':row.audio_sha256,'decoded_samples16k':count,'finite':True,
            'sample_rate_hz':16000,'split':row.split,'source_split':row.source_split,
            'speaker_id':row.speaker_id,'session_id':row.session_id,
            'unknown_language_session_placeholder':unknown_session(row),
            'parent_recording_id':row.parent_recording_id,'language':row.language,
            'dataset':row.dataset,'license':row.license,'source_revision':row.source_revision,
            'access_record':row.access_record,'content_verified_by_listening':False}
        metadata[row.source_id]={'dataset':row.dataset,'language':row.language,'condition':'speech',
                                'coverage_group':'language:'+row.language}
        for start,frames in ((0,16),(max(0,math.ceil(count/640)-64),64)):
            if any(w['source_id']==row.source_id and w['start_frame']==start for w in windows):
                continue
            valid=min(frames*640,count-start*640)
            if valid*3<4096:
                raise ValueError('New language crop is below the established minimum')
            windows.append({'source_id':row.source_id,'start_frame':start,'scored_frames':frames,
                            'valid_input_samples16k':valid,'valid_output_samples48k':valid*3})
    for name,value,jsonl in [('manifest.jsonl',[r.to_dict() for r in rows],True),
            ('input-sample-counts.json',counts,False),('groups.json',metadata,False),
            ('windows.jsonl',windows,True),('source-checks.json',checks,False)]:
        immutable(out/name,value,jsonl=jsonl)
    identity={'state':'ready','panel':'new-heldout-languages-v1','rows':len(rows),'windows':len(windows),
        'source_seconds':sum(counts.values())/16000,'languages':sorted({r.language for r in rows}),
        'known_speakers':len({r.speaker_id for r in rows if r.speaker_id}),
        'known_sessions':len({r.session_id for r in rows if r.session_id and not unknown_session(r)}),
        'unknown_session_rows':sum(unknown_session(r) for r in rows),
        'all_future_training_and_calibration_sources_excluded':True,
        'training_source_count_checked':len(training),
        'full_plan_window_count_checked':sum(len(p['windows']) for p in plans),
        'optimization_plan_identity':plans[0]['identity']['identity_sha256'],
        'calibration_plan_identity':plans[1]['identity']['identity_sha256'],
        'input_manifest_sha256':file_sha(options.manifest),
        'acquisition_report_sha256':file_sha(options.acquisition_report),
        'old_target_sha256':old_target_sha,'context_frames':29,
        'scope':'Separate validation panel; source labels and known identities verified, no listening or acoustic duplicate search',
        'files_sha256':{name:file_sha(out/name) for name in ('manifest.jsonl','input-sample-counts.json','groups.json','windows.jsonl','source-checks.json')}}
    identity['identity_sha256']=digest(identity)
    immutable(out/'ready.json',identity)
    del plans,training
    print(json.dumps({'phase':'language_panel_verified','rows':len(rows),'crops':len(windows),
                      'languages':identity['languages'],'source_seconds':identity['source_seconds']}),flush=True)
    teacher=FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cpu')
    teacher=_target_teacher(teacher,rows[0])
    teacher_hash=state_fingerprint(teacher.model.state_dict())
    if teacher_hash!=payload['identity']['data']['teacher_state_sha256']:
        raise ValueError('Pinned teacher weights differ from training teacher')
    warm=read_native_16k(Path(rows[0].audio_path).read_bytes(),rows[0])
    with torch.jit.optimized_execution(True):
        for _ in range(2):
            teacher.decode(teacher.encode(warm))
    del warm
    helper_path=Path('/workspace/fast-audiovae-convnext-20260909-r5/run_representative_stage.py')
    if file_sha(helper_path)!='72c249809abeb889a21c0391c65ec0580abd1c0796d157cdc99dbb4c55683c7a':
        raise ValueError('Original panel crop helper changed')
    spec=importlib.util.spec_from_file_location('language_panel_helper',helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    crops=[]
    with SourceCorpus(rows,teacher,cache_dir=out/'teacher-cache',input_sample_counts=counts,
            max_disk_bytes=128*1024**2,min_free_bytes=4*1024**3,max_memory_utterances=4,
            allow_prepared_source=True) as corpus:
        for index,row in enumerate(rows,1):
            record=corpus.get(row.source_id)
            current=[helper.panel_crop(record,w,29) for w in windows if w['source_id']==row.source_id]
            _verify_panel_targets(corpus,current,[row],{row.source_id:counts[row.source_id]},29)
            crops.extend(current)
            print(json.dumps({'phase':'language_targets_prepared','completed':index,'total':len(rows),
                              'source_id':row.source_id,'language':row.language}),flush=True)
        corpus_identity=corpus.identity
    if state_fingerprint(teacher.model.state_dict())!=teacher_hash or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('Teacher acquired updates or gradients')
    target_path=out/'heldout-targets.pt'
    if target_path.exists():
        saved=torch.load(target_path,map_location='cpu',weights_only=True)
        if _crop_identity(tuple(TrainingCrop(**v) for v in saved['crops']))!=_crop_identity(crops):
            raise ValueError('Immutable language targets differ')
    else:
        with target_path.open('xb') as handle:
            torch.save({'crops':[asdict(c) for c in crops]},handle)
        target_path.chmod(0o444)
    target_identity={'sha256':file_sha(target_path),'teacher':teacher.provenance,
        'teacher_state_sha256':teacher_hash,'teacher_unchanged':True,'cpu_threads':1,'device':'cpu',
        'crops':_crop_identity(crops),'corpus_identity':corpus_identity,
        'target_backend_caveat':'CPU FP32 targets. Original trend panel used CUDA FP32 targets; no cross-backend identity is claimed.'}
    immutable(out/'heldout-targets.json',target_identity)
    del teacher
    engine=RecipeV2Engine(StudentDecoder(StudentConfig(**payload['engine']['model_config'])),
                         recipe=RecipeV2Config(**payload['engine']['recipe']))
    engine.load_state_dict(payload['engine'])
    model_hash=state_fingerprint(engine.model.state_dict())
    values=evaluate_crops(engine,crops,include_signal_checks=True)
    result={'step':engine.step,'device':'cpu','cpu_threads':1,'checkpoint_sha256':receipt['sha256'],
        'panel_identity_sha256':identity['identity_sha256'],'rows':values,
        'groups':summarize_evaluation(values,metadata),'gate':waveform_gate(engine,values),
        'teacher_unchanged':True,'student_unchanged':state_fingerprint(engine.model.state_dict())==model_hash,
        'old_target_file_unchanged':file_sha(BASE/'heldout-targets.pt')==old_target_sha,
        'optimizer_updates':0,'training_panel_changed':False,'identity':identity}
    if not result['student_unchanged'] or not result['old_target_file_unchanged']:
        raise ValueError('Evaluation modified model or original targets')
    immutable(out/f'evaluation-step{engine.step:06d}.json',result)
    summary={k:v for k,v in result.items() if k not in ('rows','gate')}
    summary['final_gate_passed']=result['gate']['passed']
    summary['metrics']=[{k:v for k,v in r.items() if k!='quiet_windows'} for r in values]
    summary['student_clipped_samples']=sum(r['student_clipped_samples'] for r in values)
    summary['student_peak_abs']=max(r['student_peak_abs'] for r in values)
    immutable(out/f'summary-step{engine.step:06d}.json',summary)
    print(json.dumps({'phase':'complete','groups':result['groups'],
                      'clipped_samples':summary['student_clipped_samples']}),flush=True)


if __name__=='__main__':
    main()
