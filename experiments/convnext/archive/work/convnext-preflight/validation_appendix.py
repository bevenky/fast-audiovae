"""Create and evaluate a separate immutable whisper/breathing development panel.

Runs on the Runpod CPU, never changes training or the existing trend panel.
Labels are provenance labels, not an assertion of independently verified content.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch

from audiovae_student.comparison_data import load_comparison_plan, assert_comparison_disjoint
from audiovae_student.data import load_manifest, validate_manifest
from audiovae_student.restart_data import canonical, digest, file_sha
from audiovae_student.teacher import FrozenAudioVAE2
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.model import StudentDecoder, StudentConfig
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine
from audiovae_student.distillation_training import evaluate_crops, waveform_gate
from audiovae_student.representative_pilot import summarize_evaluation, _verify_panel_targets

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
OUT = BASE / 'remediation/validation-appendix'
SELECTED = {
    'freesound:27895': ('Breathing', '8447425da98edd1b017d326458b08a9a522ce29429f33aead437cc293d7baef1'),
    'freesound:27896': ('Breathing', '529906c268fd220d079963527fb8b3611ecf95e52b7f3f18e5b2e28478f8de5d'),
    'freesound:372237': ('Breathing', '5a6f0ae1b17119232e08b12b466f526eadd256b8451cefd4c04173346e89f6e4'),
    'freesound:183929': ('Whispering', '3d67dd77d881e8b856707f74f6f9757eb05b8e6a21287b79aafb7bd040130955'),
    'freesound:183934': ('Whispering', 'da7d7f77c9b71e92f2a348197d11e2ad801a6356c83299fb685a04ecd4d97230'),
    'freesound:167448': ('Whispering', 'ce69503c157fbc2cbfc67a4eb8cd367da65a49293f399432ee44e7289ccdb542'),
    'freesound:264051': ('Whispering', '09632e2aafba259f082114febe76f8be7a21e786671bbe157213fad4e63e4f41'),
}


def immutable(path, value, *, jsonl=False):
    data = b''.join(canonical(row) for row in value) if jsonl else canonical(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Immutable artifact changed: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(data)
        path.chmod(0o444)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    plan, calibration = [load_comparison_plan(BASE/'data/recipe-v2'/name)
                         for name in ('optimization', 'calibration')]
    training = {row.source_id: row for row in plan['rows'] + calibration['rows']}
    candidates, origins = {}, {}
    source_manifests = [BASE/'data/recipe-v2/optimization/reserved.jsonl',
        OLD/'expanded-pilot/corpus/candidate-dev.jsonl', OLD/'data-fsd-vocal/prepared/dev.jsonl',
        Path('/workspace/fast-audiovae-convnext-20260909-r5/data-expressive-topup-v1/prepared/dev.jsonl'),
        OLD/'data-expressive-originals/prepared/dev.jsonl']
    for source_manifest in source_manifests:
        if not source_manifest.exists():
            continue
        for row in load_manifest(source_manifest):
            if row.source_id in SELECTED and row.split == 'dev' and row.audio_sha256 == SELECTED[row.source_id][1]:
                if row.source_id not in candidates:
                    candidates[row.source_id] = row
                    origins[row.source_id] = str(source_manifest)
    if set(candidates) != set(SELECTED):
        raise ValueError('Expected seven reserved files are missing')
    rows = [candidates[key] for key in sorted(candidates)]
    validate_manifest(rows)
    assert_comparison_disjoint(training.values(), rows)
    if any(row.split != 'dev' or row.audio_sha256 != SELECTED[row.source_id][1] for row in rows):
        raise ValueError('Reserved split or source bytes identity changed')
    checkpoint = BASE/'audit-step1000/checkpoint-step001700.pt'
    expected_checkpoint = 'e2d4770eac1ac2f42e1812e5094cec2587597a08eb20c5927078bf3fb3b7386f'
    if file_sha(checkpoint) != expected_checkpoint:
        raise ValueError('Preserved audit checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    old_panel_ids = {row['source_id'] for row in payload['identity']['data']['heldout']['rows']}
    if old_panel_ids.intersection(candidates):
        raise ValueError('Appendix repeats an existing trend-panel source')
    old_target_sha = file_sha(BASE/'heldout-targets.pt')
    if old_target_sha != '3349f715c9ae4c4b7ac859cde935cdc949395bfc0e15915ad759c8e547a4dd2e':
        raise ValueError('Pinned original targets changed')
    counts, checks, windows, metadata = {}, [], [], {}
    for row in rows:
        raw = Path(row.audio_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.audio_sha256:
            raise ValueError(f'Audio hash mismatch: {row.source_id}')
        audio = read_native_16k(raw, row)
        count = audio.shape[-1]
        if not torch.isfinite(audio).all() or abs(count/16000-row.duration_seconds) > 1/16000+1e-10:
            raise ValueError('Invalid decoded audio or complete-file duration')
        counts[row.source_id] = count
        checks.append({'source_id':row.source_id, 'sha256_verified':row.audio_sha256,
            'decoded_samples16k':count, 'decoded_seconds':count/16000,
            'prepared_mono16k':True, 'finite':True, 'source_peak_abs':float(audio.abs().max()),
            'speaker_id':row.speaker_id, 'session_id':row.session_id,
            'parent_recording_id':row.parent_recording_id, 'split':row.split,
            'source_split':row.source_split, 'dataset':row.dataset, 'license':row.license,
            'label':SELECTED[row.source_id][0], 'content_verified_by_listening':False})
        metadata[row.source_id] = {'dataset':row.dataset,'language':row.language,
            'condition':SELECTED[row.source_id][0], 'coverage_group':'event:'+SELECTED[row.source_id][0]}
        # The short utterance-start crop exposes startup artifacts. A second,
        # distinct crop includes available real history and ends at the tail.
        for start, frames in ((0,16), (max(0, math.ceil(count/640)-64),64)):
            if any(w['source_id']==row.source_id and w['start_frame']==start for w in windows):
                continue
            valid = min(frames*640, count-start*640)
            if valid*3 < 4096:
                raise ValueError('Appendix crop is shorter than the established loss contract')
            windows.append({'source_id':row.source_id,'start_frame':start,'scored_frames':frames,
                            'valid_input_samples16k':valid,'valid_output_samples48k':valid*3})
    immutable(OUT/'manifest.jsonl', [row.to_dict() for row in rows], jsonl=True)
    immutable(OUT/'input-sample-counts.json', counts)
    immutable(OUT/'groups.json', metadata)
    immutable(OUT/'windows.jsonl', windows, jsonl=True)
    immutable(OUT/'source-checks.json', checks)
    identity = {'state':'ready','panel':'reserved-whisper-breathing-appendix-v1','rows':len(rows),
        'windows':len(windows),'source_seconds':sum(counts.values())/16000,
        'optimization_plan_identity':plan['identity']['identity_sha256'],
        'calibration_plan_identity':calibration['identity']['identity_sha256'],
        'entire_training_and_calibration_windows_checked':len(plan['windows'])+len(calibration['windows']),
        'training_source_count_checked':len(training),
        'split_source_hash_parent_and_known_people_disjoint':True,
        'old_panel_source_count':len(old_panel_ids),'old_panel_source_overlap':0,
        'old_target_sha256':old_target_sha,
        'source_manifest_sha256':{str(p):file_sha(p) for p in source_manifests if p.exists()},
        'selected_row_manifest_origins':origins,
        'context_frames':29,'sample_rate_in':16000,'sample_rate_out':48000,
        'label_evidence':'FSD50K source labels from the prior reserved inventory, not listened verification; uploader/session does not establish speaker identity',
        'deduplication_limit':'Known canonical identity and exact file hashes, not acoustic fingerprints or unknown speaker/session relationships',
        'files_sha256':{name:file_sha(OUT/name) for name in ('manifest.jsonl','input-sample-counts.json','groups.json','windows.jsonl','source-checks.json')}}
    identity['identity_sha256'] = digest(identity)
    immutable(OUT/'ready.json', identity)
    print(json.dumps({'phase':'appendix_verified','rows':len(rows),'crops':len(windows),
                      'source_seconds':identity['source_seconds']}), flush=True)
    del plan, calibration, training
    teacher = FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cpu')
    teacher = _target_teacher(teacher, rows[0])
    before = state_fingerprint(teacher.model.state_dict())
    if before != payload['identity']['data']['teacher_state_sha256']:
        raise ValueError('Frozen teacher weights differ from the live student teacher identity')
    warm = read_native_16k(Path(rows[0].audio_path).read_bytes(), rows[0])
    with torch.jit.optimized_execution(True):
        for _ in range(2):
            teacher.decode(teacher.encode(warm))
    del warm
    helper_path = Path('/workspace/fast-audiovae-convnext-20260909-r5/run_representative_stage.py')
    if file_sha(helper_path) != '72c249809abeb889a21c0391c65ec0580abd1c0796d157cdc99dbb4c55683c7a':
        raise ValueError('Established development crop helper changed')
    spec=importlib.util.spec_from_file_location('appendix_panel_helpers',helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    crops=[]
    with SourceCorpus(rows,teacher,cache_dir=OUT/'teacher-cache',input_sample_counts=counts,
            max_disk_bytes=64*1024**2,min_free_bytes=4*1024**3,max_memory_utterances=7,
            allow_prepared_source=True) as corpus:
        for row in rows:
            record=corpus.get(row.source_id)
            crops.extend(helper.panel_crop(record,w,29) for w in windows if w['source_id']==row.source_id)
            print(json.dumps({'phase':'targets_prepared','source_id':row.source_id,
                              'input_samples':record.input_samples}),flush=True)
        _verify_panel_targets(corpus,crops,rows,counts,29)
        corpus_identity=corpus.identity
    if state_fingerprint(teacher.model.state_dict()) != before or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('Teacher changed or acquired gradients')
    target_path=OUT/'heldout-targets.pt'
    if target_path.exists():
        saved=torch.load(target_path,map_location='cpu',weights_only=True)
        if _crop_identity(tuple(helper.TrainingCrop(**v) for v in saved['crops'])) != _crop_identity(crops):
            raise ValueError('Immutable appendix targets differ')
    else:
        with target_path.open('xb') as handle:
            torch.save({'crops':[asdict(c) for c in crops]},handle)
        target_path.chmod(0o444)
    target_report={'teacher':teacher.provenance,'teacher_state_sha256':before,'teacher_unchanged':True,
        'device':'cpu','cpu_threads':1,'raw_mu_same_teacher_contract':True,
        'target_backend_difference':'Appendix targets prepared CPU FP32; original trend targets were prepared CUDA FP32. No same-file cross-backend comparison is asserted.',
        'corpus_identity':corpus_identity,'crops':_crop_identity(crops),'sha256':file_sha(target_path)}
    immutable(OUT/'heldout-targets.json',target_report)
    del teacher
    engine=RecipeV2Engine(StudentDecoder(StudentConfig(**payload['engine']['model_config'])),
                         recipe=RecipeV2Config(**payload['engine']['recipe']))
    engine.load_state_dict(payload['engine'])
    fingerprint=state_fingerprint(engine.model.state_dict())
    values=evaluate_crops(engine,crops,include_signal_checks=True)
    result={'step':engine.step,'device':'cpu','cpu_threads':1,'checkpoint_sha256':expected_checkpoint,
        'panel_identity_sha256':identity['identity_sha256'],'rows':values,
        'groups':summarize_evaluation(values,metadata),'gate':waveform_gate(engine,values),
        'teacher_unchanged':True,'student_unchanged':state_fingerprint(engine.model.state_dict())==fingerprint,
        'old_target_file_unchanged':file_sha(BASE/'heldout-targets.pt')==old_target_sha,
        'optimizer_updates':0,'training_panel_changed':False,
        'label_limit':identity['label_evidence']}
    if not result['student_unchanged'] or not result['old_target_file_unchanged']:
        raise ValueError('Evaluation changed immutable model or old validation targets')
    immutable(OUT/'evaluation-step001700.json',result)
    summary={k:v for k,v in result.items() if k not in {'rows','gate'}}
    summary['source_checks']=checks
    summary['final_gate_passed']=result['gate']['passed']
    summary['metrics']=[{k:v for k,v in row.items() if k not in {'quiet_windows'}} for row in values]
    summary['source_seconds']=identity['source_seconds']
    immutable(OUT/'summary.json',summary)
    print(json.dumps({'phase':'complete','groups':result['groups'],'model_unchanged':result['student_unchanged']}),flush=True)


if __name__ == '__main__':
    main()
