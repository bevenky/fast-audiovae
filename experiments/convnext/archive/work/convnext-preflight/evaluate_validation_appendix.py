"""Evaluate the retained 3000-step student on immutable appendix targets, CPU1."""
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.model import StudentConfig,StudentDecoder
from audiovae_student.recipe_v2 import RecipeV2Config,RecipeV2Engine
from audiovae_student.distillation_training import evaluate_crops,waveform_gate
from audiovae_student.representative_pilot import summarize_evaluation
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.restart_data import file_sha,digest
from validation_appendix import BASE,OUT,immutable

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
state=json.loads((BASE/'remediation/monitoring/state.json').read_text())
receipt=next(v for v in state['checkpoints'] if v['threshold']==3000 and v['state']=='retained')
checkpoint=Path(receipt['path'])
if file_sha(checkpoint)!=receipt['sha256']:
    raise ValueError('Retained checkpoint differs from capture receipt')
payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
if payload['engine']['step']!=receipt['step'] or receipt['step']<3000:
    raise ValueError('Retained milestone has an inconsistent step')
identity=json.loads((OUT/'ready.json').read_text())
if digest({k:v for k,v in identity.items() if k!='identity_sha256'})!=identity['identity_sha256']:
    raise ValueError('Appendix identity changed')
for name,expected in identity['files_sha256'].items():
    if file_sha(OUT/name)!=expected:
        raise ValueError('Appendix manifest or crop plan changed')
target=json.loads((OUT/'heldout-targets.json').read_text())
if file_sha(OUT/'heldout-targets.pt')!=target['sha256']:
    raise ValueError('Frozen appendix targets changed')
if payload['identity']['data']['teacher_state_sha256']!=target['teacher_state_sha256']:
    raise ValueError('Milestone teacher differs from appendix teacher')
saved=torch.load(OUT/'heldout-targets.pt',map_location='cpu',weights_only=True)
crops=tuple(TrainingCrop(**v) for v in saved['crops'])
if _crop_identity(crops)!=target['crops']:
    raise ValueError('Appendix crop tensor identities differ')
metadata=json.loads((OUT/'groups.json').read_text())
engine=RecipeV2Engine(StudentDecoder(StudentConfig(**payload['engine']['model_config'])),
                     recipe=RecipeV2Config(**payload['engine']['recipe']))
engine.load_state_dict(payload['engine'])
before=state_fingerprint(engine.model.state_dict())
rows=evaluate_crops(engine,crops,include_signal_checks=True)
result={'step':engine.step,'device':'cpu','cpu_threads':1,'checkpoint_sha256':receipt['sha256'],
    'checkpoint_receipt':receipt,'panel_identity_sha256':identity['identity_sha256'],
    'targets_sha256':target['sha256'],'rows':rows,'groups':summarize_evaluation(rows,metadata),
    'gate':waveform_gate(engine,rows),'student_unchanged':state_fingerprint(engine.model.state_dict())==before,
    'optimizer_updates':0,'teacher_inference_calls':0,'training_panel_changed':False}
if not result['student_unchanged']:
    raise ValueError('Model state changed during inference')
immutable(OUT/f'evaluation-step{engine.step:06d}.json',result)
baseline=json.loads((OUT/'evaluation-step001700.json').read_text())
if {(v['source_id'],v['start_frame'],v['samples']) for v in baseline['rows']} != {(v['source_id'],v['start_frame'],v['samples']) for v in rows}:
    raise ValueError('Comparison source windows or scored lengths differ')
comparison={}
for key,now in result['groups'].items():
    if key!='all' and not key.startswith('condition/'):
        continue
    previous=baseline['groups'][key]
    comparison[key]={'step1700':previous,'step3000':now,'delta':{
        metric:now[metric]-previous[metric] for metric in
        ('teacher_waveform_mean','teacher_mel_mean','nonquiet_cosine_mean','quiet_failed_count')}}
compact={k:v for k,v in result.items() if k not in {'rows','gate'}}
compact['comparison_to1700']=comparison
compact['final_gate_passed']=result['gate']['passed']
compact['metrics']=[{k:v for k,v in row.items() if k!='quiet_windows'} for row in rows]
compact['student_clipped_samples']=sum(row['student_clipped_samples'] for row in rows)
compact['student_peak_abs']=max(row['student_peak_abs'] for row in rows)
immutable(OUT/f'summary-step{engine.step:06d}.json',compact)
print(json.dumps({'state':'complete','step':engine.step,'groups':result['groups'],
                  'clipped_samples':compact['student_clipped_samples'],'peak':compact['student_peak_abs']}),flush=True)
