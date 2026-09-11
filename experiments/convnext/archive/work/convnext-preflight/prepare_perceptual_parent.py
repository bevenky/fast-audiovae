"""Preserve the selected r6 control as a distinct immutable r7 parent."""
import json
from pathlib import Path
import shutil
import torch
from audiovae_student.restart_data import file_sha, canonical
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.training import _atomic_save

OLD=Path('/workspace/fast-audiovae-convnext-20260909-r7')
BASE=Path('/workspace/fast-audiovae-convnext-20260909-r8')
source=OLD/'training-runs/quiet-phase-v1/latest.pt'
result=json.loads((OLD/'stage-result.json').read_text())
if (result['step']!=250 or result['state']!='completed_awaiting_review' or
    not result['teacher_state_unchanged'] or not result['parent_checkpoint_unchanged'] or
    result['comparison']['numerical_promotion_checks_passed']):
    raise ValueError('Expected completed comparison rejecting the quiet-phase penalty')
state=torch.load(source,map_location='cpu',weights_only=True)
if state['engines']['control']['step']!=250 or state['sampler']['cursor']!=8000:
    raise ValueError('Selected parent exposure is incomplete')
if (source.parent/'inflight.json').exists():
    raise ValueError('Uncertain partial exposure')
source_sha=file_sha(source)
original=state['engines']['control']
phase=original.get('quiet_phase')
if not phase or phase['config']['gradient_share']!=0 or phase['activation'] is not None:
    raise ValueError('Only the disabled control extension may be removed')
plain={k:v for k,v in original.items() if k!='quiet_phase'}
selection={'source_checkpoint_path':str(source),'source_checkpoint_sha256':source_sha,'selected_arm':'control',
 'selected_engine_sha256':state_fingerprint(plain),'source_engine_sha256':state_fingerprint(original),
 'transition':'Remove only disabled quiet_phase metadata; all model/optimizer/RNG/norm values remain exact',
 'completed_evaluation_sha256':file_sha(source.parent/'evaluation-step000250.json'),
 'stage_result_sha256':file_sha(OLD/'stage-result.json'),
 'source_exposure_sha256':file_sha(source.parent/'exposure.jsonl'),
 'decision':'Retain control: quiet-phase candidate failed the predeclared midpoint/final criteria.'}
BASE.mkdir(exist_ok=True)
target=BASE/'selected-parent.pt'
if target.exists():
    raise ValueError('Parent already prepared; inspect instead of overwriting')
payload={'engine':plain,'rng':state['rng'],'identity':state['identity'],
 'selection':selection,'sampler':state['sampler'],'teacher_coverage':state['teacher_coverage']}
_atomic_save(payload,target)
check=torch.load(target,map_location='cpu',weights_only=True)
if state_fingerprint(check['engine'])!=selection['selected_engine_sha256'] or file_sha(source)!=source_sha:
    raise ValueError('Parent extraction changed state')
for name in ('heldout-targets.pt','heldout-targets.json'):
    shutil.copyfile(OLD/name,BASE/name)
shutil.copytree(OLD/'teacher-cache-dev',BASE/'teacher-cache-dev')
selection['parent_checkpoint_sha256']=file_sha(target)
selection['heldout_targets_sha256']=file_sha(BASE/'heldout-targets.pt')
(BASE/'selected-parent.json').write_bytes(canonical(selection))
print(json.dumps(selection))
