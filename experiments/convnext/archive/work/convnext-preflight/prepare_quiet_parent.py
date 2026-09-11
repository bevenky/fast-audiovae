"""Preserve the selected r6 control as a distinct immutable r7 parent."""
import json
from pathlib import Path
import shutil
import torch
from audiovae_student.restart_data import file_sha, canonical
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.training import _atomic_save

OLD=Path('/workspace/fast-audiovae-convnext-20260909-r6')
BASE=Path('/workspace/fast-audiovae-convnext-20260909-r7')
source=OLD/'training-runs/balance-v1/latest.pt'
result=json.loads((OLD/'stage-result.json').read_text())
if (result['step']!=500 or result['state']!='completed_awaiting_review' or
    not result['teacher_state_unchanged'] or not result['parent_checkpoint_unchanged'] or
    result['comparison']['numerical_promotion_checks_passed']):
    raise ValueError('Expected completed comparison rejecting the mel cap')
state=torch.load(source,map_location='cpu',weights_only=True)
if state['engines']['control']['step']!=500 or state['sampler']['cursor']!=16000:
    raise ValueError('Selected parent exposure is incomplete')
if (source.parent/'inflight.json').exists():
    raise ValueError('Uncertain partial exposure')
source_sha=file_sha(source)
selection={'source_checkpoint_sha256':source_sha,'selected_arm':'control',
 'selected_engine_sha256':state_fingerprint(state['engines']['control']),
 'completed_evaluation_sha256':file_sha(source.parent/'evaluation-step000500.json'),
 'stage_result_sha256':file_sha(OLD/'stage-result.json'),
 'source_exposure_sha256':file_sha(source.parent/'exposure.jsonl'),
 'decision':'Reject cap: waveform improvement below10% and mel regression above5%.'}
BASE.mkdir(exist_ok=True)
target=BASE/'selected-parent.pt'
if target.exists():
    raise ValueError('Parent already prepared; inspect instead of overwriting')
payload={'engine':state['engines']['control'],'rng':state['rng'],'identity':state['identity'],
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
(BASE/'parent-selection.json').write_bytes(canonical(selection))
print(json.dumps(selection))
