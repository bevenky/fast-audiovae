"""Read-only audit of an already completed pruning cut and its saved events."""
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import torch

root=Path('/tmp/fast-audiovae-progressive-pruning-v1')
out=root/'cut1-384-256'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
load=lambda p:json.loads(Path(p).read_text())
completed=load(out/'completed.json')
launch=load(out/'launch.json')
receipt=load(out/'checkpoint-step1000.json')
rows=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
assert completed['status']=='awaiting_review' and completed['failure'] is None
assert completed['cut_updates']==1000 and len(rows)==1000
assert [r['step'] for r in rows]==list(range(1,1001))
assert completed['frozen_state_preserved'] and completed['original_files_preserved']
assert receipt['frozen_state_preserved'] and launch['trainable_stages']==[2,3,4]
assert launch['automatic_next_cut'] is False and completed['automatic_next_cut'] is False
assert launch['execution_batch_size']==1 and launch['gradient_accumulation']==12
source_ids=[sid for row in rows for sid in row['source_ids']]
assert source_ids==launch['source_ids'] and len(source_ids)==len(set(source_ids))==12000
assert all(len(row['source_ids'])==12 for row in rows)
assert all(row['unique_sources']==12*row['step'] for row in rows)
teacher_checks=[check for row in rows for check in row['teacher_cache_checks']]
assert len(teacher_checks)==12000 and all(c['allclose_original_tolerance'] for c in teacher_checks)
assert all(math.isfinite(row[key]) for row in rows for key in ('total','waveform','mel','feature','step_seconds'))
checkpoint=out/'checkpoint-step1000.pt'
checkpoint_sha=sha(checkpoint)
assert checkpoint_sha==receipt['checkpoint_sha256']==completed['last_checkpoint_sha256']
state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert state['cut_updates']==1000 and state['sources_seen']==source_ids
assert len(state['selection']['stage2_indices'])==384 and len(state['selection']['stage3_indices'])==256
assert state['coefficients']==launch['coefficients'] and state['scored_samples_this_cut']==completed['scored_samples']
optimizer=state['optimizer']
assert len(optimizer['state'])==90
assert all(float(value['step'])==1000 for value in optimizer['state'].values())
assert all(g['lr']==3e-5 and tuple(g['betas'])==(.9,.99) and g['eps']==1e-8 and g['weight_decay']==0 for g in optimizer['param_groups'])
assert all(torch.isfinite(t).all().item() for t in state['group'].values())
assert all(torch.isfinite(value[key]).all().item() for value in optimizer['state'].values() for key in ('exp_avg','exp_avg_sq'))
source_checks={path:sha(path)==expected for path,expected in launch['protected'].items() if Path(path).suffix in ('.py','.json')}
assert all(source_checks.values())
process=load(root/'process-launch.json')
cmdfile=Path('/proc')/str(process['pid'])/'cmdline'
active=cmdfile.exists() and b'progressive_train.py' in cmdfile.read_bytes()
assert not active
with urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags',timeout=10) as response:tags=json.load(response)
runs=[name for name,values in tags.items() if 'overview/Percent' in values]
assert len(runs)==13
dashboard={}
for run in runs:
    url='http://127.0.0.1:8888/data/plugin/scalars/scalars?'+urlencode({'run':run,'tag':'overview/Percent'})
    with urlopen(url,timeout=10) as response:values=json.load(response)
    assert values and values[-1][1]==1000
    dashboard[run]={'last_step':values[-1][1],'last_value':values[-1][2]}
result={'passed':True,'status':completed['status'],'updates':1000,'distinct_sources':12000,
    'teacher_cache_comparisons':12000,'all_teacher_cache_comparisons_passed':True,
    'frozen_state_preserved_by_runner':True,'original_files_preserved_by_runner':True,
    'independently_rehashed_source_and_manifest_files':len(source_checks),
    'checkpoint_sha256':checkpoint_sha,'joint_optimizer_parameter_states':90,'all_adam_steps':1000,
    'weights_moments_and_losses_finite':True,'source_ledger_matches_checkpoint_and_launch':True,
    'trainer_exited':True,'elapsed_seconds_including_final_review':completed['elapsed_seconds'],
    'scored_audio_hours':completed['scored_samples']/48000/3600,
    'dashboard_metric_runs_at_step1000':dashboard,
    'automatic_next_cut':False,
    'scope':'Saved-state and event audit only. No training, inference, benchmarks, or model changes.'}
(root/'completion-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
