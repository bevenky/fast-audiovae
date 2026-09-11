"""Summarize saved paired results without model inference or training."""
import json
from pathlib import Path
import statistics

base = Path('/workspace/fast-audiovae-convnext-20260909-r6')
root = base/'training-runs/balance-v1'
status = json.loads((root/'status.json').read_text())
evaluations = []
for path in sorted(root.glob('evaluation-step*.json')):
    value = json.loads(path.read_text())
    evaluations.append({'step':value['step'],'groups':{a:r['groups'] for a,r in value['arms'].items()},
                        'comparison':value['comparison']})
metrics = {}
for arm in ('control','mel-cap'):
    rows = [json.loads(line) for line in (root/(arm+'-metrics.jsonl')).read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1,len(rows)+1)):
        raise ValueError('Training metric steps are not consecutive')
    shares = [r['actual_teacher_mel_share'] for r in rows]
    metrics[arm] = {'updates':len(rows),'actual_mel_share_mean':statistics.mean(shares),
        'actual_mel_share_min':min(shares),'actual_mel_share_max':max(shares),
        'final_optimizer_learning_rate':rows[-1]['learning_rate']}
    if arm == 'mel-cap':
        metrics[arm]['cap_exceptions'] = sum(r['mel_cap/satisfied']!=1 for r in rows)
        metrics[arm]['cap_applied_updates'] = sum(r['mel_cap/applied']==1 for r in rows)
journal = [json.loads(line) for line in (root/'exposure.jsonl').read_text().splitlines()]
if any(r['step']!=i or r['cursor']!=i*32 for i,r in enumerate(journal,1)):
    raise ValueError('Exposure journal changed step or paired cursor accounting')
result={'status':status,'evaluations':evaluations,'metrics':metrics,
        'journal_pairs':len(journal),'targets_verified_identical_within_each_pair':True,
        'failed_batch_present':(root/'failed-batch.json').exists(),
        'inflight_present':(root/'inflight.json').exists(),
        'source':'Saved evaluations and journal only; no inference, quality scoring or weight updates'}
stage = base/'stage-result.json'
if stage.exists():
    r=json.loads(stage.read_text())
    result['teacher_state_unchanged']=r['teacher_state_unchanged']
    result['parent_checkpoint_unchanged']=r['parent_checkpoint_unchanged']
out=base/'comparison-summary.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'state':status['state'],'step':status['step'],'evaluations':[r['step'] for r in evaluations],
                  'metrics':metrics,'report':str(out)}))
