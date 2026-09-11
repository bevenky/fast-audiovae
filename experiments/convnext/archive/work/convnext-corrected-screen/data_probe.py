import sys,json,collections
from pathlib import Path
B=Path('/workspace/fast-audiovae-convnext-20260909-r9')
sys.path.insert(0,str(B/'remediation/fusion-code'))
from audiovae_student.continuation_data import load_continuation_plan
r=json.loads((B/'remediation/fusion-screen/parent-receipt.json').read_text());p=load_continuation_plan(r['plan_path'])
print('PLAN',len(p['rows']),len(p['windows']),r['cursor'])
for label,ws in [('all',p['windows']),('remaining',p['windows'][r['cursor']:])]:
 c=collections.defaultdict(lambda:[0,0,set()])
 for w in ws:
  if w.condition not in (None,'speech','emotional_nonverbal'):
   v=c[w.condition];v[0]+=1;v[1]+=w.valid_input_samples16k/16000;v[2].add(w.source_id)
 print(label,json.dumps({k:[x[0],round(x[1],2),len(x[2])] for k,x in c.items()}))
print('METADATA',str(p['metadata'])[:2500]);print('ROWEXAMPLES',[x.to_dict() for x in p['rows'] if 'whistle' in x.dataset][:3]); print('EXCLUDED',len(p['excluded']), 'RESERVED',len(p['reserved']))
