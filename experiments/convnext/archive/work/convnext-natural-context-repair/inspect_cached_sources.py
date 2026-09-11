import json,os
from pathlib import Path
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
b=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
r=json.loads((b/'corrected-screen/target-cache.json').read_text())
d=torch.load(r['path'],map_location='cpu',weights_only=True,mmap=True)
rows=[]
for c in d['heldout']:
 meta=d['metadata'].get(c['source_id'],{})
 valid=c['context_frames']*1920+c['valid_scored_samples']
 rows.append({'source_id':c['source_id'],'metadata':meta,'start_frame':c['start_frame'],'context_start_frame':c['context_start_frame'],'context_frames':c['context_frames'],'scored_frames':c['scored_frames'],'valid_scored_samples':c['valid_scored_samples'],'valid_frames':valid//1920,'latent_shape':list(c['latents'].shape),'reference16k_shape':list(c['reference16k'].shape) if c['reference16k'] is not None else None})
summary=defaultdict(lambda:{'crops':0,'with_reference':0,'ge56':0,'ge56_with_reference':0,'sources':set(),'ranges':set()})
for c in rows:
 g=summary[c['metadata'].get('condition','missing')];g['crops']+=1;g['with_reference']+=c['reference16k_shape'] is not None;g['ge56']+=c['valid_frames']>=56;g['ge56_with_reference']+=c['valid_frames']>=56 and c['reference16k_shape'] is not None;g['sources'].add(c['source_id']);g['ranges'].add((c['context_frames'],c['scored_frames'],c['valid_frames'],c['latent_shape'][-1]))
for v in summary.values():v['sources']=sorted(v['sources']);v['ranges']=sorted(v['ranges'])
print(json.dumps({'receipt':r,'summary':dict(summary),'crops':rows},indent=2))
