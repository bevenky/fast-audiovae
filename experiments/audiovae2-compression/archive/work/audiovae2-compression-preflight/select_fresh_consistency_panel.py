import json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from fresh_training_data import FreshTrainingData
P=Path('/workspace/fast-audiovae-compression-20260910-v1');S=Path('/dev/shm/fast-audiovae-compression-fresh-shards-v2')
torch.set_num_threads(1)
o=json.loads((P/'pilot-selection-v1.json').read_text());provider=FreshTrainingData(P/'fresh-source-plan-v2/plan.json',P/'pilot-selection-v1.json',{k:v['rows'] for k,v in o['splits'].items()},S)
report=json.loads((P/'training-audit5000-runtime.json').read_text());selected=[]

def annotate(index,crop):
 row=provider.plan['rows'][index];a=crop['context_frames']*1920;n=crop['valid_scored_samples'];y=crop['teacher_audio'][0,0,a:a+n].double();full=n//960;rms=y[:full*960].reshape(full,960).square().mean(-1).sqrt()
 return {'index':index,'source_id':row['source_id'],'context_frames':row['context_frames'],'context_start_frame':row['context_start_frame'],'start_frame':row['start_frame'],'valid_scored_samples':n,'scored_frames':row['scored_frames'],'partial_latent_tail_samples':n%1920,'total_rms':float(y.square().mean().sqrt()),'quiet_full_windows':int((rms<=1e-3).sum()),'nearzero_full_windows':int((rms<=1e-5).sum()),'full_windows':full,'peak':float(y.abs().max()),'shard':str(S/f'{index//300*300:06d}-{(index//300+1)*300:06d}')}
for start in (300,11700):
 rows=[annotate(start+i,crop) for i,crop in enumerate(provider.take(start,300))]
 interior=start==300
 quiet=min((r for r in rows if (r['context_start_frame']>0 if interior else r['start_frame']==0)),key=lambda r:(-r['nearzero_full_windows'],-r['quiet_full_windows'],r['total_rms']))
 selected.append({**quiet,'reason':f'{"first" if start==300 else "last"}_new_shard_quiet_{"interior" if interior else "startup"}'})
 active=max((r for r in rows if (r['start_frame']==0 if interior else r['context_start_frame']>0)),key=lambda r:r['total_rms'])
 selected.append({**active,'reason':f'{"first" if start==300 else "last"}_new_shard_active_{"startup" if interior else "interior"}'})
for label in ('Crying_and_sobbing','human_whistling_source_description'):
 ids={r['source_id'] for r in report['actual_fresh_source_mix']['labeled_sources'] if label in r['labels']}
 indices=[i for i,row in enumerate(provider.plan['rows'][:12000]) if i>=300 and row['source_id'] in ids]
 i=min(indices,key=lambda i:(provider.plan['rows'][i]['valid_scored_samples']%1920==0,provider.plan['rows'][i]['valid_scored_samples']))
 selected.append({**annotate(i,provider.take(i,1)[0]),'reason':'specific_source_label/'+label,'event_in_crop_verified':False})
provider.assert_unchanged()
result={'plan_identity_sha256':provider.identity,'plan_path':str(provider.plan_path),'shards_dir':str(S),'selected':selected,'selection_policy':'Read-only cached target RMS and geometry; first/last new shards plus scarce event-source labels. No student output used.'}
p=P/'fresh-consistency-panel.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
