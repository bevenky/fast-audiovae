"""Bind an existing downloaded source selection to the current student's ledger."""
import argparse
from collections import Counter
import json
from pathlib import Path
from fresh_training_data import digest, sha, FreshTrainingData

CHECKPOINT = 'a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f'
SELECTION = 'f9b7ef7a9b567c350bd87435b423c2e178f110645effe98ae93a3fcf2d411e74'
KEYS = ('source_id','audio_sha256','parent_recording_id')


def main():
    p=argparse.ArgumentParser()
    for name in ('parent-plan','manifest','existing-selection','checkpoint','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    parent=json.loads(a.parent_plan.read_text());original=json.loads(a.manifest.read_text())
    old=json.loads(a.existing_selection.read_text())
    if digest({k:v for k,v in parent.items() if k!='identity_sha256'})!=parent['identity_sha256']:
        raise ValueError('Original source plan identity differs')
    if (old['identity_sha256']!=SELECTION or digest({k:v for k,v in old.items() if k!='identity_sha256'})!=SELECTION
        or old['parent_plan_identity_sha256']!=parent['identity_sha256']
        or old['parent_plan_sha256']!=sha(a.parent_plan)
        or parent['original_manifest_sha256']!=sha(a.manifest)
        or old['original_manifest_sha256']!=sha(a.manifest)
        or sha(a.checkpoint)!=CHECKPOINT):
        raise ValueError('Source selection or parent checkpoint identity differs')
    rows=old['appended_rows'][:30000]
    if len(rows)!=30000: raise ValueError('Insufficient appended sources')
    blocked={k:set(parent['blocked_identities'][k]) for k in KEYS}
    for row in parent['rows']:
        for k in KEYS: blocked[k].add(row['manifest_row'][k])
    for row in rows:
        src=row['manifest_row'];FreshTrainingData._geometry(row)
        if row['source_id']!=src['source_id'] or src['split']!='train': raise ValueError('Source split differs')
        for k in KEYS:
            if not src[k] or src[k] in blocked[k]: raise ValueError('Repeated or reserved '+k)
            blocked[k].add(src[k])
        if not Path(src['audio_path']).is_file(): raise FileNotFoundError(src['audio_path'])
    original_ids=[r['source_id'] for r in original['splits']['fit']['rows']]+parent['source_ids']
    ids=original_ids+[r['source_id'] for r in rows]
    if len(original_ids)!=30000 or len(ids)!=60000 or len(set(ids))!=60000: raise ValueError('Stream size differs')
    full_rows=original['splits']['fit']['rows']+parent['rows']+rows
    summaries={}
    indic=set('as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur'.split())
    for start in range(24000,60000,6000):
        part=full_rows[start:start+6000]
        langs=Counter(r['manifest_row']['language'].split('_')[0] for r in part)
        if indic-set(langs): raise ValueError('Missing Indic language in review interval')
        n=sum(r['valid_scored_samples'] for r in part)
        summaries[f'{start}:{start+6000}']={'sources':len(part),'languages':dict(langs),
            'datasets':dict(Counter(r['manifest_row']['dataset'] for r in part)),
            'scored_hours':n/48000/3600,
            'broad_expressive_percent':100*sum(r['valid_scored_samples'] for r in part if r['broad_expressive_source'])/n,
            'startup_sources':sum(r['start_frame']==0 for r in part)}
    plan={'version':'audiovae2_progressive_extension_5000_v1',
        'parent_plan_path':str(a.parent_plan.resolve()),'parent_plan_sha256':sha(a.parent_plan),
        'parent_plan_identity_sha256':parent['identity_sha256'],'original_manifest_sha256':sha(a.manifest),
        'teacher_source_sha256':parent['teacher_source_sha256'],'teacher_checkpoint_sha256':parent['teacher_checkpoint_sha256'],
        'starting_checkpoint_sha256':CHECKPOINT,'starting_optimizer_step':2000,'target_optimizer_step':5000,
        'appended_start_index':30000,'appended_rows':rows,'source_ids_sha256':digest(ids),
        'authorized_source_interval':[24000,60000],'gradient_accumulation':12,'execution_batch_size':1,
        'existing_selection_path':str(a.existing_selection.resolve()),'existing_selection_sha256':sha(a.existing_selection),
        'existing_selection_identity_sha256':SELECTION,'selection_policy':'First 30000 rows of the preserved stratified selection; crop geometry unchanged',
        'disjointness':dict.fromkeys(KEYS,True),'summaries':summaries,'generator_sha256':sha(__file__),
        'expressive_limit':'Dataset-level expressive labels do not establish a specific vocal event in each crop'}
    plan['identity_sha256']=digest(plan)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    small={'identity_sha256':plan['identity_sha256'],'plan_sha256':sha(a.out),'added_sources':len(rows),
        'original_stream_sources':len(original_ids),'available_through_step':5000,
        'intervals':{k:{'sources':v['sources'],'language_count':len(v['languages']),'all22_indic':True,
                     'scored_hours':v['scored_hours'],'broad_expressive_percent':v['broad_expressive_percent']} for k,v in summaries.items()},
        'estimated_new_tensor_bytes':sum(4*2624*(r['context_frames']+r['scored_frames']) for r in rows)}
    a.out.with_name('summary.json').write_text(json.dumps(small,indent=2)+'\n')
    print(json.dumps(small))


if __name__=='__main__': main()
