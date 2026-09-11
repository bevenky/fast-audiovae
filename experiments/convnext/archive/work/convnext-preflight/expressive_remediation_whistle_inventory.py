import json,sys,re
from pathlib import Path
R1=Path('/workspace/fast-audiovae-convnext-20260908-r1');R9=Path('/workspace/fast-audiovae-convnext-20260909-r9');sys.path.insert(0,str(R9))
from audiovae_student.data import load_manifest
from audiovae_student.comparison_data import load_comparison_plan,known_identities
from audiovae_student.acquire_fsd_vocal import canonical_uploader
plans=[load_comparison_plan(R9/'data/recipe-v2'/s) for s in ['optimization','calibration']]
rs=[r for p in plans for k in ['rows','reserved','excluded'] for r in p[k]]+load_manifest(R9/'remediation/expressive/reserved-all.jsonl')
ids={r.source_id for r in rs};respeople=set().union(*(known_identities(r) for r in rs))
old=json.loads((R1/'data-human-whistling/prepared/provenance/selection.json').read_text()); ids.update(x['source_id'] for x in old['selected']);forbidden=set(old['forbidden_fsd_clip_ids'])
rows=json.loads((R1/'data-human-whistling/research/hf-whistle-candidates.json').read_text());out=[]
for x in rows:
 i=str(x['freesound_id']);text=' '.join([str(x.get('title','')),str(x.get('description','')),str(x.get('tags',[]))])
 if 'freesound:'+i in ids or i in forbidden or ('session','freesound:uploader:'+canonical_uploader(x['username'])) in respeople:continue
 if re.search(r'human|mouth|person|lips|whistling myself|myself whistling|me whistling|whistling voice',text,re.I) and not re.search(r'bird|kettle|train whistle|steam|flute|referee|synth|slide whistle|tea pot|teapot|bottle|engine|dog',text,re.I):out.append(x)
(R9/'remediation/expressive/whistle-research-candidates.json').write_text(json.dumps(out,indent=2))
print(json.dumps([{'id':x['freesound_id'],'title':x['title'],'description':x['description'],'license':x['license'],'uploader':x['username'],'shard':x['shard'],'row_group':x['row_group']} for x in out],indent=2))
