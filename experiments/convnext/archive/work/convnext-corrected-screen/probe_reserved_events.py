import sys,json,collections
from pathlib import Path
B=Path('/workspace/fast-audiovae-convnext-20260909-r9');sys.path.insert(0,str(B/'remediation/fusion-code'))
from audiovae_student.continuation_data import load_continuation_plan
from audiovae_student.comparison_data import known_identities
r=json.loads((B/'remediation/fusion-screen/parent-receipt.json').read_text());p=load_continuation_plan(r['plan_path'])
touched=set().union(*(known_identities(x) for x in p['rows']))
from audiovae_student.data import ManifestRow
for e in p['ledger'].entries.values():touched.update(known_identities(ManifestRow.from_dict(e['row'])))
selected=[];summary=collections.Counter()
for row in p['reserved']:
 if not row.dataset.startswith('fsd50k'):continue
 rec=json.loads(Path(row.access_record).read_text());labels=rec.get('selected_metadata',{}).get('target_labels',[])
 summary.update(labels)
 if set(labels)&{'Crying_and_sobbing','Giggle','Shout','Yell'}:
  selected.append({'row':row.to_dict(),'labels':labels,'overlap_training_identity':bool(known_identities(row)&touched),'metadata':rec.get('selected_metadata',{})})
out={'summary':dict(summary),'candidates':selected};path=B/'remediation/corrected-reserved-event-candidates.json';path.write_text(json.dumps(out,indent=2));print(json.dumps({'summary':dict(summary),'candidates':[{'source':v['row']['source_id'],'labels':v['labels'],'overlap':v['overlap_training_identity']} for v in selected],'path':str(path)}))
