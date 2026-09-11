import json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
R1=Path('/workspace/fast-audiovae-convnext-20260908-r1'); R9=Path('/workspace/fast-audiovae-convnext-20260909-r9')
sys.path.insert(0,str(R9))
from audiovae_student.data import load_manifest
from audiovae_student.comparison_data import load_comparison_plan,known_identities,_primary
from audiovae_student import acquire_fsd_vocal as fsd
out=R9/'remediation/expressive';out.mkdir(parents=True,exist_ok=True)
plans=[load_comparison_plan(R9/'data/recipe-v2'/s) for s in ['optimization','calibration']]
active={r.source_id:r for p in plans for r in p['rows']}
reserved={r.source_id:r for p in plans for r in p['reserved']}
excluded={r.source_id:r for p in plans for r in p['excluded']}
allm=[]
for d in Path('/workspace').glob('fast-audiovae-convnext-*'):
 for p in d.rglob('*.jsonl'):
  if any(x in p.parts for x in ['.train-venv','teacher-target-cache','.git','remediation']):continue
  if p.name in ['dev.jsonl','test.jsonl','reserved.jsonl','manifest.jsonl','source-manifest.jsonl','candidate-dev.jsonl']:
   try: rows=load_manifest(p)
   except (ValueError,TypeError,KeyError,json.JSONDecodeError):continue
   held=[r for r in rows if r.split in {'dev','test'}]
   if held:
    allm.append({'path':str(p),'rows':len(held),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    reserved.update({r.source_id:r for r in held})
resids=set().union(*(known_identities(r) for r in reserved.values()))
actids=set().union(*(known_identities(r,people=False) for r in active.values()))
excids=set().union(*(known_identities(r,people=False) for r in excluded.values()))
rows=load_manifest(R1/'data-emogator/prepared/train.jsonl')+load_manifest(R1/'data-emogator/prepared/dev.jsonl')
counts=Counter();secs=Counter(); eligible=[]
for r in rows:
 reason=('existing_dev' if r.split!='train' else 'active_source' if known_identities(r,people=False)&actids else 'reserved_identity' if known_identities(r)&resids else 'excluded_source' if known_identities(r,people=False)&excids else 'missing_file' if not Path(r.audio_path).is_file() else 'eligible_unused')
 counts[reason]+=1;secs[reason]+=r.duration_seconds
 if reason=='eligible_unused':eligible.append(r)
fsd.TARGET_LABELS=fsd.TARGET_LABELS|{'Whispering','Breathing','Yell'}
selection=fsd.read_selection(R1/'assets/fsd50k-metadata')
existing=set(active)|set(reserved)|set(excluded)
# Existing prepared train sources outside active plan must not be downloaded twice.
prepared=[]
for d in [R1/'data-fsd-vocal',R1/'data-human-whistling',Path('/workspace/fast-audiovae-convnext-20260909-r5/data-expressive-topup-v1')]:
 for p in d.rglob('train.jsonl'):
  try: prepared+=load_manifest(p)
  except (ValueError,TypeError,json.JSONDecodeError):pass
existing.update(r.source_id for r in prepared)
fcounts=Counter(); frows=[]
for x in selection['selected']:
 ids={('source',x['source_id']),('parent',x['parent_recording_id']),('session',x['session_id'])}
 reason=('already_manifested' if x['source_id'] in existing else 'heldout_contributor' if ids&resids else 'active_or_excluded_source' if ids&(actids|excids) else 'fresh_metadata_candidate')
 fcounts[reason]+=1
 if reason=='fresh_metadata_candidate':frows.append(x)
report={'emogator_counts':dict(counts),'emogator_hours':{k:v/3600 for k,v in secs.items()},'emogator_current_primary_example':_primary(eligible[0],{}) if eligible else None,'emogator_eligible_speakers':len({r.speaker_id for r in eligible}),'historical_holdout_manifests':allm,'reserved_sources':len(reserved),'active_sources':len(active),'excluded_sources':len(excluded),'fsd_counts':dict(fcounts),'fsd_candidate_labels':dict(Counter(y for x in frows for y in x['target_labels'])),'fsd_candidates':frows}
(out/'inventory.json').write_text(json.dumps(report,indent=2))
(out/'candidate-emogator.jsonl').write_text(''.join(json.dumps(r.to_dict())+'\n' for r in eligible))
(out/'reserved-all.jsonl').write_text(''.join(json.dumps(r.to_dict())+'\n' for r in reserved.values()))
(out/'fsd-candidate-selection.json').write_text(json.dumps({**selection,'selected':frows,'selected_clip_count':len(frows)},indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'historical_holdout_manifests','fsd_candidates'}},indent=2))
