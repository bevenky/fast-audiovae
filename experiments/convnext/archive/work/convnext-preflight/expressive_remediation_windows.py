"""Immutable candidate intervals only, not an active training schedule."""
import json,sys,hashlib,shutil,gzip
from pathlib import Path
from dataclasses import replace
from collections import Counter
R9=Path('/workspace/fast-audiovae-convnext-20260909-r9');ROOT=R9/'remediation/expressive';sys.path.insert(0,str(R9))
from audiovae_student.data import load_manifest,validate_manifest
from audiovae_student.restart_data import windows_for,canonical,ConsumedLedger,digest,verify_windows
from audiovae_student.comparison_data import known_identities,load_comparison_plan,verify_comparison_windows
from audiovae_student.acquire_expressive import StorageBudget
budget=StorageBudget(ROOT,400*1024**2,4*1024**3)
ready=json.loads((ROOT/'ready.json').read_text())
for name,sha in ready['manifest_sha256'].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Staged receipt changed '+name)
def load_pretty_stream(path):
 text=path.read_text();decoder=json.JSONDecoder();pos=0;result=[]
 while pos<len(text):
  while pos<len(text) and text[pos].isspace():pos+=1
  if pos==len(text):break
  value,end=decoder.raw_decode(text,pos);result.append(value);pos=end
 from audiovae_student.data import ManifestRow
 return [ManifestRow.from_dict(r) for r in result]
rows=load_pretty_stream(ROOT/'train-candidates.jsonl');conditions=json.loads((ROOT/'fresh-conditions.json').read_text());emos=json.loads((ROOT/'emogator-conditions.json').read_text())['by_source']
plans=[load_comparison_plan(R9/'data/recipe-v2'/x) for x in ['optimization','calibration']]
old=[r for p in plans for r in p['rows']];historical=load_manifest(ROOT/'reserved-all.jsonl');freshdev=load_pretty_stream(ROOT/'fresh-dev.jsonl')
excluded=[r for p in plans for r in p['excluded']]+old
counts={r.source_id:round(r.duration_seconds*16000) for r in rows}; byid={r.source_id:r for r in rows};allwindows=[];bycond=Counter();clipcond=Counter();wholecond=Counter();excludedshort=[]
for r in rows:
 condition='emotional_nonverbal' if r.source_id in emos else conditions[r.source_id][0]
 ws=[replace(w,condition=condition) for w in windows_for(r,counts[r.source_id],minimum=3040)]
 if not ws:excludedshort.append({'source_id':r.source_id,'input_samples':counts[r.source_id],'reason':'shorter_than_3040_scored_input_samples'})
 allwindows.extend(ws);bycond[condition]+=sum(w.valid_input_samples16k for w in ws);clipcond[condition]+=len(ws);wholecond[condition]+=counts[r.source_id]
 if any(w.scored_frames>64 or w.valid_output_samples48k!=3*w.valid_input_samples16k for w in ws):raise ValueError('Scored window contract changed')
ids={w.source_id for w in allwindows};selected={k:v for k,v in byid.items() if k in ids};selectedcounts={k:counts[k] for k in ids}
body={'format_version':1,'input_sample_rate':16000,'sources':[],'provenance':{'scope':'New sources only, entire original r9 optimization and calibration sources excluded'},'exposure_policy':'Each staged scored interval once only; no active schedule mutation'}
ledger=ConsumedLedger({**body,'identity_sha256':digest(body)})
verify_comparison_windows(allwindows,selected,selectedcounts,ledger,reserved_rows=historical+freshdev,excluded_sources=excluded)
trainpath=ROOT/'train-candidates-v2.jsonl';devpath=ROOT/'fresh-dev-v2.jsonl'
budget.write(trainpath,b''.join(canonical(r.to_dict()) for r in rows));budget.write(devpath,b''.join(canonical(r.to_dict()) for r in freshdev))
if load_manifest(trainpath)!=rows or load_manifest(devpath)!=freshdev:raise ValueError('Published canonical manifest failed round-trip')
path=ROOT/'candidate-windows-v2.jsonl.gz';budget.write(path,gzip.compress(b''.join(canonical(w.to_dict()) for w in allwindows),mtime=0))
budget.write(ROOT/'candidate-window-source-ids.json',canonical(sorted(selected)))
# Duplicate manifest is avoided if its full size would consume the allocation.
report={'state':'verified_once_only_candidates_not_scheduled','publication_version':2,'supersedes_manifest_publication':'Original train-candidates.jsonl, fresh-dev.jsonl and emogator-train.jsonl contain pretty JSON object streams, not valid JSONL. Preserved unchanged but must not be used. Canonical v2 train and dev manifests are load_manifest round-trip verified. Original audio, receipts, labels and reservation identities unchanged.','window_encoding':'gzip compressed canonical JSONL, mtime zero for deterministic digest','windows':len(allwindows),'sources':len(ids),'scored_input_samples':sum(bycond.values()),'scored_hours':sum(bycond.values())/16000/3600,'whole_source_hours':sum(counts.values())/16000/3600,'condition_totals':{k:{'windows':clipcond[k],'scored_hours':v/16000/3600,'whole_hours':wholecond[k]/16000/3600,'scored_seconds':v/16000} for k,v in sorted(bycond.items())},'unusable_short_sources':excludedshort,'source_exclusions':len({r.source_id for r in excluded}),'heldout_sources':len({r.source_id for r in historical+freshdev}),'context_frames':29,'scored_frames_maximum':64,'minimum_valid_input_samples':3040,'input_hop':640,'output_input_ratio':3,'scored_interval_overlap':False,'active_plan_modified':False,'generic_emotional_nonverbal_is_not_verified_action_label':True,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,ROOT/'candidate-window-source-ids.json',trainpath,devpath]},'free_bytes':shutil.disk_usage(ROOT).free}
budget.write(ROOT/'candidate-window-inventory.json',canonical(report));print(json.dumps(report,indent=2))
