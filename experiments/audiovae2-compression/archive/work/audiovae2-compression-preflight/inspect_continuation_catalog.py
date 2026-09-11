"""Read-only capacity inventory of the existing downloaded source catalogues."""
import collections,hashlib,json
from pathlib import Path
B=Path('/workspace/fast-audiovae-convnext-20260909-r9')
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
KEYS=('source_id','audio_sha256','parent_recording_id')
def lines(p):
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
pilot=json.loads((P/'pilot-selection-v1.json').read_text())
blocked={k:set() for k in KEYS}
def block(r):
 for k in KEYS:
  if r.get(k):blocked[k].add(r[k])
for split in pilot['splits'].values():
 for row in split['rows']:block(row['manifest_row'])
reserved=[B/'data/recipe-v2/optimization/reserved.jsonl',B/'data/recipe-v2/optimization/excluded-sources.jsonl',B/'remediation/continuation-step005900/reserved.jsonl',B/'remediation/heldout-languages/language-expanded-manifest.jsonl',B/'remediation/validation-appendix/manifest.jsonl',B/'remediation/expressive/fresh-dev-v2.jsonl']
for p in reserved:
 for row in lines(p):block(row)
canonical=json.loads(Path('/tmp/fast-audiovae-recovery-20260909/canonical-panel-v1/source-inventory.json').read_text())
for row in canonical['sources']:
 if row.get('manifest_row'):block(row['manifest_row'])
rows={}
for p in [B/'data/recipe-v2/optimization/train-manifest.jsonl',B/'remediation/expressive/train-candidates-v2.jsonl']:
 for row in lines(p):
  if row['split']!='train' or any(row[k] in blocked[k] for k in KEYS):continue
  if row['source_id'] in rows and rows[row['source_id']]!=row:raise ValueError('conflicting source')
  rows[row['source_id']]=row
dedup={k:set() for k in KEYS};accepted=[]
for sid,row in sorted(rows.items()):
 if any(row[k] in dedup[k] for k in KEYS):continue
 for k in KEYS:dedup[k].add(row[k])
 accepted.append(row)
result={'candidate_sources':len(rows),'source_hash_parent_unique_capacity':len(accepted),'datasets':dict(collections.Counter(r['dataset'] for r in accepted)),'languages':dict(collections.Counter(r['language'] for r in accepted)),'licenses':dict(collections.Counter(r['license'] for r in accepted)),'blocked_identity_counts':{k:len(v) for k,v in blocked.items()},'files_checked_exists':sum(Path(r['audio_path']).is_file() for r in accepted),'total_source_hours':sum(r['duration_seconds'] for r in accepted)/3600}
(P/'continuation-catalog-capacity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
