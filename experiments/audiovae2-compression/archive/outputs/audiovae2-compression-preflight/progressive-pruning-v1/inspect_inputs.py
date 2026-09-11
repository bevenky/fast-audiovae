"""Inspect preserved inputs and existing run without model execution."""
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path('/workspace/fast-audiovae-compression-20260910-v1')
OLD = Path('/tmp/fast-audiovae-joint-recovery-v2/segment-5625-6625')
SHARDS = Path('/dev/shm/fast-audiovae-compression-fresh-shards-v2')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def schema(value):
    if isinstance(value,dict):return {k:schema(v) for k,v in value.items()}
    if isinstance(value,list):return {'length':len(value),'sample':value[:2]}
    return value

selection=json.loads((P/'pilot/channel-selection.json').read_text())
manifest=json.loads((P/'pilot-selection-v1.json').read_text())
plan=json.loads((P/'fresh-source-plan-v2/plan.json').read_text())
rows=manifest['splits']['fit']['rows']+plan['rows'][:9000]
identities={k:[row['manifest_row'][k] for row in rows] for k in ('source_id','audio_sha256','parent_recording_id')}
shards=[]
for start in range(0,9000,300):
    d=SHARDS/f'{start:06d}-{start+300:06d}'
    receipt=json.loads((d/'receipt.json').read_text())
    shards.append({'start':start,'stop':start+300,'complete':receipt['complete'],
                   'pairs_exists':(d/'pairs.pt').is_file(),'pairs_size':(d/'pairs.pt').stat().st_size,
                   'plan_identity_matches':receipt['plan_identity_sha256']==plan['identity_sha256']})
processes=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=[x.decode() for x in path.read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,UnicodeDecodeError):continue
    if any(Path(a).name in ('joint_recovery_v2.py','produce_continuation_extension_v2.py') for a in args):
        processes.append({'pid':int(path.parent.name),'args':args})
completed=json.loads((OLD/'completed.json').read_text()) if (OLD/'completed.json').exists() else None
result={'selection_schema':schema(selection),'original_fit_count':len(manifest['splits']['fit']['rows']),
        'cache_path':manifest['cache_path'],'cache_exists':Path(manifest['cache_path']).is_file(),
        'new_student_source_count':len(rows),'identity_unique_counts':{k:len(set(v)) for k,v in identities.items()},
        'first_source_metadata':rows[0]['manifest_row'],'shards':shards,'processes':processes,
        'old_run_completed':None if completed is None else {k:completed.get(k) for k in ('status','step','last_checkpoint_sha256','original_files_preserved')},
        'source_plan_sha256':sha(P/'fresh-source-plan-v2/plan.json'),
        'manifest_sha256':sha(P/'pilot-selection-v1.json')}
out=Path('/tmp/fast-audiovae-progressive-pruning-v1');out.mkdir(exist_ok=True)
(out/'input-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
