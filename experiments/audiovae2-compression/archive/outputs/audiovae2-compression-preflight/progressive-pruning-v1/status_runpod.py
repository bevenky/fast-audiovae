"""Read the first cut's actual state without starting or changing it."""
import json
from pathlib import Path
import subprocess

root=Path('/tmp/fast-audiovae-progressive-pruning-v1')
out=root/'cut1-384-256'
result={}
for name in ('process-launch.json',):
    path=root/name
    if path.exists():
        receipt=json.loads(path.read_text());pid=receipt['pid']
        cmdpath=Path('/proc')/str(pid)/'cmdline'
        result['trainer']={'pid':pid,'active':cmdpath.exists() and b'progressive_train.py' in cmdpath.read_bytes()}
path=out/'train.jsonl'
if path.exists():
    lines=path.read_text().splitlines()
    rows=[json.loads(line) for line in lines if line]
    if rows:
        latest=rows[-1]
        result['training']={k:v for k,v in latest.items() if k not in ('source_ids','teacher_cache_checks')}
        result['record_count']=len(rows)
        checks=[c for row in rows for c in row['teacher_cache_checks']]
        result['teacher_cache']={'comparisons':len(checks),'all_within_original_tolerance':all(c['allclose_original_tolerance'] for c in checks)}
        ids=[sid for row in rows for sid in row['source_ids']]
        result['sources']={'count':len(ids),'unique':len(set(ids))}
for name in ('full-width-copy.json','checkpoint-step0.json','completed.json'):
    path=out/name
    if path.exists():
        value=json.loads(path.read_text())
        if name=='full-width-copy.json':value={'passed':value['passed'],'records':len(value['records'])}
        result[name]=value
path=out/'development-step0.json'
if path.exists():
    report=json.loads(path.read_text())
    result['baseline']={'aggregate':report['aggregate'],'quiet_regions':{name:{k:row[k] for k in ('windows','failed','residual_rms','teacher_rms') if k in row} for name,row in report['quiet_regions']['regions'].items()}}
result['display_processes']=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:args=(entry/'cmdline').read_bytes().decode().split('\0')
    except (OSError,UnicodeDecodeError):continue
    if any('tensorboard' in arg or 'projector' in arg for arg in args) and any('python' in arg for arg in args[:1]):
        result['display_processes'].append({'pid':int(entry.name),'command':args})
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
path=root/'training.log'
if path.exists():result['log_tail']=path.read_text(errors='replace').splitlines()[-4:]
print(json.dumps(result,indent=2))
