"""Read the active recovery without launching work or changing its state."""
import json
from pathlib import Path
import subprocess

root=Path('/tmp/fast-audiovae-progressive-pruning-v1/recovery-1000-2000')
out=root/'segment-1000-2000'
result={}
p=root/'process-launch.json'
if p.exists():
    launch=json.loads(p.read_text());pid=launch['pid']
    cmdfile=Path('/proc')/str(pid)/'cmdline'
    result['process']={'pid':pid,'active':cmdfile.exists() and b'progressive_continue.py' in cmdfile.read_bytes()}
for name in ('restore-check.json','completed.json'):
    p=out/name
    if p.exists():result[name]=json.loads(p.read_text())
p=out/'train.jsonl'
if p.exists():
    rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if rows:
        result['latest_training']={k:v for k,v in rows[-1].items() if k not in ('source_ids','teacher_cache_checks')}
        ids=[sid for row in rows for sid in row['source_ids']]
        checks=[c for row in rows for c in row['teacher_cache_checks']]
        result['new_sources']={'count':len(ids),'unique':len(set(ids))}
        result['teacher_cache']={'comparisons':len(checks),'all_passed':all(c['allclose_original_tolerance'] for c in checks)}
for step in (1500,2000):
    p=out/f'development-step{step}.json'
    if p.exists():
        report=json.loads(p.read_text())
        result['quality_'+str(step)]={'aggregate':report['aggregate'],'quiet':report['quiet_regions']['regions']}
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
p=root/'training.log'
if p.exists():result['log_tail']=p.read_text(errors='replace').splitlines()[-6:]
(root/'latest-status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
