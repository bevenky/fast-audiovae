"""Read existing plans/receipts and count exposures; no model or tensor loading."""
from pathlib import Path
import json,hashlib

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def windows(path,limit):
    ready=json.loads((path/'ready.json').read_text());pins=ready['files_sha256']
    for n in ('windows.jsonl','input-sample-counts.json','train-manifest.jsonl'):
        if sha(path/n)!=pins[n]:raise ValueError('Changed plan '+str(path/n))
    counts=json.loads((path/'input-sample-counts.json').read_text())
    rows=[json.loads(line) for line in (path/'train-manifest.jsonl').read_text().splitlines()]
    allw=[json.loads(line) for line in (path/'windows.jsonl').read_text().splitlines()]
    w=allw[:limit] if limit is not None else allw
    if limit is not None and len(w)!=limit:raise ValueError('Missing consumed windows')
    ids={v['source_id'] for v in w}
    scored=sum(v['valid_output_samples48k'] for v in w)
    return {'path':str(path),'ready_sha256':sha(path/'ready.json'),'windows_sha256':pins['windows.jsonl'],
        'selected_windows':len(w),'plan_windows':len(allw),'selected_distinct_source_ids':len(ids),'plan_source_rows':len(rows),
        'scored_hours':scored/48000/3600,'selected_full_source_hours':sum(counts[sid] for sid in ids)/16000/3600,
        'sample_row_keys':list(w[0]),'note':'Counts sources eligible for teacher preparation; not proof all used batch or all were wrong'}

b=Path('/workspace/fast-audiovae-convnext-20260909-r9');out={}
out['r9_first5900']=windows(b/'data/recipe-v2/optimization',188800)
out['r9_calibration']=windows(b/'data/recipe-v2/calibration',None)
out['r9_5900to8090']=windows(b/'remediation/continuation-step005900',70080)
p=b/'remediation/corrected-screen/targeted-data.json';d=json.loads(p.read_text());groups={};allkeys={}
for name,pool in d['pools'].items():
    ws=[v['window'] for v in pool];ids={v['source_id'] for v in ws}
    groups[name]={'windows':len(ws),'distinct_source_ids':len(ids),'scored_hours':sum(v['valid_output_samples48k'] for v in ws)/48000/3600}
    for w in ws:allkeys[(w['source_id'],w['start_frame'],w['valid_output_samples48k'])]=w
out['corrected8490']={'path':str(p),'sha256':sha(p),'pools':groups,'unique_cached_crops':len(allkeys),
    'distinct_source_ids':len({k[0] for k in allkeys}),'source_rows':len(d['rows']),
    'unique_scored_hours':sum(v['valid_output_samples48k'] for v in allkeys.values())/48000/3600,
    'note':'Same cached pools reused across matched arms; no per-actual-batch numerical receipts retained by original preparation code'}

for root in sorted(Path('/workspace').glob('fast-audiovae-convnext-*')):
    for p in list((root/'training-runs').glob('*/identity.json'))+list((root/'training-runs').glob('*/run.json')):
        if not p.exists():continue
        d=json.loads(p.read_text());data=d.get('data',{});impl=d.get('implementation',{})
        short={'path':str(p),'sha256':sha(p),'file_bytes':p.stat().st_size,
            'batched_teacher_sha256':impl.get('batched_teacher',impl.get('batched_teacher.py')),
            'data_keys':list(data),'initialization':d.get('initialization'),'kind':d.get('kind'),
            'batch_size':d.get('batch_size'),'global_start_step':d.get('global_start_step')}
        for field in ('diagnostic','sentinel','training','heldout'):
            v=d.get(field,data.get(field))
            if isinstance(v,dict):
                short[field]={k:(len(x) if isinstance(x,list) else x) for k,x in v.items()
                              if isinstance(x,(int,float,str,list)) and k not in ('crops','rows')}
                for k in ('crops','rows'):
                    if isinstance(v.get(k),list):short[field][k+'_count']=len(v[k])
            elif isinstance(v,list):short[field+'_count']=len(v)
        out.setdefault('run_lineage',[]).append(short)
print(json.dumps(out,sort_keys=True,allow_nan=False))
