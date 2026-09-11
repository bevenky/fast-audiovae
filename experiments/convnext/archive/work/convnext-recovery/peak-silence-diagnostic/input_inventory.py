"""CPU-only inventory of already used source inputs, without model inference."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from audiovae_student.restart_data import file_sha


def run():
    torch.set_num_threads(1)
    receipt_path=Path('/dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json')
    receipt=json.loads(receipt_path.read_text())
    path=Path(receipt['path'])
    if file_sha(path)!=receipt['sha256']:raise ValueError('Source target cache changed')
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    rows=data['pools']['targeted_generator']
    if len(rows)!=12800:raise ValueError('Training pool changed')
    bins=Counter();sources={k:set() for k in ('exact_zero','nonzero_le_1e-5','1e-5_to_1e-4','1e-4_to_1e-3','above_1e-3')}
    total=0;missing=0;longest=0;long_sources=set();runs_at_least_one_second=0;exact_zero_samples=0
    for i,c in enumerate(rows):
        ref=c['reference16k']
        if ref is None:missing+=1;continue
        if c['valid_scored_samples']%3:raise ValueError('Noninteger16k input extent')
        first=c['context_frames']*640
        stop=first+c['valid_scored_samples']//3
        x=ref.flatten().numpy()[first:stop]
        if len(x)!=stop-first or not np.isfinite(x).all():raise ValueError('Invalid input window')
        total+=len(x)
        zero=(x==0);exact_zero_samples+=int(zero.sum())
        edges=np.r_[-1,np.flatnonzero(~zero),len(x)]
        runs=np.diff(edges)-1
        longest=max(longest,int(runs.max()))
        count=int((runs>=16000).sum());runs_at_least_one_second+=count
        if count:long_sources.add(c['source_id'])
        for s in range(0,len(x),320):
            v=x[s:s+320].astype(np.float64)
            r=float(np.sqrt(np.mean(v*v)))
            key='exact_zero' if r==0 else 'nonzero_le_1e-5' if r<=1e-5 else '1e-5_to_1e-4' if r<=1e-4 else '1e-4_to_1e-3' if r<=1e-3 else 'above_1e-3'
            bins[key]+=len(v);sources[key].add(c['source_id'])
        if i%2000==0:print(json.dumps({'scanned':i+1}),flush=True)
    result={'scope':'Source-input20ms RMS windows on actual12800 scored training crops; no encoder/decoder forwards',
        'cache_sha256':receipt['sha256'],'source_sha256':file_sha(__file__),'crops':len(rows),'missing_inputs':missing,
        'input_samples':total,'input_hours':total/16000/3600,
        'bins':{k:{'samples':bins[k],'seconds':bins[k]/16000,'sources':len(sources[k]),'duration_percent':100*bins[k]/total} for k in sources},
        'individual_exact_zero_samples':exact_zero_samples,
        'longest_exact_zero_run_within_crop_seconds':longest/16000,
        'at_least_one_second_exact_zero_runs':runs_at_least_one_second,'sources_with_such_runs':len(long_sources),
        'limitations':'No joining separate crops; longest-run statistic is a within-crop lower bound. Source-input levels are distinct from teacher48k output quiet thresholds. Context and right-padding excluded.'}
    target=Path('/tmp/fast-audiovae-recovery-20260909/peak-silence-diagnosis-v1/input-inventory.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
