"""Isolate backend dependence of a reproduced frozen-encoder batch discrepancy."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json, fcntl
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnostic_common import load_context, atomic_json, status
from repair_natural_history import source_rows, authentic_audio, raw_metrics
from audiovae_student.objective_comparison import state_fingerprint
from cache_probe import OUT, IDS, LENGTHS

@torch.no_grad()
def main():
    ctx=load_context(); rows,_=source_rows(ctx); teacher=ctx.teacher()
    before=state_fingerprint(teacher.model.state_dict())
    audios=[authentic_audio(rows[s])[0] for s in IDS]
    x=torch.cat([F.pad(a,(0,93440-a.shape[-1])) for a in audios]).to(teacher.device)
    prior=torch.load(OUT/'probe-latents.pt',weights_only=True,map_location='cpu')
    serial=torch.cat([teacher.encode(x[i:i+1].contiguous()).detach().cpu() for i in range(8)])
    result={'teacher_state_sha256':before,'batch_shape':list(x.shape),'experiments':[]}
    def check(name):
        status('encoder_backend_probe',policy=name)
        zs=[teacher.encode(x).detach().cpu() for _ in range(2)]
        item={'policy':name,'batch_vs_historical':raw_metrics(zs[-1],prior['batch_latents']),
              'batch_vs_serial':raw_metrics(zs[-1],serial),
              'repeat':raw_metrics(zs[0],zs[-1])}
        result['experiments'].append(item)
        atomic_json(OUT/'backend-progress.json',result)
    check('default')
    with torch.jit.optimized_execution(False):
        check('jit_optimized_false')
    with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
        check('cudnn_disabled')
    if state_fingerprint(teacher.model.state_dict())!=before:
        raise RuntimeError('Teacher state changed')
    result.update(parameter_updates=0,checkpoint_preservation=ctx.verify_files())
    atomic_json(OUT/'backend.json',result)
    status('backend_probe_complete')

if __name__=='__main__':
    with Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock').open('rb') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
