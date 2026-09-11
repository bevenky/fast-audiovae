"""Run a fixed current-checkpoint derivative audit, with no optimizer steps."""
import argparse
import fcntl
import json
from pathlib import Path

from run_diagnostic import CHECKPOINTS, ROOT, save, status
import torch
from canonical_evaluation import load_canonical_cache
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.fusion_architecture import FusionStudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha
from gradient_probe import FrozenWaveformGradientProbe
from parameter_geometry import probe_parameter_geometry


def run(out, selection):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    path,expected=CHECKPOINTS['quarter']
    if file_sha(path)!=expected:raise ValueError('Checkpoint bytes changed')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['engine']
    model=FusionStudentDecoder.from_decoder(StudentDecoder(StudentConfig(**state['model_config'])))
    model.load_state_dict(state['model'],strict=True)
    model=model.cuda().eval().requires_grad_(False)
    original=state_fingerprint(model.state_dict())
    if original!=state_fingerprint(state['model']):raise ValueError('Model restore differs')
    crops,_,receipt=load_canonical_cache(ROOT/'canonical-panel-v1/receipt.json')
    lookup={(c.source_id,c.start_frame):c for c in crops}
    selected=json.loads(Path(selection).read_text())
    if len(selected)!=10:raise ValueError('Expected the sealed10case diagnosis')
    loss_probe=FrozenWaveformGradientProbe(state,device='cuda')
    rows=[]
    for i,key in enumerate(selected):
        crop=lookup[key['source_id'],key['start_frame']]
        status('parameter_geometry',completed=i,total=len(selected),source_id=crop.source_id)
        with torch.no_grad():expected_prediction=model(crop.latents.cuda()).detach()
        rows.append(probe_parameter_geometry(model,crop,state,device='cuda',loss_probe=loss_probe,
                                             expected_prediction=expected_prediction,include_per_loss=False))
        save(out/'parameter-geometry.json',rows)
    if state_fingerprint(model.state_dict())!=original or file_sha(path)!=expected:
        raise ValueError('Frozen model or checkpoint changed')
    save(out/'complete.json',{'checkpoint_sha256':expected,'canonical_cache_sha256':receipt['sha256'],
        'model_parameters_and_buffers_unchanged':True,'optimizer_steps':0,'cases':len(rows),
        'source_sha256':{n:file_sha(Path(__file__).with_name(n)) for n in
                         ['run_parameter_geometry.py','parameter_geometry.py','gradient_probe.py']}})
    status('complete',out=str(out))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--selection',required=True)
    a=p.parse_args()
    lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
    with lock.open('a+') as h:
        fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a.out,a.selection)
