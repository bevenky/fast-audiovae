"""Matched short timing of the qualified hybrid, with unchanged public checks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import types

HERE = Path(__file__).resolve().parent
os.environ.pop('PYTORCH_MPS_PREFER_METAL', None)
for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','TORCHINDUCTOR_USE_FAST_MATH'):
    if os.environ.get(key,'0') != '0':
        raise RuntimeError(key+' must be disabled')
    os.environ[key]='0'
sys.path.insert(0,str(HERE.parent/'apple-gpu-v2'))
import experiment as exp
import repairs
import torch


def build(arm):
    if arm == 'before_compile':
        return exp.CompiledModel(exp.legacy())
    if arm not in ('hybrid','hybrid_pointwise'):
        raise ValueError(arm)
    model = exp.legacy()
    assert repairs.apply(model,'hybrid') == 6
    if arm == 'hybrid_pointwise':
        def pointwise(self,x):
            return torch.mm(self.weight[:,:,0],x[0]).unsqueeze(0)+self.bias[None,:,None]
        changed = 0
        for module in model.modules():
            if module.__class__.__name__ == '_Pointwise':
                module.forward=types.MethodType(pointwise,module); changed+=1
        assert changed==19
    return exp.CompiledModel(model)


if __name__ == '__main__':
    output=HERE/'screen-repair-r1.json'
    assert not output.exists()
    for receipt in ('hybrid-r1.json','hybrid-pointwise-r1.json'):
        qualified=json.loads((HERE/receipt).read_text())
        assert qualified['status']=='passed'
        for path in (HERE/'repairs.py',HERE/'candidates.py'):
            assert qualified['files'][str(path)]==hashlib.sha256(path.read_bytes()).hexdigest()
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
             (Path(__file__).resolve(), HERE/'repairs.py',HERE/'candidates.py')}
    original=exp.build; exp.build=build
    try:
        exp.main(argparse.Namespace(name='../apple-gpu-v4/screen-repair-r1',
            arms=['before_compile','hybrid','hybrid_pointwise'],repetitions=2))
    finally:
        exp.build=original
        if output.exists():
            result=json.loads(output.read_text())
            result.setdefault('files',{}).update(sources)
            result['repair']={'qualified_before_timing':True,'unchanged_tolerances':True,
                              'stage0':'Paired projection','stage1_to_5':'V3 two-projection form'}
            unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in sources.items())
            result['sources_unchanged']=unchanged
            if not unchanged:result.update(status='failed',error='Candidate source changed')
            output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert json.loads(output.read_text())['status']=='passed'
