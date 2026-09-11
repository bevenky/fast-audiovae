import gc
import torch
from diagnostic_common import load_context,atomic_json,status,NAMES

ctx=load_context()
rows=[]
for name in NAMES:
    engine=ctx.engine(name)
    rows.append({'name':name,'step':engine.step,'head_shape':list(engine.model.output.weight.shape),
                 'frozen_normalization':all(bool(getattr(engine.model,n).statistics_frozen) for n in ('stem_norm','affine'))})
    status('checkpoint_restored',**rows[-1])
    del engine;gc.collect();torch.cuda.empty_cache()
atomic_json(ctx.out/'preflight.json',{'checkpoints':rows,**ctx.verify_files(),'retained_updates':0})
