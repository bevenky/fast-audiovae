"""Two-source frozen-encoder boundary audit; no student or optimizer calls."""
import json,os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from diagnostic_common import load_context
from repair_natural_history import source_rows,authentic_audio,raw_metrics,tensor_hash
from audiovae_student.fusion_evaluation import _preserved_evaluation
ctx=load_context();rows,_=source_rows(ctx);teacher=ctx.teacher();out=[]
with _preserved_evaluation(teacher.model):
 for sid in ['freesound:119459','freesound:386521']:
  c=next(c for c in ctx.heldout if c.source_id==sid and c.context_start_frame==0 and c.latents.shape[-1]==64)
  a,info=authentic_audio(rows[sid]);models={}
  for mode,x in [('full',a),('prefix64',a[...,:64*640].contiguous())]:
   before=tensor_hash(x);z=teacher.encode(x.to(teacher.device)).detach().cpu()
   if tensor_hash(x)!=before:raise RuntimeError('Input mutated')
   z=z[...,:64];diff=(z-c.latents).abs();per=diff.amax(dim=(0,1)).tolist()
   models[mode]={'vs_cached':raw_metrics(z,c.latents),'per_frame_max_abs':per,'frames_over_1e_4':[i for i,v in enumerate(per) if v>1e-4], 'latent_sha256':tensor_hash(z)}
   if mode=='full':fullz=z
   else:models['full_vs_prefix64']=raw_metrics(fullz,z)
  decoded=teacher.decode(c.latents.to(teacher.device)).detach().cpu()
  models['cached_z_decode_vs_cached_target']=raw_metrics(decoded,c.teacher_audio)
  out.append({'source_id':sid,'source':info,'cached_z_sha256':tensor_hash(c.latents),'cache_key':c.cache_key,'checks':models})
print(json.dumps({'cases':out,'parameter_updates':0,'encoder_calls':4,'student_calls':0,'files':ctx.verify_files()},indent=2))
