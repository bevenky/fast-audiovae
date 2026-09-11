"""Bounded read-only investigation of the existing first-mixer alignment failure."""
import argparse,json
from pathlib import Path
import torch
import diagnose_channel_contributions as d
b=d.base;g=d.group

def metrics(a,c):
 e=(a-c).double();flat=int(e.abs().argmax());i=list(torch.unravel_index(torch.tensor(flat,device=e.device),e.shape))
 return {'shape':list(a.shape),'max_abs':float(e.abs().max()),'rms':float(e.square().mean().sqrt()),'reference_rms':float(a.double().square().mean().sqrt()),'location':[int(v) for v in i],'teacher_value':float(a[tuple(i)]),'student_value':float(c[tuple(i)]),'within_original_absolute_1e5':bool(torch.allclose(a,c,atol=1e-5,rtol=0))}

@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for n in ('assets','base-out','manifest','out'):p.add_argument('--'+n,type=Path,required=True)
 args=p.parse_args();b.policy();metadata,selection,initial,preflight,manifest,pools=d.screen.authenticate_inputs(args)
 teacher=b.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
 model=b.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices']);first=d.operations(selection)[0]
 k=selection['stage2_indices'];paths=['sr_cond_model.3','model.3.block.0','model.3.block.1','model.3.block.2.block.0','model.3.block.2.block.1','model.3.block.2.block.2']
 specs=[d.Operation(p,p,[],[],0) for p in paths]+[first];rows=[];weights={}
 for path in paths:
  tm=teacher.model.decoder.get_submodule(path);sm=model.decoder.get_submodule(path)
  if hasattr(tm,'weight_v'):
   tw=g.effective_weight(tm).detach();sw=g.effective_weight(sm).detach()
   expected=tw.index_select(1,torch.tensor(k,device=tw.device)) if isinstance(tm,torch.nn.ConvTranspose1d) else tw.index_select(0,torch.tensor(k,device=tw.device))
   weights[path]=metrics(expected,sw)
 for idx,crop in enumerate(pools['calibration']):
  z,target,valid,_=b.batch([crop])
  with d.capture_operations(teacher.model.decoder,specs) as tc:tr=b.teacher_forward(teacher,z)
  d.warm_student(model,tr['group_input'])
  with d.capture_operations(model.decoder,specs) as sc:model.group_from_input(tr['group_input'])
  row={'index':idx,'source_id':crop['source_id'],'context_frames':crop['context_frames'],'valid_scored_samples':crop['valid_scored_samples'],'layers':{}}
  for path in paths:
   t=tc[path]['output'];s=sc[path]['output'];t=t if t.shape[1]==s.shape[1] else t[:,k]
   row['layers'][path]=metrics(t,s)
  row['pointwise_input']=metrics(tc[first.name]['input'][:,k],sc[first.name]['input']);rows.append(row)
  if not row['pointwise_input']['within_original_absolute_1e5']:break
 receipt={'version':'alignment_failure_readonly_v1','diagnostic_sha256':b.sha(d.__file__),'weights':weights,'sources':rows,'threshold_unchanged':True,'neural_training':False,'no_affine_fit':True}
 b.write_json(args.out,receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
