import json
from pathlib import Path
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.model import StudentDecoder,StudentConfig
from audiovae_student.restart_data import canonical
base=Path('/workspace/fast-audiovae-convnext-20260909-r9')
audit=base/'audit-step1000'
torch.set_num_threads(1)
evaluation=json.loads((audit/'evaluation-step001700.json').read_text())
rows={(r['source_id'],r['start_frame']):r for r in evaluation['rows'] if r['student_clipped_samples']}
payload=torch.load(audit/'checkpoint-step001700.pt',map_location='cpu',weights_only=True)
model=StudentDecoder(StudentConfig(**payload['engine']['model_config']));model.load_state_dict(payload['engine']['model']);model.eval()
saved=torch.load(base/'heldout-targets.pt',map_location='cpu',weights_only=True)
results=[]
with torch.no_grad():
 for item in saved['crops']:
  c=TrainingCrop(**item)
  if (c.source_id,c.start_frame) not in rows:continue
  y=model(c.latents)[...,c.scored_slice].flatten();t=c.teacher_audio[...,c.scored_slice].flatten()
  indices=torch.where(y.abs()>1)[0]
  results.append({'source_id':c.source_id,'start_frame':c.start_frame,'samples':len(y),
   'student_peak_abs':float(y.abs().max()),'teacher_peak_abs':float(t.abs().max()),
   'teacher_clipped_samples':int((t.abs()>1).sum()),'student_clipped_samples':len(indices),
   'first_clipped_ms':float(indices.min())/48,'last_clipped_ms':float(indices.max())/48,
   'clipped_in_first_10ms':int((indices<480).sum()),'clipped_in_last_10ms':int((indices>=len(y)-480).sum()),
   'peak_sample':int(y.abs().argmax()),'peak_sample_mod480':int(y.abs().argmax())%480,
   'boundary_scope':'Coordinates relative to the scored crop; real latent history retained'})
report={'checkpoint_step':1700,'device':'cpu','optimizer_updates':0,'rows':results}
(audit/'peak-audit.json').write_bytes(canonical(report));print(json.dumps(report))
