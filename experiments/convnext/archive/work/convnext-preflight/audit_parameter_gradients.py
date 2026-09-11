"""Read-only CPU gradient and timing-alignment diagnostics; no optimizer."""
from pathlib import Path
import json, sqlite3, hashlib, io, statistics
import torch
from audiovae_student.model import StudentDecoder, StudentConfig
from audiovae_student.cache import UtteranceCache
from audiovae_student.losses import WarmupReconstructionLoss
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
b=Path('/workspace/fast-audiovae-convnext-20260908-r1')
saved=torch.load(b/'training-runs/warmup-speech-500h-v1/latest.pt',map_location='cpu',weights_only=True)
model=StudentDecoder(StudentConfig(**saved['identity']['model_config'])).eval()
model.load_state_dict(saved['model'],strict=True)
criterion=WarmupReconstructionLoss()
c=sqlite3.connect('file:'+str(b/'target-cache-500h/index.sqlite3')+'?mode=ro',uri=True)
entries={json.loads(v)['row_id']:(k,json.loads(v)) for k,v in c.execute('SELECT lookup_key,info FROM entries')}
c.close()
ids=['crema_d:1040_IEO_FEA_HI.wav','emogator:000216-10-2.mp3','2277-149896-0030','jvnv:M2_anger_regular_61.wav']
parameter_names=['adapter.weight','blocks.9.project.weight','output.weight']
parameters=dict(model.named_parameters())
weights={'teacher_spectral':15.,'teacher_waveform':1.,'reference_spectral':45.}
report={'device':'cpu','optimizer_steps':0,'checkpoint_step':saved['step'],'results':[]}
for source_id in ids:
 key,info=entries[source_id]
 raw=(b/'target-cache-500h/targets'/(key+'.pt')).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==info['file_sha256']
 p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
 rec=UtteranceCache(p['latents'],p['teacher_audio'],p['reference16k'],p['metadata'],p['cache_key']);rec.validate()
 end=min(64,rec.latents.shape[-1]); valid=min(end*1920,rec.input_samples*3)
 pred=model(rec.latents[...,:end])[...,:valid]
 target=rec.teacher_audio[...,:valid];ref=rec.reference16k[...,:valid//3]
 losses=criterion(pred,target,ref)
 gradients={name:torch.autograd.grad(weight*losses[name],[parameters[n] for n in parameter_names],retain_graph=True) for name,weight in weights.items()}
 item={'source_id':source_id,'weighted_parameter_gradient_norms':{name:{pn:float(g.norm()) for pn,g in zip(parameter_names,grads)} for name,grads in gradients.items()}}
 # Diagnostic best-lag correlation only. No shift is applied to losses or targets.
 a=pred.detach().flatten().double();t=target.flatten().double()
 a=a-a.mean();t=t-t.mean();n=a.numel();fft_n=1<<(2*n-1).bit_length()
 corr=torch.fft.irfft(torch.fft.rfft(a,n=fft_n)*torch.conj(torch.fft.rfft(t,n=fft_n)),n=fft_n)
 lag=1920
 candidates=torch.cat([corr[-lag:],corr[:lag+1]])/(a.norm()*t.norm()).clamp_min(1e-20)
 idx=int(candidates.abs().argmax())
 item['alignment_diagnostic']={'maximum_absolute_correlation_within_40ms':float(candidates[idx].abs()),'signed_correlation':float(candidates[idx]),'best_lag_samples':idx-lag,'zero_lag_correlation':float(candidates[lag])}
 report['results'].append(item)
 print(json.dumps(item),flush=True)
 del pred,losses,gradients
(b/'audit-completed-loss/parameter-gradients.json').write_text(json.dumps(report,indent=2)+'\n')
