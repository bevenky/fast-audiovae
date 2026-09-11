"""Bounded held-out FP32 teacher repeatability check, not a timing benchmark."""
from pathlib import Path
import hashlib
import json
import math
import os

import soundfile as sf
import torch
from audiovae_student.teacher import FrozenAudioVAE2

base = Path('/workspace/fast-audiovae-convnext-20260908-r1')
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
source, checkpoint = base/'assets/audio_vae_v2.py', base/'assets/audiovae.pth'
cpu = FrozenAudioVAE2.from_files(source, checkpoint, device='cpu')
gpu = FrozenAudioVAE2.from_files(source, checkpoint, device='cuda')


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compare(a, b, *, enforce=True):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    assert a.shape == b.shape
    error = a-b
    noise = error.square().mean().item()
    signal = a.square().mean().item()
    snr = 10*math.log10(max(signal,1e-30)/max(noise,1e-30))
    result = {'max_abs':error.abs().max().item(), 'rms_error':noise**.5, 'snr_db':snr,
              'bitwise_equal':torch.equal(a,b), 'differing_values':int(torch.count_nonzero(error))}
    if enforce:
        assert result['max_abs'] <= 1e-3 and snr >= 70, result
    return result

before = state_digest(gpu)
rows = []
for optimized in (True, False):
    for path in sorted((base/'parity-inputs').glob('*.wav')):
        data, sr = sf.read(path,dtype='float32')
        assert sr ==16000 and data.ndim==1
        x=torch.from_numpy(data)[None,None]
        x_gpu=x.cuda()
        with torch.jit.optimized_execution(optimized):
            mu_cpu=cpu.encode(x)
            y_cpu=cpu.decode(mu_cpu)
            encodes=[gpu.encode(x_gpu) for _ in range(5)]
            mu_gpu=encodes[-1]
            decodes=[gpu.decode(mu_gpu) for _ in range(5)]
            y_gpu=decodes[-1]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                auto_mu=gpu.encode(x_gpu)
                auto_y=gpu.decode(auto_mu)
            after_mu=gpu.encode(x_gpu)
            after_y=gpu.decode(after_mu)
        assert all(t.dtype ==torch.float32 and not t.requires_grad and not t.is_inference()
                   for t in [mu_gpu,y_gpu,auto_mu,auto_y,after_mu,after_y])
        row={'clip':path.name,'jit_optimized_execution':optimized,
             'input_samples':len(data),'raw_target_samples':y_cpu.shape[-1],
             'valid_target_samples':3*len(data),
             'cpu_cuda_latents':compare(mu_cpu,mu_gpu),'cpu_cuda_waveform':compare(y_cpu,y_gpu),
             'successive_latent_repeats':[compare(a,b) for a,b in zip(encodes,encodes[1:])],
             'successive_decode_repeats':[compare(a,b) for a,b in zip(decodes,decodes[1:])],
             'post_warmup_autocast_latents':compare(mu_gpu,auto_mu),
             'post_warmup_autocast_waveform':compare(y_gpu,auto_y),
             'post_autocast_normal_latents':compare(auto_mu,after_mu),
             'post_autocast_normal_waveform':compare(auto_y,after_y)}
        print(json.dumps(row),flush=True)
        rows.append(row)
gpu.train(True)
assert all(not m.training for m in gpu.modules())
assert all(not p.requires_grad and p.grad is None for p in gpu.parameters())
after=state_digest(gpu)
assert before==after
result={'qualification':'held-out teacher implementation parity and repeatability only; no training or RTF benchmark',
        'teacher':gpu.provenance,'teacher_state_before':before,'teacher_state_after':after,
        'teacher_all_frozen':True,'teacher_eval':True,'autocast_outputs_float32':True,
        'policy':{'tf32':False,'cudnn_deterministic':True,'cudnn_benchmark':False,
                  'deterministic_algorithms':True,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  'per_shape_encode_calls':5,'per_shape_decode_calls':5,
                  'comparison_max_abs':1e-3,'comparison_min_snr_db':70},
        'comparisons':rows}
(base/'h100-teacher-parity.json').write_text(json.dumps(result,indent=2)+'\n')
print('Frozen teacher and FP32 target parity passed',flush=True)
