"""Bounded full-capacity synthetic training checks, not speech training."""
from pathlib import Path
import gc
import json

import torch
from audiovae_student.model import StudentDecoder
from audiovae_student.training import TrainingBatch, TrainingConfig, train_fixed_batch

base = Path('/workspace/fast-audiovae-convnext-20260908-r1')
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available()
generator = torch.Generator().manual_seed(18)
latents = torch.randn(1, 64, 8, generator=generator).cuda()
time = torch.arange(8 * 1920, device='cuda') / 48000
target = (0.15 * torch.sin(2 * torch.pi * 220 * time)
          + 0.04 * torch.sin(2 * torch.pi * 3300 * time))[None, None]
results = []
for name in ('adamw', 'muon_adamw'):
    torch.manual_seed(17)
    model = StudentDecoder().cuda()
    run_name = 'preflight-synthetic-H100-full-' + name
    result = train_fixed_batch(
        model, TrainingBatch(latents, target), steps=32,
        config=TrainingConfig(optimizer=name, seed=17),
        checkpoint_path=base / 'checkpoints' / (run_name + '.pt'),
        log_dir=base / 'runs', run_name=run_name)
    record = {'run': run_name, 'qualification': 'synthetic setup check; no speech or teacher targets',
              'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'parameters': sum(p.numel() for p in model.parameters()),
              'steps': result.step, 'metrics': result.metrics,
              'step_elapsed_seconds': result.step_elapsed_seconds}
    results.append(record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('metrics','step_elapsed_seconds')}), flush=True)
    del model, result
    gc.collect()
    torch.cuda.empty_cache()
(base / 'h100-logged-preflight.json').write_text(json.dumps(results, indent=2) + '\n')
