"""Small deterministic-backward check before the settings comparison."""
import json
import torch
import run_pilot
from author_mel import AuthorMelLoss

run_pilot.policy()
assert torch.cuda.is_available()
objective = AuthorMelLoss().cuda()
generator = torch.Generator(device='cuda').manual_seed(20260910)
prediction = (torch.randn(1, 1, 4097, generator=generator, device='cuda')*.03)
target = torch.randn(1, 1, 4097, generator=generator, device='cuda')*.03
results = []
for _ in range(2):
    source = prediction.clone().requires_grad_(True)
    loss = objective(source, target).losses['teacher_mel']
    gradient, = torch.autograd.grad(loss, source)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    results.append((loss.detach(), gradient))
assert torch.equal(results[0][0], results[1][0])
assert torch.equal(results[0][1], results[1][1])
print(json.dumps({'passed': True, 'torch': str(torch.__version__),
                  'device': torch.cuda.get_device_name(),
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'loss': float(results[0][0]), 'repeated_gradient_max_difference': 0.0}))
