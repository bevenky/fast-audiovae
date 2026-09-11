"""Run the same original batch with JIT optimization disabled from process startup."""
import fcntl
from pathlib import Path
import torch
import cache_probe

cache_probe.OUT=Path('/tmp/fast-audiovae-recovery-20260909/cache-jit-from-start')
lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
with lock.open('rb') as handle, torch.jit.optimized_execution(False):
    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache_probe.main()
