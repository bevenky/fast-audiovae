"""Isolate the cuDNN backend from the first frozen-teacher forward."""
import fcntl
from pathlib import Path
import torch
import cache_probe

cache_probe.OUT=Path('/tmp/fast-audiovae-recovery-20260909/cache-cudnn-off-from-start')
lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
with lock.open('rb') as handle, torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache_probe.main()
