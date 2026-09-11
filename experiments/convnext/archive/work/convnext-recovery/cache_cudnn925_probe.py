"""Retest exact historical encoder batch with externally selected current cuDNN."""
import fcntl
from pathlib import Path
import torch
import cache_probe

if torch.backends.cudnn.version()!=92501: raise RuntimeError("Expected isolatedcuDNN9.25.1")
cache_probe.OUT=Path("/tmp/fast-audiovae-recovery-20260909/cache-cudnn925")
lock=Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
with lock.open("rb") as handle:
    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache_probe.main()
