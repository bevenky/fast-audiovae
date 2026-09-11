# Updated training environment

The new isolated environment uses **PyTorch 2.14.0, CUDA 12.6 and cuDNN 9.25.1**. It contains 69 packages. Imports and dependency checks completed successfully, apart from the single intentional cuDNN pin exception described below.

| Component | Installed version |
|---|---|
| PyTorch | 2.14.0+cu126 |
| CUDA toolkit | 12.6.3 |
| cuDNN Python package | 9.25.1.1, providing runtime 9.25.1 |
| NCCL | 2.29.3 |
| Triton | 3.8.0 |
| ONNX Runtime / ONNX | 1.29.0 / 1.22.0 |
| TensorBoard | 2.21.0 |
| NumPy | 2.5.3 |
| pytest / pip | 9.1.1 / 26.2.1 |

CUDA 12.6 is an official build of the latest PyTorch release and works with the existing 570.133.20 driver. The complete toolkit dependency set was installed together. NVIDIA's support matrix permits cuDNN 9.25.1 with CUDA 12.6 and this driver; no host-driver replacement or CUDA 13 compatibility layer was required. [PyTorch release](https://pytorch.org/blog/pytorch-2-14-release-blog/), [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html).

The official PyTorch CUDA 12.6 wheel requires cuDNN 9.10.2.21. We deliberately replaced that package with **9.25.1.1**, which fixes the reproduced encoder error. `pip check` reports exactly this one known conflict, with **no other dependency conflicts**. Package metadata was not edited. The remaining CUDA dependencies follow PyTorch's required versions; Pydantic-core and mpmath likewise retain versions required by their parent packages. This is the selected compatible stack, not an independent maximum-version upgrade of every dependency.

The original environment, cached targets and checkpoints remain intact. The new interpreter is `/tmp/fast-audiovae-recovery-20260909/venv214/bin/python`. The next training launcher can select it after target repair is complete. TensorBoard was installed in the new environment; the existing service was not restarted.

The latest-stack head and runtime measurements used CUDA path helper 1.6.0. After those tests ended, that pure-Python helper was updated to 1.8.1, its latest compatible release. No other package changed in that final step, all 57 recorded native library files were unchanged, and imports and dependency checks passed again. Installation and finalization performed no GPU API or model calls; model qualification was a separate operation.

[Final package lock](requirements-training-torch214.lock) and [versions, wheel sources and checksums](updated-environment-versions.json) record the exact environment. Because cuDNN is an explicit override, the lock is an inventory, not a promise that a single ordinary `pip install -r` will resolve it: recreate the official dependency set, apply the cuDNN-only override, and verify the same single declared exception.

The separate matched H100 throughput check completed on both environments with corrected cuDNN. Ten timed training updates after three warmups averaged 0.331 seconds on PyTorch 2.11 and 0.323 seconds on 2.14. Teacher encoding averaged 84.2 versus 84.4 ms, and decoding 123.2 versus 124.1 ms for the same eight-recording batch. These changes are small relative to the observed variation and do not establish a material speedup. Teacher latents and decoded output matched bitwise across the two corrected runtimes; optimizer restoration checks and all recorded training losses were finite. The 13 trial updates in each environment were discarded, with no saved training checkpoints. [Detailed timing comparison](runtime-comparison.json).
