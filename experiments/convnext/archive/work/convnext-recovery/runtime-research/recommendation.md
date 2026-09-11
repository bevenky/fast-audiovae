# Coherent runtime for the teacher recovery

Research date: 2026-09-09. This audit read official package metadata and the existing isolated package files. It installed nothing and ran no CUDA kernels.

Use a fresh PyTorch 2.14 CUDA 12.6 environment for the next qualification. Keep its complete dependency set together. Treat cuDNN 9.25.1 as an explicit, independently qualified dependency override until a PyTorch wheel actually permits that version. Do not promote the current mixed `PYTHONPATH` installation into a training environment. The PyTorch 2.11 plus new cuDNN result is useful as a diagnostic control, not a reason to stop evaluating the latest framework.

PyTorch 2.14 was released September 2, 2026. Its official CUDA builds are 12.6, 13.0 and 13.2; the CUDA 12.8 index has no 2.14 wheel. [Release announcement](https://pytorch.org/blog/pytorch-2-14-release-blog/), [CUDA 12.8 index](https://download.pytorch.org/whl/cu128/torch/).

## Exact package requirements

These are Linux x86_64 CPython 3.12 wheel requirements downloaded from the official index. Complete metadata, URLs and SHA-256 values are retained in `official-wheel-metadata.json` and the three adjacent `.metadata` files.

| PyTorch wheel | CUDA toolkit package | cuDNN package | NCCL package | Triton |
|---|---|---|---|---|
| 2.14.0+cu126 | 12.6.3 | nvidia-cudnn-cu12 == 9.10.2.21 | nvidia-nccl-cu12 == 2.29.3 | ~= 3.8.0 |
| 2.14.0+cu130 | 13.0.3 | nvidia-cudnn-cu13 == 9.24.0.43 | nvidia-nccl-cu13 == 2.30.7 | ~= 3.8.0 |
| 2.14.0+cu132 | 13.2.1 | nvidia-cudnn-cu13 == 9.24.0.43 | nvidia-nccl-cu13 == 2.30.7 | ~= 3.8.0 |

Sources: [cu126 metadata](https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl.metadata), [cu130 metadata](https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl.metadata), [cu132 metadata](https://download.pytorch.org/whl/cu132/torch-2.14.0%2Bcu132-cp312-cp312-manylinux_2_28_x86_64.whl.metadata).

The cu126 build also specifies the toolkit's cublas, cudart, cufft, cufile, cupti, curand, cusolver, cusparse, nvrtc and nvtx extras; CUDA bindings >=12.9.4,<13; cuSPARSELt 0.7.1; NVSHMEM 3.4.5; and nvJitLink >=12.6.85,<13. Replacing only torch and NCCL leaves other libraries from the previous toolkit. A full dependency resolution is necessary.

## Driver compatibility

cuDNN 9.25.1 for CUDA 12.x supports both CUDA 12.6 and 12.8, Hopper GPUs, and Linux drivers >=525.60.13. The observed 570.133.20 host driver satisfies this support matrix. NVIDIA also documents binary backward compatibility between cuDNN minor releases within one major version. This supports testing the newer library with a build compiled against an older cuDNN 9 release; it does not replace model correctness checks or change Python wheel requirements. [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html), [cuDNN binary compatibility](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.1/developer/forward-compatibility.html).

CUDA 13 normally requires the R580 driver family or later. NVIDIA's forward-compatibility matrix allows `cuda-compat-13-0` and `cuda-compat-13-2` with R570 on supported data-center GPUs. The libraries can be extracted into a user-owned directory and selected for one process, without changing the kernel driver. H100 belongs to the eligible hardware class. This is a documented possible route, not a tested guarantee for this host/container: capability checks and actual mapped-library validation still apply, and some features require newer kernel-mode support. It adds another layer while retaining the cuDNN 9.24 pin, so it offers no immediate advantage for this recovery. [Minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [Forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## What can honestly pass package validation

An unmodified official 2.14 wheel plus its declared dependencies can pass `pip check`. None of the three current CUDA wheel variants declares cuDNN 9.25.1. There is therefore no all-official-wheel environment that simultaneously retains these exact pins and installs cuDNN 9.25.1 under the same package name.

Pip constraints narrow existing requirements; they do not override an exact pin. An explicit uv dependency override can resolve a replacement but does not prove the original metadata is satisfied. `--no-deps`, ignoring a diagnostic, retaining an unused old cuDNN package while loading another copy, or silently editing installed metadata must not be presented as an unchanged upstream environment. [Pip constraints](https://pip.pypa.io/en/stable/user_guide/#constraints-files), [uv overrides](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides), [uv package validation](https://docs.astral.sh/uv/pip/compatibility/#pip-check).

Practical choices:

1. Create a clean official 2.14+cu126 environment and validate its original dependency set. If the recovery needs 9.25.1, explicitly record that single override, its exact wheel/library hashes and the expected package-pin discrepancy. Require no unrelated dependency failures. This is the smallest experimental migration, not a claim of upstream pin conformance.
2. If zero package-pin exceptions and cuDNN 9.25.1 are mandatory, build an auditable local PyTorch 2.14 distribution against the chosen CUDA/cuDNN stack, with accurate dependency metadata and a distinct build identity. This is a separate build and qualification effort. It is not necessary merely to diagnose the current failure. Official source-build instructions permit supported CUDA and cuDNN 9 or newer. [PyTorch source build](https://pypi.org/project/torch/2.14.0/).
3. Keep 2.11 plus 9.25.1 only as a temporary isolated control if the complete 2.14 environment cannot yet be qualified. It also overrides an exact cuDNN pin: the observed 2.11+cu128 metadata requires 9.19.0.56. Do not label it latest or package-pin clean.

## NCCL import failure, confirmed without importing torch

Read-only `nm -D --defined-only` inspection showed NCCL 2.29.3 exports `ncclCommGrow` and the installed 2.28.9 does not. The new library is not missing the symbol.

The actual isolated PyTorch 2.14 `torch/__init__.py` explains why the old library can win:

- Lines 333-348 iterate `sys.path`, select the first matching package library, and call `ctypes.CDLL` using its absolute filename.
- Lines 351-386 preload the CUDA dependencies, including NCCL.
- Lines 435-446 perform that preload before `torch._C` loads the remaining dependencies.

The launch initially exposed only the new torch directory through `PYTHONPATH`; the new NCCL and cuDNN roots were present only in `LD_LIBRARY_PATH`. The preloader could consequently select the old venv's NCCL by absolute path. For one diagnostic launch all intended package roots must precede the old site-packages. For the real recovery environment, use one fresh venv, without inherited `PYTHONPATH`, user site-packages, or old CUDA library overlays; explicitly verify loaded versions and library paths after startup. The source and symbol inspection establish the loader mechanism; a successful corrected import remains the launcher's verification responsibility.

New cache provenance should record the actual torch version, toolkit version, cuDNN runtime version and backend selection. Changing the encoder backend or cuDNN must change cache identity. Actual-batch versus serial qualification remains necessary after a version fix, because package version alone cannot certify every shape or value.
