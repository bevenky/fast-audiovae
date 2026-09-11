# Training library version audit

Checked on 9 September 2026 at 16:01 UTC. The original Runpod training environment contains **68 packages: 42 already match the latest stable PyPI release, and 26 have newer releases**. The main outdated component is the PyTorch/CUDA stack. Most data, monitoring and CPU inference libraries are already current.

This was a read-only metadata audit of `/workspace/fast-audiovae-convnext-20260908-r1/.train-venv`, using Python 3.12.3 on Linux x86_64. It installed nothing and ran no models or GPU operations. All 68 packages, requirements, version sources and compatible-wheel checks are in the [full manifest](library-version-manifest.json).

| Library | Installed | Latest stable | Action |
|---|---:|---:|---|
| [PyTorch](https://pypi.org/project/torch/) | 2.11.0+cu128 | 2.14.0 | Qualify a complete compatible CUDA bundle |
| [cuDNN, CUDA 12 package](https://pypi.org/project/nvidia-cudnn-cu12/) | 9.19.0.56 | 9.25.1.1 | Qualify with the selected PyTorch build |
| [Triton](https://pypi.org/project/triton/) | 3.6.0 | 3.8.0 | Use the selected PyTorch build's requirement |
| [ONNX Runtime](https://pypi.org/project/onnxruntime/) | 1.29.0 | 1.29.0 | Current |
| [ONNX](https://pypi.org/project/onnx/) | 1.22.0 | 1.22.0 | Current |
| [TensorBoard](https://pypi.org/project/tensorboard/) | 2.21.0 | 2.21.0 | Current |
| [NumPy](https://pypi.org/project/numpy/) | 2.5.3 | 2.5.3 | Current |
| [Hugging Face Hub](https://pypi.org/project/huggingface-hub/) | 1.30.0 | 1.30.0 | Current |
| [PyArrow](https://pypi.org/project/pyarrow/) | 25.0.1 | 25.0.1 | Current |
| [SoundFile](https://pypi.org/project/soundfile/) | 0.14.0 | 0.14.0 | Current |
| [soxr](https://pypi.org/project/soxr/) | 1.1.0 | 1.1.0 | Current |
| [Pydantic](https://pypi.org/project/pydantic/) | 2.13.5 | 2.13.5 | Current |
| [pytest](https://pypi.org/project/pytest/) | 9.0.2 | 9.1.1 | Update in the new environment |

**“Latest” must mean the latest compatible set.** The installed PyTorch build pins CUDA 12.8.1, cuDNN 9.19.0.56 and Triton 3.6.0. PyPI's default PyTorch 2.14 metadata instead selects CUDA 13.0.3, the CUDA 13 cuDNN package at 9.24.0.43, and Triton 3.8. CUDA-specific PyTorch indexes have different requirements. Installing the newest NVIDIA packages independently can break this relationship; the full manifest records every NVIDIA/CUDA version and requirement. [PyTorch package metadata](https://pypi.org/pypi/torch/json).

Three apparent outdated dependencies require special handling:

- Pydantic 2.13.5 requires `pydantic-core==2.46.5`, although 2.48.0 exists.
- SymPy 1.14.0 requires `mpmath<1.4`, so installed 1.3.0 is appropriate despite 1.4.1 being available.
- Old PyTorch requires `setuptools<82`. Latest setuptools 84.0.0 is an option for the new PyTorch environment, whose metadata no longer carries that cap.

Those constraints come from installed and publisher metadata in the manifest. Remaining ordinary maintenance updates are pip 24.0 → 26.2.1 and filelock 3.32.3 → 3.32.6. They are not fixes for the encoder's numerical issue.

The repository's CPU inference requirements already pin the latest ONNX and ONNX Runtime. Training uses PyTorch, NumPy, TensorBoard, PyArrow, Hugging Face Hub, SoundFile, soxr and the teacher's Pydantic configuration. CUDA libraries and Triton are framework dependencies rather than independent application choices.

**Audio quality scoring has a separate version contract.** PESQ, STOI, librosa, SciPy and TorchAudio are not installed in this training environment, so their separate scoring environments have not been certified current by this audit. Their latest releases are PESQ 0.0.4, pystoi 0.4.1, librosa 1.0.0, SciPy 1.18.1 and TorchAudio 2.11.0. The manifest includes the primary PyPI links. PESQ's latest release has no matching prebuilt wheel here and needs a source build. TorchAudio 2.11 now explicitly supports PyTorch 2.11 and later through its stable ABI. [TorchAudio compatibility](https://docs.pytorch.org/audio/stable/installation.html).

Our UTMOS scorer uses the pinned [SpeechMOS implementation](https://github.com/tarepan/SpeechMOS/tree/ed25eacbfa42b99156c36ebec67a733b5dbb9b79) and `utmos22_strong_step7459_v1.pt`. DNSMOS uses pinned [Microsoft models](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS). Preserve these model and preprocessing identities for historical comparisons; a newer similarly named package is not a drop-in scoring update.

The metadata check found no violated installed requirements outside optional extras. Matching wheels exist for the latest versions of all 68 installed packages, but wheel availability does not establish CUDA-driver compatibility or numerical correctness. Adopt the new environment only after the encoder reproduction and frozen teacher/student checks pass, and record its complete resolved versions.
