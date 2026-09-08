# Source and dependency notices

Project-owned source follows the repository's [Apache 2.0 license](../../../LICENSE). This directory preserves source bytes; it does not redistribute model or dataset licenses on their owners' behalf.

The SME2 matrix kernel and streaming ABI helper are from [Arm KleidiAI](https://github.com/ARM-software/kleidiai), commit `02f7b3df98c39df1884eead7461c3269feca4ddc`. Their original copyright notices and [Apache 2.0 license](native/sme/upstream/LICENSES/Apache-2.0.txt) are retained. The [upstream manifest](native/sme/upstream-provenance.json) records exact file, Git blob and SHA256 identities. The upstream quantizer is not used.

ONNX Runtime headers and the runtime are external dependencies. Header pins are in [the parent experiment](../pins/ort.json), with its [ONNX Runtime license](../licenses/ONNXRUNTIME-LICENSE). The Apple builder consumes the ORT header pins only; unrelated entries in that parent file do not add SLEEF, oneMKL or LIBXSMM dependencies to this Apple package.

NumPy, ONNX, SoundFile and optional quality-scoring dependencies retain their respective licenses and are not vendored here. Apple SDK components and system frameworks are supplied by the user's toolchain and operating system.
