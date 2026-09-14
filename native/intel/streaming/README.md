# Intel streaming kernels

This is the accepted one-thread CPU path for 40 ms and 80 ms packets.

- The second upsampling projection pair uses oneDNN 3.13.2 BRGeMM with 64-channel panels. INT8 values, scales, integer correction and dequantization stay unchanged.
- Other packet sizes use the existing oneMKL precision core. The first projection keeps its existing VNNI kernel.
- Raw-history and phase operations reuse the qualified sources in `native/amd/streaming`. Snake continues to use the existing SLEEF implementation.
- The encoder, weights, streaming state layout and batch graph are unchanged.

`tools/build_intel_streaming.py` verifies source hashes, builds the libraries and checks their ELF dependencies. Runtime libraries use `$ORIGIN`, so a prepared bundle can move between directories. The normal recipe builds the pinned CPU-only sequential oneDNN source; release qualification can reuse the exact previously verified oneDNN binary and its real build receipt.

The generated bundle includes the oneDNN license and third-party notices. No GPU or OpenMP runtime is required by these added libraries.
