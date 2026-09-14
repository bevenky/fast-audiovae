# Intel streaming transfer, 14 September 2026

Release follow-up: the exact history, phase and matrix changes are integrated in
the [Intel serving recipe](../../docs/intel-serving.md). The report below preserves
the experiment-time decisions; Snake remains separate.

Short CPU checks found a useful, modest improvement from reusing the accepted
history and phase-assembly techniques. The combined candidate reduced decoder
time by 4.9% at 80 ms and 4.3% at 40 ms. It remains an experiment; automatic
serving and the published Intel package have not changed.

## Matched timing

Intel Xeon Platinum 8280 identity exposed by a two-vCPU KVM machine. Tests use
CPU 0, one thread, ONNX Runtime 1.29.0 CPUExecutionProvider, and the existing
optimized Intel INT8 recipe. No GPU was used. The runtime was held fixed to
isolate the kernel changes; this is not a claim that 1.29 is the latest release.

Three frozen Bengali, English and Spanish FLEURS prefixes, 1.6 seconds each,
were decoded with persistent streaming state. Each variant received one warmup
per prefix and three measured repetitions. Separate resident processes alternate
the execution order. RTF is pooled completed `Session.run` time divided by audio
duration. Loading, offline encoding, IPC, assertions and hashing are excluded.

| Packet size | Existing Intel | Combined candidate | Decoder time reduction |
|---|---:|---:|---:|
| 40 ms | 0.39575 | 0.37887 | 4.27% |
| 80 ms | 0.25731 | 0.24475 | 4.88% |

All three prefixes improve in aggregate at both packet sizes. Eight of nine
individual 40 ms pairs and nine of nine 80 ms pairs improve. These short samples
establish a candidate, not a broad hardware or audio-corpus guarantee.

For context, separate short screens on the same prefixes measured stock
AudioVAE2 at 0.86148 and Pocket Mimi at 0.18011 for 80 ms streaming. Mimi was not
interleaved with the combined candidate. AudioVAE2 outputs 48 kHz from continuous
64-channel, 25 Hz latents; Pocket Mimi outputs 24 kHz from continuous 32-channel,
12.5 Hz latents. Mimi has no native 40 ms frame in this comparison. These are
decoder measurements, not encoder-plus-decoder or TTS measurements.

The separate history-only and phase-only screens measured 0.24419 and 0.24990
at 80 ms. The combined separate screen measured 0.24303. Use the alternating
table above for the combined gain; isolated percentage improvements do not add.

## What changed

1. Six high-channel depthwise/Snake regions consume raw history directly and
   calculate only the depthwise and following Snake outputs that survive the
   causal crop. They retain the required pre-Snake history computation and its
   original arithmetic policy.
2. The first upsampling phase assembly reads its preceding projection state
   directly instead of materializing the old concatenation and slicing path.
   Addition order remains `(current + previous) + bias`.

The existing Intel paired VNNI projection, oneMKL matrix kernels, weights,
quantization scales, graph inputs/outputs and all 18 persistent state tensors
remain unchanged. AMD's AOCL matrix path was not copied to Intel. Only the
applicable history and phase mechanisms were transferred.

## Correctness

| Check | Result |
|---|---|
| Isolated regions, actual saved coefficients, six lengths, zero/tiny/random input | 252/252 output and state comparisons exact |
| Region input immutability | 126/126 passed |
| Short decoder packet, complete waveform and state comparisons | 1,520/1,520 exact |
| Alternating timing runs | All 18 measured prefix pairs have exact waveform and final state hashes |
| Different chunk partitions | Five cases match across all four partitions in both variants |
| Public API empty calls, retained state, reset and reset replay | Passed |

The region checks used 0.206 seconds of native execution. The decoder gates
used about 1.3 seconds per variant. No long recordings, perceptual-quality
scorers or training runs were added. Exactness means no additional numerical
change relative to the existing Intel INT8 decoder on these inputs. It does
not mean INT8 is identical to the original FP32 model.

Two test-harness issues were corrected before accepting results. The first
standalone region export needed concrete intermediate shapes for ORT's shape
inference. Excluding empty input did not fix that failure, which occurred while
loading the reference session. The first complete graph export also put small
integer shape constants in external storage; they are now embedded, with only
FP32 payloads of at least 1,024 bytes shared externally. Neither fix changes
kernel mathematics. Failed artifacts and receipts remain in the working output
directory; the successful graph manifest records the corrected export.

## What is still expensive

The combined candidate's short 80 ms trace attributes 20.2% of kernel time to
the first paired projection, 18.7% to the other standalone precision matrix
calls, 24.5% to fused residual stacks and 10.7% to the fused C128 upsampler.
Calling all fused-stage time matrix work would be incorrect.

A separate timing shim measured the following approximate breakdown within
those stages. Its 35 calls consumed 0.133 seconds of native execution. Every
stage matched the whole decoder and remained exact with timers enabled.

| Region | Input preparation | Integer GEMM | Other integer-row work | Snake | FP32 pointwise | Remaining work |
|---|---:|---:|---:|---:|---:|---:|
| C256 residual stack | 7.8% | 21.0% | 7.2% | 38.3% | 0% | 25.7% |
| C128 upsampler | 8.1% | 23.1% | 7.7% | 37.9% | 0% | 23.2% |
| C64 residual stack | 0% | 0% | 0% | 49.4% | 29.8% | 20.8% |
| C32 residual stack | 0% | 0% | 0% | 59.7% | 15.5% | 24.8% |

GEMM nested inside row work is counted once. Remaining work includes depthwise
convolution, copies, allocation, epilogues and ORT dispatch; it is not a pure
depthwise measurement. Timers add overhead, so these are diagnostic proportions,
not replacement end-to-end latency numbers.

## Intel libraries worth testing next

The current path already uses Intel oneMKL, custom AVX-512 VNNI for the first
projection, SLEEF for sine, and specialized small FP32 kernels. Merely installing
another Intel package will not accelerate an existing custom operator.

| Priority | Implementation | Relevant experiment | Status |
|---|---|---|---|
| 1 | [oneDNN BRGeMM microkernels](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html) | Pack constant INT8 weights once and reuse a shape-specific kernel on the narrow streaming matrices | Proposed, not timed |
| 2 | [oneMKL Vector Math](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/vm-data-types-accuracy-modes-and-performance-tips.html) | High-accuracy vector sine in the measured Snake regions | Proposed, not timed |
| 3 | [oneDNN MatMul](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) | Cached primitive and prepared weight layout as a simpler matrix alternative | Proposed, not timed |
| Later | [OpenVINO](https://github.com/openvinotoolkit/openvino) and [Optimum Intel](https://huggingface.co/docs/optimum-intel/index) | A separate portable stock-graph baseline, if microkernels do not close enough of the gap | Backend work required |

For oneDNN, follow the published [microkernel example](https://uxlfoundation.github.io/oneDNN/page_cpu_brgemm_example_cpp.html)
and start with the second projection pair, each with M=3,072, K=1,024
and T=8/16. Express the multiplication as `X^T[T,K] * W^T[K,M]` so constant
weights occupy the packable B operand. Keep packed weights, kernels and scratch
space across calls. Preserve the existing activation quantizer, integer
accumulation, compensation and dequantization. Include input layout conversion
and output scatter in timing. The public microkernel API requires callers to
manage B packing and explicitly warns that zero-point attributes can move
computation into floating point. The first tiny projection is already packed
and specialized, so compare against it before replacing anything there.

For Snake, start with the current 256-element tiles. Intel recommends its
vector-math API for vectors larger than 40 elements. Use high accuracy first;
its faster reduced-accuracy modes are not justified by the current quality
requirement. High accuracy still need not be bitwise identical to SLEEF, and
small changes before activation quantization can affect integer results. Screen
the sine and complete Snake outputs before trying a short decoder comparison.

OpenVINO is a graph compiler/runtime. Optimum Intel is its integration with
Hugging Face libraries, not an additional low-level codec kernel. Our custom ORT
operations need mapping or replacement through OpenVINO's
[extension mechanism](https://docs.openvino.ai/2024/documentation/openvino-extensibility.html).
That makes a runtime switch a larger experiment than replacing a matrix call.

[Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
is archived, so it is not a new dependency candidate.
[Intel Neural Compressor](https://github.com/intel/neural-compressor) provides
quantization and compression methods; changing the current INT8 representation
would require a separate quality experiment rather than being a kernel-only fix.

The recent Intel-authored [XPU kernel work on Hugging Face](https://huggingface.co/blog/danf/intel-xpu-kernels-skill)
targets Arc GPUs with Triton. Its reported gains do not apply to this Xeon CPU.

The [1D dilated-convolution paper](https://arxiv.org/html/2104.08002v1) studies
width blocking and small matrix kernels, including Xeon 8280 measurements. Its
dense convolution workloads support investigating layout and blocking, but its
speedups cannot be assigned to our seven-tap depthwise operations. Our measured
raw-history transfer already removes some unnecessary convolution work.

The next tests should remain isolated and short: exact layer shapes and
coefficients, two warmups and three alternating repetitions, one CPU thread,
and a cumulative native execution budget near one second per microbenchmark.
Only a clear layer win should advance to the existing three-prefix decoder
screen. There is no measured basis yet to promise another 30% or 50% reduction.

## Reproduction and evidence

`rewrite.py` creates isolated history, phase and combined graphs; `build.py`
pins the existing Intel compiler/arithmetic fingerprint and static SLEEF build.
`region_check.py` checks the changed regions. `short_check.py` contains the
bounded timing, profile and public-API checks; `paired_check.py` alternates the
two resident decoder processes. `attribution/` contains the timing shim.

`results/summary.json` retains exact aggregate values, source-result hashes,
graph/build provenance and the attribution summaries. The corresponding raw
aggregate/hash reports are retained in `results/`. No audio or latent arrays are
included. Host asset paths in the scripts refer to the existing Intel test
installation; these scripts are experiment tools, not a portable installer.

The proposed library screens above are now completed in the [Intel library screen](../intel-library-screen-v1/README.md). Its paired tests use this combined candidate as their reference.
