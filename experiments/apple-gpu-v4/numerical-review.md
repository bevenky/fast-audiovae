# Transpose matrix numerical review

Read-only source and saved-result audit, 2026-09-13. No model execution or existing-file changes. The V3 candidate remains unqualified under the original `atol=1e-5, rtol=1e-4` gates.

The transpose-only and transpose-plus-pointwise candidates both stop at `stage1.residual2.history`. In the transpose-only diagnostic, one of 27,648 elements fails: maximum absolute difference 4.342198371887207e-5, RMS 1.6587406843708762e-6, maximum tolerance ratio 1.2907666544423297. The largest absolute difference and failing element need not be the same element. Earlier state differences already exist; this is the first failed recorded gate, not the first divergent arithmetic operation. The raw `before_compile` label in these qualification files aliases the candidate, not the original compiled decoder.

The history is the input to the dilation-9 depthwise convolution after its pre-Snake activation. Error can propagate through earlier transpose, residual matrix, addition and Snake operations before being stored. For literal Snake coefficients a and r, its real-arithmetic derivative is `1 + a*r*sin(2*a*x)`. This explains possible amplification, but does not identify its measured contribution. Passing tiny CPU algebra fixtures or the earlier short waveform screen does not clear the failing MPS state gate.

| Form | Exact real-arithmetic computation for output phase q and time t | FP32 distinction |
| --- | --- | --- |
| Original transpose | Sum over c of `W[c,o,q]*x[c,t] + W[c,o,q+s]*previous[c,t]`, then bias | Backend controls accumulation order |
| V3 packed projection | One MM produces separate current-head and previous-tail dot products; add those rounded results, then bias | Two independently rounded channel reductions followed by an explicit add |
| V4 paired projection | One MM over concatenated current and previous channels; add bias | One 2Cin reduction, with no exposed intermediate phase-result rounding |

The V3 indexing and one-frame activated history implement the correct algebra. It neither adds bias twice nor consumes a future frame. A different FP32 reduction tree is the strongest source-supported hypothesis, not yet an experimentally isolated cause.

The installed Torch 2.14.0 records commit `08187d9e0fba026dc8217405802ab5381dc88d90`. At this commit, transposed convolution delegates to convolution backward-input, which submits `convolution2DDataGradientWithIncomingGradientTensor` with NCHW/OIHW descriptors. Its internal reduction order is inside MPSGraph and is not specified by the PyTorch source. [Convolution.mm:821–852 and 1042–1065](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Convolution.mm#L821-L852)

Eager MPS transpose adds bias separately after that bias-free operation. Therefore a bias-in-accumulator explanation is unsupported. [Convolution.cpp:1648–1657](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/Convolution.cpp#L1648-L1657)

With the recorded default dispatch and these contiguous FP32 shapes, V3 MM uses MPSGraph matrix multiplication. V4 first-stage T1 changes the shape to `[8192,4096] @ [4096,1]`; the rank-one gate selects the native Metal GEMV route before checking `PREFER_METAL`. T2 remains a matrix operation. This is a concrete dispatch difference to separate from arithmetic grouping in 40/80 ms checks. [LinearAlgebra.mm:262–294, 481–548 and 943–1002](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/LinearAlgebra.mm#L943-L1002)

Generated Metal compilation selects safe math and precise floating-point functions when `PYTORCH_MPS_FAST_MATH=0`; the source does not support blaming an enabled fast-math setting here. [OperationUtils.mm:789–810](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/OperationUtils.mm#L789-L810)

Recommended bounded sequence:

1. On the existing failing packet, compare each transpose using the same reference activation/history. Separate bias-free output, biased output and following Snake, then compare propagated histories. Include the original compiled route so compiler effects are not assigned to the matrix rewrite.
2. Check the already prepared paired projection at both packet lengths. If it fails locally, one interleaved K ordering `[current_c, previous_c]` per channel is a distinct bounded test. There is no source evidence that either ordering matches MPSGraph's opaque reduction tree.
3. If only specific stages seed the failure, retain their original transpose and keep matrix replacements elsewhere, then apply the complete original qualification. This may preserve part of the speed gain; no magnitude is predicted.

Padding the T1 paired matrix with a second independent column can isolate GEMV versus MPSGraph if the trace implicates that case. Compensation, precision reduction or tolerance changes are not justified fixes: lower mathematical error need not mean closer agreement with the qualified reference. No candidate is accepted on average error alone.

Small evidence hashes:

| File | SHA-256 |
| --- | --- |
| `../apple-gpu-v3/screen_transpose.py` | `31fe15796d9468d785fa1d796eb89aca63e47ec2a9dd813c781aed42fea8e5d7` |
| `../apple-gpu-v3/qualify-transpose-only-r1.json` | `4fe8800df89a9d2fb2da28261ca432ec7391b8ea9590ff335574eecc9d4faa5e` |
| `../apple-gpu-v3/qualify-transpose-combined-r1.json` | `5a69471ab4fffdbefd91421c273d0959e55ac40b015224e5bbd6ef0e786a3cd5` |
| `../apple-gpu-v3/state-error-aggregate-r1.json` | `b1f62d25d3dfa2280da4da97af4e5d6aab23f41a18207469aa137d88bfd82e24` |
| `../apple-gpu-v2/mps_decoder_before.py` | `72d51d8052d8376186730fb4cd2a0e5770eb717b37fb2a5deecca2f3ac689890` |
| `candidates.py` as inspected | `adbfddcf65360537e7fb7035a818ba1754b5ee5f4383c40a33cb398bc190c43f` |
