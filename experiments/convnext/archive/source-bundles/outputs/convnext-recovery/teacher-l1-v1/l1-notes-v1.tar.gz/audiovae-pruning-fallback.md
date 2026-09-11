# AudioVAE2 pruning as a fallback

**This is a credible, lower architecture-risk fallback if the current bounded loss experiment fails.** Start with the working AudioVAE2 decoder and inherited weights, then remove expensive channel groups while distilling from an untouched teacher. Preserve the frozen encoder, 64-channel 25 Hz latents, causal timing, upsampling schedule, final waveform convolution and tanh. The existing encoder interface remains unchanged. No fallback model or experiment was created for this review.

The advantage is a known good starting function and naturally aligned stages. Unlike the frame-domain student, the thinner decoder initially retains the teacher's mechanisms for waveform-rate refinement and bounded output. Pruning can still damage those mechanisms. This lowers uncertainty about topology and alignment; it does not establish that the retained capacity is enough.

## Where the useful reduction is

An archived Intel profile attributed 41.13% of kernel time to standalone matrix operators and another 45.32% to fused residual/upsampling stages containing matrix work. Those fused costs are not independently decomposed. This is historical diagnostic evidence, not a fresh benchmark of the final streaming package. [Saved profile report](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/intel-iteration2/report.md:28).

The matching static AudioVAE2 audit counts **8.991 GMAC per output second**, excluding activations, normalization, packing and memory movement:

| Decoder arithmetic | Share of counted MACs |
|---|---:|
| Upsampling matrices/convolutions | 56.12% |
| Residual pointwise matrices | 42.09% |
| Residual depthwise convolution | 1.63% |

The 1.2, 6 and 12 kHz stages together account for 71.27% of those MACs. The **6 kHz, 256-channel stage is the largest at 30.97%**; the final 48 kHz stage accounts for only 4.19%. Therefore begin with dependency-aware channel reduction in the expensive middle stages, not wholesale removal of the final high-rate stages or Snake. These are static operation shares, not predicted CPU-time savings. [Counted shapes and formulas](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/intel-iteration2/data/architecture-costs.json).

For an affected square pointwise matrix, retaining 75% of both channel dimensions leaves about 56% of its MACs. This illustrates the leverage of width reduction; it is not a recommended uniform pruning percentage or an end-to-end speed estimate. Activations generally scale linearly with width, and fixed overhead remains. Widths must also suit the actual CPU kernels and streaming chunk sizes.

## How to make the fallback controlled

1. **Copy the trained decoder first.** Establish exact unpruned output and streaming-state parity under the authenticated teacher execution contract. Keep the original teacher and current frame-domain candidate intact. This is a separate fallback lineage.
2. **Prune coupled channel groups, not arbitrary individual weights.** Residual additions, depthwise groups, pointwise rows/columns, transposed-convolution channels, Snake parameters and streaming buffers must agree. Sparse zeros in an otherwise unchanged dense matrix do not automatically accelerate CPU execution. Materialize effective weight-normalized tensors before slicing, then reconstruct their normalization correctly. The source contains these coupled structures. [Residual block](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:75), [upsampling and output](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:176).
3. **Choose removal by measured reconstruction importance.** A first-order Taylor score from teacher-distillation loss at an exact teacher copy is zero, so it cannot rank channels. Group-removal impact or an appropriate activation-reconstruction criterion is needed. Low average activation alone is not proof that a channel is dispensable: small contributions can participate in silence cancellation. Include quiet, peaks, speech and expressive material in the training-only importance analysis.
4. **Distill corresponding stage outputs and the final waveform.** Identical strides give matching temporal grids without interpolation. If channel widths change, compare explicitly retained teacher channels or use a small training-only channel projection. Equal clock rates do not make differing feature dimensions interchangeable. Keep final teacher waveform L1 and spectral supervision primary, with calibrated auxiliary stage losses. Remove projections from deployment.
5. **Recover the complete chain, not just isolated blocks.** Local teacher-input fitting can initialize a replacement, but final recovery must feed each stage the student's own upstream activations. Otherwise teacher-forced local success may disappear when approximation errors accumulate. Allow the affected neighboring layers to adapt and retain end-to-end waveform, quiet and peak checks.

The published **DepGraph** method and its maintained **Torch-Pruning** implementation are relevant scaffolding for coupled structural removal. Their results concern other architectures/tasks; custom causal wrappers and channel parameters still require an explicit dependency audit. They do not certify this decoder. [DepGraph](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html), [author implementation](https://github.com/VainF/Torch-Pruning).

**FitNets** provides the direct precedent for intermediate teacher hints with learned dimension mappings. It supports the methodology, not a promised AudioVAE2 quality or speed result. Here, inherited topology makes temporal correspondence clearer than our current cross-architecture feature hint. [FitNets](https://arxiv.org/abs/1412.6550).

## Recommendation and limits

Keep this as the fallback after the current L1 experiment. If pursued, begin with one conservative coupled-width candidate centered on the 6 kHz stage, recover it against the teacher, and compare its complete single-thread CPU streaming runtime and the same quality panel. Continue only if it yields a material gain with preserved quality. Do not prune every layer merely for symmetry or combine low-rank factorization, activation replacement and quantization in the first candidate.

This approach should make failures easier to localize and preserve a stronger starting point. It still retains waveform-rate computation, so it may have less ultimate speed potential than a successful frame-domain decoder. Neither 4× CPU acceleration, Supertonic-like RTF, reduced training-data requirements nor teacher parity follows from pruning papers or MAC counts. The teacher's tanh bounds output, but does not guarantee phase, silence or perceptual fidelity after pruning. The loss-balance and validation-coverage findings remain relevant to either architecture.
