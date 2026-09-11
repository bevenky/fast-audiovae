# Additional pruning and distillation methods

## Current decision

Run B's reconstruction-aware initialization through the original joint recovery recipe, using the same first 24,000 training sources and the same frozen teacher. Keep the constrained startup experiment separate. This measures the change already supported by our own initialization result.

The original 5,000 checkpoint is a preserved baseline with a recorded recipe, source ledger and RNG state. A fully repeated training trajectory has not been demonstrated. A better B start does not by itself establish faster convergence or a better 5,000-step endpoint.

## What the papers add

| Method | Relevant mechanism | Application here |
|---|---|---|
| Few Sample Knowledge Distillation, CVPR 2020 | Fit compressed block responses with a small calibration set and fold the correction into convolutions | Supports reconstruction-aware initialization. Our native refits already provide this capability without a separate adapter |
| Progressive Network Grafting, AAAI 2021 | Train replacements in the surrounding teacher network, then connect and adapt them jointly | Supports retaining the original suffix while training the replacement stages against final output, then coadapting the group |
| Optimal Brain Compression, NeurIPS 2022 | Select removals using reconstruction sensitivity and compensate surviving weights | The useful future extension is better selection of which shared channels to remove, not merely refitting after the current selection |
| FocalCodec-Stream, 2025 | Adapt a difficult component, then train connected components to close the resulting mismatch | Relevant audio precedent for staged recovery; its causalizing encoder/refiner design is not a direct startup fix for our decoder |
| GRAIL, CPAL 2026 | Reconstruct reduced hidden features and fold the linear map into consumer weights | Closely related to B's direct output reconstruction, with different regularization and temporal constraints |
| Two-stage reconstruction, ESWA 2026 | Concentrate information before pruning, then reconstruct accumulated output errors | Pre-pruning preparation is a distinct possibility; accessible primary information is insufficient for implementation |

**FSKD:** its foldable 1×1 calibration provides a close precedent for efficient initial recovery. It does not establish a need for another layer here: our pointwise and native upsampler coefficients are already fitted directly. Its published results concern image networks. [Paper](https://arxiv.org/abs/1812.01839), [official implementation](https://github.com/LTH14/FSKD).

**NetGraft:** replacement blocks are trained using the surrounding teacher network and final output objective, then connected for joint adaptation. Our original suffix and waveform/mel losses implement the relevant functional principle; this is not a claim that our complete recipe reproduces NetGraft. [Paper](https://arxiv.org/abs/2012.04915), [official implementation](https://github.com/zju-vipa/NetGraft).

**Optimal Brain Compression:** its reconstruction-sensitive removal and compensation are relevant to the remaining selection gap. Its public implementation supports sparse weights and structured sparsity patterns. Adapting it to one shared set of channels across stage 2's three residual units and stage 3's upsampler would be new work. It is not a ready-to-use causal audio channel selector. [Paper](https://arxiv.org/abs/2208.11580), [official implementation](https://github.com/IST-DASLab/OBC).

**FocalCodec-Stream:** the specific distribution shift arises while causalizing WavLM. Its multi-stage recovery is relevant, but adding its refiner or lookahead would change our CPU/causality contract. It does not show that every internal feature must match exactly or that startup silence is automatically preserved. [Paper, section 2.2](https://arxiv.org/html/2509.16195v1#S2.SS2).

**GRAIL:** the full methods explicitly equate folded hidden-feature ridge reconstruction with direct consumer-output ridge least squares. Sequential statistics account for upstream pruning. This supports B's basic principle, but the solvers are not identical: B uses centered FP64 delta ridge, fits a shared bias, subtracts the current residual skip, and preserves native current/previous-frame and five-phase constraints. GRAIL's convolution formulation shares a channel reconstruction across taps. Its vision/language evidence and stated distribution-shift limitations do not establish audio startup preservation. I inspected the full paper and author compensation code; neither provides a reason to replace B's solver during this comparison. [Full methods](https://arxiv.org/html/2602.23795v2), [published paper](https://proceedings.mlr.press/v328/tang26a.html), [official code](https://github.com/TWWinde/GRAIL_Compensation/blob/main/grail-llm/lib/compensation.py).

**Two-stage reconstruction:** the primary abstract describes concentrating information in retained components before pruning, followed by reconstruction that accounts for accumulated layer errors. The pre-pruning stage differs from B's post-cut refits; post-pruning global recovery overlaps the current training objective. This assessment relies on the primary abstract and accessible introductory/section snippets, not verified full methods. I could not retrieve an open full method or author-linked implementation in the bounded search. Exact penalties, optimization schedule and transfer to causal audio remain unverified, so this is not ready-to-use code or a demonstrated startup fix. [Primary article](https://www.sciencedirect.com/science/article/pii/S0957417425025473), [DOI](https://doi.org/10.1016/j.eswa.2025.128930).

None of these papers establishes our codec's quality or CPU RTF. They support methods to test, not a guarantee of 99.99% reconstruction.

## The remaining selection gap

The actual current selector samples stage-output features from calibration, accumulates their Gram matrix, and uses a pivoted subset. It does not choose the subset by directly measuring the errors induced in all four downstream mixing operations or by explicitly preserving startup behavior. B now reconstructs those operations after selection, but selection itself is unchanged.

A later reconstruction-aware selector would score coordinated channel deletions by their downstream output effect after compensation. Startup and normal audio would remain separate measured constraints. That could preserve useful channels which a feature-only criterion undervalues, at the same final width and inference operation count. It should be compared after the current B recovery result, not mixed into this controlled run.

## What the startup experiment has now shown

The constrained-A experiment passes all 13 observed development startup windows with the same native upsampler. The remaining width can represent a passing response for these cases; restoring channels or adding an inference layer is not necessary for this observed subset.

However, that fit worsens ordinary waveform and quiet reconstruction. Startup constraints alter weights shared throughout the recording, and exact internal equality is stronger than the actual waveform requirement. The result is evidence of a tradeoff under this fitting objective, not a free repair.

Do not transplant constrained-A weights into B or the adapted 5,000-step model. Their inputs differ. Any later combination must fit on the chosen candidate's actual inputs. Training may also move away from an initially valid startup solution, so validation must continue measuring startup independently.
