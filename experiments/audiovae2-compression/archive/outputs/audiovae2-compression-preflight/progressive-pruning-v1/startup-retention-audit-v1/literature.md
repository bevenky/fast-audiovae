# Startup retention after pruning: evidence and method review

11 September 2026. Read-only review of saved reports, the declared protocols and original papers. No model execution, optimizer update, training-source change or threshold change was performed for this note.

## What our results establish

The current 384/256 student can represent a response that passes all 13 measured startup windows: constrained-A and combined-C initializations did so using existing native operators. C then lost that behavior during ordinary joint recovery, ending at 0/13 after 2,500 updates. D started at 11/13 and was at 0/13 at every later saved review. G never passed all 13, but its 2,000-update startup residual RMS was 13.163 microFS versus B's 38.457. These observations separate initial feasibility from its retention; they do not establish a capacity ceiling or identify which loss/parameter caused the drift. [C initialization](../combined-recovery-v1/initial-findings.md), [C completion](../combined-recovery-v1/findings.md), [D completion](../downstream-selection-v1/completed-findings.md), [G completion](../grail-hidden-v1/completed-findings.md).

C's fitting stream contained 1,866 near-silent first-20ms onsets among 30,000 sources. They contributed 0.050775% of scored waveform samples. Startup was present, but rare by sample exposure. That fraction is not its parameter-gradient contribution: the waveform, mel and hidden-feature reductions and Jacobians differ. It does not prove that increasing a quiet weight is the right remedy. [Exposure evidence](../combined-recovery-v1/training-onset-coverage.json).

Q failed before its first completed update because its differentiable constraint did not exactly reproduce its no-gradient pre-state. It has no trained quality result. This is a separate implementation/numerical question, not evidence for or against constrained recovery. [Q failure](../quiet-preservation-v1/failure-review.md).

## Relevant primary results and their limits

| Source | What it actually contributes | Transfer to this decoder |
|---|---|---|
| [GRAIL, section 3.1 and section 5](https://arxiv.org/html/2602.23795v1#S3) | A post-activation hidden reconstruction map is fitted on calibration statistics and folded into consumer weights, including a shared channel map across convolution taps. It is a one-time, training-free compensation. The authors identify activation-distribution shift and local block scope as limitations. | Supports our folded native initialization. It does not supply a rule that preserves startup after subsequent Adam updates. Its vision/language results are not codec-startup evidence. Our sequential actual-student-input and causal-transpose-convolution adaptation remains GRAIL-like. |
| [Gradient Episodic Memory, section 3, equations 5–11 and Algorithm 1](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) | Stored past examples define loss inequalities. The method projects a proposed gradient through a small dual quadratic program. Relating gradient signs to retained loss assumes local linearity and representative memory. | A useful precedent for finite auxiliary constraints. Our Q omits persistent memory, uses physical quiet limits and projects an actual Adam displacement. It therefore tests a different, current-batch policy. |
| [Orthogonal Gradient Descent, section 3, equations 7–9](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf) | Projects updates against stored gradients of model outputs, rather than only loss gradients. The paper explicitly discusses near-zero loss gradients at well-fitted examples and approximating current output Jacobians by those stored at an earlier solution. | Explains why tiny reconstruction gradients need not mean startup is insensitive. A stored startup subspace would avoid repeated input access, but its local approximation can become stale during substantial recovery. It is not an exact retention guarantee. |
| [Functional Regularisation of Memorable Past, sections 2–3, equation 8](https://papers.nips.cc/paper_files/paper/2020/file/2f3bbb9730639e9ea48f309d9a79ff01-Paper.pdf) | Regularizes predictions at a selected memory of inputs, with a Gaussian-process-derived weighting. Those input locations remain involved in later updates. | Supports protecting the function at training-only anchors instead of demanding unchanged internal coordinates or weights. A plain output penalty is a possible simpler transfer, but still needs a coefficient and provides a soft tradeoff. The full GP machinery is not justified by our present evidence. |
| [Adam, Algorithm 1](https://arxiv.org/pdf/1412.6980) | The displacement uses bias-corrected first and second gradient moments and per-coordinate scaling. | A raw gradient inner product is insufficient to predict this implementation's change in startup. Measure the actual proposed parameter displacement, including its optimizer state. This is not evidence that Adam itself is faulty. |

No cited paper establishes a causal-audio startup solution under our exact native geometry, current source policy and fixed losses. Their value here is a mechanism and a testable limitation, not a guarantee of teacher parity or CPU speed.

## Why a passing fit need not stay passing

Let `f_theta(z)` be the complete student waveform for the authentic teacher latent input, with real startup padding and conditioning. For a fixed startup window, write the two evaluator constraints as

`r(theta) = mean((f_theta(z) - y_teacher)^2) - residual_limit^2 <= 0`

`a(theta) = mean(f_theta(z)^2) - output_limit^2 <= 0`.

The implementation evaluates these through its prescribed FP32 RMS arithmetic; the formulas describe the mathematical quantities, not permission to change arithmetic or thresholds.

An initializer finds one point inside a finite measured feasible set. The existing recovery objective is a weighted combination of global waveform L1, mel and group MSE. It is not an optimization over that feasible set. A lower ordinary objective can coexist with larger `r` or `a` on a rare window. The fixed teacher does not change this: different examples and loss terms share the student's finite parameters. The current results demonstrate the coexistence, not the direction or strength of the individual gradient contributions.

There is also a concrete moving-input issue. A native fit may satisfy `W_up X(theta_initial) + b = Y_teacher` on calibration rows. Joint recovery changes the upstream features `X`, the upsampler, and the downstream trainable residual blocks. Keeping only that original fitted equality or only `W_up` fixed would not generally preserve the complete output. This is a mathematical limitation of the fixed-input fit, not measured attribution to one layer in C's trajectory. Full external waveform targets avoid assuming that adapted internal coordinates still mean the original teacher coordinates.

For an actual update `delta`, a differentiable constraint obeys locally

`F(theta + delta) - F(theta) = grad(F)^T delta + higher-order terms`.

The observable directional test is `grad(F)^T delta_Adam`, not only the cosine between ordinary and quiet gradients. Different coordinate scaling can change that sign; momentum can also make the actual proposal differ from the current gradient direction. If the measured nonlinear difference disagrees with the local prediction, curvature, maximum-window switching, rounding or an incorrect derivative/evaluation path must be separated before interpretation.

At exact reconstruction, `grad(r) = 2 J^T error / n` is zero even if the waveform Jacobian `J` is not. A finite update can then incur approximately `||J delta||^2 / n`. Thus a vanishing residual-loss gradient alone cannot prove preservation. Our C initializer has nonzero residual, so this limiting example is a warning about the inference, not a claim that C's actual constraint gradient is zero.

Slower updates reduce a local displacement; they do not by themselves make the feasible set invariant over many steps. Likewise, preserving a mean/DC term alone leaves possible AC waveform or phase errors. Startup at 0–20ms, the teacher's 20–40ms transient and sustained quiet need separate measurements.

## Exactly what current-batch Q can and cannot establish

The [locked Q protocol](../follow-on-experiments-plan.md) uses up to six maxima: residual and amplitude excess for startup-near, other-near and ordinary-quiet windows present in a batch. It projects the actual Adam proposal and rescans complete present cohorts after candidate displacements. That nonlinear rescan is necessary because a first-order row can miss curvature or a different maximizing window.

Two limitations remain even with perfect arithmetic:

1. A batch without startup has no startup constraint. No condition is imposed on previously passing startup outputs during that update. Protection observed when startup is present cannot establish retention between such batches.
2. If a cohort is already infeasible, its cap is its previous largest positive excess. Holding that maximum below the cap can still allow a different previously passing window to fail. It neither forces all failures to improve nor protects every window independently.

For a fixed, initially feasible anchor set, exact complete-set nonlinear checks after every accepted update could establish preservation **on that finite set**, by induction. They would not establish performance on all possible inputs, the reused development panel or deployment audio. Rejection can also make no useful learning progress; accepted nonzero displacements and ordinary quality remain necessary outcomes.

## Smallest decisive next investigation

First close the numerical question at Q's failed pre-state without retained training. Use identical native weights, input shape, context, targets, valid masks and warmed execution. Compare the waveform and selected constraint index/value across no-gradient and gradient-enabled evaluation; then compare the differentiable scalar to the canonical evaluator on the **same waveform tensor**. This separates a model-forward difference from a reduction/mask/max-selection difference. A tiny discrepancy is not automatically harmless when a guard requires equality; record its magnitude relative to the actual constraint slack. Do not silently loosen the guard or quiet limits.

After the paths agree, a bounded disposable proposal audit is more informative than another long run immediately. At a feasible C state, calculate the unchanged ordinary Adam proposal on a predeclared fitting batch and measure its predicted and actual effect on the existing six calibration startup windows. Include a naturally startup-absent batch as a distinct case. Measure waveform/feature/mel directional contributions separately if needed, but label them local sensitivities, not a unique causal decomposition of Adam's nonlinear combined update. Restore all weights, moments and RNG after each counterfactual.

The decision evidence is concrete:

- An ordinary proposal raises startup error while native no-update parity remains stable: evidence of update-induced loss of that function at that state.
- A linear prediction tracks the change: first-order interference is supported locally. Failure of that prediction calls for finite-step/sensitivity analysis, not an assertion of gradient conflict.
- An anchor-preserving displacement prevents the measured failure while retaining useful ordinary improvement: evidence to justify a bounded retention arm.
- Many zero accepted displacements or damaged ordinary audio: evidence that this particular constraint policy is too restrictive, not proof that the architecture lacks capacity.

## Minimal retention experiment, conditional on that evidence

The simplest extension of existing Q is persistent **training-only startup support**, not another layer or a new initialization family: use the same six calibration onsets and unchanged teacher/evaluator bounds at every update, refresh their constraint gradients at the actual current weights, project the same Adam proposal, and verify the complete finite set nonlinearly. Keep ordinary quiet treatment separate, the C initialization fixed, all 90 group parameters trainable and the original ordinary objective unchanged. Do not fit against the 13 development starts or transplant weights into an already adapted checkpoint. Compare a declared fresh 2,000-update arm with the matched C trajectory and report all prescribed reviews.

This extension has a source-policy consequence. Repeated anchor forwards that influence accepted updates are repeated use of those training/calibration inputs, even if they are excluded from the ordinary loss and source stream. They cannot be called mere monitoring or silently substituted for current-batch Q. The user's earlier permission to reuse downloaded data during debugging and current instruction to execute relevant experiments authorize a declared experimental anchor trial; the previous plan's narrower policy does not create a new permission barrier. Amend the experimental protocol explicitly, keep the ordinary stream at 24,000 distinct sources used once, and report recurring calibration-anchor uses and their compute in a separate ledger. Never use development inputs to construct constraints. If a later final-training policy excludes recurring inputs, OGD-style stored derivative information is a possible approximate alternative, with its stale-Jacobian limitation stated; it is not equivalent protection.

All these mechanisms are training-only and can retain the existing exported graph. None guarantees zero practical runtime difference without a benchmark, and none yet demonstrates satisfactory startup plus ordinary fidelity. Preserve C, D, G and Q's failed evidence; diagnose the measured path before deciding which pending recovery experiment to execute.
