# Direct channel-contribution diagnosis

The user approved measuring omitted contributions, testing controlled restorations, and checking whether retained features can predict the useful missing contribution. This is isolated diagnosis and affine fitting. Main neural training stays paused; original checkpoints and all teacher parameters remain unchanged.

## Decisive locations

1. Stage 2 residual unit 1 pointwise mixer. At original sliced initialization its retained inputs still match the teacher, so its first missing contribution can be identified without confounding earlier approximation.
2. Stage 3 transposed convolution, where both input and output channel sets shrink.
3. Stage 4 transposed convolution, where input channels shrink but the full output width remains.

For each linear operator, separate the selected teacher output into kept-input contribution, dropped-input contribution and one output bias. Use the actual post-Snake inputs and effective normalized weights. Preserve native transposed-convolution input/output axes, taps, phase overlap and causal trim. For later operations, separately account for drift in the retained upstream input. Verify the additive identity numerically before attributing any effect.

Report signed mean, AC and RMS contributions and their combinations for teacher-defined quiet/near-silence, startup, active and expressive regions. Avoid ratios with a nearly zero teacher waveform denominator. Use full causal history and valid-tail weights.

## Counterfactuals

- In the otherwise original teacher, remove one measured omitted-input contribution from the selected output coordinates and propagate through the original fixed downstream network. Other coordinates remain untouched. This measures that term's influence, not the effect of pruning and retraining the whole decoder.
- At the first verified matching mixer of a fresh sliced student, restore its exact missing term while leaving the remaining initial student fixed. This is a teacher-dependent oracle, not a deployable repair.
- If implemented, an all-omitted-term oracle at the six affected residual mixers and two later upsamplers should recover the original selected intermediate trajectory and full group output. It is an accounting control, not an efficient student.

Never inject selected teacher-coordinate corrections into the adapted step-5,000 representation and describe that as a valid repair. Its internal features have learned different meanings.

## Local reconstructibility experiment

Fit only the first stage-2 residual mixer in this initial bounded experiment. Use all valid stage-2 cells from the 72 original calibration recordings, with weights equal to their valid waveform-sample support, from zero to 40. The 96 development recordings are disjoint and never select fitting parameters.

Let x be retained input features and d the exact omitted pointwise contribution. Accumulate float64 weighted means and centered covariance/cross-covariance. Fix ridge strength before scoring as `lambda = 1e-6 * trace(Cxx) / 256`. Solve the affine prediction `d_hat = A*x + c`, with `c = mean(d) - A*mean(x)`. Report rank, raw and regularized condition numbers, solve residual and coefficient magnitudes. If input variance is zero, flag an intercept-only fit instead of dividing by zero.

Add A to the existing retained pointwise effective matrix and c to its bias, then correctly reconstruct weight-normalization parameters. No new inference layer, activation or phase-dependent bias is introduced. Only this isolated copied candidate is changed.

Compare plain sliced initialization, exact first-term restoration and folded affine correction on the fixed 96-source development panel. Report local omitted-term prediction separately from downstream waveform, mel, whole-group and regional errors. A local fit may be predictable yet fail to improve the full waveform because later omissions remain. Failure of this one fit is not evidence that the entire narrower architecture is incapable of learning.

## Integrity and outputs

Before Runpod execution, require independent review and focused tests for the operator decomposition, phase/bias semantics, valid masks, fitting split and effective-weight folding. Validate original teacher/cache consistency, protected file hashes and frozen states. Keep all outputs separate from the main TensorBoard run and training checkpoints. Save compact signal traces or plots for the actual quiet/onset/whistle examples so conclusions can identify signal behavior rather than rely solely on summary scores.

No new neural optimizer, long training continuation, architecture widening, commit or checkpoint promotion is authorized by this diagnostic protocol.
