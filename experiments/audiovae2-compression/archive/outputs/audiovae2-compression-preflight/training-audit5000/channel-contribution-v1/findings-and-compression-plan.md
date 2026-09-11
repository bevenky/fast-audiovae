# Recovering the teacher function after compression

The completed experiment rules out the isolated first-mixer correction as a useful decoder repair. It supports reconstructing the affected computation across a whole block, accounting for changed upstream inputs, before deciding whether to compress further. The procedure below is proposed, not running. Main training and all trained checkpoints remain unchanged. The later [distillation review and next step](distillation-next-step.md) refines the immediate priority to a controlled intermediate-hint comparison on the preserved trained student; the reconstruction procedure remains applicable to future cuts and initializer diagnosis.

## What the experiment established

The tested model was the original untrained, teacher-derived sliced initialization. It was not the adapted step-5,000 model. Channel widths were reduced at stages 2 and 3; no residual units were deleted. The original causal geometry, all nine residual units in the group, final waveform convolution and tanh remain.

We fitted the first stage-2 pointwise mixer on 72 calibration recordings and evaluated it on 96 separate development recordings. The fixed linear correction reduced omitted-contribution squared error by 76.67% on calibration and 75.86% on development. All 96 development sources improved that local score. Near-silence local error fell 93.32%. Nevertheless, final waveform MAE worsened on 61/96 sources and quiet error on 59/68 supported sources. Supplying the exact first missing contribution also produced negligible overall waveform improvement. Later omissions and their interactions remain consequential even when this first location is repaired.

Each following intervention removes the omitted-input contribution from one operation of the otherwise complete original teacher. It keeps the other teacher channels and downstream operations. The interventions therefore do not reproduce an entire pruned student, and their different coordinate scopes are not an intrinsic ranking of layer importance.

| Intervention on the same 15 diagnostic recordings | Waveform MAE against teacher | Quiet failures |
|---|---:|---:|
| Remove first stage-2 pointwise contribution | 0.004584 | 401/595 |
| Remove stage-3 upsampler contribution | 0.014881 | 595/595 |
| Remove stage-4 upsampler contribution | 0.028790 | 595/595 |
| Restore all eight omitted contributions to sliced initialization | 0.0000000132 | 0/595 |

The eight-term restoration uses original teacher features that are unavailable to a standalone small decoder. Its near-exact result validates the accounting; it is not an efficient inference solution or proof that the narrower student can synthesize those features itself.

The later upsamplers are therefore justified priorities for a reconstruction probe. Stage 4 is particularly consequential under the measured intervention. This does not uniquely identify the cause of residual errors after 5,000 training steps.

## The signal changes are observable

The saved [waveform plot](results/channel-waveforms.png) shows stationary near-silence at 10.24 seconds, a Kannada onset and a whistle interval. It includes original teacher interventions and the untrained pruned candidates in separate columns. The [snippet measurements](results/snippet-analysis.json) use the original amplitudes and absolute source offsets.

In the Spanish interior near-silence interval, removing the stage-3 contribution introduces a residual mean of about `2.06e-5`, with much smaller varying error (`1.60e-6` RMS). Removing the stage-4 contribution introduces both a mean shift (`-7.48e-5`) and varying residual (`7.66e-5` RMS). Its residual repeats closely at a 240-sample lag, with correlation 0.99875. This is an observed periodic effect of that intervention; it does not identify one erroneous padding operation or prove the origin of every trained-model artifact.

Around the whistle peak, the original waveform RMS is 0.02520. Removing the first-mixer, stage-3 and stage-4 contributions reduces it to approximately 0.02211, 0.01674 and 0.00528 respectively. This demonstrates different effects on an actual expressive signal rather than only hidden-feature error.

The signed contribution cross terms do not support a universal cancellation explanation. At the first mixer and stage-3 upsampler the pooled retained and discarded contributions predominantly reinforce each other. Stage 4 has cancellation in some quiet regions, but reinforcing contributions in pooled near-silence. Restoring correct gain, offsets and phase-dependent responses requires matching the full function; a special silence offset cannot solve all these effects.

## A repeatable compression procedure

The intended contract is: a cheaper student block receives the same latent stream and causal history, and reproduces the original block's externally visible output closely enough that the remainder of the decoder produces equivalent audio. Internal channels may acquire new meanings. They need not individually match an arbitrarily selected teacher coordinate.

1. **Define a block contract before a cut.** Record input/output widths, temporal rates, sample alignment, conditioning, causal state and history. For the current block, retain the exact stage-2 input and complete 128-channel stage-4 output as the main reference boundaries. Freeze the encoder and original teacher. The 64-channel latent interface is already shared; no separate decoder latent-prediction objective is needed.
2. **Choose cuts by recoverable output error and CPU benefit.** Activation magnitude or redundancy at one boundary alone is insufficient. Assess how a proposed removed set contributes to downstream outputs, then how well the remaining computation can reconstruct those outputs. Keep structured widths/ranks suitable for the CPU runtime. Predictable, inexpensive-to-reconstruct contributions are better candidates than merely small activations.
3. **Refit the remaining computation on the inputs it will actually receive.** When upstream compression changes an input, fit the next smaller operation using that changed input and the original target output. A fit performed only on teacher inputs can become invalid when connected to the student. Fit linear operators with a constrained least-squares initializer where practical; fold the result into their existing effective weights and bias. No additional inference operation is required for those fits.
4. **Respect the whole residual and temporal operation.** For `output = skip + branch`, fit the branch toward the desired complete output minus the actual student skip. For transposed convolutions, preserve current/previous-frame contributions, every output phase, right trim and one shared output bias. A generic phase-averaged pointwise fit is not equivalent. Refresh downstream fits whenever upstream weights change.
5. **Recover the complete block jointly.** Local fits initialize the candidate. Then optimize all retained parameters inside the block against its complete teacher boundary and final waveform/mel objectives, passing gradients through the frozen suffix. These losses already exist in our training code; the proposal improves selection, initialization and ordering rather than inventing more losses. Local feature scores are diagnostics, not acceptance criteria by themselves. Do not require 0.99 correlation before enabling the recovery objectives.
6. **Accept a cut before proposing another.** Compare against the original teacher and previous accepted candidate with fixed held-out speech, expressive, quiet and onset panels. Preserve the 0.99 active-correlation goal, but also enforce amplitude, waveform, mel, quiet/noise, transient and streaming-continuity checks. Correlation alone can hide severe attenuation. Confirm actual single-thread CPU streaming RTF; parameter count and MAC savings do not establish it. Keep the candidate only when the quality and useful speed criteria are met. A failed recovery prompts a smaller cut, different selected channels, or a different factorization, rather than automatically accumulating more cuts.

This procedure follows the selection-plus-reconstruction principle in [Channel Pruning](https://arxiv.org/html/1707.06168v2#S3.SS2), including its [residual-branch treatment](https://arxiv.org/html/1707.06168v2#S3.SS3). [Asymmetric reconstruction](https://arxiv.org/html/1505.06798v2#S3.SS3) specifically accounts for the changed inputs produced by earlier approximations. [ThiNet](https://arxiv.org/abs/1707.06342) also motivates selecting filters by downstream effects. Their image-network results do not establish achievable audio fidelity or CPU speed for this decoder.

## How the remaining layers can replace deleted work

There are two different recovery mechanisms. A predictable linear contribution can be absorbed into remaining matrices and biases, as our first-mixer fit demonstrated locally. A removed nonlinear or temporal computation cannot generally be folded algebraically across Snake into one convolution. The surviving block must relearn an approximation through its existing nonlinear units and temporal filters. If it lacks enough information, history or capacity, the proposed cut cannot meet the requested quality at that size.

For future whole-block deletion, a six-block student can target the output of the original ten-block teacher chain at the same external boundaries. It should not target only the first six teacher blocks. Deleting dilated temporal units also reduces available history unless another retained operation is redesigned to preserve it; this must be checked before training. Keeping the same strides is insufficient to prove preserved temporal information.

If width pruning proves too damaging, a separate option is to retain full-width nonlinear/residual states and factor expensive linear matrices into smaller dense factors. Two consecutive linear factors can approximate one large matrix without a new activation, but they still add a matrix call and retain more channelwise work. This is a candidate with a measurable rank/quality/CPU tradeoff, not free computation or a guaranteed solution.

For cuts inside an already trained student, hidden coordinates may have co-adapted. Local initialization can preserve the pre-cut student's complete block function at an unchanged boundary, while the original AudioVAE2 remains the final complete-group and waveform reference. Alternatively, enlarge the fitted region to an externally aligned boundary. Do not paste original teacher internal-channel corrections into the step-5,000 representation and assume their meaning is unchanged.

## Specific next experiment

Keep the current trained checkpoints. Reject the isolated first-mixer patch as a decoder repair.

Use a separate teacher-derived candidate to test reconstruction of the **complete stage-4 upsampler output** from the actual narrowed prefix input. Keep its native stride-2, four-tap, causal operation and fit its existing weights and bias. Score both its output and the final waveform through the unchanged remaining operations. This first probe tests whether the large measured downstream effect can be recovered at the current width.

If it is useful, reconstruct stage 3, then fit stage 4 again because its inputs have changed, and perform whole stages-2-to-4 recovery. If the first probe is inadequate, compare reconstruction from retained original teacher inputs with reconstruction from actual student inputs to distinguish loss of useful input information from accumulated upstream drift. Neither an unsuccessful linear probe nor the previous first-mixer result establishes a hard nonlinear capacity limit.

Do not remove further blocks or resume a long run until this compression-and-recovery procedure has demonstrated a useful result. The next experiment is not launched by this report.

## Integrity and supporting records

The successful diagnostic passed 32 focused tests on Runpod. All 420 source evaluations and the final post-intervention check matched cached teacher targets bitwise. Original teacher, checkpoints and untouched student states were preserved. The 72 calibration and 96 development source sets are disjoint, but the development set has been inspected previously and is not a fresh final qualification set. No new RTF or listening/MOS benchmark was run.

The first attempt stopped on an inconsistent absolute-only input-alignment check. The measured drift was small FP32 rounding; a documented signal-scaled check was used for the fresh successful attempt. No audio-quality threshold changed. See the [numerical amendment](alignment-gate-amendment.md), [execution audit](execution-audit.md), [independent quantitative audit](independent-analysis.md), and [raw results](results/).
