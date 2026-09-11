# Follow-up audit of the fusion objective screen

Read-only audit of the frozen source, saved scalar logs and existing reports. No inference, backward pass, optimizer update or checkpoint modification was performed. The parent remains step 8,090. The six scalar logs contain 200 updates each, steps 8,091–8,290. The comparison tail contains the parent's saved steps 7,891–8,090. Every audited implementation file matches the hash in the sealed experiment identity.

The main missed control is **gradient-scale adaptation after changing an objective or discriminator**. Preserving the old balancer was faithful to the original optimizer state, but it was not neutral when the meaning and magnitude of the loss changed. The complex discriminator result is principally a screen of a nearly reconstruction-only update mixture, not a convincing test of complex adversarial training's potential.

Evidence with per-file hashes and first/last/aggregate scalar statistics: [objective-followup-scalars.json](objective-followup-scalars.json). Existing quality results remain valid measurements of those exact trained checkpoints; this audit changes their interpretation, not their values.

## 1. The nominal loss shares did not survive the objective changes

The balancer computes each loss gradient with respect to the output waveform, masks invalid samples, averages per-example L2 norms, then divides the desired component norm by its historical exponential mean. Its decay is 0.999. The migration explicitly requires that all prior balancer state remain unchanged. The fresh-discriminator warmup changes discriminator weights, but does not recalibrate these gradient means.

| Arm | Waveform | Mel | Feature matching | Adversarial |
| --- | ---: | ---: | ---: | ---: |
| Declared target | 30.00% | 40.00% | 20.00% | 10.00% |
| Parent's preceding 200 updates | 29.44% | 39.28% | 20.90% | 10.39% |
| Control | 30.57% | 39.27% | 20.22% | 9.94% |
| Tanh | 30.43% | 39.10% | 20.38% | 10.10% |
| Short mel | 34.13% | 31.50% | 22.93% | 11.45% |
| Output filter | 30.40% | 39.27% | 20.27% | 10.06% |
| Fresh magnitude | 38.69% | 50.29% | 10.12% | 0.90% |
| Complex multiband | 43.17% | 56.56% | 0.148% | 0.122% |

These are means of each batch's **individual scaled output-gradient norm divided by the sum of those norms**, before the vectors are added. They are not parameter-gradient shares, effective optimizer-update shares, or causal contributions to quality. Rounding can affect totals.

On the first paired update, control and complex receive identical waveform and mel losses. But complex feature-matching raw norm is 0.000828 against an inherited EMA of 0.458092. Its scaled norm is 0.000362, versus control's 0.250872. Complex adversarial raw norm is 0.169226 against EMA 19.503728; scaled norm is 0.000868 versus control's 0.136267. This is a direct observed scale mismatch, not a hypothesis about a chart.

After 200 updates, the EMA numerator still retains 81.86% of its old contribution. At the parent's nearly settled EMA weight, its normalized historical mass is approximately the same. The half-life is 693 updates. Complex's combined FM/adversarial share averages just 0.270%, versus the intended 30%; in the final 20 updates it is still only 0.440%. Fresh magnitude is also underweighted, but much less severely: its average combined perceptual share is 11.015%. Therefore even the fresh-magnitude comparison does not match actual objective balance across the two discriminator families.

No balancer scale bound was hit in these logs. This is inherited normalization, not a coefficient-cap failure. A positive finite GAN loss and `perceptual_fraction=1` do not establish that its gradient had meaningful influence.

Relevant frozen source: `gradient_balancer.py:208–244`, `fusion_migration.py:182–203`, `recipe_v2.py:165–191`, `run_fusion_screen.py:159–181`.

## 2. Short mel tested three changes together

The old spectral loss is `(L1024 + L2048 + L4096) / 3`. The candidate is `(L256 + L512 + L1024 + L2048 + L4096) / 5`, with respective mel bins 16, 32, 64, 128 and 128. The new 256/512 scales have 5.33/10.67 ms windows and 1.33/2.67 ms hops at 48 kHz. Each old scale's scalar coefficient falls from one-third to one-fifth. Thus the test combines shorter analysis, redistribution away from existing scales, and the old gradient-normalization state.

The first update makes this visible without any model divergence: control mel raw norm is 0.401186, candidate 0.267856; the inherited normalization makes their scaled norms 0.496820 and 0.331845. Across the entire screen the mel share is 31.50% versus control's 39.27%, and the other branches gain relative influence. This was a valid declared five-scale replacement, but not an isolated test of shorter windows at matched effective spectral influence.

The candidate's average scalar training mel loss, 0.85136 versus 0.96624, cannot be read as a 12% quality improvement because the objective changed. The report correctly uses the unchanged evaluation criterion for quality comparisons. Its small fixed-mel gain and quiet regression warrant holding **this weighting and initialization**, not ruling out short-time supervision generally.

A cleaner follow-up would first measure per-resolution gradient norms and directions, then compare a declared short-scale mix against the old mix with comparable total spectral influence. Keeping all old scalar coefficients and blindly adding full-strength terms is not automatically fair either: it changes the total budget. State explicitly whether a trial tests redistribution within a fixed budget or additional supervision at an increased budget. Multiplying the entire mel branch by a constant is not a durable fix under a balancer that eventually divides by that branch's EMA.

Relevant frozen source: `fusion_objectives.py:24–50`, `reconstruction_v2.py:101–115, 142–157`.

## 3. Complex input does not guarantee teacher-aligned phase

The complex MRD receives raw real and imaginary STFT channels, so it can detect phase-related structure that an absolute-magnitude input discards. However, adversarial real/fake scores are evaluated on each waveform independently; the discriminator is not conditioned on the paired teacher waveform or encoder latent. Its GAN term encourages a plausible distribution, not equality of each prediction's phase to its teacher.

The feature-matching term **is paired**: it compares aligned student and teacher hidden activations. It can give phase-sensitive paired feedback when the complex discriminator has learned useful features and receives a meaningful gradient share. It is not equivalent to a direct complex-spectrum reconstruction loss, and neither it nor GAN implies 0.99 waveform correlation. The raw waveform MAE already supplies an exact paired, phase-sensitive training signal.

The complex candidate additionally changes channels from 16 to 32, five band-specific networks, representation scaling and feature structure. Its comparison tests that complete discriminator family, not real/imaginary channels alone. Both fresh arms receive 20 full-discriminator warmup updates, followed by 200 joint updates, compared with 7,590 prior discriminator updates in the original parent. The MPDs retain their prior training, but new spectral heads do not. Calling the warmup 'only new MRD optimization' would be inaccurate because the code updates trained MPDs too; this is matched between the fresh pair.

A direct paired complex-spectrum loss is a possible separate training-only candidate if phase mismatch remains after gradient diagnosis. It would need amplitude-aware scaling and evidence beyond the current waveform loss; adding it now would introduce another objective without establishing the current one's effective action.

Relevant frozen source: `discriminators.py:168–211`, `fusion_objectives.py:116 onward`, `run_fusion_screen.py:159–181`.

## 4. Clipping is pervasive, but its effect is not established

All 200 generator and all 200 discriminator updates in every arm exceed the norm-1 clipping threshold. Control median preclip norms are 313.68 for G and 29.09 for D. Complex medians are 313.18 and 21.14. The median multiplier applied to control's parameter gradients is about 0.00319 for G and 0.03437 for D. The preceding parent tail also clips on every update.

These values deserve instrumentation, but they do **not** mean learning is 314 times slower. Muon orthogonalizes matrix updates, and AdamW scales with moment estimates. Uniform clipping can leave much of the adaptive update scale unchanged while still changing direction across time through moment accumulation and relative behavior across optimizer groups. No actual per-group parameter displacement, gradient-direction agreement or clipped-versus-unclipped update comparison is logged. Raising the threshold without measuring that is not a supported big win.

The near-identical preclip generator norms across control and complex also do not imply identical useful gradients. Their component mixtures demonstrably differ. Compare update/weight norms and directions separately for Muon matrices, AdamW matrices, output head, biases and normalization affine parameters before changing clipping or optimizer settings. Any counterfactual update inspection must use isolated cloned optimizer state, not modify a preserved checkpoint.

Relevant frozen source: `recipe_v2.py:175–192`, `distillation_training.py:43`, `optimizers.py:build_optimizer_bundle`.

## 5. The current logs cannot explain quiet-versus-loud allocation

Current waveform MAE uses a global count of valid samples. For each nonzero residual its derivative magnitude is `1 / valid_sample_count`, regardless of signal level. It does not have the old inverse-RMS amplification and does not remove quiet samples. Short tails contribute proportionately by valid length. This is a verified correction already present in the recipe.

The log-mel term has magnitude-dependent derivatives above its floor, while the clamped log branch has zero derivative below the floor. There is still a linear-magnitude term and waveform term there. In control, log mel accounts for about 99.0% of the **scalar** mel loss, but this is not evidence that it accounts for 99% of the gradient. The saved balancer records only the sum of linear/log/resolution gradients per example and then averages example norms. It cannot identify which level range dominates, whether quiet loss gradients oppose waveform reconstruction, or which parameter group creates the remaining output floor.

Do not infer a lack of quiet data from the failure: prior verified coverage and the parent's receipt contain substantial low-level audio. Do not claim that the existing loss already guarantees sufficient quiet pressure either. Needed next is a frozen-weight, training-only diagnostic that attributes gradients by teacher RMS bins, complete valid quiet windows, genuine utterance start versus mature context, and exact encoded silence, preserving breaths and room tone in the target. Include pairwise waveform/mel/FM/adversarial gradient cosine and the 480-phase residual projection. A batch-average norm cannot answer those questions.

The old 250-update quiet-phase candidate already failed: quiet residual increased 5.70% and repeating residual 25.67% with a bounded 2% contribution. It used an earlier recipe and is not conclusive about every targeted quiet loss, but the same penalty must not be presented as an untried obvious fix. See `outputs/convnext-quiet-phase-comparison/result.md`.

## Minimal next sequence

1. **Establish the teacher ceiling and the useful comparison.** Report teacher reconstruction versus original audio using the same quality protocol, and student versus teacher separately. Teacher-versus-itself gives zero error by construction and cannot establish source-audio quality. Keep input/target sample-rate contracts explicit. The parent agent handles this separate benchmark; none was run in this audit.
2. **One fixed-weight gradient audit before more training.** Use a small representative training-only calibration pool, no optimizer updates, with current control, short mel and the two warmed discriminator variants. Measure effective output shares, per-resolution and per-level gradients, gradient conflicts, and actual proposed update norms on isolated optimizer copies if clipping remains a concern. This both validates the diagnosis and sets defensible scales.
3. **Repair migration state only for changed objectives.** Explicitly initialize changed loss-gradient EMAs from their new calibrated distributions, retaining old model weights, moments, calibrated model normalization and unchanged-loss state. Apply the same calibration protocol to both fresh discriminator arms, and record it as a new experiment. Do not reset the production balancer or claim exact continuation. Audit the resulting scales and quiet allocations before allowing a jump from 0.27% to 30% perceptual influence.
4. **Run only the corrected pair justified by that audit.** For the complex question this is complex versus fresh magnitude with matched data, warmup and calibrated effective balance. Use more than the present 200-update engineering screen if deciding learning potential; a bounded approximately 1,000-update paired adaptation with sparse common-panel checks is a planning option, not a convergence promise. Its discriminator must show useful gradients and noncollapsed behavior, rather than relying on a step count alone. Short-mel weighting is a separate question and should remain separate.
5. **Architecture and silence changes remain conditional.** Tanh is still a clear bounding mechanism with observed reconstruction tradeoffs; this audit found no comparable balancer confound for it. Do not add all candidate losses/filter/padding changes together. A targeted quiet residual or phase-aware objective should follow the level/direction diagnosis, not precede it.

The 200-step screen is useful for gross regressions, accounting, gradients being present and immediate adaptation costs. It is insufficient to dismiss fresh discriminator families or assert a quality-neutral final architecture, especially with the measured EMA mismatch and a single parent/seed. Existing common-panel metrics should be kept, and the premature inference about the complex objective's potential should be corrected.
