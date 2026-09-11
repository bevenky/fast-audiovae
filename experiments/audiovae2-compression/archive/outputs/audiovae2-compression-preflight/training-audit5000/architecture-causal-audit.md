# What changed inside AudioVAE2

The current student is the original AudioVAE2 decoder with two internal channel boundaries narrowed. It is not the earlier ConvNeXt student, and it does not currently delete residual units. It retains the teacher's operation types and time support, but removes many learned connections between those operations. This audit separates that deliberate approximation from a wiring defect and from the cause of the late training regression.

Only source and saved diagnostics were read. No new model execution, fitting or benchmark occurred. The current official decoder source was previously re-fetched and matched our pinned source byte for byte; its identity and training-recipe limitations are in [the upstream re-audit](upstream-reaudit.md).

## 1. Exact boundary and tensor changes

The decoder starts with 64 latent channels at 25 Hz, a depthwise temporal stem, and a 64→2,048 projection. Its first upsampler produces 1,024 channels at 200 Hz. All of that remains frozen. Stages 2–4 form the trainable replacement. Stages 5–6 and the final waveform convolution/tanh remain frozen.

| Boundary | Teacher channels/rate | Student channels/rate |
|---|---|---|
| Group input, before stage-2 conditioning | 1,024 at 200 Hz | Identical |
| After stage 2 | 512 at 1,200 Hz | 256 at 1,200 Hz |
| After stage 3 | 256 at 6,000 Hz | 128 at 6,000 Hz |
| After complete stage 4 | 128 at 12,000 Hz | Same full coordinate space |
| Stage 5 / stage 6 | 64 at 24,000 / 32 at 48,000 Hz | Original frozen modules |
| Waveform | 1 at 48,000 Hz | Original frozen seven-tap convolution and tanh |

Let `K2` denote the selected 256 of 512 stage-2 channel coordinates and `K3` the selected 128 of 256 stage-3 coordinates. The same selection propagates through every affected input, output, residual skip, depthwise filter, Snake parameter and conditioning embedding. Code stages 2/3/4 are `decoder.model[3/4/5]`. [Construction, lines 350–377](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L350).

The saved actual teacher and step-5,000 tensor shapes agree with the source construction:

| Tensor family | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|
| Upsampler effective weight `[input, output, taps]` | `[1024,512,12]→[1024,256,12]` | `[512,256,10]→[256,128,10]` | `[256,128,4]→[128,128,4]` |
| Depthwise weight in each of 3 units | `[512,1,7]→[256,1,7]` | `[256,1,7]→[128,1,7]` | `[128,1,7]`, unchanged shape |
| Pointwise weight in each of 3 units | `[512,512,1]→[256,256,1]` | `[256,256,1]→[128,128,1]` | `[128,128,1]`, unchanged shape |
| Stage-input Snake alpha | `[1,1024,1]`, unchanged shape | `[1,512,1]→[1,256,1]` | `[1,256,1]→[1,128,1]` |
| Each residual Snake alpha, 6 per stage | `[1,512,1]→[1,256,1]` | `[1,256,1]→[1,128,1]` | `[1,128,1]`, unchanged shape |
| Each conditioning scale/bias embedding | `[4,1024]`, unchanged shape | `[4,512]→[4,256]` | `[4,256]→[4,128]` |

This removes **4,939,648 effective convolution coefficients** across the group. In stage-2 and stage-3 pointwise matrices, halving both axes removes **75% of connections**, even though the channel count only halves. The full list of 21 convolution tensors, with shapes, groups, stride, dilation and normalization axis, is in [architecture-tensor-removals.json](architecture-tensor-removals.json). This count excludes bias, Snake, conditioning and weight-normalization scales.

Convolution biases follow output coordinates: 512→256 at stage 2, 256→128 at stage 3, and unchanged 128 at stage 4. Residual biases follow the residual width. Weight normalization uses effective weights before slicing. For ordinary convolutions its scale has one value per output channel; for the released transposed convolution it has one per input channel. The initializer recomputes each norm from the retained effective tensor, rather than retaining an incompatible old norm. All group parameters, including the unchanged-shape stage-4 residual stack and conditioning, can then train. Unchanged shape does not mean frozen weight. [Copying, lines 47–94 and 219–284](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L47).

## 2. Where signal paths disappear

**Stage 2:** its input and all 1,024 incoming channels are retained. At sliced initialization, its 256 upsampler outputs correspond to the selected teacher outputs. The first residual unit's Snake, depthwise temporal convolution and second Snake are channelwise, so that correspondence survives up to its pointwise mixer, subject to floating-point evaluation.

At that mixer, write `v` for the features after the second Snake and `D2` for the removed channels:

```text
teacher retained output = x[K2] + W[K2,K2] v[K2] + W[K2,D2] v[D2] + b[K2]
student output          = x[K2] + W[K2,K2] v[K2]                 + b[K2]
```

The lost term is a learned contribution, not necessarily zero. The residual identity is present and correct for retained coordinates. What disappears is the removed coordinates' state and their outgoing contribution to retained coordinates. This is the first structural loss of a contribution to a retained stage-2 output. Later units receive changed activations, so their errors are no longer just an independently omitted linear term. [Original residual unit, lines 75–99](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py#L75-L99).

**Stage 3:** its upsampler loses both the 256 omitted stage-2 input channels and 128 output channels. Even if its retained input coordinates equalled the original teacher's, the removed incoming weighted sums would already alter retained outputs. Its three pointwise mixers then lose three quarters of their coefficients, as at stage 2.

**Stage 4:** all 128 output coordinates and all three 128-wide residual units remain. But its upsampler receives only 128 of the teacher's 256 incoming channels. Copying the full 128-wide residual stack therefore does not restore the original input it was trained to process. This is an important distinction: the final full-width boundary gives an exact target space, not an exact function by construction.

After joint training, `K2/K3` describe parameter provenance, not mandatory meanings of the student's hidden coordinates. Internal selected-coordinate error cannot identify a uniquely bad unit or prove which width to restore. The full 128-channel stage-4 boundary and final audio are the comparable quantities.

## 3. What remains structurally correct

**Snake:** each retained learned alpha is copied. But Snake computes a nonlinear function of `alpha × input`; once an earlier mixer changes the input, the nonlinear phase and derivative change too. Keeping alpha and the activation family does not keep the original activation output. No newly introduced singular alpha was found in the saved audit. The smallest measured absolute alpha also occurs in the original teacher. [Original Snake, lines 49–65](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py#L49-L65).

**Upsampling phase:** the three strides remain 6, 5 and 2 with kernels 12, 10 and 4. The causal wrapper uses the same right trim, producing exactly `stride × input_length`. For an interior output phase `p`, an upsampler combines current input through tap `p` and previous input through tap `p+stride`. Pruning removes channel terms from those sums; it does not drop an entire phase or change sample alignment. A periodic-error mechanism is possible after changing learned phase-filter sums, but the present diagnostics have not established that as the cause of the gain drift. [Original wrappers and stage](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py#L20-L38).

**Conditioning:** the original four sample-rate bins and input scale/bias operation remain. Removed channels lose their conditioning entries, while retained coordinates keep theirs. The current fit uses the 48 kHz condition; retaining all four embedding rows does not establish recovered quality for other conditions.

**Temporal support:** all nine units keep dilations 1, 3 and 9, all temporal taps and zero-startup semantics. The static maximum remains 19 past group-input frames and 20 past latent frames for the complete decoder. There is no new lookahead or shorter nominal receptive field. There are fewer parallel learned temporal feature channels. Temporal support and representational capacity are separate properties. [Saved support calculation](../../audiovae2-compression-plan/static-budget.json).

**Frozen suffix and gradient path:** the student receives the teacher's exact group input, targets the complete stage-4 output and propagates waveform gradients through the unchanged suffix. The suffix is frozen, not detached. No individual narrowed hidden layer is an ongoing training target. [Group execution, lines 297–308](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L297), [training update, lines 522–544](../../../work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py#L522).

## 4. How this relates to silence and amplitude

The teacher's hidden features are not zero during acoustic near-silence. In the targeted Spanish case, the teacher head-Snake RMS is approximately 0.275 while waveform RMS is `9.62e-6`. The final projection, its temporal filtering and bias convert this nonzero representation into a tiny output. This is consistent with cancellation and attenuation; it does not alone quantify which channels cancel.

For an approximately stationary interior region, the final pre-tanh mean is approximately `bias + sum_channel(sum_tap(weight) × channel_mean)`. Changed channel means can perturb this value even while overall hidden-feature MSE falls. At this amplitude tanh is approximately linear, so it neither removes DC nor restores the teacher floor. A stronger output limiter would not address that mechanism.

The saved interventions provide firmer evidence than this explanation alone:

- Teacher group output through the student suffix produced bitwise teacher audio on all 18 inspected checkpoint/source combinations. Both target consistency and prefix equality also passed on the sampled cases.
- For Spanish, stage-4 RMS increased about 2.06% from 4,500 to 5,000 while waveform RMS increased about 12.19%. Stage-4 reconstruction error nevertheless decreased.
- At 5,000, replacing the group error by half its value reduced active residual RMS and excess gain in Spanish and Kannada. Full teacher substitution removed the error.
- Most inspected near-silence failures at 5,000 were amplitude-ceiling failures at very small absolute levels, not larger waveform residuals. The separate Kannada startup error was larger and remains a separate case.

These observations localize the source to the changed group and demonstrate sensitivity through an intact suffix. They do not prove that a specific missing channel, Snake, stage or insufficient width caused the late update regression. In local terms, waveform error depends on `J_suffix × group_error`; group MSE measures the error vector's size without accounting for every direction's sensitivity. The current waveform loss already acts through this Jacobian, so the remaining question includes optimization and relative pressure, not simply a missing mathematical gradient. [Full measured evidence](layer-diagnosis.md).

## 5. Conditional architecture choices

| Approach | Mathematical advantage | What it cannot promise |
|---|---|---|
| Reconstruction-aware initialization at current widths | Refits retained outgoing weights to approximate discarded contributions; no extra deployed operations | An affine fit cannot exactly reproduce an arbitrary nonlinear wider group; better initialization does not explain a late regression |
| Restore one internal width modestly | Adds back independent nonlinear feature channels and cross-channel paths while retaining all dilation/history | Which boundary benefits most is unproven; more channels do not automatically fix loss/update behavior |
| Full-width residual state, low-rank pointwise and upsampling matrices | Keeps every Snake/depthwise channel and full-width identity path; compresses linear combinations rather than selecting a coordinate subset | Low rank still removes directions; it retains more activation work and adds calls/buffers |
| Delete more residual units | Reduces some activation/state and matrix work | Removes nonlinear processing and temporal support when neither has been shown redundant |

Reconstruction-aware channel pruning has a precedent in [Channel Pruning](https://arxiv.org/abs/1707.06168), and next-layer-aware selection in [ThiNet](https://arxiv.org/abs/1707.06342). Those principles expose the limitation of selection followed by plain slicing; their image-model results do not establish audio fidelity.

The low-rank variant is genuinely different. A full-width pointwise matrix is replaced by two linear factors `C→r→C` with no intervening activation. Savings require `r<C/2`. A transposed-convolution matrix can be factored at input rate as `Ci→r→(2sCo)` followed by the original phase overlap and trim. The first factor must be bias-free, and the original output bias must be added once after overlap. Otherwise startup/interior bias handling changes. A shared rank across taps constrains phase-filter combinations even though phase timing remains intact. Truncated or activation-weighted factorization is approximate unless the retained rank already spans the relevant mapping. [Low-rank and asymmetric reconstruction](https://arxiv.org/abs/1505.06798).

Illustrative static budgets are 42.45% fewer MACs for current widths, about 38% fewer after modestly restoring one boundary, and 30.10% fewer for the previously described full-width group with example low ranks. None is a measured CPU speedup. Details and a controlled decision sequence are in [the conditional-options memo](conditional-student-options.md).

## 6. The evidence to obtain before more continuation

The update/accumulation comparison and the missing-channel question are separate. The latter is not closed by the existing traces: they do not yet measure `W[kept,dropped] × feature[dropped]`, and a large hidden RMS followed by a small waveform does not itself prove cancellation. The next proposed diagnostic should close that gap before further continuation or architecture selection:

1. **Measure exact omitted contributions at teacher-conditioned inputs.** Capture the full input to each affected pointwise and transposed convolution. Decompose its selected output into retained-input contribution, removed-input contribution and bias. Preserve each upsampling phase and current/previous-input contribution. Report signed means, AC/RMS and their combination on teacher-defined quiet, startup, active and expressive regions. Do not divide by a nearly zero output to manufacture an importance ratio.
2. **Verify the first mismatch at the actual pruned initialization.** At stage-2 RU1, confirm that the retained channelwise path still matches, then compare the measured teacher/student mixer difference with the calculated omitted term. For later operations, separate missing-input contribution from already-changed retained inputs. Write `A` for selected output coordinates, `B` for selected input coordinates and `D` for their removed-input complement. At a sliced linear operation, `teacher_output[A] - student_output = W[A,B] × (teacher_input[B] - student_input) + W[A,D] × teacher_input[D]`, with the corresponding temporal/phase contraction. Here W uses algebraic output/input ordering, not the native transposed-convolution storage order. Inputs are the actual tensors entering that linear operation, after any Snake. Report numerical reconstruction error for this identity, not just component norms.
3. **Run labelled restoration counterfactuals through unchanged downstream operators.** One isolation control subtracts a measured missing contribution from the otherwise original teacher operator and leaves the downstream original teacher frozen; this asks what that term does to final audio. A complementary test restores that term at the first verified matching boundary of the pruned initialization and leaves its downstream initial student fixed. This asks how much one local correction helps before subsequent omissions. Neither control reproduces the effect of retraining the entire smaller group, and their scope must be stated.
4. **Check whether retained features can predict useful removed contributions.** On training-only calibration sources, fit an affine reconstruction of the relevant outgoing response, then examine independent sources and downstream audio. Fit quality within one linear operation is evidence about that proposed initializer, not a proof of complete nonlinear-group capacity. If a contribution is predictable, reusing current widths is plausible. If an important contribution consistently cannot be reconstructed and its restoration materially improves final audio, width restoration or full-width nonlinear state becomes a better-supported question.

Do not inject teacher selected-coordinate corrections blindly into the adapted 5,000-step representation. Its internal channels and weights have co-adapted, so the initialization identity no longer describes a missing term by itself. The shared full-width stage-4 teacher substitution remains interpretable there, but does not identify an internal repair.

Any later architecture comparison must use matched source exposure and final-audio guards. A reconstruction initializer starts from a separate teacher-derived copy, not by overwriting the learned 4,500/5,000 weights. Select width or rank on training-only functional evidence, then confirm on held-out regions. This is a proposed bounded diagnosis; nothing in this memo launches it or establishes which variant is the required fix.
