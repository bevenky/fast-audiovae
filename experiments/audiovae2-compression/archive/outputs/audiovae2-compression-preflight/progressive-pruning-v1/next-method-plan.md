# A bounded downstream-aware selection experiment

Proposed, not implemented or launched. Preserve the current combined 2,500-update run and every previous checkpoint, journal, source plan and result. This is a separate 2,000-update student experiment from the original frozen teacher, at the same stage2/stage3 widths **384/256**. It changes channel selection, then applies the **unchanged B sequential native reconstruction** before recovery. It does not combine startup constraints or a new loss with the selector.

## What changes

The current selector builds an uncentered activation Gram at the teacher's stage2 output, samples at most 256 valid time cells from each of 72 calibration sources, and selects 384 channels by pivoted Cholesky. It does not include consumer weights. See `prepare_progressive_schedule.py:29–55`.

Use a **ThiNet-style consumer reconstruction criterion** instead. ThiNet selects channels using the output contribution of the following convolution, greedily minimizes the sum of removed contributions, then performs reconstruction and fine-tuning. Its published evidence is image classification, not this codec. Our multi-consumer, causal-polyphase extension below is a proposed transfer. [ThiNet, sections 3.2.1–3.2.3](https://arxiv.org/pdf/1707.06342).

There is one common retained set S of 384 original channel IDs, with a complementary removed set D of 128 IDs. Apply it consistently to the stage2 upsampler output, all stage2 per-channel parameters and residual states, the three pointwise mixers, and stage3 upsampler input. Keep stage3's 256 output channels, all nine residual units, conditioning, native temporal geometry, suffix and tanh.

## Exact screening objective

Collect inputs to the four teacher consumers on the original 72 calibration sources, with frozen weights, authenticated cached latents and the established teacher/cache checks. Use all valid cells and the original valid-waveform sample count for each cell. This is a deterministic selection-only pass; it does not use development data, optimizer updates or an extra startup multiplier.

For residual pointwise consumer m, let its full teacher weight be W_m with 512 output rows and input activation x_m(t). Channel c contributes

`f[m,c,t] = W_m[:,c] * x_m[c,t]`.

For stage3's native stride5/kernel10 upsampler, output cell `n = 5*t+p`, phase p in 0..4, has channel contribution

`f[up,c,n] = W[c,:,p]*x[c,t] + W[c,:,p+5]*x[c,t-1]`.

The previous input is zero only at the actual tensor start. Preserve available causal context before applying the scored-cell mask. Never reset the previous input at the first scored cell. These are the native current/previous mappings already implemented in `reconstruction_aware_init.py:24–52`.

Let q_m(t) be the number of valid waveform samples represented by a cell: at most 40 for a pointwise consumer and 8 for the stage3 upsampler. Use the same partial-tail weighting as B. Bias is unchanged in this screening operation and cancels from the local output difference. Define

`E_m = sum_t q_m(t) * ||sum_c f[m,c,t]||²`

`K_m[c,d] = sum_t q_m(t) * dot(f[m,c,t],f[m,d,t]) / E_m`

`K = (K_RU1 + K_RU2 + K_RU3 + K_up) / 4`

`J(D) = sum_(c,d in D) K[c,d]`.

Thus each site contributes its relative error in the complete bias-free teacher linear output. The four equal site weights are a declared experimental design choice, not a published codec constant. Normalizers are fixed once from the complete calibration teacher outputs. Reject a zero/nonfinite E_m rather than silently introducing a new numerical floor or data-dependent quiet weight.

**Keep all 512 teacher output coordinates for each residual mixer in this score**, regardless of S. Keep all 256 upsampler output coordinates and all five phases. Otherwise dropping difficult output coordinates could artificially improve the score as the target shrinks. The score concerns input-column deletion in those complete teacher consumers; it does not assert that the deployed student retains the discarded 128 residual coordinates.

Start D empty. Repeat 128 times: choose the remaining original channel c minimizing

`K[c,c] + 2*sum_(d in D) K[c,d]`.

Break exactly tied scores by the original channel ID. Retain signed cross terms; do not replace them with absolute values or call them negative cancellation without measurements. Save the full fixed K, four normalizers, selected IDs, per-removal objective and teacher/calibration/source hashes. Repeating only this arithmetic on the saved K must reproduce S exactly.

## Feasible implementation cost

Do not materialize a tensor of all channels' contributions to every output and time. For a pointwise consumer, its unnormalized K is the elementwise product of the weighted activation Gram and `WᵀW`.

For each upsampler phase accumulate four 512×512 activation products: current/current, previous/previous, current/previous and its transpose. Combine them elementwise with the corresponding current/current, previous/previous and cross-tap weight products. This preserves the correlation between the two taps and all 256 outputs. It requires no 5,120-dimensional inverse. A straightforward implementation storing all four products for all five phases needs about 40 MiB in FP64; the three pointwise Grams add 6 MiB. Chunk temporary activations. Retain FP64 accumulated statistics and use the established deterministic runtime.

The greedy arithmetic needs fewer than 65,536 candidate comparisons and small 512-element score updates. It is not a global optimum search and has no inner neural training loop.

Budget separately: **one 72-source teacher selection pass**, **four unchanged 72-source B operator-fit passes**, normal shape warmups/cache parity, and one 96-source step0 evaluation. The B fitter still performs its existing native 3,840-dimensional upsampler solve once; this proposal avoids repeating it for hundreds of candidate subsets. There is one fixed new candidate, no swaps, parameter sweep, silent fallback to the original selector or development-based support choice. Wall time and peak memory must be measured and reported; existing B's measured execution is only an estimate for planning.

## Repair and recovery remain matched

After selecting S, build a fresh slice from the original teacher. Reuse B unchanged: fit RU1, RU2, RU3, then stage3 upsampler; recompute actual current student inputs after each fit; use complete selected teacher RU output minus the current student skip; use the full teacher upsampler output; keep its centered FP64 delta ridge, shared native bias and verified FP32 WN writeback. See `reconstruction_aware_init.py:55–60,88–178`.

Train all 90 group tensors for exactly 2,000 updates, fresh AdamW 3e-5 with the existing betas/epsilon/weight decay, original initialization RNG, accumulation12, original first24,000 distinct sources in the same order, and unchanged waveform/mel/full-group loss coefficients. Preserve teacher/encoder, singleton teacher execution, geometry, precision and export graph. Do not load adapted old5000 or current combined weights. Calibration fitting may revisit its separate 72-source panel as already authorized for debugging; each recovery source is consumed once within the new run.

Compare against the completed B2k trajectory at matched 0/250/500/1,000/1,500/2,000 milestones. The original sliced 2k result is a secondary baseline and old5000 an exposure-mismatched reference. The support differs, so stage2 selected-coordinate diagnostics describe each model's teacher coordinates and cannot be compared as one common hidden basis. Stage3 full output, full group output, teacher waveform and the fixed 96-source/seven-quiet-region evaluation remain comparable.

Required integrity gates: exact source/RNG lineage, all 90 finite optimizer states and correct counters, unchanged native shapes and teacher state, existing target-cache tolerances, numerical writeback parity, fixed teacher-window identity and no overshoot invariant violation. Report quality even if the initializer or endpoint is worse. No selection rejection based on development loss, no automatic next cut or extension.

At completion report per-recording MAE/mel and active RMS ratios, cosine, full group MSE, quiet residual RMS and output excess, disjoint startup versus other-near failures, all seven existing overlapping cohorts, peaks, calibration score and runtime. Continuous errors and pass counts must both be shown; a lower mean quiet error does not imply more quiet windows pass.

## What this does and does not test

This tests whether **choosing which channels survive using their linear downstream effects**, followed by the same B repair and recovery, improves the result. The score is exact for its frozen teacher-conditioned local column-deletion problem. It is not the exact objective of the nonlinear narrowed network: earlier residual inputs drift, Snake acts on changed activations, removed residual coordinates alter later states, and B changes the coefficients. Subsequent actual-student B calibration errors and held-out waveform results decide whether the proxy transfers. A negative result is evidence against this bounded selector, not proof that all downstream-aware selection or this width is inadequate.

## Distinction from B and GRAIL

GRAIL fits a post-activation hidden reconstruction map from a reduced representation, then merges that map into the consumer weights. Its Gram/ridge formulation uses linear feature reconstruction; convolutional folding applies the same channel map to kernel taps. It does not push a map through nonlinearities. These facts and its official implementation are now verifiable. [GRAIL, section 3](https://arxiv.org/html/2602.23795v1#S3), [official code](https://github.com/TWWinde/GRAIL_Compensation).

B already fits native output errors sequentially on actual student inputs, with unrestricted native tap corrections and current-skip handling. Merely calling B a folded correction or adding another equivalent unconstrained output fit would not create a distinct experiment. A separate GRAIL-like arm would need its own pinned definition, for example reconstructing post-Snake hidden features and sharing one channel map across upsampler taps. This is a more restricted compensation parameterization with different bias and input-drift behavior, not a proven upgrade. This selector arm changes S and keeps B fixed so the comparison isolates selection.

For such a separate arm, a zero-intercept hidden map folds into each native tap while retaining the consumer bias. An affine hidden offset generally produces different constant terms across the five output phases and at startup, when the previous input does not exist. It cannot automatically be represented by one shared native upsampler bias. Any offset proposal needs a realizability proof; do not silently add phase-dependent biases or operations. B's complete phase-coded native fit already enforces its one shared output bias.

## Separate startup-preservation follow-up

The completed combined run is a mixed result, not an all-metric winner. At 2,500 updates it has MAE 0.00237865 versus the original same-budget 0.00230938 (about 3% worse), mel 0.138365 versus 0.148842 (about 7% better), and group MSE 0.00147844 versus 0.00206382 (about 28% better). Quiet residual RMS is 87.31 versus 81.07 microFS and quiet pass counts are 872 versus 1,123 of 2,544. All 13 startups fail, with startup RMS 20.7006 microFS versus its initially feasible 1.956 microFS. These are parent-verified completion aggregates. B stops at 2,000, so compare B and the combined student at 2,000, not B2k against combined2.5k as an equal-budget result.

The combined initializer established feasible startup outputs, but ordinary recovery can move away from them. Near-silent first20ms windows exist in 1,866 of 30,000 fitting sources, yet occupy only 0.0508% of scored waveform samples. That is objective exposure, not measured parameter-gradient mass or proof of which loss causes drift.

A future preservation test should constrain end-to-end startup waveform residual/output energy under actual optimizer displacements, instead of reimposing stale internal teacher coordinates on an adapted model or choosing an arbitrary inverse-frequency loss weight. Naturally present startup windows from each newly consumed source avoid oversampling. **A constraint applied only when a batch contains such windows does not protect them against future updates on batches without startup.** Continual protection would require a separately specified persistent constraint mechanism and a precise feasibility policy; finite training constraints still cannot guarantee performance on every unseen recording. Keep this experiment separate from the selector and do not claim it is already implemented.
