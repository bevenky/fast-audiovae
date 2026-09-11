# Validation plan for AudioVAE2 stage compression

Read-only preparation, 2026-09-10. This note reviews saved measurements and source code. It does not run models, training, benchmarks or tests, and does not change the existing implementation.

Final planning selection after this independent review: the first proposed experiment reduces internal width in stage groups 2–4 while preserving their external interfaces. Reducing residual depth from nine to six units across those groups is conditional follow-up, not the first experiment. The residual-depth discussion below records the initial review; the same copied-control, whole-group supervision, causal-history and quality requirements apply to width compression. This note does not authorize or report model execution.

The useful difference from the previous ConvNeXt student is that a compressed AudioVAE2 can retain the teacher's stage interfaces, causal upsampling and bounded waveform head. This makes local failures easier to isolate and allows direct feature supervision. Matching interfaces does not guarantee that fewer layers approximate the teacher well enough. The exact full copy is a control; a compressed network is an approximation from the moment trained residual units are removed.

## 1. Reference and stage boundaries

Use the pinned original AudioVAE2 training weights as the teacher and source of copied parameters. Do not initialize from an INT8 graph, the ConvNeXt student or materialized tensors with uncertain provenance. The existing [teacher loader](../../work/fast-audiovae/experiments/convnext/audiovae_student/teacher.py) pins model revision `32279effe8c19989596f05d353d1447f51d9e915`, source revision `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`, source SHA256 `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8` and checkpoint SHA256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`.

The [pinned source](../../work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py), lines 176–210 and 275–372, defines these boundaries. Rates below are intermediate feature rates, not different output sample-rate contracts.

| Boundary | Channels | Features per second | Operation reaching boundary |
| --- | ---: | ---: | --- |
| Raw latent | 64 | 25 | Frozen encoder posterior mean |
| Decoder stem | 2048 | 25 | Causal depthwise kernel 7 and pointwise projection |
| Stage 1 | 1024 | 200 | Upsample 8, then three residual units |
| Stage 2 | 512 | 1,200 | Upsample 6, then three residual units |
| Stage 3 | 256 | 6,000 | Upsample 5, then three residual units |
| Stage 4 | 128 | 12,000 | Upsample 2, then three residual units |
| Stage 5 | 64 | 24,000 | Upsample 2, then three residual units |
| Stage 6 | 32 | 48,000 | Upsample 2, then three residual units |
| Waveform | 1 | 48,000 | Snake, causal kernel 7, tanh |

Each residual unit uses dilation 1, 3 or 9, two Snake activations, a depthwise temporal convolution and a dense pointwise convolution. The optional sample-rate affine conditioning precedes each stage. Include that conditioning in the stage contract or define the boundary after it consistently. Keep the 48 kHz conditioning bucket, learned Snake parameters, all biases and terminal tanh. The configured teacher does not use the optional random-noise blocks. These are implementation facts, not a claim that every mechanism is necessary at its existing size.

The preferred black box is the complete three-residual-unit stack after an upsampler. Replace that whole stack with two or one residual units and train all remaining units in the replacement together against the teacher's complete stack output. Do not impose individual teacher-to-student layer correspondence. Its input and output channels and time grid stay identical even if the replacement changes internal width; no new feature adapter is required at an unchanged boundary. Keep the upsampler outside this first replacement and define any sample-rate conditioning placement explicitly.

Preserving the stem, six upsampling interfaces and waveform head while reducing residual depth provides an interpretable experiment. It also retains cost outside the removed units, so a layer-count reduction cannot be equated to the same percentage of CPU time saved. Existing fused kernels may expect three residual units; supporting a different count is a deployment task, not an automatic consequence of removing Python modules.

## 2. Establish the exact copy before approximation

1. Load teacher and full-copy control independently through verified checkpoint bytes. Require strict state-key and shape matches. Record source/config/checkpoint hashes, parameter names, buffers and effective weights. Prove no shared mutable parameter storage. Freeze the teacher and encoder, including parameter gradients and any mutable buffers.
2. Preserve upstream weight normalization initially. Its legacy `weight_g`/`weight_v` state is not interchangeable with modern parametrization names or a plain `weight` key. Do not load with ignored missing keys or let constructor initialization overwrite loaded values. Materialization must use the effective weight computed from the loaded normalization state, not a potentially stale cached `module.weight`.
3. If training or export needs materialized weights, verify effective-weight equality before and after removal and ensure trainable tensors remain `Parameter` objects. Our earlier [folding implementation](../../work/convnext-recovery/architecture-experiments/run_teacher_refinements_v2.py), lines 53–76, explicitly guards a modern-parametrization failure in which removal under `no_grad` registers a weight as a buffer. That observation concerns the modern API; the original AudioVAE2 uses the legacy API and needs its corresponding removal procedure.
4. Construct the optimizer only after the final module replacements, normalization representation and trainable scope are fixed. Parameter replacement changes object identities. Do not transfer optimizer moments by position or merely matching shape. Start each candidate with a fresh optimizer and the same approved recipe; a genuine resume restores exact named state, RNG and scheduler. The public teacher checkpoint does not supply the original codec's complete optimizer/training state.
5. Before fitting, compare the full-copy control with the frozen teacher on identical raw latents at every stage and waveform, then compare full-copy continuous and streaming outputs. At a minimum retain the established waveform numerical gate, `atol=1e-5, rtol=1e-4`; report actual maximum error as well. Internal tensors need an explicit dtype-aware tolerance because their scales differ. Failure of the unchanged copy is an implementation problem, not evidence against compression.
6. Initialize surviving layers from the same teacher tensors and record exactly which units were removed, replaced or made trainable. A changed residual stack cannot be required to pass exact-copy parity before training. Its initial mismatch is a diagnostic baseline, not a quality waiver.

Weight normalization is an optimizer parameterization; folding it removes its runtime calculation but does not itself compress the convolutions. A separate half-precision or INT8 export would introduce a second approximation and must not be mixed into the first architecture-quality attribution.

Do not calibrate feature/waveform/mel gradient ratios at the exact full-copy control: its reconstruction errors and gradients can all be zero. If multiple losses need balancing, calibrate on the initialized compressed candidate using predeclared training-only examples, gradients of the actual trainable group parameters and explicit zero/nonfinite handling. Inspect disposable optimizer updates with fully restored parameters, moments and RNG. Neither adding an epsilon to an undefined zero/zero ratio nor copying the previous ConvNeXt loss coefficients establishes a valid recipe here.

## 3. Three comparisons, all with correct history

Let `T_i` be the complete teacher residual stack in group i, `S_i` the complete shorter replacement, `h_T` the teacher prefix output and `h_S` the actual student prefix output at the same group input boundary. All outputs below refer to the same scored time indices, with invalid padding excluded. The loss concerns group outputs, not matching the different numbers of individual layers.

| Comparison | What it reveals | Limitation |
| --- | --- | --- |
| `S_i(h_T)` versus `T_i(h_T)` | Can the replacement approximate its stage on natural teacher features? | Teacher forcing hides accumulated upstream student error. |
| `S_i(h_S)` versus `T_i(h_S)` | Local approximation error on inputs the deployed student actually produces | The teacher is being evaluated on potentially shifted intermediate features. This is not the natural teacher waveform target. |
| Complete student `S(z)` versus complete teacher `T(z)`, plus `h_S` versus `h_T` at each boundary | Actual reconstruction and propagation of error through the whole compressed decoder | A large downstream mismatch alone does not identify the first cause. |

Use the third comparison as the acceptance target. First train all parameters of a shorter group jointly on same-input group pairs `S_i(x)` and detached `T_i(x)`. Natural teacher-prefix inputs give the clean initial approximation task. Then validate the hybrid decoder using actual student prefixes so later groups are exposed to preceding replacements, including the same-input diagnostic `S_i(h_S)` versus `T_i(h_S)`. If fitting later uses those student-prefix inputs, name that policy explicitly and keep the natural full teacher waveform target unchanged. Directly matching complete group boundary outputs is feasible because the interfaces agree. There is no need for a learned readout or one-to-one layer correspondence.

Log both raw feature error and a normalized diagnostic using fixed training-set channel scales with an explicit floor. Do not normalize every quiet feature by its own tiny RMS or independently whiten teacher and student to conceal bias. The teacher target must be detached. If `T_i(h_S)` is used as an auxiliary moving target, state that choice explicitly and stop gradients through that target; use it primarily as a localization diagnostic until its effect is established. Natural teacher `T(z)` remains the fixed reconstruction target.

To separate local error from prefix drift, the counterfactual teacher stage must see the complete corresponding student feature history. Copying teacher internal stream buffers into a student-driven stage would mix incompatible histories even if the buffer shapes match. Prefer full causal sequences or explicitly replayed bounded context during these diagnostics. Each path owns its state.

Preserve left zero padding, transposed-convolution trimming, sample-rate conditioning and phase alignment. Cropping a fresh input and resetting the encoder is not equivalent to cropping latents from an encoded full recording. The existing [teacher wrapper](../../work/fast-audiovae/experiments/convnext/audiovae_student/teacher.py) and [authenticated source helper](../../work/convnext-recovery/architecture-experiments/full_source_teacher_head.py) document this distinction and preserve whole-source target geometry. Removing dilation-9 units also shortens available receptive field; identical in/out shapes do not preserve long-memory behavior.

The encoder backend was previously corrected and its wrapper records the verified CUDA policy. Reuse that policy and authenticated full-source targets; do not silently regenerate target latents with another backend, batch shape, padding rule or gain transform. Any target preparation change needs its own parity receipt before a model comparison.

## 4. Gates that remain meaningful when the starting copy is exact

A copied decoder has zero mathematical teacher error and numerical error near zero. Neither `new_error <= 1.01 * copied_error` nor a required 10% reduction of copied quiet error is a valid allowance for approximation. Keep two different gates:

- **Implementation parity:** copying, normalization materialization, export and chunking retain the same selected model function within the declared numerical tolerance. These checks can be strict.
- **Compression quality:** compare the approximated waveform with the original teacher using an explicit nonzero error budget, original-source perceptual scores and listening. No aggregate improvement permits hidden failed sources or regions.

The following acceptance design is recommended before candidate results are inspected. Existing quiet thresholds are reusable provisional engineering checks. Final nonquiet error and perceptual noninferiority margins need to be frozen in the experiment manifest; they are not supplied by the upstream model or proven inaudibility thresholds.

| Area | Proposed gate and reporting |
| --- | --- |
| Integrity | No NaN/Inf; unchanged 64-channel latent coordinates; exact 1,920 output samples per frame; no missing, duplicated or time-shifted chunks. |
| Peak safety | Retained terminal tanh gives a mathematical bound. Require exported outputs `abs(y) <= 1 + 2e-6`, a declared rounding allowance, and inspect pre-tanh saturation, transient error and peak timing. A bounded but flattened transient still fails fidelity. |
| Quiet and breath | Reuse the [20 ms teacher-conditioned check](../../work/fast-audiovae/experiments/convnext/audiovae_student/quiet_audio.py): teacher RMS <=0.001; residual RMS <=`max(sqrt(0.02)*teacher_RMS, 1e-5)` and output RMS <=`max(10^(1/20)*teacher_RMS, 1e-5)`. Require every tested quiet window to pass for final promotion. These are provisional engineering limits, not calibrated audibility claims. |
| Nonquiet fidelity | Track absolute sample-pooled MAE/RMSE, per-source and per-region distributions, DC error, spectral linear/log components separately and nonquiet correlation. The previously stated 0.99 correlation is a reconstruction target, not proof of inaudibility or a gate that delays all useful training objectives. Freeze absolute waveform/spectral budgets rather than ratios to the zero-error copy. |
| Perceptual noninferiority | Score original-source PESQ/STOI/ESTOI, UTMOS22 and each DNSMOS component with the same pinned pipelines. Compare paired source deltas and confidence intervals, language/event strata and worst cases. Use a declared margin against the original teacher, not only the already approximate optimized output. |
| Human assessment | Blind paired listening or an actual rated listening panel on speech and edge cases. Keep raw outputs for amplitude/peak checks; any playback loudness matching must be disclosed and separate. Predicted MOS is not MUSHRA. |
| Support | Report valid samples, source counts and spectral window counts for every region. Empty regions are unavailable, not passes. Short tails retain waveform samples even when a declared spectral objective cannot score them. |

As a magnitude reference only, the currently accepted optimized Intel/AMD outputs differ from the base AudioVAE2 table by about 0.025 PESQ, 0.0016 STOI, 0.005 UTMOS22 and 0.005 DNSMOS overall. These rounded observed corpus differences are not validated noninferiority margins and must not be multiplied or silently spent again at compression plus quantization. A defensible provisional ceiling would be no larger total source-metric degradation than that accepted approximation, with separate tail/listening protection; the user must not be told this guarantees no quality loss. Apple currently has the near-exact path, so replacing its default with an approximate model needs explicit quality qualification too. [Existing quality methodology](../../work/fast-audiovae/docs/multilingual.md), [current concise results](../../work/fast-audiovae/README.md).

For spectral scoring, preserve intact valid STFT windows. Do not concatenate silent islands, independent recordings or masked snippets. Classify the windows on their original timeline into quiet, active and transition groups. Report linear and logarithmic terms separately because the log floor changes behavior near silence. Reuse the sample/element pooling semantics in [ReconstructionV2](../../work/fast-audiovae/experiments/convnext/audiovae_student/reconstruction_v2.py), with explicit accounting for spans shorter than its largest FFT. Distinguish pooled-element, equal-crop and equal-source averages rather than selecting whichever hides a regression.

## 5. Coverage and holdout discipline

Use existing downloaded, licensed material for diagnosis without redownloading or claiming a new independent evaluation. The known failure panel should include stationary encoded silence, natural pauses, low-volume breath/whisper, silence-to-speech and speech-to-silence transitions, laughter/giggling, crying, whistling, screaming/shouting, high-amplitude consonants, clipping already present in the reference, and long utterances. Generate quiet latents with the unchanged encoder; a zero latent vector is a separate synthetic probe.

Stratify by all 22 scheduled Indic languages, English, Latin American Spanish, Portuguese, Chinese, Japanese, French and Arabic where available. Record missing coverage explicitly; training presence does not establish held-out validation coverage. Include different speakers, pitch ranges, recording conditions, expressive categories and onset/tail positions. Keep original source identities, speakers where known and overlap information so multiple crops from one source do not masquerade as independent recordings.

The reused canonical panel has 144 natural sources and 282 crops; including synthetic probes it has 147 sources and 285 crops. The recent selection panel has 256 sources. Their repeated use for model decisions makes them development panels, even if disjoint from that individual trial's fitting data. The [last saved integrity audit](../convnext-recovery/teacher-l1-v1/integrity.json) also shows why aggregates are insufficient: improved mean errors coexisted with four whole-source mel regressions across two reductions and five peak-regression crops from four sources.

Before final selection, reserve a fresh untouched set with source and preferably speaker disjointness, all required language/event strata and fixed scoring rules. Do not use it to choose stage count, loss coefficients, learning rate, checkpoint or thresholds. Existing 16 kHz FLEURS metrics cover only the common source bandwidth up to 8 kHz after resampling. Teacher matching at native 48 kHz must separately score upper-band output and listening; it measures preservation of teacher synthesis, not original-source evidence of recovered information above 8 kHz.

## 6. Trusted streaming baseline and required reduction

These are saved measurements, not new runs. The earlier fully matched comparison used CPU-only ONNX Runtime 1.29.0, one inference thread, the same three Bengali/English/Spanish clips, two warmups and three randomized measured repeats. Aggregation is mean of per-clip median RTF. RTF includes every decoder call and flush divided by actual returned duration; it excludes loading, stream creation, input preparation and encoding. All 405 runs and 25,830 call records passed the recorded waveform/sample checks. AudioVAE2 outputs 48 kHz, while this Pocket continuous Mimi outputs 24 kHz with native 80 ms frames. [Protocol and results](../../work/fast-audiovae/benchmarks/streaming/one-thread.json), [explanation](../../work/fast-audiovae/docs/streaming.md#earlier-matched-one-thread-comparison).

| CPU | Chunk | Matched optimized AudioVAE2 RTF | Matched Mimi RTF | Additional decoding-time reduction to match |
| --- | ---: | ---: | ---: | ---: |
| Apple M5 Max | 80 ms | 0.22213 | 0.05471 | 75.37% |
| Apple M5 Max | 160 ms | 0.12176 | 0.03876 | 68.17% |
| AMD EPYC 9654 | 80 ms | 0.18048 | 0.12297 | 31.87% |
| AMD EPYC 9654 | 160 ms | 0.14214 | 0.11398 | 19.81% |
| Intel Xeon Platinum 8280 VM | 80 ms | 0.37679 | 0.22228 | 41.01% |
| Intel Xeon Platinum 8280 VM | 160 ms | 0.27850 | 0.18446 | 33.77% |

The subsequent Apple/Intel projection optimization is the latest documented selected one-thread recipe. Its own baseline/candidate comparisons are valid, but Mimi was not rerun. The following remaining reductions are planning estimates against historical Mimi, not freshly matched ratios:

| CPU | Chunk | Latest optimized AudioVAE2 RTF | Historical Mimi RTF | Estimated remaining reduction |
| --- | ---: | ---: | ---: | ---: |
| Apple M5 Max | 80 ms | 0.14975 | 0.05471 | 63.47% |
| Apple M5 Max | 160 ms | 0.09077 | 0.03876 | 57.30% |
| Intel Xeon Platinum 8280 VM | 80 ms | 0.31152 | 0.22228 | 28.65% |
| Intel Xeon Platinum 8280 VM | 160 ms | 0.24781 | 0.18446 | 25.56% |

The calculation is `1 - Mimi_RTF / AudioVAE2_RTF`. The latest Apple estimate needs about 2.74x and 2.34x speedups; Intel needs about 1.40x and 1.34x. AMD's earlier matched values remain the documented reference here. The [projection record](../../work/fast-audiovae/benchmarks/streaming/projection.json) uses two warmups/five repeats, and [current streaming documentation](../../work/fast-audiovae/docs/streaming.md#streaming-projection-update) explicitly warns against a fresh Mimi ratio. The older machine-readable record's note that automatic preparation was pending describes that campaign; current documentation states it is now selected automatically.

The README full-clip table uses different thread counts and is not the streaming target. Old operator percentages from full-clip profiles also cannot be assumed to decompose one-thread 80 ms inference. Removing half of residual units cannot be advertised as sufficient before exported streaming cost is measured.

## 7. Runtime qualification after a quality-qualified candidate exists

Later, rerun the compressed export, exact-copy export, currently selected optimized decoder and Mimi together at one thread on each CPU. Use a frozen identical runtime build and CPU provider, separate warmups, randomized paired order, fixed source manifests and the actual automatically selected backend. Keep GPU training separate from CPU-only inference measurement.

- Compare 80 and 160 ms chunks on all three machines; retain 40 ms correctness and latency checks for AudioVAE2 without inventing a 40 ms Mimi mode.
- Require full versus streaming parity for each candidate independently, causal future-prefix invariance, startup, variable/empty chunks, partial final chunk, reset, failure-state restoration and interleaved independent streams. Assert exact output counts. Do not reset history at every training or benchmark chunk.
- Keep independent per-stream state with bounded memory, no utterance replay and no new lookahead. Existing API emits all 1,920 samples per latent immediately and flushes no delayed tail. A compression must not obtain a better RTF by withholding samples or changing that contract.
- Report aggregate RTF, each source median, per-chunk latency distribution, startup separately and missed real-time deadlines. The old figures are warmed unpaced throughput; a paced-stream check can establish behavior under actual arrival intervals but is a different result.
- Record model/kernel hashes, provider/thread selection, memory, host contention and failures. Quantify additional export/quantization error separately from training approximation. An architecture win needs at least the requested double-digit time saving before adoption, and matching Mimi requires the larger reductions above.

The initial deliverable should therefore be an authenticated full-copy control, explicit removable stage groups, fixed development/holdout manifests and frozen acceptance budgets. That is enough to make a bounded stage-compression pilot interpretable without restarting architecture guessing or claiming that every AudioVAE2 edge case is preserved automatically.
