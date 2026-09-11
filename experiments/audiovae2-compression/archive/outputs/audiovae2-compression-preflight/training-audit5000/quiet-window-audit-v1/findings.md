# Quiet-window diagnosis before further training

Continue recovery of the existing pruned decoder, with separate startup and sustained-silence checks. Do not remove more channels, freeze stages 2 and 3, or add a silence suppressor on the basis of the current aggregate failure count. This audit found different failure mechanisms that the single quiet score had combined. It does not establish an implementation bug or an irreducible capacity limit.

Training remains paused at optimizer step 5,625. This investigation made no optimizer updates, model interventions, loss changes or acceptance-threshold changes.

## Scope and integrity

The same 96 development recordings were evaluated once at each preserved checkpoint, 4,625 and 5,625. All 2,544 quiet-window identities, teacher values, valid lengths and limits matched. Ten focused diagnostic tests passed locally and on Runpod. Both saved quality reports reproduced under the existing numeric parity check. All live teacher/cache outputs were bitwise equal, and teacher weights, frozen student modules and protected input files were preserved. The paired evaluation took 9.57 seconds on the H100, including the separately counted runtime warmups. This is not an inference-speed benchmark.

The diagnostic observes the original evaluation pass and adds source-reference zero status, DC/AC decomposition, timing and failure categories. Original 16 kHz samples were mapped to overlapping 48 kHz window positions using exact 3:1 sample-cell overlap. No waveform resampling or new audio download was performed. An exact-zero prepared-source window does not imply a zero latent vector, zero causal history or zero teacher output.

## What the failures mean

Quiet means teacher RMS at most 0.001 over 20 ms. A passing window must satisfy both a residual limit, `max(sqrt(0.02) * teacher_RMS, 0.00001)`, and an output-level limit, `max(10^(1/20) * teacher_RMS, 0.00001)`. These are provisional engineering fidelity checks, not calibrated audibility thresholds. Near-silence is the subset with teacher RMS at most 0.00001.

| All quiet windows | Step 4,625 | Step 5,625 |
| --- | ---: | ---: |
| Pass both checks | 138 | 220 |
| Fail reconstruction only | 2,175 | 2,030 |
| Fail output level only | 150 | 122 |
| Fail both | 81 | 172 |
| Total | 2,544 | 2,544 |

Most failing quiet windows therefore fail reconstruction despite staying below the output-level cap. This can reflect attenuation, waveform-shape error, or both; passing the level limit does not establish good audio. The median student/teacher RMS ratio within the final reconstruction-only failures is 0.960, and their median cosine is 0.926. These changing subsets should not be interpreted as a paired source trajectory.

There is a material mixed result: total windows exceeding the level limit increased from 231 to 294, although pooled quiet residual fell 8.31%. The increase occurs above the near-silence bin. Keep both measures visible instead of declaring all quiet behavior improved.

## Near-silence has three distinct mechanisms

### 1. The student's initial transient dominates near-silence residual energy

At step 5,625, 13 near-silent windows beginning at source sample zero account for **97.57% of all near-silence squared residual**. Their largest residual occurs at output sample 25, about 0.521 ms after startup. On the repeated zero-input startup cases the student is about 0.000706 there, while the teacher is about 0.00000509. The analogous student sample was about 0.000813 at step 4,625, so it is improving but remains incorrect.

This is a real student startup-response mismatch. It is not evidence that the same large error continues throughout silence. Two other near-silent windows fail both checks: a Finnish window at 280 ms and a Spanish window at 12.24 seconds. The Finnish residual RMS is 0.000010015, just above the 0.00001 floor; the Spanish residual is 0.000016279. These interior cases remain in the checks.

### 2. Most other near-silence failures concern a small excess output floor

Of the 184 near-silent windows, 47 pass, 122 fail only output level, and 15 fail both. All 122 output-level-only failures already meet the waveform-residual requirement. Their largest residual RMS is 0.00000435, and their median student/teacher RMS ratio is 1.186. Their output exceeds the allowed level by a median 5.70% and a maximum 12.34%; the largest output RMS in this subset is 0.00001221. The absolute numbers and the strict relative/floor limits both matter.

On the fixed 50 near-silent windows after 800 ms of source time, passes rose from **3 to 25**, and residual RMS fell from **0.000005505 to 0.000004427**, a 19.6% reduction. On the 164 exact-zero source windows after 40 ms, residual RMS fell from **0.000005015 to 0.000003997**, a 20.3% reduction. The latter set still has 119 output-level-only failures, one failure of both checks and 44 passes. This supports continued recovery; it does not establish complete silence parity.

An existing Spanish quiet span retains a small repeating residual. Its RMS fell from 0.000005120 to 0.000004202. A 240-sample phase template describes about 74.3% of its centered residual energy over 172 complete cycles; a 40-sample template describes 54.5% over 1,032 cycles. These are descriptive same-data fits, not independent proof of a specific faulty layer. Such templates capture some random variance even without a periodic mechanism.

### 3. The teacher itself produces a startup transient on exact-zero input

Ten exact-zero source windows between 20 and 40 ms account for **97.40% of residual energy across all 185 exact-zero source windows**. Their teacher waveform has a peak around -0.008588 at about 29.46 ms. At that sample the final student is only about -0.0000359. Thus the large mismatch in those windows is principally failure to reproduce a teacher transient, rather than added student noise.

This is a measured distinction between teacher fidelity and source-silence fidelity. It does not justify silently excluding the windows or changing targets. Keep the original teacher-comparison score and label this behavior separately. A blanket silence suppressor would not repair the teacher mismatch and could erase whispers, breaths or low-level transitions.

## What pruning changed and what remains uncertain

The current student narrows stages 2 and 3 inside the jointly trained stages 2–4 group. It retains all nine residual units, causal timing, dilations, Snake activations, the final convolution and tanh. The teacher encoder and the prefix supply the same group input. Frozen suffix layers are intact, and gradients traverse them into the trainable group.

Removing channels changes intermediate matrices and nonlinear feature inputs even when layer types remain the same. Earlier controlled teacher interventions showed that omitted contributions can alter startup, level and periodic response; restoring all eight omitted contributions at the fresh sliced initialization recovered the teacher numerically. Supplying the complete teacher group output to the frozen suffix also recovers the teacher waveform. These experiments locate the functional difference in the changed group, not the unchanged suffix.

They do not uniquely assign the remaining trained-checkpoint error to one missing channel or prove that the reduced width cannot learn the required function. The adapted stages co-operate: earlier isolated replacement of stage-4 internals damaged final reconstruction. That is why another isolated pruning/replacement or freezing stages 2 and 3 is not the next action.

There is no evidence for a universal DC-offset repair. Per-window DC accounts for only 2.64% of current pooled quiet residual energy and 1.43% of near-silence residual energy. A global bias subtraction would leave most measured error and could change already-correct audio.

## Training exposure and continuation

The separate [exposure audit](../joint-recovery-v1/training-exposure-audit.md) inspected all 12,000 source crops used in the last 1,000 updates. Quiet audio was 15.004% of scored samples, or 73.45 minutes; near-silence was 0.809%, or 3.96 minutes. Every update contained quiet audio, but 330 updates contained no near-silence. The current raw sample-pooled L1 loss includes these samples with the same nonzero-error derivative magnitude as active samples. There is no missing-silence mask or old inverse-RMS weighting bug.

Limited and uneven near-silence exposure may affect recovery, but the saved counts and scalar losses do not prove a regional gradient conflict or identify the best new weight. The final 250 updates actually had the least near-silence exposure while near-silence error improved. Do not treat more data or a larger loss coefficient as an established cure.

Over the completed 1,000 updates, active correlation rose 0.97601 to 0.97952, waveform MAE fell 6.32%, mel error fell 8.18%, and near-silence RMS fell 10.15%. Most sources improved, but three expressive sources remain roughly 9–13% below teacher active level. The final checkpoint is not best in every metric, and correlation is not a percentage of perceptual accuracy.

Recommended next sequence:

1. Preserve step 5,625 and the earlier checkpoints. Keep the present architecture, joint optimization, teacher targets and losses for the next controlled continuation.
2. Carry the startup, sustained-silence and low-level-content distinctions into monitoring now. Retain every original threshold and full-panel score. Track absolute output floor as well as residual, plus expressive levels.
3. Prepare new unique source crops from existing audio, measuring both startup and interior near-silence coverage. No new download is necessary. Do not tune or train on the development examples used here.
4. Continue in 1,000-update segments, reviewing the fixed panel at 500 and 1,000 updates. Proceed toward 10,000 total only while reconstruction and the separated edge-case checks continue improving without persistent material regressions. Define the review horizon in optimizer updates when implementing the new cadence; do not silently reuse a review-count trigger that assumed the old 250-update interval. Do not postpone a stalled startup or floor problem until step 10,000.
5. If a separated component stalls or repeatedly regresses, the next targeted diagnostic should compare its existing loss gradients and optimizer direction with those of active speech on training-only examples. This tests whether the current objective can improve that component without harming speech before changing loss weights, initial-state handling or capacity.

Reaching optimizer step 10,000 from 5,625 requires 4,375 further updates and 52,500 unique crops at accumulation 12. Only 3,000 are currently sealed in the unused plan, so another 49,500 must be selected and cached. A read-only inventory found 151,147 eligible existing nonempty audio files after the current exclusions. Existing audio is sufficient; the continuation cache is not yet prepared.

## Evidence

- [Paired quiet-window results](results/paired-windows.json)
- [Step 4,625 detailed results](results/quiet-step4625.json)
- [Step 5,625 detailed results](results/quiet-step5625.json)
- [Completion and preservation checks](results/completed.json)
- [Training exposure and loss audit](../joint-recovery-v1/training-exposure-audit.md)
- [Completed training review](../joint-recovery-v1/final-review-audit.md)

No training continuation, model promotion, commit or push was performed during this investigation.
