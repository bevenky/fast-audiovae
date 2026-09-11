# Joint stages 2–4 recovery

User authorized up to 1,000 additional joint updates, review every 250, then a matched prefix-freezing diagnostic only if the trajectory warrants it.

- Runpod output: `/tmp/fast-audiovae-joint-recovery-v1`; log is the same path with `.log` appended.
- Retained starting checkpoint: accumulation12 `final.pt`, optimizer step 4625, SHA256 `834f6788cb53d10422fa987a84fe516e2ccaadbc02a1c69e10f5ce750cf66c11`.
- Training PID at launch: 1041582. Full launch receipt and source hashes are saved beside this document.
- Same trainable stages 2–4, teacher/encoder/prefix/suffix frozen, AdamW learning rate 3e-5, betas (0.9, 0.99), no weight decay, original three coefficients, saved moments and RNG.
- Singleton forward/backward with 12-source pooled gradient accumulation. No projected hints or native upsampler replacement.
- New source interval 12000:24000 extends the exact 15,000-source candidate history. All 12,000 source, parent and audio hashes are distinct and held-out disjoint. About 8.159 hours of scored crops from 31.56 hours of parent recordings; 111 normalized language labels including all 22 Indic languages. Source-level expressive metadata does not prove each crop contains the named event.
- Frozen targets are produced only on Runpod by the separately versioned reserve producer. Shards are consumed only after atomic sealing and existing checksum validation.
- Review checkpoints: optimizer steps 4875, 5125, 5375 and 5625. The existing fixed 96-source panel and intermediate boundary diagnostics are used. Continuous near-silence RMS is captured from the same window-scoring pass without changing the original metrics.
- Review thresholds are engineering safeguards, not perceptual equivalence or statistical confidence claims. Two repeated material regressions of the same metric/source pause the run. Quiet count changes alone are alerts. A stall decision is eligible only after the full 1,000-update budget.
- 54 focused tests and 10 subtests passed on Runpod CPU before launch. The update-equivalence test matches the existing training update, parameters, moments and RNG.
- Startup restore passed the original exact-state and saved-quality checks. First 312 sources were in exact order and all teacher/cache comparisons were bitwise equal. No initial failure.
- TensorBoard now serves this run from its new colored display, preserving historical logs. URL: https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#custom_scalars&_smoothingWeight=0
- The existing AudioVAE training review heartbeat is active every 10 minutes. It observes the automatic 250-update reviews, diagnoses material issues and can execute the user-authorized short matched joint-versus-frozen-stages-2/3 comparison if justified. It must pause on completion and cannot extend the training budget silently.

The earlier recommendation to freeze stages 2–3 immediately was superseded by `../native-up4-reconstruction-v1/freeze-versus-joint-review.md`. Internal substitution mismatch does not prove harmful joint adaptation. No new checkpoint has been promoted, and nothing was committed or pushed.

## First scheduled review: step 4875

250 additional updates and 3,000 distinct new sources completed. The saved checkpoint, teacher/cache comparisons and frozen state checks passed. All 3,000 cached target comparisons were bitwise equal. The review action is `continue`, with count-only near-silence alert and no material regression flags.

- Waveform MAE: 0.00423436 to 0.00409770, down 3.23%.
- Mel error: 0.335225 to 0.326517, down 2.60%.
- Active waveform correlation: 0.976007 to 0.977526.
- Group output MSE: down 5.87%; quiet residual RMS: down 2.72%.
- Quiet passes: 138 to 156 of 2,544.
- Near-silence passes: 18 to 5 of 184. Continuous near residual RMS increased only 0.605%, from 2.91007e-5 to 2.92767e-5. This remains an unresolved weakness and an alert, rather than evidence of a large new noise burst.
- No overshoot samples. Waveform MAE improves on 92 of 96 sources, mel on 93, and group MSE on 95.

The second scheduled review is step 5125 (500 additional updates). Joint recovery remains active. No freezing comparison has been launched because the current result does not justify it.

## Completed budget and final decision

The run completed all 1,000 additional updates at step 5625. The final report is `final-review-audit.md`. The mixed 750-update review is preserved as `review750-audit.md`. All 29 earlier material metric/source flags cleared at the final review, with no repeated regression or stall. Six new flags remain; the conditional freezing experiment was not warranted and was not launched.

Final active waveform correlation is 0.9795208; waveform MAE is 0.003966812 and mel error is 0.3078063. Quiet passes improve to 220/2544 and near-silence passes to 47/184. Most quiet windows still fail and the 0.99 correlation target is unmet. Source-specific whispering, laughter and yelling amplitude deficits remain, so aggregate improvement does not establish teacher parity.

`final-integrity-audit.json` independently verifies exactly 1,000 sequential updates, 12,000 unique additional sources in the planned order, all 12,000 teacher/cache comparisons bitwise equal, exact scored-sample totals and preserved original files and frozen models. The final checkpoint SHA256 is `0da2f6e98a29b025b41df84dbb674e1a35fbea2629f698ef33fe62f687c18b1b`. The final correlation point is served at step 5625 in TensorBoard. Both training and teacher-target generation processes have exited.

Checksum-verified copies of all four new checkpoints, the preserved starting candidate and run records are retained in `/workspace/fast-audiovae-compression-20260910-v1/joint-recovery-completed-v1`. The originals remain unchanged. See `backup-receipt.json` for all 25 saved files and hashes.

The existing training-review heartbeat was paused after completion. No further training, new architecture experiment, commit or push was performed.
