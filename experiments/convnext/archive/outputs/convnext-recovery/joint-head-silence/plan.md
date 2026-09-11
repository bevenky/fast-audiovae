# Selective silence repair pilot

Start from the preserved step 8,890 checkpoint. Adapt its existing causal head convolution, PReLU and output projection together. Keep the encoder, latent adapter, ten ConvNeXt blocks and normalization frozen. This tests whether the existing nonlinear head can improve quiet reconstruction while retaining its useful responses elsewhere.

No silence detector, gate, new layer, output clamp or tanh is installed. Peaks remain monitored for regressions but their treatment is deferred to a separate experiment.

| Arm | Teacher-defined quiet samples | Other valid samples |
| --- | --- | --- |
| Reference | Unchanged checkpoint | Unchanged checkpoint |
| Ordinary head adaptation | Match teacher | Match teacher |
| Selective silence repair | Match teacher | Preserve original checkpoint output |

The selective arm preserves the actual teacher's quiet waveform, including its nonzero detail. Quiet masks use the existing fixed 20 ms teacher-only definition. Losses pool valid sample counts, so short tails and quieter recordings do not automatically receive larger per-sample weights. Global baseline quiet and nonquiet errors, measured on training data only, provide fixed objective scales. The selective arm uses a declared preservation multiplier of 100. This is a conservative pilot setting, not a claim that the weighting is optimal.

Use the existing deterministic split: 2,048 distinct source/audio hashes for fitting and 256 different sources for selection, excluding the canonical development panel. Both arms use the same ordered crops for 256 steps of eight crops, with singleton forward/backward accumulation to retain the verified execution geometry. Frozen body features can be cached after exact replay checks. Teacher targets come from the sealed corrected full-source cache. No teacher regeneration or synthetic fixture fitting is needed.

The restricted head uses matched fresh AdamW states, no weight decay, a global gradient norm cap of 1 and FP32 training. This is an isolated head fitting experiment, not an unchanged continuation of the full Muon/GAN recipe. A small training-only first-update calibration starts at learning rate 0.000001 and halves at most six times. Both arms must avoid increasing quiet error by more than 1% and keep nonquiet output displacement within 10% of the original reconstruction residual RMS. Use the same accepted rate for both, restore all trial state, then start the matched run. If no rate qualifies, stop instead of weakening the check.

Select among the predeclared 64, 128 and 256-step checkpoints on the separate 256-source split. No failed candidate is silently selected. Report final-arm results even when none qualifies. Then evaluate the fixed candidate on all 285 canonical development crops, retaining the existing masks, sources and sample counts. This reused panel is not a new unseen final test set.

Retention requires at least 10% lower natural quiet residual RMS, better stationary silence, no new peak regression, and the existing reconstruction screens: natural errors within 1%, speech/expressive errors within 2%, sufficiently represented language/event groups within 5%. Include mel, high-frequency and transient errors, not just average waveform loss. A stationary fixture improvement alone does not establish a fix. Screening margins do not prove inaudibility.

Verify the frozen parameters and normalization buffers, original checkpoint/cache hashes and exact output counts. If a candidate passes quality, run CPU-only one-thread streaming parity and matched timing checks before any promotion. The graph's arithmetic is unchanged, but actual runtime and numerical parity still require measurement. No candidate is automatically promoted or committed.

Run in a new experiment directory on the H100. Keep feature caches in RAM and small candidate artifacts on temporary storage because the persistent workspace has less than 1 GB free. The original training run remains paused.
