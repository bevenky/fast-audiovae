# Recipe v2: step 1,700 update

The new decoder has now passed the older control on mean waveform reconstruction at roughly the same unique-audio exposure. Restarting from scratch is not supported by this evidence. Quality remains incomplete: mel fidelity and quiet noise still trail the control, screaming/whistling remain poor, and two laughter crops now exceed full scale.

This supplements the earlier [step 1,000 metrics review](metrics-review.md). It analyzes an already completed CPU evaluation of the preserved step 1,700 checkpoint, with no additional inference or training.

| Fixed-panel measure | New step 1,000 | New step 1,700 | Prior r7 control |
|---|---:|---:|---:|
| Mean active waveform cosine | 0.5685 | 0.7015 | 0.6661 |
| Normalized waveform error | 0.4868 | 0.3376 | 0.3522 |
| Teacher mel error | 1.9971 | 1.5268 | 1.4313 |
| Median active level error, dB | -4.2128 | -1.8834 | -2.6250 |
| Mean quiet residual RMS | 0.002653 | 0.000539 | 0.000378 |
| Median 480-sample repeating residual | 0.001396 | 0.000268 | 0.000173 |
| Samples exceeding full scale | 0 | 75 | 0 |

At step 1,700 the student has consumed 34.9963 unique scored hours, compared with approximately 34.95 hours for the prior control. Training mixtures, initialization, objective, and cumulative update history differ, so this closer exposure match still does not isolate any one architectural change.

Compared with the prior control, normalized waveform error is 4.14% lower and cosine is higher on 138/167 crops. Mel error remains 6.67% higher, mean quiet residual 42.68% higher, and repeating-pattern residual 55.38% higher. Compared with the new decoder’s step 1,000, waveform and cosine improve on 166/167 crops and mel on 165/167. Quiet residual drops another 79.70%.

## Low volume is not a simple gain error

For each active crop, the least-squares optimal scalar gain is `cosine × teacher_RMS / student_RMS`, using the saved uncentered cosine. This calculation uses existing statistics and does not alter audio. The median optimal gain is 0.9985, while matching RMS alone would require a median 1.2421× boost. The best possible per-crop scalar correction reduces waveform MSE by only 0.291% at the median, or 1.06% averaged over crops.

| Condition | Cosine | Median level error | Optimal scalar gain | Gain needed only to match RMS |
|---|---:|---:|---:|---:|
| Speech | 0.8238 | -1.39 dB | 1.011 | 1.17× |
| Laughter | 0.5887 | -3.12 dB | 0.891 | 1.43× |
| Screaming | 0.0657 | -17.04 dB | 0.345 | 7.11× |
| Whistling | 0.0677 | -24.05 dB | 0.498 | 16.02× |
| Other emotion/nonverbal | 0.6674 | -2.44 dB | 0.990 | 1.32× |
| Japanese verbal/nonverbal | 0.8001 | -1.77 dB | 1.009 | 1.23× |

In particular, boosting screams or whistles by their RMS deficit would amplify a largely unmatched waveform. Scalar gain cannot recover missing timing, phase or spectral content, and positive gain cannot improve waveform cosine. This supports focusing on reconstruction quality, not an output-volume patch.

## Remaining risks and verified mechanics

- The step 1,700 output exceeds full scale on 75 of 16,465,920 scored samples. Of these, 74 are in `freesound:240901`, start frame 0, peak 1.3980; one is in `freesound:25794`, start frame 45, peak 1.0319. Both are laughter. Teacher peak values are not logged in these evaluation rows, so compare the existing target peaks before attributing this to new clipping distortion. The saved metric counts excursions, not an actual clipped playback file.
- Subsequent [peak verification](peak-audit.json) found teacher peaks of 0.9910 and 0.9948, with zero teacher excursions. Student excursions are outside the first and last 10 ms of both crops. This confirms interior student overshoot; see the [combined audit](review.md).
- All 1,540 quiet windows still fail the provisional strict checks. Mean quiet RMS is improving strongly, but it has not reached the teacher. No active crop reaches cosine 0.99; the maximum is 0.9797.
- The preserved checkpoint passes strict RecipeV2Engine loading: update 1,700, 54,400 consumed windows, and exactly 1,200 discriminator updates. Its calibrated normalization buffers match the saved calibration. Five real-condition streaming/fold parity checks pass, with maximum reported difference approximately 8.34e-7. The CPU audit changed no model weights and performed zero optimizer updates.
- This evaluation used CPU; earlier step 1,000 and prior-control evaluations used CUDA. Source/start keys and scored lengths are identical. The maximum difference in reported teacher RMS is 2.98e-8, consistent with reduction rounding, far smaller than the observed quality changes. This is not a new matched CPU-versus-GPU inference test.

Preserve this checkpoint and continue from it if the wider source/data audit finds no defect. The observed learning trajectory argues against restarting. The highest-value follow-up is to investigate the two full-scale excursions, the remaining quiet pattern, and sparse expressive reconstruction. The gradient-allocation mismatch documented in the earlier report remains a potential training-policy issue, not proof that the architecture or optimizer is broken.

Raw evidence: [current evaluation and optimal-gain summary](current-evaluation-summary.json). Checkpoint SHA256: `e2d4770eac1ac2f42e1812e5094cec2587597a08eb20c5927078bf3fb3b7386f`.
