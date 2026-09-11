# Native stage-4 reconstruction and downstream compatibility

This approved diagnostic tests why smaller intermediate feature errors did not produce better final audio in the projected-hint comparison. It fits the existing stage-4 upsampler without adding inference operations. It does not resume neural training, change the hints or learning rate, remove more channels, or promote a checkpoint.

## Starting models

Use two independent, preserved bases:

1. The original teacher-derived sliced initialization. This provides an interpretable initialization control, before downstream layers adapted.
2. The retained trained accumulation-12 candidate at optimizer step 4,625, without hints. This is the relevant current recovery candidate.

Do not mix their states or compare an untrained intervention with a trained control as if their starting points were identical. Preserve the original teacher, checkpoint files, and code receipts. The main run remains paused.

## Native operator fit

At `model.5.block.1`, capture the actual student input after its stage-4 conditioning and Snake. Fit the original full 128-channel teacher upsampler output from this actual 128-channel input using the current native stride-2, kernel-4 causal transpose convolution.

For output phase 0, taps 0 and 2 operate on the current and previous input frames. For phase 1, taps 1 and 3 do the same. Both phases share the original single output bias. Preserve zero history at startup, right trimming, sample rate and output length. A fit with independent phase biases or no previous-frame input would represent a different operator and is not allowed.

Use the same 72 training calibration recordings. Accumulate weighted sufficient statistics in float64 for a phase-coded 512-dimensional linear design with one shared intercept. Fit with a fixed ridge factor of 0.000001 times the mean diagonal of the centered design covariance. Do not select the ridge using development results. Exclude unscored context from the loss while retaining it as causal input. Weight each 12 kHz output cell by its exact number of valid 48 kHz samples, including partial tails.

Fold each fit into that candidate's existing effective convolution weights and bias, preserving weight-normalization semantics and operation geometry. The fit is a closed-form diagnostic, not an optimizer continuation.

## Controlled comparisons

Score the same 96 development recordings with their original contexts and masks. These recordings have been used for previous development decisions; they are not a new final qualification set.

| Base | Variant | Question |
|---|---|---|
| Sliced initialization | Unchanged | Starting error before learned compensation |
| Sliced initialization | Native upsampler fit | Can existing weights reconstruct the missing response from actual narrowed inputs? |
| Sliced initialization | Exact teacher upsampler output substituted | Does correct local output recover the remaining unchanged teacher operations? |
| Trained candidate | Unchanged | Current recovery reference |
| Trained candidate | Native upsampler fit, adapted residual units retained | Does local teacher matching help the current connected decoder? |
| Trained candidate | Exact teacher upsampler output substituted, adapted residual units retained | Have those units adapted to a different internal representation? |
| Trained candidate | Same native fit, original teacher stage-4 residual units restored | Does aligning both sides of the internal boundary repair a compatibility problem? |

The last variant restores only the existing three 128-channel stage-4 residual units. It retains the trained prefix, conditioning and input Snake used by the fit. It adds no layer. It is redundant for the original sliced initialization, whose stage-4 residual units already match the teacher, so do not repeat it there.

Teacher-output substitution is a diagnostic oracle. It depends on teacher features unavailable to a standalone student and is not a deployable speed improvement.

Add two controls:

- Fit a native upsampler using the retained 128 original teacher input coordinates. Score its local prediction only. Compare against actual student input fits to investigate input selection and accumulated drift. Never splice these original teacher input coordinates into the trained student and call it deployment.
- Feed the complete teacher stage-4-end output into the common frozen suffix. Confirm final waveform parity. This tests the externally aligned boundary separately from the internal upsampler boundary.

## Measurements and limits

Record local upsampler MSE overall, by output phase and by teacher-defined active/quiet/near-silence regions. Record the complete group boundary error, waveform error/correlation, mel error, active amplitude, quiet and near-silence pass counts, peaks, and fixed expressive/onset cases. Check teacher/cache targets and frozen-state preservation.

Compare every intervention with its own starting model. A lower local MSE alone is insufficient. If exact teacher output helps the initialization but harms the trained remainder, that supports downstream coadaptation. If local teacher-input reconstruction is good but actual-student-input reconstruction is poor, accumulated upstream changes deserve attention; it does not prove that all lost information is irrecoverable by nonlinear training. A failed linear fit does not prove an architectural capacity limit.

No CPU RTF benchmark is needed to interpret this fixed-geometry diagnostic. Any eventual trained candidate still requires the existing quality, streaming and CPU performance gates before adoption. Save all results, including unsuccessful interventions, without selecting a favorable intermediate checkpoint.
