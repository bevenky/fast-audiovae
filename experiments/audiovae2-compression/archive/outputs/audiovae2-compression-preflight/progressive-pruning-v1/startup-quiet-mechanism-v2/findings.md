# Startup and quiet-window diagnosis

The complete unchanged-teacher control passed. The current width cut introduces real reconstruction damage before training. Recovery through step5000 improves every quiet window, but startup behavior and quiet nonzero audio remain different problems. Keep the step5000 checkpoint and investigate weight reconstruction before another cut.

## What was checked

The control copied the full teacher architecture and weights into the student wrapper. Across all 60,000 crops, native teacher and copied student outputs were exact. All 1,153,650 quiet windows and 58,045 near-silence windows passed the relative comparison. Twelve fresh AdamW updates on 144 separate fitting sources produced zero losses, zero gradients and no weight changes. Those twelve updates do not cover every source in the larger control.

One saved cache target differed slightly from a fresh teacher call, within the original tolerance. One cold execution shape also differed on its first call before becoming exact. These small numerical exceptions do not explain the pruning residual. The control validates the wrapper and relative comparison on tested inputs; it does not establish that the original teacher produces physically zero output for every silent input.

The current experiment narrows stage2 from512 to384 channels. Stage3 remains256 channels. It retains all nine residual units in stages2–4, causal padding, strides, dilation, Snake, conditioning, and the original final convolution and tanh. The main change is channel removal, not deletion of whole residual layers.

The causal diagnostic evaluated 19 fixed sources and165 teacher-defined windows. It used original sliced step0, saved step5000, original teacher, and an independent full-width copy. Both full96-source saved metric reports reproduced, and all model states and original files were preserved. No training ran.

## Where startup damage enters

All13 startup failures already existed at step0. The unchanged stage1 output matches the teacher exactly. The cut directly removes weighted input contributions at three stage2 residual projections and the stage3 transposed convolution.

The table below adds omitted teacher contributions back only to the pristine step0 model. This is an instrumented counterfactual, not an inference implementation or a repaired step5000 checkpoint.

| Step0 intervention | Startup residual RMS, millionths of full scale |
|---|---:|
| Original sliced initialization |48.385|
| Restore residual projection1 only |54.256|
| Restore residual projection2 only |53.919|
| Restore residual projection3 only |49.771|
| Restore stage3 upsampler input contribution only |5.151|
| Restore all four contributions |0.001106|
| Trained step5000, no intervention |22.004|

Restoring the stage3 upsampler contribution alone reduces initial startup RMS error by89.35%. Restoring all four approaches numerical parity. All four local algebraic decompositions, full group output, and final waveform pass the unchanged original tolerance across all19 sources.

The evidence localizes the largest initial waveform sensitivity to the stage3 upsampler input. Individual residual repairs can worsen the waveform, so the sites must be considered together. Measured kept/dropped cross terms are positive in these regions; the result should not be described as proof of negative cross-channel cancellation.

Zero source audio does not imply zero internal activations. Encoder outputs, learned biases and conditioning still drive the decoder, while startup history is padded with zeros. Removing weighted feature contributions changes that startup response even though padding code is unchanged. At step5000, startup error is97.67% temporally varying energy and2.33% per-window DC energy. A constant bias correction cannot explain most of it.

The recovered group output still differs from the teacher before the frozen suffix: startup group-output normalized RMS error is11.52%, versus approximately3.9% in the selected interior source-zero control. This locates incomplete recovery, but does not uniquely identify which adapted weights cause it. Original teacher-coordinate interventions were deliberately not applied to the coadapted step5000 model.

## Where the remaining quiet failures occur

These groups are disjoint. Near-silence means teacher RMS at most1e-5; remaining quiet is above that but at most1e-3. The original thresholds are unchanged.

| Group | Windows | Passing at5000 | Failing at5000 |
|---|---:|---:|---:|
| Near-silence in first20ms |13|0|13|
| Other near-silence |171|171|0|
| Remaining quiet audio |2360|1340|1020|
| Total |2544|1511|1033|

Every one of the2544 quiet windows has lower residual RMS at step5000 than at initialization. Pooled residual RMS falls from1414.63 to70.263 millionths of full scale, a95.03% reduction. This comparison establishes recovery from initialization, not monotonic improvement at every intervening checkpoint.

Of the1033 failures,1000 fail only waveform residual,31 fail both residual and output-level checks, and2 fail only output level. Outside near-silence,687 failures occur after800ms. Most are not startup, clipped peaks, or missing stream chunks.

Pooled output RMS for remaining quiet audio is98.59% of teacher RMS, while waveform error is still too high. The per-window gain-error component accounts for only10.54% of its residual energy. Per-window DC accounts for11.93%. These are separate decompositions, not additive fractions. Most error is detailed waveform mismatch rather than a uniform volume or offset problem. A silence gate would erase legitimate quiet content and would not constitute teacher parity.

The broader data control found quiet samples make up15.67% of scored training samples.29,196 of60,000 crops begin at the true source start. The first20ms of those crops makes up0.397% of all scored waveform exposure, across all audio levels. Startup was not universally masked out; its small relative exposure may affect recovery but is not by itself proof of a loss-weighting problem.

## Recommended next experiment

1. Preserve the current checkpoint and do not make the next cut yet. More training could help, but these results do not justify a guarantee of99.99% reconstruction or a speculative60,000-update run.
2. Test a separate fresh initialization that reconstructs the original four mixing operations using only the retained inputs. Do not overwrite teacher-coordinate fits into the coadapted step5000 model. Prioritize the stage3 upsampler, fit its actual temporal input neighborhoods with native stride5/kernel10/shared-bias constraints, and use the real sequential student inputs when moving through the sites. For each residual mixer, fit the teacher's complete residual-unit output on retained coordinates minus the current student skip input. Recompute downstream inputs after each upstream fit; fitting four independent branches on stale inputs would leave skip drift uncorrected. Fit on the calibration/fitting split only, then judge on the unchanged held-out panel. Include true starts, sustained zeros, quiet nonzero audio, speech and expressive examples. The current calibration set has29 true-source-start crops out of72, which does not establish that silent-start coverage or influence is sufficient.
3. Store any successful reconstruction in the existing smaller weight tensors. This adds no layer or inference operation. Compare the upsampler-only and coupled four-site versions before any recovery training. A counterfactual using inaccessible dropped teacher features is not a deployable solution.
4. Continue joint recovery only if the real reduced-input model improves startup and quiet residuals without damaging active audio. Preserve output waveform and full-group behavior as the main acceptance criteria. Do not require every intermediate adapted channel to equal the original teacher coordinate.

The present channel selector uses a Gram matrix of stage-output activations, then slices weights. Selecting informative features does not preserve the downstream weighted function automatically. Reconstructing the remaining operators is the specific missing compensation this experiment should test. Success is not guaranteed: a narrower feature set may be unable to represent every omitted contribution.

## Numerical and execution limits

The first diagnostic stopped when one whisper source exceeded pointwise tolerance inside frozen stages5/6 despite passing full-group and waveform comparisons. A separate FP64 check preserved the actual FP32 effective coefficients and brought all tested boundaries within the original tolerances. V2 retains those hidden FP32 failures explicitly and requires local/full-group/waveform closure. This establishes arithmetic sensitivity, not a training corruption or a uniquely identified faulty kernel.

An additional proposed startup template/phase comparison was not executed. Automatic approval review rejected exporting detailed waveform-derived metrics and latent examples from Runpod. The findings above use only completed and previously permitted summaries. Raw audio, tensors, source reports and checkpoints remain remote. No changes were committed, promoted or trained during this diagnosis.
