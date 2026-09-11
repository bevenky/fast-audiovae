# Optimizer and silence causal comparison

The preserved step 8,890 checkpoint remains the control. These are bounded diagnostics on disposable copies, with no long training or architecture modification.

## Optimizer comparison

- Use three calibration batches and four comparison batches, each with 32 crops. Select deterministically from the corrected 12,800-pair training cache. Separate sources and recording hashes across batches and from the canonical evaluation panel.
- For each batch, restore the same checkpoint and discriminator state. Capture one actual discriminator update and its resulting balanced generator gradient. Derive the retained Muon/AdamW direction, a direction with both generator optimizer states freshly initialized, and a plain negative-gradient reference from that identical gradient.
- Choose one common parameter-displacement radius using training calibration data only. Evaluate a fixed decreasing grid. Check waveform linearity and the squared-error curvature term for both valid audio and teacher-defined quiet samples. A small waveform nonlinearity alone does not establish a safe step near silence.
- Preserve and report directions that increase a calibration metric. They are not silently treated as corrections. If no common radius satisfies the declared checks, report that limitation instead of tuning against held-out results.
- Evaluate the chosen radius on four distinct comparison batches against the same ten diagnostic crops. Also report the unscaled retained optimizer update. Equal parameter displacement does not imply equal acoustic change, so report waveform displacement separately.
- A single improving batch is insufficient to justify a new training recipe. No optimizer-only continuation will be retained by this diagnostic script.

## Architecture diagnosis

- Trace the actual encoded-silence latent through the adapter, stem, normalization, all ten blocks and output head. Explicit replay must match normal inference.
- Separate the 480-sample repeated pattern from differences between the four internal phases and variation across latent frames.
- Test startup dependence by supplying a long prefix of the observed stationary latent. Compare only after the documented receptive field.
- Solve an external linear feasibility problem for the existing final projection. Determine whether its current hidden features can express the teacher's stationary silence and report rank, conditioning and minimum weight correction. No fitted weights are installed.
- Inspect at most six natural quiet recordings selected using teacher levels before counterfactual results. Stationary-input counterfactuals are probes of mechanism, not substitutes for the natural targets.

## Interpretation

An architecture can permit a periodic artifact while still having the capacity to reproduce the correct target. Conversely, fitting a single stationary fixture does not establish a general cure for quiet speech, transients or expressive audio. Recommendations will separate these questions.
