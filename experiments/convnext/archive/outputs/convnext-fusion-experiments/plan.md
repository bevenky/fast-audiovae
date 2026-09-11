# Decoder refinement experiment

The current training run is paused at step 8,090. Preserve that exact checkpoint, optimizer states, calibration, teacher identity and exposure cursor. These are debugging branches, not a restart or a change to the production decoder.

| Candidate | Limited comparison | What would justify keeping it |
| --- | --- | --- |
| Terminal tanh | 200 updates against the unchanged control | Eliminates overshoot without material speech, transient or mel regression. Bounding alone is insufficient. |
| Short-time spectral loss | 200 updates against the same control | Improves rapid transients or quiet residual while preserving the existing fixed-scale evaluation scores. |
| Seven-tap causal output filter | Identity initialization, then 200 updates | Reduces repeating residual without smoothing away detail; passes streaming parity and sample accounting. |
| Zero startup padding | Matched checkpoint, startup and mature-context diagnostic first | A useful startup improvement; no claim that it fixes mature silence. Train only if the diagnostic warrants it. |
| Complex multiband discriminator | Fresh magnitude versus fresh complex spectral heads; 20 discriminator-only warmup updates and 200 generator updates each | A better quality trend than the equally fresh magnitude control. A short negative result is inconclusive about eventual quality. |

Each generator comparison uses the same 6,400 nonoverlapping scored windows, once per independent branch. The two discriminator comparisons use a separate 640-window warmup pool. Teacher targets are generated from the frozen encoder and decoder once and shared unchanged across branches. No latent sampling, loudness normalization, new silence penalty or learning-rate adjustment is introduced.

Supply 30 real latent frames of context for all training arms so the filter has sufficient history. On archived evaluation crops with only 29 frames and an artificial left boundary, exclude the first six scored samples uniformly across all arms and report that exclusion. True utterance starts remain scored.

Evaluation uses the same held-out sources and unchanged common quality criterion before and after. Report speech and expressive correlation, raw waveform error, fixed-scale mel error, high-frequency error, quiet residual, periodic residual, peaks above full scale, and saturation. Include the existing whisper/breath and language appendices plus an encoded-zero fixture. Aggregate scores cannot hide a failing event group.

The short-time candidate averages five spectral scales, including 5.3 ms and 10.7 ms windows. Its total mel branch budget stays fixed, so this also redistributes weighting within that branch. Complex spectral inputs keep raw amplitude; DAC's per-clip peak normalization is not copied.

Architecture variants must pass batch versus streaming sample counts and numerical agreement. Measure bounded one-thread CPU streaming overhead for tanh and the output filter. Training-only losses and discriminators add no decoder inference work. Zero padding is not eligible for folded export until boundary-dependent normalization folding is separately qualified.

This is a directional screen, not proof of final reconstruction quality or a new cross-platform RTF result. No combination is promoted just because one metric improves. Keep the original run paused until the evidence identifies the next continuation.
