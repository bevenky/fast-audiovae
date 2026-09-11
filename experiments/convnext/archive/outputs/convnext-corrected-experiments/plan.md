# Corrected decoder training comparisons

All four arms start from the same saved step-8,090 student with its trained weights, Muon/AdamW moments and calibrated model-normalization statistics. The AudioVAE2 encoder and teacher decoder remain frozen. There are no new student parameters or decoder inference operations.

| Arm | Data and discriminator views | Spectral discriminator | Preparation |
| --- | --- | --- | --- |
| Regular | Matched ordinary-speech control; random 190 ms views | Original trained magnitude MRD | Preserve existing state |
| Targeted | Explicit expressive material, natural quiet and transitions; selected 190 ms views | Original trained magnitude MRD | Preserve existing state |
| Targeted magnitude | Same targeted data and views | Fresh magnitude MRD; trained MPD retained | 64 discriminator-only updates, then 32 fixed-weight gradient-calibration batches |
| Targeted complex | Same targeted data and views | Fresh complex multiband MRD; trained MPD retained | Same preparation budget and data |

Each arm receives 400 generator updates with batch size 32, exactly 12,800 windows and 8.97777 hours of valid scored audio. Regular and targeted arms share 89.854% of their audio duration; valid length is matched at every batch position. The control is a matched ordinary-speech selection, not a replay of the original broad emotional mixture. See [data selection](data-selection.md).

This gives three interpretable comparisons: targeted versus regular tests the data-selection and view-selection package; complex versus fresh magnitude tests the discriminator representation under matched preparation; fresh magnitude versus the retained trained discriminator shows the cost or benefit of replacing and preparing spectral heads. The last comparison does not isolate gradient calibration from fresh initialization and additional discriminator updates.

Only the changed feature-matching and adversarial gradient averages are recalibrated. Student weights, model normalization and all optimizer state remain fixed during that calibration. Actual achieved gradient contributions and scale saturation are recorded throughout training. Calibration material includes ordinary speech, quiet/transitions, breathing, laughter and whisper-style material; several rarer expressive labels occur in generator training but not the small calibration pool. Check their training batches before claiming universal calibration.

Evaluation uses the existing common natural panel and synthetic teacher-encoded fixtures, plus one separately reserved yelling source. Crying and giggling still lack distinct held-out groups. Source-level expressive labels and acoustic activity are not timestamp-level semantic annotations.

Evaluate before and after each arm on identical latent histories and valid samples. The primary comparisons are teacher-relative waveform error, mel error, nonquiet correlation, absolute quiet residual, high-frequency residual, peaks and overshoot counts. Separate speech, expressive and synthetic fixtures. Do not allow an aggregate improvement to conceal a quiet or peak regression. Speech MOS predictors are not the deciding criteria for these targeted waveform experiments.

A separate read-only diagnostic decomposes quiet residual power into DC, a 480-sample component, an additional 1,920-sample component and remaining error. It also measures block-boundary jumps against the teacher and the actual input to the existing output head. These measurements describe the error; they do not prove that a particular layer caused it. The frozen teacher's repeatability is checked on the same encoded-zero trajectory.

Each final model receives CPU-only batch/stream parity checks at 80 and 160 ms, with exact sample counts and a fixed maximum absolute difference tolerance of 2e-6. These are correctness checks, not new RTF benchmarks. The student inference structure is unchanged.

All checkpoints and per-step metrics remain separate. No candidate is promoted, committed or merged automatically. A bounded improvement identifies a candidate for review, not proof of final perceptual equivalence. The original training run remains paused.
