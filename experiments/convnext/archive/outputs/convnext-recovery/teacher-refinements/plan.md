# Sequential teacher-guided decoder experiments

Retain the joint-head spectral candidate unchanged, including its archived parent and head tensors. Each standalone comparison starts from this same candidate. The frozen AudioVAE2 encoder, 64-channel latent means, teacher targets and streaming geometry remain fixed. No production promotion, commit or long training run is part of these pilots.

## Sequence

1. Compare the selective-repair objective with teacher waveform targets both inside and outside quiet regions, retaining the existing spectral reconstruction. Replace the old-student displacement selection rule with teacher-relative reconstruction checks. This is an explicit objective change, not a continuation under the old repair contract.
2. Compare ordinary head training with weight normalization on its existing convolution and projection under the corrected objective. Materialize ordinary weights before evaluation/export and verify parity. No deployed weight-normalization operation remains.
3. Test an additional short spectral branch at FFT sizes 256 and 512. Preserve the original 1024/2048/4096 branch and its coefficient. Calibrate the new branch to 25% of the original branch's initial output-gradient norm using training data only.
4. Test adapting the final existing ConvNeXt block together with the head. Capture its actual input with full existing context; require replay to match the original full forward. This changes trainable weights, not deployed topology.
5. Compare that same trainable scope with auxiliary teacher-feature guidance, after passing alignment checks. Use the first complete teacher upsampling block at 200 Hz and exact two-phase expansion from student features at 100 Hz. No temporal interpolation or claim of matching hidden-channel semantics. This tests early latent expansion, not the teacher's late waveform cancellation. Fit a training-only linear readout on 32 calibration sources, freeze it for the student comparison and remove it for deployment.

Review each result before the next stage or any combination. A combination is justified only by complementary observed improvements; changes are not stacked merely because they exist. All candidates remain experimental until the full quiet, source-level, peak and streaming criteria pass.

## Matched controls

Each arm is capped at 256 updates of eight distinct sources, using the existing 2,048-source fit and 256-source selection splits. Reuse across debugging arms is intentional; there is no source repeat within an arm. All arms use the same source order. The existing 285-crop development panel remains unchanged and is not presented as an unseen final test.

The teacher waveform objective uses separate sample-pooled quiet and active squared errors normalized by fixed candidate errors measured over the training pool. Their coefficients are one; the old 100x preservation coefficient is not transferred to a teacher-loss term. A common long-spectral coefficient is calibrated from the corrected objective on the first four training batches and reused across head arms. The repair control therefore uses the common new calibration, rather than claiming exact replay of the preceding run.

Use fresh AdamW states, FP32 and a common calibrated head learning rate no greater than 0.000001. Weight-normalization or changed trainable scope is treated as an optimizer-parameterization change. Record initial/final component gradients and actual update movement on a fixed training-only panel, without using held-out failures for calibration.

Full-source teacher-feature capture must authenticate the original source, full latents, decoded target and cache key. At teacher-feature rate 200 Hz, each scored feature cell requires all corresponding 240 waveform samples to be valid. Teacher normalization statistics and auxiliary readout fitting use training data only. Feature-loss calibration uses shared last-block parameter gradients because that auxiliary branch has no gradient with respect to the downstream waveform-output tensor.

## Reporting and retention

### Calibration amendment before fitting

The first runner stopped before fitting any arm. Its disposable single-update calibration required an existing overshoot peak to be monotonic within 0.000002, which rejected the first calibration batch at every tested learning rate despite passing the other reconstruction, spectral and overshoot-count checks. The second runner records the actual peak movement and permits at most 1% aggregate peak movement during this disposable entry check. All temporary updates are restored. Final selection and canonical peak requirements remain unchanged. Keep the first attempt as a failed calibration record, not a completed training experiment.

### Corrective objective comparison identified during the pilots

The recorded initial output gradients reveal that separate inverse-baseline-error normalization gives quiet waveform corrections about 1,016 times the active waveform gradient norm on the fixed 32-source probe. This is not an AdamW update ratio. Add one isolated head comparison using the sum of squared teacher errors over all valid samples, divided by their total count and one fixed teacher mean-square scale computed from the entire fitting set. Recalibrate the existing long-mel coefficient using the same first-four-batch procedure. Keep the checkpoint, source order, 256-update budget, optimizer, trainable head, spectral windows and final quality criteria unchanged. Report quiet and active waveform gradient energies separately. This tests the measured scaling issue; it is not a promise that global MSE will solve silence.

### Teacher initialization before feature experiments

The initial final-block attempt stopped before fitting because the first cold encoder call differed from the sealed cache by at most 0.000001848. The preceding full-source experiment had established that explicit encoder/decoder warmup is required for exact replay in this runtime. Restore that initialization using the first authenticated fitting source, require repeated warm outputs and the original cache key to match exactly, then run the feature captures. Do not replace cached latents or waveforms or relax their equality checks.

Report stationary silence, natural quiet, teacher waveform error, original mel error, source/region regressions and peak behavior separately. Record sparse quiet-window support and overlapping observations. Improving averages cannot establish no quality loss. Keep the teacher-perfect output admissible under the new selection contract.

All new computation runs in separate Runpod experiment directories. Training may use the H100; any inference timing is CPU-only and separate. Identical deployed operations are not presented as a new measured RTF result. The retained parent and candidate hashes are `f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948` and `b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2`.
