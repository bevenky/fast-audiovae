# B recovery from the original teacher

## Question and authorization

The user authorized a new 1,000–2,000-update recovery experiment from B's reconstructed teacher initialization, alongside the separate startup-preservation investigation. Test whether this initialization reaches the original 5,000-step result sooner. Do not assume that a better initial score guarantees a better converged model.

The earlier run is reproducibly specified: teacher weights, source manifests/order, random-generator state, optimizer recipe and recovery lineage are preserved. Repeating the entire training trajectory has not yet been tested. It remains the baseline, not a new teacher.

## Matched recovery experiment

Construct the original AudioVAE2 decoder from its authenticated teacher weights. Apply the existing first width cut (stage2 512 to384, stage3 unchanged256), and install the four independently sealed B operator states. Verify the complete candidate state matches the prior B artifact and reproduces its full96-source development metrics before training.

Restore the original step0 RNG state and use fresh AdamW moments. Do not load the 1,000- or5,000-step model weights or optimizer. Keep the original frozen teacher, encoder, student prefix/suffix and all90 trainable stage2–4 tensors.

Use the exact first24,000 sources and crop geometry from the original run, each once within this new run. The repeated use of prior-run data is authorized for debugging. The72 calibration and96 development sources remain excluded from gradient training.

Keep the qualified runtime and old recipe:
- FP32, TF32 off, deterministic singleton operations.
- Physical batch1, gradient accumulation12.
- AdamW learning rate3e-5, betas0.9/0.99, epsilon1e-8, weight decay0.
- Waveform coefficient1, mel0.0006674012905982311, group reconstruction0.009304078923434964.
- Same frozen-teacher latent inputs and waveform/group targets.

Review actual quality at updates0,250,500,1000,1500,2000. Preserve model/optimizer/RNG/source-ledger checkpoints at0,1000,2000. Continue the same optimizer through1000; do not reset its moments.

The run stops at2000 for comparison. No automatic next cut,5,000-update extension, promotion, commit or push.

## Measurement and repeatability

Compare B and the original sliced run at matching update counts and source/audio exposure. Use the original5,000-step model as a fixed quality reference. Report waveform MAE, active waveform cosine, mel, whole stage2–4 error, peak/overshoot, quiet RMS and pass counts, with startup and other near-silence separated.

Report individual-source improvements/regressions only as aggregate counts. Keep raw audio, latent arrays and source identities on Runpod.

A faster-convergence finding must specify which metrics and exposure milestone qualify. One trajectory does not establish statistical repeatability across seeds. Preserve the reconstruction procedure, source hashes, runtime, seed/RNG and manifests so the recipe can be replayed. No claims of guaranteed5,000-step superiority.

Training losses update every step. Quality curves update only after actual reviews. Create a separate TensorBoard directory and retain the old event files. Publish the new active dashboard on the existing8888 endpoint after a real update is observed.

## Separate startup test

The constrained-A experiment uses only calibration sources and no neural optimizer. It tests whether the native upsampler can fit ordinary reconstruction while preserving the teacher's observed startup response. It does not change this B recovery run, its losses or its initialization.

Run GPU jobs serially. This isolates the two questions and prevents GPU contention from corrupting timing. Decide whether constrained initialization merits a subsequent combined test only from its measured results.
