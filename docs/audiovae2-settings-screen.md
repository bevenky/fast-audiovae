# AudioVAE2 compression settings comparison

This short comparison tests training settings for the same stages 2–4 replacement. It does not recreate end-to-end VAE pretraining. The original encoder, stage 1 and stages 5–6 plus waveform head stay frozen. Both paths receive identical block inputs and the student matches the complete stage-4 output.

## Source audit

The [released AudioVAE2 code](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py) establishes the model, causal state and sample-rate contracts. The original weights, Snake, weight normalization, nine residual units and dilations, terminal convolution and tanh are retained. Only the approved internal channel widths change.

A [VoxCPM collaborator's loss description](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845) specifies the original AudioVAE recipe: mel weight 15, adversarial feature matching 2, generator adversarial loss 1 and constant KL weight 5e-5. It refers to the DAC loss implementation. The [schedule follow-up](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3771286634) describes a global batch of 128 two-second clips, learning rate 3e-5, cosine decay, 1,000 warmup steps and one million training steps. These disclosures precede AudioVAE2 and do not establish every V2 pretraining value.

A [later V2 discussion reply](https://github.com/OpenBMB/VoxCPM/issues/353#issuecomment-4913099637) describes additional low-band mel supervision and a different schedule. Its author's connection to the model team could not be verified from the public profile or repository record. Those claims are not used as confirmed settings in this comparison. The public TTS fine-tuning script freezes/removes the VAE before constructing its optimizer; its YAML is not a VAE training recipe.

## Four controlled runs

| Run | Mel definition | Learning rate |
|---|---|---:|
| Current, faster rate | Existing five-scale linear plus natural-log magnitude mel | 1e-4 |
| Current, lower rate | Same existing loss | 3e-5 |
| Author reference, faster rate | Original AudioVAE/DAC seven-scale log10-only mel, evaluated at 48 kHz | 1e-4 |
| Author reference, lower rate | Same author-reference loss | 3e-5 |

The reference uses windows 32/64/128/256/512/1024/2048 and mel bins 5/10/20/40/80/160/320, a quarter-window hop, magnitude power 1, floor 1e-5 and the sum of per-resolution mean log10 errors. Its DAC/audiotools reference uses centered transforms with reflected edge padding and periodic Hann windows. The current loss uses windows 256/512/1024/2048/4096, bins 32/64/64/128/128, uncentered transforms and averages resolution means. These are different definitions, so the published scalar 15 cannot be transferred to the existing loss unchanged.

Reflection is built with explicit slices, reversal and concatenation before an uncentered STFT. This produces the same reflected samples and forward loss as native centered reflection while supporting strict deterministic CUDA backward. The native padding backward failed during calibration before any retained training update. The replacement passed 37 loss/screen tests on both hosts and repeated H100 gradients were bitwise identical. Determinism remains enabled.

Each run starts from the same authenticated untrained compressed weights and uses the same ordered 768 distinct fitting sources for 256 updates. Execution is singleton FP32 with three-source gradient accumulation and one AdamW update per group. Betas 0.9/0.99, zero weight decay and epsilon 1e-8 remain fixed experimental choices. Both learning rates use the same short constant schedule; this does not reproduce the author's million-step schedule.

Raw teacher-waveform L1 and complete stage-4 MSE remain active in every run. Each mel definition receives one gradient calibration on the same separate 72-source panel; its coefficients then remain fixed across both learning rates. Disposable calibration updates restore weights and optimizer/RNG state. Reusing sources across independent comparison runs is intentional; a source is not repeated within a run. Each final checkpoint preserves optimizer state and exposure history for continuation.

The reference's GAN and discriminator feature-matching objectives are not tested here. Stage-4 feature MSE is a separate distillation loss. KL would have no gradient into this student because its encoder is frozen. Results must be described as a block-distillation settings comparison, not an exact AudioVAE2 pretraining reproduction.

## Measurements and decision

Use the same 96 development sources at steps 0, 128 and 256. Compare stage-4 MSE/normalized error, waveform MAE, nonquiet correlation, quiet residuals, failed quiet windows, peak values and overshoot counts. Retain per-source and language/expressive summaries. Measure the fixed 12-source stage-boundary panel at the final step.

Compare common metrics computed with identical definitions across all four runs. Raw training totals and different mel definitions are not interchangeable quality scores. Report tradeoffs instead of declaring a winner from one lower loss. A short run can select a promising continuation; it cannot establish 0.99 reconstruction, perceptual equivalence, or CPU RTF. Those still require longer recovery and the planned quality/runtime checks.

## Measured results at 256 updates

All four runs completed on the H100 with the same 768 distinct fitting sources, 31.47 minutes of scored audio per run, and the same 96 development sources. Frozen decoder state and the original initial checkpoint were preserved. There were no nonfinite training losses or repeated sources within an arm.

| Mel definition | Learning rate | Active waveform cosine, higher is better | Waveform MAE, lower is better | Common five-scale mel, lower is better | Full stage-4 MSE | Quiet residual RMS |
|---|---:|---:|---:|---:|---:|---:|
| Current | 1e-4 | 0.90155 | 0.009958 | 0.61252 | 0.032638 | 0.0003164 |
| Current | 3e-5 | 0.91328 | 0.008732 | 0.65827 | 0.041361 | 0.0003025 |
| Author reference | 1e-4 | 0.90279 | 0.009776 | 0.62292 | 0.032440 | 0.0002993 |
| Author reference | 3e-5 | 0.91291 | 0.008784 | 0.67075 | 0.042174 | 0.0002928 |

The lower learning rate improves final waveform reconstruction despite slower recovery of internal features and the common spectral objective. For the current loss, it improves waveform MAE on 85 of 96 sources and active correlation on 78 of 94 sources. The two lower-rate loss definitions remain close. Current mel gives slightly better waveform, expressive and full-group agreement; reference mel gives lower pooled quiet residual on 64 of 68 sources containing quiet audio, but does not improve every quiet-selected source.

Both lower-rate arms still fail 2,542 of 2,544 quiet-window reconstruction checks at step 256. All arms remain bounded by the retained tanh; zero full-scale overshoots are a structural property, not proof of transient fidelity. These are one-seed development results, not a final quality claim.

The added symmetric evaluation also favors current loss at the same lower rate on the reference seven-scale metric: 1.47469 versus 1.48510. All four candidates reproduce their original common spectral scores exactly. The current-loss lower-rate checkpoint was therefore continued with its optimizer state to the planned 1,000 updates, consuming the next 2,232 unique sources. Active correlation reached 0.94468, waveform MAE 0.006786, and quiet residual RMS 0.0002442. Quiet failures remain 2,429 of 2,544. See the [quiet-audio audit](audiovae2-quiet-audit.md) for target replay, source-silence checks, failure categories, data coverage and the remaining whistling gain regression.

The report archive SHA-256 is `155dc744000d58214cd3c6eb1e31897a4b56c7a647cc29c7dccdd0f9cf06b654`. Numeric reports preserve per-source scores, fixed boundary traces, source exposure, model identities and checkpoint hashes. Original checkpoints remain on Runpod.
