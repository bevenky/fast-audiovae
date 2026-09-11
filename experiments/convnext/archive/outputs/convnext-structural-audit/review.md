# Structural review of the decoder training recipe

9 September 2026. Further training is paused. The r6 and r7 comparisons completed; no perceptual trial or final training run has started. This review used source, saved metrics, and read-only CPU inspection of checkpoint tensors. No new forward experiment or benchmark was run for the review.

## Main conclusion

The central problem is that we have not established a complete, coherent training baseline for the new decoder. We implemented a vocoder-style generator but kept its adversarial and feature-matching training disabled, then adjusted reconstruction losses to solve symptoms. The two controlled adjustments were valid comparisons, but they were the wrong priority before resolving the overall recipe.

There is no current evidence that the frozen teacher is absent, latent coordinates are mismatched, the student lacks enough past context, or its ten ConvNeXt blocks are still frozen near initialization. There is also no proof that the proposed decoder can attain teacher quality at its intended speed.

## Confirmed recipe gaps

### 1. The quality-training stage has never trained this student

The selected model has only received teacher waveform and mel reconstruction gradients. Its discriminator optimizer remains unused. The original entry condition required near-final 0.99 waveform correlation and strict quiet acceptance before allowing GAN and feature matching.

That is circular as a training strategy: perceptual training is intended to help produce realistic fine waveform structure, yet we require that reconstruction to be almost solved before trying it. The [Supertonic paper](https://arxiv.org/html/2503.23108v3#S3.SS1.SSS2) includes mel, adversarial and feature-matching losses in autoencoder training. The [Vocos implementation](https://github.com/gemelo-ai/vocos/blob/main/vocos/experiment.py) uses an optional step-based mel warmup, default zero, rather than a correlation prerequisite.

The proposed new readiness gate still contains an arbitrary relative-noise threshold. In r7 control, quiet residual rose from 0.000339166 at the midpoint to 0.000377614 at the end: 11.34%, but approximately 0.93 dB. That fails the declared 10% test and must remain a recorded failure. It does not, by itself, scientifically establish instability or justify indefinitely postponing the missing training stage. We should revise the next experiment's policy explicitly, rather than cherry-pick a different earlier checkpoint to make this one pass.

### 2. The normalization calibration described in the plan was not performed

The plan called for calibration using fixed student weights and training-only audio: calibrate the stem first, then the final normalization under that calibrated stem. The actual pilot instead froze both BatchNorm running-statistic sets after 200 updates while upstream weights were changing. Mel training began after 250 updates.

With momentum 0.1, about 95.3% of the frozen statistics' EMA weight comes from the last 29 updates. This is not a calibration over the whole representative dataset.

The checkpoint's statistics are finite and not visibly singular. Frozen normalization is an affine mapping and its scale/offset remain trainable, so this is a conditioning and calibration gap, not proof of a capacity loss or the cause of attenuation. After freezing, training and evaluation use the same function.

### 3. Loss weighting is more complicated than “75% waveform, 25% mel”

The waveform objective divides each clip's mean absolute error by its teacher RMS, floored at 0.001. Its derivative per scored sample is:

`sign(student - teacher) / (batch_size * clip_samples * max(teacher_RMS, 0.001))`

For equal-length examples, a clip at RMS 0.001 therefore receives 30 times the waveform gradient per sample of one at RMS 0.03. One global multiplier for the waveform branch does not undo that relative example weighting. Short tails also carry the same per-example loss weight as full crops, despite data quotas being specified by duration. About 27% of the latest planned windows are partial windows.

This is an actual objective choice, but it does not mathematically force attenuation: exact teacher audio still minimizes the waveform loss. We have not established that the weighting caused the observed volume deficit. Reducing the mel share did not fix quality and should not be promoted.

### 4. We treated adaptation to frozen AudioVAE2 latents as easier than we had established

The teacher uses successive synthesis stages at 200, 1,200, 6,000, 12,000, 24,000 and 48,000 Hz. The student does temporal processing at 100 Hz and predicts 480 samples per output frame directly. Its four-phase adapter must turn each frozen 40 ms latent into four useful 10 ms representations.

This is a different synthesis problem from jointly training an encoder and decoder to share a convenient latent space. A teacher supplies consistent targets, but output-only distillation does not transfer learned decoder weights or guarantee rapid convergence.

The combined 256-by-64 adapter has rank 64 and condition number 9.59. Individual phase maps are much less well conditioned: approximately 11,133, 2,750, 3,593 and 451. Because the decoder is causal, the first output phase cannot use later phases of the same current latent. This is a specific potential weakness in how the first 10 ms receives latent information. It is not proof of information loss or the measured failure. An information-preserving identity path to every phase is a defensible design to consider before another restart, rather than adding a penalty to its output artifacts.

The direct head can represent arbitrary 480-sample blocks; 100 Hz hidden processing is not a 50 Hz audio bandwidth limit. The uncertainty concerns learning sample phase and continuity efficiently. Upstream biases can also generate a repeated output template even when the final projection has no bias.

### 5. Data coverage has changed during the diagnostic lineage

The initial corrected pilot used mainly English and 13 Indic languages. Later continuations expanded to 110 speech languages. That broadens exposure but changes the distribution while normalization stays frozen. Coverage count alone does not establish quality in every language.

Conservative whole-file retirement after using some windows is stricter than preventing repeated scored intervals. It has exhausted scarce crying, giggling, shouting and whistling material for this continuation. The latest event quota was filled by laughter, screaming and German whispers. Future diagnostics should use an explicitly fixed mixture and retain unused intervals without replaying already scored audio. Independent debugging experiments may reuse the downloaded training pool, as authorized; held-out data remains excluded.

## Checks that do not support a fundamental wiring failure

| Question | Current evidence |
|---|---|
| Is the teacher frozen and actually used? | Yes. Both decoders use the same original encoder posterior mean; waveform targets come from the original decoder at fixed 48 kHz. |
| Are latent coordinates and sample clocks matched? | Raw 64-channel means, 640 input samples per latent, 1,920 output samples per latent. Targets are decoded over complete utterances before aligned cropping. |
| Is past context too short? | Student: current plus 29 past latents. Teacher dependency arithmetic: current plus 19–20 past latents. The supplied context covers both. |
| Are the residual blocks still disabled by their initial scale? | No. Mean absolute learned scales are 0.0294–0.0372, and none of 5,120 coefficients remains below 0.00001. |
| Is Snake missing from the released Supertonic 3 pattern? | No. The inspected release uses GELU in the blocks and PReLU in the head. Adding Snake would be a new architecture change. |
| Do the latest data indicate a plateau? | No. The retained control continues improving, although it remains far from final acceptance. |

Checkpoint details and dependency arithmetic are in [the tensor/context evidence](checkpoint-and-context.json). The latest source checkpoint SHA is `4854cd0deefe1199beadb2a2b3aaa692f655666d22180bf7d769e4ed92c9b215`.

## What the completed comparisons actually established

| Selected control | Total learned updates in this lineage | Waveform cosine | Normalized waveform error | Teacher mel error | Median level deficit |
|---|---:|---:|---:|---:|---:|
| Corrected pilot r5 | 1,009 | 0.5254 | 0.4900 | 1.6493 | 9.84 dB |
| After r6 | 1,509 | 0.6342 | 0.3778 | 1.5218 | 3.28 dB |
| After r7 | 1,759 | 0.6661 | 0.3522 | 1.4313 | 2.62 dB |

The selected lineage has consumed approximately 34.953 scored hours. Updates from discarded models do not count toward this model's learning.

The mel-cap candidate improved waveform error by only 0.95% and worsened mel error by 5.78%. The quiet-phase candidate increased mean quiet residual by 5.70% and its repeating-pattern residual by 25.67%. Neither is retained. All final 0.99 and strict quiet goals remain unmet; these numbers are not percentages of perceived quality.

## Recommended next decision

Before another training launch, settle one complete baseline in writing:

1. Preserve the frozen encoder, raw latents, original teacher and aligned continuous targets. No extra latent-matching loss is needed simply to give both decoders the same latent input.
2. Resolve normalization calibration and the adapter's per-phase information path as explicit architecture/conditioning decisions. Do not silently change an existing checkpoint's inference function by replacing statistics.
3. Specify the waveform/example weighting, mel reconstruction, and a short declared warmup followed by GAN/feature matching. Use mechanical checks to establish readiness: correct targets, masks, finite gradients and causal streaming. Keep 0.99 and quiet fidelity as visible final reconstruction goals rather than prerequisites for starting the objective.
4. Use one stable, representative diagnostic mixture and a meaningful sustained budget. Review actual phase, amplitude, event quality and learning trends at planned checkpoints, instead of changing coefficients every few hundred updates.

If a complete baseline still cannot learn phase and detail, then change the synthesis head or retain a low-rate portion of the original teacher decoder. The current evidence does not justify assuming that more loss tuning, more GPUs, or an optimizer replacement will solve it.

No new architecture changes, source commits, releases or training launches were made as part of this structural review. The prepared perceptual-trial code remains unlaunched.
