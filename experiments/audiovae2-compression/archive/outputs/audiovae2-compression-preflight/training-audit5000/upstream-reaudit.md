# AudioVAE2 upstream re-audit

Checked 10 September 2026. This is a source and saved-evidence review; it made no model updates or new forward passes.

The released decoder does not contain a missing silence correction, normalization layer or output limiter that explains our remaining failures. Our teacher source is current, and the compressed decoder retains its causal geometry, Snake activations, conditioning, residual depth and bounded waveform head. Width reduction nevertheless changes the function substantially. Recovering that function is the work the experiment must still accomplish.

## Source identity and version boundaries

The current official GitHub revision is `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`. Its `audio_vae_v2.py` is byte-identical to our pinned teacher: SHA256 `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8`. The model repository is at revision `32279effe8c19989596f05d353d1447f51d9e915`; its configuration still specifies 64 latent channels, encoder rates `[2,5,8,8]`, decoder rates `[8,6,5,2,2,2]`, and 16 kHz input with 48 kHz output. [Official decoder](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py), [released configuration](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/config.json). Download URLs, file hashes and the exact source comparison are recorded in [receipt.json](upstream-reaudit-sources/receipt.json).

Three versions must remain separate:

| Version | Established interface and recipe evidence |
|---|---|
| Original AudioVAE | The original paper describes causal 16 kHz reconstruction, DAC-like convolutions, mel and adversarial objectives, and KL weight `5e-5`. [Original paper, §3.5](https://arxiv.org/html/2509.24650v1#S3.SS5) |
| VoxCPM1.5 AudioVAE | Its released configuration specifies 44.1 kHz, five encoder stages `[2,3,6,7,7]` and five reversed decoder rates. This is a different checkpoint and configuration. [Official configuration](https://huggingface.co/openbmb/VoxCPM1.5/blob/main/config.json) |
| AudioVAE2 | The V2 report specifies the asymmetric causal codec and sample-rate conditioning, but does not publish a complete codec-pretraining recipe. Its §3.5 AdamW/cosine description concerns latent-generating TTS training. [V2 report, §3.2 and §3.5](https://arxiv.org/html/2606.06928v1) |

The current repository tree contains TTS fine-tuning code, not a standalone V2 codec-pretraining trainer/configuration. Crucially, the released fine-tuning script removes `audio_vae` from `base_model` at line 189 before constructing AdamW at line 199. Its optimizer defaults cannot establish the VAE optimizer. [Official script, lines 185–203](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/scripts/train_voxcpm_finetune.py#L185-L203).

## What the training disclosures establish

The collaborator's older AudioVAE disclosure specifies mel weight 15, discriminator feature matching 2, generator adversarial loss 1 and constant KL `5e-5`. It references DAC's seven mel windows `32…2048`, bands `5…320`, magnitude power 1, log clamp `1e-5` and zero linear-magnitude weight. A follow-up specifies batch 128, two-second clips, LR `3e-5`, cosine decay, 1,000-step warmup, one million updates and clipping thresholds 10/1,000 for discriminator/generator. It does not specify every optimizer field. [Loss disclosure](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845), [training disclosure](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3771286634). Another contributor confirms the DAC basis. [Contributor response](https://github.com/OpenBMB/VoxCPM/issues/175#issuecomment-3841828687).

Issue 353 contains a later V2-specific claim about higher KL, an additional 8 kHz-band mel term and a longer schedule. The API currently identifies that comment's author association as `NONE`; this review has not independently authenticated it as an author disclosure. This does not prove it is false, but it cannot supply a confirmed missing requirement. The long reproduction script in issue 175 is also a user's implementation, not the released trainer. [V2-specific comment](https://github.com/OpenBMB/VoxCPM/issues/353#issuecomment-4913099637).

Neither authenticated AudioVAE reply documents exponential moving average (EMA) model weights. The inspected official DAC trainer also optimizes and checkpoints the generator directly, without an EMA path or warmup. Its base configuration specifies AdamW betas `0.8/0.99`, LR `1e-4` and exponential decay factor `0.999996`; its scheduler steps after every generator update. The configured batch size is 72, while the trainer reports effective batch as actual local batch times world size. Thus 72 alone is not a confirmed multi-GPU global batch. [DAC configuration, lines 26–29 and 71–74](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/conf/base.yml), [DAC trainer, lines 246–278 and 290–312](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/scripts/train.py#L246-L312). Learning-rate decay therefore has precedent in both disclosed older AudioVAE and DAC recipes. EMA remains a separate proposed intervention, not an authenticated omitted recipe component. Neither precedent selects a decay schedule or validates an EMA repair for this short decoder-compression run.

## Architecture comparison

| Released V2 mechanism | Location in official source | Current compressed group |
|---|---|---|
| Left-only zero padding and transposed-convolution right trimming | Lines 20–38 | Preserved |
| Weight-normalized convolutions and learned per-channel Snake | Lines 41–65 | Effective weights and selected Snake channels copied; all retained trainable-group parameters can adapt |
| Upsampler and three residual units with dilations 1, 3, 9 | Lines 176–209 | All nine units across stages 2–4 retained |
| Per-stage sample-rate scale/bias conditioning | Lines 218–267, 323–352 | Preserved; narrowed coordinates follow their channels |
| Final Snake, neighboring-sample convolution and tanh | Lines 310–314 | Original frozen suffix retained |
| Deterministic posterior-mean encoding | Lines 489–501 | Same frozen encoder and cached raw latents |

These are direct source observations, not inferred improvements. [Exact source lines](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py#L20-L501). No noise injection is enabled in this configuration. No quiet-specific gate or DC subtraction appears in its decoding path. Tanh bounds peaks but cannot force small inputs to reproduce a particular micro-amplitude floor.

Preserving the layer types does not preserve the trained calculation. A pointwise projection computes `y[j] = sum_i W[j,i] * x[i] + b[j]`. Removing half the input channels removes their contributions, including cancellations, even for an output channel we retain. Upsamplers also mix channels. Subsequent Snake and residual calculations operate on these changed inputs. Retaining all dilations preserves temporal support, not cross-channel capacity or waveform equality. This is a mathematical consequence of the implemented slicing, not evidence of a slicing bug. See [`group_model.py`, lines 227–269](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L227).

## Consequences for this diagnosis

Our current objective is explicitly a compression objective: sample-pooled teacher waveform L1, five-scale mel reconstruction and full-width stage-4 feature MSE. It is not the original VAE's pretraining objective. [`run_pilot.py`, lines 149–196](../../../work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py#L149).

With a frozen encoder, its posterior KL has no gradient with respect to the compressed decoder, so adding KL cannot repair these decoder errors. The same raw latent input already preserves the encoder contract; hidden decoder features need not be called or constrained as latents. Adversarial and discriminator feature-matching objectives could later help perceptual detail, but their omission does not establish the cause of the observed small DC drift or recent broad amplitude change. Reusing their original numerical weights with different losses and reductions would not reproduce the original balance.

The saved targeted diagnosis already found exact teacher-to-cache waveform agreement and exact waveform recovery when the true teacher stage-4 output was passed through the student's frozen suffix. Thus the released suffix can reproduce the target; the narrowed group's remaining error is sufficient to account for the measured downstream differences. That does not locate a uniquely faulty residual unit, prove insufficient capacity, or distinguish an unfavorable optimizer update from an objective tradeoff. [Measured layer diagnosis](layer-diagnosis.md).

The appropriate next step is the authorized replay of the actual regressing updates: inspect which loss and parameter changes move the shared group output into directions the frozen suffix amplifies. No source finding here justifies another architecture change, additional pruning, a silence clamp, or importing an unverified V2 schedule before that replay.
