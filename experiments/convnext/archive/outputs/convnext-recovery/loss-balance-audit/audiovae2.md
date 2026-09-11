# AudioVAE2 transfer audit

Read-only source review, 2026-09-10. No inference, training or benchmarks were run for this audit. The retained student and all experiment runners remain unchanged.

The teacher supplies useful architectural mechanisms, but the public material does not establish a special silence loss or a complete reproducible AudioVAE2 pretraining recipe. We should separate what its implementation guarantees from what its training may have learned.

## Verified implementation

The frozen source is [audio_vae_v2.py](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:1), SHA256 `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8`. Its architecture agrees with the [official implementation](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py). Local line references below refer to that frozen copy.

| Mechanism | Teacher implementation | Relevant student difference |
|---|---|---|
| Latent interface | Raw encoder posterior mean, 64 channels, 25 Hz; no inference whitening or posterior sampling. Lines 489–501. | Receives those same coordinates; internal four-phase expansion is a learned decoder representation. |
| Progressive synthesis | Six transposed-convolution stages; three residual units with dilations 1, 3, 9 after each. Lines 176–210, 295–307. | Ten residual ConvNeXt blocks work at 100 Hz. |
| Activation and parameters | Learned per-channel Snake and weight-normalized convolutions. Lines 41–65, 75–99. | GELU inside blocks, PReLU before final projection. |
| Waveform head | Snake, causal kernel-seven waveform convolution, then tanh. Lines 310–315. | Direct projection emits 480 adjacent samples per frame; no terminal bound or waveform-rate filter. |
| Boundaries | Zero left padding; preprocessing right-pads to whole encoder hops. Lines 20–28, 441–448. | Replicated left boundaries. |
| Conditioning | Optional sample-rate affine embeddings before stages; default decode uses output rate. Lines 218–267, 345–353, 469–473. | Fixed 48 kHz teacher target. No separate rate-control requirement. |

The [released V2 configuration](https://huggingface.co/openbmb/VoxCPM2/blob/main/config.json) sets input/output to 16/48 kHz, encoder rates `[2,5,8,8]`, decoder rates `[8,6,5,2,2,2]`, latent width 64 and decoder width 2048. It omits noise/depthwise overrides; the implementation defaults are `use_noise_block=False`, `depthwise=True`, `cond_type="scale_bias"`, and `cond_out_layer=False` (local lines 359–372). Thus optional random-noise injection is not an explanation for the configured teacher's successful quiet output.

Current student evidence: [block implementation and replicated padding](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/convnext/audiovae_student/model.py:75), [construction and direct waveform head](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/convnext/audiovae_student/model.py:215). The student already has residual paths; saying that it is missing residual learning would be incorrect.

## What the publications establish

The [VoxCPM2 report, section 3.2](https://arxiv.org/html/2606.06928#S3.SS2), confirms the asymmetric causal codec, rate factors and 64-dimensional 25 Hz interface. It describes inheritance from the original causal AudioVAE but does not disclose exact V2 decoder loss weights, spectral windows, optimizer settings or silence-specific treatment. Section 3.5's flow-matching/stop objective and AdamW schedule concern the TTS backbone, not codec reconstruction. The reported two-million-hour corpus likewise must not be presented as a measured requirement for our decoder distillation.

The [original AudioVAE paper, section 3.5](https://arxiv.org/html/2509.24650#S3.SS5), explicitly describes separate VAE training using mel reconstruction, multi-period/multi-scale adversarial training and KL regularization weighted `5e-5`. It calls the causal convolutional architecture DAC-like. This supports the lineage of the training ideas, but does not prove exact unchanged V2 hyperparameters, discriminator preprocessing or auxiliary feature-distillation losses.

The [official fine-tuning script](https://github.com/OpenBMB/VoxCPM/blob/main/scripts/train_voxcpm_finetune.py#L169-L188) saves a reference to the AudioVAE and removes it from the model before constructing the optimizer. Its `loss/diff` and `loss/stop` defaults therefore provide no missing codec training recipe.

For comparison, [DAC's released configuration](https://github.com/descriptinc/descript-audio-codec/blob/main/conf/base.yml#L39-L65) uses mel windows 32 through 2048 at 44.1 kHz and includes adversarial feature matching. These are DAC settings, not verified AudioVAE2 settings. Our unchanged long-mel branch uses 1024/2048/4096 samples at 48 kHz, or 21.3/42.7/85.3 ms; the isolated short branch adds 256/512, or 5.3/10.7 ms. See [local reconstruction configuration](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/convnext/audiovae_student/reconstruction_v2.py:19). Removing inverse-STFT from inference does not remove STFT from a training loss.

## Implications for the failures

- **Silence:** there is no explicit detector or mute gate in the decoder. Tanh has derivative approximately one around zero, so it cannot materially suppress a small residual. The network must learn the correct cancellation for actual quiet latents. Zero audio must still use its encoded posterior mean; zeroing the latent is a different input. Matching padding is relevant to startup, not a sufficient explanation for residuals deep inside a long recording.
- **Peaks:** the teacher has a genuine output bound that our student lacks. Weight normalization does not bound activations or guarantee an output ceiling. Copying tanh would guarantee a ceiling but would initially alter ordinary peaks too; it needs adaptation and is an additional deployed activation.
- **Transients and repeating residual:** progressive high-rate processing provides local waveform coupling that a block projection does not explicitly impose. This is a structural difference, not proof that the current head cannot represent the desired output. More short-window supervision may guide existing capacity without deploying more layers; it cannot manufacture missing capacity if subsequent evidence establishes that limitation.
- **Phase:** magnitude mel losses do not uniquely specify phase. Aligned teacher waveform error remains useful. DAC's [complex multiband discriminator](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py#L94-L161) sees real/imaginary STFT components. But its [preprocessing](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py#L193-L202) removes DC and peak-normalizes each input to 0.8. Consequently copying that discriminator would not provide an absolute amplitude or silence-noise guarantee. Its exact use in V2 is unverified.

## Transfers consistent with the CPU constraint

1. Keep the encoder and raw latent contract fixed. Audit the actual waveform/mel gradient balance and score per-source, quiet, active and transition errors. This is a training correction, not an AudioVAE2 feature claim. A KL term cannot improve a frozen encoder's posterior and is inappropriate as a decoder-only repair.
2. Training-time weight normalization can be materialized into the existing convolution weights before export. This changes optimizer geometry without adding runtime operators; the WN experiment already tests this rather than assuming it works. Preserve exact zero-row handling and verify post-removal weights remain parameters and outputs retain numerical parity.
3. Short-time spectral supervision, adversarial losses and discriminator feature matching cost training compute only. Their gradients must be checked against teacher waveform reconstruction; none guarantees silence or peak preservation. Judge completed isolated results before adding combinations.
4. Allowing an existing late block and its head to adapt jointly, or using a discarded auxiliary teacher-feature readout, also adds no deployed work. These are our proposals, not documented AudioVAE2 methods. Intermediate channels at different sample rates cannot be matched directly without a validated alignment/readout.
5. Do not present Snake replacement, another progressive upsampler, a waveform-rate convolution, tanh, a silence gate, or generic normalization as free. Fixed affine folding is conditional on operator order and boundary semantics; in particular affine shifts across nonlinearities or zero-padded convolutions are not automatically equivalent.

The most defensible fusion uses the teacher's unchanged latent/target contract and well-scaled training supervision while keeping the low-rate student computation. Its public architecture does not justify abandoning measured guard failures, nor does a missing upstream training detail justify assuming that another layer will solve them.
