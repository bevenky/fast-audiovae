# Published methods for the next decoder revision

Reviewed September 10, 2026. This is a research and design review. No model, training run, benchmark or production kernel was changed for it. The earlier custom intermediate-rate head is an unselected hypothesis; its arithmetic estimate is not a quality or runtime result.

The strongest reference for our teacher/student problem is **StreamCodec2**, alongside **DLL-APNet** for spectral vocoder design. **APNet2 and ESTVocoder** supply actual audio evidence for ConvNeXt V2. These are more useful starting points than selecting a new upsampling arrangement solely from its operation count. None establishes a modification that fixes all our failures at unchanged CPU RTF.

## What Supertonic already took from this literature

Supertonic's published decoder combines a Vocos-derived ConvNeXt body with a WaveNeXt-inspired direct waveform head, adding a wider nonlinear head, PReLU, causal convolutions and dilations. Our student already contains that broad structure. The saved Supertonic 3 graph audit shows LayerNorm/GELU blocks and the direct PReLU head, with no GRN or Snake operation. The 2025 paper is a reference design, not complete documentation of the later release's training. [Supertonic paper](https://arxiv.org/html/2503.23108v3#A1.SS1.SSS2).

Original WaveNeXt replaces inverse STFT with a bias-free linear projection and waveform reshape. It explicitly names ConvNeXt V2 as future work. It does not document an extra silence-removal layer that we omitted. [Author's WaveNeXt paper](https://www.okamotocamera.com/preprint_asru_2023_okamoto.pdf).

## The relevant papers and their limits

| Method | Relevant published result | Decision for our work |
|---|---|---|
| ConvNeXt V2 | GRN improves feature diversity; its original evidence concerns vision. | Consider audio evidence below, but do not copy temporal-global normalization into streaming. |
| APNet2 | Explicit amplitude/phase prediction, ConvNeXt V2, spectral supervision and improved discriminators. | Useful source for training methods and an established spectral-head alternative. |
| ESTVocoder | Direct Vocos V1-to-V2 ablation improves PESQ from 3.51 to 3.58 and ViSQOL from 4.863 to 4.878. | Evidence that V2 can help audio, not proof it fixes our errors. Its pitch-based excitation is a separate change. |
| StreamCodec2 | Distillation into a smaller causal MDCT codec. | Closest reference for intermediate teacher guidance at unchanged deployed student structure. |
| DLL-APNet | Distillation into a causalized amplitude/phase vocoder. | Relevant spectral alternative; released implementation and complete streaming contract remain verification gaps. |
| WaveNeXt 2 | Iterative residual denoising improves perceptual results. | Extra passes and centered STFT conditioning do not fit an unchanged compute/causal contract. |
| ComVo | Complex-valued spectral modeling and discriminator supervision. | Training-only discriminator transfer is possible, but reported gains are mixed and preprocessing matters. |
| RepNeXt | Training branches can collapse into a single deployed convolution. | A zero-added-operation training parameterization, not extra deployed nonlinear capacity. |

### ConvNeXt V2 and APNet2: distinguish evidence from names

Standard GRN measures each channel's energy over spatial positions. In APNet2's actual `[batch,time,channel]` tensors, it reduces over the entire time axis. Consequently, a naive port reads future frames; applying it independently to chunks changes the output when chunk sizes change. Its input-dependent scale cannot be folded into fixed weights. Running or framewise replacements would be adaptations needing their own evidence. [Official GRN code](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py#L98-L109), [APNet2 implementation](https://raw.githubusercontent.com/redmist328/APNet2/main/models.py).

APNet2 also has symmetric convolution padding and centered iSTFT. Its backbone ablation replaces V2 with ResNet, so it does not isolate the benefit of GRN over our existing ConvNeXt. ESTVocoder provides the more direct V1/V2 comparison. Neither evaluates our 48 kHz frozen-latent silence and nonverbal panel. [APNet2 paper](https://arxiv.org/pdf/2311.11545), [ESTVocoder ablation](https://arxiv.org/html/2411.11258#S3.SS3).

### Distillation papers: useful guidance, incomplete teacher matching

StreamCodec2 uses trainable linear projections to align intermediate student and teacher feature dimensions. Its 16 kHz results improve student PESQ from 2.650 to 2.744; the teacher remains at 3.132. The reported 910 MFLOPs and 20 ms latency concern that codec, not our decoder RTF. It is a precedent for feature guidance, not evidence that distillation removes every quality gap. [StreamCodec2](https://arxiv.org/html/2509.13670v1#S2.SS3).

Its paper also leaves GRN's exact implementation unspecified. The available [MDCTCodec predecessor](https://github.com/PB20000090/MDCTCodec) uses global-time normalization, while the [StreamCodec repository](https://github.com/PB20000090/StreamCodec) contains demonstrations. This does not establish that StreamCodec2 is noncausal; it means a complete causal implementation has not been verified here.

DLL-APNet reports UTMOS 3.90 for causal APNet2 and 3.98 with distillation, compared with 4.00 for its noncausal teacher. Its experiments use 16 kHz English speech and report FLOPs, not matched CPU RTF. The paper causalizes convolutions but does not resolve GRN's aggregation or the full synthesis alignment sufficiently to certify our streaming contract. The located official repository supplies demonstrations rather than a model implementation. [Paper](https://arxiv.org/html/2509.13667v1), [repository](https://github.com/redmist328/DLL-APNet).

### Newer methods are not automatically cheaper

WaveNeXt 2's own one-core CPU table gives RTF 0.06 for base WaveNeXt, 0.10 for two GAN refinement passes and 0.20 for four. These are 24 kHz vocoder results on that paper's setup. They must not be compared directly with our RTF or Supertonic's full TTS throughput. [WaveNeXt 2](https://arxiv.org/html/2605.25506v1#S4).

ComVo's discriminator-only replacement slightly improves PESQ and spectral error while slightly worsening UTMOS in its ablation. Its actual discriminator removes DC and independently normalizes each waveform's peak. That cannot alone supervise our absolute quiet floor or output amplitude. Our existing experimental real/imaginary-input discriminator preserves amplitude and is not ComVo's complex-valued network. [ComVo ablation](https://arxiv.org/html/2603.11589v1#S5.SS3), [preprocessing code](https://raw.githubusercontent.com/hs-oh-prml/ComVo/main/exp/discriminators.py).

## Transfer to the failures we actually measured

These are proposed applications, not published fixes for this checkpoint.

| Failure | Most relevant mechanism | Constraint |
|---|---|---|
| Repeating stationary-silence residual | Jointly learn nonlinear head features and synthesis weights; use paired spectral feedback to expose the unwanted pattern. | Existing features can represent this fixture. Neither GRN nor overlap guarantees cancellation, and a universal silence correction already harmed real audio. |
| Natural quiet, breathing and whispering | Preserve absolute waveform supervision; guide input-dependent features on real low-level trajectories. | Quiet is not zero. Do not peak-normalize every clip, subtract all DC or force every constant latent to produce silence. |
| Interior laughter/speech overshoots | Jointly train a bounded output, as in AudioVAE2. | Tanh guarantees the range for finite inputs, not faithful transient shape; it does not remove a quiet residual. |
| Whistling, screams and rapid transients | Paired complex spectral supervision and selected intermediate feature guidance. | Existing waveform loss already contains phase information. This changes how errors train the model, not what the teacher knows. |
| Startup, tails and chunk boundaries | Explicit causal histories and synthesis alignment. | A paper's causal-convolution label does not verify normalization, overlap handling or exact output counts. |
| Crying, giggling and other nonverbals | Source-disjoint event-verified training and evaluation alongside speech. | The current panel has sparse named groups and no distinct crying group. No architecture can establish quality on unmeasured conditions. |

The teacher's final causal waveform convolution and tanh are visible in the [official AudioVAE2 source](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py). Our own evidence separating stationary, natural quiet and peak behavior is in the [completed experiment report](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/head-experiments/report.md).

## Recommended design direction

1. **Keep the frozen encoder, the same 64-channel latent input and the trained causal body.** The failed fixed-readout fits do not establish that ten ConvNeXt blocks lack sufficient capacity.
2. **Use the distillation papers to specify training-only guidance at selected decoder stages.** Channel projections alone cannot align AudioVAE2's different temporal rates. Select compatible timestamps and receptive fields from our instrumentation, and supervise features on the student path that actually produces audio. Those selected student stages must be trainable: retaining the body means preserving its topology and checkpoint, not freezing every weight. Export removes the auxiliary projections. An adapter that merely learns around frozen, mismatched features would not solve the main reconstruction problem.
3. **Transfer APNet2's paired real/imaginary spectral error carefully.** Compute it on actual student and teacher waveforms, retaining absolute waveform supervision. Avoid raw phase-angle errors near zero energy. Do not add its predicted-spectrum consistency loss to a waveform-first decoder: the STFT of an actual waveform is already a consistent spectrum. Its training implementation supplies the reference, but weights require our corrected gradient calibration. [APNet2 training code](https://raw.githubusercontent.com/redmist328/APNet2/main/train.py).
4. **Keep bounded-output adaptation separate.** Retain checkpoint 8,890 as control. Train the existing nonlinear head jointly against final teacher waveforms before concluding another waveform synthesis layout is necessary. Appending tanh to unchanged weights or fitting only a linear pre-tanh target is not equivalent.
5. **If we replace the synthesis architecture, use the published spectral family as the comparator.** A StreamCodec-style real MDCT/IMDCT head or APNet/DLL-APNet amplitude-phase/iSTFT head is a better-founded alternative to our untested intermediate-rate head. This changes the output representation; it does not require changing the encoder or latent interface in principle. Reusing our body, scaling to 48 kHz and retaining our streaming contract are still adaptations. The CPU budget must include transforms, output heads, normalization and overlap work. Choose one after that paper-to-code contract audit, not an architecture sweep.

For a literal zero-added-operation option, normalization-free linear training branches can fuse back into the current causal kernels using the [RepNeXt](https://arxiv.org/abs/2406.16004) technique. Keep the same support, grouping and activation position. This alters optimization rather than enlarging the final function class; it is not my primary proposed cure for silence or peaks.

No reviewed paper establishes teacher-equivalent fidelity on all our edge cases with zero added CPU time. Training-only supervision can preserve the inference graph exactly. New input-dependent layers or a synthesis transform require runtime validation before the same claim is justified.
