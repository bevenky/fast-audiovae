# Selective silence adaptation: results and next steps

The bounded experiments improved quiet waveform reconstruction, but neither qualified for promotion. The original checkpoint at step 8,890 and all frozen teacher-pair caches remain unchanged. No further training or CPU timing campaign is running.

Both candidates adapted the existing head convolution, shared PReLU and final projection. The encoder, latent adapter, ten ConvNeXt blocks and normalization stayed frozen. Both used the same 2,048 distinct training recordings, approximately 83.6 minutes of scored audio, including 11.1 minutes of teacher-defined quiet samples. A separate 256-source split supplied selection checks. Each arm ran 256 updates, with eight singleton crops accumulated per update. The first-update calibration accepted the shared learning rate of 0.000001.

The ordinary adaptation arm matched teacher waveforms both inside and outside quiet windows. The selective arm matched the teacher in quiet windows and penalized departure from the original student's nonquiet output. Both were deliberately restricted waveform-fitting experiments. They did not use the full student's mel, adversarial or feature-matching training recipe.

| Canonical development panel metric | Original | Ordinary head adaptation | Selective silence repair |
| --- | ---: | ---: | ---: |
| Stationary silence residual RMS | 0.000043793 | 0.000028039 | 0.000032706 |
| Reduction in stationary error | Reference | 35.97% | 25.32% |
| Natural quiet residual RMS | 0.000258495 | 0.000248479 | 0.000249703 |
| Reduction in natural quiet error | Reference | 3.87% | 3.40% |
| Natural waveform MAE | 0.01160303 | 0.01159502 | 0.01159883 |
| Natural mean mel error | 1.048141 | 1.046556 | 1.051917 |
| Maximum absolute peak | 1.296632 | 1.304268 | 1.296757 |
| Scored overshoot observations | 588 | 573 | 585 |
| Natural quiet windows failing the existing strict check | 3,386 / 3,434 | 3,342 / 3,434 | 3,350 / 3,434 |

The canonical panel retains all 285 crops, 147 sources and 26,206,830 scored samples. It is a reused development panel, not an unseen final test. Overshoot observations can overlap across crops and are not distinct physical events. The stationary fixture was evaluated without fitting a synthetic silence anchor.

Selective repair reduced natural quiet error on all 87 natural recordings containing scored quiet samples. On the independent 256-source selection split, its nonquiet waveform displacement from the original student was 87.2% smaller in RMS than the ordinary adaptation arm. This supports the usefulness of preservation during a focused repair. It does not prove inaudibility or unchanged quality.

## The remaining quality gap

Both candidates passed the existing aggregate waveform, mel, high-frequency and transient screens. Neither achieved the required 10% natural quiet improvement, and both increased peaks in some existing overshoot crops. Consequently neither was selected or promoted.

A deeper source-level audit found a gap in the aggregate quality guards:

| Source-level mel result, 144 natural sources | Ordinary adaptation | Selective repair |
| --- | ---: | ---: |
| Sources with increased mel error | 66 | 117 |
| Sources with more than 1% increased mel error | 19 | 25 |
| Largest increase | 13.20% | 12.69% |

The largest increase in both arms occurs on the same Cantonese recording, whose teacher RMS is approximately 0.000771. Its baseline mel error is 0.277836; selective repair increases that to 0.313080. Quiet waveform improvement therefore does not establish preservation of weak spectral detail. Attenuation or averaging of imperfectly reconstructed components is a plausible mechanism, but these results do not prove that it is the complete explanation. A listening comparison has not been performed for these candidates.

The shared failure under two different nonquiet targets points first to the waveform-only objective used in this pilot. It does not identify the PReLU, convolution or projection as a defective layer. The selective validation quiet RMS ratio continued improving at every checkpoint: 0.97957, 0.97516, 0.97221 and 0.96942 at steps 64, 128, 192 and 256. There is no established architectural plateau.

The runner evaluated at its fixed 64-step interval, including step 192. The initial prose plan omitted that interval when listing checkpoints. None qualified at any interval, so this reporting discrepancy did not change candidate selection.

## Preserve the AudioVAE2 contract and cheap synthesis

The frozen AudioVAE2 encoder already provides exactly the same 64-channel latent means to teacher and student. Latent matching is not an additional approximation to optimize here. The decoder must reproduce the teacher's post-tanh waveform more faithfully from those fixed inputs. Retain the causal ConvNeXt body and direct waveform synthesis inspired by Supertonic 3.

The original student's output is useful as a temporary preservation reference. Making it the permanent target outside silence would preserve its existing errors and prevent full convergence toward AudioVAE2 there. Teacher reconstruction remains the eventual target everywhere.

Recommended next sequence:

1. Keep the architecture unchanged and compare the present focused objective with one that restores explicit teacher spectral reconstruction. Reuse existing mel machinery on valid contiguous audio, separately inspect quiet, active and transition regions, and strengthen per-source regression checks. Do not zero masked waveform samples before a spectral transform. Calibrate any new loss weighting on training data rather than reusing stale balancing statistics. This adds no inference operations.
2. Under the qualified objective, compare updating only the final projection with updating the existing nonlinear head jointly. Use the same source schedule and finite-update calibration rule. Previous ridge fits used different objectives and optimization, so they do not isolate whether nonlinear adaptation caused the new gains.
3. If that comparison demonstrates a limitation of the existing head, the smallest true architecture candidate is channelwise PReLU. Repeat the existing scalar slope across all 2,048 channels to preserve the starting function, then adapt the slopes with the head. It adds 2,047 parameters, about 8 KiB, without new convolutions or history. CPU equivalence still needs measurement. There is no evidence yet that the shared slope causes our error. The original [PReLU paper](https://arxiv.org/html/1502.01852v1#S2.SS1) describes both channelwise and shared variants; its image-classification results do not establish an audio-quality benefit here.

Keep peak bounding as a separate, jointly adapted teacher-waveform experiment after selecting the silence approach. Tanh does not address the measured stationary quiet residual. No wider body, extra ConvNeXt blocks or temporal synthesis stage is currently justified by these results.

## Verification and artifacts

All 22 dedicated harness checks passed on PyTorch 2.14.0 with CUDA 12.6 and cuDNN 9.25.1 on the H100. Locally, those and five existing CPU contract checks passed. Every cached pre-head replay exactly matched the full singleton forward. Frozen parameters, normalization buffers, original engine state, retained checkpoint and sealed caches passed preservation checks.

The bounded fit and evaluation phase took 48.9 seconds after initial context loading; integrity loading and final exit checks add time. This is GPU experiment duration, not CPU RTF. Because neither candidate qualified, no candidate CPU performance run or deployment promotion followed.

The full reports and small experimental head artifacts are retained on the pod under `/tmp/fast-audiovae-recovery-20260909/joint-head-silence-v2`. The local [results.json](results.json) contains the source identity, calibration, selection histories, canonical aggregates and row summaries, preservation receipts and hashes of the remote artifacts. The original production checkpoint was not replaced or committed.
