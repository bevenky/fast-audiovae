# Latent, target and time-alignment audit

Audit date: 2026-09-09. Scope: read-only inspection of the training implementation and pinned original AudioVAE2 source, plus a small CPU synthetic context check. No checkpoint was modified, no training was started, and no GPU or quality benchmark was run. The completed run's measured losses and checkpoint weights are being examined separately.

**No structural sample-rate, latent-scaling, time-index or insufficient-context bug was found in this path.** The strongest confirmed issue is interpreting the current reconstruction objective as a direct measure of teacher imitation or perceptual quality. Its two reference targets differ, its main original-reference term is uniform-bin log-STFT rather than the mel loss described in the recipe, and this phase has no adversarial or feature-matching training.

## Verified data path

File references below are relative to `work/fast-audiovae/experiments/convnext/audiovae_student/` unless stated otherwise.

| Boundary | Verified behavior | Evidence |
| --- | --- | --- |
| Original assets | Complete source/checkpoint SHA-256 verified before loading, strict original state keys, no optimized weights | `teacher.py:94–165` |
| Actual source waveform | Exact file SHA checked, soundfile reads FP32 mono16k without gain or rate conversion, sample count must match manifest | `source_corpus.py:65–74`, `348–366` |
| Encoder | Full utterance; original upstream right-zero-padding to a multiple of640; raw posterior mean,64 channels,25 frames/second | `teacher.py:216–225`; pinned upstream `audio_vae_v2.py:441–448`, `489–501` |
| Decoder target | Same raw means passed to frozen original decoder, explicit48k conditioning,1,920 samples per latent, no noise block | `teacher.py:121–128`, `227–237`; `cache.py:176–180` |
| Cache | Continuous raw decoder output retained, original sample count and prepared PCM hash retained, detached FP32 targets | `cache.py:61–116`, `163–186` |
| Crop | Same latent interval and matching640/1,920 sample intervals for original/teacher; real left context; only right padding | `cache.py:275–303` |
| Scored region | Context and invalid tail removed before any FFT, independently for each example | `cache.py:250–258`; `batching.py:119–129` |
| True batching | Only right latent padding; different context lengths keep their own score offsets; equal-length score groups preserve per-example mean | `batching.py:105–137` |
| Student output ordering | Channel/phase adapter gives four ordered internal frames per latent; each internal frame emits480 contiguous samples | `model.py:133–147` |
| Teacher state | Frozen FP32, eval, disabled autocast, explicit TF32-off guard; whole-utterance calls do not reuse streaming state | `teacher.py:178–243`; `source_corpus.py:348–375` |

The pinned source's SHA was recomputed during this audit and is `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8`, matching the wrapper's constant.

The teacher uses constant48k bandwidth conditioning, which the student can absorb into its learned parameters. There is no omitted varying conditioning signal or stochastic noise input in this teacher contract. The source encoder's strided layers consume a complete640-sample input frame before its corresponding latent is available; emitting1,920 samples from that latent is consistent with this40ms codec interface.

## Receptive field and crop context

The student's ten dilations sum to18. Its stem, blocks and head therefore require `6 + 6*18 + 2 =116` internal100Hz history frames, exactly29 latent frames. For any sample belonging to latent frame `t`, the earliest possible student latent input is `t-29`. The training entry point checks that configured context covers this value (`model.py:43–45`, `source_training.py:154–155`).

An exact backward index calculation through the pinned teacher's final causal kernel, six residual/transpose stages and input kernel shows that every output sample in frame `t` depends only on latents `t-20..t`; some output phases have earliest input `t-19`. Teacher layers are in upstream `audio_vae_v2.py:20–38`, `75–99`, `176–205`, `286–315`. Thus the29-frame student context also covers the teacher's historical dependency. No unavailable future latent is required.

A CPU toy used the real student module with the full default dilation schedule, small channel widths and residual scales set to1.0 so deep history errors could not hide behind tiny initialization. Cropped versus whole outputs matched exactly at scored starts0,1,28,29,30,40 and64. All available-history transitions, including the first window that discards earlier utterance samples, were covered. Results are in `alignment-toy-checks.json`. This validates indexing and context behavior, not trained quality or full-size numerical performance.

The different teacher zero-padding and student replicate-padding at a true utterance start are an architectural boundary choice, not accidental crop resetting. All later scored windows receive sufficient real context. Startup fidelity should still be assessed on trained outputs.

## Confirmed objective and evaluation limitations

1. **An exact teacher imitation does not generally have zero total loss.** Current loss is `15*teacher_log_STFT + 1*teacher_waveform_L1 + 45*original_reference_log_STFT` (`losses.py:14–21`, `83–111`). At student=teacher, the first two terms are zero but the original-reference term remains. A total of22 could, for example, come entirely from an original-reference distance of22/45, about0.489. This is an illustration, not a measured floor. Teacher-as-prediction on the exact same dev inputs is needed before interpreting the completed run's total.

2. **The implemented reference loss differs from the proposed mel recipe.** It averages absolute log differences uniformly over FFT bins (`losses.py:83–107`); it has no mel projection. The plan describes multiresolution mel and adopts coefficient45 (`docs/convnext-training-plan.md:184–204`). Uniform-bin log-STFT changes the perceptual emphasis and numerical scale, particularly around low-energy bins. The borrowed coefficient is not a validated equivalent. This is a confirmed recipe difference, with an unproven contribution to the observed plateau.

3. **This remains reconstruction warmup.** The loss has no discriminator, feature matching, learned speech-feature or teacher-feature term. Therefore its scalar cannot establish naturalness or waveform-phase quality. The high-level architecture alone does not transfer the quality of a GAN-trained vocoder.

4. **Native-source highband is not directly supervised in this phase.** The actual trainer always reads prepared16k original references (`source_corpus.py:65–74`, `366`; `losses.py:99–107`). Frequencies above8k receive teacher supervision only, even when a retained original file had a higher native rate. This is not a rate mismatch, but it limits what the present objective can learn beyond the teacher.

5. **Final dev loss evaluates only each utterance's leading window.** `_fixed_dev_crop` uses the first64 frames, at most2.56seconds, and zero context (`corpus_training.py:218–228`; `source_training.py:122–140`). It is not a whole-utterance or steady-state streaming quality measure. Leading silence or startup behavior can affect the aggregate disproportionately.

6. **Loss aggregation weights examples equally, not audio duration.** A qualifying short tail can receive the same scalar weight as a2.56-second example (`batching.py:134–137`; `sampling.py:122–125`, `159–162`). This is deliberate serial/batch parity, not lost samples, but the changing duration/language mixture of a one-pass corpus can change the training loss distribution. A single last-batch total should not be mistaken for a fixed-panel convergence curve.

## Structural hypotheses, not confirmed faults

The100Hz trunk is not limited to50Hz audio bandwidth: its480-channel head predicts480 full-rate waveform samples per internal frame. There is no additional latent information bottleneck beyond the same fixed teacher encoder. However, predicting independent contiguous10ms waveform blocks without an explicit periodic source, overlap synthesis or learned high-rate refinement places more burden on training to learn phase and inter-block continuity. This is a plausible inductive-bias disadvantage, not proof that this architecture cannot match the teacher.

Small LayerScale initialization, the optimizer partition, learning-rate schedule, actual learned phase behavior and block-boundary artifacts require separate checkpoint/optimizer evidence. None is established as the cause by this alignment audit.

Before redesigning the decoder, the most informative existing-run evidence is: teacher-as-prediction objective components on the same dev clips; student-to-teacher waveform/spectral components separately from student-to-original components; and trained waveform inspection at480-sample block boundaries. No new training or architecture change is justified by the number22 alone.
