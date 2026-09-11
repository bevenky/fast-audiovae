# Next steps toward teacher-quality reconstruction

**Current execution update:** The user authorized implementation and launch. The chosen target is now 0.99, with 0.95 retained as a milestone. The remaining comparison controls were implemented and passed 48 focused checks locally and on Runpod. The two-arm H100 comparison has started. Its [current protocol](../convnext-objective-comparison/protocol.md) supersedes the older preparation status and thresholds below. Full checkpoints are every 500 updates because of available disk space; evaluations remain every 100.

9 September 2026. The Runpod status was checked live: the previous run ended at step 500 with `gate_failed`, and no GPU training process is running. No new training experiment has been launched. This plan replaces the earlier informal suggestion to adjust the loss balance and try again.

## Acceptance: 0.95 is the target, not the whole quality test

Use zero-lag waveform cosine of at least **0.95 on every nonquiet fitted diagnostic crop**, rather than passing on the mean. Retain waveform L1 at most half the silence-baseline error and RMS amplitude within 1 dB of the teacher. Require two consecutive scheduled evaluations before calling the diagnostic successful. Beginning and interior positions must both pass, with fixed normalization statistics. No fitted time shifts, gain adjustment or polarity correction may improve the score.

Quiet clips need explicit output-noise checks because cosine becomes unreliable near silence. Keep them visible as a separate panel; the existing 0.001 RMS quiet floor is a provisional diagnostic tolerance, not a claim of inaudible noise. Final acceptance needs a stricter assessment of added noise in quiet intervals, including listening, rather than relying on whole-clip amplitude alone.

The same 0.95 per-clip reconstruction target applies to the declared nonquiet held-out acceptance set after representative training. Report failures individually, by language and condition. Do not silently lower that target if it proves difficult. The small unseen sentinel set is an early warning system, not a requirement that a model trained on only 16 utterances already generalize perfectly.

Cosine 0.95 is not 95% perceptual quality. For equal-RMS aligned signals, it still permits a relative squared waveform error of `2 × (1 − 0.95) = 0.10`, or 10 dB waveform SNR. Amplitude, spectral error, quiet noise, PESQ, STOI, UTMOS, DNSMOS and blind listening therefore remain separate checks. The importance of distinguishing signal-error definitions is discussed in the [SI-SDR paper](https://arxiv.org/abs/1811.02508).

## 1. Correct the measurement and acceptance code

**Completed locally:** the per-clip threshold is now 0.95. The initial and final parameter-gradient probes now use the full diagnostic batch, matching the training batch instead of comparing two-example gradients with moving averages learned on 32 examples. Reports record their example count, scored samples and normalization mode. Eighteen focused CPU checks passed. The previous run's reports and source archive have not been rewritten.

**Before the next launch:** add actual parameter-update measurements at the adapter, a middle projection and the output head, together with gradient clipping and active loss shares. Raw gradient norms alone do not establish the size of Muon or AdamW updates. Bind the new run to its exact parent checkpoint, source, objective, schedule and data identities. Treat changed settings as an explicit experimental fork, not an exact resume.

The longer protocol, two-consecutive-evaluation rule and separate perceptual-readiness decision below still need runner implementation. The code is not being described as already supporting these unimplemented transitions.

## 2. One controlled objective comparison

Fork the same saved step-500 checkpoint into two runs. Copy model weights, frozen normalization, optimizer state, latent cache, crops and ordering. Preserve the full architecture and the same raw 64-channel encoder latents in both.

| Run | Objective | Question |
| --- | --- | --- |
| A | Current waveform plus mel objective | Does it continue to fit with a sustained optimization budget? |
| B | Waveform-only reconstruction control | Is the competing mel objective preventing accurate sample reconstruction? |

Both runs get the same schedule: a 50-update ramp from the current 0.00002 learning rate to 0.0002, then a constant 0.0002 during this bounded comparison. Cap each at **2,000 additional updates**. Record identical checkpoint/evaluation intervals of 100 updates and preserve the original checkpoint. This schedule is a proposed experiment setting, not a proven fix or a claim about Supertonic's training.

The active waveform weight is renormalized in B; this is part of the objective intervention and must be logged. Keep the moving-average implementation otherwise unchanged in this first comparison. Neither run updates teacher or encoder weights, consumes new main-training windows, or enables discriminators. These repeated examples are an isolated learnability test whose fitted weights will not seed the representative pilot.

Stop early only for two consecutive complete diagnostic passes, nonfinite training, or material output instability. Otherwise stop at the declared cap and evaluate the trajectory. A failure within that cap does not prove an architectural limit. Do not repeatedly extend the run without identifying what the extension will resolve.

## 3. Let the result choose the next change

- **B passes and A does not:** waveform reproduction is possible with this architecture. Add mel back gradually from the successful diagnostic state and verify a faster-adapting balancer. Keep the 0.95 reconstruction target. This diagnoses objective competition without claiming the old learning rate alone was wrong.
- **Both pass:** select using paired reconstruction, amplitude and noise results, then move to representative training. Do not add architectural changes merely because they were previously considered.
- **Neither passes, but both are still improving:** inspect convergence rate and actual update sizes before deciding on one explicit budget extension. A short deadline is not a capacity test.
- **Neither passes and progress has stalled:** compare one normalization alternative with the existing frozen-statistics path, holding objective, weights and optimizer fixed. The current reports do not establish BatchNorm freezing as the cause.
- **Only after these checks fail:** examine the adapter and waveform head, then the backbone. Preserve the encoder interface and causal output contract. Architecture changes require their own isolated comparison.

The purpose is to identify the limiting factor. It is not a broad optimizer, architecture or learning-rate sweep. Muon already uses its AdamW-compatible RMS adjustment; there is no verified missing factor of 100 in its learning rate.

## 4. Representative reconstruction and perceptual training

After the small-set diagnostic establishes learnability, start a fresh student under the selected recipe on the prepared **20.007 hours of unused scored audio**. Keep the encoder and original decoder frozen; batch/cache teacher inference. Use the reserved multilingual and expressive validation data without training on it. The small diagnostic's fitted weights are discarded to preserve evaluation and exposure accounting.

Do not require unseen cosine 0.95 before ever enabling the implemented adversarial and feature-matching losses. That would conflate final quality acceptance with readiness to test the training components intended to improve quality. A bounded perceptual trial needs a separate, explicit checkpoint review: stable finite reconstruction, valid amplitude and quiet behavior, and an improving held-out reconstruction trend. It does not waive the final 0.95 target. The current evidence does not justify starting that trial yet.

Introduce MPD, MRD and feature matching gradually while retaining waveform and mel anchors. Keep reconstruction and perceptual metrics separate; an adversarial loss has no teacher-zero interpretation. Snapshot before the transition and roll back a regression in reconstruction or added noise. No discriminator or mel-analysis computation is exported into CPU inference.

## 5. Final quality and speed comparison

On a fixed multilingual and expressive acceptance set, compare original AudioVAE2, the new student and the saved baselines using the same recordings and preprocessing. Require the per-clip reconstruction checks above; assess paired PESQ, STOI, UTMOS and DNSMOS changes and their uncertainty instead of accepting a favorable mean alone. Use blind listening for subtle and expressive failures. MUSHRA results require human ratings.

Then verify exact streaming sample counts, no missing chunks, and full-file versus streaming agreement. Measure one-thread decoder RTF on Intel, AMD and Apple at matched 80 and 160 ms streaming calls, and compare like-for-like decoder workloads. Quality and speed are separate acceptance conditions; neither a cosine threshold nor Supertonic's whole-TTS RTF establishes the other.

All changes remain on `Convnext`, uncommitted. The next training comparison is planned but not started.
