# Fresh target consistency and channel-copy audit

The source audit found no missing causal history, offset, masking or channel-copy defect. It did identify a numerical verification gap: each fresh target passes repeated full-source decoder agreement, but the new producer does not check the teacher when it re-decodes the shorter cached latent crop. Mathematical equivalence is strong evidence, but does not by itself prove floating-point parity across different convolution shapes.

No model or training code changed. This review ran no GPU forward. A separate, already authorized diagnostic will compare six representative fresh crops using its existing teacher waveform computation.

## Exact data path

1. The producer authenticates each prepared source recording, preserves its amplitude, and runs the original frozen encoder on the entire source in FP32 with cuDNN disabled. It uses the raw 64-channel mean latents. It does not re-encode an isolated crop.
2. It decodes the complete latent sequence with the original 48 kHz teacher. Every source receives two identical decodes and must pass bitwise equality after shape warmup.
3. Only then does it slice both latents and teacher waveform from `context_start_frame`. Nonstartup crops retain 30 latent frames before scoring. Both arrays use the same absolute offset and the ratio of 1,920 waveform samples per latent frame.
4. Training re-decodes this cached latent crop to obtain the teacher's group input and full 128-channel stage-4 target. Its waveform and mel losses compare against the cached full-source teacher waveform, not against a student-generated target.

Evidence: [producer](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/audiovae2-compression-preflight/produce_continuation_pairs.py:58), [training batch and teacher trace](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py:100), [actual training path](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py:522).

## Causal-history derivation

The teacher uses left zero padding for causal ordinary convolutions and right trimming for causal transposed convolutions. There is no temporal normalization, attention, stochastic noise block or future-input dependence in this pinned decoder. Snake and sample-rate conditioning are pointwise in time.

A residual stack with kernel 7 and dilations 1, 3 and 9 can reach 78 earlier samples at its own stage rate. A transposed convolution with stride `s` and kernel `2*s` maps an output index `t` back no earlier than `floor(t/s)-1`. Therefore, tracing a complete stage backward maps the earliest index to `floor((t-78)/s)-1`. The stem adds six latent samples of history. The final waveform convolution adds six waveform samples.

Applying that recurrence to the actual rates `[8, 6, 5, 2, 2, 2]` gives:

| Boundary | Maximum prior latent frames needed |
|---|---:|
| Stage 1 output | 17 |
| Stage 2 output | 19 |
| Stage 3 output | 19 |
| Stage 4 output | 20 |
| Stage 5 output | 20 |
| Stage 6 output | 20 |
| Final waveform | 20 |

The final waveform reaches back 19 latent frames in 1,650 of its 1,920 output phases and 20 frames in the other 270 phases. Thirty retained frames exceed that maximum by ten frames. Width reduction preserves these temporal operators and all nine residual units, so the student's receptive-field bound is unchanged.

For source-start crops, both full-source and cropped decoding begin at absolute zero and use the same zero padding. For interior crops, the first scored sample is beyond the complete receptive field of the crop's artificial start. Future source audio is unnecessary. Right storage padding is excluded by the true valid-sample count.

Evidence: [causal convolution definitions](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:20), [residual units](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:75), [decoder construction](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:277).

## Masking and feature alignment

`batch()` scores precisely `[context_frames*1920 : context_frames*1920 + valid_scored_samples]`. It does not score the retained context or padded right tail. The mel loss receives intact contiguous valid slices, rather than a waveform with invalid samples zeroed into its interior. Singleton accumulation uses one shared three-source denominator, so a short tail is not assigned the same weight as a full crop.

At stage 4, each feature time cell corresponds to four waveform samples. The feature loss sums the four validity indicators and uses that count as its weight. A partially valid terminal cell therefore receives weight 1, 2 or 3 rather than 4. All 128 stage-4 channels are compared against the original teacher. The denominator is valid waveform samples multiplied by 128. Teacher features are detached.

The full retained context is still processed before the waveform loss is sliced. Gradients can therefore reach contextual group outputs that contribute through the frozen suffix to a scored waveform sample. Masking does not sever that route.

Evidence: [loss and pooling code](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py:149).

## Channel-copy and conditioning audit

- Effective weight-normalized weights are computed from the current `g` and `v` before slicing, avoiding stale cached `weight` attributes.
- Transposed convolution axes are correctly treated as input channel 0 and output channel 1. Ordinary pointwise weights use output axis 0 and input axis 1. Depthwise channels use one shared input/output index.
- Snake parameters and output biases use their corresponding channel coordinates. New weight-normalization parameters reproduce the sliced effective tensor; zero-norm rows have a safe representation.
- Stage-3 conditioning selects the narrowed stage-2 output coordinates. Stage-4 conditioning selects the narrowed stage-3 output coordinates. Conditioning is applied once at the input of each stage.
- The group input remains the complete original stage-1 output, before stage-2 conditioning. The group output remains all 128 original stage-4 coordinates. Prefix, suffix and final waveform layers remain independent frozen copies.
- All three residual units and dilations are retained within each altered stage. Removing contributions from discarded channels is the intentional approximation being trained, not an indexing error.

Evidence: [effective weight installation](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/group_model.py:38), [channel slicing](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/group_model.py:227), [conditioning and whole-group construction](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/audiovae2-compression/group_model.py:263).

## Minimum numerical guard

Compare the already computed cropped teacher waveform with the cached full-source target on exactly the valid mask. Retain the original preflight tolerance: `atol=1e-5`, `rtol=1e-4`. Report maximum absolute error, residual RMS, and quiet-window residual RMS separately. In a quiet region, an absolute tolerance alone can hide an error large relative to the target amplitude, so those diagnostics matter even when `allclose` passes.

The six selected fresh indices are 419, 319, 11735, 11982, 477 and 9527. They span the first and last newly generated shards, source-start and interior crops, actual near-zero windows, active high-amplitude audio, a long-offset crying source and a partial-tail whistling source. No source from the reused initial 300-source seed is included. Exact identities, paths and cached-target statistics are in `fresh-consistency-panel.json`.

This representative check closes a process gap; it cannot certify all 11,700 newly generated sources. If the sample fails, diagnose the mismatch before attributing it to training or changing the decoder. If it passes with errors negligible relative to the observed student residual, a systematic fresh-target mismatch becomes less plausible. A future production receipt can record the same comparison during teacher feature extraction, using the teacher forward that training already performs, without adding a second neural forward.

## Completed representative numerical check

The subsequent six-source GPU diagnostic reproduced the cached teacher waveform bitwise on every valid sample for all six selected crops, including their quiet and near-silent samples. Maximum absolute error was zero in each case. This supports the causal-history derivation for the sampled first/last-shard, startup/interior and partial-tail cases. It remains a six-source sample rather than certification of the complete fresh set. Evidence: `regressing-layer-diagnosis-v1.json`, `fresh_targets`.
