# Corrected decoder: implementation and first diagnostic

9 September 2026. Branch `Convnext`. Run `corrected-waveform-preflight-v1`.

The corrected training implementation is in place. The bounded H100 diagnostic completed 500 updates in 80.15 seconds and stopped because reconstruction did not pass. No larger training run or adversarial stage started. This is evidence of incomplete learning, not proof that this decoder architecture cannot reproduce the teacher.

## The latent contract is preserved

The frozen AudioVAE2 encoder produces raw 64-channel posterior means at 25 Hz. The original decoder and the new decoder receive the same cached tensor:

```text
z = frozen_encoder(audio)
target = frozen_original_decoder(z)
prediction = trainable_lightweight_decoder(z)
```

There is no second encoder, sampled latent noise or student-predicted latent vector. The new decoder's internal adapter changes its internal representation while accepting the original latent interface unchanged. A test verifies the exact cached latent values arrive at that adapter. Teacher and encoder parameters are excluded from optimization, and cached teacher targets are detached.

Consequently, matching two independently generated latent vectors is not a required loss here: there is only one input latent tensor. What still requires learning is the new mapping from that tensor to the teacher's 48 kHz waveform. Preserving the input representation does not transfer the original decoder's trained weights or guarantee reconstruction quality. The existing AudioVAE2 16 kHz encoder input and 48 kHz waveform output interface is retained.

## Implemented changes

- Preserved the full ten-block ConvNeXt decoder and direct nonlinear waveform head.
- Added masked training BatchNorm at the two documented sites, with fixed-statistics evaluation and folding for deployment.
- Added true multi-resolution mel reconstruction and a normalized waveform error against the same frozen teacher waveform.
- Added MPD/MRD discriminators, adversarial loss and feature matching. These are implemented but were not activated in this reconstruction diagnostic.
- Added explicit gradient balancing, per-clip amplitude and waveform checks, immutable checkpoints and exact source/configuration identities.
- Added bounded, length-bucketed teacher inference and a new cache. The optimizer loop reads cached targets rather than rerunning the teacher every update.
- Prepared 20.007 hours of unused scored audio for a later pilot, with separate diagnostic and held-out utterances. None of those main-pilot hours has been trained in this run.

STFT/mel analysis and discriminators are training components. They are not added to CPU inference. The official paper and graph support the selected components, but the complete Supertonic 3 training source is not public; unreported settings are explicitly our choices. See the [correction plan](plan.md) and [layer comparison](supertonic3-decoder-comparison.md).

## What passed

The final experimental CPU suite passed 384 tests in aggregate, with two skipped asset/GPU-specific tests. Two staging failures were resolved by including an existing configuration fixture; their targeted rerun passed. These checks cover implementation correctness, not perceptual quality. [Check record](implementation-checks.json).

Actual H100 teacher batching passed comparison against serial inference on seven mixed-length clips. Retained waveform agreement was at least 112.65 dB, or exact. The largest reported latent absolute difference was 0.0000153 and the largest valid-waveform absolute difference was 0.00000242. These are batching-related floating-point differences. Within the student run, both decoder paths use the same cached latent values. Initial preparation of 32 utterances took 7.08 seconds including qualification. This was not a matched serial-versus-batched speed benchmark, so no acceleration ratio is claimed. [Teacher qualification](teacher-batch-qualification.json).

The final one-thread CPU folding and streaming check passed over eight chunks and 30,720 samples. Maximum absolute differences were 0.000000644 for folding and 0.00000154 for streaming. No new RTF measurement was made. [Run summary](summary.json).

## What did not pass

The diagnostic fitted 16 utterances at two fixed positions each. These intentionally repeated diagnostic crops are excluded from main-data exposure. A separate 16-utterance sentinel set also used two positions each. Each set contained 31 nonquiet crops and one quiet crop.

| Measure after 500 updates | Fitted diagnostic crops | Unseen sentinel crops |
| --- | ---: | ---: |
| Mean nonquiet waveform cosine, initial | -0.0002 | -0.0012 |
| Mean nonquiet waveform cosine, final | 0.286 | 0.043 |
| Best nonquiet waveform cosine | 0.754 | 0.115 |
| Nonquiet crops with cosine at least 0.9 | 0 / 31 | 0 / 31 |
| Nonquiet crops with lower waveform L1 than silence | 4 / 31 | 0 / 31 |
| Nonquiet crops within 1 dB of teacher amplitude | 9 / 31 | 7 / 31 |

All nonquiet fitted crops improved their cosine and waveform error from initialization, but none met the complete reconstruction gate. The quiet sentinel exposed added noise: teacher RMS was 0.000121 versus student RMS 0.014764. Lower mel loss or a favorable mean cannot conceal that failure. [Initial evaluation](evaluation-step000000.json), [final evaluation](evaluation-step000500.json), [compact metrics](preflight-metrics-summary.json).

These measurements are waveform diagnostics, not PESQ, STOI, UTMOS, DNSMOS or human ratings. They use a different panel and objective from the previous 10,000-step run; their loss values cannot be compared directly with the old score near 22.

## Interpretation and next investigation

The original extreme suppression of waveform gradients has been reduced. In the saved final parameter-gradient probe, mel-to-waveform scaled norm ratios were approximately 1.22 at the adapter, 2.45 at a middle projection, and 4.95 at the output projection. These ratios are still uneven, but are no longer orders of magnitude apart. [Initial gradients](gradient-initial.json), [final gradients](gradient-final.json).

Two limits matter. First, the live final batch had output-gradient norms of 0.50 waveform and 1.20 mel despite equal target shares: the 0.999-decay moving average lagged the changing mel gradients. Second, the standalone final probe evaluated two examples with statistics accumulated from batches of 32, so its absolute scaled norms are not directly comparable with the live norms. Its within-probe parameter ratios are descriptive, not proof of a causal bottleneck.

Before changing architecture, the next bounded comparison should use consistent gradient measurement batches and compare waveform-led fitting with the current balance on the same examples. The learning rate also fell tenfold within this short diagnostic. This budget does not establish that the model is unable to fit, and another long run with an unexplained objective would not resolve that uncertainty cleanly.

The strict memorization gate remains useful on fitted diagnostic crops. Requiring every unseen sentinel crop to reach cosine 0.9 after fitting just 16 utterances is too demanding as an early generalization criterion; sentinels should initially expose instability and noise, with broad quality assessed after representative training. Relaxing that sentinel rule would not change this result because the fitted diagnostic set also failed.

Early normalization freezing remains a hypothesis, not an established cause. The worst saved amplitude results occurred before freezing began, and measurements improved afterward.

No additional experiment was launched after this failure. All source changes remain uncommitted. The previous run and checkpoint are preserved.
