# Recipe v2: saved-metrics audit

Historical snapshot. See the [combined audit](review.md) for the newer step 1,700 evaluation and final recommendation.

Read-only snapshot at 2026-09-09T08:23:38Z, through update 1,522. No new inference, optimizer update, audio transfer, or checkpoint transfer was performed for this report.

The run is learning. Its fixed-panel errors improved strongly from step 500 to 1,000, and recent training losses continue falling. These logs do not support a restart solely because a displayed loss appears flat. They also do not establish acceptable audio quality: quiet-region noise, attenuated amplitude, and very weak screaming/whistling reconstruction remain substantial.

## Fixed development panel

All three evaluations use the same 167 source/start pairs, 16,465,920 scored samples and exactly matching reported teacher RMS. The prior control is an older trained student with a different cumulative training budget and data mix, so this is a valid quality comparison, not an isolated recipe experiment.

| Measure | New step 500 | New step 1,000 | Prior r7 control |
|---|---:|---:|---:|
| Mean active waveform cosine | 0.2215 | 0.5685 | 0.6661 |
| Mean normalized waveform error | 0.9299 | 0.4868 | 0.3522 |
| Mean teacher mel error | 2.9534 | 1.9971 | 1.4313 |
| Median active level error, dB | -2.82 | -4.21 | -2.62 |
| Mean quiet residual RMS | 0.010597 | 0.002653 | 0.000378 |
| Median 480-sample repeating residual RMS | 0.009937 | 0.001396 | 0.000173 |
| Quiet windows passing strict checks | 0 / 1,540 | 0 / 1,540 | 0 / 1,540 |
| Samples clipped above full scale | 0 | 0 | 0 |

Step 500→1,000 reduced normalized waveform error by 47.6%, mel error by 32.4%, mean quiet residual by 75.0%, and median repeating residual by 85.9%. Nevertheless, current step 1,000 has worse normalized waveform error on all 167 crops than the prior control and better cosine on only 6.

## Remaining failures

| Condition | Step 500 cosine | Step 1,000 cosine | Step 1,000 median level error |
|---|---:|---:|---:|
| Speech | 0.2685 | 0.6835 | -3.77 dB |
| Laughter | 0.1465 | 0.4049 | -6.46 dB |
| Screaming | 0.0035 | 0.0118 | -12.62 dB |
| Human whistling | 0.0108 | 0.0319 | -16.24 dB |
| Other emotion/nonverbal | 0.2197 | 0.5477 | -4.87 dB |
| Japanese verbal/nonverbal | 0.2285 | 0.6311 | -3.72 dB |

At step 1,000, 160 of 165 active crops are attenuated by more than 1 dB, and 32 active crops have waveform absolute error at least as large as predicting silence. Mean cosine is improving despite this gain bias. Signed median gain changed from −2.82 to −4.21 dB; median absolute gain error changed only slightly, 4.33→4.24 dB, because the earlier model also over-amplified many examples.

The quiet residual is spread across beginning and interior crops: mean RMS 0.002703 versus 0.002571. The repeating-pattern median is 0.001418 versus 0.001357. This argues against a problem confined to stream startup; it does not identify a specific layer as the cause.

Near-silent windows with teacher RMS below −100 dBFS have median output RMS 0.001716 and median gain +45.0 dB. Quiet windows in −100 to −80 dBFS have +31.9 dB median gain; −80 to −60 dBFS windows have +15.7 dB. Across all quiet windows, the median residual is 49.7 times its provisional per-window limit. These limits are engineering checks, not an audibility calibration.

## Training mechanics and gradient allocation

- Every saved update has 32 examples, finite numeric metrics, and the correct monotonically increasing step. Discriminator updates equal max(0, step−500) throughout the snapshot. GAN and feature matching both start on update 501 and reach their full declared mix on update 1,000.
- The new `train/total` chart is raw waveform plus mel loss only. It omits adversarial loss, feature matching and gradient balancing; it is not the complete training objective. The legacy normalized waveform/mel fields remain separate.
- Calibration occurred once at step 500: 512 examples in 16 batches, two fixed-weight passes, 30,736 selected latent frames, 122,944 internal frames. Both passes have the same exact input hash, frozen statistics and unchanged model parameters. Calibration took 25.60 seconds and consumed zero optimizer updates.
- Every recorded generator update invoked clipping at norm 1. Median preclip generator norm fell from 4,074.9 during steps 401–500 to 162.0 during the latest 100 steps. Every discriminator update after step 550 also clips; its latest median preclip norm is 17.2. Because AdamW and Muon transform gradients, these clipping factors do not directly predict parameter displacement or prove learning is stalled.
- No loss branch reported a zero output gradient or hit the balancer scale cap. First-use EMA behavior on update 501 is correct: adversarial and feature-matching current norms equal their initial per-loss EMAs and their scaled norms exactly equal their tiny initial scheduled shares.

| Recent steps | Waveform norm share | Mel norm share | Feature matching | Adversarial |
|---|---:|---:|---:|---:|
| 1001-1100 | 14.9% | 51.2% | 23.5% | 10.4% |
| 1201-1300 | 15.4% | 50.9% | 22.0% | 11.6% |
| 1423-1522 | 17.0% | 50.5% | 20.8% | 11.6% |
| Declared full-stage shares | 30% | 40% | 20% | 10% |

These are fractions of the sum of individual scaled output-gradient norm magnitudes. They are not projections onto the combined vector and are not measured parameter-gradient shares. The slow EMA currently underestimates rising mel and perceptual gradients while waveform gradients stay nearly constant, so waveform reconstruction receives a smaller relative component than the declared 30%. This is a real allocation discrepancy to assess, but these logs alone do not prove that changing the EMA or shares would fix gain/noise.

Recent steps 1,423–1,522 still improve over 901–1,000: mean training waveform MAE 0.02440→0.01811 (−25.7%) and mean mel 2.0206→1.4634 (−27.6%). These are different unique training examples, whereas the step 500/1,000 comparison above uses the same heldout panel. The discriminator loss remains finite, with recent median 1.395; there is no observed discriminator loss collapse to zero.

## What the evidence warrants

Continue the structural audit without treating a scalar chart as proof of failure. The highest-priority unresolved issues are the widespread low-level residual and phase pattern, systematic amplitude attenuation, and sparse expressive reconstruction. The allocation discrepancy deserves attention if the structural audit finds a specific mechanism, but a fresh restart or another learning-rate change is not established by these saved metrics. There is no saved development evaluation beyond step 1,000, so later training-loss improvements must not be advertised as confirmed heldout quality gains.

Detailed machine-readable evidence: [metrics and evaluations](metrics-and-evaluation.json), [prior-control comparison](prior-control-comparison.json). The root audit separately verifies the live checkpoint tensors and source/data provenance.
