# Late replay trajectory audit

The replay exhibits large, repeated gain swings in the four diagnostic cases. The three speech cases finish at their highest sampled gain, but their median gains are close to the teacher. This is stronger evidence for unstable gain during optimization than for a smooth, irreversible volume increase.

**Qualification remains failed.** The replay diverges numerically at its first update. All four final source MAEs also miss the existing comparison tolerance. These are newly measured trajectories of a close replay, not recovered historical outputs from the original run. The exactness requirement has not been relaxed. No further training, inference or benchmark was run for this analysis; it uses saved JSON only.

## Fixed-case behavior

Gain is student active-audio RMS divided by teacher RMS. A value of 100% means matched amplitude. Measurements occur every 25 updates, including both endpoints, giving 21 observations per source.

| Diagnostic source | Step 4500 | Step 5000 | Median | Observed range | Direction reversals |
|---|---:|---:|---:|---:|---:|
| Spanish | 100.04% | 112.12% | 99.05% | 81.42–112.12% | 10 |
| Kannada | 104.12% | 113.33% | 98.41% | 88.61–113.33% | 9 |
| Kashmiri | 107.30% | 114.36% | 99.20% | 89.51–114.36% | 9 |
| Whistling | 87.09% | 89.02% | 87.09% | 64.05–106.38% | 14 |

There are 20 adjacent intervals. Speech gain rises in 11, 13 and 12 intervals respectively, and falls in the rest. The equal-source mean of the three speech gains reverses direction 11 times. It falls by 10.36 percentage points between 4900 and 4925, then rises by 10.63 points in the last 25 updates. All three individual speech gains reach their sampled maximum at 5000. The late endpoint therefore captures a high-gain phase in this replay.

This does not make the existing endpoint failure acceptable, and does not establish that the original intermediate history had identical values. It does explain why evaluating only every 500 updates can miss substantial variation.

Whistling has a different problem as well: its median gain is only 87.09%, with still larger swings. Its best sampled amplitude and lowest waveform error do not occur at the same checkpoint. Amplitude, correlation and waveform reconstruction must remain separate checks.

## Shared motion is partial, not a single volume knob

Descriptive Pearson correlations across the saved diagnostic series:

| Speech pair | Gain levels, 21 points | Gain changes, 20 intervals |
|---|---:|---:|
| Spanish / Kannada | 0.451 | 0.279 |
| Spanish / Kashmiri | 0.693 | 0.553 |
| Kannada / Kashmiri | 0.831 | 0.773 |

Whistle gain changes correlate with Spanish at 0.719, but with Kannada at -0.091 and Kashmiri at -0.023. The response is therefore partly shared and partly dependent on the input. This supports investigating the learned group's sensitivity; it does not support assuming that one global output-gain adjustment repairs every case. These are descriptive correlations of four selected failures, with no population or causal claim.

## Quiet behavior and loss interpretation

The scored sample counts and active masks remain unchanged across all 21 observations. Quiet-window failures barely move: Spanish 47–48, Kannada 65–69, Kashmiri always 2 and whistle always 61. The final counts equal the initial counts. The large active-gain swings do not resolve the quiet reconstruction problem.

The 25-update averages of the actual training total range from 0.00499 to 0.00910. These are different fitting batches, so their fluctuations cannot be read as fixed-panel quality improvement. Held-out mel and stage-4 MSE were not measured at these extra 25-step snapshots; claiming that they fall monotonically over this trajectory would be unsupported. The original 96-source milestone reports remain the evidence for those longer-interval trends.

## Actual updates and source composition

The dot product between each minibatch's original weighted gradient and its actual AdamW displacement is positive on 33 of 500 updates, or 6.6%. Positive means the displacement is locally uphill for that current batch's loss. The median gradient/update cosine is -0.210, ranging from -0.567 to +0.131. Adam momentum need not descend every individual minibatch, so these observations are not by themselves an optimizer implementation bug.

Six of the last 25 updates have positive dots, compared with a median of one per 25-update interval. That final interval also contains the largest positive mean speech-gain shift. This is a temporal association in the replay. It does not prove that those six updates, a particular source or momentum alone caused the gain change. The diagnostic snapshots do not provide each intermediate update's held-out gradient or actual before/after held-out loss.

The 1,500 sources remain varied: 582 FLEURS, 297 IndicVoices, 263 LibriSpeech, and the remaining 358 from the expressive corpora and vocal-sound sources already selected. Each 25-update interval contains multiple datasets. The final interval contains nine datasets. Its pooled target RMS is 0.0768, within the segment's block range of 0.0636–0.0936; its quiet-sample fraction is 17.8%, within the observed 11.3–19.3%. There is no obvious abrupt dataset or signal-level transition that uniquely explains the final increase. Coarse dataset labels do not rule out individual difficult sources or finer domain effects.

All 1,500 replayed cached teacher targets match the fresh teacher waveform bitwise over the scored samples, with maximum error zero. This rejects stale target waveforms for the inspected segment. It does not repair the separate replay exactness failure.

## Implication for the next experiment

A controlled comparison of larger effective accumulation is a plausible way to test whether the gain swings can be reduced without changing the architecture or losses. This audit does not prove that effective batch size 3 caused the problem or that increasing it will fix it. Compare the same ordered source exposure and fixed quality panel, and judge the whole gain trajectory, quiet behavior and final quality rather than selecting one favorable endpoint.

With matched source exposure, a larger effective batch also reduces the number of optimizer updates and changes Adam's history measured in examples. The result would test the practical batching policy, not isolate gradient variance alone. The present evidence does not justify an untested momentum reset, blind loss reweighting or another architecture change.

Raw evidence: `case-trajectories.json`, `replay.jsonl`, `completed.json`. Exact derived statistics and input hashes: `replay-trajectory-audit.json`. Endpoint tensor and cache qualifications: `discrepancy-summary.json`. No model parameters or existing training artifacts were changed by this analysis.
