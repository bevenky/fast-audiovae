# Completed representative pilot: saved-metric audit

The run finished its declared **1,009 updates and 20.007 hours of unique scored windows**. It stopped at `budget_completed_awaiting_review`; no perceptual/adversarial phase started. This report analyzes saved evaluation JSON, scalar training logs and teacher-coverage metadata. It does not load checkpoints, generate audio, measure inference RTF, or run perceptual metrics.

## Progress is real, but the target is not close yet

The fixed development panel has 84 recordings and 167 crops. Two crops are globally quiet, so waveform-cosine aggregates use 165 nonquiet crops. All evaluations use the same panel and frozen-statistics evaluation mode.

| Update | Mean nonquiet cosine | Normalized waveform error | Teacher mel error | Mean quiet-window residual RMS | Median quiet output excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.0007 | 1.3617 | 3.3544 | 0.029198 | +37.66 dB |
| 200 | 0.0729 | 1.3003 | 3.3432 | 0.020280 | +34.54 dB |
| 400 | 0.1349 | 0.7203 | 2.6114 | 0.006305 | +24.23 dB |
| 600 | 0.2783 | 0.6104 | 2.0514 | 0.002611 | +16.77 dB |
| 800 | 0.3779 | 0.5775 | 1.7140 | 0.001615 | +13.36 dB |
| 1,000 | 0.5190 | 0.4913 | 1.6622 | 0.001082 | +10.43 dB |
| 1,009 | 0.5254 | 0.4900 | 1.6493 | 0.001179 | +10.29 dB |

The final 209 updates improved mean cosine from 0.3779 to 0.5254 and waveform error by 15.2%. This does **not** look like the previous stalled run. Mel improvement slowed, and the strict quiet pass count remains zero, but the continuous quiet residual measurements show substantial progress that binary pass/fail hides. The small residual increase in the final nine updates is worth monitoring; it does not establish a new plateau or identify a cause.

The best individual final crop has cosine 0.9497. No crop reaches 0.95 or 0.99. Correlation is not a percentage of perceptual quality.

## Major remaining problem: quiet noise and weak active audio coexist

At the final update, **164 of 165 nonquiet crops are more than 1 dB too quiet**. Median whole-crop level error is **-9.84 dB**. Turning the output volume up would also amplify the excess audio in pauses, so a single output-gain adjustment cannot solve this.

| Development category | Recordings / crops | Mean cosine | Median output level error |
| --- | ---: | ---: | ---: |
| Speech | 51 / 101 | 0.6263 | -9.20 dB |
| Laughter | 6 / 12 | 0.3646 | -12.59 dB |
| Screaming | 7 / 14 | 0.0057 | -20.04 dB |
| Human whistling | 2 / 4 | 0.0354 | -20.90 dB |
| Generic emotion/nonverbal | 12 / 24 | 0.5121 | -10.42 dB |
| Japanese verbal/nonverbal mixture | 6 / 12 | 0.6309 | -5.13 dB |

The generic category has two globally quiet crops excluded from its cosine and level summary. Emotion labels should not be relabeled as crying, giggling or other unverified events.

Speech improves steadily. Laughter improves but remains far behind. Screaming remains effectively uncorrelated, with mean waveform error approximately equal to predicting silence. Whistling's cosine peaked at 0.0453 at update 800 and ends at 0.0354, while ordinary speech improves. These event-specific deficits deserve separate coverage and reconstruction checks before a broad quality claim.

Median nonquiet level error changed from -0.30 dB at update 200 to -9.74 at 400, -20.63 at 600, -20.58 at 800 and -9.84 at 1,009. This coincides with changes in training stage and model learning; it does **not** isolate mel, normalization or another component as the cause. Near-correct amplitude of early unrelated noise is not successful reconstruction either. Training/evaluation normalization, gain behavior and objective balance need an explicit audit.

## Quiet residuals

All **1,540 quiet 20 ms windows** fail the strict residual criterion. Their mean residual is nevertheless 94.2% lower than at update 200. At the final step, 1,529 also exceed the output-energy limit; eleven satisfy that upper-energy limit but still have the wrong waveform.

The 149 windows with teacher RMS below 0.00001 have median student RMS **0.000937**, approximately -60.6 dBFS, and median excess **+39.87 dB**. The 152 windows with teacher RMS between 0.00001 and 0.0001 also have median student RMS close to 0.001. This suggests a remaining output floor across very quiet conditions. Saved RMS values cannot distinguish DC, periodic artifacts and broadband noise; waveform inspection is required for that distinction.

The largest excess includes pauses in the laughter source `freesound:25794` and a Telugu speech source. The problem spans event and speech data. The thresholds are provisional engineering limits, not independently established audibility thresholds.

Actual training targets contain 116,553 windows below -80 dBFS, about 38.7 minutes by valid sample counts, plus 393,626 windows between -80 and -60 dBFS. There are 114,642 quiet transitions, 11,040 utterance-start crops, 5,134 partial-latent crops and 4,020 partial windows. Teacher targets contain no clipped samples. Quiet coverage is therefore present; "no quiet data" does not explain the remaining error by itself.

## Language and aggregation limits

There are 18 identified languages, but individual language groups have only 1–10 recordings. The largest group, Japanese, includes verbal/nonverbal examples. Conditions, speakers, durations and sources differ, so these are diagnostics rather than controlled language rankings.

Final language-group mean cosine ranges from English 0.4764, Assamese 0.5332 and Gujarati 0.5384 to Hindi 0.7355, Malayalam 0.7447, Nepali 0.7618 and Sindhi 0.7847. Every language still misses reconstruction acceptance. Nine scheduled Indic languages and Chinese/Arabic remain outside this panel; crying/giggling coverage also remains unqualified.

The reported all-crop mean is 0.5254. Giving each recording equal weight gives **0.5271**, close to the reported value. Giving each scored sample equal weight gives **0.5704**, because longer speech crops perform better than short nonverbal crops. Giving the 18 identified language groups equal weight gives **0.6318**, but excludes unspecified-language nonverbals. That larger number must not replace the inclusive result. Report speech and nonverbals separately alongside the overall score.

Beginning versus interior crops have nearly identical cosine means, **0.5269 versus 0.5239**, and median level errors, **-9.69 versus -9.99 dB**. The saved metrics do not suggest that the failures are confined to crop starts, though they cannot independently prove full streaming equivalence or sample alignment.

## Interpretation for the next decision

Keep the completed checkpoint as a recoverable baseline. The current run is learning, so these data do not justify throwing away the architecture. Before extending training, resolve active-speech attenuation, distinguish the remaining quiet output floor, verify actual next-stage expressive coverage and preserve the fixed development panel. Do not introduce several loss changes at once or declare progress using cosine alone. The current run has not established teacher-level quality, expressive reconstruction or deployment readiness.

Full scalar summaries: [completed-metrics-summary.json](completed-metrics-summary.json). Per-crop metadata across all seven evaluations: [completed-crop-metrics.jsonl](completed-crop-metrics.jsonl).
