# Loss-balance diagnostic result

Both arms completed 500 updates on 16,000 identical, unique scored windows each. The teacher and original parent checkpoint remain unchanged. No cap exceptions, uncommitted paired updates or target mutations were found.

| Fixed development measure | Control | Mel cap |
|---|---:|---:|
| Waveform cosine | 0.6342 | 0.6397 |
| Normalized waveform error | 0.3778 | 0.3742 |
| Teacher mel error | 1.5218 | 1.6098 |
| Median level error, dB | -3.28 | -3.07 |
| Mean quiet residual RMS | 0.000419 | 0.000438 |
| Median quiet 480-sample pattern RMS | 0.000224 | 0.000232 |

Decision: retain the control. The cap delivered only 0.95% lower waveform error against the required 10%, while mel error increased 5.78%, exceeding the 5% limit. The configured 25% current-batch cap itself worked on every update; its implementation was not evidence of better quality.

Both arms had zero clipping or active-audio collapse. Neither passes the final 0.99 reconstruction target or the strict quiet-window goal. These are teacher-reconstruction diagnostics, not perceptual-quality percentages. The fixed panel remains 84 recordings, 167 crops and 18 identified speech languages; it does not qualify every training language.

The training sequence covered all 22 scheduled Indic languages and 110 speech languages overall. It contained 9.9495 scored hours per arm. Source labels for expressive files are not dense event annotations.

Next: the approved isolated quiet-phase trial, starting from the retained control. All checkpoints and original results are preserved. No changes have been committed or released.
