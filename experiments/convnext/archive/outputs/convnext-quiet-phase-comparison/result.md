# Quiet-phase comparison result

Both arms completed 250 updates on the same 8,000 unique scored windows. The frozen teacher and original parent were verified unchanged. The 2% penalty budget was respected on all updates.

| Fixed-panel measure | Control | Quiet-phase penalty |
|---|---:|---:|
| Waveform cosine | 0.6661 | 0.6661 |
| Normalized waveform error | 0.3522 | 0.3529 |
| Teacher mel error | 1.4313 | 1.4410 |
| Median level error, dB | -2.62 | -2.64 |
| Quiet residual RMS | 0.000378 | 0.000399 |
| Repeating-pattern residual RMS | 0.000173 | 0.000217 |

Decision: retain the control. The penalty increased mean quiet residual by 5.70% and the repeating-pattern residual by 25.67%. It also failed the required midpoint direction. Neither arm meets the final 0.99 or strict quiet-window goals.

Further training is paused for the requested [structural training audit](../convnext-structural-audit/review.md). The prepared GAN/feature-matching trial has not launched. No inference architecture changed, and nothing has been committed or released.
