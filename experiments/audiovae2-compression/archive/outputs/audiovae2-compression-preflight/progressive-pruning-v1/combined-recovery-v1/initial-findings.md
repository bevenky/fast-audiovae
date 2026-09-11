# Combined initialization: saved-result audit

The combined candidate passes all **13/13 startup windows** at **1.955965 microFS residual RMS**, a **93.01% reduction** from B. It uses B's actual upstream inputs, preserves all three B residual mixers, and changes only the existing native upsampler. The saved run reports baseline parity, native constraint/writeback success, and preservation of teacher, outer layers, and B stage2 operators.

| Metric | B initialization | Combined initialization | Change |
|---|---:|---:|---:|
| Waveform MAE | 0.00662276 | 0.00688386 | +3.94% |
| Mel error | 0.22257185 | 0.23451410 | +5.37% |
| Group MSE | 0.00249605 | 0.00271076 | +8.60% |
| Overall quiet residual RMS | 0.00013631 | 0.00014837 | +8.84% |
| Nonquiet correlation | 96.5930% | 96.3448% | -0.2482 percentage points |

| Disjoint passed windows | B | Combined |
|---|---:|---:|
| Startup, 13 total | 0 | 13 |
| Other near-silence, 171 total | 171 | 169 |
| Remaining quiet, 2,360 total | 580 | 447 |
| All quiet, 2,544 total | 751 | 629 |

The net loss of 122 quiet passes comprises 13 startup gains, two other-near losses, and 133 fewer remaining-quiet passes. These are net category counts, not a per-window transition analysis. The two current near-silence failures are amplitude-only and after 800 ms. The ten 20–40 ms source-reference-zero windows still fail the reconstruction limit. These regional views overlap and must not be added to the disjoint table.

The fixed 96-source support is unchanged: 11,246,952 waveform samples, 8,809,920 nonquiet samples, and 2,437,032 quiet samples. Zero overshoot samples are reported; peak level is descriptive, not a ranking criterion.

This is a successful startup correction with a broader reconstruction tradeoff. It does not establish that the correction survives unrestricted joint training, that all silence is repaired, or that trained quality improves. The approved 2,500-step recovery will supply that separate evidence; no training result is inferred here.

Source: `combined-startup-init-v1/aggregate-results.json`; SHA256 `7ee2b3bf2a5dbb00b52c071cdfd22c068f79878d7cf472888359aacfdc435a2c`. This note uses saved aggregate data only, with no model runs or remote operations.
