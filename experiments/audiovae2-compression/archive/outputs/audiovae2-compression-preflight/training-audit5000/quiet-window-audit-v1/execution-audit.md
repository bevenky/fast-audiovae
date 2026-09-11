# Read-only quiet-window audit: execution and interpretation limits

The paired audit completed successfully. It evaluated the same 96 development sources at preserved optimizer steps 4625 and 5625, reproducing the saved metrics at the existing absolute 1e-6 / relative 1e-5 tolerances with exact counts. All 192 teacher-versus-cache checks were bitwise equal. Both checkpoint files, frozen student modules and teacher state were preserved. No optimizer, model intervention or promotion occurred.

The H100 was idle before launch: no GPU process or training/target worker. The audit took 9.568 seconds after model loading. Ten focused CPU checks passed locally and remotely. There were 192 scored source/checkpoint evaluations, each with a teacher and a student evaluation, plus 33 teacher and 33 student group/suffix warmup forwards at the first checkpoint; the second checkpoint reused the warmed shapes. Raw audio and checkpoints remain on Runpod.

## What failed

| Unchanged 20ms quiet-window criterion | 4625 | 5625 |
|---|---:|---:|
| Total quiet windows | 2544 | 2544 |
| Passed | 138 | 220 |
| Residual-only failure | 2175 | 2030 |
| Output-RMS-only failure | 150 | 122 |
| Both failed | 81 | 172 |
| Any output-RMS failure | 231 | 294 |
| Pooled quiet residual RMS | 0.000176647 | 0.000161971 |
| Window-DC share of residual energy | 1.063% | 2.644% |

Thus broad quiet failures principally measure waveform mismatch, not excessive output amplitude. Most residual energy remains after removing each window's mean; this observation does not distinguish phase, timing, attenuated quiet detail and other alternating errors. Among residual-only windows at 5625, median least-squares gain is 0.880204, median centered cosine 0.910967, and median residual/limit 2.116608. The gain is dot(student,teacher)/teacher_energy, not an RMS ratio.

Aggregate improvement conceals a regression: 95 previously residual-only windows now fail both criteria. The total amplitude failures increased by 63. The saved paired-window table records every transition; there was no threshold change.

## Near-silent subset and startup

There are 184 teacher-near windows (nonzero teacher RMS <=1e-5), from 15 sources. Near failures fell 166→137. At 5625 these are 122 output-RMS-only and 15 both. Median output/cap ratio across all near windows is 1.021892; its 90th percentile is 1.123344. These ratios use the acceptance cap, not teacher RMS.

Thirteen first-20ms near windows contribute 97.566% of near residual energy. The other 171 windows have pooled residual RMS 4.23243e-6, below the 1e-5 residual floor, with 122 amplitude-only and two both failures. This separates a large startup error from small sustained amplitude-margin failures. It does not make either accepted under the existing criterion.

Original 16k source silence and teacher silence differ. There are 185 exactly zero source windows, but no exactly zero teacher window. Ten exact-source-zero windows at source times 20–40ms have teacher RMS about 6.3283e-4 and student RMS about 1.1448e-5 at 5625. The student is much quieter there, and fails teacher reconstruction. All source-zero first-40ms windows account for 99.936% of source-zero residual energy. Outside those first40ms, the 164 source-zero windows have pooled residual RMS 3.99690e-6, with 119 amplitude-only and one both failure. These are observed source/teacher differences; this audit does not establish which encoder/decoder mechanism produces them.

Startup is not a complete explanation: 1610 of 1762 quiet windows after absolute source time 800ms still fail at 5625, mainly residual-only (1477). The paired source/crop indices and context frames are retained in each row.

## Measurement limits

The quiet tests are provisional engineering tolerances, not measured audibility. The 96-source panel is reused development data, not a new independent test set. The source-reference RMS/exact-zero comparison weights native 16k cells over their three corresponding 48k sample slots; it is not a resampled waveform comparison.

Phase templates at 2,4,8,40,240 and1920 samples require at least eight complete cycles in each contiguous quiet run. These are in-sample descriptive templates: even random residuals contribute a finite-cycle mean. They do not identify a tone or a causal layer. No architectural or loss change follows from a phase-template maximum alone.

## Artifacts

- `results/quiet-step4625.json` and `results/quiet-step5625.json`: full saved quality, all quiet windows, source/reference/DC/AC statistics, phase templates and teacher/cache checks.
- `results/paired-windows.json`: exact window pairing and failure transitions.
- `results/completed.json`, `results/preservation.json`, `results/launch.json`: runtime, protected file hashes and frozen-state receipts.
- `execution-summary.json`: compact summaries, linear-interpolation quantiles and input SHA256 hashes derived only from the saved JSON.
- `launch-receipt.json`, `run.log`: idle-GPU prelaunch check, exact command and execution log.

Frozen runner SHA256: `cf197cf04d190aaa0e1546286aa267c1658177c1458cab16f7b0c0cc31839dc9`. Remote output: `/tmp/fast-audiovae-quiet-audit-v1`. No additional model evaluation was needed to produce this summary.
