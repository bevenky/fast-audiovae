# B reconstruction recovery: completed 2,000-update comparison

B improved recovery at equal training exposure, but did not reach the old 5,000-step model by 2,000 updates. Its final waveform MAE was only 1.41% better than the original student at 2,000 updates; mel error was 12.32% lower, whole-group MSE 38.32% lower, and aggregate quiet residual RMS 10.39% lower. These improvements do not mean all quiet cases improved.

The run used the same first 24,000 distinct fitting sources, original frozen teacher and encoder, original step 0 RNG, fresh AdamW, unchanged losses and coefficients, and physical batch 1 with accumulation 12. All 90 stage2–4 tensors trained jointly. B's four calibrated native operators were the initialization change. The separate constrained-A initializer was not used.

## Fixed 96-source quality

Cosine is waveform similarity, not perceptual accuracy. RMS values below are in millionths of waveform full scale.

| Model / update | Active cosine | Waveform MAE | Mel error | Group MSE | Quiet RMS | Quiet passed /2544 |
|---|---:|---:|---:|---:|---:|---:|
| B initialization |0.965930|0.00662276|0.222572|0.00249605|136.315|751|
| B 1,000 |0.991102|0.00277857|0.156031|0.00172731|91.102|268|
| B 1,500 |0.993078|0.00241563|0.142888|0.00151780|78.197|903|
| B 2,000 |0.992375|0.00261465|0.141059|0.00148276|78.734|1242|
| Original 2,000 |0.991865|0.00265208|0.160883|0.00240390|87.866|1066|
| Original 5,000 |0.995233|0.00196372|0.126230|0.00146269|70.263|1511|

At matched 2,000 updates, waveform MAE improved for 53 of 96 recordings and worsened for 43. Mel and group MSE improved for all 96. Against the old 5,000-step model, B 2,000 had worse waveform MAE for 94 of 96 recordings and worse mel error for 93. The5,000-step comparison is a quality reference with greater training exposure, not a matched-budget experiment.

Recovery was not monotonic: from B 1,500 to2,000, waveform MAE increased 8.24% and active cosine fell, while mel and group error continued improving. Quiet RMS increased 0.69%, even as the number of passing quiet windows rose. Threshold counts and continuous errors answer different questions. No milestone had full-scale overshoot; final peak magnitude was 0.988997.

## What remains in quiet audio

These three final groups are disjoint and account for all 2544 quiet windows:

| Group | Passed | Failed | Residual RMS | Failure details |
|---|---:|---:|---:|---|
| Near-silence in the first 20ms |0|13|38.457|All 13 fail both residual and output-level checks.|
| Other near-silence |136|35|1.457|All 35 failures concern output level only; pooled output-limit excess RMS is 0.257.|
| Remaining quiet audio |1106|1254|81.688|1106 residual-only,38 output-level-only,110 both.|

The seven existing diagnostic cohorts overlap. Their measured comparison is:

| Cohort | Original 2,000 passed | B 2,000 passed | Original RMS | B RMS |
|---|---:|---:|---:|---:|
| All quiet |1066/2544|1242/2544|87.866|78.734|
| All near-silence |171/184|136/184|5.499|10.340|
| Near-silence first 20ms |0/13|0/13|19.763|38.457|
| Nonzero source-reference quiet |902/2359|1101/2359|90.639|81.621|
| Exact-zero source reference 20–40ms |0/10|10/10|158.800|59.476|
| Exact-zero source reference after 40ms |164/164|131/164|1.407|1.260|
| Near-silence after 800ms |50/50|25/50|1.516|1.863|

Thus B recovers the teacher's 20–40ms transient better, while its first 20ms response remains worse. The 35 other-near failures are small output-level excesses despite low reconstruction error; they should not be described as the same failure as startup, nor dismissed as solved. The ordinary quiet residual remains the largest failed-window population. No audibility or perceptual-quality conclusion was measured here.

## Integrity and next step

The aggregate completion audit confirms 24,000 distinct sources in original order, all 24,000 cached-teacher comparisons passing the unchanged tolerance, all 90 Adam counters at 2,000, finite moments and weights, unchanged optimizer settings, protected files and frozen tensors, and identical validation window/region identities at every review. The old 5,000 aggregate is bound to its protected checkpoint receipt. The final checkpoint SHA256 is `43ef31f30cca751167cf91152f3ab039592b06860f0ffbd60531c8862c783954`.

Training covered 16.33355 hours of scored audio. Recorded training-update time was 1201.586 seconds and total elapsed time 1309.675 seconds. This is one preserved trajectory, not a statistical repeatability study or an inference RTF benchmark.

Retain B and the original 5,000 checkpoint. The user has now authorized a separate combined-initialization 2,500-update experiment: reconstruct fresh B, refit its native upsampler's startup constraints on B's actual inputs, then use the unchanged joint recovery recipe. That experiment had not started when this review was written. Its initial quality is a measured baseline, not an automatic rejection criterion; ordinary recovery may still lose startup feasibility. No automatic promotion, further cut, or 5,000-update extension is implied.

Evidence: [aggregate completion audit](aggregate-completed-audit.json), [matched recovery plan](plan.md), and [execution record](execution.md). Only aggregate results and provenance were collected; raw recordings, tensors and source-level reports remain on Runpod.
