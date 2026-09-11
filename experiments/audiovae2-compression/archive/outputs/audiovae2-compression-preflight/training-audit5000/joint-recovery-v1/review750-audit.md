# Review after 750 joint updates

Allowing the final 250 authorized updates is consistent with the frozen review policy. This is a mixed checkpoint, not an all-clear result: the first material source regressions have appeared, but they do not repeat flags from the preceding review. There are no nonfinite-output, overshoot, frozen-state or teacher/cache failures. The final 1,000-update review must determine whether the same regressions persist under the prescribed comparisons; no automatic freezing follows from this report.

| Measure | Versus start | Versus 500 updates |
|---|---:|---:|
| Waveform MAE |−3.40%|+2.30%|
| Mel error |−5.30%|−0.50%|
| Group-output MSE |−13.37%|−4.14%|
| Waveform MSE |−11.56%|+0.70%|
| Quiet residual RMS |−7.66%|−2.03%|
| Near-silence residual RMS |−5.04%|−5.26%|

Correlation is 0.978605, compared with 0.978765 at 500 updates and 0.976007 initially. The pooled active RMS ratio improves from 0.97145 to 0.97616 relative to the teacher. Thus the newest waveform regression is not a global volume reduction.

## Individual sources

There are 29 material metric/source flags: 22 waveform MAE, five active-level and two quiet-RMS flags. None was present at the previous review. Compared with 500 updates, MAE worsens in 73 of 96 sources, while group MSE improves in 91. Compared with the starting checkpoint, MAE still improves in 70, mel in 94 and group MSE in 95.

The five active-level flags are Luganda, Telugu, Polish and Azerbaijani attenuation, plus amplification on `freesound:402835`. For example, Telugu's active RMS ratio changes 0.99877→0.91790 and its MAE rises 64.74% from the preceding review, or 43.02% from start. The amplified source changes 1.00104→1.10701, with MAE +17.29% from the preceding review but only +3.15% from start.

Among the 22 MAE flags, 17 worsen their RMS gain error and 18 lose active cosine similarity. These are overlapping groups. Therefore both level and waveform-shape errors contribute. Cosine is not a frequency-resolved phase measurement, so these results cannot identify a phase defect. The exact active normalized-MSE identity is `(gain−1)^2 + 2*gain*(1−cosine)`. For Telugu and Luganda, the level-error term increases while the angular term decreases, which directly supports an amplitude explanation for their active squared-error deterioration. Other sources, including Mandarin and Manipuri, mainly worsen the angular term. This does not explain their MAE sample by sample.

## Quiet audio and internal boundaries

Quiet RMS improves in 49 of 68 quiet-bearing sources. The eligible material quiet flags are Luganda (+1.85e-5 RMS versus the previous review) and Armenian (+1.15e-5). Near-silence RMS improves in 13 of 15 source subsets and globally reaches 2.76352e-5. Yet near-silence passes fall 5→2 out of 184 and quiet passes fall 184→163 out of 2,544. These count changes must be read alongside continuous residuals and the separate output-amplitude condition; they are not evidence that overall quiet error increased.

On the fixed 12-source boundary panel, stage1 remains exactly equal to the teacher. Complete stage4, stage5 and stage6 error decreases from the previous review, but active pre-tanh cosine declines 0.97705→0.97575 and waveform cosine 0.97831→0.97672. The intervening frozen final Snake and output convolution can respond differently to residual direction despite better hidden MSE. This is a measured discrepancy across that segment, not proof of a faulty layer or a uniquely causal explanation. Stage2/3 selected-coordinate diagnostics remain partial and do not establish that joint adaptation is harmful.

The saved checkpoint confirms 9,000 teacher/cache checks and preserved frozen state. Retain the 500-update and 750-update checkpoints for the final comparison; do not change objectives, layers, or review thresholds mid-run.
