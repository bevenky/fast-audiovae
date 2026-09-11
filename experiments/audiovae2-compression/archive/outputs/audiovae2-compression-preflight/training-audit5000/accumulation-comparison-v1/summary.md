# Larger accumulation materially improves recovery, but does not fix silence

The approved comparison is complete. Pooling twelve recordings per update improves the current narrowed AudioVAE2 student over pooling three, using the same starting model, AdamW state and 1,500 training recordings. This supports keeping the architecture for the next controlled continuation. It does not establish that all regional quality problems are solved or that batch size alone explains the original regression.

## Matched comparison

Both arms restored the original step-4,500 checkpoint exactly, kept the same learning rate and losses, and processed one recording per forward pass. The control made 500 optimizer updates; accumulation twelve made 125. Both consumed 176,445,657 scored training samples, approximately 1.02 hours. The final optimizer counters are 5,000 and 4,625. Audio exposure, rather than update count, is the comparison axis.

All 15 focused tests passed locally and on Runpod. Both arms passed the full 96-recording starting-quality checks; their starting aggregates were exactly equal. All 3,000 teacher-target checks, 1,500 in each arm, matched bitwise. Frozen components and original files were preserved. The two-arm run took 248.5 seconds. No new CPU inference benchmark was run.

| Full 96-recording panel | Common start | Accumulation 3 | Accumulation 12 |
|---|---:|---:|---:|
| Waveform MAE, lower is better | 0.00489207 | 0.00552586 | **0.00423436** |
| Waveform MSE, lower is better | 0.00027442 | 0.00030278 | **0.00022695** |
| Active waveform cosine, final goal 0.99 | 0.971874 | 0.972513 | **0.976007** |
| Mel error, lower is better | 0.350438 | 0.342079 | **0.335225** |
| Full group-output MSE, lower is better | 0.01180799 | 0.01168822 | **0.01162473** |
| Quiet residual RMS, lower is better | 0.00018144 | 0.00017686 | **0.00017665** |
| Quiet windows passing / 2,544 | 104 | 119 | **138** |
| Near-silence windows passing / 184 | 3 | 4 | **18** |
| Full-scale overshoot samples | 0 | 0 | 0 |

Accumulation twelve lowers final waveform MAE by **23.37% versus the control**, and **13.44% versus the common start**. MSE falls 25.05% versus control. Waveform MAE improves beyond the existing numerical comparison tolerance on 94 of 96 recordings versus both the control and common start; the other two remain within that tolerance. This is not merely a favorable aggregate at the expense of other waveform MAEs.

Other objectives are not uniformly better per recording: versus control, mel error improves on 81 and worsens on 15, while group MSE improves on 44 and worsens on 52. Their pooled improvements are 2.00% and 0.54% respectively. The whole-panel weighted objective improves, but these individual results must remain visible.

Across the full active panel, pooled student/teacher RMS gain falls from 1.08101 to 0.96858. This removes the broad excess gain but leaves mild under-amplification: 90 of 94 active sources are below teacher level. Absolute log-gain error improves on 69 sources and worsens on 25. Lower waveform error therefore does not imply an amplitude match on every recording.

## The trajectory supports a real stability improvement

The same four diagnostic recordings were evaluated at all 26 matched-exposure points. Speech gain variability falls approximately 52–60% in the three speech cases. Their mean waveform MAEs across the complete trajectories also improve; the conclusion does not rely only on the endpoint. Whistling becomes steadier but remains under-amplified on average. These are three speech cases and one whistle, not population-wide gain statistics.

![Matched-exposure gain trajectories](results/gain-comparison.png)

The larger accumulation also makes fewer updates and extends Adam's memory when measured in recordings. The experiment demonstrates the effect of that practical policy, not an isolated measurement of gradient variance. One segment is not a guarantee of long-run convergence.

## What remains unresolved

Quiet residual RMS improves only **0.12% versus control**. Most quiet and near-silence windows still fail their unchanged checks. The better pass counts do not justify calling silence repaired. Active waveform correlation is still below the final 0.99 goal. Whistling level and spectral detail also require continued attention.

Quiet pass counts improve on 14 recordings but worsen on four: Welsh loses two passing windows, Armenian five, Pashto one and Urdu one. Near-silence passes improve on 12 recordings and worsen on one, where freesound277554 loses one passing window. These are regional regressions despite the aggregate improvement; neither endpoint is qualified for release.

The bounded original output head produces no full-scale overshoots in either arm. Excess loudness relative to the teacher and full-scale clipping are distinct issues.

## Decision

Keep the current architecture and carry accumulation twelve forward as the better supported training policy. Plan a continuation from the preserved 1,000-step anchor using explicit source-exposure milestones and a correctly updated optimizer/source ledger. Do not relabel the 125-update diagnostic as a 500-update run, select an intermediate checkpoint, or overwrite the preserved main run. No further training has started.

If a steadier continuation still leaves a reconstruction plateau, the first alternative is a separate reconstruction-aware initialization at the same widths: fit the existing reduced matrices to teacher outputs instead of only slicing out channels. It changes no deployed layer or runtime graph. Compare it against an equally fresh plain-sliced student under matched fitting exposure; do not overwrite the trained checkpoints with a fresh initializer.

Only if reconstruction evidence shows that the reduced dimensions are insufficient should the next architecture candidate restore one internal width modestly. Full-width residual states with low-rank matrix projections are a separate later option. Do not remove more residual units now. See the [conditional student decision memo](../conditional-student-options.md) for the tradeoffs and evidence required; its MAC estimates are not measured CPU speedups.

Evidence: [independent analysis](independent-analysis.md), [execution audit](execution-audit.md), [raw comparison result](results/completed.json), [predeclared protocol](../accumulation-protocol.md).
