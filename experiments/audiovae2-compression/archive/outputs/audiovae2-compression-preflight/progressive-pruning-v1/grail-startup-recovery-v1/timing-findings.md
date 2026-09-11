# Recovery training timing audit

Snapshot: 2026-09-11T12:06:22.251815+00:00. Step 1130/2000. Existing journals and source only; no new benchmark or change to the running experiment.

| Component | Mean seconds/update | Share of timed update |
|---|---:|---:|
| Ordinary student optimization | 0.5874 | 21.7% |
| Startup protection, projection and verification | 2.1208 | 78.2% |
| Total timed update | 2.7130 | 100% |

The matched fresh64 control measured 0.620 seconds/update ordinarily and 2.651 seconds/update with protection, a 4.28x core-update slowdown. Its ordinary learning component stayed essentially unchanged (39.67 versus38.38 seconds over64updates). The current run's ordinary component remains0.587seconds/update.

Source-level explanation: six calibration starts impose twelve inequalities. The projection enumerates all2^12=4096 active-constraint subsets using small CPU FP64 solves. Each update also performs at least eighteen six-anchor scoring forwards in total, plus six gradient forwards with two gradient traversals per source. Further nonlinear correction adds scans and gradients. Observed averages are24.016 scoring forwards and9.016 gradient-source forwards per update. Accepted-normal steps average3.428seconds versus2.010seconds without an accepted normal correction. The4096 base-solver counter is not a count of every extra normal solve.

These are structural counts, not separate time attribution within the auxiliary total. The exact shares spent in QP solves, model gradients, state copies and GPU synchronizations have not been separately profiled.

Observed elapsed3155.23seconds includes3065.65seconds of timed updates,15.42seconds validation, a waiting field fixed to zero in the runner, and74.15seconds outside the timed categories. The waiting field does not independently measure data or other waiting. Outside time includes untimed warmup/data/checkpoint/other overhead; it is not all attributed to a specific operation. Validation is about0.49% of observed elapsed and does not explain the slowdown.

At this snapshot all protected calibration checks pass and there are zero zero-displacement updates. The last128updates average2.493seconds, faster than the lifetime average. A single GPU snapshot showed76% utilization; it cannot establish sustained utilization or allocate bottleneck time. The existing GPU work also synchronizes individual Gram-dot results to CPU and copies parameters, optimizer moments and gradients for rollback; these costs are not separately timed. No training code, optimizer, batch size or inference graph changed in this audit.

Potential later optimization: reuse verified anchor results and workspaces where mathematically identical, and qualify a more efficient exact constrained solver against the current exhaustive reference. Changing optimizers alone does not remove this protection overhead. No speedup has been measured for these proposals.
