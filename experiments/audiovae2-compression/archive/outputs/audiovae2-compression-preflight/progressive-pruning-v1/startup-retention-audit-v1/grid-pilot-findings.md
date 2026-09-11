# Extended-grid startup retention: completed comparison

11 September 2026. The new fresh-C 64-update pilot completed. All model, optimizer, RNG, source and protected-file checks passed. The initial state, development evaluation, calibration anchors, first fitting objective and ordered 768 unique ordinary sources matched the original retention pilot. All recorded non-timing scalars through update 42 agreed exactly. Only six additional positive backtracking fractions were introduced; no trained checkpoint was saved.

| Endpoint measurement | Original retention grid | Extended grid |
|---|---:|---:|
| Development startup passing | 12/13 | 11/13 |
| Calibration startup passing | 6/6 | 6/6 |
| Waveform MAE | 0.00542730 | 0.00541925 |
| Mel error | 0.237675 | 0.237486 |
| Active waveform cosine | 0.970705 | 0.970787 |
| All quiet windows passing | 456/2,544 | 455/2,544 |
| Quiet residual RMS, microFS | 155.426 | 155.283 |
| Rejected weight updates | 21/64 | 16/64 |
| Rejected weight updates in last 16 | 16/16 | 15/16 |
| Mean accepted fraction in last 16 | 0 | 0.0001220703 |
| Full-scale overshoot samples | 0 | 0 |
| Complete disposable pilot time | 166.57 s | 172.15 s |

Pilot time includes preparation, validation and preservation checks; it is not CPU inference RTF. Quiet cohorts overlap. The reused development panel is not an untouched final test.

## What changed

The first old rejected update, 43, now accepts 1/32. Updates 44, 45 and 47 also accept 1/32, while 46 accepts 1/64. Update 48 rejects every fraction. Only update 49 moves afterward, accepting 1/512; updates 50 through 64 all reject weight movement. The last-16 mean fraction is 0.012207% of a full proposal, with fifteen exact zero displacements. Adam moments still advance by the same declared policy.

Waveform MAE changes by -0.1484% relative to the old retention arm. That small difference does not establish a practical recovery improvement. Development startup passing also worsens by one, despite preserving every calibration anchor. The canonical startup failures remain amplitude-only; the teacher's 20–40ms transient remains 0/10 in both arms.

## Decision

Do not extend either retention grid to 2,000 updates. The grid cutoff was a real, avoidable source of the first rejection, but fixing it did not resolve the sustained movement problem. Do not keep shrinking the global learning rate or automatically retry a longer run.

The next relevant bounded investigation is a separately declared corrected-direction comparison on a reproduced stalled state: include all individual constraints in the local feasibility assessment, then use a bounded correction driven by measured nonlinear output excess if a linearly feasible proposal still fails. Verify the true waveform with unchanged thresholds after every candidate and retain the ordinary-quality checks. A feasible local direction would still need a fresh short recovery pilot before a full run.

The first-zero experiment already shows omitted linear constraints for some large fractions and actual failure despite all twelve linear predictions passing for smaller fractions. Thus neither omitted rows alone nor final metric rounding alone explains the result. The late extended-grid stall has not been separately decomposed; the available data do not establish a pure curvature cause or a fundamental architecture-capacity limit.

Keep both completed grids as distinct evidence, along with the ordinary control, initializers, B/C/D/G checkpoints and failed Q. G plus retention remains conditional on a useful retention rule and its own feasible fresh initialization. Further width cuts and standalone constrained-A training remain deferred.

Evidence: [new aggregate](anchor-grid-pilot-v1-aggregate.json), [original matched pilots](pilot-findings.md), [first-zero diagnosis](first-zero-findings.md), [primary-source nonlinear feasibility review](feasible-update-review.md).
