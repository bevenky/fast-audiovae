# Proposed recovery evaluation contract

Seal this configuration before running either arm. These are proposed pilot screening margins, not measurements of perceptual equivalence or final reconstruction targets.

Both arms start at targeted step 8,490 and finish after 400 generator updates at 8,890. The sole planned difference is the joint generator learning-rate multiplier: 1 versus 0.25. Preserve initial optimizer moments, discriminator state, generator loss-balancer state, frozen model normalization and the same teacher-latent/target cache identity. Use identical reconstruction windows and discriminator-view positions and preserve each final checkpoint. Do not combine a tanh, new discriminator, new loss weights, clipping changes, new normalization, different batch size, or a momentum reset in this comparison.

## Quality panel

Use all 285 sealed held-out crops from 147 sources, including 282 natural crops from 144 sources and three six-second fixtures. The original `evaluate_fusion` provides one unchanged criterion and mask policy. Retain its shared six-sample interior exclusion: 618 excluded samples total; 26,206,830 scored samples overall, 25,342,830 natural. This exclusion is a comparison policy, not missing streaming output.

The helper adds groups from every row without another model call. Natural/synthetic membership uses exactly the three fixture identities and metadata checks. Preserve unknown source conditions; keep existing speech/expressive cohort semantics. Conditions are case-normalized for report lookup only. The full panel must contain 4,073 teacher-defined quiet windows: 3,434 natural plus 639 synthetic. Group/source counts, target-derived quiet masks, target RMS, criterion settings and source/time boundaries must match before/control/candidate exactly.

Primary quiet metric is pooled residual RMS across teacher-defined 20 ms windows, weighted by valid samples. Do not substitute mean window RMS, equal-crop MSE, the 18-example diagnostic subset, or the 40 ms component decomposition. For encoded zero, retain both the full six-second result and the 2–6 second steady result; derive the latter from existing quiet-window records, not a new encoder call.

## Proposed decision gates

- Candidate natural pooled quiet RMS at least 10% lower than simultaneous control and strictly lower than the starting checkpoint.
- Candidate steady encoded-zero residual RMS strictly lower than both control and starting checkpoint.
- Natural raw MAE and diagnostic mel no more than 1% worse than either control or starting checkpoint.
- Speech and expressive raw MAE/mel no more than 2% worse than starting checkpoint.
- Each source-labelled event and language cohort with at least two sources: raw MAE/mel no more than 5% worse than starting checkpoint. All smaller groups remain visible for review; do not describe them as validated.
- Natural maximum peak no more than 1% higher, and sample-pooled peak-excess energy no more than 10% higher, than the starting checkpoint. Peak-excess energy is `mean(max(abs(student)-1,0)^2)` and is gathered during the evaluator's existing forwards; all teacher samples are below full scale. These are non-regression screens, not a peak fix. Counts in overlapping crops are not unique physical events. Report affected source counts, overshoot counts and saturation counts alongside these gates. Zero or undefined denominators do not silently pass a ratio gate.

At the end, report raw per-group and per-source changes as well as pass/fail, including Sindhi speech, laughter, whistling, screaming, breathing and whispering. A failed guardrail means review the preserved checkpoint; it does not authorize retuning a margin after seeing the result. A single-seed 400-step pass is permission to consider a further bounded continuation, not a release-quality guarantee. Current 0.99 correlation and strict quiet thresholds stay visible but are not prerequisites to finish this pilot.

## Coverage and exposure safeguards

Compare train/held-out source IDs, decoded audio hashes, parent recording/time intervals and known speaker/session identities, including reserved appendices. Cross-arm replay is intentional. Any reuse of previously seen training windows within this debugging continuation must be explicitly declared using the user's debugging-data exception; do not report it as new unique corpus exposure. If generating new windows, ensure scored intervals do not overlap earlier intervals in this same branch; causal left context can overlap and must not be counted as new exposure.

The two laughter encoder-cache discrepancies must be resolved or explicitly isolated before the training cache is used. Do not overwrite old held-out targets while comparing to old metrics. A changed cache requires a new preparation identity and matched before/control/candidate evaluation; the old baseline cannot silently stand in for it.

Run finite-state health checks early, without repeated full quality evaluation. The broad panel is evaluated before and after only. Health audio should cover encoded zero, a natural quiet example, laughter, whistling, Sindhi peaks and ordinary speech, with exact identities saved. Teacher and student parameter/buffer hashes, all gradient routes and exact planned sample/discriminator-view counts should remain auditable. CPU streaming parity must remain a separate correctness check before promotion; these H100 quality evaluations are not RTF measurements.
