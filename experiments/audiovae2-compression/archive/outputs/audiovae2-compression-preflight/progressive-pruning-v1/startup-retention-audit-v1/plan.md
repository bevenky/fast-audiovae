# Startup retention: current plan

11 September 2026. The matched 64-update pilots, first-zero diagnosis and extended-grid pilot have completed. The grid extension fixes the avoidable zero at update 43 but still rejects 15 of the final 16 displacements, with only 0.148% lower MAE and 11/13 development startup passes. No full 2,000-update retention run has started. The next relevant bounded test compares all individual constraint directions before considering a separate nonlinear correction. See [current report](report.md), [grid findings](grid-pilot-findings.md) and [first-zero findings](first-zero-findings.md). Historical pre-pilot plan details follow. All previous experiments, checkpoints and source snapshots remain preserved.

## Evidence behind the change

D and G completed their independent 2,000-update experiments. G improved waveform MAE and aggregate quiet passing against matched B, but neither D nor G achieved startup passing at its trained endpoint. Constrained-A and combined-C initializations did pass all 13 measured development starts; C subsequently lost that property under ordinary recovery. Those results remain separate and unchanged.

The original current-batch Q trial stopped before its first completed update because gradient-enabled and no-gradient constraints differed. The numerical audit found a small cold-path difference with zero acceptance flips in the measured panel. Exact repeated-path qualification now addresses that guard failure without changing thresholds.

A fresh-C disposable probe then reproduced loss of startup passing after only four ordinary updates: **13/13 to 2/13**, all amplitude-only failures. Waveform residuals still passed. Interventions showed that updated upstream and downstream trainable blocks can produce the failure even with the upsampler restored; protecting only that operator would miss the observed interaction. Both first-update selected constraint gradient dot products were negative. Maximum-window switching and finite-step effects prevent a claim that positive first-order gradient conflict has been proved. The same first-batch ordinary objective also worsened by 36.19%, so parameter movement alone is not evidence of useful learning. [Measured findings](measured-findings.md), [probe evidence](probe-v1-aggregate.json).

## Original matched pilot protocol, now completed

| Pilot | Common starting point and budget | Only experimental difference |
|---|---|---|
| Ordinary control | Independent fresh original-teacher-derived C initialization, fresh AdamW, original RNG, first 768 ordinary sources, 64 updates | Original waveform, mel and group-feature recovery. |
| **R: startup-only retention** | Same initialization, optimizer, ordinary source order and 64-update budget | Repeated constraints from six fixed calibration startup windows. |

Both retain the existing 384/256 student, all nine residual units, all 90 trainable group tensors, frozen teacher/encoder/outer student stages, original precision and accumulated batch of 12. Ordinary loss coefficients and AdamW settings do not change. Numerical qualification uses the same declared execution policy in the matched pilots.

R refreshes two maxima over the six anchors: waveform-residual excess and output-amplitude excess, each capped at zero. It constrains the actual Adam displacement across all 90 tensors and rescans **all six** windows nonlinearly before accepting a step. Original quiet bounds are unchanged. Current-batch ordinary-quiet and other-near constraints are excluded from this first retention test. The original Q protocol is preserved but superseded as the next full 2,000-update experiment because it does not protect startup between batches that contain startup.

## Decision after the pilots

A full 2,000-update R experiment is conditional on intact calibration guards, useful nonzero parameter updates and a reviewable ordinary-quality trajectory against the matched control. Report accepted fractions, zero-movement counts, correction norms, Adam counters, auxiliary cost and the unchanged development metrics. Adam moments advance once even when an accepted displacement is zero; those steps must not be called effective learning. Preserving anchors by blocking almost every update is not success.

Development remains evaluation-only. The probe already contains a hybrid that passes all six calibration anchors but only 12/13 development starts, demonstrating that anchor feasibility is not a generalization guarantee. Check all 13 development starts and all seven quiet cohorts, including continuous residual and output-level measures; do not substitute pass counts for ordinary reconstruction quality.

The potential full run keeps **24,000 distinct ordinary sources used once**, with **six recurring calibration sources counted separately** in an exposure and compute ledger. Each pilot uses 768 ordinary sources; R's recurring anchors are additional training involvement. This declared reuse falls within the existing data-reuse authorization. No development constraints, new loss weights or inference operations are introduced.

G remains a promising independent candidate. Combining G initialization with startup retention is conditional future work after the C-based comparison establishes useful retention; it is not running now. Constrained-A, C, D, G, failed Q and all controls remain registered. No further width cut, automatic promotion or run beyond the declared budget is proposed. [Experiment register](../experiment-register.md), [method rationale](literature.md).
