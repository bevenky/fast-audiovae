# L1 head pilot: method audit

Reviewed saved results on 2026-09-10. No additional model calls, tests or experiments were run for this audit. Evidence: [summary](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-l1-v1/audit-summary.json), [coefficient calibration](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-l1-v1/coefficient-calibration.json), [rate calibration](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-l1-v1/rate-calibration.json), [actual-update probes](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-l1-v1/l1-actual-update-probes.json).

The L1 experiment corrected the severe quiet-gradient imbalance and made small improvements in both overall reconstruction objectives. It did not fix natural quiet or qualify for replacement. Importantly, quiet waveform error also worsened on the fitting panel, so the remaining failure cannot be attributed solely to held-out generalization.

## Calibration worked as specified

All nine batches of eight sources had finite, positive waveform and mel parameter-gradient norms. No batch was skipped. The fixed mel coefficient was **0.001979136889896455**, the median of the nine complete-head norm ratios. The original `1e-6` learning rate passed every disposable quality check on its first attempt, without halvings.

Across the nine isolated AdamW updates, each evaluated on the same complete 72-source panel:

| Raw loss | Mean predicted change, gradient dot actual displacement | Mean measured finite change |
|---|---:|---:|
| Teacher waveform L1 | −0.00000143764 | −0.00000135216 |
| Long mel | −0.000179944 | −0.000122259 |

Both aggregate directions improved. The actual steps included AdamW preconditioning and clipping, not a substituted SGD approximation. These calibration updates were discarded, and the retained pilot used fresh optimizer state.

Raw L1 assigned **13.9609%** of waveform-output gradient energy to quiet samples on the 72-source panel, essentially unchanged at the beginning, step 16 and end. The full 2,048-source fit pool contains **13.3079%** quiet valid samples; that latter percentage is sample accounting, not an independently aggregated gradient measurement across all training batches. This removes the previous pooled-MSE gradient starvation without returning to inverse-regional-error amplification.

## Equalizing typical batches does not equalize every layer or pooled panel

| Probe | Weighted mel/wave parameter norm | Weighted mel/wave output norm | Mel parameter-gradient energy in head bias |
|---|---:|---:|---:|
| Beginning | 2.50× | 17.81× | 95.97% |
| Step 16 | 3.36× | 17.79× | 95.05% |
| End | 3.07× | 18.11× | 94.98% |

These are whole-panel accumulated gradients. A median of per-batch ratios does not promise their equality: gradients from different examples can reinforce or cancel. One calibration batch still had weighted mel norm **13.79×** waveform norm; the median deliberately prevented that outlier from setting the coefficient. Most waveform parameter-gradient energy instead fell in `output.weight`, approximately **58–66%**.

The head-bias concentration is a useful localization clue. It does **not** prove that mel's bias gradient caused the quiet regression: these probes did not separate quiet versus active parameter gradients, and AdamW transforms the update further. Nor does the larger output norm invalidate a calibration explicitly performed at parameters.

## Actual later steps do not show a broken optimizer

| Disposable probe | Common-panel measured waveform change | Common-panel measured mel change |
|---|---:|---:|
| Beginning, fresh AdamW | −1.6614e−6 | +1.2880e−4 |
| After 16 retained steps | −3.1595e−7 | −1.7309e−4 |
| After 256 retained steps | −7.6306e−8 | −8.1009e−6 |

The beginning update improves its own batch's waveform **and** mel losses while raising mel on the broader panel. Its gradient-dot-displacement correctly predicts that broader-panel conflict. This is compatible with the nine-update aggregate passing; individual batches were not required to improve both global losses.

The three probes use the same evaluation panel but **different update batches and different optimizer moments**. Therefore their magnitudes are not a controlled learning-rate or momentum comparison. Their measured gradient norms before clipping were 0.00537, 0.00491 and 0.00826, all below the cap of one; clipping did not constrain these three particular steps. Negative waveform/mel cosines alone likewise do not establish failure when the actual update improves both losses.

## Remaining failures are specific, not an overall divergence

Changes relative to the retained candidate:

| Evaluation | Waveform MAE | Overall mel | Quiet waveform RMS |
|---|---:|---:|---:|
| Fixed 72-source fitting panel | −0.253% | −0.358% | **+0.343%** |
| 256-source selection set | −0.352% | −0.282% | **+0.318%** |
| Canonical 144-source natural panel | −0.307% | −0.187% | **+0.395%** |

Mel in the first two rows pools spectral elements; canonical mel retains its historical crop-average reduction. Compare each row with its own baseline, not absolute values across rows.

Quiet-region mel improved **2.345%** on the fitting panel and **1.959%** on selection, while quiet waveform RMS worsened. That shows a real distinction between spectral similarity and sample-aligned quiet error. It does not establish whether quiet waveform MAE improved, because that regional MAE was not separately stored. Raw L1 and a whole-panel mel loss do not guarantee improvement of every region's squared-error metric.

Stationary silence improved **13.01%**, but natural quiet did not. The canonical peak maximum decreased from **1.29261 to 1.28966**, and overshoot observations fell from **589 to 575**. Nevertheless, five crop records across four sources regressed in peak magnitude or overshoot count. Three canonical sources exceeded the 1% mel regression guard: Bengali **+1.487%**, Bodo **+1.361%**, Maithili **+1.022%**. These are individual source results, not conclusions about entire languages.

The supported conclusion is that correcting the recent objective improved its behavior but was insufficient for the strict local fidelity goals. The evidence points to continuing tradeoffs between aggregate spectral/waveform fitting and quiet/local peak preservation, not a demonstrated optimizer malfunction. It does not yet distinguish limited head capacity from loss geometry or insufficient event coverage. Do not promote this candidate or combine failed variants as established fixes. Preserve the current candidate and these measurements; any next diagnosis should use the localized head-bias finding and regional error disagreement to ask a narrower question.
