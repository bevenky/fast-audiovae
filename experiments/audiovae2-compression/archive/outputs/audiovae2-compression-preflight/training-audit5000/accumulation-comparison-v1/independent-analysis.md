# Accumulation 3 versus 12: independent result audit

The larger accumulation is a meaningful improvement in waveform fidelity and stability on this controlled comparison. It merits continuation as a training policy. It does not resolve the quiet-audio failures, establish perceptual equivalence to the teacher, or improve every source and internal representation.

Both arms start from the same original step 4500 weights, optimizer state and RNG, and consume the same ordered 1,500 distinct sources. Accumulation 3 makes 500 updates and ends at optimizer step 5000; accumulation 12 makes 125 updates and ends at step 4625. Both see 176,445,657 scored samples. Singleton forward execution, learning rate, coefficients, architecture, teacher and frozen decoder sections are unchanged. The complete source ledger, state restoration, frozen-state preservation and teacher/cache equality checks pass. The original training artifacts remain unchanged. No checkpoint was automatically promoted.

## Full development panel

This is the same 96-source development panel, with 11,246,952 valid samples, 94 sources eligible for active correlation, 2,544 quiet windows and 184 near-silence windows in both arms. Lower errors are better; correlation is the mean of active per-source waveform cosines. The initial column is their common step 4500 checkpoint.

| Metric | Initial | Accumulation 3 | Accumulation 12 |
|---|---:|---:|---:|
| Waveform MAE |0.00489207|0.00552586|**0.00423436**|
| Waveform MSE |0.000274416|0.000302783|**0.000226950**|
| Active waveform correlation |97.1874%|97.2513%|**97.6007%**|
| Common mel error |0.350438|0.342079|**0.335225**|
| Stage2–4 output MSE |0.0118080|0.0116882|**0.0116247**|
| Unchanged weighted objective |0.00523582|0.00586291|**0.00456625**|
| Quiet residual RMS |0.000181435|0.000176862|0.000176647|
| Quiet windows passing |104/2544|119/2544|138/2544|
| Near-silence windows passing |3/184|4/184|18/184|
| Maximum absolute waveform sample |0.97998|0.97462|0.97056|
| Samples above full scale |0|0|0|

Accumulation 12 improves MAE 23.37% and the original weighted objective 22.12% over the control endpoint. It also improves over the shared starting checkpoint. Its linear mel error decreases from 0.00451614 to 0.00311478 and log mel error from 0.337563 to 0.332110 relative to control. These are measured using the same original loss definitions and fixed coefficients, not a new favorable scoring rule.

The objective is `MAE + 0.0006674012905982311 × mel + 0.009304078923434964 × group MSE`. Its improvement does not override the separate quality checks.

## Stability across the entire matched trajectory

Every one of the26 prespecified snapshots is included, at 0, 60,…,1,500 consumed sources. No best checkpoint is selected. Each case contributes equal weight. The speech summary covers only the three fixed Spanish, Kannada and Kashmiri recordings; the whistle summary is one recording. Temporal standard deviation is calculated across all 26 gains for each recording, then averaged across recordings.

| Trajectory measure | Accumulation 3 | Accumulation 12 | Reduction |
|---|---:|---:|---:|
| Speech mean absolute log RMS gain error |0.0390892|0.0173965|55.50%|
| Speech mean temporal RMS gain standard deviation |0.0518913|0.0224922|56.66%|
| Speech mean waveform MAE |0.00467082|0.00404324|13.44%|
| Whistle mean absolute log RMS gain error |0.138462|0.137026|1.04%|
| Whistle temporal RMS gain standard deviation |0.0716892|0.0455731|36.43%|
| Whistle mean waveform MAE |0.000876313|0.000803637|8.29%|

The fluctuations become smaller. Their direction does not reverse less frequently: the speech cases have 14/17/15 reversals with accumulation 3 versus 16/18/18 with accumulation 12. Therefore the evidence supports smaller excursions, not elimination of oscillations or a monotonic trajectory.

Whistle average gain remains low:0.8782 versus 0.8732 of the teacher. Its final gain improves from 0.8902 to 0.9240, but the nearly unchanged mean absolute log-gain error shows that endpoint improvement must not be mistaken for fixing its persistent level bias.

## Regressions and unresolved issues

- **Waveform:**94 of 96 sources improve MAE; two quiet recordings worsen slightly. Welsh increases by 2.64e-7 and Luxembourgish by 6.27e-7. Both changes are smaller than the existing 1e-6 absolute numerical reproduction tolerance, but their measured regressions are retained. This tolerance is not a perceptual quality criterion. Active correlation improves 92 of 94 sources and worsens for Greek by 0.000342 and Lithuanian by 0.000075.
- **Spectral fidelity:**81 sources improve common mel error and 15 worsen. The largest relative increases are Kurdish 3.11%, Marathi 2.96% and Hebrew 2.51%. Linear mel improves 95 sources; log mel improves 76 and worsens 20.
- **Feature alignment:**Stage2–4 MSE improves 44 sources and worsens 52, even though the pooled value improves 0.54%. Largest relative regressions include Hungarian 10.88%, Dogri 10.33% and Freesound36399 at 9.49%. The quiet-selected cohort's pooled group MSE also worsens. The result does not establish uniformly better internal teacher matching.
- **Gain bias:**Across all 94 active sources, pooled RMS gain moves from 1.08101 to 0.96858 and median gain from 1.07298 to 0.97175. Absolute log-gain distance improves 69 sources and worsens 25. The larger accumulation removes the broad excess gain at this endpoint but leaves 90 of 94 sources below the teacher's RMS. Mean absolute log-gain error falls from 0.076785 to 0.034797; this is closer overall, not perfectly calibrated amplitude.
- **Silence:**Quiet residual RMS improves only 0.121%. Still 2,406 of 2,544 quiet windows and 166 of 184 near-silence windows fail. Quiet pass counts improve for 14 sources but worsen for 4: Welsh loses 2 passing windows, Armenian 5, Pashto 1 and Urdu 1. Near-silence passes improve for 12 sources but Freesound277554 loses 1. The whistle's 61 quiet windows still all fail. Thresholds and window coverage were not changed.
- **Expressive coverage:**All 11 expressive recordings improve waveform MAE; that cohort's sample-pooled MAE falls from 0.0190376 to 0.0173988. Its mean active correlation remains 93.4600%, below the full-panel 97.6007%. This is progress, not a claim expressive reconstruction matches the teacher.

## Interpretation and continuation decision

Use accumulation 12 for a further bounded continuation if training continues, preserving its actual optimizer state and the consumed-source ledger. Keep the existing quiet, near-silence, whistle, peak and per-source spectral checks. The remaining 0.99 correlation goal and eventual perceptual comparisons have not been met by this test. Do not classify the candidate as a complete quality fix or silently accept the source regressions.

This is one matched source segment from one starting checkpoint, not a seed-replicated experiment or untouched final test. Accumulation changes both gradient averaging and the number of Adam updates per source exposure. With unchanged beta values, it also changes the amount of audio represented by the optimizer's moment history. The comparison establishes that this larger-accumulation policy works better here; it does not uniquely prove that small-batch gradient variance caused the original late regression.

Machine-readable values, per-source differences, quiet-window changes and input-report hashes are in [independent-comparison.json](results/independent-comparison.json). All input reports are preserved in [results](results/). The paired gain plot is [gain-comparison.png](results/gain-comparison.png).
