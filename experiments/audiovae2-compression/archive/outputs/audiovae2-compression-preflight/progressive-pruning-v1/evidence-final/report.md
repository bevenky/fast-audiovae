The first cut is learning well, but I recommend more recovery at **384/256 before another cut**. Preserve its1,000-step checkpoint. This review used saved reports only and launched no training or evaluation.

| Metric | Pruned step0 | Step500 | Step1000 |
|---|---:|---:|---:|
| Active waveform cosine |0.743507|0.985474|0.988697|
| Waveform MAE |0.021871|0.003465|0.003286|
| Mel error |1.029526|0.242775|0.201765|
| Complete group MSE |0.046927|0.005535|0.003539|
| Active output RMS / teacher |11.85%|97.50%|93.65%|
| Quiet windows passing |0/2544|272/2544|358/2544|
| Near-silence passing |0/184|170/184|167/184|
| Peak absolute amplitude |0.080676|0.980502|0.989506|

The last500 updates reduce MAE5.16%, mel16.89%, group MSE36.06%, and aggregate quiet residual RMS16.76%. Mel and group MSE improve on all96 sources. MAE improves on67 but regresses on29. This is meaningful continued learning, not a flat loss curve. The0.99 cosine milestone remains unmet and would not by itself establish perceptual or amplitude parity.

The important regression is active level. The whistling source428921 falls from108.78% to81.25% of teacher RMS. Its waveform MAE increases58.90% and cosine falls0.99164→0.97860, while mel and group MSE improve. Hausa, Indonesian, Luo, Javanese, and Portuguese examples also become quieter while waveform MAE increases. Laughter343940 falls97.41%→87.63% in level despite lower waveform error. Whispering, breathing, shouting and screaming sources show improved reconstruction. Source labels describe the recordings, not verified timestamps of every expressive event.

| Quiet cohort | Passing500→1000 | Residual RMS500→1000 | Output-limit excess RMS500→1000 |
|---|---:|---:|---:|
| All quiet |272→358 of2544|134.40→111.87µ|17.04→9.22µ|
| All near-silence |170→167 of184|6.05→6.47µ|3.42→2.30µ|
| Near-silent first20ms |0→0 of13|21.35→19.85µ|12.85→8.58µ|
| Exact source-zero20–40ms |0→0 of10|348.60→289.57µ|0→0|
| Exact source-zero after40ms |164→164 of164|2.02→3.54µ|0→0|
| Near-silence after800ms |49→49 of50|2.24→3.52µ|0.039→0µ|
| Quiet with nonzero source input |108→194 of2359|137.69→114.61µ|17.68→9.56µ|

µ means one millionth of full-scale waveform amplitude. These cohorts overlap. Sustained source silence now passes all164 checks, which is encouraging, but its output RMS falls9.23→6.77µ against teacher9.60µ. Residual rises while DC-removed residual falls1.91→1.50µ. Interior near-silence behaves similarly. This indicates a growing per-window offset component with reduced varying error, rather than renewed noise amplification. The pass thresholds permit small differences; passing does not mean exact reconstruction. Startup and the teacher's20–40ms transient remain separate unresolved cases, although both improve.

The12-source boundary panel also improves from500 to1000: complete stage4 NRMSE0.2273→0.1755, frozen-suffix stage5 0.1692→0.1336, stage6 0.1561→0.1260, waveform0.1945→0.1808. The shared stage1 prefix remains exactly equal to the teacher. Internal stage2 coordinates are descriptive because the narrowed representation can redistribute information. Stage3 retains all256 channels in this cut despite the report's generic selected-coordinate label. The complete group and waveform remain the deciding comparisons.

Completion reports1,000 updates,12,000 distinct-source positions, no failure, preserved frozen state and protected files, and a retained checkpoint. No overshoot samples were recorded. The final checkpoint receipt and report agree. Full source-ledger and file-hash verification is handled separately by the main audit. No CPU speed or perceptual benchmark has been run for this cut.

The next decision should preserve the recovered width while checking whether continued joint recovery improves amplitude matching as well as waveform and spectral error. Starting the next width cut now would add a new disruption before that tradeoff is resolved. This recommendation does not claim that extra training is guaranteed to remove the remaining errors.

The reported active levels use `overview_window_metrics.by_source[source_id].active_student_energy` and `.active_teacher_energy`, added by the existing `UnifiedMonitor` observer. Pooled level is `sqrt(sum(student_energy) / sum(teacher_energy))`; the whistling value uses the same ratio for `freesound:428921` alone. The energies cover identical teacher-defined active windows and use valid-sample weighting. These are measured waveform RMS ratios, not inferred perceptual loudness. Exact input energies and report paths are saved in `quality-review.json` under `active_level_provenance`.
