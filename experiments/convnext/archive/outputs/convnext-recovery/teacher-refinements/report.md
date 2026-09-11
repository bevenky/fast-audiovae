# Teacher-guided decoder refinement results

All seven bounded experiments have completed. None qualifies for adoption. Keep the retained joint-head spectral candidate and original checkpoint unchanged. No combination is justified by these results, and no further trial has been started.

## What was compared

Every arm starts from the same retained candidate, with the frozen AudioVAE2 encoder, identical 64-channel latent means and sealed teacher waveforms. Each receives 256 updates on 2,048 distinct fitting sources, about 83.6 minutes of scored audio. Source order is fixed, with no repetition within an arm. Debugging arms intentionally reuse the same data.

Selection uses 256 separate sources. The fixed development panel contains 144 natural recordings represented by 282 crops, plus three synthetic fixtures, for 285 crops total. Overlapping crops are observations, not independent recordings. This repeatedly inspected panel is not an unseen final test.

Head experiments use fresh AdamW at `1e-6`. Final-block and auxiliary experiments share their own calibrated rate of `2.5e-7`. They match exposure and initialization, but the final-block versus head comparison does not isolate trainable scope at an identical learning rate. Disposable calibration updates are restored; their 1% peak allowance does not relax final quality checks.

## Completed results

Changes below are paired against the retained candidate on the same development inputs. Negative error changes are better. Mel is the existing crop-mean reconstruction error against the teacher. “Source failures” counts whole-source mel regressions above 1% under the legacy aggregation; regional failures are reviewed separately.

| Experiment | Waveform MAE | Mel error | Natural quiet RMS | Stationary RMS | Maximum peak | Source failures |
|---|---:|---:|---:|---:|---:|---:|
| Retained candidate | Reference | Reference | Reference | Reference | 1.29261 | 0 |
| Selective repair control | +0.129% | −0.708% | −0.609% | −9.878% | 1.29014 | 2 |
| Teacher waveform targets throughout | +0.724% | −1.188% | −0.944% | −6.861% | 1.30524 | 21 |
| Teacher targets + weight normalization | +0.724% | −1.187% | −0.941% | −6.864% | 1.30534 | 21 |
| Teacher targets + short spectral loss | +0.821% | −1.275% | −0.611% | −5.142% | 1.30436 | 22 |
| Final existing block + head | +0.452% | −1.038% | −0.756% | −2.711% | 1.30041 | 16 |
| Globally pooled teacher waveform loss | −0.246% | +0.616% | +0.379% | +16.566% | 1.29065 | 56 |
| Final block + auxiliary teacher features | +0.451% | −1.032% | −0.741% | −4.578% | 1.30060 | 15 |

The retained reference has MAE `0.01161279`, mel `1.01888956`, natural quiet RMS `0.0002532622`, stationary residual RMS `0.0000222316`, and 589 overshoot observations. These are teacher-reconstruction measurements, not listening scores or MUSHRA results.

Weight normalization offers no useful measured benefit over ordinary teacher-directed head training. The short spectral branch reduces some transition failures, but worsens other errors. Final-block adaptation reduces source failures from 21 to 16 compared with the teacher-head arm, while giving weaker silence improvement. Its lower calibrated rate also limits causal attribution. None supplies demonstrated complementary gains that justify stacking changes.

Against its matched final-block control, auxiliary guidance improves stationary residual RMS by 1.919% and reduces source failures from 16 to 15. Natural quiet RMS worsens by 0.0157%, mel by 0.0055%, and maximum peak by 0.0151%; waveform MAE improves by only 0.0008%. This is a small, mixed signal, not a qualified repair.

Pooling improves waveform error and reduces total overshoot observations to 568, but raises stationary noise by 16.6%. Five crop observations still have worse overshoot counts or peaks. A lower global maximum is insufficient.

Against original checkpoint 8,890, all seven preserve average mel and quiet improvements inherited from the retained candidate. Nevertheless, whole-source mel failures remain at 2, 7, 7, 6, 3, 0 and 3 respectively in table order; peak-regressing crop counts are 3, 6, 6, 7, 6, 4 and 6. Even pooling, with zero whole-source failures against the original, retains the sparse `freesound:220655` quiet spectral failure. Improving an aggregate or an intermediate candidate is not a no-regression result.

## What the diagnostics explain

Separate inverse-error normalization initially made quiet waveform output gradients about 1,016 times the active gradient norm on the fixed probe. Pooling removes that inflation, but quiet then contributes only 0.00439% of calibration waveform gradient energy. It also exposes a calibration limitation: equal waveform/mel output gradients become waveform parameter gradients 11.6–15.5 times larger than mel. The saved parameter directions locally trade mel quality for waveform error. They do not reconstruct every AdamW update or prove an architecture ceiling. See the [objective audit](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/objective-audit.md).

The [coverage audit](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/source-coverage.md) also limits conclusions: crying and human whistling lack selection coverage; crying, giggling and chuckling lack verified canonical examples. Labels identify source recordings, not event timing within each crop.

## Integrity and inference cost

The first final-block attempt stopped before training on cold teacher-cache mismatch. Restoring the previously established three-call encoder/decoder startup protocol produced bitwise-identical second/third outputs and exact original cached latents, waveform and full-source cache key. No target or tolerance changed. The warmup receipt preserves teacher state SHA256 `8da1691a055ac7eee3d06750846fc1ad1f887adfec13c31340cab7397fde1501`; stabilized latent and waveform hashes are recorded in [the late summary](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/late-v3-summary.json). This startup behavior is not evidence that historical training was wrong.

Remote validation reports 105 passing checks across the existing, pooled-objective and warmup suites. All 2,048 auxiliary teacher-feature captures authenticated the full source and reproduced cached latents/targets exactly. The training-only readout remained frozen and was excluded from student export. Saved receipts confirm preservation of the original engine, checkpoint, retained candidate and canonical inputs. No candidate was promoted.

Deployed topology is unchanged. Weight normalization folds into ordinary weights with zero measured folding difference and zero runtime normalization operations. The auxiliary readout is training-only and excluded from student export. No new CPU RTF was measured, so these runs establish neither an inference speedup nor final streaming/perceptual equivalence.

Retain the existing candidate. The clearest next design issue is calibration against actual shared-parameter gradients while balancing teacher waveform, spectral and quiet errors. Current evidence does not justify adding layers or simply stacking these variants. Review that objective contract before considering another bounded experiment; this report starts no new tests or training.

## Saved artifacts

All pilot checkpoints, reports, source snapshots and stopped-attempt logs are archived on Runpod at `/workspace/fast-audiovae-convnext-20260909-r9/retained-candidates/teacher-refinements-20260910/teacher-refinements-20260910.tar.gz`. Every one of the 173 archive members and the durable copy passed SHA-256 verification. The archive is 183,460,219 bytes with SHA-256 `6e1aecc3ad838cefe64b717acff545682ef01cbd64063595570913db2c8f8031`.

The [archive receipt](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/archive-receipt.json) contains the member manifest. The [comparison integrity checks](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/comparison-integrity.json) confirm identical baseline metrics and fitting-source selection across all completed runs. No commit, merge, production promotion or sustained training run was made.
