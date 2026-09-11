# Native stage-4 reconstruction execution audit

The bounded diagnostic completed successfully. All protected initial/trained group tensors, frozen modules, teacher weights and original files were preserved. Main training remains paused. No fitted model was adopted.

The identical native upsampler fit behaves very differently with adapted versus original teacher residual units. This is direct evidence of downstream coadaptation: closeness to the original internal teacher feature is not, by itself, the trained student's correct internal contract.

| Trained4625 intervention | Wave MAE | Mel | Final group MSE | Active least-squares gain | Quiet failures | Near-silence failures |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00423436 | 0.335225 | 0.0116247 | 0.9371 | 2406 / 2544 | 166 / 184 |
| native_fit | 0.02034975 | 0.507896 | 0.0152263 | 1.5215 | 2544 / 2544 | 184 / 184 |
| teacher_up4 | 0.01978298 | 0.363531 | 0.0091968 | 1.6405 | 2491 / 2544 | 183 / 184 |
| native_fit_original_residuals | 0.00490318 | 0.459561 | 0.0070760 | 0.9021 | 2455 / 2544 | 179 / 184 |

Here, least-squares gain means sum(student×teacher)/sum(teacher²) over pooled active samples. It differs from the active RMS ratio sqrt(sum(student²)/sum(teacher²)). The latter is0.968584 for baseline,1.607603 for native fit,1.663155 for teacher-up4 oracle and0.938856 for fit with original residuals.

The exact teacher-up4 oracle has zero local feature error, yet increases the trained model's least-squares gain to1.641 and waveform MAE by4.67×. Using the same fitted upsampler with the original teacher residual stack lowers MAE from0.020350 to0.004903, but this still trails the unchanged trained model at0.004234. Its mel and quiet-window failures also remain worse. None is a replacement candidate on this evidence.

The trained-input native fit reduces held-out up4 MSE from0.00827351 to0.00149656 (81.91%), and near-silence feature MSE from0.01848965 to0.00094038 (94.91%). Both phases improve. This strong local improvement does not transfer through the adapted residual stack unchanged. The original-residual control has exactly the same up4 local errors as the adapted-residual fit, isolating the downstream difference.

Fitting from the teacher's original retained inputs yields held-out local MSE0.000304315, versus0.001496562 from actual trained inputs and0.003793481 from sliced-initial inputs. That demonstrates an additional upstream representation difference. It does not prove a nonlinear capacity limit or establish that restoring width is necessary.

The original sliced model with exact teacher-up4 injection recovers waveform MAE4.80e-9, all96-source active correlation essentially1 and zero quiet/near failures. Teacher full-group-end injection into the common frozen suffix is bitwise perfect on every scored sample. These controls verify the external boundary and suffix implementation independently of the adapted internal coordinates.

## Execution and arithmetic checks

- Fixed72 calibration sources, disjoint96 development sources. No tuning on development data.
- Actual native stride2/kernel4, both phases, current and previous input frames, one shared output bias. Weights were fit by centered float64 ridge with fixed factor1e-6 and0..4 valid-waveform sample weights per12kHz cell.
- All three regressions have full512 design rank. Ridge condition:47,276 for sliced inputs,165,389 for trained inputs,78,141 for retained teacher inputs. Normal-equation residuals are at most5.70e-15.
- Actual WN-folded operator versus phase-design parity passed the existing atol1e-5/rtol1e-4 check. Max differences were1.31e-6 for sliced and3.87e-7 for trained; these are numerical agreement, not bitwise equality.
- The saved trained4625 baseline passed the pre-existing absolute1e-6/relative1e-5 metric check.
- All1,296 recorded pristine teacher/cache comparisons are bitwise equal:1,224 retained main-run comparisons plus72 from the receipt recovery. These repeat the fixed72 fitting and96 development sources across variants; they are not1,296 unique sources. All scoped interventions restored original group tensors exactly.
- Ten focused tests passed locally and on Runpod. The main diagnostic took46.60 seconds after loading. The receipt-only recovery below took2.90 seconds after loading.

## Reporting correction

The teacher-retained-input solver receipt and its fitting-error score accidentally used the same filename. The score overwrote the solver JSON; all measured model results remained intact. The parent approved one identical72-source teacher-input pass to recover the missing solver metadata. It used a new script and new filename, preserved every earlier output, and performed no additional development evaluation or neural training. The recovered receipt is `results/teacher-retained-input-solver-recovered.json`, SHA256 `5663a5dad06d2ad69999faf9ab783584c7380fee3a36e170688756a907449b2b`.

The small waveform arrays in `results/waveform-snippets.npz` include teacher and interventions for the fixed Spanish, Kannada, Kashmiri and whistling sources. `waveform-snippets.json` records exact48kHz source/crop offsets. Large fitted tensors remain on Runpod. Original results and source hashes are recorded in `execution-receipt.json`.

No CPU runtime benchmark was performed. Native operation geometry remains unchanged, but this is not a measured latency claim. The diagnostics neither authorize promotion nor prove that the whole compressed model cannot match the teacher after a better matched training process.
