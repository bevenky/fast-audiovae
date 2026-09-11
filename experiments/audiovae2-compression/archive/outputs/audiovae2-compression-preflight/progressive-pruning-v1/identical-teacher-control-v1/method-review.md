# Exact-teacher control: independent method review

This is a read-only source and saved-evidence audit. It adds no model results. The control must wait for verified completion of the current 5,000-update run.

Source references below use `work/fast-audiovae/experiments/audiovae2-compression/` for `run_pilot.py` and `group_model.py`, and `work/fast-audiovae/experiments/convnext/audiovae_student/` for `quiet_audio.py` and `reconstruction_v2.py`. Cohort observation and source alignment are implemented in `joint_recovery_v2.py:162–185`, `joint_recovery_gates_v2.py:25–72`, and `audit_quiet_windows_v1.py:24–37,55–88` in the compression directory.

The existing `../evidence/cut1-384-256/full-width-copy.json` contains 12 singleton cases. Every copy waveform, group boundary, cached target and singleton repeat error is exactly zero. However, `run_pilot.py:260–278` runs this preflight under `no_grad`, using the twelve-source boundary panel. It did not evaluate an unpruned copy through the complete current 96-source panel and seven quiet cohorts. The additional control is therefore useful.

## Expected identity behavior

`group_model.py:97–107` clones original decoder state independently. With all original stage-2 and stage-3 channels selected, `build_student` skips narrowing and retains the cloned raw weight-normalization parameters. The control must start there, rather than widening a trained narrow model. Compare state keys, shapes, exact values and independent storage before inference.

`quiet_audio.py:46–79` excludes context and padding before arithmetic and divides each window norm by the square root of its actual valid count. `quiet_audio.py:109–112` requires residual RMS at most `max(sqrt(.02)*teacher_rms, 1e-5)` and output RMS at most `max(10**(.05)*teacher_rms, 1e-5)`. If prediction equals teacher exactly, both tests necessarily pass. This includes a nonzero teacher floor and teacher startup transients. Exact original 16 kHz source silence does not imply that its teacher reconstruction is zero.

Seven cohorts overlap. They must retain their recorded individual denominators, not be added as a partition. In the current panel, exact identity should pass all 2,544 quiet windows and all 184 near-silence windows, including every window in the startup, transient, sustained and interior subsets. No active samples can legitimately produce an undefined cosine; otherwise self-cosine is approximately one, subject to floating-point reduction. Peak and output RMS should match the teacher, not become zero.

## Separate the possible error paths

1. **Metric identity:** score cached target against itself with the same masks and quiet observer. Independently recompute a few window RMS values and margins, including partial tails. This does not depend on a neural implementation.
2. **Network identity:** compare the published native decoder `forward`, the helper `teacher_trace`, and the independent full-width student's full and split paths. A common helper alone cannot exclude a helper error. Use the unchanged FP32, singleton, sample-rate-conditioned, causal execution and shape warmups.
3. **Cache consistency:** compare the live teacher to the immutable cached full-source target on the scored span separately from excluded context and tail. `run_pilot.py:100–125` removes `context_frames*1920` samples and scores only `valid_scored_samples`. Sufficient causal history establishes geometric coverage, not numerical equality between different full-source and crop execution shapes.
4. **Gradient-mode consistency:** before any update, compare grad-enabled and no-grad outputs of the same full-width copy on identical inputs. Existing preflight and `singleton_forward_check` both use no-grad. Their exact results and the new CPU tests do not establish H100 grad-mode parity.

For every comparison, report bitwise equality plus maximum absolute error, RMS and quiet-window margins. The existing `atol=1e-5, rtol=1e-4` compatibility check is not an identity test: its absolute tolerance equals the quiet residual floor. Do not loosen the quiet limits to make the control pass.

## Loss and optimizer interpretation

The current objective is sample-pooled waveform L1, magnitude mel reconstruction and full-group MSE (`run_pilot.py:175–195, 522–545`; `reconstruction_v2.py:81–133`). At identical waveform and group targets, these losses and their student gradients should be finite zero. No GAN or KL objective is active in this compression recovery path.

If a frozen cached target differs even slightly from the live grad-enabled teacher copy, nonzero gradients are legitimate. The L1 derivative depends on the residual sign, not its magnitude. On fresh AdamW's first step with zero weight decay, the per-parameter displacement is approximately `-lr*g/(abs(g)+eps)`. Consequently, tiny waveform discrepancies need not imply tiny parameter movement. Conversely, old nonzero Adam moments can move parameters even when the current gradient is zero. Use fresh optimizer state and distinguish cache or execution sensitivity from an evaluator defect; stop to interpret the pre-update evidence before attributing later drift.

An identity pass would establish that this pipeline recognizes teacher equality. It would not prove that the provisional quiet thresholds are perceptually calibrated or that channel pruning can reach identity with further training. Keep this zero-error control separate from percentage-reduction charts whose baseline would otherwise divide by zero.

## Additive runner review

The reviewed `identical_teacher_control.py` now checks completion and GPU idleness before loading models, runs the cached-self baseline on the scoring GPU, authenticates its quiet-window identity against the saved 5,000-step panel, and scores the native teacher and independent copy through the unchanged full evaluator. The no-update grad-mode probe precedes the complete consumed-source audit. Native-path mismatch and cache differences outside the original tolerance prevent actual updates; within-tolerance cache rounding remains a separate sensitivity outcome. The disposable Adam test stops at its first nonzero loss, gradient or weight change and then records the full panel again. This does not alter the preserved student or teacher.

No blocking source issue remained in the final reviewed source, SHA-256 `6c2a9066d02a996ecab0a2ef167cbebf151470183d634a25e5dc557dd3b812b2`. Its existing three warmup forwards now retain the first and third no-grad and grad-enabled group/waveform outputs, without additional forwards. They report first-versus-third behavior and first-grad versus first/third no-grad behavior for each newly encountered shape. First outputs are detached and cloned before later calls. This records initial grad-enabled execution in this isolated control, as well as warmed parity; it does not recreate historical backend state or prove what occurred at the start of the earlier run. The final status and counters retain warmup or pre-update sensitivity even when later updates happen to be zero. `progressive_train.py:162–169` originally warmed only no-grad execution, so that historical distinction remains explicit.
