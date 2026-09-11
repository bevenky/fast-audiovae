# Exact-teacher control completed

The independently loaded full-width student reproduces the native original teacher exactly on all 60,000 source crops consumed by the current training run. All 1,153,650 quiet windows and 58,045 near-silence windows pass the unchanged checks. The same 96-recording evaluation has zero waveform, mel and group error both before and after the optimizer probe, including 13/13 startup windows passing.

All twelve fresh-AdamW updates on 144 distinct fitting sources have zero loss, zero gradients and unchanged model weights. The original teacher, source files, caches and final training checkpoint remain preserved. This clears the tested copied-model forward path and the bounded optimizer cases; it does not validate the pruned model, promise identical behavior on unseen inputs or establish a zero gradient on every consumed source.

The result is classified `cache_sensitive`, not an unconditional bitwise cache pass. One source at global index 2707 differs from the saved waveform cache by maximum 2.5332e-7 and RMS 3.6112e-8, within the original tolerance; its native teacher and student are identical and its quiet check passes. That source is outside the 144 optimizer-probe sources. One of 59 shapes also showed a small first-call difference that disappeared after warmup. Neither observation reproduced the pruned model's startup or quiet-window failures. Raw evidence and completed.json remain on Runpod.

The control confirms that the current 13 startup failures are not inevitable under the evaluator. Existing first-cut reports already show those failures before any recovery training. Their persistence after recovery requires investigation of the current width cut and its learned response, rather than assuming either an evaluator defect or an intrinsic capacity limit.

## Existing data and objective exposure

A read-only aggregation of all saved audit records counted 7,058,693,412 valid waveform samples. Quiet windows contain 15.674% of those samples. There are 29,196 crops beginning at original-source time zero, and 30,804 with preceding context. The first 20 ms of all true-start crops accounts for 0.3971% of valid waveform samples. This is all amplitude levels, not a measured silent-startup fraction or gradient share.

The current waveform objective averages absolute error over valid samples; group error is also sample weighted. Startup samples are included when their original-source start is scored. Therefore broad quiet audio and source starts were not omitted wholesale. The small temporal share of startup behavior may help explain why aggregate improvements can coexist with persistent onset error, but does not prove loss weighting is its root cause. A new weighting rule is not selected here.

The run is finished. Automatic follow-up remains paused. The user has separately authorized the focused current-cut diagnosis recorded in ../startup-quiet-mechanism-v1/plan.md. No further parity sweep or training is scheduled automatically.
