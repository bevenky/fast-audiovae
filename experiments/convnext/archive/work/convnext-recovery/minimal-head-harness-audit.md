# Bounded architecture experiment harness audit

This is a read-only audit of the existing harnesses, not a new run. The retained reference is the quarter-rate control at step 8890. No production checkpoint, data cache or optimizer policy was changed.

## Restore and isolate each arm

1. Load the runtime 2.14 diagnostic context through `diagnostic_common.load_context()`. Preserve the deterministic runtime policy and one CPU thread.
2. Load the quarter checkpoint with `torch.load(..., map_location="cpu", weights_only=True, mmap=True)`, verify its pinned SHA, create `ctx.engine("targeted", device="cuda")`, and call `native_chain.restore_quarter(engine, payload["engine"])`. This verifies the complete engine identity, not just model weights.
3. Restore `payload["rng"]` through `_restore_rng` before each arm. Restore the saved crop RNG immediately before training, after any initialization or evaluation. Bind the identical planned discriminator positions with `bind_views(engine, ctx.data["pools"]["targeted_generator"])`.
4. Apply only the declared architecture migration on the disposable engine. Preserve every original parameter name, tensor and optimizer state, all normalization buffers, discriminator state, loss-balancer state, training clocks and calibration. New parameters need separately declared empty optimizer state.
5. Train each arm on the same immutable, ordered, source-disjoint subset of the corrected teacher-pair cache. All arms can share examples for a matched debug comparison; do not repeat or overlap scored intervals within an arm. Use the unchanged no-architecture continuation as the primary control.

`build_fusion_engine` cannot directly restore the quarter checkpoint: it accepts only original `recipe_v2` parents, while the quarter checkpoint is `fusion_recipe_v1`. `restore_quarter` is the existing exact experimental resume helper.

For a parameter-free tanh fork, `FusionStudentDecoder.from_decoder(engine.model, FusionArchitectureConfig(terminal_tanh=True))` and `_rebind_student_optimizer` preserve the old parameter groups and their state. Record the changed architecture explicitly in `fusion_migration`. Do not masquerade as an unchanged control checkpoint.

## Data and evaluation contracts

Use `load_training_overlay(ctx, receipt, required_counts={"targeted_generator": 12800})` and `load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, receipt)`. Those validate corrected whole-source encoder means, post-tanh teacher targets, tensor hashes, sample counts and ordered crop geometry. No new teacher generation is necessary.

The canonical panel has 285 crops, 147 sources and 26,206,830 scored samples, including 282 natural crops from 144 sources. Interior crops exclude six samples consistently for all variants; the total exclusion is 618 samples. The complete encoded-zero fixture is six seconds, with the stationary score taken only over seconds two through six. Quiet membership comes from sealed teacher-only 20 ms window masks, never candidate amplitudes.

Use `build_canonical_evaluation(engine, crops, metadata, receipt)` unchanged before and after. It already collects common waveform, mel, pooled quiet residual, loud peaks, excess energy, saturation, speech/expressive/language/source breakdowns and every individual crop. Do not select candidate settings using this heldout panel. Evaluate final candidates against both the unchanged checkpoint and the matched continuation.

The existing `compare_canonical_reports` is not directly reusable: its underlying comparison hard-codes step 8490 to 8890. A new comparison must declare the new start/end steps and reuse the metric contracts without falsifying old step fields. Likewise, `run_update_experiment.py` is hard-coded to that previous experiment and should supply utilities, not be launched unchanged.

All reports must distinguish scored overshoot observations from deduplicated physical events. Existing crops can overlap. A bounded head's zero overshoot is a structural range guarantee, not quality equivalence. Report raw MAE, mel, transient peaks, saturation, quiet error and per-source regressions alongside it.

## Optimizer and training traps

- Restored quarter learning rate is 0.00005. Do not apply the old quarter-rate migration again accidentally.
- `RecipeV2Engine.train_step` resets every generator optimizer group's LR from `engine.learning_rate()` every step. Setting a group's LR alone will not survive. Any rate policy change must update recipe/config consistently and appear in every relevant arm.
- `train_step` requires a gradient for every student parameter. A head-only experiment cannot simply freeze the body; it requires an explicit trainable-parameter training-step implementation and optimizer partition. Native hybrid routing requires all twenty ConvNeXt matrices to remain trainable.
- Standard Muon routing rejects unexpected trainable module roots. The existing rebind helper only special-cases its seven-tap output filter. A new residual head needs explicit new-parameter routing and named state receipt.
- Keep discriminator view locations and RNG identical; discriminator weights should evolve separately within each arm, not be copied from a control after each step.
- Keep loss-balancer and normalization state initially identical. Earlier changed objectives inherited stale gradient statistics, making those objective comparisons confounded. A head-only function change is separate from an objective change.
- The recently qualified 1/16 parameter displacement was not a validated 16x LR reduction. Its four independent single-step results cannot be promoted to a long-run optimizer policy without a declared test.

## Avoid repeating old negative candidates

The prior 200-step tanh candidate, rescored on corrected canonical pairs, removed overshoot but increased natural MAE 0.49%, natural quiet RMS 1.66%, and compressed the maximum peak to 0.922 while teacher peaks reach about 0.995. Repeating plain tanh is justified only by an explicit improved adaptation hypothesis or as a necessary matched control.

The previous single-channel seven-tap causal output filter increased quiet RMS 10.52% and waveform MAE 0.14%. It is not equivalent to the teacher's learned 32-channel, seven-tap synthesis. Copying that scalar filter again does not test the diagnosed multichannel cancellation mechanism.

Natural quiet error is predominantly varying rather than a stationary phase template. Removing one template improved the earlier aggregate only 0.65%. Tying phases or filtering away quiet detail has no established benefit. The unchanged-architecture control is essential because finite update magnitude already explains measured quiet regressions.

The lowest inference-cost synthesis test is a calibrated change in the existing readout, which can fold into its existing weights. Training-only quiet anchors and representative speech guards are necessary; fitting only the heldout encoded-zero fixture would contaminate the test and would not establish natural-quiet improvement.

## CPU cost and state checks

Copy each final model to CPU; use one thread and identical batch/streaming inputs. Run matched warmups followed by interleaved baseline/candidate measurements, with output duration as the RTF denominator. Benchmark the full decoder as well as any added head operation. Do not label H100 diagnostic time a CPU RTF.

For existing stateless tanh, use `diagnose_streaming(..., frames_per_chunk=2 or 4)` and require exact sample count plus max absolute batch/streaming discrepancy no greater than 2e-6. A new stateful synthesis operation additionally needs empty-input, state-reset, final partial input, and boundary tests; its history must appear in the architecture identity. No additional lookahead or dropped samples is acceptable.

Use a fresh output directory and the existing shared runner lock. Keep originals immutable, write evidence with source/data/checkpoint hashes, and preserve the reference checkpoint before any bounded arm. Check available storage before saving full optimizer checkpoints: the workspace mount was nearly full in the prior run. Model-only experimental artifacts should be clearly labeled non-resumable if that is the deliberate storage policy.
