# Conditional quiet phase engine

Implemented locally in the new `quiet_phase_engine.py` subclass, without changing the base engine, current comparison runner, model or inference path. Nothing from this component has been uploaded or trained.

## Starting a separate paired trial

Construct `QuietPhaseDistillationEngine` with the exact parent model, training, reconstruction-loss, discriminator and balancer configuration. Its `phase_config` defaults to `QuietPhaseEngineConfig(gradient_share=0.0)`. If the selected parent has a version-2 capped balancer, pass the matching `MelGradientCapConfig` as `mel_cap`; do not remove that configuration.

1. Call `load_state_dict(parent_engine_state)`. The subclass deep-copies the incoming payload to prevent CPU optimizer states from aliasing and changing the parent.
2. Call `fork_reconstruction(fork_config)`. This requires unchanged exact loaded state and already frozen normalization. Use `freeze_normalization_step=0`, a fixed reconstruction objective and `quiet_gradient_share=0`.
3. Keep the control phase-disabled. For the candidate, call:

```python
activation = engine.enable_phase(
    gradient_share=0.02,
    policy={
        "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "teacher_state_unchanged": True,
        "parent_checkpoint_sha256": verified_parent_file_sha,
        "evidence_sha256": verified_phase_diagnostic_file_sha,
    },
)
```

The 0.02 share is an example for the proposed first bounded comparison, not a validated optimal value. The configured maximum is 0.05. Positive configuration alone cannot start training: activation requires the exact-load/fork transition and teacher-identified evidence. The engine records parent, model, fixed-statistic and fork fingerprints itself.

The policy refers to caller-verified files. The caller remains responsible for checking the frozen teacher's actual state and continuous target-cache identities. This engine owns no teacher model and cannot independently verify its weights. Targets and latents remain detached during updates.

## Gradient behavior

The phase loss uses the standalone complete-window, teacher-conditioned 480-sample residual-template criterion. It measures the current mean per-example output-gradient norm, scales the phase gradient to at most its declared pre-sum share, and transfers that budget from the current reconstruction gradient. It preserves the exact reconstruction gradient when the phase term is ineligible, has zero gradient, or the base gradient has no available norm budget. The coefficient cap is inherited from the existing balancer and explicitly reported.

Logged scalar metrics include raw/scaled phase norm, coefficient, target and achieved pre-sum share, base scaling, final combined norm, zero/bound conditions, complete/quiet/eligible windows, active examples, min/max eligible windows per example and mean active residual-template RMS. This share is not a percentage of parameter updates or final audio quality. The standalone `quiet_phase_metrics` API retains full per-example evidence when required.

The candidate rejects mixing the general quiet loss, changing its reconstruction objective, or activating GAN training. It adds no output filtering, template subtraction, muting or inference layer.

## Checkpoints and verification

`state_dict()` adds a separate `quiet_phase` extension. Plain parent states load only with phase disabled. Candidate resumes require the identical `QuietPhaseEngineConfig`, teacher policy, fixed statistics, parent/fork evidence and base configuration. Construct a resume engine using `QuietPhaseEngineConfig(**state["quiet_phase"]["config"])`, restore any version-2 mel cap, then load the state exactly. Enabling again on resume is not required or allowed.

Thirteen focused CPU integration tests pass, alongside the twenty standalone loss tests. They check exact disabled/base updates, parent payload preservation, teacher and latent gradient ownership, poisoned masked context/tails, bounded actual share, finite zero/ineligible behavior, no discriminator updates, exact candidate resume with a version-2 capped balancer, changed-policy/config/statistics rejection and the activation transition. These are correctness checks, not model quality or speed evidence.
