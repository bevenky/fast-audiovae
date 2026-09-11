# Joint generator-rate recovery screen

Run only after the encoder/cache reproduction result releases the exact cache and data identities through a separate resolution receipt. This harness does not decide that the historical preparation problem is resolved.

Both arms begin at the preserved targeted step 8,490 and run 400 updates with batch size 32. They use the same 12,800 targeted-generator windows and the same planned 190 ms discriminator views. The windows were previously used by this starting checkpoint; reuse is explicitly for debugging. Within each arm, scored intervals do not repeat or overlap. Causal context may overlap without being scored.

| Arm | Joint generator rate | Everything else |
| --- | ---: | --- |
| Current rate | 0.0002 | Existing recipe |
| Quarter rate | 0.00005 | Existing recipe |

Both generator optimizers receive the changed rate together. Initial model weights, optimizer moments, discriminator state, loss EMA and fixed normalization are identical. Moments, discriminator weights and EMA estimates evolve normally during each arm. There is no new warmup, calibration, loss, data selection, model layer or encoder update.

For the approved corrected-domain continuation, regenerate both latent means and decoder targets from authentic complete sources using the verified teacher backend. Supply them through `--training-receipt`. The full 12,800-window overlay must retain the exact original indexed source/crop schedule and reference waveforms; its new latent/target/cache identities are recorded separately. Both rate arms use that same fresh overlay. The original cache remains an immutable baseline, and its prior strict numerical-screen failures are not relabeled as passes.

## Execution and preservation

- Refuse to start without a ready cache-resolution receipt identifying the parent checkpoint, target cache and data plan hashes.
- Preserve original files and refuse automatic replay of any experiment with an existing identity file.
- Record implementation hashes, input identities, every training update and the exact discriminator views.
- Evaluate the complete sealed 285-crop panel before and after each arm using the original scoring masks. Add peak-excess energy during the same evaluator forwards and derive encoded-zero steady-state RMS from seconds 2 through 6.
- For this corrected-encoder trial, also provide the separately sealed canonical receipt. Evaluate its 285 matched crops before and after each arm, with its own teacher-defined quiet masks and counts. Keep `before.json` / `after.json` historical; write `before-canonical.json` / `after-canonical.json` separately. Both domains must pass the same unchanged pilot gates. Never combine their samples or replace historical targets.
- Check a fixed small health panel at updates 0, 25, 100 and 400. Nonfinite outputs or a catastrophic amplitude above 4 stop the arm. These are failure guards, not quality acceptance.
- Save atomic checkpoints every 100 updates. Use `/tmp/fast-audiovae-recovery-20260909/update-results`, where the current machine has more space; preserve a 512 MiB reserve beyond the pending checkpoint allocation. This temporary mount is recorded and requires later durable transfer.
- Verify CPU streaming sample counts and batch/stream numerical agreement at 80 and 160 ms. This is correctness, not a new RTF benchmark.
- Preserve both final checkpoints. Report gates without automatically promoting either candidate.

## Preregistered pilot gates

The executable definitions are imported from `evaluation_audit.GATES` and hashed into the experiment identity before the first update. They are engineering screening margins, not perceptual-equivalence claims.

- Natural pooled quiet RMS: at least 10% below the matched current-rate control and strictly below the starting checkpoint.
- Encoded-zero steady RMS: strictly below both control and starting checkpoint.
- Natural raw MAE and unchanged mel error: no more than 1% above either control or starting checkpoint.
- Speech and expressive cohort raw MAE/mel: no more than 2% above the starting checkpoint.
- Language and event groups with at least two sources: no more than 5% raw MAE/mel regression from the starting checkpoint. All smaller groups and per-source changes remain visible for review.
- Natural maximum peak: no more than 1% above baseline. Peak-excess energy: no more than 10% above baseline. This stage does not claim to solve the remaining amplitude-bound problem.
- Missing metrics, undefined required ratios, changed target/mask contracts or failed correctness checks cannot pass.

Passing identifies a candidate for further continuation, not a release. Final 0.99 correlation, quiet fidelity, amplitude fidelity, expressive coverage and listening remain separate requirements.

## Required resolution receipt

The controlling process supplies a JSON object with:

```json
{
  "format_version": 1,
  "training_ready": false,
  "parent_checkpoint_sha256": "exact targeted final checkpoint hash",
  "target_cache_sha256": "exact verified cache hash",
  "data_plan_sha256": "exact sealed data-plan file hash",
  "requires_canonical_evaluation": true,
  "requires_fresh_training_pairs": true,
  "training_receipt_sha256": "fresh training receipt file hash",
  "training_target_contract_sha256": "fresh training contract identity hash",
  "training_target_cache_sha256": "fresh training pairs file hash",
  "canonical_receipt_sha256": "exact canonical receipt file hash",
  "canonical_target_contract_sha256": "canonical receipt contract identity hash",
  "canonical_target_cache_sha256": "exact canonical heldout cache hash",
  "resolution": "What was established and which baseline family is released",
  "evidence": ["path to the resolution evidence"]
}
```

The default context loader and evaluator bind the current sealed historical cache and panel. The canonical receipt is a separate versioned contract for evaluating corrected-encoder inputs. Its cache and metadata are checked before any updates, its source/crop geometry must match the historical panel, and its receipt/cache hashes are checked again at completion. The controlling resolution must set `requires_canonical_evaluation` to `true` when it releases a corrected-domain trial. Any canonical hash binding also makes that panel mandatory. Free-text resolution notes do not replace this machine-enforced field. A supplied canonical panel must pass even if its requirement flag was omitted.

Entry point: `run_update_experiment.py --cache-resolution RECEIPT --training-receipt FRESH_TRAINING_RECEIPT --canonical-receipt CANONICAL_RECEIPT --output-dir OUTPUT`. The runtime must include the existing training package, frozen diagnostic helpers, recovery evaluator, canonical-target helper and corrected-training overlay directories on `PYTHONPATH`. The script acquires the paused parent runner's lock and its own lock before loading data. `comparison-historical.json` and `comparison-canonical.json` retain separate gate results; `comparison.json` passes only if both pass. No candidate is automatically promoted.
