# Active finite experiment sequence

The corrected queue uses the exact same validated controller and experiment source. Its only changes are the predecessor process and output locations. It waits for the fresh downstream-selection repeat, then runs GRAIL initialization, GRAIL recovery2000 and quiet-preserving recovery2000. All experimental outputs, logs and event files now use capacity-checked Runpod shared memory.

The preceding downstream run performed2000updates and all quality reviews, but synthetic pytest fixtures consumed `/tmp` headroom before its final checkpoint. The endpoint weights were not saved. Its failed receipt, earlier checkpoints, measurements and failed queue remain unchanged. See [failure record](../downstream-selection-v1/storage-failure.md). This is not a model-quality or numerical failure.

Active roots:

| Item | Runpod location | Status at launch |
|---|---|---|
| Downstream+B fresh repeat | `/dev/shm/fast-audiovae-downstream-recovery-20260911-v2/results` | Running, PID1100124; initial state/quality/RNG/Adam parity passed. First19update losses and source order exactly matched the preserved first run. |
| GRAIL initializer | `/dev/shm/fast-audiovae-grail-hidden-20260911-v1/results` | Queued, no model execution yet. |
| GRAIL2000 recovery | `/dev/shm/fast-audiovae-grail-recovery-20260911-v1/results` | Queued after initializer. |
| Quiet-preserving2000 recovery | `/dev/shm/fast-audiovae-quiet-recovery-20260911-v1/results` | Queued after GRAIL. |
| Finite controller | `/dev/shm/fast-audiovae-independent-trials-20260911-v2` | Active. Aggregate status is `status.json`. |

- Configuration SHA256 `9fec3a65dd665962fdb33303bc277ef376895aa7d4efa37b603d0ddf787d92f9`. [Exact configuration](config.json).
- Driver remains `/workspace/fast-audiovae-independent-trials-20260911-v1/independent_trial_queue.py`, SHA256 `e55b6f584daec80de5cb64665e49d75be5ef867eabad5656962f47809b60ce9e`.75local/remote controller and aggregate-audit tests already passed. No repeated tests or synthetic fixture writes are needed.
- GRAIL source remains `/workspace/fast-audiovae-grail-hidden-20260911-v1/code`;34local/remote tests passed. Quiet source remains `/workspace/fast-audiovae-quiet-recovery-20260911-v1/code`;48local/remote tests passed.
- The original teacher-derived initializers, original RNG, fresh Adam,24,000distinct training sources, fixed96development panel and exact2000update budgets remain unchanged. No old trained student checkpoint is installed.
- At retry start2,069,962,752bytes of shared memory were free, against a1GiBoperating allowance for all three runs and their retained checkpoints/logs. Keep the pod running to retain shared-memory data. Qualification is complete; no other temporary test producers are launched.
- TensorBoard port8888 serves the active retry; PID1100174 at handoff. The controller switches to subsequent trials only after initial parity and a real update. Previous event files remain untouched.

For completion review, use `/workspace/fast-audiovae-independent-trials-20260911-v1/independent_trial_aggregate.py`, SHA256 `a4d27c0fbf155372db1b2f8a3777b81b4bbddb7c157647db1ed28756f1fa556a`, with the qualified `/tmp/fast-audiovae-recovery-20260909/venv214/bin/python`. Run CPU-only with `--run RESULTS --reference REFERENCE`. D/G reference `/tmp/fast-audiovae-progressive-pruning-v1/reconstruction-b-recovery-v1/results`; Q reference `/workspace/fast-audiovae-combined-recovery-20260911-v1/results` (matched to its2000prefix). Set PYTHONPATH from the queue's G recovery environment so the existing pure metric-summary modules are available; keep CUDA_VISIBLE_DEVICES empty for this saved-result audit. It performs no inference, optimization or audio/latent loading.

Return only filtered aggregate JSON, never raw training logs, source manifests, per-recording values, audio, model weights or latents. Independently verify each checkpoint and source ledger, preserve failed evidence, and keep startup/other-near/ordinary-quiet measurements separate. No automatic continuation, next cut, model promotion, benchmark, commit or push.

The existing AudioVAE training-review follow-up is active for this finite sequence. It stays quiet on unchanged progress, performs aggregate-only saved-checkpoint audits at completion, and pauses on technical failure or after the final comparison. It cannot extend training, launch duplicate jobs or change methods.

## Completion review, 11 September 2026

Downstream selection + B completed2,000updates and saved its final checkpoint. The CPU-only saved-result audit passed. All24,000sources were unique and matched B's ordered prefix. The fresh repeat exactly reproduced all2,000loss records and all six saved reviews from the preserved storage-failed attempt. [Completed findings](../downstream-selection-v1/completed-findings.md).

The downstream result is mixed: MAE2.26%better than B, but mel0.62%worse and quiet passes1,158versus1,242of2,544. Near-silence improved, while startup remains0/13and the20–40ms teacher transient worsened. No promotion or extension was made.

GRAIL initialization has completed and its independent recovery is running. Quiet-preserving recovery remains queued after GRAIL. Controller, child, configuration and source identities passed the live check; no dashboard warnings were present. The launch-status table above is retained as historical launch evidence.

Next completion review: GRAIL recovery has now completed 2,000 updates and passed the same CPU-only saved-checkpoint audit. Waveform MAE is 5.29% lower than B and quiet passes are 1,419/2,544; startup remains 0/13 despite lower continuous error. The teacher's 20–40 ms transient is still worse than B. [G findings](../grail-hidden-v1/completed-findings.md). The controller has launched the independent quiet-preservation arm. Shared-memory free space was 1,646,473,216 bytes at handoff, with no controller or dashboard failures. The follow-up remains active for Q completion or a technical failure.
# Current state: paused after Q technical failure

D and G both completed 2,000 updates and passed their saved-checkpoint audits. Q passed initial parity but stopped before its first completed update with `Grad-enabled constraint differs from its no-grad pre-state`. The controller has stopped and the existing follow-up is paused for review. [Q failure review](../quiet-preservation-v1/failure-review.md). All results and checkpoints remain preserved; no retry, source change or threshold change was made. The earlier running-state entries below are historical observations.
