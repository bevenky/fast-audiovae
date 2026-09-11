# Quiet-preserving recovery trial

Authorized independent2,000-update trial. Source and method reviews passed;48 local tests and48 qualified Runpod CPU tests passed. Model execution is pending the sequential queue.

- Code archive SHA256: `cb2b9dd4a128ea36ce680df66b99ed06be832bd065617a771a40ad3b23275a74`;33,516bytes, code/tests/hashes only.
- Remote CPU log SHA256: `03c7cf21c5aff88e1e5ecc571d9311395f03a196c623bbe1ee898b92e1ef59fa`.
- Source root: `/workspace/fast-audiovae-quiet-recovery-20260911-v1/code`.
- Reserved output root: `/dev/shm/fast-audiovae-quiet-recovery-20260911-v1`. Existing disk volumes lack space for all new optimizer checkpoints; this uses available Runpod shared memory, without deleting old artifacts. These new files require the pod to remain running.
- Fresh original teacher factory → sealed B native operators → sealed combined startup operator. No trained group/checkpoint weights are installed; original step0 supplies only original RNG/empty Adam recipe and provenance.
- The starting complete decoder must equal combined C's untrained state `56687c88adf30f4e770537ae7cd13ff7424bacbcc65c6bbabda4ace28c3610f4`.
- Comparator is completed C at matched0/250/500/1000/1500/2000, identical first24,000 distinct sources and ordinary objective.
- The only training change projects the actual Adam displacement onto current-batch quiet constraints, then checks nonlinear behavior and backtracks if required. All90 group tensors participate.
- Adam advances once per unique12-source batch. Actual nonzero weight updates, zero displacements, intervention counts, ordinary update time and auxiliary time are reported separately.
- Startup/other near silence/ordinary quiet stay separately measured. Current-batch protection cannot guarantee preservation on absent startup cases or all individual windows below a failing cohort maximum.
- No new inference operations, loss coefficients, source repeats within a run, automatic extension, checkpoint promotion or next cut.

[Full fixed protocol](../follow-on-experiments-plan.md). No GPU result is claimed yet.

Operational update: follow the [active corrected queue](../independent-trials-v2/execution.md). GRAIL initialization output moved to `/dev/shm/fast-audiovae-grail-hidden-20260911-v1/results`; source code and method are unchanged.
# Failure update, 11 September 2026

The independent arm launched after GRAIL completed but stopped before its first completed update. Initialization parity passed; the new constraint path raised `Grad-enabled constraint differs from its no-grad pre-state`. All artifacts are preserved and the finite controller and follow-up are stopped. [Failure review](failure-review.md). No retry or code/threshold change was made.
