# Gradual channel pruning: first cut

The first cut completed 1,000 updates on Runpod in 669.9 seconds including its final review, and stopped awaiting review. The previous joint-recovery run ended at step 6,625 and is retained as a historical comparison. This is a fresh student initialized from the original AudioVAE2 weights.

## Approved experiment

Narrow one internal channel boundary at a time, recover jointly, and review before another cut. Keep all nine residual units in stages 2 through 4, along with the original activations, temporal dilations, causal padding, upsampling, conditioning, frozen prefix, and frozen waveform suffix.

| Point | Stage 2 width | Stage 3 width | Status |
|---|---:|---:|---|
| Original teacher copy | 512 | 256 | Numerical control passed |
| First cut | 384 | 256 | Completed 1,000 updates; retained for further recovery |
| Second cut | 384 | 192 | Requires first-cut review |
| Third cut | 256 | 192 | Requires review and additional disjoint source preparation |
| Fourth cut | 256 | 128 | Requires review and additional disjoint source preparation |

The nested channel selections reproduce the previous 256/128 endpoint exactly. Later cuts inherit the recovered effective weights from the prior student, while the original teacher stays fixed. AdamW starts with fresh moments at each width cut because weight-normalized parameters change coordinates after slicing. Within a cut, optimizer moments are continuous.

All 90 stage 2 through 4 parameter tensors train jointly. The encoder, teacher, prefix and suffix stay frozen. Gradients pass through the frozen suffix into the trainable group. The losses remain raw sample-pooled waveform L1, multiscale mel, and complete group-output MSE, with coefficients 1, 0.0006674012905982311, and 0.009304078923434964 respectively. AdamW uses learning rate 0.00003, betas 0.9/0.99, epsilon 1e-8 and no weight decay. Physical batch size is one, with 12 distinct recordings accumulated per update. The qualified runtime is PyTorch 2.14.0+cu126, cuDNN 92501, FP32 with TF32 disabled.

## Validation and data

All 57 focused model, trainer and monitor tests passed locally and on Runpod. All 72 original-teacher calibration comparisons were bitwise equal to the saved targets. The independent full-width decoder-copy control passed its 12 cases. The actual first pruned model was measured before its first optimizer update.

The first cut uses 12,000 distinct source recordings already on Runpod. Source IDs, audio hashes and parent recording IDs were checked for uniqueness. The mixture includes all 22 scheduled Indian language codes, English, Latin American Spanish and Portuguese, Chinese, Japanese, French, Arabic, other FLEURS languages and expressive datasets. Presence does not mean equal representation: this first cut contains only 12 Japanese-tagged sources. Expressive coverage is supported by dataset provenance; the preparation report does not establish duration of each expressive event.

The existing immutable source stream contains 30,000 recordings, sufficient for the first two 12,000-source cuts. Further cuts fail closed until enough additional disjoint audio is prepared. Earlier-model audio may be reused by this new student, but a recording is not repeated within this progressive student.

The same 96-recording validation panel is measured at updates 0, 500 and 1,000. Its seven quiet groups remain separate: all quiet windows, near-silence, startup first 20 ms, the teacher's 20 to 40 ms transient on source silence, sustained source silence after 40 ms, interior near-silence after 800 ms, and quiet nonzero reference. These subsets overlap. Reports retain continuous residual and output levels, threshold excess and pass counts. No silence-specific loss or inference gate was added.

The first cut's untrained baseline has active waveform cosine 0.7435066, waveform MAE 0.02187068, and 0 of 2,544 quiet windows passing. These are measurements immediately after pruning, not a trained result. Percentage error reductions use this actual cut baseline. The exact-copy control is kept separate. A 99% waveform-correlation milestone is not 99% perceptual accuracy, and neither 99.99% recovery nor a 10x runtime improvement is promised.

## Running files and dashboard

- Remote root: `/tmp/fast-audiovae-progressive-pruning-v1`
- Trainer output: `cut1-384-256`
- Launch receipt: `process-launch.json`
- Trainer log: `training.log`
- Source: `code/progressive_train.py`, SHA256 `159571f9eba7b283e9360fb3882c64dbab7fe48b80b9bc7a773310a41871ea25`
- Original unchanged dependencies: `/workspace/fast-audiovae-compression-20260910-v1/code`
- TensorBoard events: `tensorboard/cut1`
- Previous run: `/tmp/fast-audiovae-joint-recovery-v2`

[Open the new TensorBoard overview](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#custom_scalars&_smoothingWeight=0).

One overview contains 13 colored metric runs. Details contains raw training losses each update and all seven silence groups. Quality changes only at 0, 500 and 1,000; the progress line updates each step. Use the clean URL above so an old saved run filter does not hide the new curves. Historical event files and checkpoints are preserved. Browser verification found that the default "Ignore outliers in chart scaling" hid sparse quality measurements among frequent progress points. That option was disabled and the chart was fitted to all data; the full correlation target and the 500-update points are now visibly present.

The runner saves the pre-cut state and checkpoints at 0, 500 and 1,000. It exits awaiting review after 1,000. The follow-up monitor reviews 500 and 1,000, then pauses. It does not launch another cut, extend the old run, promote the model, or commit changes automatically.

## Completed first-cut review

The saved checkpoint SHA256 is `4272198a8d76168564651c49960ef370690859981a5591ba0de39d42053eef56`. All 12,000 teacher/cache comparisons passed. The source ledger contains exactly 12,000 distinct recordings and matches the launch and checkpoint. All 90 AdamW parameter states reached update 1,000; weights, moments and losses are finite. The runner preserved frozen state and original files, and an independent audit rehashed 13 source and manifest files. All 13 TensorBoard metric runs expose their final step 1,000 values.

| Measurement | Update 500 | Update 1,000 |
|---|---:|---:|
| Active waveform cosine | 98.55% | 98.87% |
| Waveform MAE | 0.003465 | 0.003286 |
| Mel error | 0.242775 | 0.201765 |
| Quiet windows passing | 272 / 2,544 | 358 / 2,544 |
| Sustained source silence passing | 164 / 164 | 164 / 164 |
| Interior near-silence passing | 49 / 50 | 49 / 50 |
| Startup near-silence passing | 0 / 13 | 0 / 13 |
| Pooled active RMS relative to teacher | 97.50% | 93.65% |
| Full-scale overshoot samples | 0 | 0 |

Overall waveform, mel and group errors improve, but 29 of 96 individual recordings have higher waveform MAE between updates 500 and 1,000. All 96 improve in mel and group error. For the held-out whistle `freesound:428921`, active RMS falls from 108.78% to 81.25% of the teacher and waveform error rises 58.90%. RMS ratios are computed from stored energy on teacher-active windows; they are not perceptual loudness scores. Startup and quiet nonzero reference still need recovery. Sustained and interior silence continue passing most or all checks, although their residual RMS rises as output becomes lower.

Recommendation: retain the completed first cut and recover further at 384/256 before narrowing another boundary. Do not treat the correlation improvement or all-pass sustained-silence result as complete reconstruction recovery. A further segment needs a reviewed continuation that preserves this checkpoint's optimizer, RNG and source cursor; it was not launched by the completion monitor. Full analysis is in `evidence-final/report.md`, `quality-review.json` and `completion-integrity-audit.json`.
