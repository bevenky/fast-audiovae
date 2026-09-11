# AudioVAE2 group compression plan

Status: implemented on `audiovae2-compression`. The real H100 numerical preflight passed on 168 calibration/development sources. CPU ONNX export and source streaming checks passed, including exact output counts, reset and interleaved streams. The measured block input matches the teacher exactly. A four-setting, 256-update comparison has completed; see [the settings comparison](audiovae2-settings-screen.md). No inference-speed improvement has been established. This branch starts from `main` at `7150ae9`; the previous ConvNeXt candidates and production runtime remain available.

The objective is to replace a complete teacher group with a smaller group that learns the same input-to-output mapping. For example, six jointly trained student layers can approximate the combined function of ten teacher layers. Their intermediate activations need not correspond. The external feature coordinates, timing and causal contract must agree; equality of the resulting function is a training objective, not a guarantee from deleting layers.

## The first replacement

Keep the original frozen encoder and decoder weights as the reference. Its raw posterior-mean latents remain 64 channels at 25 Hz. This experiment keeps AudioVAE2's existing 16 kHz input and 48 kHz output interface.

Treat decoder stages 2 through 4 as one group:

```text
                                      Complete group
Frozen stem + stage 1 ── x ──┬── original stages 2–4 ── h_teacher
                            └── smaller stages 2–4  ── h_student
                                                        │
                                   frozen stages 5–6 + waveform head
                                                        │
                                                   student audio
```

Both paths receive exactly the same `x`, including its causal history. Match `h_student` directly to `h_teacher`, then match the student audio to the full original teacher audio. Backpropagate through the frozen suffix into the smaller group; freezing suffix weights must not detach this path.

| Contract | Teacher and student |
|---|---|
| Group input | 1,024 channels at 200 Hz, before stage-2 sample-rate conditioning |
| Group output | 128 channels at 12,000 Hz, after the full stage-4 residual stack |
| Expansion | 60 output feature frames per group input frame |
| Full decoder output | 1,920 waveform samples per 40 ms latent frame |
| Boundary behavior | Causal, zero-filled startup, unchanged transposed-convolution phase and trimming |
| Outside the group | Original encoder, stem, stage 1, stages 5–6, final waveform convolution and tanh |

No learned adapter is needed at these shared boundaries. Internal widths can change because the whole group, rather than each internal stage, is the supervised replacement.

| Internal stage | Teacher channels | Smaller group channels | First candidate residual units |
|---|---|---|---|
| 2: 200 → 1,200 Hz | 1,024 → 512 | 1,024 → 256 | Keep all three |
| 3: 1,200 → 6,000 Hz | 512 → 256 | 256 → 128 | Keep all three |
| 4: 6,000 → 12,000 Hz | 256 → 128 | 128 → 128 | Keep all three |

Each retained unit keeps Snake, depthwise temporal convolution, pointwise projection and its residual skip. The first candidate preserves all dilations 1, 3 and 9 and therefore the teacher's temporal support. The terminal convolution still couples adjacent waveform samples, and tanh still bounds output. Those properties address specific weaknesses of the earlier direct-waveform student, but do not establish that compression preserves quiet audio or transient fidelity.

## Why start with width, then consider fewer layers

The source-level count is 8.991 GMAC per audio second. Upsampling matrices account for 56.12%, residual pointwise matrices 42.09%, and depthwise convolutions about 1.63%. These are arithmetic counts, excluding activation and memory cost. The three-stage group accounts for 71.27% of that total.

| Complete group replacement | Decoder GMAC/s | Reduction from teacher | Role |
|---|---:|---:|---|
| Original widths, nine residual units | 8.9912 | 0% | Exact-copy control |
| Original widths, six units | 8.0610 | 10.35% | Depth-only diagnostic if needed |
| Narrower internals, nine units | 5.1741 | 42.45% | First recovery candidate |
| Narrower internals, six units | 4.7823 | 46.81% | Conditional next candidate |

Width reduction supplies most of the arithmetic opportunity. Deleting three more units saves another 7.57% of the width-only candidate's MACs but reduces the group's maximum past support from 19 to 15 input frames. Across the whole decoder, maximum support changes from 20 to 19 past latent frames. Retaining more input in a buffer cannot restore a dependency the smaller architecture cannot represent.

Start with the nine-unit narrower group to preserve that capacity. If it recovers quality but remains too slow, jointly distill its six-unit replacement. Initialize that later candidate from the accepted narrower model while retaining the original AudioVAE2 as the teacher. Additional Snake and state savings may matter beyond the MAC count and must be measured.

## Where the same logic applies

| Replacement boundary | What may change inside | Initial decision |
|---|---|---|
| Complete residual stack at any stage | Three units become two or one, all jointly adapted | Useful isolation diagnostic; too little saving alone |
| Several adjacent upsampling stages | Internal widths and residual depth, with unchanged outer shape and stride product | Primary stages 2–4 candidate |
| One residual branch with a full-width skip | Channelwise branch narrows, projection returns to original channels | Possible fallback if narrowing the larger group fails |
| One dense matrix | Two lower-rank factors with no intervening activation | Later option; extra calls and buffers can offset savings |
| Stem or high-rate output tail | A complete replacement preserving its outer contract | Retain initially; less arithmetic leverage and direct sensitivity to latent/output behavior |

Simply removing an upsampler changes the output sample rate. Merging upsamplers while preserving their stride product would be a separate synthesis-module design. Matching every retained layer is unnecessary, and may prevent the smaller group from redistributing the teacher's computation.

## Implementation and fitting sequence

1. **Authenticate the control.** Load the pinned original weights independently as teacher and copy, verify every key and shape, freeze the teacher and encoder, and compare every group boundary and final waveform. Verify continuous and streaming equivalence before compression. Preserve the currently verified teacher backend and full-source target-generation policy. Batching is enabled only after parity against sealed singleton references, including internal boundary tensors.
2. **Prepare the narrower group.** Add an explicit stage-width configuration instead of assuming that every stage halves channels. Couple each channel selection across adjacent transposed convolutions, residual skips, depthwise groups, Snake parameters and sample-rate conditioning. Materialize effective weight-normalized weights before slicing, then reconstruct the new normalization parameters. Copy all surviving weights; train all parameters inside the replacement together.
3. **Freeze initialization and data.** Select channel coordinates using a training-only calibration panel with speech, quiet windows and expressive events. A deterministic rank-revealing subset of uncentered teacher activations is the proposed initialization, with selected indices recorded. It is not a proof of importance or preserved cancellation. Do not rank removals using the gradient of a zero-error teacher-copy loss. Do not choose channels or coefficients using final holdout scores.
4. **Check deployment feasibility early.** Export the unchanged control and the narrower shape configuration. Ensure actual dimensions reach the existing packed matrix and streaming kernels; verify no dense zero-filled replacement or Python fallback hides the savings. The first candidate keeps three residual units, fitting the existing fused-stack structure more closely. A later two-unit candidate requires that structure to accept a variable unit count. Establish shape/state/export correctness before a long fitting run.
5. **Fit the complete group.** Use a whole-group feature objective, final teacher-waveform L1 and multiscale spectral/mel reconstruction from the start. Normalize feature diagnostics with fixed training-set channel scales, not each quiet example's RMS. Pool valid samples correctly. Calibrate loss coefficients on the initialized compressed group and inspect actual disposable optimizer updates before freezing the recipe. The initial pilot uses fresh AdamW with no weight decay, betas 0.9/0.99 and epsilon 1e-8. The raw whole-group MSE is calibrated against waveform and mel parameter gradients; it is not independently whitened. Do not reuse unrelated ConvNeXt moments or add an optimizer comparison to this architecture experiment.
6. **Run a bounded recovery pilot.** First validate gradients, target alignment and sample accounting. Use an initial 1,000-update pilot with diagnostics at step 0, 256 and 1,000. Publish source exposure and audio hours beside steps. Only continue toward 10,000 if fixed development losses and regional errors show recovery. Do not wait for 0.99 correlation before enabling reconstruction objectives. The pilot uses three distinct sources per update through three singleton forwards with pooled gradient accumulation. An initial batched check exceeded the strict hidden-feature numerical tolerance before any fitting update, so the pilot uses the verified singleton teacher path. Learning rate and fixed loss coefficients are calibration outputs written to the launch manifest before fitting starts.
7. **Qualify the recovered model.** Evaluate native-rate waveform/spectral errors, silence, transitions, peaks, languages and nonverbals. Then run the existing source-referenced PESQ, STOI/ESTOI, UTMOS and DNSMOS pipelines, followed by blind listening. Human MUSHRA scores require actual ratings. Add adversarial/feature matching fine-tuning only if reconstruction recovers and listening/spectral evidence identifies remaining texture loss; use a step-based schedule, not near-perfect correlation as an entry requirement.

If one smaller group is insufficient and another group is later replaced, audit both teacher-prefix and actual student-prefix inputs. For local error, compare `S_group(x_student)` with `T_group(x_student)` using matching history and detached teacher targets. Keep the original full-teacher waveform target to prevent accumulated approximation errors from becoming the new reference.

The first pilot has one changed architectural variable: the internal widths of this complete group. It does not change activations, normalization type, sample rate, output head or numerical precision at the same time. Proposed calibration choices remain unvalidated until the preflight is completed.

## Data and storage

Reuse the existing Runpod corpus. A metadata check on 2026-09-10 found 186,641 nonempty source paths and 513.445 declared hours, including development data. The train split contains 183,802 sources and 510.927 hours. Excluding the known reserved source/hash/parent-recording union leaves approximately 501.698 hours across 181,008 sources. This verifies metadata and file presence, not a new decode or content-hash audit of every file.

Maintain all 22 scheduled Indic languages and the requested wider language mix. Use the existing expressive material, but ensure the fitting and evaluation manifests include laughter, crying, whistling, shouting, whispering, breathing and transitions. Downloaded language presence does not prove enough held-out speakers or coverage of Latin American accents. Do not treat generic unknown-session identifiers as real speakers and remove entire language groups accidentally.

Previously reused test panels are development panels. Reserve a fresh source-disjoint final panel; enforce speaker disjointness when real speaker identities exist and explicitly report unknown identities. All downloaded eligible data can be reused during debugging. Within a final training run, track source/crop exposure and honor the user's no-repeat policy. Candidates need matched data exposure for attribution.

The H100 was idle at inspection, but persistent storage had only about 182 MB free. Resolve durable checkpoint headroom before launch. Preserve downloaded audio and retained models; use only verified disposable caches for reclamation. Do not cache all intermediate teacher features across 500 hours: the 128-channel, 12 kHz group output alone would occupy about 11 TB in FP32. Generate bounded minibatch targets or a small disposable cache, and retain restartable manifests and durable checkpoints.

## Quality and speed decisions

The unchanged-copy numerical check and compressed-model quality check are different gates. The former retains the established waveform tolerance `atol=1e-5, rtol=1e-4`; the latter needs a nonzero approximation budget. A ratio against the exact teacher's zero reconstruction error is invalid.

Keep absolute quiet-audio checks, 0.99 nonquiet correlation as a final reconstruction target, bounded output and transient/peak fidelity. Report every failed source or region; a good global mean does not waive silence or expressive failures. Freeze the perceptual noninferiority margins before inspecting candidate quality results. Small metric differences alone do not establish inaudibility.

Qualify CPU-only, one-thread streaming on Intel first, at 80 and 160 ms, with a fresh paired Mimi comparison. Latest saved optimized AudioVAE2 RTF is 0.31152 / 0.24781 on Intel; the earlier Mimi results are 0.22228 / 0.18446. Those imply planning reductions of about 29% / 26%, but Mimi was not rerun in the latest optimization campaign. They are historical targets, not a current paired result. Apple and AMD follow after Intel demonstrates a useful gain. AudioVAE2 outputs 48 kHz and Mimi 24 kHz; report that difference explicitly.

Require exact sample counts and no additional lookahead, missing chunks, duplicated output samples or delayed flush. Use bounded causal state without recomputing the complete utterance at every call. Record full versus streaming parity, startup, 40 ms AudioVAE2 correctness, reset, variable tails and independent streams. Measure the final automatically selected exported runtime and account for any additional quantization error separately. A 42% arithmetic reduction is an opportunity to reach the Intel target, not a predicted 42% RTF improvement or a guarantee of matching Mimi on every CPU.

## Method references

[BERT-of-Theseus](https://aclanthology.org/2020.emnlp-main.633/) provides a precedent for training compact replacements inside a larger fixed model. Its NLP results and stochastic replacement schedule are not an audio-quality guarantee. [DepGraph](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) and its [author implementation](https://github.com/VainF/Torch-Pruning) support dependency-aware structural pruning, but custom causal convolutions, conditioning and streaming states still require explicit handling. These methods support the experiment design; the pinned AudioVAE2 code and this project's measurements determine its contracts and targets.

## Training dashboard

The active continuation resumes the selected 1,000-update checkpoint toward 5,000 updates, with validation and durable checkpoints every 500 updates. It retains AdamW at `3e-5`, the saved optimizer moments, RNG state, and fixed reconstruction coefficients. Teacher and encoder remain frozen. The next 12,000 fitting sources are distinct from the original 3,000 and the reserved panels; another 15,000 sources are reserved for a conditional continuation to 10,000. Frozen targets are prepared from downloaded Runpod audio in sealed shards. Training waits for missing shards without repeating data.

Open TensorBoard's **Custom Scalars → Progress → All quality metrics** for one overview chart. **Active waveform correlation (%)** is the main reconstruction trace; the flat 99 line applies only to that trace. Error-reduction traces use the fixed untrained baseline, quiet-window traces show passing percentages, and whistling level is a percentage of the teacher's level. Peak amplitude is shown as percent full scale. Raw values remain in **Text → Monitor → Latest raw metrics** and ordinary Scalars. Older run logs are preserved outside the active dashboard.

The **Training progress to 5000 steps (%)** trace shows actual completed updates between validation points. A separate CPU process reads the training log every ten seconds, without running the model or generating quality values. **Text → Monitor → Live status** shows the current step, latest validation, next milestone and prepared-data status. TensorBoard must use `--load_fast false --reload_multifile true --reload_multifile_inactive_secs 86400` so both event writers remain visible. The browser refresh interval can add a short delay.

The served display uses `tensorboard_display_v2.py` to read the original logs and give each metric a distinct color. The sidebar entries are metrics from the same training run; **Details** contains raw values and status. Labels state the 99% correlation target, 100% quiet-window pass targets, 100% full-scale peak limit and 100% teacher-level match for whistling. Error reductions show a 100% mathematical ideal, meaning zero error, rather than an established acceptance threshold. Original scalar values, steps and timestamps are preserved. Parentheses are omitted from display labels because TensorBoard 2.21's Custom Scalars color parser misreads them as part of the run name and fails to draw curves. The original training event directory remains the source; the served directory is `tensorboard-display-v2`. Extending the training target requires updating the display's explicit progress-label mapping too.

Physical batch size remains one, accumulating three distinct sources per optimizer update. A bounded check found a 12.28% forward/backward timing reduction for three equal-length sources, but mixed-length batching missed two strict gradient checks. Only 23.1% of the earlier updates had three equal lengths, so this is not a 12% whole-run speedup. Keep the validated singleton path for this continuation. The check made no optimizer updates and does not establish an audio-quality regression from the small numerical differences.

At 5,000 updates, inspect the fixed development trend in waveform, mel, group output, quiet residuals and expressive level before extending to 10,000. An improving global correlation alone does not override a worsening quiet or expressive result. This continuation does not automatically promote a model to production.

TensorBoard monitors the PyTorch training process. All three objectives begin at update1: raw teacher waveform L1, five-scale mel linear/log reconstruction, and raw whole-group feature MSE. Display each raw objective, each weighted contribution, and the complete weighted total. A displayed total must include every active term.

The reconstruction target is0.99 waveform correlation over teacher-active20ms windows. Display the mean, minimum and count of sources below that target. This is not a universal99percent perceptual-accuracy measurement. Quiet audio has separate absolute residual and amplitude checks; output bounds do not replace transient fidelity, source-referenced metrics or listening.

At step0,256 and1000, log fixed development results for each language, expressive label and selected quiet/transition group. Log direct full-width stage4 output error plus stage5,stage6, pre-tanh and waveform diagnostics. Stage2/3 diagnostics compare only retained teacher channel coordinates; their representations are free to adapt and those intermediate comparisons do not add losses or serve as final-quality gates. Streaming startup, interior joins and final tails are checked separately from neural-stage boundaries. Keep teacher and encoder frozen, and retain exact source/target identities in the run metadata.
