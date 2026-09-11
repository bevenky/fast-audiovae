# Diagnose the reconstruction gap before changing the decoder

The next investigation should distinguish four possible causes: inconsistent inputs or scoring, losses that oppose exact reconstruction, useful information that the trained output head does not use, and limitations in the learned upstream representation. The current evidence does not establish that the decoder architecture is incapable of matching the teacher.

This document is based on a new read-only source audit, analysis of the saved four-arm results, and a direct read of the original step-8,090 checkpoint configuration. No model forward passes, optimizer steps, fine-tuning, architecture changes or training restarts were performed for this audit. The diagnostics below are proposed, not completed measurements.

## What the audit found

**The detailed component panel has a selection bug.** `run_corrected_screen.py` selects lowercase condition names, while saved metadata uses `Laughter`, `Screaming`, `Whispering`, `Breathing` and `human_whistling_source_description`. Most of these records do not have its alternative `group="expressive"` field. As a result, every component report contains only three synthetic fixtures and the added yelling recording. The primary evaluation still contains all 285 crops; its quality results remain valid. The detailed DC/480/1920 decomposition does not yet cover natural silence broadly or the bad laughter peaks.

**The gradient audit stops before the network and optimizer.** `corrected_calibration.py` detaches the predicted waveform and differentiates the losses with respect to that waveform. Actual training then routes this gradient through shared model parameters, clips the global parameter-gradient norm, and applies native Muon to 20 hidden matrices and AdamW to the remaining parameters. A helpful waveform-space gradient can become an unhelpful change elsewhere through shared parameters, momentum or preconditioning. Current reports do not measure that final effect.

In the targeted arm's final fixed audit, quiet samples are 15.747% of the batch but contain 82.241% of combined output-gradient energy. Their waveform-only gradient has residual cosine 0.6777; the combined gradient has cosine 0.1771. Feature-matching and adversarial gradients have cosines 0.00293 and 0.00115 there. This is one audit batch, not a population estimate. It does not support assuming that silence simply receives too little gradient.

**The actual output matrix is 480 × 2,048, with no output bias.** A direct checkpoint read confirmed these dimensions, the 512-channel hidden body and the raw-repeat phase adapter. The output matrix can have full row rank. We have not measured its rank or conditioning. A small-dimensional head is not an established limitation. Upstream biases and normalization offsets can still produce a repeating output even though the final projection has no bias.

**Rare peaks and average error are different objectives.** The waveform objective is sample-weighted L1. Its derivative magnitude does not increase with residual amplitude away from zero, so a few severe overshoots need not dominate training. This is an objective property, not proof that it causes the observed peaks.

## First diagnostic pass: fixed checkpoints and identical inputs

Use the parent checkpoint and the targeted and complex candidates. Preserve weights, buffers, discriminator states, optimizer state, loss-balancer state and RNG. Keep the existing quality thresholds and unadjusted scores. Include matched ordinary speech controls, interior natural silence, speech-to-silence transitions, startup, every unique overshoot event, and the available expressive groups. Select by the canonical evaluation categories and observed failures, with assertions for expected membership. Deduplicate physical peak events by source and absolute sample position.

| Priority | Diagnostic | What it distinguishes |
| --- | --- | --- |
| 1 | Teacher self-check and loss behaviour approaching the teacher | Invalid scoring or objectives opposing the desired target |
| 2 | Trace each loss into shared model parameters | Harmful interference hidden by waveform-space gradient statistics |
| 3 | Expand residual analysis and separate head-weight changes from feature changes | Stable offsets, temporal structure, peak errors and where changes arise |
| 4 | Fit a diagnostic readout on frozen features, tested on separate sources | Information already present but poorly decoded versus inadequate current features |
| 5 | Matched-history and timing/level sensitivity checks | Context, alignment or nonlinear response that aggregate correlation conceals |

### 1. Does the objective prefer the exact teacher output?

First evaluate the teacher against itself through the same masks and quality functions. Reconstruction errors should be zero and quiet checks should pass. Exact all-zero correlation is undefined and must not be treated as a failure.

Then substitute the cached teacher waveform for the detached student output in the generator losses. Keep the same discriminator, example weights, view offsets and loss scales. Waveform, mel and feature-matching errors should vanish. The adversarial derivative need not vanish at the teacher: its objective still asks the fixed discriminator for a real label. A nonzero adversarial loss alone is not evidence of a problem; inspect gradient magnitude, direction and location.

On the same valid samples, evaluate `y(a) = (1-a) * student + a * teacher` for `a = 0, 0.5, 0.9, 0.99, 1`. Record each loss and its derivative toward the teacher, separately for quiet, active speech and peak neighbourhoods. Keep EMA scales frozen so this measures the current objective geometry rather than a changing weighting rule. Record the exact pipeline behaviour at equality without interpreting any zero-error balancing degeneracy as a normal training step.

If approaching the teacher reliably increases the effective objective in problematic regions, we have evidence of objective conflict. If all terms support convergence there, look next at parameter routing and representation. This interpolation is a diagnostic path through waveform space, not necessarily a waveform path realizable by the student.

### 2. Follow the gradient through the network

Compute per-loss parameter gradients for the adapter, normalization, each temporal block, final hidden projection and waveform head. Separate the gradient originating in quiet samples from that originating in active samples before routing each through the same model Jacobian.

Measure pairwise angles and the directional derivative of diagnostic quiet-residual energy and teacher-relative peak-excess energy along each parameter gradient. These diagnostic functions do not become new training losses. Track actual clipping multipliers from the saved logs; a configured clipping threshold does not show how often it acts.

If speech and silence compete in the same parameter groups, this localizes a mechanism that the output-only audit misses. Gradient-conflict research motivates examining directions, not just magnitudes, but its proposed algorithms are not automatically a solution for this decoder. [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782).

### 3. Locate the repeating residual and peak changes

Repeat the detailed DC/480/1920 decomposition across the complete natural failure panel. Report absolute component power and RMS as well as fractions, split startup from interior silence, and compare block-boundary jumps with ordinary waveform differences. Percentages can increase merely because another component shrank.

Capture the actual vector entering the waveform projection. With output `y = W h`, the change between two checkpoints can be decomposed exactly:

`change in y = (W1-W0)(h1+h0)/2 + (W1+W0)(h1-h0)/2`.

The first term associates output changes with the head weights; the second associates them with upstream features. Report both and their cancellation, particularly at laughter overshoots and quiet intervals. Also separate each feature vector into its quiet phase mean and deviation, then project both through the existing head. This distinguishes a stable feature offset from varying residuals without adding a layer or fitting a corrective filter.

Periodic structure can arise from the synthesis parameterization or discriminator gradients. The image-domain literature motivates checking both; our direct nonoverlapping waveform head does not have the uneven-overlap mechanism of a conventional transposed convolution. Periodicity alone therefore does not identify the cause. [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/).

### 4. Is the information already in the learned features?

Collect the actual 2,048-dimensional vectors before the output projection and their aligned teacher blocks. Fit a regularized least-squares readout on training sources only, choosing regularization on separate validation sources. Evaluate on untouched sources or speakers, with conditioning and rank reported. Compare the current projection, a fitted projection of identical dimensions with no bias, and an affine diagnostic with an intercept.

This is a diagnostic coefficient fit, not student fine-tuning or a deployed replacement. Least squares optimizes squared error; report that alongside unchanged MAE, mel, quiet and peak measures rather than claiming a universal reconstruction bound.

- Improvement on held-out sources with identical dimensions means useful information is present but the trained readout uses it poorly.
- Improvement only with an intercept points toward offset representation or calibration, though earlier layers may already be capable of supplying that offset.
- Training-only improvement indicates overfitting.
- Little improvement does not prove inadequate architectural capacity: the upstream network may still learn better features.

This adapts the principle of frozen-representation probes to continuous waveform reconstruction. Simply projecting teacher blocks into the unrestricted column space of a full-row-rank output matrix would be uninformative. [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644).

### 5. Verify history, alignment and sensitivity

The current model has 116 past internal frames of theoretical history, approximately 1.16 seconds. Training supplies 30 latent context frames; that covers its own finite receptive field. This does not establish that the teacher needs no longer history or that the model effectively uses the available history.

Decode the same latent sequence continuously and with controlled prefixes in both models, retaining the same absolute scored samples. Measure the teacher's own full-versus-window discrepancy before interpreting the student's error. Avoid re-encoding arbitrary crops, which would change the input latents.

Report bounded local timing-offset and gain diagnostics separately from the unmodified primary scores. A large alignment-adjusted improvement suggests phase/timing error rather than missing spectral content. Do not use arbitrary time warping, clipping or fitted gain to claim improved delivered quality.

For quiet/peak sensitivity, create modest amplitude changes in the input audio and re-encode each through the frozen encoder. Compare the teacher and student responses. Do not assume scaling latent vectors is equivalent to scaling audio. Jacobian and finite-difference checks can identify excessive student gain or brittle channel directions; theoretical receptive field and effective sensitivity are different measurements. [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128).

## Only if the fixed-weight diagnosis remains ambiguous

An isolated native-optimizer replay is the next discriminator between objective and optimizer behaviour. Clone the checkpoint and all native optimizer state, compute one disposable update, and measure its predicted `J * delta` and actual audio change on an untouched probe panel. Compare the combined loss with leave-one-loss-out and zero-current-gradient controls, keeping the other scales fixed. Include momentum and weight decay. Separate fixed-discriminator analysis from exact next-step replay, which must also reproduce the discriminator update that normally occurs before the generator update.

These are counterfactual effects, not additive percentages: Muon, AdamW and global clipping are nonlinear in the combined gradient. The retained checkpoints, sampler and training ledger must remain untouched. This replay is proposed separately from the fixed-weight pass.

A tiny-set fitting diagnostic is a later option only if representation versus learnability remains unresolved. Success would show that the existing architecture can fit those examples, not that it generalizes. Failure would still require checking optimizer and conditioning before declaring a capacity limit. Another broad 400-step model variant is not the next diagnostic.

The immediate recommendation is to complete priorities 1–3 first, then use the frozen-feature readout and history/sensitivity checks to resolve whichever explanation remains. Do not add losses, filters, layers or more silence weighting until the measurements identify what they need to correct.
