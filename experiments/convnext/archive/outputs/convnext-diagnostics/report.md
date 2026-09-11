# Student decoder diagnostic audit

The strongest evidence concerns the size and interaction of the actual parameter updates, plus a separate failure to control rare peaks. The tests do not establish that the architecture is incapable of matching AudioVAE2. They also do not support treating a small output filter, an output bias, or removal of GAN as a complete solution.

A separate new gap concerns encoder-cache reproducibility: two laughter recordings do not reproduce their cached latents when the verified original source is re-encoded. Their cached latent/teacher pairs remain internally consistent. The historical input or encoder-preparation difference needs clarification before restarting training.

The parent at step 8,090 and the targeted and complex candidates at step 8,490 were preserved. Training stayed paused. All diagnostic optimizer updates used disposable state, and every retained checkpoint hash remained unchanged. No architecture, loss, optimizer policy or deployed decoder was changed.

**What was run.** The campaign used the original frozen teacher cache and 285 held-out crops from 147 sources, including 282 natural crops from 144 sources. It covered teacher self-scoring, objective interpolation, gradients through every parameter group, native optimizer replays, fractional update paths, parameter-group interventions, the complete residual and peak panel, head singular values, source-separated readout fitting, and teacher/student history and amplitude probes. The readout used 1,024 training windows from 878 sources, with 703 fitting sources and 175 tuning sources. The H100 ran training diagnostics; these are not CPU RTF measurements. All 42 helper tests passed remotely: 28 main, eight supplement and six natural-context repair tests.

## The retained quality gap

These are the unchanged full natural-audio panel scores against the frozen teacher output. They are not quality scores against original recordings or newly trained results.

| Checkpoint | Mean nonquiet cosine | Waveform MAE | Quiet residual RMS, pooled | Maximum peak | Failed quiet windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent 8,090 | 0.91286 | 0.012814 | 0.0003461 | 1.2535 | 3,434 / 3,434 |
| Targeted 8,490 | 0.91991 | 0.012552 | 0.0003067 | 1.2655 | 3,434 / 3,434 |
| Complex 8,490 | 0.92027 | 0.012546 | 0.0003122 | 1.2634 | 3,434 / 3,434 |

Teacher self-scoring gave exactly zero reconstruction error and passed all 4,073 quiet windows, including the synthetic fixtures. Its maximum peak on this panel is 0.9948. This verifies internal consistency of the checks; it does not validate their thresholds as human audibility limits. The current limits remain provisional engineering checks.

## Actual updates explain more than output-gradient measurements

A controlled native D-then-G update used the same 32 training crops at each checkpoint. The resulting audio was scored on 18 deliberately difficult held-out crops. Percentages below are changes in equal-crop mean squared error or peak-excess energy, not corpus-wide quality estimates or RMS changes. Peak-excess energy is `mean(relu(abs(student) - max(1, abs(teacher)))²)` over valid samples; it is distinct from peak amplitude and the number of overshoots.

| Checkpoint | Quiet squared error | Overall waveform squared error | Peak-excess energy |
| --- | ---: | ---: | ---: |
| Parent | -16.49% | +0.259% | +31.46% |
| Targeted | +34.96% | +0.741% | +17.54% |
| Complex | +49.02% | +0.578% | +12.48% |

The parameter-gradient prediction says quiet error should fall at all three checkpoints. Smaller displacements along the exact same native update confirm a locally helpful direction:

| Fraction of the same update | Targeted quiet squared error | Complex quiet squared error | Targeted peak-excess energy | Complex peak-excess energy |
| --- | ---: | ---: | ---: | ---: |
| 10% | -3.24% | -2.72% | +1.62% | +1.14% |
| 25% | -5.23% | -3.52% | +4.10% | +2.89% |
| 50% | -1.01% | +3.68% | +8.39% | +5.93% |
| 100% | +34.96% | +49.02% | +17.54% | +12.48% |

The full displacement reverses the quiet benefit. Peak error, however, worsens even at 10%. These are separate problems: reducing the displacement alone does not make this update favorable for both. This interpolation is not a learning-rate sweep, because optimizer moments, clipping and decay were computed only once at the saved rate. It cannot select a production learning rate or establish what happens across many batches.

There is also substantial interaction between parameter groups. On targeted, applying only the cached Muon-parameter displacement changes quiet error by -0.15%; applying only the AdamW-parameter displacement gives +0.44%; applying both gives +34.96%. The corresponding complex values are +1.91%, +8.35% and +49.02%. These effects are nonadditive. Squared-error cross terms can produce nonadditivity even for a linear model; the numbers do not prove an optimizer conflict or a nonlinear defect. They implicate the combined displacement as the quantity to assess, not one universally bad optimizer.

Applying only blocks 8 and 9's cached displacement worsens quiet error by 14.44% in targeted and 27.96% in complex, while improving overall waveform MSE. Applying the output projection's displacement alone slightly improves quiet error. This localizes an important quality tradeoff upstream of the final projection. It does not justify freezing those blocks without a separate solution test.

Removing the current GAN or feature-matching gradient does not fix the quiet regression. Removing waveform reconstruction substantially worsens overall error. With all current gradients set to zero, the saved optimizer state and weight decay still produce a substantial update. Their separate contributions were not isolated. The exact D-before-G step closely matches the fixed-D replay, so that ordering does not explain the observed regression on this batch.

Every generator update in the saved 8,090-step history and each 400-step comparison arm was clipped. Recent median preclip norms are about 313–333 with a threshold of 1. This does not mean adaptive parameter updates become 313–333 times smaller. No loss-balancer coefficient hit its configured bounds. The complex arm's achieved FM+GAN norm share averaged 40.20% versus the intended 30%; current FM norms averaged 1.50 times their EMA estimate. This is a remaining controller-calibration gap, not evidence of coefficient saturation.

## The loss still rewards the teacher

Along the tested output interpolation, the weighted objective decreases at all five tested interpolation points with the checkpoint EMA coefficients held fixed. For targeted it goes from 59.490 at the student, to 0.656 at 99% interpolation, to 0.00244 at the teacher. Quiet, active and peak regions have favorable combined gradient projections before equality. Waveform, mel and feature-matching losses and their returned gradients become exactly zero at equality.

An adversarial gradient remains at the teacher. That alone does not prove the teacher is not a local optimum: L1 losses have a cusp at equality, where automatic differentiation selects a zero subgradient. The tested direction provides no evidence that the current objective stops rewarding teacher agreement. It does not prove that every desired waveform is realizable by the student parameters.

## The noise is not only a repeating output template

The corrected component panel includes all 285 crops. The previous component selector accidentally examined only three encoded fixtures and one natural recording. The original full quality panel was unaffected.

For 1,409 complete natural quiet 40 ms cycles in crops with enough cycles for decomposition:

| Checkpoint | DC power | Repeating 480-sample component | Additional 1,920-sample component | Remaining variation |
| --- | ---: | ---: | ---: | ---: |
| Parent | 4.5% | 42.1% | 5.1% | 48.4% |
| Targeted | 14.7% | 17.2% | 6.9% | 61.2% |
| Complex | 12.9% | 20.2% | 6.9% | 60.0% |

Targeted reduces absolute 480-periodic error power from 4.55e-8 to 1.42e-8, while remaining variation barely changes, from 5.24e-8 to 5.08e-8. Removing a fixed repeating pattern therefore cannot account for most remaining natural quiet error. The component masks differ intentionally from the primary 20 ms quiet checks; these percentages must not replace the primary RMS metric.

The exact parent-to-candidate decomposition attributes much more waveform-change power to changed upstream features than to head weights: approximately 42 times as much for targeted and 39 times for complex in natural quiet audio, with partial cancellation. This identifies where the recent changes occurred, not the original cause of every artifact.

After deduplicating overlapping crop observations, overshoot affects 248, 282 and 283 physical samples across six, five and six sources respectively. None of the 37, 42 and 43 contiguous overshoot events peaks within six samples of a 480-sample boundary. The largest peak occurs in a Sindhi speech recording. Overshoot is not confined to laughter or chunk joins.

## The output head is not collapsed, and a fitted replacement is not an all-metric fix

All heads are 480 by 2,048, have numerical rank 480, and condition numbers of 6.51, 6.77 and 6.91. A collapsed output basis is ruled out. This says nothing conclusive about whether upstream features can represent every needed waveform.

The initial readout grid selected its strongest regularizer, so the diagnostic was extended. All six extended probes select an interior relative ridge value of 0.3. Fitting MSE improves 12.9–15.0%, but source-disjoint tuning MSE remains 1.3–1.6% worse than the learned head. On untouched natural audio, the same-dimension bias-free readout produces these changes:

| Checkpoint | Waveform MAE | Mel error | Quiet residual RMS | Maximum peak |
| --- | ---: | ---: | ---: | ---: |
| Parent | +2.23% | -12.35% | +2.45% | 1.2735 |
| Targeted | +1.97% | -11.90% | +13.94% | 1.2819 |
| Complex | +1.44% | -9.96% | +9.13% | 1.2729 |

Adding an intercept does not remove the tradeoff. The tested ridge readouts did not produce an all-metric improvement. It does not establish an architectural capacity limit or an optimum across every possible readout-training method.

## History, level and alignment

The final natural panel contains eight distinct sources, two each for speech, laughter, whistling and quiet audio. Seven have authenticated complete prepared source recordings for amplitude re-encoding, spanning all four groups. The Yell source remains in the history panel but lacks an entry in the manifests used by this amplitude pass; its earlier separate amplitude probe is not silently merged into this panel. Historical quality scores are unchanged.

With 30 latent history frames, the largest teacher reset-versus-longest-history error is 1.51e-6 in waveform amplitude; all student errors are at most 8.35e-7. RMS discrepancies are at most 1.66e-7 for the teacher and 1.20e-7 for the students, far smaller than their reconstruction errors. The 29-, 30- and 40-frame results differ only at this numerical scale on these examples. All 32 teacher/student future-perturbation checks preserve earlier output bitwise. Insufficient decoder history and decoder future leakage are not supported as causes on this panel. These are empirical checks, not a proof about every encoder or deployment path.

Re-encoding the original input at gains 1, 0.5, 0.1, 0.01 and 0 reveals a persistent excess output floor. At zero input, teacher RMS is about 9.63e-6, versus 2.16e-4 for parent, 1.43e-4 for targeted and 1.23e-4 for complex. The newer students still emit about 15 and 13 times the teacher's RMS. This continues to matter when real expressive material becomes very quiet; it is not only a startup artifact.

Bounded lag and gain fitting explains less than 4% of the selected targeted speech/laughter/whistle MSE. One complex whistling example is different: a six-sample timing adjustment and a 1.154 gain reduce MSE by 55.75%. Its raw cosine is already 0.99276. This demonstrates that a high cosine alone does not certify correct level or waveform fidelity, and that timing/level errors are case-dependent. Fitted corrections never replace the unmodified scores or delivered audio. Quiet-case improvements at the minimum allowed gain mostly show attenuation of unwanted output, not recovered quiet signal.

**A new historical source/encoder-preparation discrepancy needs resolution.** For laughter sources `freesound:119459` and `freesound:386521`, freshly encoding the authenticated complete source yields latent maximum differences of 1.03 and 1.37 from the cache. The resulting teacher waveforms have cosine 0.9610 and 0.9568 to the cached targets on the selected intervals. The other five authenticated sources match their cached baselines exactly or to small numerical differences.

A bounded two-source follow-up compared full recording encoding with encoding only its first 64 latent frames' worth of audio. Full versus prefix encoding is identical or differs by at most 1.23e-5 in the latent, while both differ substantially from the historical cache across all 64 frames. This rules out that specific right-boundary explanation. Decoding the cached latents still reproduces their cached teacher waveform exactly or within 3.99e-7. The original cache files also match their recorded hashes, their latent prefixes match the held-out tensors, and the stored input audio is bitwise identical to the authenticated source. Their metadata declares the same frozen source/weights and deterministic FP32 posterior-mean policy. File drift, input gain/channel/resampling differences, and this truncation explanation are therefore excluded by these checks. The paired supervision is internally coherent; the unresolved issue is historical encoder execution/preparation reproducibility despite those declared identities, not established target corruption or evidence to discard every cache.

The independently re-encoded amplitude experiment remains internally matched, but its laughter baseline cannot be presented as identical to the archived baseline. The two baseline families remain separate. The underlying preparation cause and its prevalence across the larger corpus are not yet established.

## Gap register

The original gap list is retained separately. This register records what this campaign established and what remains open.

| ID | Gap or hypothesis | Diagnostic status |
| --- | --- | --- |
| G01 | Detailed component panel omitted most natural cases | Diagnostic gap closed: all 285 crops examined; original quality panel was valid. |
| G02 | Output gradients were being used to infer optimizer effects | Measurement gap closed: gradients, 21 native counterfactual updates, three fixed-update paths, and parameter subsets measured. |
| G03 | Quiet-gradient concentration might mean silence needs more weight | Not established. Combined infinitesimal direction helps quiet audio; full native displacement can reverse the benefit. Quiet and active gradients are not globally opposed on the routing panel. |
| G04 | Loss balancing drift | Confirmed: complex FM+GAN averages 40.20% rather than 30%; EMA lag remains. Coefficient-bound saturation is ruled out. |
| G05 | Current losses stop rewarding exact teacher reconstruction | Not supported on the tested paths. Reconstruction losses vanish at the teacher and combined objective decreases toward it. Residual GAN gradient is not by itself proof of an impossible optimum. |
| G06 | Collapsed or undersized output basis | Collapsed basis ruled out: rank 480, moderate conditioning. Architectural learnability remains unresolved. |
| G07 | Repeating silence residual could be fixed only at the head | Incomplete explanation. Most remaining natural quiet error is not the repeating 480-sample component; upstream feature changes dominate recent quiet waveform changes. |
| G08 | Universal quiet failures might be a scoring artifact | Teacher self-check passes. Student residual and level failures are real under these checks; audibility calibration of the thresholds remains open. |
| G09 | Rare peak overshoot | Confirmed and separate from the quiet step-size problem. Peak error worsens even along small fractions of the current update. Peak events are interior and not limited to laughter. |
| G10 | Insufficient or mismatched teacher history | No meaningful decoder-history deficit on the eight natural cases; 30 frames matches longer context to numerical precision. |
| G11 | Timing or gain error could explain the reconstruction gap | Case-dependent. Small gains for most tested speech and expressive cases; one complex whistle has a substantial gain/timing component. High correlation alone misses level errors. |
| G12 | Validation coverage and independence | Still incomplete: tiny language cohorts and missing separately reviewed expressive groups; unknown speaker identities limit independence claims. |
| G13 | Earlier bundled interventions prevent causal attribution | Still applies to the earlier 400-step comparisons. New fixed-state counterfactuals localize current behavior but do not retrospectively isolate every historical change. |
| G14 | Finite-update stability near quiet audio | Newly confirmed on this panel: locally helpful quiet directions become harmful at the full targeted/complex update. Generality across training batches is not established. |
| G15 | Interactions between parameter families | Newly confirmed on this panel: the joint Muon/AdamW parameter displacement has much larger quiet error than either subset alone. Not proof that one optimizer should be removed. |
| G16 | Frozen-feature readout generalization | Extended grid resolves the initial boundary limitation. Fitted heads still trade waveform/quiet fidelity for mel gains on unseen sources; bias alone is not a complete fix. |
| G17 | Historical input/encoder preparation reproducibility | Newly observed in two laughter sources: fresh full-source encoding differs materially from cached latents; cached latent-to-target decoding remains consistent. The source of the difference and corpus-wide prevalence remain unresolved. |

The diagnostic implementation also had a raw-audio availability gate that wrongly excluded cached natural latents from history tests. The original four-case history report and first attempted supplement remain preserved as incomplete coverage. The final natural-source panel removes that gate and asserts coverage before model execution. No primary evaluation scores were replaced.

## Limits and preserved state

These probes characterize three checkpoints and a small controlled update panel. They do not establish population-wide learning dynamics. Known source, audio-hash, recording, speaker and session identities were grouped for the readout split; speaker IDs are missing for 562 of the 878 selected sources, and session IDs for 24. Unknown-identity leakage cannot be ruled out by this probe. Full training includes all 22 scheduled Indic languages, but some held-out language groups are very small. Separately reviewed crying, giggling and shouting evaluation groups remain incomplete.

Legacy normalized evaluation-loss fields and current raw training losses use different reductions. The report uses raw waveform error, unchanged mel evaluation, and explicit aggregation labels rather than comparing their totals as if they were interchangeable.

No tiny-set memorization training or new broad model variant was run. Architectural learnability remains an open question; the current diagnosis provides concrete update and representation gaps without declaring a capacity ceiling. All retained checkpoints and the frozen teacher cache remain preserved, and sustained training remains paused.

Evidence: [compact results](results.json), [objective and native-update measurements](objectives-updates.json), [fixed-step path](native-step-path.json), [feature summary](feature-summary.json), [extended readout summary](ridge-extension-summary.json), [optimizer history](optimizer-history.json), [natural history and amplitude](natural-history.json), [laughter preparation check](laughter-boundary.json), [original cache provenance](laughter-cache-provenance.json), and [gaps recorded before the run](known-gaps-before-run.md).
