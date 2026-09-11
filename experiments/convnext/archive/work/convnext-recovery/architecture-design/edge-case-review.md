# Edge-case review for a budgeted synthesis-head change

This is a design review, not new benchmark evidence. It uses the current step-8,890 head-experiment results and the earlier layer and optimizer diagnostics. No source code or checkpoint was changed.

## What needs to be explained

| Observed failure | What the evidence supports | What an architectural change could do | What it would not establish |
|---|---|---|---|
| Stationary encoded silence has a repeating 10 ms residual | The final frame-to-480-sample readout maps nearly stationary features to unequal waveform positions. Teacher tanh contributes essentially nothing here. Teacher's final multichannel convolution cancels phase variation. | Allow final features and synthesis weights to learn together; introduce cheap shared local waveform synthesis in place of some existing head work. | Flat output for every constant hidden state is not a valid codec requirement. A constant latent may encode a sustained periodic sound. |
| Natural quiet audio still differs from the teacher | 81.93% of its residual power varies with the input. Student and teacher quiet RMS are already close. A universal silence correction explained only 0.65% RMS improvement. | Preserve changing low-level content while learning the final nonlinear mapping jointly. A small multichannel synthesis filter can condition its result on learned feature channels. | A silence detector, blanket attenuation, fixed silence subtraction or exact one-fixture constraint would discard or distort useful information. |
| Short overshoots in speech and laughter | There are 57 distinct events across six recordings, lasting 0.0208–0.3125 ms. None is within six samples of a 480-sample block boundary. Teacher pre-tanh maxima around 3 become approximately 0.995. Student head is unbounded. | A jointly adapted tanh output supplies the teacher's mathematical range property. Local multichannel waveform synthesis may improve transient shape. | An overlap/window fix cannot be claimed to fix a demonstrated join discontinuity. The observed peaks are interior. Bounding is not proof of faithful transient reconstruction. |
| Weak breath/whisper/whistle/scream reconstruction | Current panel nonquiet cosine is approximately 0.659/0.747/0.771/0.764, respectively, across only 3/4/2/7 sources. Laughter is about 0.893 across six sources. | Better learned sample-neighbor coupling may help broadband breath detail and rapid transitions, while preserving phase information. | There is no localized architectural diagnosis proving that one head change will fix these classes. Do not repeat the obsolete 0.07 correlation claim. Crying has no independently named condition group in this current panel. |
| Quiet regressions after a training update | Actual finite optimizer displacement overshot a helpful quiet direction; the optimizer also reversed a peak-improving raw-gradient direction. | Better parameterization could improve conditioning, but this is unproved. | A new layer does not remove optimizer interference or guarantee stable updates. |
| Startup and streaming differences | Replicate padding differs from the teacher's zero-padding contract; steady residual remains after the receptive field. All original decoder streaming count checks passed. | Any new synthesis state must preserve causal startup, chunk partitioning, empty calls and tail counts. | Changing startup padding is not a supported fix for the continuing silence residual. |

The present validation panel is development data and has been examined repeatedly. Multilingual/expressive coverage and small condition groups limit claims of generality. Architecture cannot substitute for a fresh, source-disjoint final quality assessment.

## What the failed head experiment actually establishes

The shared silence readout fit reduced steady residual 88.43% but increased natural quiet error 2.20%, waveform MAE 2.16% and high-frequency magnitude error 5.74%. The full four-phase fit improved the fixture 99.98% but worsened those real-audio measures further. The teacher's pre-tanh linear readout migration increased validation MAE 13.28%.

These results reject fixed-feature linear repair and the tested hard constraints. They do not show that the backbone cannot learn the teacher or that tanh is intrinsically harmful. The original nonlinear head features were frozen. A fair next architecture experiment should learn its nonlinear synthesis features and weights together, primarily against the teacher's actual final waveform. Exact teacher pre-tanh copying is not required to produce the same post-tanh waveform.

## Strong candidate: reallocate head compute to a multichannel synthesis head

Current head, after the unchanged 100 Hz ConvNeXt body:

`512 channels → causal k3 convolution → 2048 channels → PReLU → 480 outputs → flatten to 48 kHz`

A minimal candidate preserving the nonlinear head width:

`512 channels → causal k2 convolution → 2048 channels → PReLU → 2×480 outputs → two 48 kHz feature channels → causal k7 convolution, 2→1 → tanh`

This spends less of the head budget mixing 100 Hz history and more mixing neighboring waveform samples across learned channels. The entire ten-block body remains. It removes one 10 ms head history frame and adds six waveform samples of past state, without future lookahead. The two channels are freely learned feature signals, not a quiet/loud gate or a post-hoc copy of the waveform.

The mechanism is closer to the teacher's final multichannel waveform convolution than an appended scalar smoothing filter. Crucially, the feature channels and final filter train together. A fixed scalar filter on the already synthesized waveform has only one signal to work with and previously harmed quality.

Approximate multiplication counts per output second are 412.877 million for the current head and 406.995 million for this candidate, excluding activations, bias, reshaping and memory movement. This is an arithmetic budget, not a speed prediction. It also changes parameters and temporal factorization; exact checkpoint migration is not available by merely copying every current head weight.

If a wider compute margin is necessary, a four-channel candidate with a 1024-wide k3 nonlinear head is about 355.238 million MAC/s before tanh. That retains k3 temporal support but halves nonlinear head width. There is no evidence that this width reduction is quality-neutral. Do not test a large width/channel sweep before deciding which tradeoff is justified.

The two-channel candidate is the more conservative capacity choice; the four-channel candidate gives more runtime margin. Neither reproduces the teacher's full 32-channel high-rate representation. That is intentional budget reallocation, but its adequacy remains a hypothesis.

## Alternative: causal overlapping waveform synthesis

A low-rate head that emits a longer waveform kernel and overlap-adds the previous frame's tail is also plausible. It can be causal with no lookahead and only bounded tail state. It creates dependence on adjacent hidden frames with relatively little work when compensated by narrower head width.

However, overlap alone does not guarantee phase-consistent DC response. For constant hidden features, unequal polyphase filter sums can still produce a periodic waveform. Windowing can suppress or redistribute a pattern without preserving detail. More importantly, the actual peak events are not joins. Describe overlap as a useful synthesis inductive bias, not as a diagnosed cure for current overshoot or all quiet failures.

For this specific evidence, a small jointly learned multichannel waveform synthesis stage transfers the observed teacher mechanism more directly. An overlap head remains a competing hypothesis, not something to stack on the first candidate immediately.

## What should stay out of the proposed change

- No universal flat-response constraint for all constant latents or features. It can restrict sustained-tone synthesis, including whistling.
- No teacher-silence feature subtraction presented as free additional capacity. With affine compensation it is a reparameterization; without it, the learned function changes. Four-phase anchoring already showed real-audio costs.
- No weakening of the final ConvNeXt block. Its local peak derivative conflicts with useful ordinary-speech and quiet reconstruction derivatives.
- No claim that tanh fixes silence. It is almost identity at the relevant amplitudes.
- No claim that reduced MACs imply equal or lower single-thread streaming RTF. Kernel dispatch, memory traffic and layout must be measured in the eventual implementation.

## Evaluation interpretation

Use a matched existing-head nonlinear-training control when eventually testing a replacement. Otherwise any improvement could be caused by extra training rather than the new architecture. Keep the preserved backbone as the starting point, initially frozen; do not restart all learned layers from random weights. Use the actual post-tanh teacher waveform as the main reconstruction target. Monitor synthetic silence as one condition alongside real quiet trajectories, without a new mandatory silence equality constraint.

An output bound can be guaranteed mathematically. Teacher-equivalent fidelity, all-edge-case coverage and no extra CPU time cannot be guaranteed from this design. A candidate should be retained only if it preserves detailed real-audio reconstruction, improves the relevant failure populations and meets the measured original CPU budget.

## Evidence read

- `outputs/convnext-recovery/head-experiments/report.md` and `results/baseline_raw.json`
- `outputs/convnext-recovery/optimizer-silence-causal/architecture-analysis.md` and `natural-layer-results.md`
- `outputs/convnext-recovery/peak-silence-diagnosis/report.md`
- `outputs/convnext-recovery/candidate-rescore/report.md`
- `work/fast-audiovae/experiments/convnext/audiovae_student/model.py`

## Review of the richer 6 kHz candidate

After comparison with the independently proposed multirate head, prefer this replacement over the two-channel option above:

`512@100 Hz → causal k3, 1920 channels → PReLU → phase shuffle to 32@6 kHz → causal 32→32 k3 → PReLU → causal 32→1 synthesis, stride 8/kernel 16 → tanh`

This introduces actual nonlinear feature interaction at a submillisecond scale and leaves approximately 23.36% head-MAC headroom, rather than the two-channel option's tight 1.4% margin. Its 316.416 million head MAC/s is still only about a 3.8% reduction in the complete decoder. The new feature convolution adds 64 history floats; synthesis adds eight waveform-tail values, totaling 72 additional state floats. Existing body/head state remains.

The 6 kHz features do not imply a 3 kHz waveform bandwidth limit. They are 32 learned feature channels, not a downsampled mono waveform. The final linear synthesis has eight waveform output phases per feature frame; a full-row-rank 8-by-32 polyphase mapping can represent all eight phases and therefore full 48 kHz output bandwidth. This is a capacity observation, not evidence that trained whistling or high-frequency audio will match the teacher.

Aliasing and periodic noise remain risks: the 60-way feature shuffle and final stride-eight synthesis can generate unequal phase responses. Two-fold overlap does not enforce equal polyphase gains, and nonlinear feature processing does not supply an anti-alias guarantee. The design offers a better place to learn local cancellation and transient structure; it does not mathematically eliminate those artifacts. Do not impose a universal constant-feature-to-flat-waveform constraint to manufacture that guarantee.

For streaming, each original latent still yields 4×60×8 = 1920 waveform samples. Synthesis emits the current eight samples after adding the previous frame's tail, saves the next eight, and applies tanh after the overlap sum. A batch causal reference right-trims the trailing eight synthesis samples; a centered transposed convolution would violate this alignment. It needs neither future lookahead nor additional latent buffering, but initial state, final tail handling and variable chunk partitioning must be verified.

A matched continuation of the unchanged nonlinear head remains the required control. New-head quality cannot be credited to architecture if the baseline receives no equivalent training. Full-band whistle/scream errors, breath/whisper detail, low-level periodic spectral lines, natural quiet trajectories and peak shape must remain separate outcomes. Neither a bounded maximum nor a stationary-silence improvement qualifies the replacement alone.
