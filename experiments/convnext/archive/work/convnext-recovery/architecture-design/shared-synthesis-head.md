# A compute-budgeted local synthesis head

This is an architecture proposal, not a measured quality or CPU improvement. No model has been changed or benchmarked for this note.

## Evidence and limits

The actual student uses a 512-channel temporal backbone at 100 Hz, then a causal 512-to-2048 kernel-3 convolution, scalar PReLU, and a 2048-to-480 linear projection. The 480 values are contiguous waveform samples. Its steady hidden response can therefore produce a periodic waveform. The teacher's final multichannel convolution nearly cancels its periodic component, and its tanh bounds peaks. These are different mechanisms.

The fixed-readout experiments prove neither that the backbone is wrong nor that a new architecture is necessary. They show that the tested silence anchors altered useful natural-audio predictions. Natural quiet residual is predominantly input-dependent, so fixing one stationary response does not solve quiet speech. Overshoot occurs inside recordings too, so boundary smoothing alone does not solve it.

## Recommended architectural candidate if a head replacement is desired

Keep the encoder, 64-dimensional latents, latent adapter, all ten ConvNeXt blocks and 100 Hz backbone. Replace the complete current synthesis head with:

1. Causal convolution: 512 to 1,920 channels, kernel 3, followed by PReLU.
2. Reshape each 1,920-vector into 60 successive 32-channel frames. This produces 32 channels at 6 kHz.
3. Shared causal convolution: 32 to 32 channels, kernel 3, followed by PReLU.
4. Learned shared waveform synthesis: 32 to 1 channels, transposed-convolution kernel 16, stride 8, implemented as causal finite-impulse-response overlap accumulation.
5. Tanh, trained jointly with the synthesis head against the actual post-tanh teacher waveform.

The essential change is a short nonlinear multichannel waveform synthesis path. It learns interactions between neighboring short waveform regions, rather than filtering a completed scalar waveform. This borrows a useful AudioVAE2 mechanism while retaining low-rate bulk computation. It is not a miniature copy of the complete expensive AudioVAE2 decoder.

## Exact arithmetic budget

One multiply-accumulate is counted as one MAC. Biases, activations, normalization, tensor layout and dispatch are excluded here.

| Head operation | Current MAC/s | Proposed MAC/s |
|---|---:|---:|
| Low-rate causal projection | 100 × 512 × 2048 × 3 = 314,572,800 | 100 × 512 × 1920 × 3 = 294,912,000 |
| Current waveform projection | 100 × 2048 × 480 = 98,304,000 | Removed |
| Shared local nonlinear projection | None | 6000 × 32 × 32 × 3 = 18,432,000 |
| Shared waveform synthesis | None | 6000 × 32 × 16 = 3,072,000 |
| **Total** | **412,876,800** | **316,416,000** |

This saves 23.36% of head MACs. Including the current stem and ten blocks, it saves approximately 3.80% of decoder convolution/linear MACs. It adds 192,000 intermediate PReLU values and 48,000 tanh values per second. The first PReLU shrinks from 204,800 to 192,000 values per second. The rearrangement produces 192,000 intermediate feature values per second.

Current head convolution/linear weights total 4,128,768. Proposed weights total 2,952,704: 2,949,120 in the low-rate projection, 3,072 in the shared local convolution, and 512 in the synthesis kernel. Include biases and PReLU parameters separately according to implementation. In particular, the final synthesis must not add a per-input-frame bias before overlap; an optional waveform bias is added exactly once per output sample.

Lower MACs do not establish equal CPU latency. Additional short convolutions, a feature rearrangement, activation implementations and dispatch may erase the savings. Any statement of no CPU overhead requires an optimized one-thread implementation and matched 80/160 ms streaming and batch measurements. This budget makes CPU neutrality plausible, not guaranteed.

## Causal streaming definition

Let `v[n,c]` be the refined 6 kHz features. For sample position `r` from 0 through 7:

`y[8*n+r] = sum_c (K[c,r]*v[n,c] + K[c,r+8]*v[n-1,c])`.

This reads no future feature. Its implementation emits eight samples immediately and retains eight accumulated tail values. The preceding kernel-3 local convolution retains two 32-channel feature frames. Additional state is therefore 64 + 8 = 72 floats per batch item, beyond the unchanged 512 × 2 history of the low-rate head convolution. The local refinement and shared synthesis together use up to four 6 kHz feature positions, corresponding to a roughly 32-sample local span. This is in addition to the preserved backbone receptive field.

For each 40 ms encoder latent, four internal frames become 240 intermediate frames and exactly 1,920 output samples. Do not append a synthesis tail at the end: that tail predicts samples beyond the requested latent horizon. A consistent batch crop and streaming output count are mandatory. Empty calls must preserve all state. New-layer startup histories must be defined explicitly and used identically in training, batch and streaming; do not silently inherit centered or symmetric convolution padding.

The added local stages also extend the complete receptive field. The existing model needs 116 past 100 Hz frames. The new local convolution and synthesis together can read three past 6 kHz feature positions. For positions 0, 1 and 2 within a 100 Hz frame, those positions belong to the previous low-rate head frame, whose own computation needs 116 earlier low-rate frames. The worst-case requirement is therefore 117 past 100 Hz frames, or 30 encoder latent frames when whole 40 ms frames supply context. It is not correct to retain the old 29-latent context requirement merely because the backbone is unchanged.

For an interior crop that currently supplies exactly 29 latent frames of context, the first 24 waveform samples can depend on missing history. The existing six-sample exclusion still leaves 18 unsupported scored samples. Evaluation must rebuild the prefix from authenticated full-source latents to provide 30 frames, keeping the original absolute scored interval and mask unchanged, or run continuously from the true source beginning. Genuine source startup follows the declared startup boundary condition and is not an artificial interior-crop truncation. Do not discard another 18 scored samples to conceal this dependency. No evaluation data was changed for this design note.

## Which failures this can and cannot address

| Failure | Mechanistic reason to test this head | What is not guaranteed |
|---|---|---|
| Stationary 10/40 ms residual | Shared local synthesis and jointly adapted features can learn cancellation with nearby feature channels | Periodic features can still produce periodic output. Sharing or overlap alone does not guarantee silence |
| Natural quiet and whispering | Input-dependent nonlinear synthesis can preserve small changing trajectories without a silence gate | No guarantee of lower quiet error; evaluate actual teacher-selected windows and recording-level outcomes |
| Interior peak overshoot | Tanh gives a mathematical output bound while the preceding path learns its compression | A bound does not establish transient fidelity. Adding tanh to the old weights is not a valid migration |
| Laughter, crying, shouting and screaming | Local multichannel synthesis can coordinate rapid adjacent-sample changes before bounding | Reduced global hidden width and shared synthesis could lose important detail |
| Whistling and high-frequency tones | Learned synthesis filters can represent fine waveform structure rather than imposing scalar smoothing | Both polyphase stages can produce artifacts. Check harmonic images and actual high-frequency reconstruction |
| Startup, tails and stream boundaries | Explicit histories and overlap state define a causal exact-length operator | Correct waveform counts do not prove numerical batch/stream agreement |
| Language diversity | The same audio mechanism operates for every language | Language neutrality is not measured equivalence. Retain separate language and expressive controls |

## Why simpler alternatives are weaker

A longer direct 960-sample output with 480-sample causal overlap is cheaper to conceptualize but still gives `y[r]=(W_first[r]+W_tail[r])h` for constant hidden input. The sum can differ across all 480 positions, so overlap does not structurally remove the silence tone. A 1536-wide kernel-3 head with 960 outputs costs 383.3856M MAC/s, but trades nonlinear width for overlap with no exact checkpoint migration.

A linear constant-preserving interpolator from low-rate channels is insufficient for arbitrary broadband speech synthesis. If every constant latent is forced to produce DC, legitimate sustained periodic sounds may also become unrepresentable. Only a particular encoded-silence trajectory is a valid silence anchor; this is not permission to zero all constant latents.

Adding a scalar output filter after the completed waveform repeats an already weak experiment and does not provide the teacher's learned multichannel cancellation mechanism. Lowering the last ConvNeXt block's gain also has demonstrated local reconstruction-versus-peak tradeoffs.

## Initialization and decision

There is no exact function-preserving map from the existing 2048 nonlinear units and arbitrary 480-row readout into this shared narrower head. A channel truncation presented as preserving the checkpoint would be incorrect. Preserve the entire existing model as control; retain the backbone weights in a separate candidate; train the new nonlinear head against frozen teacher outputs before considering backbone adaptation.

The narrower intermediate projection and shared synthesis are real capacity changes. A joint fine-tune of the existing head remains a lower-risk first comparison and costs no new architecture. If the user specifically wants an architectural candidate, the proposal above is more motivated than a scalar filter because it creates the missing local nonlinear multichannel synthesis while allocating less matrix work. Only quality and runtime results can establish whether this is the better tradeoff.

Existing evidence: `outputs/convnext-recovery/head-experiments/report.md`, `outputs/convnext-recovery/optimizer-silence-causal/architecture-analysis.md`, and `outputs/convnext-recovery/optimizer-silence-causal/natural-layer-results.md`. Actual structure: `work/fast-audiovae/experiments/convnext/audiovae_student/model.py`.
