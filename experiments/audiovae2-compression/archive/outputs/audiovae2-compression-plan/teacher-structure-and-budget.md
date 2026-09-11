# Teacher structure and grouped compression budget

This is a static source/arithmetic audit. No checkpoint was loaded, no neural operation or benchmark ran, and no existing model or experiment source changed. The [reproducible analysis](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/audiovae2-compression-audit/static_budget.py) reads frozen Python syntax and saved costs using the standard library only. Its [JSON output](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/audiovae2-compression-plan/static-budget.json) records both input hashes and all formulas.

The objective for a compressed group is **`S_group(x) ≈ T_group(x)` for the same input and aligned complete group output**. A student with fewer units learns the combined teacher mapping. Surviving student units need not reproduce individual teacher units. Copying their weights only provides an initialization; an untrained removal test cannot decide whether the smaller trained group works.

## Exact teacher inventory

The decoder has 64-channel 25 Hz latent input and 48 kHz waveform output. It starts with a causal depthwise kernel seven and a 64-to-2048 pointwise projection, followed by six stages. Every stage has sample-rate affine conditioning, Snake, a transposed convolution, then three residual units. Each unit is Snake → depthwise kernel seven → Snake → dense pointwise projection → full-width residual addition, with dilations 1, 3 and 9. The tail is Snake → 32-to-1 causal waveform convolution, kernel seven → tanh. See [frozen teacher, residual units](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:75), [stage construction](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:176), and [decoder construction](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:270).

| Stage/module | Input channels/rate | Output channels/rate | Upsampling GMAC/s | Residual GMAC/s | Total GMAC/s |
|---|---|---|---:|---:|---:|
| 1 / `decoder.model.2` | 2048 / 25 Hz | 1024 / 200 Hz | 0.838861 | 0.633446 | 1.472307 |
| 2 / `decoder.model.3` | 1024 / 200 Hz | 512 / 1200 Hz | 1.258291 | 0.956621 | 2.214912 |
| 3 / `decoder.model.4` | 512 / 1200 Hz | 256 / 6000 Hz | 1.572864 | 1.211904 | 2.784768 |
| 4 / `decoder.model.5` | 256 / 6000 Hz | 128 / 12000 Hz | 0.786432 | 0.622080 | 1.408512 |
| 5 / `decoder.model.6` | 128 / 12000 Hz | 64 / 24000 Hz | 0.393216 | 0.327168 | 0.720384 |
| 6 / `decoder.model.7` | 64 / 24000 Hz | 32 / 48000 Hz | 0.196608 | 0.179712 | 0.376320 |

The complete decoder is **8.9912432 GMAC per audio second**, matching the [saved architecture audit](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/intel-iteration2/data/architecture-costs.json). Upsampling is **56.12%**, residual pointwise matrices **42.09%**, residual depthwise kernels **1.63%**, and initial/final convolutions **0.16%**. Counting only depthwise MACs would miss the main matrix budget. Conversely, these MACs exclude nonlinearities and memory work: the model also evaluates Snake on **48,793,600 feature values per audio second**.

## Three different meanings of compression

**A. Fewer residual units, unchanged individual stage interfaces.** Replace each complete three-unit stack by two jointly adapted units, or one. All surviving weights can be copied directly. Removing the middle unit and starting with dilations 1 and 9 is a concrete initialization, not a proven optimal choice. Strides and widths stay fixed, but history shrinks. Three-to-two across all six stages saves 14.57% whole-decoder MACs and 28.54% Snake evaluations. Three-to-one across all stages saves 29.15% MACs and 57.08% Snake evaluations. Therefore depth reduction alone cannot honestly be advertised as a 50% arithmetic reduction.

**B. Smaller internals inside each residual unit, unchanged individual stage interfaces.** Two useful alternatives exist. A rank-`r` factorization of the dense `C×C` projection costs `2Cr`, so `r=C/2` saves no matrix MACs; `r=C/4` halves those MACs. Alternatively, because the first Snake/depthwise/Snake path is channelwise, retain only `r` branch channels, project `r→C`, and keep the full-`C` identity skip. This costs `r(C+7)` instead of `C(C+7)` and reduces both Snake passes. It needs a channel-selection-aware implementation. Neither alternative requires the stage's external width to change.

**C. Change widths only inside a larger group with fixed outer boundaries.** Individual internal stages may become narrower while the complete group's input and output remain identical to the teacher. This is compatible with grouped distillation. It is different from shrinking every stage boundary throughout the decoder, which would also change the final head input and is outside the initial proposal.

For all weight slicing/factorization, first materialize the teacher's effective weight. Slicing `weight_v` columns while retaining old `weight_g` incorrectly rescales the matrix. Initialize new normalization parameters from the sliced effective matrix, with explicit zero-row handling. Depthwise selected rows, corresponding Snake alphas and untouched biases have direct inheritance mappings.

## First material group candidate

Treat **stages 2 through 4 as one box**. Its teacher input is `[B,1024,T]` at 200 Hz, and output is `[B,128,60T]` at 12 kHz. The input is the output of `decoder.model.2`, before stage-2 conditioning; the output is `decoder.model.5` after its entire residual stack. Stage 2's conditioning can remain unchanged, while conditioning vectors at narrowed internal boundaries must use the same selected channel coordinates.

| Internal stage | Teacher channels | Proposed student channels | Teacher units | Student units |
|---|---|---|---:|---:|
| 2 | 1024→512 | 1024→256 | 3 | 3 |
| 3 | 512→256 | 256→128 | 3 | 3 |
| 4 | 256→128 | 128→128 | 3 | 3 |

The primary student keeps **all nine residual units inside this box**, including the teacher's dilations and temporal support, while narrowing the internal representations. Keep the encoder, initial decoder stem, stage 1, stages 5 and 6, final waveform convolution and tanh unchanged. A later depth-reduction candidate uses six jointly adapted units, but that is a separate decision after evaluating the width-only group.

| Candidate | Whole-decoder GMAC/s | MAC reduction | Snake-element reduction |
|---|---:|---:|---:|
| Teacher | 8.9912432 | 0% | 0% |
| Depth-only control: box retains original widths, two units/stage | 8.0610416 | 10.35% | 15.11% |
| **Primary: narrower box, three units/stage** | **5.1741296** | **42.45%** | **15.42%** |
| Later depth reduction: narrower box, two units/stage | 4.7822960 | 46.81% | 26.13% |
| More aggressive bound: narrower box, one unit/stage | 4.3904624 | 51.17% | 36.83% |

The primary candidate reduces the box's own MACs by **59.57%**, creating a material opportunity while retaining the teacher's temporal support, high-rate tail and bounded output. The later six-unit candidate reduces whole-decoder MACs another **7.57% relative to the primary candidate**, but shortens temporal support; it is not part of the first change. If that later experiment starts from an accepted width-only student, its target remains the **original full AudioVAE2 teacher**, not the compressed student's output.

These are architecture budgets, **not measured CPU speedups or evidence of retained quality**. Existing fused kernels assume particular widths and three-unit stacks; exporting a smaller model without adapting dispatch/packing can forfeit part of the gain. Keeping the depth change separate makes its incremental quality and cost effects interpretable.

## Boundary and streaming contract

- Each transposed convolution has kernel `2s`, stride `s`, and right trim `s`, so it returns exactly `sT` frames. Output position `n` depends on input `floor(n/s)` and the preceding input frame, with no future input requirement. Preserve these phase/trim conventions.
- Every residual depthwise convolution has left zero padding `6d`; the original and primary student stacks need 78 previous stage-rate frames. A later two-unit stack with dilations 1 and 9 would need 60, reducing temporal support without adding lookahead.
- The teacher and primary student stage-2-to-4 boxes both need at most **19 preceding 200 Hz input frames** by exact support arithmetic. Their whole decoders both need at most **20 preceding latent frames**. The later two-unit box would need 15 preceding group-input frames and 19 preceding latent frames overall. Use the original teacher's full context for target generation and training slices. These are receptive-support bounds, not added-delay or time-to-first-audio measurements.
- Preserve zero-filled startup, per-layer streaming states, 1920 output samples per latent frame, the same conditioning value, and exact crop coordinates. A shape match alone is insufficient evidence of streaming parity. Module prehooks receive already conditioned input; group boundary capture must not accidentally apply conditioning twice.
- For training, capture one authenticated teacher prefix output and the teacher's complete group output for the same full-context source. Feed that same prefix to the student group. Match group outputs directly in their shared 128-channel coordinate system, and propagate waveform losses through the unchanged downstream decoder. Freeze downstream weights without wrapping its student path in `no_grad`.
- Train the group jointly. Teacher-forcing every internal student stage independently would conceal the distribution change generated by the preceding compressed stage. Internal per-layer matching is optional diagnostic evidence, not the required optimization target.

The final convolution and tanh remain intact in the proposed main candidate. The first question is whether the smaller group can match its teacher boundary and downstream waveform, including real quiet input, nonverbals and peaks. The static audit establishes the compression opportunity and exact interfaces; it does not settle that quality question.
