# Numerical divergence: English and Expressive0

The first changed computation is stage 0 transpose, not history publication. Both same-input alternatives introduce small FP32 differences there. The first English tolerance crossing occurs later in the unchanged stage 0 residual pointwise operation, before stage 1 transpose. No individual local transpose check fails the existing tolerance.

Evidence: two completed four-packet traces, 7 latent frames each, partition `[2,1,2,2]`. Each contains 892 aggregate tensor comparisons, 26 independently evolved histories per arm, and four separate compiled-V3 public calls. All 96 same-input transpose comparisons pass; no model/GPU work was run for this analysis.

| Fixture / route | First bitwise difference | First tolerance crossing |
|---|---|---|
| English, eager V3 | Packet 0, stage0.transpose; max 2.4438e-6 | Packet 3, stage0.residual2.pointwise; 2 elements, ratio 1.1293 |
| English, paired | Packet 0, stage0.transpose; max 1.6689e-6 | Packet 3, stage2 transpose **input**; 1 element, ratio 1.3938; all saved histories pass |
| Expressive0, eager V3 | Packet 0, stage0.transpose; max 2.2650e-6 | Packet 3, stage5.residual2.residual_add; 1 element, ratio 1.2424; all saved histories pass |
| Expressive0, paired | Packet 0, stage0.transpose; max 1.1921e-6 | Packet 3, stage5.residual2.history; 1 element, ratio 1.1482 |

V3 has a full propagated operation trace; the paired arm has transpose, incoming-transpose, history and audio checkpoints. Therefore its first reported crossing is the first **observed** boundary, not necessarily its earliest internal operation crossing. Packet 3 begins at latent frame 5.

## Does compilation cause the failure?

No: eager V3 already fails English `stage1.residual2.history` on packet 3 (max 4.0889e-5, one element, ratio 1.2155). Compiled V3 fails the same history (max 4.3422e-5, one element, ratio 1.2908). Its worst reference error is slightly larger, but compilation is unnecessary for the failure. Both compiled and eager V3 pass all Expressive0 saved histories.

Compiled and eager-V3 reference-error statistics differ for 92 of 104 histories per fixture, starting at stage0.residual1.history. These are comparisons against the original eager reference, not direct compiled-versus-eager-V3 tensor comparisons. They cannot locate or measure a compiler-specific added error. All traced final audio comparisons pass. Internal diagnostic crossings do not replace or weaken the existing waveform/state acceptance gates.

## Narrow repair targets

1. **Paired projection only at stage 0; V3 two-dot form at stages 1–5.** This preserves matrix fast paths. Paired has lower same-input local RMS in all 8 stage-0 checks, while V3 has lower RMS in all 40 later-stage checks. The all-paired expressive state regression therefore gives no reason to use paired projection everywhere. The hybrid remains untested.
2. **Original transpose at stage 0; V3 at stages 1–5**, if the hybrid fails. This removes the first rounding perturbation exactly and retains five optimized stages. It directly targets the English divergence that is already amplified before stage 1. It is also untested; no claim that later local perturbations cannot cross the state gate.
3. If a new trace first crosses after stage 1 despite an exact stage-0 path, add only stage 1 to that original-transpose fallback. Do not begin by replacing the failing downstream depthwise/history copy: it stores already-divergent pre-Snake input. Avoid a stage-5-only fallback inferred solely from the late expressive symptom; upstream propagated errors already exceed local stage-5 errors.

Mean same-input local RMS against **original FP32**, across four packets (not FP64 accuracy):

| Fixture | Stage | V3 RMS | Paired RMS | Paired lower / 4 |
|---|---:|---:|---:|---:|
| English | 0 | 2.2384e-07 | 1.0927e-07 | 4 |
| English | 1 | 1.3929e-07 | 2.0936e-07 | 0 |
| English | 2 | 6.5692e-08 | 9.4602e-08 | 0 |
| English | 3 | 4.5541e-08 | 6.5178e-08 | 0 |
| English | 4 | 1.6879e-08 | 2.3440e-08 | 0 |
| English | 5 | 1.1947e-08 | 1.6484e-08 | 0 |
| Expressive0 | 0 | 2.2324e-07 | 1.1308e-07 | 4 |
| Expressive0 | 1 | 1.2990e-07 | 2.0339e-07 | 0 |
| Expressive0 | 2 | 7.5408e-08 | 1.0864e-07 | 0 |
| Expressive0 | 3 | 6.0156e-08 | 8.4936e-08 | 0 |
| Expressive0 | 4 | 4.8075e-08 | 6.8220e-08 | 0 |
| Expressive0 | 5 | 4.9763e-08 | 7.0198e-08 | 0 |

For expressive packet 3, paired stage5 transpose propagated max error is 1.4320e-5, versus its worst same-input local error across this trace of 2.3842e-6. Its stage4.residual2.history has already reached 2.7776e-5. The final failing history is therefore not evidence of a isolated stage-5 implementation defect.

These data establish changed reduction grouping and propagation relative to the accepted FP32 path, not true mathematical accuracy. All local gates pass; FP64 evidence would be needed before calling one algebraic form more accurate. Small output error is compatible with larger internal errors because later transforms can attenuate or cancel them.

Saved inputs:
- [trace-english-r1.json](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/trace-english-r1.json), SHA256 `c7eec467aa9c14bd8ca2fe2cd85fa21b4d5f1bf3739f408da50a52e0dba621d6`.
- [trace-expressive-r1.json](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/trace-expressive-r1.json), SHA256 `625c0db3481a0f4ec29e906c3b4bb4306f462e87f42f38a15279b7056a205a15`.

Source: [trace_divergence.py](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/trace_divergence.py), [transpose forms](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/candidates.py), [legacy residual/history semantics](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v2/mps_decoder_before.py:77).
