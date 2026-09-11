# Archived target verification

These scripts perform no training, student inference, automatic repair or cache overwrite. Run them explicitly in the qualified PyTorch environment with cuDNN 9.25.1. Both take the existing exclusive training lock and require a new output directory. They pin the historical source plan, 15.5 GB target cache, source implementation, checkpoint and complete teacher state.

Start with the bounded source check:

```sh
python audit_encoder_sample.py --encoder-sample 32 --out /tmp/conditional-source-sample-v1
```

It selects the first 32 distinct source IDs in the original gradient-calibration pool order. It verifies each full prepared source file, exact sample count and unchanged-gain reader policy, then checks bitwise agreement with the archived crop's input reference. It calls the pinned model's original singleton encoder with corrected cuDNN enabled, compares real cached latent frames at their absolute context offsets, and reports all 64 channels. Separately, it decodes the archived latents and compares their archived waveform targets. Latent tolerances are max absolute error 2e-5 and RMS 2e-6. A sampled failure is evidence about that source, not an estimate of all affected sources.

The exhaustive conditional-pair check is separate:

```sh
python validate_conditional_targets.py --out /tmp/conditional-targets-v1
```

It verifies all 17,328 unique training crops from 4,749 sources across four original pools, plus all 285 historical heldout crops. Duplicate windows must have identical geometry and tensor hashes. Equal-length batches contain at most 32 crops, with no added batch padding. Each encountered batch shape has a singleton comparison sentinel. The first eight numerical failures also receive a singleton diagnostic re-decode; these checks never turn a failed batch comparison into a pass.

Only valid scored samples enter comparisons, retaining the historical exclusion of six initial scored samples for crops whose context starts inside a recording. Every crop must satisfy max absolute error <=1e-5 and RMS <=1e-6. Undefined, nonfinite or missing results fail. Results include per-crop JSONL and pooled sample-weighted RMS, maximum errors and failure counts by pool, language and condition. No gain fit, alignment, clipping or relative tolerance is applied.

The pinned teacher decoder's reset influence ends before output sample 36,750. Starting with its latent kernel7, propagate `L=6`; each stride `s` adds `L=(L+1)*s+78` for the transpose convolution and three kernel7 residuals at dilation 1, 3 and 9. Strides are 8, 6, 5, 2, 2 and 2; the final kernel7 adds six. Twenty latent frames cover that support. Interior training crops contain 30 real context frames, while historical heldout crops normally contain 29. The code checks each actual crop's scored start against the support bound; it never fabricates context at recording startup.

A conditional-pair pass establishes that the archived targets remain T(z) for the archived latents. It does not establish that those latents equal the corrected encoder's output. The source sample answers that separate question. Both historical and canonical heldout gates remain required before any continuation decision.
