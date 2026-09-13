# Apple GPU numerical repair

[report.md](report.md) explains the divergence and the qualified repair. [selected-candidate.json](selected-candidate.json) records the selected implementation and evidence hashes.

The first upsampler uses a paired matrix projection; later upsamplers retain the V3 projection form. Original weights, state schema, causal dependencies and validation tolerances remain unchanged. Both repaired variants passed the short state/waveform panel. The selected variant adds the 19 pointwise matrix replacements. CPU remains the default. Public-loader integration is separate.

These are retained workspace source snapshots using the V2 and V3 harnesses and verified local model assets. Failed and diagnostic attempts remain included. Interleaved projection has CPU algebra checks only; it and retain_first were not run on the GPU. No audio, model weights or raw state values are included.
