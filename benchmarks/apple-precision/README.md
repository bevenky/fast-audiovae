# Apple precision evidence

The existing Apple FP32 path is retained. The final SME2 screen was promising but did not establish a reliable 10% speed improvement. Full Apple INT8 perceptual testing was not performed.

- `sdot/`: initial slower port, native checks, matched screen and interrupted later validation.
- `sme2/`: final native/protocol checks, exact build/graph identities and completed three-clip screen.
- `diagnostics/`: separate instrumented SDOT profile with original node details and traces. Worker times overlap.
- `retained-fp32/`: completed fresh 60-clip stock/FP32 waveform audit with 156 checks and zero failures; fresh scoring completed all 12 metrics with zero errors.
- `publication.json`: source identities, original and normalized evidence hashes, and publication limits.

Private paths and background process names were normalized. All numerical timing/check results and artifact digests are retained. Saved configs reference omitted models, weights, binaries and prepared cases; they are audit records, not runnable downloads. [Build instructions](../../experiments/apple-precision/README.md) and [interpretation](../../docs/apple-precision.md) explain how to use the source and read the results.
