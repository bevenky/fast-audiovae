# Intel serving qualification

The installed, unpublished development wheel reproduces the accepted matrix-only
candidate exactly. See [serving validation](../../docs/intel-serving.md) for the
matched protocol, scope and timings. `result.json` contains waveform/state hashes
and aggregate measurements, with no audio, latent values or model weights.

`graph-qualification.json` records the initial graph-only check using ONNX 1.19.1.
The installed package subsequently regenerated the same pinned graph under ONNX
1.22.0. `runtime-generation-migration.json` records reuse of that unchanged
completed recipe for the Python-only library-lifetime fix. `unload-trace.txt`
preserves the system-call evidence for that fix. The final public load and cache
reuse checks are in `cache-check.json` and `result.json`.

The setup failures and superseded development wheels remain in the working
validation outputs. No published release artifacts were replaced.
