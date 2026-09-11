This is a diagnostic-only encoder trace. Run it in a fresh process with the frozen recovery and diagnostic directories on `PYTHONPATH`:

```sh
python trace_encoder.py --out /tmp/encoder-layer-trace-batch-first
```

Startup order is fixed: construct teacher, two B8 warmups, B8 trace, two equally padded B1 warmups, B1 trace. There is no earlier encoder warmup. Row4 is the original source119459; all eight source bytes, ordering and lengths are taken from `cache_probe.py` and authenticated through the existing source loader.

The trace covers encoder leaf convolutions and Snake activations. It retains only the selected row's first2048 time positions of each input/output, and reports full shapes, errors, channel/time distributions and effective weight hashes after the original weight-normalization pre-hook. `fc_logvar` is captured but excluded from the mean-latent first-divergence decision. Hooked outputs must remain within the existing tolerance of unhooked warm outputs or the script stops with a diagnostic failure report.

Outputs are `trace.json` and `captured-prefixes.pt`; existing outputs cannot be overwritten. The capture contains diagnostic activations and final selected-row latents, not source audio or weight tensors. Per-layer hashes read full effective weights but do not retain their arrays.

`--replay-first` optionally adds a64-output reference for the first tolerance failure using the captured inputs and verified effective weights. Eager FP32 stays on the original device; promoted-FP64 functional arithmetic runs on CPU. This changes local execution shape/backend and is explicitly a mathematical diagnostic, not a bitwise gate or production replacement. It executes after the main traces so it cannot alter their first-call order. Leave it off while the parent process is isolating startup/JIT behavior.

Offline helper checks require only NumPy:

```sh
PYTHONPATH=. python -W error -m unittest discover -p test_trace_encoder.py -v
```

No model or GPU operation was executed while preparing these files.
