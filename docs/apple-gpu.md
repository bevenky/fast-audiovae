# Apple GPU decoding

CPU and streaming remain the defaults. Apple GPU decoding is explicit:

```sh
python -m pip install 'fast-audiovae[gpu]==0.4.0' --find-links https://github.com/bevenky/fast-audiovae/releases/expanded_assets/v0.4.0
```

```python
from fast_audiovae import load

decoder = load(device="gpu")
with decoder.stream() as stream:
    audio = stream.decode_chunk(latents)  # float32 NumPy [1, 64, L]
```

Apple GPU support is included in the v0.4.0 wheels. The optional extra installs PyTorch 2.14.x; ordinary CPU installation does not require it. GPU execution requires Apple Silicon and MPS. GPU requests fail if the device or required operation is unavailable; there is no CPU fallback. CPU worker counts are unrelated to GPU execution, so keep the default `threads=1` with this device.

## Execution

The model uses the pinned, original AudioVAE2 weights in float32 and emits 48 kHz audio. The six upsamplers are expressed as packed matrix projections, and the 19 pointwise convolutions use matrix calls. The first upsampler reduces the current and previous contributions together; later stages project them separately. This combination passed the retained waveform and internal-history checks without changing the tolerance. Snake, conditioning, causal history and the output tanh remain in place.

Streaming load prepares compiled 40/80 ms steps before returning. First compilation can take time; PyTorch's compiler disk cache can be reused on later loads. `decoder.prepare()` can warm those steps again without changing an existing stream. Larger chunks execute the same model eagerly on MPS, avoiding an unbounded set of compiled shapes. `load(device="gpu", mode="batch")` also uses eager MPS with independent zero history.

Each stream retains 647,424 bytes of history on the GPU. Calls include owned input upload, a finite-value check of audio and every history tensor, GPU completion, and an owned NumPy output on the CPU. History advances only after successful completion. Empty calls, flush, reset and independent streams retain the CPU API semantics.

Fast math and CPU fallback must be disabled. `PYTORCH_MPS_PREFER_METAL` must be unset, including an explicit value of `0`, because this PyTorch release tests its presence. Other backends and reduced precision are not qualified by these results.

## Evidence

The final integrated-source comparison and CPU applicability check are retained in [the V6 report](../experiments/apple-gpu-v6/report.md). The comparison uses the actual published AudioVAE2 class, original checkpoint and official stateful streaming API on MPS. Pocket Mimi uses its continuous decoder on MPS. Both references retain their upstream inference behavior.

The shorter historical probes are preserved in [the experiment history](apple-gpu-experiments.md), including rejected candidates. The V4 paced-arrival probe measured optimized 80 ms service time at RTF 0.07531, with no deadline misses in 16 calls. That probe used the experimental implementation and a different session; it is not the integrated back-to-back RTF or a live latency guarantee. Real packet arrival gaps can change GPU service time.

Numerical agreement is separate from a listening study. The README's PESQ/STOI/MOS scores remain the previously measured CPU panel; they were not rerun or relabeled as a GPU quality study.
