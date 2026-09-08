# Apple CPU precision experiments

Use the public Apple FP32 runtime. The final SME2 panels experiment passed its correctness checks but saved only 3.0% of decoder time, below the 10% adoption threshold. Its fresh automated quality scores were slightly lower than FP32. INT8 remains an explicit experiment and is not selected automatically.

| Decoder | Final matched RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.092772 |
| Recommended Apple FP32 | 0.024685 |
| SME2 panels INT8 | 0.023938 |
| Mimi | 0.028114 |

Apple M5 Max, ONNX Runtime 1.29 CPUExecutionProvider, four ORT threads, ten fixed clips, two warmups and five repetitions. Values are equal-clip means. The 95% clip-and-repeat bootstrap interval for time saved was 0.86% to 5.86%. All 1,620 execution checks passed; all 60 full INT8 waveforms and 120 prefixes matched the earlier Apple INT8 implementation bitwise. This is not equality to FP32. See [quality, profiling and limits](../../docs/apple-precision.md) and [compact evidence](../../benchmarks/apple-precision/sme2-panels.json).

[sme2-panels/](sme2-panels/README.md) contains the current experiment's build, rewrite, check and portable campaign instructions. It quantizes the same 22 selected large products, prepares bounded packed time panels and fuses four late regions. Nine smaller products, inputs, outputs, nonlinearities and depthwise convolution remain FP32. The native wrapper checks SME2 support, streaming vector length and floating-point mode. No weights, corpus, audio or binaries are included.

`sdot/` and `sme2/` preserve the earlier experiments. Their separate three-clip screens used different FP32 controls and should not be compared directly with the final campaign. The old [publication manifest](../../benchmarks/apple-precision/publication.json) remains available; the [pinned ORT header hashes](pins/ort.json) and dependency licenses are shared.

For fresh quality scoring, use the exported FLOAT WAV manifest and the matching source manifest with the pinned metric assets:

```sh
python experiments/apple-precision/quality/score_quality.py \
  --scorer-dir experiments/apple-precision/quality/scorer \
  --sources /path/to/source-manifest.json \
  --variant-manifest /path/to/local/quality-inputs.json \
  --baseline fast_fp32 --trim-audiovae2-right-padding \
  --assets /path/to/pinned-metric-assets \
  --output /path/to/new/comparison.json
```

Keep the explicit `--scorer-dir`; the frozen wrapper's default uses its former workspace layout. It verifies scorer and asset hashes and forces CPU scoring. Metrics use the common 16 kHz source bandwidth and are separate from decoder timing. Predictor scores do not replace a listening panel, and no predictor weights are bundled.
