# Source and dependency notices

The experiment sources and copied fast-audiovae support sources are provided under the repository's Apache License 2.0, reproduced in `LICENSE`. Existing source notices are retained. `source-map.json` identifies unchanged copies and packaging adapters. No model architecture implementation, model weights or dataset recordings are distributed here.

The build uses these separately supplied dependencies:

| Dependency | Version/source | Terms included |
|---|---|---|
| ONNX Runtime | 1.29.0, Microsoft | MIT, `licenses/onnxruntime-LICENSE` |
| LIBXSMM | commit `55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1` | BSD 3-Clause, `licenses/libxsmm-LICENSE.md` |
| SLEEF, through the FP32 native library | 3.9.0 | Boost Software License 1.0, `licenses/sleef-LICENSE.txt` |
| oneMKL | 2026.1.0, Intel | Intel Simplified Software License and vendor third-party notices, `licenses/onemkl/` |

The accepted implementation requires separately supplied dependency libraries and headers. The rejected iteration3 archive additionally preserves generated SLEEF 3.9.0 inline headers under their original Boost Software License, reproduced at `archive/iteration3/SLEEF-LICENSE.txt`. No dependency binaries are bundled. These notices do not relicense dependencies. The oneMKL notice files come from the official `onemkl-license` 2026.1.0 package; its package and file hashes are preserved in `licenses/onemkl/provenance.json`. Follow each dependency's distribution terms when assembling binaries.

ONNX and NumPy are Python tooling dependencies installed by the user. Model and dataset licenses continue to apply to any separately supplied graphs, weights and evaluation audio.
