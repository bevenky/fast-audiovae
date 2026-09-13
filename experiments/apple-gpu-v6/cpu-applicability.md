# Does the final GPU matrix rewrite apply to Apple CPU?

The selected CPU implementation already replaces the 19 pointwise convolutions and six transposed convolutions with matrix operations. Its six upsamplers use two phase-projection matrices each, for 31 matrices total. Moving those convolutions to matrices is therefore not a new CPU optimization.

The GPU hybrid has one additional difference at the first upsampler: it combines current and prior activated inputs into one reduction, `W[8192,4096] @ X[4096,T]`, instead of two `W[8192,2048]` reductions. This preserves the real-valued formula but changes FP32 accumulation order. Later GPU upsamplers use separate phase reductions, which is already the CPU approach.

| Detail | Selected CPU 5918 | Final GPU hybrid |
| --- | --- | --- |
| First current projection | Persistent N64-packed LIBXSMM SME | Part of one paired matrix |
| First previous projection | Persistent KleidiAI multitile SME | Part of the same paired matrix |
| First history | Projected previous phases, `[1,8192,1]` | Activated input, `[1,2048,1]` |
| Phase assembly | Native ordered sum, bias and layout, updating projected history | Reshape paired projection and add bias, updating activated history |
| Pointwise stages | Already weight-left matrices | Weight-left matrices |

The paired reduction is not a direct replacement under the existing CPU state interface. A production port would need its own state representation and numerical qualification. A CPU benchmark cannot assume the MPS speedup transfers: the current CPU matrices already use persistent weight packing and specialized SME kernels.

The selected CPU recipe supports one and four ORT threads with the same graph. Its custom first-pair nodes retain `threads=1`; the four-thread setting does not automatically split these native kernels. The later opt-in parallel scheduling code present in source is not enabled by the immutable 5918 graph.

## Bounded diagnostic

`cpu_applicability.py` prepares an isolated first-upsample regional comparison against the exact native graph and libraries selected by `outputs/apple-streaming-promotion-v1/config-r3.json`. The candidate uses standard ORT MatMul for the GPU paired formula. Both sides use ORT 1.30 CPU only, one or four intra-op threads and one inter-op thread, with spinning disabled.

The scope is the two real first-projection weights at 40/80 ms (`T=1/2`), with deterministic synthetic signed, quiet and zero activations. Each path evolves its own mathematically corresponding history. Every math output is compared at the existing `atol=1e-5, rtol=1e-4`; inputs and old histories must remain unchanged, and new history must be owned. Per setting: three paired math calls, two warmups per arm and six balanced randomized timing pairs. Total: 88 completed session calls, capped at two seconds of call time. One-time packing and session construction are excluded; all recurring input concatenation, matrix dispatch, phase layout, bias and output/state work remain inside `session.run`.

This tests one readily available CPU backend and formula, not a new specialized paired SME kernel, full decoder performance, arbitrary packet lengths or production compatibility.

## Completed result

The single declared screen completed with 88 calls and 0.306385 seconds of completed session time. All 12 regional-output comparisons passed the unchanged tolerance; maximum absolute difference was `4.8428774e-8`. Inputs and old histories remained unchanged, and next-history ownership checks passed. These are intermediate first-upsampler outputs, not final decoded waveforms. The baseline's projected next-history was checked for shape and finite values; its values were exercised by the subsequent independent-history output comparisons, not separately recomputed by an oracle.

| ORT threads | Packet | Existing CPU mean region time | Paired formula mean region time | Median paired time reduction | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 40 ms | 1.325 ms | 4.747 ms | −255.71% | 0/6 |
| 1 | 80 ms | 1.297 ms | 9.659 ms | −665.17% | 0/6 |
| 4 | 40 ms | 1.287 ms | 4.826 ms | −270.07% | 0/6 |
| 4 | 80 ms | 1.059 ms | 3.055 ms | −192.78% | 0/6 |

Negative reductions mean slower. The four-thread labels describe ORT's intra-op setting; the frozen control's two custom matrix kernels remain internally single-threaded. Corresponding one-thread/four-thread inputs and histories are identical. All pair values are retained; none were removed. The process used CPUExecutionProvider only and authenticated the selected graph, complete native library closure and source files before execution and afterward.

The protocol completed successfully, but every performance gate failed. Retain the selected CPU implementation for both thread settings. No full-decoder experiment followed. This rejects the tested standard-ORT implementation of the paired formula, not every possible specialized native implementation of the same algebra. It also does not establish why a particular CPU backend dispatch was slower, because no hardware-counter or binary-dispatch tracing was performed.

Artifacts: `cpu-applicability-r1.json` (raw checks and all 24 timing pairs), `cpu-applicability-r1.plan.json` (sealed before native calls), `cpu-applicability-r1.log` (process log), and `cpu_applicability.py` (standalone source).

Source pointers:

- `work/fast-audiovae-apple-gpu/src/fast_audiovae/recipes/apple_selected.py`: `MATRICES`, `_rewrite_matrices`, `_rewrite_phases`.
- `work/fast-audiovae-apple-gpu/native/apple/streaming/libxsmm_panel/native.cpp`: exact first-pair guards, persistent N64 packing, serial and opt-in parallel entry points.
- `work/fast-audiovae-apple-gpu/native/apple/streaming/phase/native_ops.cpp`: `FinishRow`, projected history update.
- `outputs/apple-gpu-v4/candidates.py` and `repairs.py`: paired first stage and separate later-stage reductions.
- `outputs/apple-gpu-v4/selected-candidate.json`: selected `hybrid_pointwise` identity and qualification scope.
