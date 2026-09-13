# Compiled matrix candidate: source and accounting audit

Read-only inspection of saved generated code and `transpose-r1.json`. No GPU tests or benchmarks were run. [Full machine-readable evidence](compiled-matrix-audit.json) includes both cache hashes, every operation shape, source lines and all twelve paired observations.

## Actual generated routes

| Packet | Remaining external convolutions | External MM calls | Generated Metal calls | Distinct generated kernels |
|---|---:|---:|---:|---:|
| 40 ms | 20 | 25 | 84 | 71 |
| 80 ms | 20 | 25 | 85 | 72 |

All six transpose convolutions and all 19 pointwise convolutions are replaced by matrix multiplication. The remaining convolutions are exactly **19 depthwise plus one final ordinary K7 convolution**. All 45 operation shapes and routes match the original topology at each packet size.

The original wrapper had 45 external convolutions plus 83/84 generated Metal calls. The candidate has 20 convolutions plus 25 MM calls plus 84/85 generated Metal calls. It therefore does **not** save 25 total math calls or reduce the logical multiplication count. The logical totals remain 400,978,368/760,628,096 MACs. This supports an explanation involving different convolution-versus-MM dispatch and generated layout/assembly, without isolating their individual contributions.

## Matrix shapes and layout

All matrices use weight-left `A @ B`, with reduction dimension Cin and contiguous output rows over time. Pointwise inputs are BCT views; there is no activation-axis copy for the MM input. The stem is `[2048,64] @ [64,1/2]`; the six residual groups use `[C,C] @ [C,T]` at C/T pairs 1024/(8,16), 512/(48,96), 256/(240,480), 128/(480,960), 64/(960,1920), and 32/(1920,3840), three calls per group.

| Transpose stage | Packed A | B, 40 ms | B, 80 ms |
|---|---|---|---|
| stage0.transpose | [16384, 2048] | [2048, 2] | [2048, 3] |
| stage1.transpose | [6144, 1024] | [1024, 9] | [1024, 17] |
| stage2.transpose | [2560, 512] | [512, 49] | [512, 97] |
| stage3.transpose | [512, 256] | [256, 241] | [256, 481] |
| stage4.transpose | [256, 128] | [128, 481] | [128, 961] |
| stage5.transpose | [128, 64] | [64, 961] | [64, 1921] |

The transpose result is phase-major `[Cout,2s,T+1]`. Generated kernels index its current/previous phases directly, then perform bias, Snake and history concatenation together. In the 40 ms wrapper, generated kernel 10 (line 209) demonstrates that fusion; kernel 13 (line 299) combines phase sum, bias and residual addition. The wrapper does not materialize a standalone BCT transpose result just to pass it to the first depthwise stage. This is source evidence about generated assembly, not a measured GPU duration.

Packed weights add 165,314,560 bytes alongside retained original buffers, prepared once before compilation/timing. Original activated-input state semantics remain unchanged. `reinterpret_tensor` calls are views, not proof of data copies. Allocation counts and alignment guards likewise do not establish actual memory traffic or backend scratch usage.

## Recomputed matched timing

Each row below has six measured streams per arm: three 960 ms clips, two repetitions, 5.76 seconds of emitted audio per arm. All packet and flush durations were summed independently and agree with the saved RTF summary.

| Packet | Original compiled RTF | Combined RTF | Pooled reduction | Median paired reduction | Wins |
|---|---:|---:|---:|---:|---:|
| 40 ms | 0.122989474 | 0.059955960 | 51.2511% | 51.7000% | 6/6 |
| 80 ms | 0.065275318 | 0.031657683 | 51.5013% | 50.7430% | 6/6 |

These are within-run completed public-API comparisons. They support a large improvement for this short candidate screen, not statistical significance, sustained performance or exclusive GPU attribution. The matched generated wrappers were identified by their complete shapes and source structure; the original timing receipt did not pin their cache filenames. Their newly recorded hashes establish this audit snapshot, not a retroactive execution trace.

## Cache evidence

- 40 ms: `outputs/apple-gpu-v2/inductor-cache/cd/ccdusqpxcjnjtuyl6c43jyq6hzoymzjmkrhkfbexpvu3jmwcgrnd.py`
  SHA256 `c5c44918dd7d3edbe75d2bea92ca7c3c2b0dde4a7e923cc22af957dedd1e8079`
- 80 ms: `outputs/apple-gpu-v2/inductor-cache/me/cmexqr66vmhux3jze6n7bt3zs2slu7hvgr7fj4hm7tiqsqmpufxx.py`
  SHA256 `249031b583e0a35bab57d88abd22ec7ca231d3d9bdc9f6b7acdcb59109bd6e3d`

The screen source SHA matches the recorded receipt, and the saved compiler counters are unchanged after preparation. No existing runner or receipt was modified.
