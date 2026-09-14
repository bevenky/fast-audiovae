# A2: raw-history Snake → DW7 → Snake

Prepared experiment, not a production route. No Linux build, numerical run or timing was performed during local preparation. Python sources passed AST parsing.

The control is the existing `SnakeDW7SnakeF32` plus its exact streaming `Concat`, next-history `Slice` and output `Slice`. The candidate exposes `[x, raw_history, W, bias, alpha_pre, reciprocal_pre, alpha_post, reciprocal_post]` and returns `[new_output, next_raw_history]`. History remains chronological raw pre-Snake input, with shape `[1,C,6*d]`. T=0 copies history and returns empty output. T smaller than the halo shifts the old tail and appends only the new raw input.

The appended C entry point reuses the original source's `snake_tile`, `choose_dw`, ordered tap loop, SLEEF AVX512 sine and tile256. It applies pre-Snake to the required history, then evaluates DW and post-Snake only for new samples. It does not replace the trained formula, reciprocal, coefficients, state representation or residual consumer. Local transformed scratch is bounded by 310 floats plus one256-float DW tile. All scratch is call-local; coefficients are immutable. The ORT bridge preserves the original channel `ParallelFor` and `row_batches`; each C invocation has one thread and no OpenMP.

At C1024/d9, the old region computes62/70 DW and post-Snake positions for T8/T16. A2 computes8/16, but still pre-activates54 history positions, adds a separate history pre-Snake call, and updates raw history. The position reduction is not a predicted decoder speedup. Splitting SIMD call groups can change low bits even with the same polynomial; the original `atol=1e-5, rtol=1e-4` gate is retained, and bitwise status is reported separately. Raw next-history must be byte-identical.

## Build and run

Run on the coordinated Linux x86 CPU host, using the same compiler, SLEEF archive and pinned API29 headers as the original base build:

```sh
python a2/build.py --repo REPO --baseline-build BASE_BUILD_JSON \
  --sleef-prefix SAME_SLEEF_PREFIX --ort-include PINNED_API29_INCLUDE \
  --output NEW_BUILD_DIRECTORY

python a2/screen.py --source EMBEDDED_FP32_STREAM_GRAPH --source-sha GRAPH_SHA \
  --build NEW_BUILD_DIRECTORY/build.json --build-sha BUILD_JSON_SHA \
  --node EXACT_C1024_D9_TRIPLE_NODE --ort-version 1.30.0 --threads 1 \
  --timing --output NEW_RUN_DIRECTORY
```

Every output directory is create-exclusive. Failed compiler output or a failed numerical/cap gate remains in its JSON. The build does not load a model or run native probes. It authenticates source pins, original library, headers, SLEEF and compiler/flag equality, then links with `-Bsymbolic` to isolate the candidate's copy of the C core.

The screen validates all six actual C1024/C512 × d1/3/9 regions before any timing. Twelve patterns per region cover empty, single, three-frame tail, 40/80ms shapes, halo−1/halo/halo+1, tile257, zero, quiet and large alternating finite values. Each compares the original wrapper and candidate and checks exact raw next-history and input preservation. Both arms also receive uneven partition, future perturbation and unrelated interleaved-stream checks:72 numerical cases and12 state-check records total. It uses genuine model coefficients but synthetic region inputs and history; zero input is not an encoded silence claim.

Optional timing is restricted to the one C1024/d9 region at T8/T16, two warmups and three interleaved paired trials per shape. Three pairs have order AB, BA, AB; the small order imbalance is reported, not hidden. Timings include `Session.run`, allocation, native dispatch and raw-state copy. Load, metadata queries, reference comparisons and session construction are excluded. All regional calls, including math and warmups, share a2-second cumulative call cap. No full codec is run. Four ORT threads are a separate explicitly requested invocation, not an automatic expansion.

`rewrite.py --source GRAPH --source-sha SHA --node NAME --output NEW_DIRECTORY` prepares only that one region. It proves unchanged external input/output schemas, every initializer and every other node. It does not register libraries, modify a bundle, select winners or certify that unrelated source nodes execute FP32. The caller must supply the independently authenticated FP32 control and retain its complete precision/library manifest. Root owns any later regional qualification, selection and full-bundle integration.
