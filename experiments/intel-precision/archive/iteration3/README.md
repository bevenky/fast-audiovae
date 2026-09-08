# Rejected Intel iteration3 experiments

This archive preserves later optimization work without changing the accepted selective INT8 implementation. Seven complete-decoder screens passed their 67 correctness checks each, but none reached the required 10% time reduction. The best aggregate reduction was 4.54%. See the [paired results](../../../../benchmarks/intel-precision/iteration3/results.md) and [raw evidence](../../../../benchmarks/intel-precision/iteration3/summary.json).

`source/` retains the original workspace/pipeline implementation, direct and wider VNNI variants, guarded quantization experiments, generated exact SLEEF sine headers and final DW/post-Snake fusion. Candidate directories are separate snapshots. Their existing builders and launchers retain historical dependency paths; they are research records, not a portable installation recipe.

`source-manifest.json` identifies each original source and its digest. C/C++ and Python files retain their source bytes. JSON host paths may be normalized, with both digests recorded. The exact generated SLEEF headers use the included SLEEF license; other dependency notices are in the parent package.

The raw evidence manifest also names omitted disassembly dumps and numerical fixtures by hash. No models, audio, binaries, NPZ arrays or duplicate tar archives are included. The small native and row-check programs are source only. No archived candidate is selected by the runtime.
