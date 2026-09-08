# AMD source rebuild record

`rebuild-provenance.json` captures the exact source, header closure, compiler commands and output object hashes from the executed AMD builds. `evidence/fused-build-r2/build.json` and `evidence/aocl-build-r2/build.json` retain the original complete records.

The four shared objects can be compiled directly from the frozen release sources. They do not need a oneMKL build first:

- `matrix.o`: `support/matrix/fused_pointwise.c`, C11, `FX_WITH_LIBXSMM=1`.
- `projection.o`: `support/upsample/projection.c`, C11, `UP_WITH_LIBXSMM=1`.
- `stage.o`: `fused/stage_precision.cpp`, C++17.
- `upsample.o`: `fused/upsample_precision.cpp`, C++17.

All four use GCC/G++ 13.3.0 with `-O3 -fPIC -fvisibility=hidden -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror`. Matrix and projection include the pinned LIBXSMM headers. The stage sources include ORT API29 and the supplied native, matrix, stage, upsample and precision headers. The exact include order and commands are in the JSON record. Compile first, then apply the recorded AOCL core and stage link commands. The intermediate oneMKL shared-library links in the historical fused builder are unnecessary for this source-only route.

Dependencies:

- AOCL-DLP commit `c577191304a3db0029f2f12fcacbc8ad296a645d`, official source archive SHA256 `7c93b699d4cb147fb698bb04acde0cbc67947d970cb5bc36623a3579c0dc816a`. The reused library was built with GCC13.3, CMake3.28.3, Release, OpenMP, examples/tests/benchmarks disabled. Its executed library SHA256 is `6fb0e10f2074367c4b038beb14647b8c02cc67116f973d10b4a02f9ca01974b0`.
- LIBXSMM commit `55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1`; the source archive and generated include closure are pinned in the release and build manifest. Executed static archive SHA256 is `58caed0bea06dbfd70edc8b846f08bc42d8656a8a27fcb34005390c3da978f4c`.
- Existing CPU native library uses ABI1, ORT API29, SLEEF3.9.0 and an explicit guarded AVX512 backend. Its exact source/header and library hashes are captured.

A publication builder should parameterize these dependency locations and compile all four objects itself. It must verify actual source/header inputs before and after compilation and record actual object/library hashes. Moving paths or changing compilers does not inherit the historical binary identity automatically. The current experiment adapter deliberately requires the observed AOCL binary hash; a general rebuild path must bind a newly built library to its pinned source/configuration instead of pretending every rebuild has that binary hash.

AOCL can create OpenMP teams, but this adapter sets and verifies a one-worker calling-thread policy for every integer GEMM. Four outer ORT workers own disjoint output rows or time segments. The adapter assumes ownership of the isolated AOCL thread-local policy. It does not provide arbitrary thread-state preservation for a host application already using that same AOCL library.

The exact coefficient graph is unchanged except for 14 matrix shard counts and four stage segment counts changing from two to four. This source-only record is not an additional performance or quality claim.
