# Accepted AMD one-thread streaming closure

These five source/header files are byte copies of the accepted AMD experiments,
recorded individually in `sources.json`. Their source paths there describe
provenance only; installation does not import or compile experiment directories.

- `pair/`: the first two INT8 projections share preparation; unchanged integer
  reductions for lengths1–4, existing AOCL core fallback for longer inputs.
- `history/`: six early DW/Snake chains retain the existing raw input history.
  The wrapper includes the unchanged pinned `native/x86/native_kernels.c`, using
  the same static SLEEF archive, compiler and arithmetic flags as the base build.
- `phase/`: first-stage ordered `(current + previous) + bias` with unchanged
  cached previous-projection state. Empty input copies state.

`tools/build_amd_streaming.py` builds these libraries without loading them. It
adds no system dependency installation. Runtime dependencies use leaf SONAMEs
and `$ORIGIN`; the existing canonical bundle supplies AOCL and its CPU runtime.
The existing SLEEF/AOCL notices remain part of the canonical x86 payload.

`recipes/amd_streaming.py` verifies canonical stream `8bb688f8…`, applies only
the accepted pair, six histories and first phase, and requires final `7fc62af2…`.
All initializers, graph endpoints and18 histories/683264bytes remain unchanged.
The canonical full/batch graph remains available. This route is selected only
for AMD, streaming mode and one requested worker by the outer recipe dispatcher.

Prebuilt payload roles are `amd_stream_pair`, `amd_stream_history` and
`amd_stream_phase`. Missing roles are rejected. No unqualified Snake, reordered
matrix arithmetic, activated-history representation or later candidate belongs
to this closure. Newly linked binaries still require target-host qualification.
