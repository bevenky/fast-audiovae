# Intel precision evidence

These are saved CPU results. Packaging them did not execute a model or collect new timings. The concise interpretation is in [Intel precision results](../../docs/intel-precision.md).

- `summary.json` combines the completed 60-clip validation, 250 timing calls, automated quality and listening summary.
- `campaign-results.json` and `campaign-config.json` retain individual gates, timings, graph/library/case hashes and the frozen protocol.
- `quality-results.json` and `quality/` retain automated per-clip scores, predictor provenance and paired comparisons. Historical pre-listening fields remain as recorded; `human-ratings.json` is the later listening analysis.
- `human-ratings.json` and `human-ratings-capture.json` retain the eight-clip, single-listener pilot and explicit unrated observations. Audio is omitted.
- `profile-summary.json` and `profiles/` retain the separate representative operator and native-counter profiles. Worker times can overlap and are not decoder wall-time shares. Profiled timings are not the headline RTF.
- `iteration3/` retains seven rejected paired screens, native checks, build records and diagnostics. The required decoder-time improvement was 10%.

`publication.json` maps every copied result to its original SHA256 and published SHA256. JSON host path prefixes may be normalized to symbolic placeholders; measurements and artifact hashes remain unchanged. A record's embedded source hashes still identify the original files, not their normalized copies. Models, audio, predictor assets, compiled libraries and large numerical fixtures are omitted.
