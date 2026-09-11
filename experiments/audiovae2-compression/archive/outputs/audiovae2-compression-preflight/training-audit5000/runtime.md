# Training runtime and data audit at step 5000

Training reached its requested step 5000 decision point and stopped normally. The frozen-teacher target producer also finished. The displayed curve is therefore no longer receiving updates; neither training nor the producer is stuck in a running process. This read-only audit constructed no model and ran no neural forward.

## Execution and accounting

- All 4000 continuation updates are present, in exact order 1001 through 5000. The checkpoint source ledger exactly matches original 3000 plus the approved 12,000 fresh sources.
- All 15,000 source IDs, audio hashes and parent recording IDs are distinct. There is no calibration/development overlap under those three identifiers. No conditional-reserve source was consumed. Speaker disjointness is not claimed.
- All 40 shard receipts, pair-file hashes, tensor shapes/dtypes/finiteness and crop geometries passed. All 39 newly generated shards also passed per-source provenance/key checks; the first shard is the previously qualified unused-cache seed.
- The saved optimizer contains 90 parameter states, each at step 5000. Learning rate remains 0.00003, betas 0.9/0.99, epsilon 1 e-8 and weight decay 0. All optimizer tensors are finite.
- The fixed loss coefficients remain waveform 1, group feature 0.009304078923434964 and mel 0.0006674012905982311. All milestone frozen-state receipts and final preservation checks passed.
- Every update checks missing/nonfinite gradients and objective values. The 161 logged gradient norms are all positive and finite: median 0.0808, range 0.0194 to 0.918. No runtime error was found in the run log.

## Time and apparent pauses

The continuation took 1709.82 seconds through its last update, approximately 28.5 minutes. Target preparation generated 11,700 new full-source pairs in 1660.22 seconds, about 7.05 sources/s overall. The existing 300-source seed allowed training to start immediately.

The trainer waited at 17 distinct updates for the next immutable shard. There were 19 logged polling waits, each requesting 30 seconds, totaling 570 seconds of requested sleep. No source cursor advanced during a wait. This accounts for roughly one third of continuation wall time. The sampled update duration was 0.277 seconds median and 0.275 seconds mean; these are host-synchronized update samples, not GPU-only timings. Teacher production ran concurrently, so these timings should not be compared directly with a GPU-idle microbenchmark.

## Actual data coverage

The complete run scored 10.20917 hours from 15,000 distinct recordings. The new 12,000 contributed 8.15753 hours. Context samples and right padding were excluded. All 22 scheduled Indic languages and the requested international groups are present. The precise source counts and scored seconds for every language and dataset are in runtime.json.

Broad emotional/nonverbal dataset membership covers 19.77% of the complete run, but includes neutral speech. It must not be called 19.77% verified nonverbal events. Audited labels establish 898 distinct sources with at least one specific event/style label. A label describes the recording, not a timestamped event in the selected crop.

| Audited source label | Sources in full 15000 | Scored seconds from those sources |
|---|---:|---:|
| Laughter | 267 | 553.59 |
| Screaming | 96 | 180.56 |
| Yell | 59 | 119.02 |
| Shout | 7 | 15.48 |
| Giggle | 26 | 42.10 |
| Chuckle_and_chortle | 12 | 25.35 |
| Crying_and_sobbing | 12 | 30.64 |
| human_whistling_source_description | 5 | 9.99 |
| Whispering | 72 | 91.18 |
| explicit_whisper_style | 211 | 526.77 |
| Breathing | 143 | 251.85 |

Labels can overlap, so counts and seconds in that table must not be summed. FSD labels require at least two present-and-predominant votes and no negative vote. Human whistling uses previously reviewed source descriptions. The particularly thin recorded exposure is whistling and crying: fewer than 10 seconds and 31 seconds respectively across the complete run. Event presence within those crops remains unverified.

## Measured quiet target coverage

All 12,000 fresh cached teacher targets were inspected on their actual scored 20 ms windows, including partial tails. No encoder/decoder was run.

| Teacher target RMS | Windows | Valid seconds |
|---|---:|---:|
| rms_exact_zero | 0 | 0.00 |
| 0<rms<=1 e-5 | 11,096 | 221.60 |
| 1 e-5<rms<=1 e-4 | 48,520 | 967.60 |
| 1 e-4<rms<=1 e-3 | 162,176 | 3240.08 |
| rms>1 e-3 | 1,246,984 | 24937.84 |

Fresh data contains 4429.28 seconds below or equal to RMS 0.001 (15.08% of scored duration), including 1189.20 seconds at or below RMS 0.0001 and 221.60 seconds at or below RMS 0.00001. Thus quiet training targets are present. There are no exactly-zero decoded teacher windows; this does not show that the original recordings lack silence.

## Training losses are still changing

| Mean across 500 updates | Steps 1001–1500 | Steps 4501–5000 | Change |
|---|---:|---:|---:|
| total | 0.009907 | 0.006527 | -34.12% |
| waveform | 0.009288 | 0.006144 | -33.85% |
| mel | 0.520198 | 0.365881 | -29.66% |
| feature | 0.029253 | 0.014888 | -49.11% |

These are different fresh-source groups, so they are descriptive training statistics rather than a matched quality comparison. The fixed development panel is the appropriate basis for deciding whether continued training is improving fidelity. This audit establishes no execution failure, no source reuse, and no absent quiet data. It does identify sparse source-level whistling/crying coverage. It does not establish why the remaining reconstruction errors persist.

## Evidence and limits

- runtime.json contains exact hashes, source mappings, sample counts, all shard receipt identities, optimizer settings and timing statistics.
- source-label-map.json records the reviewed taxonomy/vote evidence and its file hashes.
- Frozen state is supported by the in-process storage/version checks and preserved-state receipts. An independent full final teacher-tensor dump was not recorded.
- No training, model, loss, threshold, source order, producer or checkpoint was modified during this audit.
