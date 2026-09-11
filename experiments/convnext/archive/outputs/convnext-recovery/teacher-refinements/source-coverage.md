# Expressive source coverage audit

The selected data has recoverable expressive labels even though every training and selection `condition` field is null. Joining the selected source IDs to the pinned FSD50K labels and annotation votes, plus reviewed whistling and explicit whisper metadata, identifies **341 of 2,048 fitting sources**, **46 of 256 selection sources**, and **23 of 144 natural canonical sources** with specific event or style labels. These are distinct-source counts, not sums of overlapping labels.

| Source label | Fit | Selection | Canonical |
|---|---:|---:|---:|
| Laughter | 71 | 10 | 6 |
| Screaming | 36 | 5 | 7 |
| Crying / sobbing | 4 | 0 | 0 |
| Human whistling | 3 | 0 | 2 |
| Giggling | 11 | 3 | 0 |
| Chuckle / chortle | 3 | 0 | 0 |
| Shout | 6 | 0 | 0 |
| Yell | 42 | 3 | 1 |
| FSD whispering | 37 | 9 | 4 |
| Explicit Thorsten whisper style | 66 | 10 | 0 |
| Breathing | 68 | 6 | 3 |

The remaining **1,707 fit**, **210 selection**, and **121 canonical** sources lack a specific audited event/style label. Many are ordinary speech or broadly emotional material; a missing event label does not mean the audio is nonexpressive.

FSD labels require at least two present-and-predominant votes and no negative-presence vote. This matters: raw training taxonomy lists 49 Shout sources, but only **6** pass that vote rule. Raw Laughter falls from 84 to **71** and Screaming from 40 to **36**. Yell remains separate from Shout.

**We cannot establish event presence within each selected crop from these files.** The annotations describe recordings and contain no dense event timestamps. For example, selected crops from the four crying-labeled sources total about **7.61 seconds**, and crops from the three human-whistle sources total about **6.00 seconds**; these are scored audio durations from labeled recordings, not verified crying or whistling durations. No audio was listened to or decoded in this audit.

The clearest coverage gaps are the absence of crying and human-whistle sources from selection, and the absence of crying from canonical evaluation. The canonical panel also lacks verified giggling, chuckling and Shout sources, although it includes one Yell recording. The first 32 calibration sources contain two laughter sources, one FSD whispering source and one Thorsten whisper source, with none of the other listed events.

No data, runner, checkpoint or quality threshold changed. The join uses the same selection identity as the completed spectral comparison: `c3c2f0335b00761e1425d0c40a12d9882305bd5440c0f3758bc4d982d77ced7f`. The [JSON evidence](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/teacher-refinements/source-coverage.json) records every source mapping, votes, metadata hashes and counts.

Sources: [selected source identities](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/head-experiments/selection.json), [pinned FSD labels](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-preflight/expressive-research/fsd50k/dev.csv), [pinned annotation votes](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-preflight/expressive-research/fsd50k/pp_pnp_ratings_FSD50K.json), [archived whistling source evidence](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-preflight/expressive-research/human-whistling/source-evidence.json), [earlier explicit-style source inventory](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-quiet-pilot/coverage-audit.json), and [canonical results](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/spectral-head-comparison/results.json).
