# Expressive supplement and remaining gaps

A separate supplement is prepared on Runpod. No files were added to the active 20-hour training plan or the fixed 84-recording development panel. No new audio was downloaded to the Mac.

| Version 2 | Training | Development |
|---|---:|---:|
| Recordings | 26 | 8 |
| Audio seconds | 127.13 | 57.73 |
| Distinct uploader groups | 16 | 5 |
| Whispering | 12 | 4 |
| Breathing | 12 | 3 |
| Yell | 2 | 1 |

These are small supplemental checks, not evidence that expressive quality has been achieved. Uploader groups are conservative source identities, not independently identified physical speakers. Two entirely unused whisper contributors and one entirely unused Yell contributor were reserved before selecting training recordings. The source partition remains official FSD50K train; internal development records retain that provenance. Yell remains a separate label from Shout.

The 34 files use the pinned FSD mirror and per-file CC0 or CC-BY-3.0 licenses. Individual WAVs were downloaded with up to four requests in parallel. The importer checked pinned content hashes, input format, finite audio, decoded duplicates and cross-split identities. Attribution, original release WAVs and prepared 16 kHz files are retained. No files were quarantined. The separate 31-file version 1 audit additionally verified all file hashes again, found zero clipping and measured 36.23 seconds of training input and 12.42 seconds of development input below -60 dBFS in 20 ms windows. Teacher-output energy has not been measured on these new files yet.

## Does the co-label filter discard useful crying or giggling?

Ordinary speech, breathing and other human vocal co-labels are already allowed. Removing only the remaining vocal-only co-label restriction yields no additional unacquired crying or giggling recordings that satisfy the current commercial-license and two-positive-predominant-vote checks.

It yields two unacquired Shout recordings, Freesound IDs 221958 and 101314. Both also carry Cheering and Human_group_actions labels. Both contributors have already appeared in earlier data. Neither was acquired in this step. They could be reviewed as group-vocal training examples under an explicit policy, but cannot provide new-contributor validation. They would not fill the crying or giggling gap.

## Remaining work

Fresh crying and giggling sources still need acquisition and source review. The existing strict FSD event pool has been consumed. Changing emotion labels into action labels would not resolve this gap. The saved broader Freesound mirror metadata can be searched for new actions, but its previous scan retained only whistling keyword matches; counts for new crying or giggling candidates have not been established.

The whistling metadata scan contains 301 keyword matches. Many remaining matches describe instruments, synthetic sounds or edited recordings. No additional human-whistle recording was qualified here. Twenty-four earlier selected whistle files were rejected because their mirror sample rate differed from their original source page. Recovering those would require inspecting the actual originals and preserving their real provenance, rather than silently accepting a mismatch. Freesound provides original-file download through its authenticated API. [Freesound download documentation](https://freesound.org/docs/api/resources_apiv2.html#download-sound-oauth2-required)

Version 1 reports remain unchanged. [Version 2 readiness](expressive-topup-v2-ready.json), [version 1 audio checks](expressive-topup-prepared.json), [source selection](expressive-topup-selection.json), and [co-label audit](expressive-colabel-policy-audit.json) retain the detailed evidence.
