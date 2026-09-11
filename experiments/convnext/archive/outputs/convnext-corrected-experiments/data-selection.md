# Data for the corrected decoder experiments

Each experiment starts from the trained step-8,090 student. The regular and targeted generator selections contain **12,800 windows, totaling 8.978 hours each**. Every position has exactly the same valid audio duration in both selections. They share 89.854% of their scored audio duration as ordinary speech.

| Generator selection | Duration share | Scored seconds | Windows |
| --- | ---: | ---: | ---: |
| Common ordinary speech | 89.854% | 29,040.64 | 11,344 |
| Targeted explicit expressive/whisper sources | 5.077% | 1,640.94 | 816 |
| Targeted contiguous quiet | 2.170% | 701.44 | 274 |
| Targeted quiet/active transitions | 2.899% | 936.96 | 366 |

The targeted rows replace 10.146% of the regular selection. Both selections cover all 22 scheduled Indic languages, English, and a broad additional language mixture. All 22 Indic languages also occur in the separate discriminator warmup and fixed-weight gradient-calibration pools.

**The control is a matched ordinary-speech selection, not an exact replay of the earlier training mixture.** It contains 99.630% speech and 0.370% sources carrying expressive or mixed nonverbal labels. Requiring full-length replacement candidates excluded many short generic-emotion recordings. The targeted selection contains 5.077% explicitly labeled expressive/whisper material. Results therefore compare these recorded mixtures, not the old nominal 5% broad-emotion recipe.

| Targeted source label | Scored seconds | Distinct sources |
| --- | ---: | ---: |
| Whistling | 45.23 | 5 |
| Crying and sobbing | 53.12 | 9 |
| Giggling | 44.05 | 23 |
| Laughter | 255.64 | 126 |
| Screaming | 188.77 | 65 |
| Shout | 15.98 | 6 |
| Yell | 268.45 | 52 |
| Whispering, two explicit source categories combined | 526.80 | 198 |
| Breathing | 220.60 | 126 |
| Chuckling | 22.31 | 8 |

These are durations of crops from labeled recordings, **not measured durations of the named event**. Source labels and acoustic activity select the 190 ms discriminator view; they do not establish semantic event timestamps. A high-energy interval could still contain another sound. Whistling and several other categories remain limited by the number of independent recordings.

Quiet candidates contain at least one uninterrupted second with input RMS at most 0.001. Transition views contain at least 60 ms of low-level audio and 60 ms of active audio. The available long-quiet pool was smaller than the initial allocation, so qualified transitions filled the remainder without relaxing acoustic thresholds. These checks use source input, not teacher-output quiet labels.

The user's debugging exception permits **1,096.62 seconds across 517 targeted generator windows** already encountered by the parent student. Each is identified in the sealed selection. No scored interval repeats within a new arm, including its preparatory pools. Causal history may overlap intentionally. Existing held-out identities and normalization-calibration exclusions remain protected. Source file hashes, decoded lengths, mono 16 kHz format, and finite sample values were verified on Runpod. No audio was downloaded to the Mac.

The unchanged primary development panel is supplemented by **one previously reserved 7.149-second Yell recording**. Its source, parent, file hash, and known speaker/session identities do not intersect training. This is one additional diagnostic, not population-level evidence. Separate reviewed crying, giggling, and shouting held-out cases remain unavailable. A generic Shout co-label on the Yell recording does not establish independent reviewed Shout coverage.

Selection identity: `1963ec9e863bc2f0ea00e259266554fe658275b6e411b6750dcad93df8d39c32`.
