# Saved validation audit through 5,000 steps

The student improves substantially against the 1,000-step anchor, but the final 500 updates introduce a broad amplitude-related regression. Near-silence also regressed early and has not recovered. These findings do not support automatically declaring5,000 the best checkpoint or extending training solely because correlation is below0.99.

| Step | Active cosine | Waveform MAE | Common mel | Group MSE | Quiet RMS | Quiet passes | Near passes | Peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.944676 | 0.00678626 | 0.505503 | 0.02498578 | 0.00024423 | 115/2544 | 91/184 | 0.954666 |
| 1500 | 0.954157 | 0.00598112 | 0.464879 | 0.02039645 | 0.00021831 | 30/2544 | 2/184 | 0.899189 |
| 2000 | 0.961902 | 0.00542407 | 0.431668 | 0.01754479 | 0.00020767 | 55/2544 | 2/184 | 0.971128 |
| 2500 | 0.963616 | 0.00535605 | 0.403883 | 0.01571390 | 0.00019866 | 94/2544 | 1/184 | 0.973291 |
| 3000 | 0.964121 | 0.00516940 | 0.394672 | 0.01433094 | 0.00019538 | 92/2544 | 4/184 | 0.966982 |
| 3500 | 0.969434 | 0.00500821 | 0.368514 | 0.01340744 | 0.00019110 | 108/2544 | 4/184 | 0.975240 |
| 4000 | 0.966616 | 0.00506046 | 0.369479 | 0.01263926 | 0.00019147 | 80/2544 | 16/184 | 0.950940 |
| 4500 | 0.971874 | 0.00489207 | 0.350438 | 0.01180799 | 0.00018144 | 104/2544 | 3/184 | 0.979977 |
| 5000 | 0.972531 | 0.00553008 | 0.342111 | 0.01169538 | 0.00017688 | 118/2544 | 4/184 | 0.974619 |

Every checkpoint has96 sources,94 active-source cosine scores,11,246,952 valid samples,8,809,920 active samples,2,544 quiet windows and 184 near-silence windows. All source order, per-source support, mel element counts and teacher-energy invariants match exactly. No overshoot samples are reported. The local development manifest hash matches the authenticated training launch.

## Improvement versus the anchor

From 1,000 to 5,000, waveform MAE improves 18.51%, common mel 32.32%, group MSE 53.19%, and quiet RMS 27.58%. All 94 active-source cosines and all 96 source mel/group errors improve. However,11 source MAEs worsen.

## The final 500 updates are a tradeoff

From 4,500 to 5,000, waveform MAE worsens 13.04% on 78/96 sources, waveform MSE worsens 10.21%, and linear mel error worsens 19.45%. Log mel improves 2.61%, so the combined mel still improves. Correlation improves only 0.0657 percentage points overall, while42/94 individual sources regress. Group MSE improves only 0.95% and worsens on 41/96 sources.

Active pooled student/teacher RMS gain increases1.02093 to 1.08102. Median per-source gain increases1.00301 to 1.07354, and sources louder than the teacher increase52/94 to 83/94. Correlation is invariant to positive scaling; it can improve while amplitude-sensitive error worsens. This establishes amplitude drift in the saved measurements, not its underlying optimization cause.

| Largest late MAE contributors | Language | Gain4,500 | Gain5,000 | MAE4,500 | MAE5,000 |
|---|---|---:|---:|---:|---:|
| kashmiri:1970324837177077_chunk_1.flac | ks | 1.0730 | 1.1438 | 0.0124425 | 0.0174002 |
| ur_pk:train:17356430244359180807.wav | ur | 1.0409 | 1.1330 | 0.0055785 | 0.0095324 |
| fr_fr:train:16715901780663941383.wav | fr | 1.0549 | 1.1433 | 0.0072536 | 0.0102616 |
| et_ee:train:16713145029159361863.wav | et | 1.0534 | 1.1156 | 0.0064349 | 0.0090120 |
| af_za:train:12687107890474022499.wav | af | 1.0759 | 1.1607 | 0.0049554 | 0.0070885 |
| konkani:4222124650671062_chunk_1.flac | kok | 1.0646 | 1.1480 | 0.0117348 | 0.0138232 |
| sanskrit:2251799813734813_chunk_1.flac | sa | 1.0291 | 1.1770 | 0.0072451 | 0.0093111 |
| cmn_hans_cn:train:10162125752145534325.wav | cmn | 1.0001 | 1.1501 | 0.0062271 | 0.0082072 |

## Quiet and near-silence

Quiet pooled RMS improves, but near-silence passes collapse91/184 at 1,000 to 2/184 at 1,500 and 4/184 at 5,000. All 91 prior passes are lost in their 11 source groups; the four current passes are in two other sources. Spanish es_419 falls42/44 to 0/44, and Kannada falls22/42 to 0/42. Continuous near-silence residuals were not saved by the overview observer, so the precise DC/noise/amplitude cause cannot be established from these files alone.

The overall quiet pass rate is4.52% at 1,000 and 4.64% at 5,000, despite27.58% lower aggregate quiet RMS. Existing thresholds are provisional engineering checks, not calibrated audibility thresholds. Neither a flat pass rate nor a low aggregate RMS alone describes the whole failure.

## Expressive sources and language coverage

| Source label | Cosine1,000 | Cosine4,500 | Cosine5,000 | MAE4,500 | MAE5,000 | Active RMS gain5,000 |
|---|---:|---:|---:|---:|---:|---:|
| Whispering | 0.755825 | 0.905604 | 0.910158 | 0.0070799 | 0.0071854 | 1.0336 |
| Crying_and_sobbing | 0.937432 | 0.966876 | 0.960247 | 0.0156808 | 0.0173437 | 1.0677 |
| Giggle | 0.931732 | 0.965626 | 0.969794 | 0.0186833 | 0.0180661 | 1.0728 |
| Laughter | 0.820520 | 0.885933 | 0.885642 | 0.0347079 | 0.0344437 | 0.9474 |
| explicit_whisper_style | 0.868112 | 0.947203 | 0.948900 | 0.0134815 | 0.0135451 | 1.0120 |
| Chuckle_and_chortle | 0.817585 | 0.932748 | 0.957930 | 0.0200992 | 0.0174939 | 1.0257 |
| Breathing | 0.742600 | 0.846610 | 0.853961 | 0.0018543 | 0.0018199 | 0.8420 |
| Shout | 0.889200 | 0.951892 | 0.957476 | 0.0146710 | 0.0148948 | 1.0604 |
| human_whistling_source_description | 0.948382 | 0.984126 | 0.978090 | 0.0007943 | 0.0008607 | 0.8897 |
| Screaming | 0.861367 | 0.941758 | 0.940489 | 0.0312628 | 0.0329994 | 0.9971 |
| Yell | 0.798448 | 0.859617 | 0.850075 | 0.0883707 | 0.0919784 | 0.9301 |

The single whistling source improves relative to 1,000 but regresses after 4,500: cosine0.984126 to 0.978090 and MAE0.0007943 to 0.0008607, even while amplitude moves87.09% to 88.97% of the teacher. Amplitude matching alone is not reconstruction quality. Crying, screaming and yelling also worsen in the final 500 updates. Laughter is effectively flat in late cosine/MSE, although MAE and mel improve slightly. Breathing remains weak at 0.853961 cosine but is improving.

At 5,000, the 11 expressive crops average0.928433 cosine, versus 0.979739 for66 ordinary-speech crops. The lowest individual scores include yelling0.850075, breathing0.853961, laughter0.885642, whispering0.910158, Tamil0.911841 and Pashto0.913741. Every active source nevertheless improves in cosine relative to 1,000.

The panel contains 78 normalized language labels plus und and all 22 scheduled Indic language codes. Most languages and expressive categories have only one crop. These observations are per-recording diagnostics, not statistically reliable language rankings. Full paired histories for every source and language/cohort are in validation-audit.json. Source-level expressive labels lack event timestamps, so a label does not prove every scored sample contains that event.

## Interpretation and next decision

This is not a global learning plateau: the mel and group losses continue improving and most long-term reconstruction measures improve. It is a conflict between progress measures plus unresolved edge cases. Preserve 4,500,5,000 and the 1,000 anchor. Investigate the late amplitude shift and the early near-silence loss using the training audit before deciding on 10,000. No new architecture, loss weight or learning-rate change follows from these saved metrics alone. A final untouched quality/listening/CPU evaluation is still separate from these repeatedly used development crops.

No model, audio, inference or benchmark was run for this audit.
