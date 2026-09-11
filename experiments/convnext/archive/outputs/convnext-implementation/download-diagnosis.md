# Download diagnosis

8 September 2026. All audio remains on Runpod.

The 440-hour FLEURS/LibriSpeech acquisition completed after parallel archive downloads were enabled. The remaining Indic and whistle jobs read selected audio embedded in Parquet files through the Hub filesystem interface.

A bounded Indic probe read 68.4 MB in 13.13 seconds through the existing range path. The native Hub/Xet downloader fetched the entire 356.36 MB pinned file in 12.91 seconds. The overlapping 705 audio files and their metadata matched, and the full downloaded file passed its upstream SHA-256 check. This is a single-file transport comparison, not an end-to-end corpus speedup measurement.

The native option preserves the existing selection policy, source order, original audio bytes and durable receipts. It uses six workers, a shared 6 GiB temporary staging cap and an 8 GiB free-space reserve. Completed temporary files are removed after their Parquet readers close. All 19 focused CPU tests pass on Runpod.

The whistle job is less efficient: one selected recording typically requires reading a group containing 59 recordings. It continues with its reviewed source and sample-rate checks. Its remaining tail has not been replaced with a custom binary-format reader.

The training workflow requires every configured source to finish and every IndicVoices language to reach two verified hours. Consequently the final planned pool is approximately 511 hours. Reaching 500 aggregate hours alone does not start training.

[Transfer evidence](indic-io-probe.json).
