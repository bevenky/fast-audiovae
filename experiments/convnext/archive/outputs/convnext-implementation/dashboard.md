# Training dashboard

[Open training and held-out loss curves](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=%5E(train%7Cvalidation)%2Ftotal%24)

Select `warmup-speech-muon-v1`. Its first real speech warmup has completed 1,000 updates. `train/total` measures sampled training crops; `validation/total` measures fixed crops from 25 separate LibriSpeech clips. Held-out loss fell from 171.89 to 40.89. These are reconstruction losses, not MOS or audible-quality scores.

Clear the tag filter to see loss components, learning rates, gradient norm, language exposure and step duration. The two `preflight-synthetic-*` runs are earlier 32-update setup checks, not speech quality comparisons.

TensorBoard is running on the existing Runpod. The link remains available while the service and Pod run. The opened browser has automatic reload enabled every 10 seconds. In another browser, enable Settings > Reload data if needed.

Logs and checkpoints remain under `/workspace/fast-audiovae-convnext-20260908-r1/`. TensorBoard reads only its `runs/` subdirectory. The idle Jupyter service was stopped to reuse exposed port 8888, with its restart settings saved privately on the Pod.

[Open corpus download progress](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=corpus%2F). Select `corpus-preparation-500h`. The persistent workflow records source hours while downloading, then starts `warmup-speech-500h-v1` after the corpus passes verification. Until then, the earlier step-1,000 loss curve remains unchanged.
