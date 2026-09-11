"""Bounded authenticated full-source encoder and conditional-target audit.

No training, target repair, or student inference. A separate command performs
the exhaustive conditional-pair scan; this command selects32or64 real sources.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json
import time
from validate_conditional_targets import (BASE, ASSETS, EXPECTED_DATA_SHA, EXPECTED_RECEIPT_SHA,
    EXPECTED_CACHE_SHA, EXPECTED_TEACHER_STATE, sha, digest, tensor_sha, write_json, metric, score_bounds)


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.batching import _validate_crop
    from audiovae_student.data import ManifestRow
    from audiovae_student.source_corpus import read_native_16k
    from audiovae_student.teacher import FrozenAudioVAE2
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.restart_data import digest as plan_digest
    if args.encoder_sample not in (32, 64):
        raise ValueError("Registered sample sizes are32or64 unique sources")
    args.out.mkdir(parents=True, exist_ok=False)
    if sha(BASE / "targeted-data.json") != EXPECTED_DATA_SHA or sha(BASE / "target-cache.json") != EXPECTED_RECEIPT_SHA:
        raise ValueError("Original data plan or cache receipt changed")
    data = json.loads((BASE / "targeted-data.json").read_text())
    receipt = json.loads((BASE / "target-cache.json").read_text())
    cache = Path(receipt["path"])
    if receipt["sha256"] != EXPECTED_CACHE_SHA or sha(cache) != EXPECTED_CACHE_SHA:
        raise ValueError("Archived target cache changed")
    saved = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if saved["identity"] != receipt["identity"]:
        raise ValueError("Archived cache identity mismatch")
    if plan_digest({k: v for k, v in data.items() if k != "identity_sha256"}) != data["identity_sha256"]:
        raise ValueError("Plan identity mismatch")
    selected = []; seen = set()
    for index, raw in enumerate(saved["pools"]["gradient_calibration"]):
        if raw["source_id"] in seen:
            continue
        crop = TrainingCrop(**raw); _validate_crop(crop); score_bounds(crop)
        window = data["pools"]["gradient_calibration"][index]["window"]
        if (crop.source_id, crop.start_frame, crop.valid_scored_samples) != (
                window["source_id"], window["start_frame"], window["valid_output_samples48k"]):
            raise ValueError("Cached crop and planned selection disagree")
        selected.append((index, crop)); seen.add(crop.source_id)
        if len(selected) == args.encoder_sample:
            break
    if len(selected) != args.encoder_sample:
        raise ValueError("Insufficient unique sample sources")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.version() != 92501:
        raise ValueError("Requires corrected cuDNN9.25.1")
    teacher = FrozenAudioVAE2.from_files(ASSETS / "audio_vae_v2.py", ASSETS / "audiovae.pth", device="cuda")
    before = state_fingerprint(teacher.model.state_dict())
    if before != EXPECTED_TEACHER_STATE or before != saved["identity"]["teacher_state_sha256"]:
        raise ValueError("Teacher state differs from historical targets")
    result = {"version": "authenticated-training-encoder-sample-v1", "parameter_updates": 0,
              "selection": "First unique sources in sealed gradient_calibration pool order; no selection by error",
              "sample_sources": len(selected), "selection_indices": [i for i, c in selected],
              "source_scope": "Complete supplied source-manifest segment, not cropped input",
              "data_plan_sha256": EXPECTED_DATA_SHA, "cache_sha256": EXPECTED_CACHE_SHA,
              "teacher_state_sha256": before, "teacher_assets": teacher.provenance,
              "actual_encoder_backend": "Pinned original model.encode at FP32, singleton full source, cuDNN enabled9.25.1; explicit qualified backend override of wrapper policy",
              "decoder_backend": "FrozenAudioVAE2.decode FP32 with cuDNN enabled9.25.1",
              "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version(), "tf32": False, "deterministic": True},
              "latent_tolerance": {"max_abs": 2e-5, "rms": 2e-6},
              "scope_warning": "Sample findings cannot establish how many of the4749sources are affected",
              "script_sha256": sha(__file__), "cases": [], "complete": False}
    start_time = time.time()
    with torch.no_grad():
        for index, crop in selected:
            raw_row = data["rows"][crop.source_id]
            row = ManifestRow.from_dict(raw_row)
            # Reproduce SourceCorpus's default reader only. Refuse any custom
            # gain/reader policy rather than silently treating it as unchanged.
            native = row.original_sample_rate_hz == 16000 and all(
                v.casefold().split(":", 1)[0].strip() in {"none", "unchanged"}
                for v in (row.resampler_policy, row.gain_policy))
            prepared = row.resampler_policy in {"prepared-native-16000-float-v1", "prepared-soxr-vhq-to-16000-v1"} and row.gain_policy == "no-additional-gain-normalization"
            if prepared and ((row.resampler_policy == "prepared-native-16000-float-v1") != (row.original_sample_rate_hz == 16000)):
                raise ValueError("Prepared source rate policy mismatch")
            if not native and not prepared:
                raise ValueError("Unqualified source transformation for " + crop.source_id)
            path = Path(row.audio_path)
            if not path.is_absolute():
                raise ValueError("Relative audio path requires the original explicit source root")
            payload = path.read_bytes()
            if __import__("hashlib").sha256(payload).hexdigest() != row.audio_sha256:
                raise ValueError("Source SHA mismatch: " + crop.source_id)
            audio = read_native_16k(payload, row)
            if audio.shape != (1, 1, data["counts"][crop.source_id]) or not bool(torch.isfinite(audio).all()):
                raise ValueError("Complete source sample contract mismatch")
            if abs(audio.shape[-1] / 16000 - row.duration_seconds) > 1 / 16000 + 1e-10:
                raise ValueError("Source duration mismatch")
            start = crop.context_start_frame * 640
            input_samples = min(crop.latents.shape[-1] * 640, audio.shape[-1] - start)
            if input_samples <= 0 or crop.reference16k is None:
                raise ValueError("Missing authentic cached input reference")
            reference = audio[..., start:start + input_samples]
            if not reference.equal(crop.reference16k[..., :input_samples]):
                raise ValueError("Prepared source does not bitwise match archived input; gain/reader mismatch")
            input_hash = tensor_sha(audio)
            zinput = audio.to(teacher.device)
            # Bypass only the wrapper's conservative cuDNN-off scope, retaining
            # the pinned model's exact encode implementation and weights.
            with torch.autocast(device_type="cuda", enabled=False):
                fresh_z = teacher.model.encode(zinput, 16000)
            if fresh_z.shape != (1, 64, (audio.shape[-1] + 639) // 640) or fresh_z.dtype != torch.float32 or not bool(torch.isfinite(fresh_z).all()):
                raise ValueError("Fresh encoder output contract mismatch")
            if tensor_sha(zinput) != input_hash:
                raise ValueError("Encoder mutated authenticated input")
            real_frames = min(crop.latents.shape[-1], fresh_z.shape[-1] - crop.context_start_frame)
            z = fresh_z[..., crop.context_start_frame:crop.context_start_frame + real_frames].cpu()
            cached = crop.latents[..., :real_frames]
            latent = metric(z, cached)
            latent["passed"] = latent["finite"] and latent["max_abs"] <= 2e-5 and latent["rms"] <= 2e-6
            channels = (z.double() - cached.double()).abs().amax(dim=(0, 2)).tolist()
            # This intentionally decodes archived z, not fresh_z. It tests
            # conditional pair coherence separately from encoder correctness.
            old_z = crop.latents.to(teacher.device)
            old_hash = tensor_sha(old_z)
            conditional = teacher.decode(old_z).cpu()
            if tensor_sha(old_z) != old_hash:
                raise ValueError("Decoder mutated archived latents")
            a, b = score_bounds(crop)
            case = {"source_id": crop.source_id, "pool_index": index,
                    "source_audio_sha256": row.audio_sha256, "prepared_audio_sha256": input_hash,
                    "input_samples16k": audio.shape[-1], "reference_bitwise_equal": True,
                    "gain_policy": row.gain_policy, "resampler_policy": row.resampler_policy,
                    "additional_gain_or_resampling": False, "context_start_frame": crop.context_start_frame,
                    "real_latent_frames_compared": real_frames, "padded_latent_frames_excluded": crop.latents.shape[-1] - real_frames,
                    "fresh_encoder_vs_archived_latents": latent, "per_channel_max_abs": channels,
                    "conditional_decode_vs_archived_target": metric(conditional[..., a:b], crop.teacher_audio[..., a:b])}
            result["cases"].append(case)
            write_json(args.out / "progress.json", result)
            print(json.dumps({"completed": len(result["cases"]), "expected": len(selected), "source_id": crop.source_id,
                              "latent_max_abs": latent["max_abs"], "conditional_max_abs": case["conditional_decode_vs_archived_target"]["max_abs"]}), flush=True)
    after = state_fingerprint(teacher.model.state_dict())
    preserved = before == after and sha(cache) == EXPECTED_CACHE_SHA and sha(BASE / "targeted-data.json") == EXPECTED_DATA_SHA
    result.update({"complete": True, "teacher_state_after_sha256": after, "originals_unchanged": preserved,
                   "teacher_remained_frozen": not any(p.requires_grad for p in teacher.model.parameters()) and not any(m.training for m in teacher.model.modules()),
                   "latent_failed_sources": sum(not c["fresh_encoder_vs_archived_latents"]["passed"] for c in result["cases"]),
                   "conditional_failed_sources": sum(not c["conditional_decode_vs_archived_target"]["passed"] for c in result["cases"]),
                   "elapsed_seconds": time.time() - start_time})
    result["passed"] = preserved and result["teacher_remained_frozen"] and not result["latent_failed_sources"] and not result["conditional_failed_sources"]
    write_json(args.out / "receipt.json", result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-sample", type=int, choices=(32, 64), default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import fcntl
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main(args))
