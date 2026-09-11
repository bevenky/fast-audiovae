"""Regenerate explicitly selected training crops from authenticated full sources.

Immutable historical cache and crop geometry; independent singleton encoder
reference; actual padded encoder batches checked before targets are accepted.
Only the teacher runs. This script never creates a training optimizer.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from validate_conditional_targets import (BASE, ASSETS, EXPECTED_DATA_SHA,
    EXPECTED_RECEIPT_SHA, EXPECTED_CACHE_SHA, EXPECTED_TEACHER_STATE,
    sha, tensor_sha, metric, score_bounds)
from audiovae_student.cache import TrainingCrop
from audiovae_student.data import ManifestRow
from audiovae_student.source_corpus import read_native_16k
from audiovae_student.teacher import FrozenAudioVAE2
from audiovae_student.batching import _validate_crop
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import digest

GEOMETRY = ("source_id", "start_frame", "context_start_frame", "context_frames",
            "scored_frames", "valid_scored_samples")
VERSION = "full-source-corrected-training-pairs-v1"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def window(x, start, length):
    if type(start) is not int or type(length) is not int or start < 0 or length < 1:
        raise ValueError("Window needs a nonnegative start and positive length")
    part = x[..., start:start + length]
    return F.pad(part, (0, length - part.shape[-1])).contiguous().clone()


def validate_source_geometry(crop, input_samples):
    """Allow right storage padding, but never count it as authentic audio."""
    _validate_crop(crop)
    for name in ("start_frame", "context_start_frame"):
        value = getattr(crop, name)
        if type(value) is not int or value < 0:
            raise ValueError("Source frame coordinates must be nonnegative integers")
    if type(input_samples) is not int or input_samples < 1:
        raise ValueError("Authentic source must contain real input samples")
    if crop.context_start_frame + crop.context_frames != crop.start_frame:
        raise ValueError("Inconsistent absolute crop coordinates")
    if crop.start_frame * 1920 + crop.valid_scored_samples > input_samples * 3:
        raise ValueError("Scored crop exceeds authentic source samples")
    score_bounds(crop)


def validate_teacher_tensor(value, shape, label):
    if (tuple(value.shape) != tuple(shape) or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())):
        raise ValueError(label + " shape/dtype/finite mismatch")


def authenticated_audio(data, sid, crops):
    row = ManifestRow.from_dict(data["rows"][sid])
    native = row.original_sample_rate_hz == 16000 and all(
        v.casefold().split(":", 1)[0].strip() in {"none", "unchanged"}
        for v in (row.resampler_policy, row.gain_policy))
    prepared = row.resampler_policy in {"prepared-native-16000-float-v1", "prepared-soxr-vhq-to-16000-v1"} and row.gain_policy == "no-additional-gain-normalization"
    if prepared and ((row.resampler_policy == "prepared-native-16000-float-v1") != (row.original_sample_rate_hz == 16000)):
        raise ValueError("Prepared source rate policy mismatch")
    if not native and not prepared:
        raise ValueError("Unqualified transformation " + sid)
    path = Path(row.audio_path)
    if not path.is_absolute():
        raise ValueError("Source path must be absolute")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row.audio_sha256:
        raise ValueError("Source hash differs " + sid)
    audio = read_native_16k(payload, row)
    if (audio.shape != (1, 1, data["counts"][sid]) or audio.dtype != torch.float32
            or not bool(torch.isfinite(audio).all())):
        raise ValueError("Source samples/shape differ " + sid)
    if abs(audio.shape[-1] / 16000 - row.duration_seconds) > 1 / 16000 + 1e-10:
        raise ValueError("Source duration differs " + sid)
    for c in crops:
        validate_source_geometry(c, audio.shape[-1])
        if c.reference16k is None or not window(audio, c.context_start_frame * 640, c.latents.shape[-1] * 640).equal(c.reference16k):
            raise ValueError("Archived reference differs from authentic source " + sid)
    return audio, row


def compare_latents(actual, reference):
    report = metric(actual, reference)
    # Explicit abs+rel reference qualification, same FP32 contract for every
    # source. Historical-cache comparisons remain descriptive, not this gate.
    diff = (actual.double() - reference.double()).abs()
    limit = 2e-5 + 2e-5 * reference.double().abs()
    report["passed"] = report["finite"] and bool((diff <= limit).all()) and report["rms"] <= 2e-6
    report["max_scaled_error"] = float((diff / limit).max())
    return report


def main(args):
    if args.out.exists():
        raise ValueError("Refusing to overwrite output")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.version() != 92501:
        raise ValueError("Requires cuDNN9.25.1")
    if sha(BASE / "targeted-data.json") != EXPECTED_DATA_SHA or sha(BASE / "target-cache.json") != EXPECTED_RECEIPT_SHA:
        raise ValueError("Original plan/receipt changed")
    data = json.loads((BASE / "targeted-data.json").read_text())
    receipt = json.loads((BASE / "target-cache.json").read_text())
    cache = Path(receipt["path"])
    if sha(cache) != EXPECTED_CACHE_SHA:
        raise ValueError("Original cache changed")
    saved = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if saved["identity"] != receipt["identity"] or digest({k:v for k,v in data.items() if k != "identity_sha256"}) != data["identity_sha256"]:
        raise ValueError("Cache/data identity differs")
    selected = {}
    for spec in args.pool:
        name, count = spec.split(":")
        count = int(count)
        if name in selected or name not in saved["pools"] or not 0 < count <= len(saved["pools"][name]):
            raise ValueError("Invalid pool selection")
        selected[name] = [TrainingCrop(**c) for c in saved["pools"][name][:count]]
        for i,c in enumerate(selected[name]):
            w = data["pools"][name][i]["window"]
            if (c.source_id,c.start_frame,c.valid_scored_samples) != (w["source_id"],w["start_frame"],w["valid_output_samples48k"]):
                raise ValueError("Planned crop differs")
    by_source = defaultdict(list)
    for name, crops in selected.items():
        for i,c in enumerate(crops): by_source[c.source_id].append((name,i,c))
    # Authenticate every selected source before model calls. Retain only a
    # small audio batch at once; verify again immediately before inference.
    inventory = {}
    for sid, entries in by_source.items():
        a,row = authenticated_audio(data,sid,[c for _,_,c in entries])
        inventory[sid] = {"samples16k":a.shape[-1],"source_audio_sha256":row.audio_sha256,"prepared_audio_sha256":tensor_sha(a)}
    atomic_json(args.out / "inventory.json", inventory)
    teacher = FrozenAudioVAE2.from_files(ASSETS / "audio_vae_v2.py", ASSETS / "audiovae.pth", device="cuda")
    if state_fingerprint(teacher.model.state_dict()) != EXPECTED_TEACHER_STATE:
        raise ValueError("Teacher state changed")
    result = {"version":VERSION,"pool_counts":{k:len(v) for k,v in selected.items()},
        "sources":len(by_source),"parameter_updates":0,"runtime":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version()},
        "teacher":teacher.provenance,"teacher_state_sha256":EXPECTED_TEACHER_STATE,
        "generator_sha256":sha(__file__),
        "pool_selection":"Ordered crop prefixes; source count is reported separately",
        "encoder_policy":("Full-source singleton cuDNN-disabled FP32 raw_mu reference only; no batched encoder results used" if args.reference_only else "Full-source singleton cuDNN-disabled FP32 raw_mu reference; every actual padded cuDNN9.25.1 batch row must match reference"),
        "decoder_policy":"Singleton full-source decode from reference latents, cuDNN9.25.1; crop only after continuous decode",
        "latent_batch_tolerance":{"atol":2e-5,"rtol":2e-5,"max_rms":2e-6},"batch_checks":[],"complete":False,
        "batch_qualification_performed":not args.reference_only,
        "prior_batch_failure_evidence":args.prior_batch_failure_evidence}
    if args.reference_only and not args.prior_batch_failure_evidence:
        raise ValueError("Reference-only generation requires recorded reason/evidence")
    replacements = {k:[None]*len(v) for k,v in selected.items()}
    ids = sorted(by_source, key=lambda sid:inventory[sid]["samples16k"])
    start_time = time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),args.batch_size):
            batch_ids = ids[start:start+args.batch_size]
            audios = [authenticated_audio(data,sid,[c for _,_,c in by_source[sid]])[0] for sid in batch_ids]
            for sid,a in zip(batch_ids,audios):
                if tensor_sha(a) != inventory[sid]["prepared_audio_sha256"]: raise ValueError("Source changed after inventory")
            padded = ((max(a.shape[-1] for a in audios)+639)//640)*640
            if not args.reference_only:
                x = torch.cat([F.pad(a,(0,padded-a.shape[-1])) for a in audios]).to("cuda")
                # Batch-first call reproduces the ordering that exposed the old bug.
                zbatch = teacher.model.encode(x,16000)
                validate_teacher_tensor(zbatch, (len(batch_ids),64,padded//640), "Actual batch encoder")
            for i,(sid,a) in enumerate(zip(batch_ids,audios)):
                frames=(a.shape[-1]+639)//640
                with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
                    z=teacher.model.encode(a.to("cuda"),16000)
                validate_teacher_tensor(z, (1,64,frames), "Reference encoder")
                if not args.reference_only:
                    comparison=compare_latents(zbatch[i:i+1,:,:frames],z)
                    result["batch_checks"].append({"source_id":sid,"batch":len(batch_ids),"padded_samples16k":padded,"comparison":comparison})
                    if not comparison["passed"]:
                        atomic_json(args.out / "failed.json", result)
                        raise ValueError("Fresh batch/reference mismatch " + sid)
                y=teacher.decode(z)
                validate_teacher_tensor(y, (1,1,frames*1920), "Fresh teacher decoder")
                z=z.cpu();y=y.cpu()
                key=digest({"version":VERSION,"source":inventory[sid],"teacher_state_sha256":EXPECTED_TEACHER_STATE,"z":tensor_sha(z),"y":tensor_sha(y)})
                for name,index,old in by_source[sid]:
                    length=old.latents.shape[-1]
                    fresh=replace(old,latents=window(z,old.context_start_frame,length),
                        teacher_audio=window(y,old.context_start_frame*1920,length*1920),
                        reference16k=window(a,old.context_start_frame*640,length*640),cache_key=key)
                    validate_source_geometry(fresh, a.shape[-1])
                    if any(getattr(old,k)!=getattr(fresh,k) for k in GEOMETRY):raise ValueError("Crop geometry changed")
                    replacements[name][index]=fresh
            completed=min(start+args.batch_size,len(ids))
            atomic_json(args.out / "progress.json",{"completed_sources":completed,"total_sources":len(ids),"elapsed_seconds":time.monotonic()-start_time})
            print(json.dumps({"completed_sources":completed,"total_sources":len(ids),"elapsed_seconds":time.monotonic()-start_time}),flush=True)
    frozen=not any(p.requires_grad for p in teacher.model.parameters()) and not any(m.training for m in teacher.model.modules())
    if not frozen or state_fingerprint(teacher.model.state_dict()) != EXPECTED_TEACHER_STATE or sha(cache)!=EXPECTED_CACHE_SHA:
        raise ValueError("Teacher or immutable cache changed")
    if sha(BASE / "targeted-data.json") != EXPECTED_DATA_SHA or sha(BASE / "target-cache.json") != EXPECTED_RECEIPT_SHA:
        raise ValueError("Original plan/receipt changed during generation")
    if any(c is None for crops in replacements.values() for c in crops):raise ValueError("Missing replacement crop")
    result.update(complete=True,teacher_remained_frozen=True,original_cache_unchanged=True,elapsed_seconds=time.monotonic()-start_time)
    atomic_json(args.out / "generation.json",result)
    # Seal using the same contract implementation consumed by diagnostics and
    # the 400-step runner; the output never replaces the archived cache.
    from types import SimpleNamespace
    from training_overlay import seal_training_overlay
    ctx=SimpleNamespace(pools=selected,receipt={"target_cache_sha256":EXPECTED_CACHE_SHA},
        identity={"data_plan_sha256":EXPECTED_DATA_SHA},
        parent={"identity":{"data":{"teacher_state_sha256":EXPECTED_TEACHER_STATE}}})
    seal_training_overlay(ctx, replacements, args.out/"pairs.pt", provenance={
        "generation_path":str(args.out/"generation.json"),"generation_sha256":sha(args.out/"generation.json"),
        "inventory_sha256":sha(args.out/"inventory.json"),"source_count":len(ids),
        "encoder_policy":result["encoder_policy"],"decoder_policy":result["decoder_policy"],"runtime":result["runtime"]})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--pool",action="append",required=True)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--reference-only",action="store_true")
    p.add_argument("--prior-batch-failure-evidence")
    args=p.parse_args()
    if not 1<=args.batch_size<=8:raise ValueError("Batch size outside bounded range")
    lock=Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args)
