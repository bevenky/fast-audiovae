"""CPU-only target occupancy audit; no model construction or forward calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("This data-only audit requires CUDA_VISIBLE_DEVICES=''")
import torch
import run_pilot as base

BIN_NAMES = ("exact_zero","positive_to_1e-5","over_1e-5_to_1e-4","over_1e-4_to_1e-3","over_1e-3")


def bin_index(rms):
    if rms == 0: return 0
    if rms <= 1e-5: return 1
    if rms <= 1e-4: return 2
    if rms <= 1e-3: return 3
    return 4


def empty_bin():
    return {"windows":0,"samples":0,"full_windows":0,"partial_windows":0,"source_ids":set()}


@torch.no_grad()
def measure(crops):
    buckets = {kind:{name:empty_bin() for name in BIN_NAMES} for kind in ("all","actual_startup_sources")}
    startup_count = 0
    for crop in crops:
        source = crop["source_id"]
        start = crop["context_frames"]*1920
        stop = start+crop["valid_scored_samples"]
        audio = crop["teacher_audio"]
        if audio.device.type != "cpu" or start < 0 or not start < stop <= audio.shape[-1]:
            raise ValueError("Cached teacher must have a valid CPU scored extent")
        target = audio[...,start:stop]
        measured = base.quiet_window_metrics(target,target,torch.ones_like(target,dtype=torch.bool))
        startup = crop["context_frames"]==0 and crop["start_frame"]==0 and crop["context_start_frame"]==0
        startup_count += startup
        for row in measured["windows"]:
            name = BIN_NAMES[bin_index(row["teacher_rms"])]
            for kind in (("all","actual_startup_sources") if startup else ("all",)):
                item = buckets[kind][name]
                item["windows"] += 1
                item["samples"] += row["valid_samples"]
                item["full_windows"] += row["valid_samples"]==960
                item["partial_windows"] += row["valid_samples"]!=960
                item["source_ids"].add(source)
    for kind in buckets.values():
        for row in kind.values():
            row["sources"] = len(row["source_ids"])
            row["source_ids"] = sorted(row["source_ids"])
            row["seconds"] = row["samples"]/48000
            row["hours"] = row["samples"]/48000/3600
    total = sum(c["valid_scored_samples"] for c in crops)
    if total != sum(row["samples"] for row in buckets["all"].values()):
        raise AssertionError("Window binning lost scored samples")
    return {"sources":len(crops),"actual_startup_sources":startup_count,"samples":total,
            "seconds":total/48000,"hours":total/48000/3600,"bins":buckets,
            "ordered_source_ids_sha256":hashlib.sha256(json.dumps([c["source_id"] for c in crops],
                separators=(",",":"),ensure_ascii=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError("Preserve the earlier audit")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): raise RuntimeError("GPU must remain unavailable")
    # This prevents an accidental module call if the audit is extended later.
    def forbid_forward(*args,**kwargs): raise RuntimeError("Neural/module forwards are forbidden in this data audit")
    torch.nn.Module._call_impl = forbid_forward
    preflight = json.loads(args.preflight.read_text())
    identity = preflight["identity"]
    if preflight.get("passed") is not True or base.sha(args.manifest)!=identity["manifest_sha256"]:
        raise ValueError("Expected the authenticated numerical preflight and manifest")
    if base.sha(base.__file__) != identity["runner_sha256"]:
        raise ValueError("Original data-loader source changed")
    quiet_module = base.sys.modules["audiovae_student.quiet_audio"]
    if base.sha(quiet_module.__file__) != identity["shared_helpers_sha256"]["quiet_audio"]:
        raise ValueError("Original quiet metric source changed")
    started = time.monotonic()
    manifest,pools,receipt = base.load_data(args.manifest)
    if receipt["sha256"] != identity["cache_sha256"] or len(pools["fit"])!=3000:
        raise ValueError("Expected the same3000-source fitting pool")
    first,later = pools["fit"][:768],pools["fit"][768:3000]
    result = {"version":"audiovae2_cached_target_rms_bins_v1","complete":True,
              "cpu_only":True,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
              "neural_forward_calls":0,"training_updates":0,"threads":torch.get_num_threads(),
              "torch":str(torch.__version__),"script_sha256":base.sha(__file__),
              "preflight_sha256":base.sha(args.preflight),"manifest_sha256":base.sha(args.manifest),
              "teacher_cache_sha256":receipt["sha256"],"quiet_metric_sha256":base.sha(quiet_module.__file__),
              "measurement":"Existing FP32 teacher RMS on scored-crop20ms windows; include every valid sample in partial tails",
              "thresholds":"exact0;0<RMS<=1e-5;1e-5<RMS<=1e-4;1e-4<RMS<=1e-3;RMS>1e-3",
              "startup_definition":"No context and absolute source start0; subset includes all scored windows from these sources, not only their first window",
              "first768":measure(first),"remaining2232":measure(later)}
    if base.sha(args.manifest)!=result["manifest_sha256"] or base.sha(args.preflight)!=result["preflight_sha256"]:
        raise RuntimeError("Input receipts changed during the audit")
    result["elapsed_seconds"] = time.monotonic()-started
    base.write_json(args.out,result)
    print(json.dumps({"complete":True,"out":str(args.out),"elapsed_seconds":result["elapsed_seconds"],
                      "first768_sources":len(first),"remaining_sources":len(later)}),flush=True)


if __name__ == "__main__": main()
