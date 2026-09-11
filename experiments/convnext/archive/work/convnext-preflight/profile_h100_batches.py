"""Disposable training throughput probe; never saves or advances a training checkpoint."""

from __future__ import annotations

import gc
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time

import torch

from audiovae_student.batching import batched_scored_crop_loss
from audiovae_student.corpus_training import (
    CachedCorpus, CorpusTrainingConfig, draw_training_crop, scored_crop_loss,
)
from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.optimizers import build_optimizer_bundle


BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "training-runs/warmup-speech-muon-v1/latest.pt"
INDEX = BASE / "target-cache/index.json"
REPORT = BASE / "h100-batch-profile.json"
WARMUP, MAX_BATCH = 2, 64
MEASURED_UPDATES = {"serial8_existing_checks": 24, "batch8_existing_checks": 64,
                    "batch8_aggregated_checks": 64, "batch32_aggregated_checks": 24,
                    "batch64_aggregated_checks": 16}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_report(report):
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(REPORT)


class UtilizationSampler:
    def __init__(self):
        self.samples = []
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        def read():
            for line in self.process.stdout:
                try:
                    gpu, memory, used = [float(value.strip()) for value in line.split(",")]
                    self.samples.append({"time": time.monotonic(), "gpu_percent": gpu,
                                         "memory_percent": memory, "memory_used_mib": used})
                except ValueError:
                    continue
        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def finish(self, start, end):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.thread.join(timeout=5)
        # The device reports a rolling interval; omit its initial second so an
        # earlier idle/setup window cannot masquerade as active utilization.
        samples = [row for row in self.samples if start + 1 <= row["time"] <= end]
        return {
            "query_period_ms": 100,
            "excluded_initial_measured_seconds": 1,
            "samples": samples,
            "gpu_percent_mean": statistics.mean(row["gpu_percent"] for row in samples) if samples else None,
            "gpu_percent_median": statistics.median(row["gpu_percent"] for row in samples) if samples else None,
            "gpu_percent_max": max(row["gpu_percent"] for row in samples) if samples else None,
            "note": "nvidia-smi rolling utilization samples during measured updates, not an idle snapshot",
        }


def native_checked(tensors, aggregate):
    if aggregate:
        return bool(torch.stack([torch.isfinite(value).all() for value in tensors]).all())
    return all(bool(torch.isfinite(value).all()) for value in tensors)


def main():
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError("Another GPU compute process is active; refusing overlapping probe: " + processes)
    torch.set_num_threads(1)
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint_hash = digest(CHECKPOINT)
    started_load = time.monotonic()
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved["step"] != 1000:
        raise ValueError("Probe requires the completed step-1000 checkpoint")
    identity = saved["identity"]
    if identity["runtime"]["matmul_tf32"] or identity["runtime"]["cudnn_tf32"]:
        raise ValueError("Previous training did not use the expected FP32 TF32-off contract")
    torch.backends.cudnn.deterministic = identity["runtime"]["cudnn_deterministic"]
    config = CorpusTrainingConfig(**identity["training_config"])
    loss_config = WarmupLossConfig(**identity["loss_config"])
    corpus = CachedCorpus(INDEX, max_cached_utterances=1000)
    if corpus.data_fingerprint != identity["data_fingerprint"]:
        raise ValueError("Probe corpus differs from the checkpoint corpus")
    verified_load_seconds = time.monotonic() - started_load
    started_preload = time.monotonic()
    for index in range(len(corpus.entries["train"])):
        corpus.get("train", index)
    preload_seconds = time.monotonic() - started_preload
    sampler = random.Random(1071)
    started_crops = time.monotonic()
    # Every smaller case uses the same prefix of each 64-example draw, and all
    # cases retain the original variable context and partial-tail sampling.
    crop_sets = [[draw_training_crop(corpus, config, loss_config, sampler)
                  for _ in range(MAX_BATCH)] for _ in range(WARMUP + max(MEASURED_UPDATES.values()))]
    crop_seconds = time.monotonic() - started_crops
    report = {
        "purpose": "Disposable student training throughput, no inference RTF and no quality evaluation",
        "checkpoint_sha256": checkpoint_hash, "checkpoint_step": saved["step"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
        "device_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "tf32_matmul": False, "tf32_cudnn": False, "autocast": False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "checkpoint_runtime": identity["runtime"],
        "train_caches": len(corpus.entries["train"]), "dev_caches": len(corpus.entries["dev"]),
        "data_fingerprint": corpus.data_fingerprint, "loss_config": identity["loss_config"],
        "training_config": identity["training_config"],
        "optimizer": "same native Muon twenty hidden matrices plus complementary AdamW, restored step1000 state",
        "load": {"checkpoint_and_verified_index_seconds": verified_load_seconds,
                 "preload_train_caches_seconds": preload_seconds,
                 "prepare_fixed_crops_seconds": crop_seconds,
                 "fixed_crops": len(crop_sets) * MAX_BATCH},
        "timing_scope": "CPU crop tensors preloaded; includes CPU-to-CUDA transfers, model forward, losses, backward, gradient clipping, optimizer and checks; excludes disk reads, sampling, checkpoint writes, dev and TensorBoard",
        "cases": [],
    }
    atomic_report(report)
    cases = [("serial8_existing_checks", 8, False, False),
             ("batch8_existing_checks", 8, True, False),
             ("batch8_aggregated_checks", 8, True, True),
             ("batch32_aggregated_checks", 32, True, True),
             ("batch64_aggregated_checks", 64, True, True)]
    for name, batch_size, batched, aggregate in cases:
        measured_updates = MEASURED_UPDATES[name]
        gc.collect()
        torch.cuda.empty_cache()
        model = StudentDecoder(StudentConfig(**identity["model_config"])).to("cuda").train()
        model.load_state_dict(saved["model"])
        optimizer = build_optimizer_bundle(model, optimizer=config.optimizer,
                                            lr=config.learning_rate, weight_decay=config.weight_decay,
                                            betas=config.betas)
        # AdamW's noncapturable step counters remain CPU tensors. Deep-copy the
        # restored state so a disposable case cannot advance the next one's
        # in-memory initial counter even though the checkpoint file is unchanged.
        optimizer.load_state_dict(deepcopy(saved["optimizer"]))
        initial_adamw_steps = sorted({float(state["step"]) for state in
                                      optimizer.optimizers["adamw"].state.values() if "step" in state})
        if initial_adamw_steps != [1000.0]:
            raise ValueError("Every case must restore the same step1000 optimizer state")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        criterion = WarmupReconstructionLoss(loss_config)
        sampler_util = UtilizationSampler()
        sampler_util.start()
        timings, measured_metrics, scored_samples, padding, groups = [], [], 0, 0, []
        start_measure = None
        try:
            for update, crop_set in enumerate(crop_sets[:WARMUP + measured_updates]):
                crops = crop_set[:batch_size]
                if update == WARMUP:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_bytes = torch.cuda.memory_allocated()
                    start_measure = time.monotonic()
                torch.cuda.synchronize()
                start_step = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                metric_rows = []
                if batched:
                    result = batched_scored_crop_loss(model, crops, criterion, "cuda")
                    values = result.mean
                    if not bool(torch.isfinite(values["total"])):
                        raise FloatingPointError("Nonfinite loss")
                    values["total"].backward()
                    if aggregate:
                        # Exactly one metric transfer; finite check also covers all components.
                        numbers = torch.stack([value.detach() for value in values.values()]).cpu().tolist()
                    else:
                        # Preserve the existing scalar-at-a-time metric transfer behavior.
                        numbers = [float(value.detach()) for value in values.values()]
                    metric_rows.append(dict(zip(values, numbers)))
                else:
                    for crop in crops:
                        values = scored_crop_loss(model, crop, criterion, "cuda")
                        if not bool(torch.isfinite(values["total"])):
                            raise FloatingPointError("Nonfinite loss")
                        (values["total"] / batch_size).backward()
                        metric_rows.append({key: float(value.detach()) for key, value in values.items()})
                if not all(math.isfinite(value) for row in metric_rows for value in row.values()):
                    raise FloatingPointError("Nonfinite metric")
                gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                if len(gradients) != len(parameters) or not native_checked(gradients, aggregate):
                    raise FloatingPointError("Missing or nonfinite gradient")
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                if not native_checked(parameters, aggregate):
                    raise FloatingPointError("Nonfinite parameter after optimizer step")
                number_norm = float(norm)
                if update == 0:
                    first_loss = {key: statistics.mean(row[key] for row in metric_rows)
                                  for key in metric_rows[0]}
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start_step
                if update >= WARMUP:
                    timings.append(elapsed)
                    measured_metrics.append({key: statistics.mean(row[key] for row in metric_rows)
                                             for key in metric_rows[0]})
                    scored_samples += sum(crop.valid_scored_samples for crop in crops)
                    if batched:
                        padding += result.padded_latent_frames
                        groups.append(len(result.groups))
                    else:
                        groups.append(batch_size)
                print(json.dumps({"case": name, "update": update + 1, "seconds": elapsed,
                                  "warmup": update < WARMUP, "grad_norm": number_norm}), flush=True)
            end_measure = time.monotonic()
            utilization = sampler_util.finish(start_measure, end_measure)
            entry = {"name": name, "status": "passed_finite_checks", "batch_size": batch_size,
                     "model_microbatch": batch_size if batched else 1,
                     "warmup_updates": WARMUP, "measured_updates": measured_updates,
                     "initial_adamw_steps": initial_adamw_steps,
                     "first_update_loss_before_weight_change": first_loss,
                     "step_seconds": timings, "mean_step_seconds": statistics.mean(timings),
                     "median_step_seconds": statistics.median(timings),
                     "examples_per_second": batch_size * measured_updates / sum(timings),
                     "scored_audio_seconds_per_second": scored_samples / 48000 / sum(timings),
                     "baseline_allocated_gib": baseline_bytes / 2**30,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                     "actual_measured_scored_audio_seconds": scored_samples / 48000,
                     "padded_latent_frames": padding,
                     "loss_groups_per_update": groups, "losses": measured_metrics,
                     "utilization": utilization}
            report["cases"].append(entry)
        except Exception as error:
            end = time.monotonic()
            utilization = sampler_util.finish(start_measure or end, end)
            report["cases"].append({"name": name, "status": "failed", "error": repr(error),
                                     "utilization": utilization})
            atomic_report(report)
            raise
        finally:
            # No resulting model, optimizer, waveform or training checkpoint is saved.
            if "result" in locals():
                del result
            if "values" in locals():
                del values
            if "gradients" in locals():
                del gradients
            if "norm" in locals():
                del norm
            del model, optimizer, criterion, parameters
            gc.collect()
            torch.cuda.empty_cache()
        atomic_report(report)
    report["checkpoint_unchanged"] = digest(CHECKPOINT) == checkpoint_hash
    if not report["checkpoint_unchanged"]:
        raise RuntimeError("Original training checkpoint changed during the disposable probe")
    atomic_report(report)
    print(json.dumps({"report": str(REPORT), "completed": True}), flush=True)


if __name__ == "__main__":
    main()
