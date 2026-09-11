"""Rescore saved candidates on a sealed corrected panel; never retrain.

Only the model and common diagnostic criterion are constructed. Discriminators,
optimizers and teacher models are not instantiated. Imports are GPU-free until
main() explicitly creates evaluation models on the chosen device.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n").encode()).hexdigest()


def read_pinned(path, expected):
    if sha(path) != expected:
        raise ValueError("Sealed file changed: " + str(path))
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def checkpoint_metadata(spec, plan):
    """Load CPU weights and verify every saved identity before evaluation."""
    import torch
    from audiovae_student.objective_comparison import state_fingerprint
    complete = read_pinned(spec["complete_receipt"], spec["complete_receipt_sha256"])
    migration = read_pinned(spec["migration_receipt"], spec["migration_receipt_sha256"])
    identity = read_pinned(spec["experiment_identity"], spec["experiment_identity_sha256"])
    if sha(spec["checkpoint"]) != spec["checkpoint_sha256"] or complete["checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Candidate checkpoint changed")
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    state = saved["engine"]
    expected_format = "fusion_screen_v1" if spec["group"] == "fusion-screen" else "corrected_screen_v1"
    if (saved["format_version"] != expected_format or state["format_version"] != "fusion_recipe_v1"
            or state["fusion_variant"] != spec["expected_variant"]
            or saved["parent_sha256"] != plan["parent_checkpoint_sha256"]
            or saved["experiment_identity_sha256"] != digest(identity)
            or saved["generator_updates"] != spec["expected_updates"]
            or state["step"] != spec["expected_step"]
            or complete["updates"] != spec["expected_updates"] or complete["step"] != spec["expected_step"]):
        raise ValueError("Saved checkpoint experiment, variant or training clock differs")
    label = saved["variant"] if spec["group"] == "fusion-screen" else saved["arm"]
    if label != spec["name"] or canonical(saved["migration"]) != canonical(migration) or canonical(state["fusion_migration"]) != canonical(migration):
        raise ValueError("Saved inference migration differs from its receipt")
    after = json.loads(Path(spec["historical_after_report"]).read_text())
    parameter_hash = state_fingerprint(state["model"])
    if (parameter_hash != after["model_identity"]["parameter_state_sha256"]
            or after["evaluated_step"] != state["step"]
            or canonical(state["loss_config"]) != canonical(after["criterion_config"])):
        raise ValueError("Saved model or diagnostic criterion differs from the historical report")
    return saved, identity, after, {
        "checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_path": spec["checkpoint"],
        "parameter_state_sha256": parameter_hash, "step": state["step"],
        "generator_updates": saved["generator_updates"], "discriminator_only_updates": saved["discriminator_only_updates"],
        "historical_after_report_sha256": sha(spec["historical_after_report"]),
        "experiment_identity_sha256": saved["experiment_identity_sha256"],
        "parent_sha256": saved["parent_sha256"], "fusion_variant": state["fusion_variant"],
        "model_config": canonical(state["model_config"]), "migration": canonical(migration),
        "loss_config": canonical(state["loss_config"]),
        "training_target_identity_sha256": saved.get("training_target_identity_sha256"),
        "data_plan_sha256": saved.get("data_plan_sha256")}


def build_evaluation_model(saved, identity, after, device):
    import torch
    from audiovae_student import model as model_module, fusion_architecture as architecture_module
    from audiovae_student import losses_distillation as loss_module, quiet_audio as quiet_module
    from audiovae_student import fusion_evaluation as evaluation_module
    from audiovae_student.model import StudentConfig, StudentDecoder
    from audiovae_student.fusion_architecture import FusionArchitectureConfig, FusionStudentDecoder
    from audiovae_student.losses_distillation import DistillationLossConfig, DistillationReconstructionLoss
    from audiovae_student.quiet_audio import QuietAudioConfig
    from audiovae_student.objective_comparison import state_fingerprint
    state = saved["engine"]; variant = state["fusion_variant"]
    sources = {}
    for module in (model_module, architecture_module, loss_module, quiet_module, evaluation_module):
        path = Path(module.__file__)
        expected = (identity.get("source_sha256", {}).get(path.name)
                    or identity.get("sources", {}).get("audiovae_student/" + path.name))
        if expected is None or sha(path) != expected:
            raise ValueError("Inference or diagnostic implementation differs from saved experiment: " + path.name)
        sources[str(path)] = expected
    architecture = FusionArchitectureConfig(**state["fusion_migration"]["architecture"])
    expected_architecture = FusionArchitectureConfig(terminal_tanh=variant == "tanh",
        causal_output_filter=variant == "filter", zero_startup_padding=variant == "zero_padding")
    if architecture != expected_architecture:
        raise ValueError("Saved architecture and variant label disagree")
    with torch.random.fork_rng(devices=[]):
        model = StudentDecoder(StudentConfig(**state["model_config"]))
        if any(asdict(architecture).values()):
            model = FusionStudentDecoder.from_decoder(model, architecture)
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        criterion = DistillationReconstructionLoss(DistillationLossConfig(**state["loss_config"]))
    model = model.to(device=device, dtype=torch.float32)
    criterion = criterion.to(device=device, dtype=torch.float32).eval()
    parameter_hash = state_fingerprint(model.state_dict())
    reconstructed_identity = {"class": type(model).__module__ + "." + type(model).__qualname__,
        "base_config": model.config.to_dict(),
        "architecture_config": model.fusion_config.to_dict() if hasattr(model, "fusion_config") else None,
        "fusion_migration": state["fusion_migration"], "parameter_state_sha256": parameter_hash}
    if canonical(reconstructed_identity) != canonical(after["model_identity"]):
        raise ValueError("Restored inference function or normalization buffers do not match saved model identity")
    engine = SimpleNamespace(model=model, criterion=criterion, device=torch.device(device), step=state["step"],
        config=SimpleNamespace(quiet_audio=QuietAudioConfig(**state["config"]["quiet_audio"])),
        fusion_migration=deepcopy(state["fusion_migration"]))
    return engine, sources


def value_change(control, candidate):
    if control is None or candidate is None or not math.isfinite(control) or not math.isfinite(candidate):
        return {"control": control, "candidate": candidate, "difference": None,
                "percent_change": None, "ratio_defined": False}
    return {"control": control, "candidate": candidate, "difference": candidate - control,
            "percent_change": 100 * (candidate / control - 1) if control != 0 else None,
            "ratio_defined": control != 0}


def compare_pair(pair, reports):
    control = reports[pair["control"]]; candidate = reports[pair["candidate"]]
    if (control["evaluation_contract_sha256"] != candidate["evaluation_contract_sha256"]
            or control["canonical_target_cache_sha256"] != candidate["canonical_target_cache_sha256"]
            or control["rescore_runtime"] != candidate["rescore_runtime"]):
        raise ValueError("Matched comparison used different inputs, criterion, masks or runtime")
    a, b = control["saved_checkpoint"], candidate["saved_checkpoint"]
    for name in ("step", "generator_updates", "discriminator_only_updates", "experiment_identity_sha256",
                 "parent_sha256", "training_target_identity_sha256", "data_plan_sha256", "loss_config"):
        if a[name] != b[name]:
            raise ValueError("Candidates do not have matched training history: " + name)
    fields = ("raw_mae", "mel", "nonquiet_cosine_mean", "quiet_residual_rms", "maximum_peak",
              "peak_excess_energy", "scored_overshoot_samples", "near_saturation_samples")
    groups = {}
    if set(control["recovery_metrics"]) != set(candidate["recovery_metrics"]):
        raise ValueError("Candidate and control groups differ")
    for group in control["recovery_metrics"]:
        ca = control["recovery_metrics"][group]; cb = candidate["recovery_metrics"][group]
        if any(ca[key] != cb[key] for key in ("crops", "sources", "samples", "quiet_windows", "quiet_samples")):
            raise ValueError("Matched group accounting differs")
        groups[group] = {"crops": ca["crops"], "sources": ca["sources"],
                         "metrics": {field: value_change(ca.get(field), cb.get(field)) for field in fields}}
    zero = {key: value_change(control["encoded_zero_steady_2_to_6_seconds"][key],
                              candidate["encoded_zero_steady_2_to_6_seconds"][key])
            for key in ("quiet_residual_rms", "quiet_student_rms")}
    high_frequency = {key: value_change(control["summary"][key], candidate["summary"][key]) for key in (
        "high_frequency_magnitude_mae", "high_frequency_log_magnitude_mae", "high_frequency_complex_residual_rms")}
    return {**pair, "groups": groups, "encoded_zero_steady_2_to_6_seconds": zero,
            "high_frequency": high_frequency, "evaluation_contract_sha256": control["evaluation_contract_sha256"],
            "interpretation": "Descriptive matched rescore, not retraining, significance, perceptual equivalence or adoption approval."}


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from canonical_evaluation import load_canonical_cache, build_canonical_evaluation
    plan = json.loads(args.plan.read_text())
    if plan["version"] != "saved-candidate-canonical-rescore-v1":
        raise ValueError("Unsupported candidate inventory")
    read_pinned(plan["canonical_receipt_path"], plan["canonical_receipt_sha256"])
    crops, metadata, canonical_receipt = load_canonical_cache(plan["canonical_receipt_path"])
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    if args.device != "cuda" or torch.backends.cudnn.version() != 92501:
        raise ValueError("Rescore requires the shared qualified CUDA/cuDNN9.25.1 runtime")
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "tf32": False,
               "cudnn_benchmark": False, "deterministic": True, "dtype": "float32"}
    requested = args.pairs or [pair["name"] for pair in plan["pairs"]]
    selected_pairs = [pair for pair in plan["pairs"] if pair["name"] in requested]
    if {pair["name"] for pair in selected_pairs} != set(requested):
        raise ValueError("Unknown pair requested")
    model_ids = list(dict.fromkeys(name for pair in selected_pairs for name in (pair["control"], pair["candidate"])))
    # Verify identities and data separation for every selected checkpoint before
    # a model forward. No heldout crop or source is silently dropped.
    proofs = {}; common_loss = None
    for name in model_ids:
        spec = plan["models"][name]
        saved, identity, after, proof = checkpoint_metadata(spec, plan)
        if common_loss is None: common_loss = proof["loss_config"]
        if proof["loss_config"] != common_loss:
            raise ValueError("Requested models do not share the original diagnostic criterion")
        if spec["group"] == "fusion-screen":
            training_ids = {row["source_id"] for row in identity["training_windows"]}
        else:
            path = Path(spec["experiment_identity"]).parent / "targeted-data.json"
            data = read_pinned(path, identity["data_plan_sha256"])
            training_ids = set(data["rows"])
        if training_ids & {crop.source_id for crop in crops}:
            raise ValueError("Corrected panel source overlaps this historical candidate's training data")
        proof["heldout_training_source_overlap"] = 0
        proof["historical_panel_crop_count"] = len(after["rows"])
        proofs[name] = proof
        del saved, identity, after
    write_json(args.out / "inventory.json", {"plan_sha256": sha(args.plan), "runtime": runtime,
        "checkpoints": proofs, "canonical_target_cache_sha256": canonical_receipt["sha256"],
        "canonical_contract_sha256": canonical_receipt["contract"]["identity_sha256"],
        "crops": len(crops), "sources": len(metadata), "parameter_updates": 0})
    reports = {}; started = time.time()
    for name in model_ids:
        print(json.dumps({"stage": "rescore_start", "model": name, "completed": len(reports), "total": len(model_ids)}), flush=True)
        saved, identity, after, proof = checkpoint_metadata(plan["models"][name], plan)
        engine, sources = build_evaluation_model(saved, identity, after, args.device)
        report = build_canonical_evaluation(engine, crops, metadata, canonical_receipt)
        if canonical(report["model_identity"]) != canonical(after["model_identity"]):
            raise ValueError("Evaluator changed saved inference identity")
        report["saved_checkpoint"] = proofs[name]
        report["rescore_runtime"] = runtime
        report["rescore_sources"] = sources
        report["rescore_notes"] = {
            "training": "Saved weights unchanged; no optimizer or discriminator was constructed",
            "loss": "Common historical diagnostic, not the candidate-specific short-mel training objective",
            "panel": "All285 corrected input/target crops, including the later reserved Yell source",
            "historical_comparison_limit": "Raw differences from old reports combine corrected targets, runtime and for200-step arms one added crop; only matched current rescoring isolates candidate ranking."}
        output = args.out / (name.replace("/", "__") + ".json")
        write_json(output, report); reports[name] = report
        del saved, identity, after, engine; gc.collect(); torch.cuda.empty_cache()
    comparisons = {pair["name"]: compare_pair(pair, reports) for pair in selected_pairs}
    unchanged = all(sha(plan["models"][name]["checkpoint"]) == plan["models"][name]["checkpoint_sha256"] for name in model_ids)
    unchanged = unchanged and sha(canonical_receipt["path"]) == canonical_receipt["sha256"]
    if not unchanged:
        raise ValueError("Saved checkpoints or canonical cache changed during rescore")
    summary = {"complete": True, "parameter_updates": 0, "teacher_calls": 0, "discriminator_calls": 0,
        "checkpoints_unchanged": True, "canonical_cache_unchanged": True, "runtime": runtime,
        "models": model_ids, "pairs": comparisons, "elapsed_seconds": time.time() - started,
        "evaluation_scope": "Teacher reconstruction, silence, peaks, expressive/language and spectral diagnostics. No PESQ/UTMOS/DNSMOS/listening rerun or CPU RTF measurement.",
        "adoption_decision": "Not made by this script; evaluate matched quality tradeoffs and historical training limitations."}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"stage": "rescore_complete", "models": len(reports), "pairs": len(comparisons), "output": str(args.out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--pairs", nargs="+")
    args = parser.parse_args()
    import fcntl
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args)
