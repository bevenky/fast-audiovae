"""Prepare fresh-student optimization/calibration metadata from existing audio.

Run on the training host after source review. This script downloads no audio,
loads no model, and does not launch training or modify earlier plans.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


OLD = Path("/workspace/fast-audiovae-convnext-20260908-r1")
PILOT = Path("/workspace/fast-audiovae-convnext-20260909-r5")
PREVIOUS = Path("/workspace/fast-audiovae-convnext-20260909-r7/data/quiet-phase-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previous-plan", type=Path, default=PREVIOUS)
    parser.add_argument("--full-manifest", type=Path, default=OLD / "expanded-pilot/corpus/source-manifest.jsonl")
    parser.add_argument("--supplement-manifest", type=Path,
                        default=PILOT / "data-expressive-topup-v1/versions/v3-jsonl/train.jsonl")
    parser.add_argument("--optimization-windows", type=int, default=320000)
    parser.add_argument("--calibration-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from audiovae_student.comparison_data import load_comparison_plan
    from audiovae_student.data import load_manifest
    from audiovae_student.recipe_v2_data import plan_recipe_v2_data, write_recipe_v2_plan
    from audiovae_student.restart_data import file_sha, identity

    previous = load_comparison_plan(args.previous_plan)
    pinned = previous["identity"]["provenance"]["source_files_sha256"]
    observed = {}

    def verify_file(path):
        current = file_sha(path)
        if current != pinned.get(str(path)):
            raise ValueError("Source metadata changed from its pinned prior plan: " + str(path))
        observed[str(path)] = current

    by_id = {}
    for path in (args.full_manifest, args.supplement_manifest):
        verify_file(path)
        for row in load_manifest(path):
            if row.split != "train":
                continue
            if row.source_id in by_id and identity(row) != identity(by_id[row.source_id]):
                raise ValueError("Duplicate source ID changed immutable metadata")
            by_id[row.source_id] = row
    rows = list(by_id.values())
    counts = {row.source_id: round(row.duration_seconds * 16000) for row in rows}
    labels, receipts = {}, {}
    for row in rows:
        if row.dataset in {"fleurs", "librispeech", "indicvoices", "emogator", "crema_d"}:
            continue
        if row.access_record not in receipts:
            path = Path(row.access_record)
            verify_file(path)
            receipts[row.access_record] = json.loads(path.read_text())
        receipt = receipts[row.access_record]
        values = list(receipt.get("selected_metadata", {}).get("target_labels", []))
        if row.dataset.startswith("freesound_human_whistle_"):
            values.append("human_whistling_source_description")
        if "whisper" in str(receipt.get("labels", {})).lower():
            values.append("explicit_whisper_style")
        labels[row.source_id] = sorted(set(values))

    plan = plan_recipe_v2_data(rows, counts, event_labels=labels,
        reserved_rows=previous["reserved"], excluded_sources=previous["excluded"],
        optimization_windows=args.optimization_windows, calibration_windows=args.calibration_windows,
        minimum_input_samples=3040, seed=args.seed)
    selected = {row.source_id: row for purpose in ("optimization", "calibration") for row in plan[purpose]["rows"]}
    missing = [source_id for source_id, row in selected.items() if not Path(row.audio_path).is_file()]
    if missing:
        raise ValueError("Selected original prepared files are missing: " + str(missing[:10]))
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_student": True, "prior_models_training_audio_reuse_authorized": True,
        "reservation_source_plan_identity": previous["identity"]["identity_sha256"],
        "old_student_consumed_ledger_applied": False, "source_files_sha256": observed,
        "planner_sha256": file_sha(Path(__file__)),
        "data_helper_sha256": file_sha(args.source_root / "audiovae_student/recipe_v2_data.py"),
        "source_root": str(args.source_root), "existing_audio_files": len(selected),
        "source_audio_hash_and_length_revalidation": "Mandatory in SourceCorpus before frozen-teacher target preparation",
        "calibration_contract": "Separate training windows, two fixed-weight no-gradient passes, no optimization scored overlap"}
    if not args.dry_run:
        ready = write_recipe_v2_plan(plan, args.output, provenance=provenance)
    else:
        ready = None
    print(json.dumps({"report": plan["report"], "ready": ready, "output": str(args.output),
                      "published": not args.dry_run}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
