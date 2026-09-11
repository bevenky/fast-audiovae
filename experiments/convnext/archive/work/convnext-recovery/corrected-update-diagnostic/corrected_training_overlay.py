"""Generator-facing sealing API; consumer contract lives in training_overlay."""
from pathlib import Path
from types import SimpleNamespace

from training_overlay import seal_training_overlay as _seal


def seal_training_overlay(out, replacements, selected, *, source_provenance,
                          base_target_cache_sha256, base_data_plan_sha256, teacher_state_sha256):
    context = SimpleNamespace(pools=selected,
        receipt={"target_cache_sha256": base_target_cache_sha256},
        identity={"data_plan_sha256": base_data_plan_sha256},
        parent={"identity": {"data": {"teacher_state_sha256": teacher_state_sha256}}})
    return _seal(context, replacements, Path(out) / "pairs.pt", provenance=source_provenance)
