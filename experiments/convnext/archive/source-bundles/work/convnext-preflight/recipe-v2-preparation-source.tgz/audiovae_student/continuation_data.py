"""Extend one student's source-conservative exposure lineage for a new trial."""
from .data import ManifestRow
from .restart_data import _counts, identity
from .comparison_data import parent_student_ledger


def continuation_ledger(previous, used_rows, counts, *, parent_checkpoint_sha256,
                        used_plan_identity_sha256, provenance=None):
    """Retire all earlier-lineage and just-used files, without older-model data.

    The supplied earlier ledger belongs to this student, not a global collection
    of independent experiments. Previously partial entries are conservatively
    retired whole as well. Shared A/B exposure is supplied once, not counted twice.
    """
    used_rows = tuple(used_rows)
    _counts(used_rows, counts)
    by_id, combined_counts = {}, {}
    for key, entry in previous.entries.items():
        by_id[key] = ManifestRow.from_dict(entry['row'])
        combined_counts[key] = entry['input_samples']
    for row in used_rows:
        if not previous.whole_untouched(row):
            raise ValueError('Newly used comparison source overlaps its prior student lineage')
        if row.source_id in by_id:
            if identity(row) != identity(by_id[row.source_id]):
                raise ValueError('Source identity changed across continuation lineage')
            raise ValueError('Current comparison repeated a prior source')
        by_id[row.source_id] = row
        combined_counts[row.source_id] = counts[row.source_id]
    return parent_student_ledger(by_id.values(), combined_counts,
        checkpoint_sha256=parent_checkpoint_sha256,
        provenance={**(provenance or {}),
                    'inherited_student_ledger_sha256': previous.identity_sha256,
                    'newly_consumed_plan_identity_sha256': used_plan_identity_sha256,
                    'inherited_sources': len(previous.entries),
                    'newly_consumed_sources': len(used_rows),
                    'deduplication_policy': 'Retire complete r5 and common r6 source files once; old independent models remain reusable'})
