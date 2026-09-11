"""Student lineage must accumulate once across shared-data comparison arms."""
from dataclasses import replace

import pytest

from audiovae_student.comparison_data import parent_student_ledger, plan_comparison
from audiovae_student.continuation_data import continuation_ledger
from test_restart_data import row, counts, WINDOW


def test_chain_retires_r5_and_r6_once_without_old_independent_sources():
    old, recent, available = row('r5'), row('r6'), row('older-independent')
    prior = parent_student_ledger([old], counts([old]), checkpoint_sha256='a' * 64)
    chained = continuation_ledger(prior, [recent], counts([recent]),
        parent_checkpoint_sha256='b' * 64, used_plan_identity_sha256='c' * 64)
    assert len(chained.entries) == 2
    assert not chained.whole_untouched(old) and not chained.whole_untouched(recent)
    assert chained.whole_untouched(available)
    assert chained.document['provenance']['inherited_student_ledger_sha256'] == prior.identity_sha256
    alias = replace(available, audio_sha256=recent.audio_sha256)
    assert chained.overlaps(alias, 0, 100)
    with pytest.raises(ValueError, match='prior student lineage'):
        continuation_ledger(chained, [recent], counts([recent]),
            parent_checkpoint_sha256='d' * 64, used_plan_identity_sha256='e' * 64)


def test_exhausted_rare_events_are_not_replayed_to_fill_a_followup_quota():
    used = row('cry-used', samples=WINDOW, language='und')
    prior = parent_student_ledger([used], counts([used]), checkpoint_sha256='a' * 64)
    corpus = [used] + [row(f'{lang}-{i}', language=lang, samples=WINDOW * 4)
                      for lang in ('en', 'hi', 'ar') for i in range(5)]
    plan = plan_comparison(corpus, counts(corpus), prior, event_labels={'cry-used': ['Crying_and_sobbing']},
        windows_count=32, required_indic=('hi',), required_other=('ar',))
    assert len(plan['windows']) == 32
    assert all(w.source_id != 'cry-used' for w in plan['windows'])
    assert any(item['bucket'] == 'events' and item['reallocated_to'] == 'english'
               for item in plan['metadata']['drained_capacity'])
