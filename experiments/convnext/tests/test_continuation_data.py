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


from copy import deepcopy
import gzip
import hashlib
import json
import math

from audiovae_student import continuation_data as cd
from audiovae_student.comparison_data import load_comparison_plan, write_comparison_plan
from audiovae_student.restart_data import FixedWindowSampler, canonical, digest, windows_for
from audiovae_student.sampling import SamplerExhausted


def synthetic_catalog(rows=(), *, conditions=None, windows=None, reserved=()):
    rows = tuple(rows)
    conditions = conditions or {r.source_id: 'emotional_nonverbal' for r in rows}
    values = tuple(windows) if windows is not None else tuple(
        replace(w, condition=conditions[r.source_id]) for r in rows
        for w in windows_for(r, counts(rows)[r.source_id], minimum=3040))
    result = {'rows': rows, 'counts': counts(rows), 'windows': values,
              'reserved': tuple(reserved), 'conditions': dict(conditions),
              'identity': {'publication_sha256': 'a' * 64,
                           'files_sha256': {name: 'b' * 64 for name in cd._SUPPLEMENT_FILES}}}
    body = cd._catalog_body(result)
    result['identity'] = {**body, 'identity_sha256': digest(body)}
    return result


def generic(identifier, *, samples=WINDOW, **kwargs):
    return replace(row(identifier, samples=samples, dataset='emogator', language='und', **kwargs),
                   license='Apache-2.0')


def make_original(tmp_path, *, original=None, windows=None, inherited=None):
    original = original or row('original', samples=20 * WINDOW + 3040, language='en')
    calibration = row('calibration', samples=WINDOW, language='en')
    ww = tuple(windows) if windows is not None else tuple(
        replace(w, condition='speech') for w in windows_for(original, counts([original])[original.source_id], minimum=3040))
    previous = inherited or parent_student_ledger((), {}, checkpoint_sha256='a' * 64)
    plan = {'rows': [original], 'counts': counts([original]), 'windows': ww, 'ledger': previous,
            'reserved': [], 'excluded': [], 'metadata': {'fixture': True, 'minimum_output_samples': 9120}, 'seed': 47}
    write_comparison_plan(plan, tmp_path/'original')
    plan = load_comparison_plan(tmp_path/'original')
    sampler = FixedWindowSampler(plan['windows']); sampler.take_batch(4)
    parent = {'checkpoint_sha256': '1' * 64, 'run_identity_sha256': '2' * 64,
              'journal_sha256': '3' * 64, 'metrics_sha256': '4' * 64, 'step': 2, 'batch_size': 2}
    return {'original_plan': plan, 'sampler_state': sampler.state_dict(), 'parent': parent,
            'calibration_rows': [calibration], 'calibration_counts': counts([calibration]),
            'calibration_windows': tuple(windows_for(calibration, WINDOW, minimum=3040))}


def default_catalog():
    explicit = replace(row('breathing', samples=3040, dataset='fsd50k_vocal_cc_by_3', language='und'),
                       license='CC-BY-3.0')
    rows = [explicit] + [generic(f'generic-{i}') for i in range(6)]
    return synthetic_catalog(rows, conditions={r.source_id: ('Breathing' if r is explicit else 'emotional_nonverbal') for r in rows})


def plan_context(context, supplement=None, **kwargs):
    return cd.plan_fixed_window_continuation(**context, supplement=supplement or default_catalog(), **kwargs)


def test_fixed_prefix_preserves_partial_tail_calibration_and_all_original_order(tmp_path):
    context = make_original(tmp_path)
    result = plan_context(context)
    original = context['original_plan']['rows'][0]
    calibration = context['calibration_rows'][0]
    assert result['ledger'].overlaps(original, 0, 4 * WINDOW)
    assert not result['ledger'].overlaps(original, 4 * WINDOW, 5 * WINDOW)
    assert result['ledger'].overlaps(calibration, 0, WINDOW)
    kept = tuple(w for w in result['windows'] if w.source_id == original.source_id)
    assert kept == tuple(context['original_plan']['windows'][4:])
    assert kept[-1].valid_input_samples16k == 3040
    allocation = result['metadata']['allocation']
    assert allocation['supplement_explicit_input_samples'] == 3040
    assert allocation['supplement_generic_input_samples'] == WINDOW
    assert allocation['achieved_broad_fraction'] >= .05
    assert allocation['supplement_generic_input_samples'] - allocation['required_generic_input_samples'] < WINDOW
    assert result['metadata']['expected_total_steps'] == 2 + math.ceil(len(result['windows']) / 2)
    assert any(w.condition == 'emotional_nonverbal' for w in result['windows'])
    assert cd.verify_continuation_plan(result, **context, supplement=default_catalog())['passed']


def test_empty_control_exactly_preserves_tail_and_does_not_claim_five_percent(tmp_path):
    context = make_original(tmp_path)
    catalog = synthetic_catalog()
    result = cd.plan_fixed_window_continuation(**context, supplement=catalog, target_expressive_fraction=0)
    assert result['windows'] == tuple(context['original_plan']['windows'][4:])
    assert result['metadata']['allocation']['mode'] == 'tail_only_control'
    assert result['metadata']['allocation']['achieved_broad_fraction'] == 0
    with pytest.raises(SamplerExhausted):
        cd.plan_fixed_window_continuation(**context, supplement=catalog)


@pytest.mark.parametrize('damage', ['cursor', 'step', 'sampler_identity', 'parent_hash', 'original_window'])
def test_changed_parent_prefix_or_original_identity_rejected(tmp_path, damage):
    context = make_original(tmp_path)
    if damage == 'cursor': context['sampler_state']['cursor'] += 1
    elif damage == 'step': context['parent']['step'] += 1
    elif damage == 'sampler_identity': context['sampler_state']['identity_sha256'] = 'f' * 64
    elif damage == 'parent_hash': context['parent']['checkpoint_sha256'] = 'unbound'
    else:
        values = list(context['original_plan']['windows'])
        values[0] = replace(values[0], condition='Breathing')
        context['original_plan']['windows'] = values
    with pytest.raises(ValueError): plan_context(context)


def test_reserved_people_and_original_hash_aliases_are_excluded_before_selection(tmp_path):
    context = make_original(tmp_path)
    base = default_catalog()
    reserved = replace(row('new-dev', samples=WINDOW, language='und'), split='dev')
    blocked = generic('blocked-person', speaker=reserved.speaker_id)
    alias = replace(generic('old-alias'), audio_sha256=context['original_plan']['rows'][0].audio_sha256)
    catalog = synthetic_catalog([*base['rows'], blocked, alias],
        conditions={**base['conditions'], blocked.source_id: 'emotional_nonverbal', alias.source_id: 'emotional_nonverbal'})
    result = plan_context(context, catalog, reserved_rows=[reserved])
    ids = {w.source_id for w in result['windows']}
    assert blocked.source_id not in ids and alias.source_id not in ids
    assert result['metadata']['allocation']['filtered_candidate_windows'] == {
        'reserved_identity': 1, 'original_or_calibration_interval': 1}


def test_known_new_heldout_overlap_with_original_fails_instead_of_dropping_original(tmp_path):
    context = make_original(tmp_path)
    dev = replace(row('unsafe-dev'), split='dev', speaker_id=context['original_plan']['rows'][0].speaker_id)
    with pytest.raises(ValueError, match='Reserved identity'):
        plan_context(context, reserved_rows=[dev])


def test_inherited_lineage_is_retained_and_blocks_a_new_file_alias(tmp_path):
    inherited_row = row('inherited-old')
    inherited = parent_student_ledger([inherited_row], counts([inherited_row]), checkpoint_sha256='d'*64)
    context = make_original(tmp_path, inherited=inherited)
    base = default_catalog()
    alias = replace(generic('inherited-alias'), audio_sha256=inherited_row.audio_sha256)
    catalog = synthetic_catalog([*base['rows'], alias], conditions={**base['conditions'], alias.source_id: 'emotional_nonverbal'})
    result = plan_context(context, catalog)
    assert result['ledger'].entries[inherited_row.source_id]['retired_whole']
    assert result['ledger'].overlaps(alias, 0, 3040)
    assert alias.source_id not in {w.source_id for w in result['windows']}


def test_unknown_fleurs_placeholder_does_not_become_an_invented_person(tmp_path):
    source = replace(row('original', samples=20*WINDOW+3040, language='en'),
                     speaker_id=None, session_id='fleurs:unknown-session-group:en_us')
    context = make_original(tmp_path, original=source)
    dev = replace(row('new-dev', language='en'), split='dev', speaker_id=None, session_id=source.session_id)
    assert plan_context(context, reserved_rows=[dev])['metadata']['allocation']['target_met']


@pytest.mark.parametrize('damage', ['replay_prefix', 'replay_calibration', 'drop_original', 'reorder_original', 'change_condition', 'ledger', 'original_source'])
def test_contextual_verification_rejects_self_consistently_rehashed_bad_plan(tmp_path, damage):
    context = make_original(tmp_path)
    catalog = default_catalog()
    result = plan_context(context, catalog)
    windows = list(result['windows'])
    if damage == 'replay_prefix': windows.append(context['original_plan']['windows'][0])
    elif damage == 'replay_calibration':
        calibration = context['calibration_rows'][0]
        result['rows'] = (*result['rows'], calibration)
        result['counts'][calibration.source_id] = WINDOW
        windows.append(context['calibration_windows'][0])
    elif damage == 'drop_original':
        windows.pop(next(i for i,w in enumerate(windows) if w.source_id == 'original'))
    elif damage == 'reorder_original':
        positions = [i for i,w in enumerate(windows) if w.source_id == 'original']
        windows[positions[0]], windows[positions[1]] = windows[positions[1]], windows[positions[0]]
    elif damage == 'change_condition':
        i = next(i for i,w in enumerate(windows) if w.condition == 'emotional_nonverbal')
        windows[i] = replace(windows[i], condition='Crying_and_sobbing')
    elif damage == 'ledger': result['ledger'] = parent_student_ledger((), {}, checkpoint_sha256='b'*64)
    else:
        result['rows'] = tuple(replace(r, audio_sha256='f'*64, parent_recording_id='forged-parent') if r.source_id=='original' else r for r in result['rows'])
    result['windows'] = tuple(windows)
    cd._seal_plan(result)
    with pytest.raises(ValueError):
        cd.verify_continuation_plan(result, **context, supplement=catalog)


def test_plan_roundtrip_identity_and_file_tampering(tmp_path):
    context = make_original(tmp_path)
    result = plan_context(context)
    output = tmp_path/'continuation'
    ready = cd.write_continuation_plan(result, output)
    loaded = cd.load_continuation_plan(output)
    assert loaded['identity'] == ready
    assert cd.verify_continuation_plan(loaded, **context, supplement=default_catalog())['passed']
    assert cd._windows_digest(loaded['windows']) == FixedWindowSampler(loaded['windows']).identity_sha256
    values = json.loads((output/'metadata.json').read_text()); values['expected_total_steps'] += 1
    (output/'metadata.json').write_bytes(canonical(values))
    with pytest.raises(ValueError, match='metadata changed'):
        cd.load_continuation_plan(output)


def test_selection_is_deterministic_and_weighted_by_valid_samples_not_clips(tmp_path):
    context = make_original(tmp_path)
    short = generic('short', samples=3040)
    long = generic('long', samples=6*WINDOW)
    catalog = synthetic_catalog([short, long])
    first = plan_context(context, catalog)
    second = plan_context(context, catalog)
    assert first['metadata']['continuation_identity_sha256'] == second['metadata']['continuation_identity_sha256']
    added = [w for w in first['windows'] if w.source_id in {'short', 'long'}]
    assert sum(w.valid_input_samples16k for w in added) == first['metadata']['allocation']['supplement_generic_input_samples']
    assert any(w.source_id == 'long' for w in added)


def test_large_interval_index_handles_sparse_same_parent_without_linear_scans():
    source = row('large', samples=WINDOW*30000)
    index = cd._IntervalIndex()
    for i in range(10000): index.add(source, i*WINDOW*2, i*WINDOW*2+3040)
    index.seal()
    assert index.overlaps(source, 19998*WINDOW, 19998*WINDOW+1)
    assert not index.overlaps(source, 19998*WINDOW+3040, 19999*WINDOW)
    alias = replace(source, source_id='alias', parent_recording_id='other-parent')
    assert index.overlaps(alias, 19998*WINDOW, 19998*WINDOW+1)


def write_catalog_files(directory, catalog):
    directory.mkdir()
    train = b''.join(canonical(r.to_dict()) for r in catalog['rows'])
    dev = b''.join(canonical(r.to_dict()) for r in catalog['reserved'])
    files = {
        'train-candidates-v2.jsonl': train, 'fresh-dev-v2.jsonl': dev,
        'candidate-windows-v2.jsonl.gz': gzip.compress(b''.join(canonical(w.to_dict()) for w in catalog['windows']), mtime=0),
        'candidate-window-source-ids.json': canonical(sorted({w.source_id for w in catalog['windows']})),
        'candidate-window-inventory.json': canonical({'publication_version': 2, 'windows': len(catalog['windows']),
            'sources': len({w.source_id for w in catalog['windows']}), 'context_frames': 29,
            'minimum_valid_input_samples': 3040, 'scored_frames_maximum': 64}),
        'emogator-conditions.json': canonical({'by_source': {k: {'condition': v, 'intended_emotion': 'unknown'}
            for k,v in catalog['conditions'].items() if v == 'emotional_nonverbal'}}),
        'fresh-conditions.json': canonical({k: [v] for k,v in catalog['conditions'].items() if v != 'emotional_nonverbal'}),
        'new-dev-contributor-reservation.json': canonical({'fixture': True})}
    for name,payload in files.items(): (directory/name).write_bytes(payload)
    ready = {'publication_version': 2, 'state': 'verified_staged_not_trained',
             'accepted_files': {name: {'path': str(directory/name), 'sha256': hashlib.sha256(payload).hexdigest()} for name,payload in files.items()}}
    path = directory/'ready-v2.json'; path.write_bytes(canonical(ready))
    return path


def test_loader_verifies_canonical_v2_and_preserves_generic_label(tmp_path):
    path = write_catalog_files(tmp_path/'catalog', default_catalog())
    loaded = cd.load_supplemental_catalog(path, expected_sha256=cd.file_sha(path))
    assert loaded['windows'] == default_catalog()['windows']
    assert loaded['conditions']['generic-0'] == 'emotional_nonverbal'
    with pytest.raises(ValueError, match='reviewed receipt'):
        cd.load_supplemental_catalog(path, expected_sha256='f'*64)


@pytest.mark.parametrize('damage', ['pretty_manifest', 'altered_file', 'version', 'condition', 'short_window'])
def test_loader_rejects_old_publication_and_inconsistent_canonical_data(tmp_path, damage):
    path = write_catalog_files(tmp_path/'catalog', default_catalog())
    ready = json.loads(path.read_text())
    if damage == 'version': ready['publication_version'] = 1
    else:
        name = 'train-candidates-v2.jsonl' if damage in {'pretty_manifest','altered_file'} else 'candidate-windows-v2.jsonl.gz'
        target = path.parent/name
        if damage == 'pretty_manifest':
            target.write_text(''.join(json.dumps(r.to_dict(),indent=2)+'\n' for r in default_catalog()['rows']))
        elif damage == 'altered_file': target.write_bytes(target.read_bytes()+b'\n')
        else:
            windows = list(default_catalog()['windows'])
            windows[-1] = replace(windows[-1], condition='Crying_and_sobbing') if damage == 'condition' else replace(
                windows[-1], valid_input_samples16k=3039, scored_frames=5)
            target.write_bytes(gzip.compress(b''.join(canonical(w.to_dict()) for w in windows),mtime=0))
        if damage != 'altered_file': ready['accepted_files'][name]['sha256'] = cd.file_sha(target)
    path.write_bytes(canonical(ready))
    with pytest.raises(ValueError): cd.load_supplemental_catalog(path)
