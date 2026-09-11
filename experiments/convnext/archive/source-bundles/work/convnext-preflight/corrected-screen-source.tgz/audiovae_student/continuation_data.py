"""Extend one student's source-conservative exposure lineage for a new trial."""
from .data import ManifestRow
from .restart_data import _counts, identity
from .comparison_data import parent_student_ledger

# The earlier whole-file retirement API below remains unchanged. The versioned
# fixed-window APIs after it operate on scored intervals, preserving unused tails.


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


from bisect import bisect_left
from collections import Counter, defaultdict, deque
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from .data import load_manifest, validate_manifest
from .comparison_data import (EVENTS, known_identities, load_comparison_plan,
                              write_comparison_plan)
from .restart_data import (ConsumedLedger, HOP, RATE, PilotWindow, canonical, digest,
                           file_sha, merge_intervals, _order)
from .sampling import SamplerExhausted, normalize_language

_KIND = 'fixed_window_continuation_v1'
_MINIMUM = 3040
_GENERIC = 'emotional_nonverbal'
_EXPLICIT = frozenset(EVENTS)
_BROAD = _EXPLICIT | {_GENERIC}
_SUPPLEMENT_FILES = frozenset({
    'train-candidates-v2.jsonl', 'fresh-dev-v2.jsonl', 'candidate-windows-v2.jsonl.gz',
    'candidate-window-source-ids.json', 'candidate-window-inventory.json',
    'emogator-conditions.json', 'fresh-conditions.json', 'new-dev-contributor-reservation.json'})


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError(name + ' must be a SHA256')
    return value


def _sequence_digest(values):
    """Hash canonical JSON arrays without materializing the complete JSON text."""
    h = hashlib.sha256(b'[')
    for i, value in enumerate(values):
        if i:
            h.update(b',')
        h.update(canonical(value).rstrip(b'\n'))
    h.update(b']\n')
    return h.hexdigest()


def _rows_digest(rows):
    return _sequence_digest(r.to_dict() for r in rows)


def _windows_digest(windows):
    return _sequence_digest(w.to_dict() for w in windows)


def _lines_sha(values):
    h = hashlib.sha256()
    for value in values:
        h.update(canonical(value))
    return h.hexdigest()


def _merge_sources(*groups):
    rows, counts = {}, {}
    for source_rows, source_counts in groups:
        source_rows = tuple(source_rows)
        _counts(source_rows, source_counts)
        for row in source_rows:
            prior = rows.get(row.source_id)
            if prior is not None and (prior.to_dict() != row.to_dict()
                                      or counts[row.source_id] != source_counts[row.source_id]):
                raise ValueError('Source identity/count changed across continuation inputs')
            rows[row.source_id] = row
            counts[row.source_id] = source_counts[row.source_id]
    return rows, counts


class _IntervalIndex:
    """Bulk-built merged intervals with logarithmic overlap queries per key."""
    def __init__(self):
        self.intervals = defaultdict(list)
        self.starts = {}
        self.whole = set()

    def add(self, row, start, stop, *, whole=False):
        origin = row.parent_start_seconds * RATE
        self.intervals[('parent', row.parent_recording_id)].append(
            (math.floor(origin + start), math.ceil(origin + stop)))
        self.intervals[('sha256', row.audio_sha256)].append((start, stop))
        if whole:
            self.whole.update(known_identities(row, people=False))

    def seal(self, *, reject_overlap=False):
        for key, intervals in self.intervals.items():
            ordered = sorted(intervals)
            if reject_overlap:
                last_stop = -1
                for start, stop in ordered:
                    if start < last_stop:
                        raise ValueError('Repeated/overlapping scored interval: ' + str(key))
                    last_stop = stop
            merged = merge_intervals(ordered)
            self.intervals[key] = merged
            self.starts[key] = [pair[0] for pair in merged]
        return self

    def overlaps(self, row, start, stop):
        if known_identities(row, people=False) & self.whole:
            return True
        origin = row.parent_start_seconds * RATE
        for key, lo, hi in (
                (('parent', row.parent_recording_id), math.floor(origin + start), math.ceil(origin + stop)),
                (('sha256', row.audio_sha256), start, stop)):
            starts = self.starts.get(key, ())
            i = bisect_left(starts, hi) - 1
            if i >= 0 and self.intervals[key][i][1] > lo:
                return True
        return False

    @classmethod
    def from_ledger(cls, ledger):
        result = cls()
        for entry in ledger.document['sources']:
            row = ManifestRow.from_dict(entry['row'])
            for start, stop in entry['intervals']:
                result.add(row, start, stop, whole=entry['retired_whole'])
        return result.seal()


def _window_interval(window, rows, counts):
    if not isinstance(window, PilotWindow) or window.source_id not in rows:
        raise ValueError('Window has no canonical source row')
    row = rows[window.source_id]
    if (any(type(v) is not int for v in (window.start_frame, window.scored_frames,
                                       window.valid_input_samples16k))
            or window.start_frame < 0 or not _MINIMUM <= window.valid_input_samples16k <= 64 * HOP
            or window.scored_frames != math.ceil(window.valid_input_samples16k / HOP)
            or window.dataset != row.dataset or window.language != normalize_language(row.language)
            or (window.condition is not None and not isinstance(window.condition, str))):
        raise ValueError('Window violates the original scored-sample contract')
    start = window.start_frame * HOP
    stop = start + window.valid_input_samples16k
    if stop > counts[window.source_id]:
        raise ValueError('Window exceeds its canonical source length')
    return row, start, stop


def _check_windows(windows, rows, counts, *, forbidden=None, reserved=(), excluded=()):
    reserved_keys = set().union(*(known_identities(r) for r in reserved))
    excluded_keys = set().union(*(known_identities(r, people=False) for r in excluded))
    checked_sources = set()
    result = _IntervalIndex()
    for window in windows:
        row, start, stop = _window_interval(window, rows, counts)
        if row.source_id not in checked_sources:
            if (known_identities(row) & reserved_keys
                    or known_identities(row, people=False) & excluded_keys):
                raise ValueError('Reserved identity in continuation: ' + row.source_id)
            checked_sources.add(row.source_id)
        if forbidden is not None and forbidden.overlaps(row, start, stop):
            raise ValueError('Consumed/calibration scored interval replay: ' + row.source_id)
        result.add(row, start, stop)
    return result.seal(reject_overlap=True)


def _catalog_body(catalog):
    return {'kind': 'canonical_expressive_catalog_v2',
            'publication_sha256': catalog['identity']['publication_sha256'],
            'files_sha256': catalog['identity']['files_sha256'],
            'rows_sha256': _rows_digest(catalog['rows']), 'counts_sha256': digest(catalog['counts']),
            'windows_sha256': _windows_digest(catalog['windows']),
            'conditions_sha256': digest(catalog['conditions']),
            'reserved_sha256': _rows_digest(catalog['reserved'])}


def _verify_catalog(catalog):
    body = _catalog_body(catalog)
    if catalog['identity'] != {**body, 'identity_sha256': digest(body)}:
        raise ValueError('Supplemental catalog identity changed')
    if set(body['files_sha256']) != _SUPPLEMENT_FILES:
        raise ValueError('Incomplete canonical v2 supplemental catalog')
    _sha(body['publication_sha256'], 'supplement publication')
    for value in body['files_sha256'].values():
        _sha(value, 'supplement file')
    if catalog['rows']:
        validate_manifest(catalog['rows'], training_only=True)
    _counts(catalog['rows'], catalog['counts'])
    rows = {r.source_id: r for r in catalog['rows']}
    if set(catalog['conditions']) != set(rows):
        raise ValueError('Supplemental conditions do not cover the source manifest')
    for window in catalog['windows']:
        if (window.condition != catalog['conditions'].get(window.source_id)
                or window.condition not in _BROAD):
            raise ValueError('Supplemental action/generic condition changed')
    _check_windows(catalog['windows'], rows, catalog['counts'])


def load_supplemental_catalog(ready_path, *, expected_sha256=None):
    """Read and hash-check canonical v2 metadata only; never read audio/teachers.

    The published receipt establishes which prepared audio was audited earlier.
    Source bytes still require validation by the operational handoff/SourceCorpus.
    """
    ready_path = Path(ready_path).resolve(strict=True)
    publication_sha256 = file_sha(ready_path)
    if expected_sha256 is not None and _sha(expected_sha256, 'expected supplement') != publication_sha256:
        raise ValueError('Supplement publication differs from the reviewed receipt')
    ready = json.loads(ready_path.read_text())
    if (ready.get('publication_version') != 2 or ready.get('state') != 'verified_staged_not_trained'
            or set(ready.get('accepted_files', {})) != _SUPPLEMENT_FILES):
        raise ValueError('Only the complete canonical v2 publication is accepted')
    paths, hashes = {}, {}
    for name, descriptor in ready['accepted_files'].items():
        path = Path(descriptor['path'])
        if (path.is_symlink() or path.name != name or path.resolve().parent != ready_path.parent
                or file_sha(path) != _sha(descriptor['sha256'], name)):
            raise ValueError('Supplemental file/path identity changed: ' + name)
        paths[name], hashes[name] = path, descriptor['sha256']
    def manifest(name):
        rows = load_manifest(paths[name])
        if _lines_sha(r.to_dict() for r in rows) != hashes[name]:
            raise ValueError('Supplement manifest is not canonical JSONL')
        return tuple(rows)
    rows, reserved = manifest('train-candidates-v2.jsonl'), manifest('fresh-dev-v2.jsonl')
    if any(r.split not in {'dev', 'test', 'regression'} for r in reserved):
        raise ValueError('Supplement dev reservation contains training audio')
    windows = []
    with gzip.open(paths['candidate-windows-v2.jsonl.gz'], 'rb') as stream:
        for line in stream:
            value = json.loads(line)
            if canonical(value) != line:
                raise ValueError('Supplement windows are not canonical JSONL')
            window = PilotWindow(**{k: value[k] for k in PilotWindow.__dataclass_fields__})
            if value != window.to_dict():
                raise ValueError('Supplement window fields changed')
            windows.append(window)
    fresh = json.loads(paths['fresh-conditions.json'].read_text())
    generic = json.loads(paths['emogator-conditions.json'].read_text())['by_source']
    conditions = {}
    for row in rows:
        if row.source_id in generic:
            if row.dataset != 'emogator' or generic[row.source_id].get('condition') != _GENERIC:
                raise ValueError('Generic nonverbal provenance changed')
            conditions[row.source_id] = _GENERIC
        else:
            labels = fresh.get(row.source_id)
            if not isinstance(labels, list) or not labels or labels[0] not in _EXPLICIT:
                raise ValueError('Missing reviewed explicit-action condition')
            conditions[row.source_id] = labels[0]
    source_ids = sorted({w.source_id for w in windows})
    if source_ids != json.loads(paths['candidate-window-source-ids.json'].read_text()):
        raise ValueError('Supplement source inventory differs from windows')
    inventory = json.loads(paths['candidate-window-inventory.json'].read_text())
    if (inventory.get('publication_version') != 2 or inventory.get('windows') != len(windows)
            or inventory.get('sources') != len(source_ids) or inventory.get('context_frames') != 29
            or inventory.get('minimum_valid_input_samples') != _MINIMUM
            or inventory.get('scored_frames_maximum') != 64):
        raise ValueError('Supplement inventory differs from the current crop contract')
    result = {'rows': rows, 'counts': {r.source_id: round(r.duration_seconds * RATE) for r in rows},
              'windows': tuple(windows), 'reserved': reserved, 'conditions': conditions,
              'identity': {'publication_sha256': publication_sha256, 'files_sha256': hashes}}
    body = _catalog_body(result)
    result['identity'] = {**body, 'identity_sha256': digest(body)}
    _verify_catalog(result)
    return result


def _verify_original(plan):
    published = plan['identity']
    body = {k: v for k, v in published.items() if k != 'identity_sha256'}
    if published.get('identity_sha256') != digest(body):
        raise ValueError('Original plan identity changed')
    hashes = {
        'train-manifest.jsonl': _lines_sha(r.to_dict() for r in plan['rows']),
        'windows.jsonl': _lines_sha(w.to_dict() for w in plan['windows']),
        'input-sample-counts.json': hashlib.sha256(canonical(plan['counts'])).hexdigest(),
        'ledger.json': hashlib.sha256(canonical(plan['ledger'].document)).hexdigest(),
        'reserved.jsonl': _lines_sha(r.to_dict() for r in plan['reserved']),
        'excluded-sources.jsonl': _lines_sha(r.to_dict() for r in plan['excluded']),
        'metadata.json': hashlib.sha256(canonical(plan['metadata'])).hexdigest()}
    if (hashes != published.get('files_sha256')
            or published.get('fixed_sampler_identity') != _windows_digest(plan['windows'])
            or published.get('ledger_identity') != plan['ledger'].identity_sha256):
        raise ValueError('Original plan contents differ from its published identity')


def _context(original_plan, sampler_state, parent, calibration_windows, calibration_rows,
             calibration_counts, reserved_rows):
    _verify_original(original_plan)
    required = {'checkpoint_sha256', 'run_identity_sha256', 'journal_sha256', 'metrics_sha256', 'step', 'batch_size'}
    if set(parent) != required:
        raise ValueError('Parent binding requires exact checkpoint/run/journal/metrics/step/batch fields')
    for name in required - {'step', 'batch_size'}:
        _sha(parent[name], name)
    if type(parent['step']) is not int or parent['step'] < 0 or type(parent['batch_size']) is not int or parent['batch_size'] < 1:
        raise ValueError('Invalid parent step/batch size')
    windows = tuple(original_plan['windows'])
    if (set(sampler_state) != {'format_version', 'identity_sha256', 'cursor'}
            or sampler_state['format_version'] != 1
            or sampler_state['identity_sha256'] != _windows_digest(windows)
            or type(sampler_state['cursor']) is not int
            or sampler_state['cursor'] != min(parent['step'] * parent['batch_size'], len(windows))
            or parent['step'] > math.ceil(len(windows) / parent['batch_size'])):
        raise ValueError('Parent sampler cursor/step differs from its exact committed prefix')
    calibration_windows, calibration_rows = tuple(calibration_windows), tuple(calibration_rows)
    if not calibration_windows:
        raise ValueError('Original calibration windows are required')
    rows, counts = _merge_sources((original_plan['rows'], original_plan['counts']), (calibration_rows, calibration_counts))
    validate_manifest(rows.values(), training_only=True)
    reserved = tuple(original_plan['reserved']) + tuple(reserved_rows)
    inherited = _IntervalIndex.from_ledger(original_plan['ledger'])
    all_original = _check_windows((*windows, *calibration_windows), rows, counts,
        forbidden=inherited, reserved=reserved, excluded=original_plan['excluded'])
    prefix = windows[:sampler_state['cursor']]
    parent_binding = {**deepcopy(parent), 'original_plan_identity': deepcopy(original_plan['identity']),
                      'sampler_state': deepcopy(sampler_state)}
    calibration_binding = {'windows_sha256': _windows_digest(calibration_windows),
                           'rows_sha256': _rows_digest(calibration_rows), 'counts_sha256': digest(calibration_counts)}
    entries = deepcopy(original_plan['ledger'].entries)
    for window in (*prefix, *calibration_windows):
        row, start, stop = _window_interval(window, rows, counts)
        if row.source_id not in entries:
            entries[row.source_id] = {'row': row.to_dict(), 'input_samples': counts[row.source_id],
                                      'retired_whole': False, 'intervals': []}
        entry = entries[row.source_id]
        if (identity(ManifestRow.from_dict(entry['row'])) != identity(row)
                or entry['input_samples'] != counts[row.source_id]):
            raise ValueError('Inherited source identity changed')
        entry['intervals'].append([start, stop])
    for entry in entries.values():
        entry['intervals'] = merge_intervals(entry['intervals'])
    body = {'format_version': 1, 'input_sample_rate': RATE,
            'sources': [entries[k] for k in sorted(entries)],
            'provenance': {'kind': _KIND, 'parent': parent_binding,
                           'inherited_ledger_sha256': original_plan['ledger'].identity_sha256,
                           'calibration': calibration_binding},
            'exposure_policy': 'Inherited exclusions plus actual committed optimizer prefix and original calibration intervals; causal context is not rescored exposure'}
    ledger = ConsumedLedger({**body, 'identity_sha256': digest(body)})
    return {'rows': rows, 'counts': counts, 'ledger': ledger, 'parent': parent_binding,
            'calibration': calibration_binding, 'reserved': reserved, 'all_original': all_original,
            'inherited': inherited, 'remaining': windows[sampler_state['cursor']:]}


def _plan_body(plan):
    metadata = {k: v for k, v in plan['metadata'].items() if k != 'continuation_identity_sha256'}
    return {'kind': _KIND, 'rows_sha256': _rows_digest(plan['rows']),
            'windows_sha256': _windows_digest(plan['windows']), 'counts_sha256': digest(plan['counts']),
            'ledger_sha256': plan['ledger'].identity_sha256, 'reserved_sha256': _rows_digest(plan['reserved']),
            'excluded_sha256': _rows_digest(plan['excluded']), 'metadata': metadata, 'seed': plan['seed']}


def _seal_plan(plan):
    plan['metadata']['continuation_identity_sha256'] = digest(_plan_body(plan))
    return plan


def _check_plan_seal(plan):
    if (plan['metadata'].get('kind') != _KIND
            or plan['metadata'].get('continuation_identity_sha256') != digest(_plan_body(plan))):
        raise ValueError('Continuation identity changed')


def _catalog_order(windows, rows, seed):
    by_id = defaultdict(list)
    for window in windows:
        by_id[window.source_id].append(window)
    result = []
    for row in _order((rows[k] for k in by_id), seed):
        result.extend(sorted(by_id[row.source_id], key=lambda w: w.start_frame))
    return result


def _interleave_remaining(original, supplemental):
    """Keep original relative order and spread added valid samples through it."""
    queues = [deque(original), deque(supplemental)]
    totals = [sum(w.valid_input_samples16k for w in q) for q in queues]
    emitted = [0, 0]
    result = []
    while queues[0] or queues[1]:
        if not queues[0]: choice = 1
        elif not queues[1]: choice = 0
        else: choice = 0 if emitted[0] * totals[1] <= emitted[1] * totals[0] else 1
        window = queues[choice].popleft()
        emitted[choice] += window.valid_input_samples16k
        result.append(window)
    return tuple(result)


def plan_fixed_window_continuation(original_plan, sampler_state, *, parent, calibration_windows,
        calibration_rows, calibration_counts, supplement, reserved_rows=(),
        target_expressive_fraction=.05, seed=47):
    """Plan metadata only; a runner must independently prove the parent is stopped.

    Retain every original unconsumed window in relative order, use all eligible
    explicit-action windows, then the smallest deterministic generic prefix that
    reaches the requested broad fraction by valid scored samples. No wrapping,
    example-count weighting, invented action labels, or live-state mutation.
    """
    if type(seed) is not int or isinstance(target_expressive_fraction, bool):
        raise ValueError('Invalid continuation allocation configuration')
    fraction = Fraction(str(target_expressive_fraction))
    if not 0 <= fraction < 1:
        raise ValueError('Expressive fraction must be between zero (empty control only) and one')
    _verify_catalog(supplement)
    if fraction == 0 and supplement['windows']:
        raise ValueError('Zero target is reserved for an explicitly empty tail-only control')
    reserved_rows = tuple(reserved_rows) + tuple(supplement['reserved'])
    context = _context(original_plan, sampler_state, parent, calibration_windows,
                       calibration_rows, calibration_counts, reserved_rows)
    original = context['remaining']
    if not original or (fraction > 0 and all(w.condition in _BROAD for w in original)):
        raise ValueError('A broad mixture needs original nonexpressive remainder; declare a new speech supply after the old plan is exhausted')
    combined_rows, combined_counts = _merge_sources((context['rows'].values(), context['counts']),
                                                    (supplement['rows'], supplement['counts']))
    reserved_keys = set().union(*(known_identities(r) for r in context['reserved']))
    excluded_keys = set().union(*(known_identities(r, people=False) for r in original_plan['excluded']))
    eligible, filtered = [], Counter()
    for window in supplement['windows']:
        row, start, stop = _window_interval(window, combined_rows, combined_counts)
        if known_identities(row) & reserved_keys:
            filtered['reserved_identity'] += 1
        elif known_identities(row, people=False) & excluded_keys:
            filtered['excluded_source'] += 1
        elif (context['inherited'].overlaps(row, start, stop)
              or context['all_original'].overlaps(row, start, stop)):
            filtered['original_or_calibration_interval'] += 1
        else:
            eligible.append(window)
    explicit = _catalog_order([w for w in eligible if w.condition in _EXPLICIT], combined_rows, seed)
    generic = _catalog_order([w for w in eligible if w.condition == _GENERIC], combined_rows, seed)
    original_samples = sum(w.valid_input_samples16k for w in original)
    original_broad = sum(w.valid_input_samples16k for w in original if w.condition in _BROAD)
    explicit_samples = sum(w.valid_input_samples16k for w in explicit)
    numerator = (fraction.numerator * original_samples - fraction.denominator * original_broad
                 - (fraction.denominator - fraction.numerator) * explicit_samples)
    needed = max(0, (numerator + fraction.denominator - fraction.numerator - 1)
                 // (fraction.denominator - fraction.numerator))
    selected_generic, generic_samples = [], 0
    for window in generic:
        if generic_samples >= needed: break
        selected_generic.append(window)
        generic_samples += window.valid_input_samples16k
    if generic_samples < needed:
        raise SamplerExhausted('Insufficient once-only generic supply for the declared expressive fraction')
    added = _catalog_order(explicit + selected_generic, combined_rows, seed + 1)
    windows = _interleave_remaining(original, added)
    ids = {w.source_id for w in windows}
    rows = tuple(combined_rows[k] for k in sorted(ids))
    counts = {k: combined_counts[k] for k in sorted(ids)}
    total = original_samples + explicit_samples + generic_samples
    broad = original_broad + explicit_samples + generic_samples
    metadata = {'kind': _KIND, 'parent': context['parent'], 'calibration': context['calibration'],
        'scope': 'Pure immutable metadata plan; stopped-parent checkpoint/journal verification is mandatory at handoff',
        'expected_total_steps': parent['step'] + math.ceil(len(windows) / parent['batch_size']),
        'original_remaining_windows': len(original), 'original_remaining_windows_sha256': _windows_digest(original),
        'supplement_identity': deepcopy(supplement['identity']), 'selected_supplement_windows_sha256': _windows_digest(added),
        'selected_supplement_windows': len(added), 'context_frames': 29, 'minimum_valid_input_samples': _MINIMUM,
        'minimum_output_samples': _MINIMUM * 3,
        'allocation': {'mode': 'tail_only_control' if fraction == 0 else 'broad_expressive',
            'target_fraction_numerator': fraction.numerator, 'target_fraction_denominator': fraction.denominator,
            'original_input_samples': original_samples, 'original_broad_input_samples': original_broad,
            'supplement_explicit_input_samples': explicit_samples, 'supplement_generic_input_samples': generic_samples,
            'required_generic_input_samples': needed, 'total_input_samples': total, 'broad_input_samples': broad,
            'achieved_broad_fraction': broad / total, 'target_met': broad * fraction.denominator >= total * fraction.numerator,
            'overshoot_policy': 'All eligible explicit actions retained; generic prefix overshoots by less than its last selected whole window',
            'generic_is_not_verified_named_action': True, 'filtered_candidate_windows': dict(filtered)},
        'condition_input_samples': dict(Counter({name or 'unlabeled': sum(w.valid_input_samples16k for w in windows if w.condition == name)
                                              for name in sorted({w.condition for w in windows}, key=str)}))}
    result = _seal_plan({'rows': rows, 'counts': counts, 'windows': windows, 'ledger': context['ledger'],
                        'reserved': context['reserved'], 'excluded': tuple(original_plan['excluded']),
                        'metadata': metadata, 'seed': seed})
    verify_continuation_plan(result, original_plan, sampler_state, parent=parent,
        calibration_windows=calibration_windows, calibration_rows=calibration_rows, calibration_counts=calibration_counts,
        reserved_rows=reserved_rows, supplement=supplement)
    return result


def verify_continuation_plan(plan, original_plan, sampler_state, *, parent, calibration_windows,
        calibration_rows, calibration_counts, reserved_rows=(), supplement=None):
    """Contextually reject forged/replayed plans even if their self-hash is valid."""
    _check_plan_seal(plan)
    context = _context(original_plan, sampler_state, parent, calibration_windows, calibration_rows,
                       calibration_counts, reserved_rows)
    metadata = plan['metadata']
    if (metadata['parent'] != context['parent'] or metadata['calibration'] != context['calibration']
            or plan['ledger'].identity_sha256 != context['ledger'].identity_sha256):
        raise ValueError('Continuation parent/prefix/calibration ledger changed')
    required_reservations = {canonical(r.to_dict()) for r in context['reserved']}
    if not required_reservations <= {canonical(r.to_dict()) for r in plan['reserved']}:
        raise ValueError('Continuation dropped required evaluation reservations')
    if tuple(plan['excluded']) != tuple(original_plan['excluded']):
        raise ValueError('Continuation dropped original source exclusions')
    rows, counts = _merge_sources((plan['rows'], plan['counts']))
    _merge_sources((context['rows'].values(), context['counts']), (plan['rows'], plan['counts']))
    if set(rows) != {w.source_id for w in plan['windows']}:
        raise ValueError('Continuation source manifest must exactly cover selected windows')
    validate_manifest(plan['rows'], training_only=True)
    _check_windows(plan['windows'], rows, counts, forbidden=_IntervalIndex.from_ledger(context['ledger']),
                   reserved=plan['reserved'], excluded=plan['excluded'])
    expected = {canonical(w.to_dict()) for w in context['remaining']}
    found = tuple(w for w in plan['windows'] if canonical(w.to_dict()) in expected)
    if found != context['remaining']:
        raise ValueError('Unconsumed originals were dropped, altered, repeated or reordered')
    added = tuple(w for w in plan['windows'] if canonical(w.to_dict()) not in expected)
    if (metadata['original_remaining_windows_sha256'] != _windows_digest(found)
            or metadata['original_remaining_windows'] != len(found)
            or metadata['selected_supplement_windows_sha256'] != _windows_digest(added)
            or metadata['selected_supplement_windows'] != len(added)
            or metadata['expected_total_steps'] != parent['step'] + math.ceil(len(plan['windows']) / parent['batch_size'])):
        raise ValueError('Continuation counts, order or global budget changed')
    for window in added:
        if window.condition not in _BROAD:
            raise ValueError('Supplement has an undeclared nonexpressive condition')
    if supplement is not None:
        _verify_catalog(supplement)
        if metadata['supplement_identity'] != supplement['identity']:
            raise ValueError('Continuation supplemental publication changed')
        allowed = {canonical(w.to_dict()) for w in supplement['windows']}
        if any(canonical(w.to_dict()) not in allowed for w in added):
            raise ValueError('New interval/condition is absent from canonical supplement')
        _merge_sources((plan['rows'], plan['counts']), (supplement['rows'], supplement['counts']))
        if not {canonical(r.to_dict()) for r in supplement['reserved']} <= {canonical(r.to_dict()) for r in plan['reserved']}:
            raise ValueError('Continuation dropped supplemental dev reservations')
    allocation = metadata['allocation']
    original_samples = sum(w.valid_input_samples16k for w in found)
    original_broad = sum(w.valid_input_samples16k for w in found if w.condition in _BROAD)
    explicit = sum(w.valid_input_samples16k for w in added if w.condition in _EXPLICIT)
    generic = sum(w.valid_input_samples16k for w in added if w.condition == _GENERIC)
    total, broad = original_samples + explicit + generic, original_broad + explicit + generic
    for key, value in {'original_input_samples': original_samples, 'original_broad_input_samples': original_broad,
            'supplement_explicit_input_samples': explicit, 'supplement_generic_input_samples': generic,
            'total_input_samples': total, 'broad_input_samples': broad}.items():
        if allocation[key] != value:
            raise ValueError('Continuation valid-sample accounting changed')
    fraction = Fraction(allocation['target_fraction_numerator'], allocation['target_fraction_denominator'])
    if (not 0 <= fraction < 1 or (fraction == 0 and (added or allocation.get('mode') != 'tail_only_control'))
            or broad * fraction.denominator < total * fraction.numerator
            or allocation['target_met'] is not True or allocation['achieved_broad_fraction'] != broad / total):
        raise ValueError('Continuation expressive target is unfulfilled or misstated')
    numerator = fraction.numerator * original_samples - fraction.denominator * original_broad - (fraction.denominator-fraction.numerator)*explicit
    needed = max(0, (numerator + fraction.denominator-fraction.numerator-1)//(fraction.denominator-fraction.numerator))
    if allocation['required_generic_input_samples'] != needed:
        raise ValueError('Required supplemental valid-sample count changed')
    if supplement is not None:
        all_rows, all_counts = _merge_sources((context['rows'].values(), context['counts']),
                                              (supplement['rows'], supplement['counts']))
        reserved_keys = set().union(*(known_identities(r) for r in plan['reserved']))
        excluded_keys = set().union(*(known_identities(r, people=False) for r in plan['excluded']))
        eligible = []
        for window in supplement['windows']:
            row, start, stop = _window_interval(window, all_rows, all_counts)
            if (not known_identities(row) & reserved_keys
                    and not known_identities(row, people=False) & excluded_keys
                    and not context['inherited'].overlaps(row, start, stop)
                    and not context['all_original'].overlaps(row, start, stop)):
                eligible.append(window)
        explicit_expected = _catalog_order([w for w in eligible if w.condition in _EXPLICIT], all_rows, plan['seed'])
        generic_expected, generic_count = [], 0
        for window in _catalog_order([w for w in eligible if w.condition == _GENERIC], all_rows, plan['seed']):
            if generic_count >= needed: break
            generic_expected.append(window)
            generic_count += window.valid_input_samples16k
        if tuple(_catalog_order(explicit_expected + generic_expected, all_rows, plan['seed']+1)) != added:
            raise ValueError('Continuation changed explicit-priority or minimal generic-prefix selection')
    return {'passed': True, 'continuation_identity_sha256': metadata['continuation_identity_sha256'],
            'parent_step': parent['step'], 'segment_windows': len(plan['windows']),
            'expected_total_steps': metadata['expected_total_steps'], 'parent_quiescence_verified': False}


def write_continuation_plan(plan, directory):
    """Publish a new comparison-compatible directory; never replace parent state."""
    _check_plan_seal(plan)
    return write_comparison_plan(plan, directory, provenance={
        'kind': _KIND, 'continuation_identity_sha256': plan['metadata']['continuation_identity_sha256'],
        'parent': plan['metadata']['parent']})


def load_continuation_plan(directory):
    """Load a sealed plan; contextual parent verification is still mandatory."""
    plan = load_comparison_plan(directory)
    plan['seed'] = plan['identity']['seed']
    _check_plan_seal(plan)
    if plan['identity']['provenance'] != {
            'kind': _KIND, 'continuation_identity_sha256': plan['metadata']['continuation_identity_sha256'],
            'parent': plan['metadata']['parent']}:
        raise ValueError('Published continuation provenance changed')
    return plan
