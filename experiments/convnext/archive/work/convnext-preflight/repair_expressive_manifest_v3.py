"""Normalize only supplemental manifest serialization; keep all audio and v2 intact."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path('/workspace/fast-audiovae-convnext-20260909-r5')
PARENT = ROOT / 'data-expressive-topup-v1/versions/v2'
TARGET = PARENT.parent / 'v3-jsonl'
EXPECTED = {
    'train.jsonl': '286e0f38000a8836b1bab9211f8a14520eb15bc82a4a97cda0a78175b4f285f5',
    'dev.jsonl': '07b2ebdc58c600202463aee25221ce922491a8eac20b53cd647673814c615df6',
}
sys.path.insert(0, str(ROOT))
from audiovae_student.data import ManifestRow, load_manifest, validate_manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def read_consecutive_objects(path):
    text = path.read_text(encoding='utf-8')
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    cursor, rows = 0, []
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        value, cursor = decoder.raw_decode(text, cursor)
        rows.append(ManifestRow.from_dict(value))
    return rows


def write_json(path, data):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    if TARGET.exists():
        raise FileExistsError(f'New version already exists: {TARGET}')
    parent_hashes = {p.name: sha(p) for p in PARENT.iterdir() if p.is_file()}
    parent_ready = json.loads((PARENT / 'ready.json').read_text())
    if any(parent_hashes.get(name) != value for name, value in EXPECTED.items()):
        raise ValueError('Parent manifest differs from reviewed version 2')
    rows, parse_errors = {}, {}
    for split in ('train', 'dev'):
        path = PARENT / (split + '.jsonl')
        try:
            load_manifest(path)
        except ValueError as error:
            parse_errors[split] = str(error)
        else:
            raise ValueError('Parent already parses as normal JSONL; repair no longer applies')
        rows[split] = read_consecutive_objects(path)
        if len(rows[split]) != parent_ready['clips_by_split'][split]:
            raise ValueError('Parent count mismatch')
        if any(row.split != split for row in rows[split]):
            raise ValueError('Row split differs from manifest split')
    validate_manifest(rows['train'] + rows['dev'])
    validate_manifest(rows['train'], reserved_rows=rows['dev'], training_only=True)
    overlaps = {}
    for key in ('source_id', 'audio_sha256', 'parent_recording_id', 'speaker_id', 'session_id'):
        values = [{getattr(row, key) for row in rows[split] if getattr(row, key)}
                  for split in ('train', 'dev')]
        overlaps[key] = len(values[0] & values[1])
    if any(overlaps.values()):
        raise ValueError('Supplement train/development identity overlap')
    audio_hashes, receipts = {}, []
    for row in rows['train'] + rows['dev']:
        receipt_path = Path(row.access_record)
        receipt = json.loads(receipt_path.read_text())
        original = Path(receipt['original_path'])
        prepared = Path(row.audio_path)
        if Path(receipt['prepared_path']) != prepared:
            raise ValueError('Prepared provenance path mismatch')
        for path, expected in ((original, receipt['original_audio_sha256']),
                               (prepared, row.audio_sha256)):
            actual = sha(path)
            if actual != expected:
                raise ValueError(f'Audio hash differs from saved provenance: {path}')
            audio_hashes[str(path)] = actual
        if receipt['prepared_audio_sha256'] != row.audio_sha256:
            raise ValueError('Prepared provenance hash mismatch')
        receipts.append({'source_id': row.source_id, 'provenance_path': str(receipt_path),
                         'provenance_sha256': sha(receipt_path),
                         'original_path': str(original), 'original_sha256': audio_hashes[str(original)],
                         'prepared_path': str(prepared), 'prepared_sha256': audio_hashes[str(prepared)]})
    staging = Path(tempfile.mkdtemp(prefix='.v3-jsonl-', dir=PARENT.parent))
    try:
        for split, values in rows.items():
            path = staging / (split + '.jsonl')
            with path.open('x', encoding='utf-8') as handle:
                for row in values:
                    handle.write(json.dumps(row.to_dict(), sort_keys=True,
                                            separators=(',', ':'), allow_nan=False) + '\n')
            loaded = load_manifest(path)
            if [r.to_dict() for r in loaded] != [r.to_dict() for r in values]:
                raise ValueError('Normalized records or order changed')
        validate_manifest(load_manifest(staging / 'train.jsonl'),
                          reserved_rows=load_manifest(staging / 'dev.jsonl'), training_only=True)
        validate_manifest(load_manifest(staging / 'train.jsonl') + load_manifest(staging / 'dev.jsonl'))
        if {p.name: sha(p) for p in PARENT.iterdir() if p.is_file()} != parent_hashes:
            raise ValueError('Parent version changed during repair')
        if any(sha(path) != value for path, value in audio_hashes.items()):
            raise ValueError('Audio bytes changed during metadata repair')
        ready = {
            'version': 3, 'state': 'normalized_validated_not_trained',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'root': str(TARGET), 'operation': 'JSON serialization only; all records and order preserved',
            'parent': {'path': str(PARENT), 'files_sha256': parent_hashes,
                       'readiness_document': parent_ready},
            'files_sha256': {name: sha(staging / name) for name in EXPECTED},
            'clips': sum(len(values) for values in rows.values()),
            'clips_by_split': {split: len(values) for split, values in rows.items()},
            'seconds_by_split': {split: sum(r.duration_seconds for r in values)
                                 for split, values in rows.items()},
            'contributors_by_split': {split: len({r.session_id for r in values if r.session_id})
                                      for split, values in rows.items()},
            'verified_now': {'standard_load_manifest': True, 'validate_manifest': True,
                             'exact_field_and_order_preservation': True,
                             'cross_split_overlap_counts': overlaps,
                             'parent_files_unchanged': True, 'original_audio_hashes': len(receipts),
                             'prepared_audio_hashes': len(receipts), 'audio_bytes_unchanged': True,
                             'audio_decoded': False, 'clipping_or_energy_recomputed': False},
            'inherited_only': 'Prior selection, per-file licenses, contributor assignment, '
                              'decoding/clipping/energy and acquisition checks remain parent evidence; '
                              'they were not newly established by this serialization repair.',
            'audio_receipts': receipts, 'parent_standard_parser_errors': parse_errors,
            'script_sha256': sha(__file__),
            'validator_source_sha256': sha(ROOT / 'audiovae_student/data.py'),
            'active_training_or_panel_modified': False,
            'audio_downloaded_or_modified': False,
        }
        write_json(staging / 'ready.json', ready)
        if TARGET.exists():
            raise FileExistsError(TARGET)
        os.rename(staging, TARGET)
        result = {k: v for k, v in ready.items() if k not in ('audio_receipts', 'parent')}
        result['parent'] = {'path': str(PARENT), 'files_sha256': parent_hashes}
        result['ready_sha256'] = sha(TARGET / 'ready.json')
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    main()
