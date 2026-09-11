"""Stage small official IndicVoices validation samples without touching training.

Run on the pod only. Parquet audio is read by bounded row group over HTTP ranges;
no archive, model, teacher, shard cache, or GPU is used. All source bytes survive
unchanged, and every selected recording is checked against the entire fixed
optimization and normalization-calibration source manifests, not exposure logs.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq
from audiovae_student.acquire_indic import REPO, REVISION, LANGUAGES, _COLUMNS, source_identifiers, make_row
from audiovae_student.data import load_manifest, validate_manifest

R9 = Path('/workspace/fast-audiovae-convnext-20260909-r9')
R1 = Path('/workspace/fast-audiovae-convnext-20260908-r1')
ROOT = R9/'remediation/heldout-languages'
CAP = 400 * 1024**2
FREE = 4 * 1024**3
GROUP_CAP = 100 * 1024**2
LANGS = ['bodo', 'dogri', 'konkani', 'kashmiri', 'maithili', 'manipuri', 'odia', 'sanskrit', 'santali']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024**2): h.update(block)
    return h.hexdigest()


def check_budget(extra=0):
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    free = shutil.disk_usage(ROOT).free
    if used + extra > CAP or free - extra < FREE:
        raise RuntimeError(f'Staging budget would be exceeded: used={used}, free={free}, extra={extra}')
    return used, free


def write(path, data):
    if isinstance(data, dict): data = (json.dumps(data, sort_keys=True, indent=2)+'\n').encode()
    check_budget(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def keys(row):
    d = row if isinstance(row, dict) else row.to_dict()
    return {(k, d[k]) for k in ['source_id','audio_sha256','parent_recording_id','speaker_id','session_id'] if d.get(k)}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'indic-manifest.jsonl').exists():
        raise RuntimeError('Existing completed manifest must be reviewed, never overwritten by a fresh acquisition')
    check_budget()
    required = [R9/'data/recipe-v2'/split/name for split in ['optimization','calibration']
                for name in ['train-manifest.jsonl','reserved.jsonl','excluded-sources.jsonl']]
    others = [R1/'expanded-pilot/corpus/candidate-dev.jsonl', R1/'expanded-pilot/corpus/source-manifest.jsonl',
              R1/'data-bootstrap/dev.jsonl',R1/'data-expanded-core/dev.jsonl',
              Path('/workspace/fast-audiovae-convnext-20260909-r5/data-expressive-topup-v1/prepared/dev.jsonl')]
    others += [R1/f'{name}/prepared/dev.jsonl' for name in ['data-expressive-originals','data-emogator','data-fsd-vocal','data-human-whistling']]
    paths = required + others
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise RuntimeError(f'Exclusion inventory is incomplete: {missing}')
    files, excluded, all_reserved = [], set(), []
    for path in paths:
        rows = load_manifest(path)
        # Mixed historical inventories contribute their declared evaluation rows;
        # full current train/calibration and their reservations are unconditional.
        filtered = rows if path in required else [r for r in rows if r.split != 'train']
        files.append({'path':str(path), 'sha256':digest(path), 'total_rows':len(rows), 'exclusion_rows':len(filtered)})
        all_reserved.extend(filtered)
        for row in filtered: excluded.update(keys(row))
    selected, selected_keys = [], set()
    report = {'status':'acquiring', 'revision':REVISION,'official_source_split':'valid',
              'target_recordings_per_language':4, 'languages':{}, 'exclusion_files':files,
              'all_optimization_and_calibration_rows_excluded':True,
              'identity_policy':'Reject every known source, SHA256, parent, speaker, or session collision; acoustic re-encoding detection is not claimed',
              'storage_cap_bytes':CAP, 'minimum_free_bytes':FREE, 'started_unix':time.time()}
    api, fs = HfApi(), HfFileSystem()
    with fs.open(f'datasets/{REPO}@{REVISION}/README.md','rb', cache_type='none') as f:
        card = f.read(2*1024**2)
    if b'license: cc-by-4.0' not in card.lower():
        raise RuntimeError('Pinned official card no longer verifies the expected license')
    write(ROOT/'indicvoices-pinned-README.md',card)
    for language in LANGS:
        stats = {'language':LANGUAGES[language], 'selected':0, 'rejected':{}, 'groups_scanned':0}
        rejected = Counter()
        report['languages'][language] = stats
        try:
            files_for_language = sorted(x.path for x in api.list_repo_tree(REPO, repo_type='dataset', revision=REVISION, path_in_repo=language)
                                        if '/valid-' in x.path and x.path.endswith('.parquet'))
            if not files_for_language: raise RuntimeError('No pinned official valid Parquet files')
            for parquet_path in files_for_language[:3]:
                if stats['selected'] >= 4: break
                with fs.open(f'datasets/{REPO}@{REVISION}/{parquet_path}', 'rb', block_size=1024**2, cache_type='none') as stream:
                    parquet = pq.ParquetFile(stream)
                    for group_index in range(min(parquet.num_row_groups,24)):
                        if stats['selected'] >= 4: break
                        check_budget()
                        metadata = parquet.read_row_group(group_index, columns=_COLUMNS).to_pylist()
                        stats['groups_scanned'] += 1
                        candidates = []
                        pending_keys = set()
                        for index, item in enumerate(metadata):
                            try: ids = source_identifiers(item, language)
                            except ValueError:
                                rejected['invalid_published_identity'] += 1; continue
                            k = keys(ids)
                            if k & excluded:
                                rejected['known_existing_identity'] += 1; continue
                            if k & (selected_keys | pending_keys):
                                rejected['same_selected_speaker_or_recording'] += 1; continue
                            if not 2 <= float(item['duration']) <= 20:
                                rejected['outside_2_to_20_seconds'] += 1; continue
                            candidates.append((index,item,ids))
                            pending_keys.update(k)
                            if len(candidates) >= 4-stats['selected']: break
                        if not candidates: continue
                        group = parquet.metadata.row_group(group_index)
                        compressed = sum(group.column(j).total_compressed_size for j in range(group.num_columns)
                                         if group.column(j).path_in_schema == 'audio_filepath.bytes')
                        if compressed > GROUP_CAP:
                            rejected['audio_row_group_over_100_MiB'] += 1; continue
                        blobs = parquet.read_row_group(group_index, columns=['audio_filepath.bytes'])['audio_filepath'].to_pylist()
                        for index, item, ids in candidates:
                            payload = blobs[index]['bytes']
                            audio_path = ROOT/'audio'/language/ids['filename']
                            receipt_path = ROOT/'receipts'/language/(ids['filename']+'.json')
                            row, receipt = make_row(item,payload,language=language,audio_path=audio_path,
                                                    parquet_path=parquet_path,row_group=group_index,row_number=index,access_record=receipt_path)
                            row = replace(row,split='dev',source_split='valid')
                            if keys(row) & (excluded|selected_keys):
                                rejected['decoded_source_hash_or_identity_collision'] += 1; continue
                            validate_manifest([row],reserved_rows=all_reserved)
                            receipt['manifest'] = row.to_dict()
                            receipt['official_source_split'] = 'valid'
                            receipt['provenance_policy'] = 'Untouched embedded original audio; canonical speaker/session retained; never assigned to training'
                            receipt['pinned_dataset_card_sha256'] = hashlib.sha256(card).hexdigest()
                            write(audio_path,payload)
                            if digest(audio_path) != row.audio_sha256: raise RuntimeError('Written audio hash mismatch')
                            write(receipt_path,receipt)
                            selected.append(row); selected_keys.update(keys(row)); stats['selected'] += 1
                            write(ROOT/'indic-staged.jsonl', ''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in selected).encode())
                        del blobs
            stats['status'] = 'ready_for_panel_review' if stats['selected'] >= 2 else 'coverage_gap'
        except Exception as error:
            stats['status'] = 'blocked'
            stats['error_type'] = type(error).__name__
            stats['error'] = str(error).split('?')[0][:600]
        stats['rejected'] = dict(rejected)
        stats['seconds'] = sum(r.duration_seconds for r in selected if r.language == LANGUAGES[language])
        write(ROOT/'indic-progress.json',report)
        print(json.dumps({language:stats}),flush=True)
    if selected:
        validate_manifest(selected,reserved_rows=all_reserved)
        for row in selected:
            if digest(Path(row.audio_path)) != row.audio_sha256: raise RuntimeError('Post-download source integrity failed')
        write(ROOT/'indic-manifest.jsonl',''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in selected).encode())
    for file in report['exclusion_files']:
        if digest(Path(file['path'])) != file['sha256']: raise RuntimeError('Exclusion plan changed during acquisition')
    report.update({'status':'completed_staging', 'selected_recordings':len(selected), 'seconds':sum(r.duration_seconds for r in selected),
                   'known_speakers':len({r.speaker_id for r in selected}), 'known_sessions':len({r.session_id for r in selected}),
                   'validation':{'manifest_schema_and_reserved_identities': bool(selected),'source_bytes_hashes':bool(selected),
                                 'exclusion_files_unchanged':True,'no_training_or_teacher_operations_requested':True},
                   'storage_used_bytes':check_budget()[0], 'workspace_free_bytes':check_budget()[1], 'completed_unix':time.time()})
    write(ROOT/'indic-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='exclusion_files'}),flush=True)


if __name__ == '__main__': main()
