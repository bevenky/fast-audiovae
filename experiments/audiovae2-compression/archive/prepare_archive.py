#!/usr/bin/env python3
"""Inventory or copy reviewed research text without changing Git or source files.

Default is inventory only. --copy requires a fresh destination archive in each
experiment family. JSON projections are explicitly labelled and retain the
original digest; they are not substitutes for the private raw evidence.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

CODE = {'.py', '.sh', '.c', '.cc', '.cpp', '.h', '.hpp', '.toml', '.yaml', '.yml', '.diff', '.patch', '.lock'}
TEXT = CODE | {'.md', '.html', '.txt', '.json'}
LARGE_REPORT_COLLECTIONS = {
    'rows', 'windows', 'by_source', 'by_source_id', 'sources', 'source_ids',
    'source_rows', 'snippets', 'phase_templates', 'batches', 'source_ledger',
    'records', 'train_records', 'window_records', 'observations', 'crop_metadata',
    'cache_directories', 'workspace_directories', 'shards', 'per_source',
    'per_window', 'files', 'directory_entries',
}
PAYLOAD_NAMES = re.compile(r'^(?:waveforms?|latents?|tensors?|weights?|audio_samples|sample_values|reference_samples|prediction_samples|reference_waveform|prediction_waveform|phase_templates|snippets)$', re.I)
SAFE_VECTOR = re.compile(r'(shape|dims|widths|indices|indexes|selected|selection|kept|channels|histogram|counts|eigenvalues|singular_values|percentiles|quantiles|fractions|source_interval)', re.I)
SIGNATURES = {
    'private_key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'hf_token': re.compile(r'\bhf_[A-Za-z0-9]{24,}\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{28,}\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'literal_bearer': re.compile(r'Bearer\s+[A-Za-z0-9._~+/-]{24,}'),
    'literal_credential': re.compile(r'''\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*['"][A-Za-z0-9_./+=:-]{16,}['"]''', re.I),
    'embedded_audio': re.compile(r'data:audio/|<audio\b', re.I),
    'embedded_binary': re.compile(r'data:application/octet-stream|data:.*?;base64,', re.I),
}
MAX_TEXT = 4 * 1024 * 1024
MAX_JSON = 40 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def project_json(value, path='', removed=None, prune_bulk=False):
    """Drop numerical payloads; reduce bulk collections only in large reports."""
    if removed is None:
        removed = []
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child = f'{path}.{key}' if path else key
            if key.lower() in {'phase_templates', 'snippets'} and isinstance(item, (dict, list)):
                removed.append(child)
                continue
            if prune_bulk and key.lower() in LARGE_REPORT_COLLECTIONS and isinstance(item, (dict, list)):
                removed.append(child)
                continue
            if PAYLOAD_NAMES.match(key) and isinstance(item, list) and any(isinstance(x, (int, float, list)) for x in item):
                removed.append(child)
                continue
            result = project_json(item, child, removed, prune_bulk)
            if result is not _OMIT:
                out[key] = result
        return out
    if isinstance(value, list):
        if value and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            if len(value) > 16 and not SAFE_VECTOR.search(path):
                removed.append(path)
                return _OMIT
        if len(value) > 1024 and not all(isinstance(x, (str, int)) for x in value):
            removed.append(path)
            return _OMIT
        out = []
        for index, item in enumerate(value):
            result = project_json(item, f'{path}[{index}]', removed, prune_bulk)
            if result is not _OMIT:
                out.append(result)
        return out
    return value


_OMIT = object()


def roots(workspace):
    result = []
    for p in sorted((workspace / 'outputs').iterdir()):
        if p.is_dir() and (p.name.startswith('convnext') or p.name.startswith('audiovae2-compression')):
            result.append((p, 'convnext' if p.name.startswith('convnext') else 'audiovae2-compression'))
    for p in sorted((workspace / 'work').iterdir()):
        if not p.is_dir() or any(x in p.name for x in ('venv', 'uv-cache')):
            continue
        if p.name.startswith('convnext') or p.name in {'audiovae2-compression-preflight', 'audiovae2-compression-audit'}:
            result.append((p, 'convnext' if p.name.startswith('convnext') else 'audiovae2-compression'))
    return result


def candidate(data, name, origin):
    suffix = Path(name).suffix.lower()
    if suffix == '.html' and 'research' in origin and 'paper' in Path(name).stem.lower():
        return None, 'third_party_full_paper_manifest_only', []
    if suffix not in TEXT and not Path(name).name.startswith(('LICENSE', 'NOTICE', 'COPYING')):
        return None, 'excluded_format', []
    limit = MAX_JSON if suffix == '.json' else MAX_TEXT
    if len(data) > limit:
        return None, 'oversize', []
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError:
        return None, 'non_utf8', []
    if '\x00' in content:
        return None, 'binary_content', []
    hits = [key for key, pattern in SIGNATURES.items() if pattern.search(content)
            and not (suffix in CODE and key == 'embedded_audio')]
    if hits:
        return None, 'sensitive_or_embedded_payload', hits
    if suffix == '.json':
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            return None, 'invalid_json', []
        # Source plans/configuration retain identity metadata, loss weights,
        # spectral windows and other non-payload collections without rewriting.
        configuration = bool(re.search(r'(config|plan|selection|manifest|recipe|requirements)', name, re.I))
        if len(data) > 2 * 1024 * 1024 and re.search(r'(inventory|dev_clips_info|pp_pnp_ratings|soundata-index)', name, re.I):
            return None, 'bulk_catalog_manifest_only', []
        removed = []
        projected = project_json(value, removed=removed, prune_bulk=len(data) > 2 * 1024 * 1024 and not configuration)
        if projected is _OMIT:
            return None, 'raw_json_array', []
        if removed:
            envelope = {'archive_projection_version': 1, 'original_relative_path': origin,
                        'original_sha256': sha(data), 'omitted_field_paths': removed,
                        'note': 'Labelled projection. Omitted numerical payloads and listed bulk report collections remain outside Git; configuration metadata is retained.',
                        'report': projected}
            encoded = (json.dumps(envelope, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
            return encoded, 'aggregate_projection', []
    return data, 'exact_text', []


def build(workspace, repo):
    records, payloads = [], {}
    family_roots = roots(workspace)
    for source_root, family in family_roots:
        for path in sorted(source_root.rglob('*')):
            if not path.is_file() or path.is_symlink() or any(p in {'__pycache__', '.pytest_cache', '.git', '.venv', 'node_modules'} for p in path.parts):
                continue
            relative = path.relative_to(workspace).as_posix()
            if path.name.endswith(('.tgz', '.tar.gz')):
                container_hash = sha(path.read_bytes())
                with tarfile.open(path, 'r:gz') as archive:
                    members = archive.getmembers()
                    if len(members) > 20000:
                        raise RuntimeError(f'Archive member count exceeds audit bound: {relative}')
                    for member in members:
                        if not member.isfile():
                            continue
                        item = PurePosixPath(member.name)
                        if item.is_absolute() or '..' in item.parts:
                            raise RuntimeError(f'Unsafe archive member path: {relative}')
                        if '__pycache__' in item.parts or member.size > MAX_JSON:
                            continue
                        if Path(member.name).suffix.lower() not in TEXT:
                            continue
                        stream = archive.extractfile(member)
                        data = stream.read()
                        origin = f'{relative}!/{member.name}'
                        destination = f'experiments/{family}/archive/source-bundles/{relative}/{member.name}'
                        yield_record(records, payloads, data, member.name, origin, destination, container_hash)
                records.append({'origin': relative, 'status': 'container_manifest_only', 'original_bytes': path.stat().st_size, 'original_sha256': container_hash})
                continue
            if path.suffix.lower() not in TEXT and not path.name.startswith(('LICENSE', 'NOTICE', 'COPYING')):
                records.append({'origin': relative, 'status': 'excluded_format', 'original_bytes': path.stat().st_size})
                continue
            data = path.read_bytes()
            destination = f'experiments/{family}/archive/{relative}'
            yield_record(records, payloads, data, path.name, relative, destination)
    # Preserve this existing relative link from the method HTML without editing
    # the historical page or duplicating unrelated runtime experiment histories.
    linked_doc = workspace / 'work/fast-audiovae/docs/cpu-kernel-results.md'
    if linked_doc.is_file():
        yield_record(records, payloads, linked_doc.read_bytes(), linked_doc.name,
                     'work/fast-audiovae/docs/cpu-kernel-results.md',
                     'experiments/audiovae2-compression/archive/work/fast-audiovae/docs/cpu-kernel-results.md')
    summary = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
    for record in records:
        summary[record['status']]['files'] += 1
        summary[record['status']]['bytes'] += record.get('archived_bytes', 0)
    return {'version': 'audiovae2_text_archive_selection_v1',
            'selection_roots': [str(p.relative_to(workspace)) for p, _ in family_roots],
            'excludes': ['audio', 'latents', 'weights', 'model binaries', 'raw JSONL/logs', 'embedded media', 'credential signatures', 'large raw numerical JSON collections'],
            'source_files_modified': False, 'git_index_modified': False,
            'summary': dict(summary), 'records': records}, payloads


def yield_record(records, payloads, data, name, origin, destination, container_hash=None):
    record = {'origin': origin, 'original_bytes': len(data), 'original_sha256': sha(data)}
    if container_hash:
        record['container_sha256'] = container_hash
    try:
        selected, status, hits = candidate(data, name, origin)
    except (ValueError, OverflowError):
        selected, status, hits = None, 'nonfinite_json', []
    record['status'] = status
    if hits:
        record['signature_categories'] = hits
    if selected is not None:
        if destination in payloads:
            if payloads[destination] != selected:
                raise RuntimeError(f'Conflicting archive destination: {destination}')
        payloads[destination] = selected
        record.update(destination=destination, archived_bytes=len(selected), archived_sha256=sha(selected))
    records.append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    repo = (args.repo or workspace / 'work/fast-audiovae').resolve()
    manifest_path = args.manifest or workspace / 'outputs/archive-preparation/selection-manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Manifest destination already exists; use a new --manifest.')
    manifest, payloads = build(workspace, repo)
    if args.copy:
        # Complete no-overwrite preflight before creating any archive files.
        collisions = [name for name in payloads if (repo / name).exists()]
        if collisions:
            raise RuntimeError(f'Refusing to overwrite {len(collisions)} archive files.')
        for relative, data in payloads.items():
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            if sha(target.read_bytes()) != sha(data):
                raise RuntimeError('Archive copy verification failed.')
    manifest['copied'] = args.copy
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'copied': args.copy, 'archive_files': len(payloads),
                      'archive_bytes': sum(map(len, payloads.values())),
                      'manifest': str(manifest_path), 'summary': manifest['summary']}, indent=2))


if __name__ == '__main__':
    main()
