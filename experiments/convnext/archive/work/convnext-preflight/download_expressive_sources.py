"""Download the reviewed original archives, with bounded disk use and receipts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

BASE = Path('/workspace/fast-audiovae-convnext-20260908-r1')
DEST = BASE / 'data-expressive-originals'
SOURCES = [
    dict(name='thorsten-emotional-v02', filename='thorsten-emotional_v02.tgz',
         url='https://www.openslr.org/resources/110/thorsten-emotional_v02.tgz',
         license='CC0-1.0', review_url='https://www.openslr.org/110/', max_bytes=600_000_000),
    dict(name='jnv-ver3', filename='jnv_corpus_ver3.zip',
         url='https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip',
         license='CC-BY-SA-4.0',
         review_url='https://sites.google.com/site/shinnosuketakamichi/research-topics/jnv_corpus',
         max_bytes=600_000_000),
    dict(name='vocalsound-44k', filename='vs_release_44k.zip',
         url='https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1',
         license='CC-BY-SA-4.0', review_url='https://github.com/YuanGongND/vocalsound',
         max_bytes=6_000_000_000),
]


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_archive(path, filename):
    if filename.endswith('.zip'):
        if not zipfile.is_zipfile(path):
            raise ValueError('Response is not a ZIP archive')
    else:
        with tarfile.open(path, 'r:gz') as archive:
            if archive.next() is None:
                raise ValueError('Downloaded archive is empty')


def download(source):
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / source['filename']
    status_path = DEST / (source['name'] + '.json')
    if path.exists() and status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get('state') == 'complete' and old.get('url') == source['url']:
            verify_archive(path, source['filename'])
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != old['archive_sha256']:
                raise ValueError('Existing archive hash mismatch: ' + source['name'])
            return old
    status = dict(source, state='downloading', bytes=0, started_at=time.time())
    temporary = path.with_suffix(path.suffix + '.partial')
    write_json(status_path, status)
    try:
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'fast-audiovae-dataset-preparation'})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) > source['max_bytes']:
                raise ValueError('Archive exceeds download bound')
            status['etag'] = response.headers.get('ETag')
            digest = hashlib.sha256()
            last = 0.0
            while chunk := response.read(2**20):
                if status['bytes'] + len(chunk) > source['max_bytes']:
                    raise ValueError('Download exceeded byte bound')
                if shutil.disk_usage(DEST).free < 8 * 2**30:
                    raise RuntimeError('Stopped to preserve 8 GiB disk reserve')
                output.write(chunk)
                digest.update(chunk)
                status['bytes'] += len(chunk)
                if time.time() - last > 5:
                    status['updated_at'] = time.time()
                    write_json(status_path, status)
                    last = time.time()
        verify_archive(temporary, source['filename'])
        temporary.replace(path)
        status.update(state='complete', archive_sha256=digest.hexdigest(), completed_at=time.time(),
                      upstream_checksum_verified=False,
                      selection_status='downloaded original archive; split and audio validation pending')
    except Exception as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}', updated_at=time.time())
    write_json(status_path, status)
    return status


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, SOURCES))
    print(json.dumps(results, indent=2), flush=True)
    raise SystemExit(0 if all(x['state'] == 'complete' for x in results) else 1)
