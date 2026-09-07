"""Fetch six SHA256-pinned ONNX Runtime 1.29.0 C/C++ API headers.

Only an explicit call to fetch() or this CLI can access the network. Existing
headers are verified and reused; --offline requires all six files locally.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.29.0'
BASE_URL = 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.29.0/include/onnxruntime/core/session/'
HEADER_SHA256 = {
    'onnxruntime_c_api.h': 'acc0cf4b3f28d39339c76770d76164bb7a0637dc89f5fde764b4017b632f6743',
    'onnxruntime_cxx_api.h': '9c63ed5bf0427ddf1c10a99aca0beeec77318ce77d2ea6f7ae76d7d933660f67',
    'onnxruntime_cxx_inline.h': 'c508dbb7d8203003a2584b8712118cb88e5b4ef1e1c7876a9974156fb7489301',
    'onnxruntime_ep_c_api.h': 'e6c986c9e98583f8113b2c6bc3864814883b806d501cf24da4d239c45753e235',
    'onnxruntime_error_code.h': '5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721',
    'onnxruntime_float16.h': '88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(include):
    """Verify all six local headers without writes or network access."""
    include = Path(include)
    for name, expected in HEADER_SHA256.items():
        path = include/name
        if not path.is_file():
            raise RuntimeError('Missing pinned ORT header: ' + name)
        if sha256(path) != expected:
            raise RuntimeError('ORT header SHA256 mismatch: ' + name)
    return {name: expected for name, expected in HEADER_SHA256.items()}


def fetch(include=None, offline=False):
    """Return include/version/header hashes; download only missing verified files."""
    include = Path(include or ROOT/'.deps/onnxruntime/include').resolve()
    if offline:
        hashes = verify(include)
        return {'include': str(include), 'version': VERSION, 'header_sha256': hashes,
                'downloaded': [], 'offline': True}
    include.mkdir(parents=True, exist_ok=True)
    downloaded = []
    with (include.parent/'.headers.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Do not hide corruption by overwriting an existing mismatched header.
        for name, expected in HEADER_SHA256.items():
            path = include/name
            if path.exists() and (not path.is_file() or sha256(path) != expected):
                raise RuntimeError('ORT header SHA256 mismatch: ' + name)
        for name, expected in HEADER_SHA256.items():
            path = include/name
            if path.exists():
                continue
            with urllib.request.urlopen(BASE_URL+name, timeout=60) as response:
                data = response.read(4*1024*1024+1)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Downloaded ORT header SHA256 mismatch: ' + name)
            temporary = path.with_suffix('.download')
            temporary.write_bytes(data)
            temporary.replace(path)
            downloaded.append(name)
        hashes = verify(include)
        record = {'include': str(include), 'version': VERSION, 'header_sha256': hashes,
                  'source_url': BASE_URL, 'downloaded': downloaded, 'offline': False}
        manifest = include.parent/'headers.json'
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(manifest)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include', help='Destination include directory; defaults to .deps/onnxruntime/include')
    parser.add_argument('--offline', action='store_true', help='Verify existing headers without downloads or writes')
    print(json.dumps(fetch(**vars(parser.parse_args())), indent=2))
