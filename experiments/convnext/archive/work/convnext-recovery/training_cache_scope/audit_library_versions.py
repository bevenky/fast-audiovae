"""Read-only active-environment inventory and parallel primary PyPI lookup.

No target package is imported; importlib.metadata reads installed metadata.
No pip install, model calls, CUDA initialization or environment changes occur.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
import json, platform, sys, time, urllib.request, urllib.error
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version, InvalidVersion

CURRENT_TAGS = set(sys_tags())
PYTHON = Version(platform.python_version())


def python_ok(value):
    return not value or SpecifierSet(value).contains(PYTHON, prereleases=True)


def file_compatible(file):
    if file.get('yanked') or not python_ok(file.get('requires_python')):
        return False
    if file.get('packagetype') != 'bdist_wheel':
        return False
    try:
        return bool(parse_wheel_filename(file['filename'])[3] & CURRENT_TAGS)
    except ValueError:
        return False


def lookup(name):
    url = 'https://pypi.org/pypi/' + name + '/json'
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'audio-library-version-audit/1.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {'url': url, 'status': 'not_on_pypi'}
            if attempt == 2:return {'url': url, 'status': 'lookup_failed', 'error': str(error)}
            time.sleep(attempt + 1)
        except (OSError, ValueError) as error:
            if attempt == 2:return {'url': url, 'status': 'lookup_failed', 'error': str(error)}
            time.sleep(attempt + 1)
    info = payload['info']; versions = []
    for raw, files in payload['releases'].items():
        try:v = Version(raw)
        except InvalidVersion:continue
        if not v.is_prerelease and not v.is_devrelease and any(not f.get('yanked') for f in files):
            versions.append((v, raw, files))
    versions.sort(reverse=True)
    latest = versions[0] if versions else (Version(info['version']), info['version'], payload['urls'])
    wheel_version = next((raw for v, raw, files in versions if any(file_compatible(f) for f in files)), None)
    python_version = next((raw for v, raw, files in versions
                           if any(not f.get('yanked') and python_ok(f.get('requires_python')) for f in files)), None)
    files = latest[2]
    wheel = next((f for f in files if file_compatible(f)), None)
    return {'url': url, 'project_url': 'https://pypi.org/project/' + name + '/', 'status': 'ok',
        'latest_stable': latest[1], 'latest_info_version': info['version'],
        'latest_info_requires_python': info.get('requires_python'),
        'latest_info_requires_dist': info.get('requires_dist') or [],
        'latest_python_compatible_distribution': python_version,
        'latest_compatible_wheel': wheel_version,
        'latest_has_compatible_wheel': wheel is not None,
        'latest_compatible_wheel_filename': wheel['filename'] if wheel else None,
        'latest_non_yanked_sdist': any(f.get('packagetype') == 'sdist' and not f.get('yanked')
                                      and python_ok(f.get('requires_python')) for f in files),
        'latest_release_upload_utc': max((f.get('upload_time_iso_8601', '') for f in files), default=None)}


installed = {}
for dist in metadata.distributions():
    name = dist.metadata.get('Name')
    if not name:continue
    canonical = canonicalize_name(name)
    installed[canonical] = {'name': name, 'installed_version': dist.version,
        'installed_requires_python': dist.metadata.get('Requires-Python'),
        'installed_requires_dist': dist.requires or [],
        'installed_metadata_path': str(getattr(dist, '_path', 'unknown')),
        'requested_marker_present': dist.read_text('REQUESTED') is not None}

# Include declared inference dependencies even when this training environment
# does not install them. Missing libraries are not silently treated as current.
additional = ['onnxruntime', 'onnx', 'pesq', 'pystoi', 'torchmetrics', 'torchcodec',
              'dnsmos', 'utmos', 'accelerate', 'safetensors', 'pytest', 'uv',
              'torchaudio', 'scipy', 'librosa', 'datasets']
names = sorted(set(installed) | set(additional))
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(lookup, name): name for name in names}
    lookups = {futures[f]: f.result() for f in as_completed(futures)}

dependency_conflicts = []
for name, row in installed.items():
    row['pypi'] = lookups[name]
    latest = row['pypi'].get('latest_stable')
    if latest:
        current, fresh = Version(row['installed_version']), Version(latest)
        row['version_comparison'] = ('older' if current.base_version != fresh.base_version and current < fresh
                                     else 'newer' if current.base_version != fresh.base_version and current > fresh else 'same_base_version')
    for text in row['installed_requires_dist']:
        try:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra': ''}):continue
            dependency = canonicalize_name(req.name)
            found = installed.get(dependency)
            if found is None or (req.specifier and not req.specifier.contains(found['installed_version'], prereleases=True)):
                dependency_conflicts.append({'package': name, 'requirement': text,
                    'installed_dependency_version': found['installed_version'] if found else None})
        except (ValueError, InvalidVersion) as error:
            dependency_conflicts.append({'package': name, 'requirement': text, 'parse_error': str(error)})

result = {'format_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only active training environment package metadata and primary PyPI JSON; no package installations or GPU calls',
    'python_executable': sys.executable, 'python_version': platform.python_version(),
    'platform': platform.platform(), 'machine': platform.machine(),
    'installed_package_count': len(installed), 'pypi_queries': len(lookups), 'parallel_requests': 8,
    'packages': [installed[name] for name in sorted(installed)],
    'additional_not_installed': {name: lookups[name] for name in additional if name not in installed},
    'active_installed_dependency_conflicts': dependency_conflicts,
    'notes': ['Latest compatible wheel means Python/ABI/platform compatibility only, not CUDA/driver or dependency-solver compatibility.',
              'PyPI latest stable may differ from a framework CUDA-index build; do not independently upgrade NVIDIA companion packages.',
              'Only active non-extra installed Requires-Dist constraints are checked; this is not a complete prospective upgrade solve.']}
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
