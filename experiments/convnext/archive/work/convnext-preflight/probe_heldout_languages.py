"""Read official held-out catalogs on the pod; never download audio."""
import json
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq
import urllib.request
import urllib.parse
from audiovae_student.acquire_indic import REPO, REVISION, _COLUMNS

root = Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation/heldout-languages')
root.mkdir(parents=True, exist_ok=True)
report = {'indic_revision': REVISION, 'indic': {}, 'fleurs': {}}
api, fs = HfApi(), HfFileSystem()
for language in ['bodo', 'dogri', 'konkani', 'kashmiri', 'maithili', 'manipuri', 'odia', 'sanskrit', 'santali']:
    try:
        files = sorted([x.path for x in api.list_repo_tree(REPO, repo_type='dataset', revision=REVISION, path_in_repo=language) if '/valid-' in x.path and x.path.endswith('.parquet')])
        with fs.open(f'datasets/{REPO}@{REVISION}/{files[0]}', 'rb', block_size=1024**2, cache_type='none') as stream:
            p = pq.ParquetFile(stream)
            m = p.read_row_group(0, columns=_COLUMNS).to_pylist()
            g = p.metadata.row_group(0)
            audio_bytes = sum(g.column(j).total_compressed_size for j in range(g.num_columns) if g.column(j).path_in_schema == 'audio_filepath.bytes')
            report['indic'][language] = {'files': files, 'row_groups': p.num_row_groups, 'first_group_rows': len(m), 'first_group_audio_compressed_bytes': audio_bytes, 'first_metadata': m[0]}
    except Exception as e:
        report['indic'][language] = {'error_type': type(e).__name__, 'error': str(e).split('?')[0][:500]}
    print(json.dumps({language: report['indic'][language]}), flush=True)
for language in ['cmn_hans_cn', 'ar_eg', 'de_de']:
    try:
        with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':'google/fleurs','config':language,'split':'test','offset':0,'length':4}), timeout=60) as r:
            entry = {'http_status': r.status}
            j = json.load(r)
            entry.update({'num_rows_total':j.get('num_rows_total'), 'first_row_keys':list(j['rows'][0]['row']), 'audio_format': str(j['rows'][0]['row'].get('audio',[]))[:200]})
        report['fleurs'][language] = entry
    except Exception as e:
        report['fleurs'][language] = {'error_type': type(e).__name__, 'error': str(e).split('?')[0][:200]}
    print(json.dumps({language: report['fleurs'][language]}), flush=True)
(root/'catalog-probe.json').write_text(json.dumps(report, indent=2)+'\n')
