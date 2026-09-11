"""Create a new immutable appendix revision including the Cantonese addition."""
from collections import Counter
import json
import time
from pathlib import Path
from audiovae_student.data import load_manifest,validate_manifest
from audiovae_student.comparison_data import assert_comparison_disjoint
from acquire_heldout_languages import ROOT,write,digest,check_budget

if (ROOT/'language-expanded-ready.json').exists():raise RuntimeError('Expanded appendix is already sealed')
base=json.loads((ROOT/'language-ready.json').read_text())
if digest(Path(base['manifest_path']))!=base['manifest_sha256']:raise RuntimeError('Original sealed appendix changed')
original=load_manifest(base['manifest_path'])
addition_path=ROOT/'cantonese-addition/fleurs-manifest.jsonl'
addition=load_manifest(addition_path) if addition_path.exists() else []
if any(r.language!='yue_hant_hk' for r in addition):raise RuntimeError('Unexpected addition language')
assert_comparison_disjoint(addition,reserved_rows=original)
rows=original+addition
validate_manifest(rows)
if len({r.audio_sha256 for r in rows})!=len(rows):raise RuntimeError('Duplicate new audio bytes')
path=ROOT/'language-expanded-manifest.jsonl'
write(path,''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in rows).encode())
ready={**base,'manifest_path':str(path),'manifest_sha256':digest(path),'recordings':len(rows),
       'recordings_by_language':dict(Counter(r.language for r in rows)),
       'duration_seconds':sum(r.duration_seconds for r in rows),
       'unknown_speaker_rows':sum(r.speaker_id is None for r in rows),
       'missing_requested_languages':([] if len(addition)>=2 else ['yue']),
       'base_manifest_sha256':base['manifest_sha256'],'cantonese_manifest_path':str(addition_path),
       'cantonese_manifest_sha256':digest(addition_path) if addition else None,
       'cantonese_report_path':str(ROOT/'cantonese-addition/fleurs-report.json'),
       'storage_used_bytes':check_budget()[0],'workspace_free_bytes':check_budget()[1],
       'completed_unix':time.time()}
write(ROOT/'language-expanded-ready.json',ready)
print(json.dumps(ready,indent=2))
