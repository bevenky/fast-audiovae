"""Seal metadata-only full-plan checks for the independently staged language set."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import soundfile as sf
from audiovae_student.data import load_manifest,validate_manifest
from audiovae_student.comparison_data import assert_comparison_disjoint
from acquire_heldout_languages import R9,ROOT,digest,write,check_budget

parser=argparse.ArgumentParser()
parser.add_argument('--ready',default='language-ready.json')
parser.add_argument('--output',default='language-validation.json')
args=parser.parse_args()
if Path(args.ready).name!=args.ready or Path(args.output).name!=args.output:raise ValueError('Use staging filenames only')
ready=json.loads((ROOT/args.ready).read_text())
path=Path(ready['manifest_path'])
if digest(path)!=ready['manifest_sha256']:raise RuntimeError('Sealed manifest changed')
rows=load_manifest(path)
validate_manifest(rows)
indic_report=json.loads((ROOT/'indic-report.json').read_text())
reserved=[]
for item in indic_report['exclusion_files']:
    p=Path(item['path'])
    if digest(p)!=item['sha256']:raise RuntimeError('Exclusion manifest changed')
    rr=load_manifest(p)
    reserved.extend(rr if '/data/recipe-v2/' in str(p) else [r for r in rr if r.split!='train'])
assert_comparison_disjoint(rows,reserved_rows=reserved)
plans={}
for mode in ['optimization','calibration']:
    base=R9/'data/recipe-v2'/mode
    sources=load_manifest(base/'train-manifest.jsonl')
    byid={r.source_id:r for r in sources}
    count=0; seen=set(); h=hashlib.sha256()
    with (base/'windows.jsonl').open('rb') as f:
        for line in f:
            h.update(line)
            if not line.strip():continue
            w=json.loads(line)
            if w['source_id'] not in byid:raise RuntimeError('Future window is absent from excluded source manifest')
            seen.add(w['source_id']);count+=1
    assert_comparison_disjoint(rows,reserved_rows=sources)
    plans[mode]={'entire_manifest_rows_excluded':len(sources),'manifest_sha256':digest(base/'train-manifest.jsonl'),
                 'entire_future_window_count_checked':count,'window_source_count':len(seen),'windows_sha256':h.hexdigest(),
                 'all_windows_resolve_to_excluded_sources':True,'new_panel_known_identity_overlap':False}
hashes=set()
for row in rows:
    if row.split!='dev' or row.source_split not in {'valid','validation','test'}:raise RuntimeError('Unofficial held-out split')
    if row.license!='CC-BY-4.0' or row.enhanced or row.teacher_cache_key is not None:raise RuntimeError('Unexpected license/enhancement/teacher transformation')
    if digest(Path(row.audio_path))!=row.audio_sha256:raise RuntimeError('Audio changed')
    if row.audio_sha256 in hashes:raise RuntimeError('Duplicate new audio bytes')
    hashes.add(row.audio_sha256)
    receipt=json.loads(Path(row.access_record).read_text())
    if receipt['manifest']!=row.to_dict():raise RuntimeError('Receipt identity does not match manifest')
    with sf.SoundFile(row.audio_path) as a:
        if a.samplerate!=16000 or a.channels!=1 or a.frames!=receipt['source']['decoded_frames']:raise RuntimeError('Source decode identity changed')
        if abs(a.frames/16000-row.duration_seconds)>1e-9:raise RuntimeError('Source duration differs')
result={'status':'passed','manifest_path':str(path),'manifest_sha256':digest(path),'rows':len(rows),
        'by_language':dict(Counter(r.language for r in rows)),'by_official_partition':dict(Counter(r.source_split for r in rows)),
        'complete_plan_checks':plans,'exclusion_snapshot_files':len(indic_report['exclusion_files']),
        'exact_audio_hashes_unique_and_verified':True,'receipt_schema_and_decoded_lengths_verified':True,
        'official_splits_preserved':True,'known_source_hash_parent_speaker_session_disjoint':True,
        'speaker_session_unknown_for_fleurs':sum(r.dataset=='fleurs' for r in rows),
        'limitations':['Four recordings per requested language is a bounded coverage check, not a reliable language-level quality estimate.',
                       'FLEURS speaker/session identities are not published. Its explicit unknown-session placeholders are retained and do not establish unseen-speaker generalization.',
                       'Disjointness covers supplied identities and exact source-byte hashes, not acoustic fingerprint detection of unknown re-encodings.'],
        'stored_bytes':check_budget()[0],'workspace_free_bytes':check_budget()[1]}
write(ROOT/args.output,result)
print(json.dumps(result,indent=2))
