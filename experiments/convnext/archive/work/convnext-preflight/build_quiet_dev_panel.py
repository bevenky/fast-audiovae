"""Freeze a metadata-selected development panel; never select by model outputs."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import tarfile
import numpy as np
import soundfile as sf

BASE = Path('/workspace/fast-audiovae-convnext-20260908-r1')
SEL = Path('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1')
OUT = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')
if (OUT/'ready.json').exists():
    raise RuntimeError('Panel is immutable: choose a new version instead of overwriting it')

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def identities(rows, key):
    return {r[key] for r in rows if r.get(key)}

def overlaps(a, b):
    result = {}
    for key in ['source_id', 'audio_sha256', 'parent_recording_id', 'speaker_id', 'session_id']:
        shared = identities(a, key) & identities(b, key)
        result[key] = {'count': len(shared), 'values': sorted(shared)}
    return result

ledger = json.loads((SEL/'ledger.json').read_text())
seen = [s['row'] for s in ledger['sources']]
pilot = read_rows(SEL/'train-manifest.jsonl')
diagnostic = read_rows(SEL/'diagnostic.jsonl') + read_rows(SEL/'sentinel.jsonl')
reserve = read_rows(SEL/'dev-reserve.jsonl')
expressive = []
for name in ['data-expressive-originals', 'data-emogator', 'data-fsd-vocal', 'data-human-whistling']:
    expressive += read_rows(BASE/name/'prepared/dev.jsonl')
bootstrap = read_rows(BASE/'data-bootstrap/dev.jsonl')
blocked = seen + pilot + diagnostic
blocked_keys = {k: identities(blocked, k) for k in ['source_id','audio_sha256','parent_recording_id']}
blocked_known = {k: {v for v in identities(blocked,k) if 'unknown-session-group' not in v}
                 for k in ['speaker_id','session_id']}
dev = {r['source_id']: r for r in reserve + expressive + bootstrap}
eligible = [r for r in dev.values() if not any(r.get(k) in v for k,v in {**blocked_keys,**blocked_known}.items())]

def provenance(row):
    return json.loads(Path(row['access_record']).read_text())

def events(row):
    if row['dataset'].startswith('freesound_human_whistle_'):
        return ['human_whistling_source_description']
    if row['dataset'].startswith('fsd50k_'):
        return provenance(row).get('selected_metadata', {}).get('target_labels', [])
    return []

selected = {}
groups = defaultdict(list)
def choose(candidates, count, group):
    candidates = sorted(candidates, key=lambda r: hashlib.sha256(('quiet-dev-panel-v1:'+group+':'+r['source_id']).encode()).hexdigest())
    picked = []
    used_speakers = set()
    # Prefer distinct known speakers before filling from remaining recordings.
    for pass_number in [0,1]:
        for row in candidates:
            if row['source_id'] in picked or row['source_id'] in selected or row['duration_seconds'] < 1366/16000:
                continue
            speaker = row.get('speaker_id')
            if pass_number == 0 and speaker and speaker in used_speakers:
                continue
            selected[row['source_id']] = row
            picked.append(row['source_id'])
            if speaker: used_speakers.add(speaker)
            groups[group].append(row['source_id'])
            if len(picked) == count:
                return

indic = ['as','bn','gu','hi','kn','ml','mr','ne','pa','sd','ta','te','ur']
for language in indic:
    choose([r for r in eligible if r['dataset']=='indicvoices' and r['language']==language],3,'speech:'+language)
choose([r for r in eligible if r['dataset']=='librispeech'],4,'speech:en')
choose([r for r in eligible if r['dataset']=='jvnv'],4,'speech:ja')
for event in ['Laughter','Screaming','human_whistling_source_description']:
    choose([r for r in eligible if event in events(r)],100,'event:'+event)
choose([r for r in eligible if r['dataset']=='emogator'],12,'generic_emotion_nonverbal')
choose([r for r in eligible if r['dataset']=='jvnv'],6,'jvnv_verbal_and_nonverbal')

# Import six original historical benchmark files, not any model reconstruction.
with tarfile.open(OUT/'legacy-originals-v1.tar.gz') as archive:
    for member in archive.getmembers():
        if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
            raise RuntimeError('Unexpected archive member')
        target = OUT/member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        content=archive.extractfile(member).read()
        if target.exists():
            if target.read_bytes()!=content:
                raise RuntimeError('Refusing to overwrite different retained original')
        else:
            target.write_bytes(content)
legacy_provenance = json.loads((OUT/'legacy-originals-provenance.json').read_text())
old_reserve = json.loads((BASE/'reserved-evaluation.json').read_text())
reserved = {r['uid']:r for r in old_reserve['samples']}
legacy_rows=[]
for sample in legacy_provenance['samples']:
    expected=reserved[sample['uid']]
    for key in ['language','original_audio_path','sha256','dataset_id']:
        assert sample[key] == expected[key], (sample['uid'],key)
    lang=sample['language']
    path=OUT/'originals'/(sample['uid']+'.wav')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sample['sha256']
    row=dict(dataset='fleurs',source_revision=legacy_provenance['revision'],
        source_id=f"{lang}:test:{sample['original_audio_path']}",
        source_url=f"https://huggingface.co/datasets/google/fleurs/resolve/{legacy_provenance['revision']}/data/{lang}/audio/test.tar.gz",
        audio_path=str(path),audio_sha256=sample['sha256'],
        parent_recording_id=f"fleurs:{lang}:audio:{sample['original_audio_path']}",parent_start_seconds=0,
        speaker_id=None,session_id=f'fleurs:unknown-session-group:{lang}',language=lang.split('_')[0],
        sample_rate_hz=16000,original_sample_rate_hz=16000,bandwidth_hz=8000,
        bandwidth_class='speech_band',bandwidth_evidence='Original mono 16 kHz source; 8 kHz Nyquist upper bound, not a measured physical bandwidth estimate',
        native_recording=True,enhanced=False,duration_seconds=sample['samples']/16000,split='dev',source_split='test',
        license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
        attribution='FLEURS: Google Research, Conneau et al.',access_record=str(OUT/'legacy-originals-provenance.json'),
        gain_policy='none: preserve original amplitude',resampler_policy='none: original mono 16000 Hz',teacher_cache_key=None)
    assert not any(row.get(k) in v for k,v in blocked_keys.items())
    selected[row['source_id']]=row
    legacy_rows.append(row)
    groups['speech:'+row['language']].append(row['source_id'])

rows=sorted(selected.values(),key=lambda r:r['source_id'])
assert len(rows)<=96
counts={}
checks=[]
windows=[]
energy=Counter()
for row in rows:
    payload=Path(row['audio_path']).read_bytes()
    actual=hashlib.sha256(payload).hexdigest()
    assert actual==row['audio_sha256'],row['source_id']
    audio,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
    assert rate==16000 and audio.shape[1]==1 and np.isfinite(audio).all()
    samples=len(audio)
    assert abs(samples/16000-row['duration_seconds'])<=1/16000+1e-10
    counts[row['source_id']]=samples
    rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    bins=Counter()
    for start in range(0,samples,320):
        piece=audio[start:start+320].astype(np.float64)
        value=float(np.sqrt(np.mean(piece*piece)))
        db=20*math.log10(max(value,1e-12))
        label='<-100' if db < -100 else '-100:-80' if db < -80 else '-80:-60' if db < -60 else '-60:-40' if db < -40 else '>=-40'
        bins[label]+=len(piece)
        energy[label]+=len(piece)
    checks.append({'source_id':row['source_id'],'audio_sha256':actual,'samples16k':samples,'source_rms':rms,
                   'input_20ms_energy_samples_by_dbfs':dict(bins),'verified_event_labels':events(row)})
    scored_frames=64 if samples>=40960 else 16
    total_frames=(samples+639)//640
    starts=[0]
    interior=max(0,(total_frames-scored_frames)//2)
    if interior>0: starts.append(interior)
    for start in starts:
        valid=min(scored_frames*640,samples-start*640)
        if valid<1:continue
        windows.append({'source_id':row['source_id'],'row_id':row['source_id'],'dataset':row['dataset'],
                        'language':row['language'],'start_frame':start,'scored_frames':scored_frames,
                        'valid_input_samples16k':valid,'valid_output_samples48k':valid*3,
                        'position':'beginning' if start==0 else 'interior'})

checks_overlap={name:overlaps(rows,other) for name,other in [('old_training_ledger',seen),('prepared_pilot',pilot),('diagnostic_and_sentinel',diagnostic)]}
for result in checks_overlap.values():
    assert not any(result[k]['count'] for k in ['source_id','audio_sha256','parent_recording_id'])
# All canonical known speakers/sessions must be disjoint. The original FLEURS
# language-wide unknown placeholders are disclosed, never renamed as speakers.
for result in checks_overlap.values():
    assert not result['speaker_id']['count']
    assert all('fleurs:unknown-session-group:' in key for key in result['session_id']['values'])

files={'manifest.jsonl':''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows),
       'input-sample-counts.json':json.dumps(counts,sort_keys=True,indent=2),
       'windows.jsonl':''.join(json.dumps(r,sort_keys=True)+'\n' for r in windows),
       'source-checks.json':json.dumps(checks,sort_keys=True,indent=2),
       'groups.json':json.dumps(groups,sort_keys=True,indent=2)}
for name,content in files.items():
    path=OUT/name
    if path.exists(): raise RuntimeError('Refusing to overwrite '+str(path))
    path.write_text(content)
report={
    'state':'ready','created_utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'windows':len(windows),
    'whole_source_minutes':sum(counts.values())/16000/60,
    'scored_minutes':sum(w['valid_input_samples16k'] for w in windows)/16000/60,
    'by_language':dict(sorted(Counter(r['language'] for r in rows).items())),
    'by_dataset':dict(sorted(Counter(r['dataset'] for r in rows).items())),
    'groups':{k:len(v) for k,v in groups.items()},
    'files_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files},
    'source_manifests_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SEL/'dev-reserve.jsonl',SEL/'diagnostic.jsonl',SEL/'sentinel.jsonl',SEL/'train-manifest.jsonl',BASE/'reserved-evaluation.json']},
    'old_ledger_identity':ledger['identity_sha256'],'overlap_checks':checks_overlap,
    'source_energy_20ms_input_seconds':{k:v/16000 for k,v in energy.items()},
    'selection_policy':'Deterministic source-metadata ranking before decoding or model scoring; retain all 15 available explicit event dev files; no model-output selection',
    'source_validation':'All selected exact file bytes SHA256 verified, decoded finite mono16k and exact counts, no resampling/gain changes',
    'heldout_policy':'Development only, never a new sealed test. Existing dev and historical benchmark files have informed earlier audits. Source/file/parent identities excluded from old and new training.',
    'fleurs_identity_caveat':'No speaker/session identities in FLEURS; language-wide unknown placeholder retained. Old training shares three unknown groups; this is unverified identity, not evidence of same or different speakers. Original test split and original filename/text-ID exclusions retained.',
    'missing_indic_languages':['brx','doi','kok','ks','mai','mni','or','sa','sat'],
    'missing_requested_language_coverage':['Chinese Mandarin','Chinese Cantonese','Arabic'],
    'other_pilot_language_without_heldout_panel':['German'],
    'missing_explicit_vocal_events':['Crying_and_sobbing','Giggle','Shout','explicit_whisper_style'],
    'source_energy_note':'Input16k energy measured only after selection; teacher48k silence coverage still must be measured during target creation.',
}
report['identity_sha256']=digest(report)
(OUT/'ready.json').write_text(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps(report,indent=2,sort_keys=True))
