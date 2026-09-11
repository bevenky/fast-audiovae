"""Deterministic training-only calibration selection from existing audited metadata."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
selection_path = ROOT / 'outputs/convnext-recovery/head-experiments/selection.json'
coverage_path = ROOT / 'outputs/convnext-recovery/teacher-refinements/source-coverage.json'
selection = json.loads(selection_path.read_text())
coverage = json.loads(coverage_path.read_text())
assert selection['identity_sha256'] == coverage['selection_identity_sha256']
fit = selection['splits']['fit']['sources']
metadata = coverage['splits']['fit']['source_map']
by_id = {row['source_id']: row for row in fit}
chosen = []
reasons = {}
def add(sid, reason):
    assert sid in by_id
    if sid not in chosen:
        chosen.append(sid)
    reasons.setdefault(sid, []).append(reason)
for row in fit[:32]:
    add(row['source_id'], 'retains_previous_calibration_anchor')
indic = ['as_in','bn_in','brx','doi','gu_in','hi_in','kn_in','kok','ks','mai','ml_in','mni','mr_in','ne_np','or_in','pa_in','sa','sat','sd_in','ta_in','te_in','ur_pk']
other = ['en','es_419','pt_br','cmn_hans_cn','ja_jp','fr_fr','ar_eg']
for lang in indic + other:
    candidates = [r for r in fit if r['language'] == lang]
    assert candidates, lang
    present = [r for r in candidates if r['source_id'] in chosen]
    add((present or candidates)[0]['source_id'], 'language:' + lang)
labels = list(coverage['splits']['fit']['verified_label_counts'])
for tag in labels:
    candidates = [sid for sid, row in metadata.items() if tag in row['verified_labels']]
    candidates.sort(key=lambda sid: (metadata[sid]['crop']['start_frame'] != 0,
        metadata[sid]['crop']['actual_scored_samples'] >= 122874,
        -metadata[sid]['crop']['actual_scored_samples'] if metadata[sid]['crop']['actual_scored_samples'] < 122874 else metadata[sid]['crop']['actual_scored_samples'], sid))
    assert candidates, tag
    # Prefer whole short recordings of useful duration; annotations remain recording-level.
    for sid in candidates[:2]:
        add(sid, 'expressive_source:' + tag)
target = ((len(chosen) + 7) // 8) * 8
for row in fit:
    if len(chosen) == target:
        break
    if row['source_id'] not in chosen:
        add(row['source_id'], 'complete_calibration_batch')
assert len(chosen) == target and len(set(chosen)) == target
# Interleave expressive and ordinary speech sources without altering fitting order.
rng = random.Random(20260910)
events = [sid for sid in chosen if metadata[sid]['verified_labels']]
speech = [sid for sid in chosen if not metadata[sid]['verified_labels']]
rng.shuffle(events); rng.shuffle(speech)
batches = [[] for _ in range(target//8)]
for bucket in (events, speech):
    for sid in bucket:
        options = [i for i,b in enumerate(batches) if len(b) < 8]
        index = min(options, key=lambda i:(len(batches[i]),i))
        batches[index].append(sid)
source_ids = [sid for b in batches for sid in b]
assert all(len(b) == 8 for b in batches)
assert set(source_ids) == set(chosen)
assert set(source_ids).isdisjoint({r['source_id'] for r in selection['splits']['validation']['sources']})
rows = [{'source_id':sid,'fit_order_index':next(i for i,r in enumerate(fit) if r['source_id']==sid),
         'reasons':reasons[sid],**metadata[sid]} for sid in source_ids]
out = {
 'format_version':'teacher_l1_calibration_sources_v1',
 'source_ids':source_ids,'source_count':len(source_ids),'batch_size':8,
 'selection_identity_sha256':selection['identity_sha256'],
 'sources_sha256':hashlib.sha256(json.dumps(source_ids,separators=(',',':')).encode()).hexdigest(),
 'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (selection_path,coverage_path)},
 'indic_languages':indic,'additional_requested_languages':other,
 'label_counts':{tag:sum(tag in metadata[sid]['verified_labels'] for sid in source_ids) for tag in labels},
 'all_rows_training_fit_only':True,'fitting_order_unchanged':True,
 'policy':'Calibration only. Fixed source metadata and deterministic selection, no selection/canonical outcomes used. Short start-zero sources preferred to reduce event/crop uncertainty. Source labels are not dense event timestamps; scored active audio and whole-source coverage require separate verification. Original 2048 fitting sources/order and 256 selection sources unchanged.',
 'rows':rows,
}
path = ROOT / 'outputs/convnext-recovery/teacher-l1-v1/calibration-sources.json'
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'source_count':len(source_ids),'labels':out['label_counts'],'languages':len(set(r['language'] for r in rows)),'batches':len(batches),'manifest':str(path)},indent=2))
