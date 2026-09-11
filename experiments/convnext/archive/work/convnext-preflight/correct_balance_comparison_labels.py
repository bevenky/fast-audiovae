"""Version only JNV/JVNV condition labels; preserve the exact comparison sequence."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r6')
sys.path.insert(0, '/workspace/fast-audiovae-convnext-20260909-r5')
spec = importlib.util.spec_from_file_location('audiovae_student.comparison_data', BASE / 'planning-code/comparison_data.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
from audiovae_student.restart_data import canonical, file_sha

old = mod.load_comparison_plan(BASE / 'data/comparison-v1')
corrected = [replace(w, condition={'jnv': 'jnv_nonverbal', 'jvnv': 'jvnv_verbal_and_nonverbal'}[w.dataset])
             if w.condition == 'speech' and w.dataset in {'jnv', 'jvnv'} else w for w in old['windows']]
changed = []
for previous, updated in zip(old['windows'], corrected):
    before, after = previous.to_dict(), updated.to_dict()
    condition_a, condition_b = before.pop('condition'), after.pop('condition')
    if before != after:
        raise ValueError('Condition-only correction changed a scored interval')
    if condition_a != condition_b:
        changed.append(updated)
metadata = dict(old['metadata'])
metadata['other_language_scope'] = 'Identified-language recordings include JNV nonverbal and JVNV mixed material with explicit condition labels; the other-language bucket is not exclusively ordinary speech'
metadata['mixed_japanese_condition_correction'] = {
    'sources': len({w.source_id for w in changed}), 'windows': len(changed),
    'scored_input_samples': sum(w.valid_input_samples16k for w in changed),
    'scored_seconds': sum(w.valid_input_samples16k for w in changed) / 16000,
    'japanese_fleurs_speech_seconds': sum(w.valid_input_samples16k for w in corrected
                                        if w.language == 'ja' and w.dataset == 'fleurs') / 16000,
    'scored_intervals_and_order_unchanged': True}
plan = {**old, 'windows': corrected, 'metadata': metadata, 'seed': old['identity']['seed']}
provenance = {**old['identity']['provenance'],
              'supersedes_metadata_plan_identity': old['identity']['identity_sha256'],
              'condition_correction_script_sha256': file_sha(Path(__file__)),
              'planner_module_sha256': file_sha(BASE / 'planning-code/comparison_data.py'),
              'version_change': 'Truthful labels for JNV/JVNV; no change to audio, scored windows, order, counts or exclusions'}
ready = mod.write_comparison_plan(plan, BASE / 'data/comparison-v2', provenance=provenance)
loaded = mod.load_comparison_plan(BASE / 'data/comparison-v2')
report = {'ready': ready, 'metadata': loaded['metadata'],
          'parent_excluded_sources': len(loaded['ledger'].entries),
          'reserved_sources': len(loaded['reserved']), 'excluded_diagnostic_sources': len(loaded['excluded'])}
(BASE / 'data-plan-v2.json').write_bytes(canonical(report))
print(json.dumps({'identity': ready['identity_sha256'], **metadata['mixed_japanese_condition_correction']}))
