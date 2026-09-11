"""Read-only JSON/code provenance inventory. Standard library only, no models."""
from pathlib import Path
import hashlib,json,os

def small(v,depth=0):
    if isinstance(v,dict):
        if depth>=5:return {'_dict_items':len(v),'_keys':list(v)[:15]}
        return {k:small(x,depth+1) for k,x in v.items()}
    if isinstance(v,list):
        if len(v)>8:return {'_list_items':len(v),'_first':small(v[:2],depth+1)}
        return [small(x,depth+1) for x in v]
    return v

result={'scope':'Read-only file bytes and JSON parsing, no Torch import or GPU calls','roots':[]}
for base in sorted(Path('/workspace').glob('fast-audiovae-convnext-*')):
    entry={'path':str(base),'code':[],'receipts':[]}
    for directory,dirs,files in os.walk(base):
        rel=Path(directory).relative_to(base); depth=len(rel.parts)
        dirs[:]=[d for d in dirs if not d.startswith('.') and d not in {'assets','audio','clips','wav','flac','downloads','node_modules','tensorboard','expressive','heldout-languages','diagnostics-v1','features'}]
        if depth>=4:dirs.clear()
        for name in files:
            p=Path(directory)/name
            if name in {'batched_teacher.py','source_corpus.py','corpus_training.py','source_training.py','prepare_targets.py','preflight_distillation.py','recipe_v2_pilot.py','recipe_v2_continuation.py','representative_pilot.py','run_fusion_screen.py','run_corrected_screen.py','run_expanded_pilot.py'}:
                raw=p.read_bytes();s=raw.decode();entry['code'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':p.stat().st_mtime_ns,
                    'prefetch_calls':s.count('.prefetch('),'actual_batch_checks':'each_actual_batch_vs_serial_fp32_v2' in s,
                    'old_single_qualification':'serial_vs_right_padded_whole_utterance_fp32_v1' in s,
                    'encode_call_lines':[{'line':i,'text':line.strip()} for i,line in enumerate(s.splitlines(),1) if any(x in line for x in ('.prefetch(','.encode(','.get(', 'prepare_utterance_cache('))][:30]})
            if name not in {'run.json','status.json','training-launch.json','initial-train-prefill.json','teacher-prefill.json','teacher-coverage.json','heldout-targets.json','ready.json','summary.json','parent-receipt.json','target-cache.json','identity.json','committed-handoff.json','handoff.json'}:continue
            if name in {'identity.json','ready.json'} and depth>3:continue
            if p.stat().st_size>50_000_000:continue
            try:
                raw=p.read_bytes();doc=json.loads(raw)
                entry['receipts'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'document':small(doc)})
            except (ValueError,OSError):continue
    result['roots'].append(entry)
print(json.dumps(result,sort_keys=True,allow_nan=False))
