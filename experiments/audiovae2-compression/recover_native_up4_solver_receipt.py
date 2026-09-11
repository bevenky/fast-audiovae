"""Recover only a collided solver receipt using the unchanged fixed72 fit pass."""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import torch
import diagnose_native_up4_v1 as diagnostic

P=Path('/workspace/fast-audiovae-compression-20260910-v1')
out=Path('/tmp/fast-audiovae-native-up4-reconstruction-v1')
destination=out/'teacher-retained-input-solver-recovered.json'
weights=out/'teacher-retained-input-solver-recovered.pt'
if destination.exists() or weights.exists():raise FileExistsError('Recovery receipts must be new')
launch=json.loads((out/'launch.json').read_text())
protected={str(p):diagnostic.base.sha(p) for p in out.iterdir() if p.is_file()}
if any(diagnostic.base.sha(p)!=v for p,v in launch['protected'].items()):raise ValueError('Original run identity changed')
args=SimpleNamespace(base_out=P/'pilot',manifest=P/'pilot-selection-v1.json',assets=Path('/workspace/fast-audiovae-convnext-20260908-r1/assets'))
diagnostic.base.policy()
_,selection,_,_,_,pools=diagnostic.screen.authenticate_inputs(args)
ids=[c['source_id'] for c in pools['calibration']]
if ids!=launch['fit_source_ids'] or len(ids)!=72:raise ValueError('Original fixed fitting split changed')
teacher=diagnostic.base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
state=diagnostic.resume.continuation.teacher_versions(teacher);started=time.monotonic()
with torch.no_grad():w,b,report=diagnostic.fit_model(None,teacher,pools['calibration'],selection,teacher_selected=True)
if diagnostic.resume.continuation.teacher_versions(teacher)!=state:raise RuntimeError('Teacher changed')
if any(diagnostic.base.sha(p)!=v for p,v in protected.items()):raise RuntimeError('Original results changed')
if any(diagnostic.base.sha(p)!=v for p,v in launch['protected'].items()):raise RuntimeError('Original input files changed')
torch.save({'format':diagnostic.VERSION,'weight':w.cpu(),'bias':b.cpu(),'source_ids':ids,
    'original_launch_sha256':protected[str(out/'launch.json')]},weights)
report.update(recovery_reason='Original solver JSON was overwritten by its fitting-error score sharing the same filename; that score remains unchanged.',
    recovery_scope='Identical teacher-retained-input72-source fit only; no development forward, new variant or neural training',
    original_files_preserved=True,no_neural_training=True,automatic_promotion=False,
    recovery_script_sha256=diagnostic.base.sha(__file__),original_launch_sha256=protected[str(out/'launch.json')],
    existing_fit_score_sha256=protected[str(out/'teacher-retained-input-fit.json')],
    coefficient_sha256=diagnostic.base.sha(weights),elapsed_seconds=time.monotonic()-started)
diagnostic.base.write_json(destination,report)
print(json.dumps({'receipt':str(destination),'sha256':diagnostic.base.sha(destination),'seconds':report['elapsed_seconds'],
    'rank':report['effective_rank'],'condition':report['ridge_condition'],'solver_residual':report['normal_equation_relative_residual']}))
