"""Locate the first strict teacher target discrepancy in the failed fit batch."""
import json
from pathlib import Path
import torch
from diagnostic_common import load_context, atomic_json, status
from run_update_experiment import load_canonical_panel
from training_overlay import load_training_overlay
from bounded_head import capture_teacher_pre_tanh, TeacherCanonicalMismatch
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.objective_comparison import state_fingerprint

ROOT=Path('/tmp/fast-audiovae-recovery-20260909')
ctx=load_context()
panel=load_canonical_panel(ctx,{'requires_canonical_evaluation':True},ROOT/'canonical-panel-v1/receipt.json')
overlay=load_training_overlay(ctx,'/dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json',required_counts={'targeted_generator':12800})
selection=json.loads((ROOT/'head-experiments-v1/selection.json').read_text())
teacher=ctx.teacher();teacher.model.eval()
before=state_fingerprint(teacher.model.state_dict())
zero=next(c for c in panel['crops'] if c.source_id=='encoded_zero')
with torch.no_grad():
    for _ in range(3): teacher.decode(zero.latents.to(teacher.device))
rows=[]
for position in range(448,512):
    crop=overlay['pools']['targeted_generator'][selection['splits']['fit']['indices'][position]]
    try:
        capture_teacher_pre_tanh(teacher,crop,QuietAudioConfig())
    except TeacherCanonicalMismatch as error:
        row={'fit_position':position,**error.diagnostic}
        rows.append(row);status('capture_discrepancy',**row)
        atomic_json(ROOT/'head-experiments-v1/capture-discrepancy.json',{'failures':rows,'teacher_state_unchanged':state_fingerprint(teacher.model.state_dict())==before})
        break
else:
    raise RuntimeError('Failed range did not reproduce; investigate startup/environment')
