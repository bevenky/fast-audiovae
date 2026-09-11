"""One CPU evaluation of an immutable retained checkpoint on the original panel."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine
from audiovae_student.distillation_training import evaluate_crops, waveform_gate
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.representative_pilot import summarize_evaluation
from audiovae_student.monitoring import signal_summary
from audiovae_student.restart_data import canonical

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
step = payload['engine']['step']
out = args.base/'remediation/evaluations'
out.mkdir(parents=True, exist_ok=True)
path = out/f'evaluation-step{step:06d}.json'
if path.exists():
    raise ValueError('This checkpoint already has an evaluation')
engine = RecipeV2Engine(StudentDecoder(StudentConfig(**payload['engine']['model_config'])),
                       recipe=RecipeV2Config(**payload['engine']['recipe']))
engine.load_state_dict(payload['engine'])
engine.model.eval()
before = state_fingerprint(engine.model.state_dict())
saved = torch.load(args.base/'heldout-targets.pt', map_location='cpu', weights_only=True)
crops = [TrainingCrop(**v) for v in saved['crops']]
identity = _crop_identity(crops)
if identity != payload['identity']['data']['heldout']['crops']:
    raise ValueError('Heldout targets do not match the trained run')
rows = evaluate_crops(engine, crops, include_signal_checks=True)
for row, crop in zip(rows, crops, strict=True):
    assert (row['source_id'], row['start_frame']) == (crop.source_id, crop.start_frame)
    teacher = crop.teacher_audio[...,crop.scored_slice]
    row['teacher_peak_abs'] = float(teacher.abs().max())
    row['teacher_samples_at_or_above_full_scale'] = int((teacher.abs() >= 1).sum())
if before != state_fingerprint(engine.model.state_dict()) or identity != _crop_identity(crops):
    raise ValueError('Evaluation mutated model or frozen targets')
with args.checkpoint.open('rb') as handle:
    checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
report = {'step': step, 'device': 'cpu', 'threads': 1, 'optimizer_updates': 0,
          'checkpoint_sha256': checkpoint_hash, 'rows': rows,
          'groups': summarize_evaluation(rows, payload['identity']['heldout_metadata']),
          'gate': waveform_gate(engine, rows), 'signal_summary': signal_summary(rows),
          'model_and_targets_unchanged': True}
temporary = path.with_suffix('.json.tmp')
temporary.write_bytes(canonical(report))
temporary.replace(path)
compact = {k:v for k,v in report.items() if k != 'rows'}
compact['peak_rows'] = [{k:r[k] for k in ('source_id','start_frame','samples','student_peak_abs',
                         'student_clipped_samples','teacher_peak_abs')} for r in rows if r['student_clipped_samples']]
(out/f'summary-step{step:06d}.json').write_bytes(canonical(compact))
print(json.dumps({'step':step,'all':report['groups']['all'],'signals':report['signal_summary'],
                  'model_and_targets_unchanged':True}))
