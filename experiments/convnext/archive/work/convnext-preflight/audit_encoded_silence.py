"""Separate synthetic boundary checks through the real frozen encoder on CPU."""
import hashlib
import json
from pathlib import Path

import torch

from audiovae_student.data import ManifestRow
from audiovae_student.source_corpus import read_native_16k
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import quiet_window_metrics
from audiovae_student.restart_data import file_sha
from audiovae_student.teacher import FrozenAudioVAE2


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
checkpoint = BASE / 'remediation/monitoring/checkpoints/checkpoint-step005000.pt'
torch.set_num_threads(1)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = StudentDecoder(StudentConfig(**saved['engine']['model_config']))
model.load_state_dict(saved['engine']['model'], strict=True)
model.eval()
teacher = FrozenAudioVAE2.from_files(OLD / 'assets/audio_vae_v2.py', OLD / 'assets/audiovae.pth', device='cpu')
before = {'teacher': state_fingerprint(teacher.model.state_dict()), 'student': state_fingerprint(model.state_dict())}
metadata = saved['identity']['heldout_metadata']
source = next(r for r in saved['identity']['data']['heldout']['rows']
              if metadata[r['source_id']]['condition'] == 'speech' and r['duration_seconds'] >= 1)
row = ManifestRow.from_dict(source)
raw = Path(row.audio_path).read_bytes()
assert hashlib.sha256(raw).hexdigest() == row.audio_sha256
speech = read_native_16k(raw, row)[..., :16000].clone()
cases = [('digital_zero_1s', torch.zeros(1, 1, 16000)),
         ('digital_zero_6s', torch.zeros(1, 1, 96000)),
         ('zero_1s_then_reserved_speech_1s', torch.cat((torch.zeros_like(speech), speech), dim=-1))]
rows = []
with torch.no_grad():
    for name, audio in cases:
        z = teacher.encode(audio)
        target = teacher.decode(z)[..., :audio.shape[-1] * 3]
        prediction = model(z)[..., :target.shape[-1]]
        state, pieces = model.initial_state(), []
        for start in range(z.shape[-1]):
            piece, state = model.forward_stream(z[..., start:start + 1], state)
            pieces.append(piece)
        streaming = torch.cat(pieces, dim=-1)[..., :target.shape[-1]]
        torch.testing.assert_close(streaming, prediction, rtol=3e-5, atol=3e-5)
        quiet = quiet_window_metrics(prediction, target)
        qrows = [w for w in quiet['windows'] if w['is_quiet']]
        row = {'case': name, 'input_sha256': hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
            'input_samples': audio.numel(), 'output_samples': prediction.numel(),
            'latents_are_actual_encoder_outputs': True, 'latent_rms': float(z.square().mean().sqrt()),
            'teacher_rms': float(target.square().mean().sqrt()),
            'student_rms': float(prediction.square().mean().sqrt()),
            'residual_rms': float((prediction - target).square().mean().sqrt()),
            'peak_abs': float(prediction.abs().max()), 'full_scale_samples': int((prediction.abs() >= 1).sum()),
            'quiet_window_count': quiet['quiet_window_count'], 'quiet_failed_count': quiet['quiet_failed_count'],
            'mean_quiet_residual': sum(w['residual_rms'] for w in qrows) / len(qrows) if qrows else None,
            'streaming_max_abs_error': float((streaming - prediction).abs().max()),
            'startup_residual_rms_first_200ms': float((prediction[..., :9600] - target[..., :9600]).square().mean().sqrt())}
        if name == 'digital_zero_6s':
            row['steady_residual_rms_last_2s'] = float((prediction[..., -96000:] - target[..., -96000:]).square().mean().sqrt())
        if 'reserved_speech' in name:
            row['reserved_speech_source_id'] = source['source_id']
        rows.append(row)
after = {'teacher': state_fingerprint(teacher.model.state_dict()), 'student': state_fingerprint(model.state_dict())}
assert before == after
result = {'step': saved['engine']['step'], 'checkpoint_sha256': file_sha(checkpoint),
    'device': 'cpu', 'threads': 1, 'rows': rows, 'weights_unchanged': before == after,
    'optimizer_updates': 0, 'original_fixed_panel_changed': False,
    'scope': 'Three synthetic encoder-boundary diagnostics, separate from natural held-out quality scores',
    'silence_not_represented_by_zero_latent_vectors': True, 'synthetic_audio_entered_training': False}
out = BASE / 'remediation/encoded-silence-audit.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
