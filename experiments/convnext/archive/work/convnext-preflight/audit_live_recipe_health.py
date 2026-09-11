"""Read checkpoint and committed metrics without inference or optimizer work."""
import json
import math
from pathlib import Path
import statistics

import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine, _norm_fingerprint
from audiovae_student.recipe_v2_pilot import implementation_identity
from audiovae_student.restart_data import digest, file_sha


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
RUN = BASE / 'training-runs/decoder-recipe-v2'
torch.set_num_threads(1)
checkpoint = torch.load(RUN / 'latest.pt', map_location='cpu', weights_only=True)
step = checkpoint['engine']['step']
state = checkpoint['engine']
rows = []
for line in (RUN / 'metrics.jsonl').read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        break
    if row['step'] <= step:
        rows.append(row)
assert len(rows) == step
assert digest(rows) == checkpoint['metrics_sha256']
engine = RecipeV2Engine(StudentDecoder(StudentConfig(**state['model_config'])),
                       recipe=RecipeV2Config(**state['recipe']))
engine.load_state_dict(state)
calibration = state['calibration']
names = ('teacher_waveform', 'teacher_mel', 'feature_matching', 'adversarial')


def summary(selected):
    result = {'updates': len(selected), 'step_first': selected[0]['step'],
              'step_last': selected[-1]['step']}
    for key in (*names, 'discriminator', 'gradient_norm', 'discriminator_gradient_norm',
                'learning_rate', 'perceptual_fraction'):
        values = [r[key] for r in selected if key in r]
        if values:
            result[key] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                           'min': min(values), 'max': max(values)}
    result['scaled_component_norm_fractions'] = {}
    for name in names:
        values = [r[name + '/scaled_norm'] / sum(r[n + '/scaled_norm'] for n in names)
                  for r in selected if all(n + '/scaled_norm' in r for n in names)]
        result['scaled_component_norm_fractions'][name] = statistics.mean(values) if values else None
        result[name + '_coefficient_saturation_updates'] = sum(r.get(name + '/saturated', 0) for r in selected)
    result['nonfinite_numeric_values'] = sum(not math.isfinite(v) for r in selected
        for v in r.values() if isinstance(v, (int, float)))
    return result


result = {
    'checkpoint_step': step,
    'strict_engine_load_passed': True,
    'metrics_prefix_matches_checkpoint': True,
    'sampler_cursor': checkpoint['sampler']['cursor'],
    'teacher_coverage': checkpoint['teacher_coverage'],
    'optimizer': state['recipe']['optimizer'],
    'discriminator_updates': state['discriminator_updates'],
    'balancer_updates': state['balancer']['updates'],
    'balancer_mel_cap': state['balancer'].get('mel_cap'),
    'calibration_completed_step': calibration['completed_step'],
    'calibration_fixed_parameters': calibration['report']['parameters_unchanged'],
    'calibration_norm_buffers_unchanged': _norm_fingerprint(state['model']) == calibration['fixed_buffers_sha256'],
    'calibration_stem': calibration['report']['stem'],
    'calibration_final': calibration['report']['final'],
    'statistics_frozen': {n: bool(state['model'][n + '.statistics_frozen']) for n in ('stem_norm', 'affine')},
    'bound_implementation_changes': [name for name, sha in checkpoint['identity']['implementation'].items()
        if implementation_identity().get(name) != sha],
    'historical_1501_1700': summary(rows[1500:1700]),
    'historical_2801_3000': summary(rows[2800:3000]),
    'latest_200': summary(rows[-200:]),
    'optimizer_updates_performed': 0,
    'model_inference_performed': False,
}
destination = BASE / 'remediation/current-health-audit.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
