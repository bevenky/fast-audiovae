"""Read-only CPU streaming parity for final trained fusion-screen decoders.

Uses saved frozen teacher targets. No GPU, teacher execution, optimizer step,
runtime measurement, model export or checkpoint modification occurs here.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.restart_data import digest, file_sha


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
ARMS = ('control', 'tanh', 'filter', 'short_mel', 'fresh_magnitude', 'complex')
PATTERNS = {'80ms': (2,), '160ms': (4,), 'irregular': (1, 0, 3, 2, 0, 1)}
ATOL = 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_metadata(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_cases(base, screen, parent, identity):
    original_path = base / 'heldout-targets.pt'
    require(file_sha(original_path) == parent['identity']['data']['heldout_targets']['sha256'],
            'Original heldout target hash differs')
    originals = [TrainingCrop(**item) for item in torch.load(original_path, map_location='cpu', weights_only=True)['crops']]
    require(_crop_identity(originals) == parent['identity']['data']['heldout']['crops'],
            'Original heldout crop identity differs')
    metadata = parent['identity']['heldout_metadata']
    nonquiet = [crop for crop in originals if crop.latents.shape[-1] >= 50
        and float(crop.teacher_audio[..., crop.scored_slice].double().square().mean().sqrt()) >= 1e-3
        and metadata.get(crop.source_id, {}).get('condition') == 'speech']
    require(bool(nonquiet), 'No sufficiently long heldout nonquiet speech crop')
    speech = sorted(nonquiet, key=lambda c: (c.source_id, c.start_frame))[0]
    synthetic_path = screen / 'synthetic-targets.pt'
    synthetic = [TrainingCrop(**item) for item in torch.load(synthetic_path, map_location='cpu', weights_only=True)['crops']]
    zero = [crop for crop in synthetic if crop.source_id == 'encoded_zero']
    require(len(zero) == 1, 'Expected exactly one frozen encoded-zero fixture')
    zero = zero[0]
    require(zero.reference16k is not None and torch.count_nonzero(zero.reference16k) == 0,
            'Encoded-zero fixture was not generated from actual zero input')
    require(zero.context_start_frame == zero.context_frames == zero.start_frame == 0,
            'Encoded-zero fixture does not start at the utterance boundary')
    require(zero.valid_scored_samples == zero.latents.shape[-1] * DECODER_HOP > 2 * 48000,
            'Encoded-zero fixture has insufficient duration or a padded target tail')
    expected = {(row['source_id'], row['start_frame']): row for row in identity['heldout']['crops']}
    for crop in (zero, speech):
        _validate_crop(crop)
        require(_crop_identity([crop])[0] == expected[(crop.source_id, crop.start_frame)],
                'Chosen source/latent/target tensors differ from the frozen screen identity')
        require(crop.latents.device.type == crop.teacher_audio.device.type == 'cpu', 'Target tensors must remain CPU')
    return {'encoded_zero': zero, 'heldout_nonquiet_speech': speech}, {
        str(original_path): file_sha(original_path), str(synthetic_path): file_sha(synthetic_path)}


def compare_stream(model, latents, expected, pattern):
    def same_state(left, right):
        return (left.normalization_layout == right.normalization_layout
            and torch.equal(left.started, right.started)
            and len(left.histories) == len(right.histories)
            and all(torch.equal(a, b) for a, b in zip(left.histories, right.histories)))
    pieces, counts, position, index, empty_calls = [], [], 0, 0, 0
    exact_counts, empties_preserve_state = True, True
    with model.stream() as stream:
        initial_state = stream.state
        initial_empty = stream.decode_chunk(latents[..., :0])
        empty_calls += 1
        empties_preserve_state &= initial_empty.shape == (1, 1, 0) and same_state(stream.state, initial_state)
        while position < latents.shape[-1]:
            frames = min(pattern[index % len(pattern)], latents.shape[-1] - position)
            index += 1
            state = stream.state
            chunk = stream.decode_chunk(latents[..., position:position + frames])
            exact_counts &= tuple(chunk.shape) == (1, 1, frames * DECODER_HOP)
            if frames == 0:
                empty_calls += 1
                empties_preserve_state &= same_state(stream.state, state)
            else:
                pieces.append(chunk)
                counts.append({'input_frames': frames, 'output_samples': chunk.shape[-1]})
            position += frames
        final_state = stream.state
        tail = stream.decode_chunk(latents[..., :0])
        empty_calls += 1
        empties_preserve_state &= tail.shape == (1, 1, 0) and same_state(stream.state, final_state)
        flush = stream.flush()
        exact_counts &= flush.shape == (1, 1, 0) and stream.frames_decoded == latents.shape[-1]
        output = torch.cat(pieces, dim=-1)
        exact_counts &= tuple(output.shape) == tuple(expected.shape)
        if exact_counts:
            difference = output.double() - expected.double()
            maximum = float(difference.abs().max())
            rms = float(difference.square().mean().sqrt())
        else:
            maximum = rms = None
        state_finite = all(bool(torch.isfinite(history).all()) for history in stream.state.histories)
    finite = bool(torch.isfinite(output).all()) and state_finite
    return {'pattern_latent_frames': list(pattern), 'input_frames': latents.shape[-1],
        'expected_output_samples': expected.shape[-1], 'actual_output_samples': output.shape[-1],
        'nonempty_calls': len(pieces), 'empty_calls': empty_calls, 'per_call_sample_counts': counts,
        'sample_counts_passed': bool(exact_counts), 'empty_calls_preserve_state': bool(empties_preserve_state),
        'empty_state_comparison': 'exact_tensor_values_and_layout; buffer object identity may change',
        'finite_output_and_state': finite, 'max_abs_difference': maximum, 'rms_difference': rms,
        'absolute_tolerance': ATOL, 'relative_tolerance': 0,
        'passed': bool(exact_counts and empties_preserve_state and finite and maximum <= ATOL)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--screen', type=Path)
    args = parser.parse_args()
    screen = args.screen or args.base / 'remediation/fusion-screen'
    output = screen / 'trained-stream-checks.json'
    require(not output.exists(), 'Refusing to overwrite existing trained-stream evidence')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), 'CUDA must not be initialized')
    receipt = json.loads((screen / 'parent-receipt.json').read_text())
    identity = json.loads((screen / 'experiment-identity.json').read_text())
    parent_path = screen / 'parent.pt'
    parent_sha = file_sha(parent_path)
    require(parent_sha == receipt['checkpoint_sha256'] == identity['parent_sha256'], 'Parent checkpoint differs')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    cases, target_hashes = load_cases(args.base, screen, parent, identity)
    loaded_sources = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('audiovae_student.') and path and Path(path).suffix == '.py':
            path = Path(path)
            loaded_sources[str(path)] = file_sha(path)
            require(loaded_sources[str(path)] == identity['source_sha256'][path.name],
                    'Frozen experiment implementation changed: ' + path.name)
    rows = {}
    for arm in ARMS:
        directory = screen / arm
        complete = json.loads((directory / 'complete.json').read_text())
        after = json.loads((directory / 'after.json').read_text())
        checkpoint_path = directory / 'final.pt'
        checkpoint_sha = file_sha(checkpoint_path)
        require(checkpoint_sha == complete['checkpoint_sha256'], 'Final checkpoint hash differs: ' + arm)
        final = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        require(final['format_version'] == 'fusion_screen_v1' and final['variant'] == arm
            and final['engine']['format_version'] == 'fusion_recipe_v1'
            and final['engine']['fusion_variant'] == arm
            and final['experiment_identity_sha256'] == digest(identity)
            and final['parent_sha256'] == parent_sha
            and final['generator_updates'] == identity['generator_steps'] == complete['updates'] == 200
            and final['engine']['step'] == complete['step'] == after['evaluated_step'],
            'Final checkpoint experiment or update identity differs: ' + arm)
        parameter_sha = state_fingerprint(final['engine']['model'])
        require(parameter_sha == after['model_identity']['parameter_state_sha256'],
                'Final model tensors differ from saved quality report: ' + arm)
        engine, migration = build_fusion_engine(parent['engine'], arm, device='cpu')
        require(json_metadata(migration) == json_metadata(final['migration'])
            == json_metadata(final['engine']['fusion_migration']), 'Variant migration differs: ' + arm)
        # Build the correct function first, then load final trained model tensors.
        # Do not load the experimental checkpoint as an unchanged base decoder.
        engine.model.load_state_dict(final['engine']['model'], strict=True)
        model = engine.model.eval()
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
                'Model escaped CPU FP32')
        require(state_fingerprint(model.state_dict()) == parameter_sha, 'Loaded model differs')
        model_rng = torch.get_rng_state().clone()
        result = {'checkpoint_sha256': checkpoint_sha, 'parameter_state_sha256': parameter_sha,
            'saved_model_identity_sha256': after['model_state_sha256'], 'final_step': final['engine']['step'],
            'generator_updates': final['generator_updates'], 'cases': {}}
        with torch.inference_mode():
            for case_name, crop in cases.items():
                latents = crop.latents.detach().cpu()
                expected = model(latents)
                require(tuple(expected.shape) == (1, 1, latents.shape[-1] * DECODER_HOP),
                        'Batch decoder sample accounting changed')
                result['cases'][case_name] = {
                    'source_id': crop.source_id, 'start_frame': crop.start_frame,
                    'input_frames': latents.shape[-1], 'full_output_samples': expected.shape[-1],
                    'streaming': {name: compare_stream(model, latents, expected, pattern)
                                  for name, pattern in PATTERNS.items()}}
                if case_name == 'encoded_zero':
                    teacher = crop.teacher_audio.detach().cpu().double()
                    prediction = expected.double()
                    start = 2 * 48000
                    residual = prediction[..., start:] - teacher[..., start:]
                    result['encoded_zero_steady'] = {'start_sample': start,
                        'samples': residual.shape[-1], 'teacher_rms': float(teacher[..., start:].square().mean().sqrt()),
                        'student_rms': float(prediction[..., start:].square().mean().sqrt()),
                        'residual_rms': float(residual.square().mean().sqrt()),
                        'residual_mae': float(residual.abs().mean()),
                        'student_peak_abs': float(prediction[..., start:].abs().max()),
                        'teacher_peak_abs': float(teacher[..., start:].abs().max())}
        if arm == 'filter':
            result['trained_filter'] = {'weight_key': 'output_filter.conv.weight',
                'tap_order': 'oldest_to_current_sample',
                'taps': model.output_filter.conv.weight.detach().cpu().flatten().tolist()}
        require(state_fingerprint(model.state_dict()) == parameter_sha, 'Evaluation mutated trained model')
        require(torch.equal(torch.get_rng_state(), model_rng), 'Model inference changed global RNG')
        result.update(model_tensors_preserved=True, inference_rng_preserved=True,
            optimizer_updates=0, teacher_inference=False)
        result['passed'] = all(check['passed'] for case in result['cases'].values() for check in case['streaming'].values())
        rows[arm] = result
        print(json.dumps({'arm': arm, 'passed': result['passed'],
            'max_abs_difference': max(check['max_abs_difference'] for case in result['cases'].values()
                                      for check in case['streaming'].values()),
            'steady_zero_residual_rms': result['encoded_zero_steady']['residual_rms']}), flush=True)
        del engine, model, final
        gc.collect()
    require(file_sha(parent_path) == parent_sha and not torch.cuda.is_initialized(), 'Parent or device policy changed')
    result = {'format_version': 1, 'kind': 'trained_fusion_decoder_streaming_correctness',
        'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'torch_version': str(torch.__version__), 'cuda_initialized': False,
        'parent_checkpoint_sha256': parent_sha, 'experiment_identity_sha256': digest(identity),
        'script_sha256': file_sha(Path(__file__)), 'implementation_sha256': loaded_sources,
        'target_hashes': target_hashes, 'selected_crops': _crop_identity(list(cases.values())),
        'runtime_benchmark': False, 'teacher_inference': False, 'optimizer_updates': 0,
        'output_trimming': False, 'validation_scope': 'two fixed cases per trained arm; no comprehensive quality or encoder streaming claim',
        'checks': len(ARMS) * len(cases) * len(PATTERNS),
        'passed': all(row['passed'] for row in rows.values()), 'arms': rows}
    temporary = output.with_suffix('.json.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({'output': str(output), 'passed': result['passed'], 'checks': result['checks']}), flush=True)


if __name__ == '__main__':
    main()
