"""Readiness-only audit of the selected completed r7 control on Runpod.

Loads saved evaluations and the exact saved development targets. It performs
cache-alignment verification, two gradient probes and CPU fold/stream checks.
It never constructs a training corpus, updates a model, or measures RTF.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

from audiovae_student.cache import TrainingCrop
from audiovae_student.data import load_manifest
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import DistillationEngine, DistillationTrainingConfig
from audiovae_student.gradient_balancer import GradientBalancerConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.perceptual_readiness import (assess_perceptual_readiness,
    collect_readiness_evidence, evaluation_evidence)
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.representative_pilot import _atomic_json
from audiovae_student.restart_data import file_sha
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import CHECKPOINT_SHA256, FrozenAudioVAE2
from audiovae_student.training import _restore_rng


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r8')
SOURCE_ROOT = Path('/workspace/fast-audiovae-convnext-20260909-r7')
SOURCE_RUN = SOURCE_ROOT / 'training-runs/quiet-phase-v1'
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')
PANEL_HELPER = Path('/workspace/fast-audiovae-convnext-20260909-r5/run_representative_stage.py')
PANEL_HELPER_SHA = '72c249809abeb889a21c0391c65ec0580abd1c0796d157cdc99dbb4c55683c7a'
SOURCE_ARCHIVE_SHA = '21a79bd63654db79b0da68e424f6fba8cbd774759751cc459028da8ffd033bbc'


def selected_parent():
    selection_path, parent_path = BASE / 'selected-parent.json', BASE / 'selected-parent.pt'
    selection = json.loads(selection_path.read_text())
    if selection.get('selected_arm') != 'control':
        raise ValueError('This readiness audit is for the explicitly selected control only')
    source_path = Path(selection['source_checkpoint_path']).resolve(strict=True)
    if source_path != (SOURCE_RUN / 'latest.pt').resolve(strict=True):
        raise ValueError('Selected source is not the completed r7 comparison checkpoint')
    if (file_sha(parent_path) != selection['parent_checkpoint_sha256']
            or file_sha(source_path) != selection['source_checkpoint_sha256']):
        raise ValueError('Selected parent or source checkpoint checksum changed')
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    engine = deepcopy(source['engines']['control'])
    extension = engine.pop('quiet_phase', None)
    if (not isinstance(extension, dict) or extension.get('format_version') != 1
            or extension.get('config', {}).get('gradient_share') != 0
            or extension.get('activation') is not None):
        raise ValueError('Only the explicitly disabled quiet-phase extension may be removed')
    if state_fingerprint(engine) != state_fingerprint(parent['engine']):
        raise ValueError('Extracted parent differs from the selected control core')
    if (engine['step'] != 250 or engine.get('perceptual_start') is not None
            or engine.get('gate') is not None or engine['balancer']['format_version'] != 1):
        raise ValueError('Selected parent must be the completed plain reconstruction control at update 250')
    if state_fingerprint(parent['rng']) != state_fingerprint(source['rng']):
        raise ValueError('Extracted parent RNG differs from its source checkpoint')
    run_identity = json.loads((SOURCE_RUN / 'identity.json').read_text())
    if source['identity'] != run_identity:
        raise ValueError('Source checkpoint and original comparison identity disagree')
    if run_identity['data']['teacher_checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Original comparison did not use the pinned teacher')
    if run_identity['implementation'] != {name: file_sha(SOURCE_ROOT / 'experiments/convnext/audiovae_student' / (name + '.py'))
                                           for name in run_identity['implementation']}:
        raise ValueError('Original frozen comparison implementation changed')
    return parent, selection, run_identity, {
        'selected_parent_sha256': selection['parent_checkpoint_sha256'],
        'source_checkpoint_sha256': selection['source_checkpoint_sha256'],
        'selected_parent_core_sha256': state_fingerprint(engine),
        'disabled_quiet_phase_removed': extension,
        'source_identity_sha256': file_sha(SOURCE_RUN / 'identity.json'),
        'selected_parent_metadata_sha256': file_sha(selection_path),
    }


def load_panel():
    if file_sha(PANEL_HELPER) != PANEL_HELPER_SHA:
        raise ValueError('Original development panel loader changed')
    spec = importlib.util.spec_from_file_location('readiness_panel_loader', PANEL_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows, counts, metadata, windows, audit = helper.load_dev_panel(PANEL)
    if len(rows) != 84 or len(windows) != 167:
        raise ValueError('The fixed 84-source, 167-crop panel changed')
    return rows, counts, metadata, windows, audit


def load_engine(saved):
    if 'quiet_phase' in saved or 'perceptual_trial' in saved:
        raise ValueError('Readiness must load the verified plain parent core')
    model = StudentDecoder(StudentConfig(**saved['model_config'])).to('cuda')
    engine = DistillationEngine(model, config=DistillationTrainingConfig(**saved['config']),
        loss_config=DistillationLossConfig(**saved['loss_config']),
        balancer_config=GradientBalancerConfig(**saved['balancer']['config']),
        discriminators=AudioDiscriminators(DiscriminatorConfig(**saved['discriminator_config'])))
    engine.load_state_dict(saved)
    if state_fingerprint(engine.state_dict()) != state_fingerprint(saved):
        raise ValueError('Loaded model, optimizer, EMA, or crop RNG differs from the parent')
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'readiness')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    verification = {'training_updates': 0, 'rtf_measured': False, 'student_evaluations_reused': [125, 250],
                    'launcher_sha256': file_sha(Path(__file__)), 'source_archive_sha256': file_sha(BASE / 'source.tgz')}
    teacher, dev, engine, selection = None, None, None, None
    teacher_before, engine_before, heldout_before = None, None, None
    status = 'failed'
    try:
        if verification['source_archive_sha256'] != SOURCE_ARCHIVE_SHA:
            raise ValueError('The validated r8 source archive changed')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        parent, selection, run_identity, selected_audit = selected_parent()
        verification.update(selected_audit)
        panel_rows, counts, metadata, windows, panel_audit = load_panel()
        if run_identity['metadata'] != metadata:
            raise ValueError('Original evaluation metadata differs from the fixed panel')
        verification['panel'] = panel_audit
        print(json.dumps({'phase': 'parent_and_panel_verified', 'parent_step': parent['engine']['step'],
                          'sources': len(panel_rows), 'crops': len(windows)}), flush=True)
        teacher = FrozenAudioVAE2.from_files(OLD / 'assets/audio_vae_v2.py', OLD / 'assets/audiovae.pth', device='cuda')
        # Exactly the r7 preparation identity and warmup source, not a new
        # data-dependent choice. This one utterance is warmup, never training.
        first = load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')[0]
        teacher = _target_teacher(teacher, first)
        teacher_before = state_fingerprint(teacher.model.state_dict())
        if teacher_before != run_identity['data']['teacher_state_sha256']:
            raise ValueError('Reloaded teacher state differs from the original comparison')
        warmup_bytes = Path(first.audio_path).read_bytes()
        if hashlib.sha256(warmup_bytes).hexdigest() != first.audio_sha256:
            raise ValueError('Teacher warmup source changed')
        warmup = read_native_16k(warmup_bytes, first).to(teacher.device)
        for _ in range(2):
            teacher.decode(teacher.encode(warmup))
        del warmup, warmup_bytes
        verification['teacher_warmup_passes'] = 2
        verification['teacher_model_state_sha256'] = teacher_before
        dev = SourceCorpus(panel_rows, teacher, cache_dir=BASE / 'teacher-cache-dev', input_sample_counts=counts,
            max_disk_bytes=1024**3, min_free_bytes=4*1024**3, max_memory_utterances=84, allow_prepared_source=True)
        before_cache = dev.metrics()
        heldout_path = BASE / 'heldout-targets.pt'
        descriptor = json.loads((BASE / 'heldout-targets.json').read_text())
        heldout_before = file_sha(heldout_path)
        if descriptor['sha256'] != heldout_before or descriptor['dev_corpus_identity'] != dev.identity:
            raise ValueError('Copied exact development targets or corpus identity changed')
        if descriptor != run_identity['data']['heldout_targets']:
            raise ValueError('Copied target descriptor differs from the original comparison')
        targets = torch.load(heldout_path, map_location='cpu', weights_only=True)
        crops = tuple(TrainingCrop(**fields) for fields in targets['crops'])
        if _crop_identity(crops) != descriptor['crops'] or _crop_identity(crops) != run_identity['heldout']:
            raise ValueError('Held-out raw latent or teacher target hashes changed')
        expected = {(w['source_id'], w['start_frame']): w['valid_output_samples48k'] for w in windows}
        if len(crops) != 167 or {(c.source_id, c.start_frame): c.valid_scored_samples for c in crops} != expected:
            raise ValueError('Saved target windows differ from the published development panel')
        engine = load_engine(parent['engine'])
        engine_before = state_fingerprint(engine.state_dict())
        _restore_rng(parent['rng'])
        evaluations = {}
        for name, step in (('previous', 125), ('current', 250)):
            path = SOURCE_RUN / f'evaluation-step{step:06d}.json'
            saved = json.loads(path.read_text())
            if saved['step'] != step or saved['arms']['control']['rows'][0]['evaluated_step'] != step:
                raise ValueError('Saved evaluation update differs from its filename')
            evaluations[name] = evaluation_evidence(crops, saved['arms']['control']['rows'],
                run_identity=run_identity, model_config=engine.model.config.to_dict())
            verification[name + '_evaluation_source_sha256'] = file_sha(path)
            _atomic_json(output / (name + '.json'), evaluations[name])
        print(json.dumps({'phase': 'checking_frozen_targets_streaming_and_gradients',
                          'panel_crops': len(crops), 'training_updates': 0, 'rtf_measured': False}), flush=True)
        evidence = collect_readiness_evidence(engine, crops, corpus=dev, panel_rows=panel_rows,
                                             sample_counts=counts, context_frames=29)
        _atomic_json(output / 'evidence.json', evidence)
        report = assess_perceptual_readiness(engine, crops, current=evaluations['current'],
            previous=evaluations['previous'], evidence=evidence)
        after_cache = dev.metrics()
        verification['dev_cache_delta'] = {k: after_cache[k] - before_cache[k] for k in
            ('memory_hits', 'disk_hits', 'cache_misses', 'prepared_utterances', 'evictions', 'teacher_seconds')}
        verification['heldout_target_file_unchanged'] = file_sha(heldout_path) == heldout_before
        verification['student_state_unchanged'] = state_fingerprint(engine.state_dict()) == engine_before
        verification['teacher_state_unchanged'] = state_fingerprint(teacher.model.state_dict()) == teacher_before
        verification['selected_parent_file_unchanged'] = file_sha(BASE / 'selected-parent.pt') == selection['parent_checkpoint_sha256']
        verification['source_checkpoint_file_unchanged'] = file_sha(Path(selection['source_checkpoint_path'])) == selection['source_checkpoint_sha256']
        if not all(verification[k] for k in ('heldout_target_file_unchanged', 'student_state_unchanged',
                    'teacher_state_unchanged', 'selected_parent_file_unchanged', 'source_checkpoint_file_unchanged')):
            raise RuntimeError('Readiness audit changed or raced its model, teacher, parent, or targets')
        _atomic_json(output / 'report.json', report)
        status = 'ready_for_bounded_trial' if report['ready_for_bounded_trial'] else 'not_ready'
        print(json.dumps({'phase': 'readiness_finished', 'status': status, 'checks': report['checks'],
            'final_acceptance_passed': report['final_acceptance']['passed'], 'automatic_activation': False,
            'training_updates': 0}), flush=True)
    except BaseException as error:
        verification['error'] = f'{type(error).__name__}: {error}'
        _atomic_json(output / 'failure.json', {'state': 'readiness_check_failed',
            'ready_for_bounded_trial': False, 'error': verification['error'], 'automatic_activation': False})
        raise
    finally:
        if dev is not None:
            dev.close()
        verification.update(status=status, readiness_elapsed_seconds=time.monotonic() - started)
        if teacher is not None and teacher_before is not None:
            verification['teacher_state_unchanged'] = state_fingerprint(teacher.model.state_dict()) == teacher_before
        if engine is not None and engine_before is not None:
            verification['student_state_unchanged'] = state_fingerprint(engine.state_dict()) == engine_before
        if selection is not None:
            verification['selected_parent_file_unchanged'] = file_sha(BASE / 'selected-parent.pt') == selection['parent_checkpoint_sha256']
            verification['source_checkpoint_file_unchanged'] = file_sha(Path(selection['source_checkpoint_path'])) == selection['source_checkpoint_sha256']
        _atomic_json(output / 'verification.json', verification)


if __name__ == '__main__':
    main()
