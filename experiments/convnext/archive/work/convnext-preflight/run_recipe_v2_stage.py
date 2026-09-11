"""Start the complete versioned decoder recipe on the existing training host."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

from audiovae_student.cache import TrainingCrop, sample_training_crop
from audiovae_student.comparison_data import load_comparison_plan, assert_comparison_disjoint, verify_comparison_windows
from audiovae_student.data import load_manifest
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.recipe_v2 import RecipeV2Engine, RecipeV2Config, calibration_batch
from audiovae_student.restart_data import canonical, file_sha, digest
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import CHECKPOINT_SHA256, FrozenAudioVAE2
from audiovae_student.training import _atomic_save

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')
PRIOR = Path('/workspace/fast-audiovae-convnext-20260909-r7')


def crops_for(corpus, windows):
    corpus.prefetch(list(dict.fromkeys(w.source_id for w in windows)), max_batch_size=8,
                    max_total_input_samples=1_920_000)
    result = []
    for w in windows:
        c = sample_training_crop(corpus.get(w.source_id), w.start_frame, w.scored_frames, 29)
        if c.valid_scored_samples < w.valid_output_samples48k:
            raise ValueError('Target shorter than planned valid interval')
        result.append(replace(c, valid_scored_samples=w.valid_output_samples48k))
    return result


def preflight(corpus, plan, calibration, teacher, teacher_hash):
    """Three distinct training batches exercise the real full-size model and D."""
    torch.manual_seed(47)
    recipe = RecipeV2Config(total_steps=4, reconstruction_warmup_steps=2,
                            learning_rate_warmup_steps=1, perceptual_ramp_steps=2)
    model = StudentDecoder(StudentConfig(normalization_mode='masked_batch_norm',
                                          adapter_mode='raw_repeat_phase_bias')).cuda()
    engine = RecipeV2Engine(model, recipe=recipe)
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        metrics.append(engine.train_step(crops_for(corpus, plan['windows'][step * 32:(step + 1) * 32])))
    calibration_crops = crops_for(corpus, calibration['windows'][:32])
    report = engine.calibrate(lambda: iter([calibration_batch(calibration_crops, 'cuda')]),
        provenance={'split': 'train', 'purpose': 'independent_full_model_mechanical_preflight',
                    'calibration_plan': calibration['identity']['identity_sha256'],
                    'windows': [w.to_dict() for w in calibration['windows'][:32]]})
    before = state_fingerprint(engine.discriminators.state_dict())
    metrics.append(engine.train_step(crops_for(corpus, plan['windows'][64:96])))
    assert engine.discriminator_updates == 1 and metrics[-1]['feature_matching'] > 0
    assert before != state_fingerprint(engine.discriminators.state_dict())
    assert teacher_hash == state_fingerprint(teacher.model.state_dict())
    # Exact checkpoint loader must accept the actual production-sized state.
    state = engine.state_dict()
    engine.load_state_dict(state)
    model.eval()
    z = calibration_crops[0].latents.cuda()
    with torch.no_grad():
        batch = model(z)
        state = model.initial_state(1)
        pieces = []
        for i in range(z.shape[-1]):
            piece, state = model.forward_stream(z[..., i:i+1], state)
            pieces.append(piece)
        streamed = torch.cat(pieces, dim=-1)
        torch.testing.assert_close(streamed, batch, atol=3e-5, rtol=3e-5)
        folded = model.fold_normalization()
        torch.testing.assert_close(folded(z), batch, atol=3e-5, rtol=3e-5)
    result = {'state': 'passed', 'kind': 'full_size_real_teacher_mechanical_preflight',
        'optimization_windows': 96, 'distinct_updates': 3, 'discriminator_updates': 1,
        'discriminator_heads': len(engine.discriminators.periods) + len(engine.discriminators.resolutions),
        'metrics': metrics, 'calibration': report, 'teacher_state_unchanged': True,
        'streaming_max_abs_error': float((streamed-batch).abs().max()),
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
        'source_archive_sha256': file_sha(BASE/'source.tgz'),
        'launcher_sha256': file_sha(Path(__file__)),
        'meaning': 'Correctness and memory check only; not a trained quality or speed result'}
    (BASE / 'full-model-preflight.json').write_bytes(canonical(result))
    print(json.dumps({key:value for key,value in result.items() if key not in {'metrics','calibration'}}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    root = BASE / 'data/recipe-v2'
    plan, calibration = [load_comparison_plan(root / name) for name in ('optimization', 'calibration')]
    ready = json.loads((root / 'ready.json').read_text())
    if (ready['identity_sha256'] != digest({k:v for k,v in ready.items() if k != 'identity_sha256'})
            or ready['optimization_plan_identity'] != plan['identity']['identity_sha256']
            or ready['calibration_plan_identity'] != calibration['identity']['identity_sha256']):
        raise ValueError('Combined optimization/calibration identity changed')
    rows = {r.source_id:r for r in plan['rows'] + calibration['rows']}
    counts = {**plan['counts'], **calibration['counts']}
    verify_comparison_windows(plan['windows'] + calibration['windows'], rows, counts, plan['ledger'],
                              reserved_rows=plan['reserved'], excluded_sources=plan['excluded'])
    helper_path = Path('/workspace/fast-audiovae-convnext-20260909-r5/run_representative_stage.py')
    if file_sha(helper_path) != '72c249809abeb889a21c0391c65ec0580abd1c0796d157cdc99dbb4c55683c7a':
        raise ValueError('Original panel loader changed')
    spec = importlib.util.spec_from_file_location('original_panel_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    panel_rows, panel_counts, metadata, panel_windows, panel_audit = helper.load_dev_panel(PANEL)
    assert_comparison_disjoint(rows.values(), panel_rows)
    teacher = FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py', OLD/'assets/audiovae.pth', device='cuda')
    first = load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')[0]
    teacher = _target_teacher(teacher, first)
    before = state_fingerprint(teacher.model.state_dict())
    payload = Path(first.audio_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != first.audio_sha256:
        raise ValueError('Teacher qualification warmup source changed')
    audio = read_native_16k(payload, first).cuda()
    for _ in range(2):
        teacher.decode(teacher.encode(audio))
    del audio
    corpus = SourceCorpus([rows[key] for key in sorted(rows)], teacher, cache_dir=BASE/'teacher-cache-train',
        input_sample_counts=counts, max_disk_bytes=2*1024**3, min_free_bytes=4*1024**3,
        max_memory_utterances=64, allow_prepared_source=True)
    dev = None
    try:
        initial = BASE / 'initial-train-prefill.json'
        if not initial.exists():
            qualification = corpus.prefetch(list(dict.fromkeys(w.source_id for w in plan['windows'][:32])),
                max_batch_size=8, max_total_input_samples=1_920_000)
            initial.write_bytes(canonical({'corpus_identity':corpus.identity,'report':qualification}))
        qualification = json.loads(initial.read_text())
        if qualification['corpus_identity'] != corpus.identity:
            raise ValueError('Training source corpus changed')
        if args.preflight_only:
            preflight(corpus, plan, calibration, teacher, before)
            return
        precheck = json.loads((BASE/'full-model-preflight.json').read_text())
        if (precheck.get('state') != 'passed' or precheck['source_archive_sha256'] != file_sha(BASE/'source.tgz')
                or precheck['launcher_sha256'] != file_sha(Path(__file__))):
            raise ValueError('Full-size mechanical validation must match the exact launched source')
        dev = SourceCorpus(panel_rows, teacher, cache_dir=BASE/'teacher-cache-dev', input_sample_counts=panel_counts,
            max_disk_bytes=1024**3, min_free_bytes=4*1024**3, max_memory_utterances=84, allow_prepared_source=True)
        target_path = BASE / 'heldout-targets.pt'
        descriptor = json.loads((BASE/'heldout-targets.json').read_text())
        if (file_sha(target_path) != '3349f715c9ae4c4b7ac859cde935cdc949395bfc0e15915ad759c8e547a4dd2e'
                or descriptor['sha256'] != file_sha(target_path) or descriptor['dev_corpus_identity'] != dev.identity):
            raise ValueError('Pinned original development targets changed')
        saved = torch.load(target_path, map_location='cpu', weights_only=True)
        heldout = tuple(TrainingCrop(**fields) for fields in saved['crops'])
        if _crop_identity(heldout) != descriptor['crops']:
            raise ValueError('Original development crop hashes changed')
        from audiovae_student.representative_pilot import _verify_panel_targets
        _verify_panel_targets(dev, heldout, panel_rows, panel_counts, 29)
        if before != state_fingerprint(teacher.model.state_dict()):
            raise ValueError('Frozen teacher changed during preparation')
        identity = {'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'teacher_state_sha256':before,
            'teacher_batch_qualification':qualification,'source_corpus':corpus.identity,
            'plan':ready,'panel':panel_audit,'heldout_targets':descriptor,
            'heldout':{'crops':_crop_identity(heldout), 'rows':[r.to_dict() for r in panel_rows],
                'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'source_corpus':dev.identity,
                'input_sample_counts':panel_counts},
            'source_archive_sha256':file_sha(BASE/'source.tgz'),'launcher_sha256':file_sha(Path(__file__)),
            'preflight_sha256':file_sha(BASE/'full-model-preflight.json')}
        from audiovae_student.recipe_v2_pilot import run_recipe_v2
        output = BASE / 'training-runs/decoder-recipe-v2'
        print(json.dumps({'phase':'starting_fresh_complete_recipe','total_steps':10000,
                          'teacher_unchanged':True,'calibration_after_step':500,'GAN_first_update':501}),flush=True)
        result = run_recipe_v2(corpus, plan, heldout, panel_rows, output,
            calibration_windows=calibration['windows'], calibration_rows=calibration['rows'],
            calibration_counts=calibration['counts'], calibration_provenance={'split':'train',
                'plan_identity':calibration['identity'],'purpose':'fixed_weight_sequential_normalization',
                'data_plan_sha256':file_sha(root/'data-plan.json')}, data_identity=identity,
            recipe=RecipeV2Config(), device='cuda', resume_from=output/'latest.pt' if args.resume else None,
            max_updates=args.max_updates, heldout_metadata=metadata, log_dir=BASE/'tensorboard',
            run_name='decoder-recipe-v2')
        result['teacher_state_unchanged'] = before == state_fingerprint(teacher.model.state_dict())
        if not result['teacher_state_unchanged']:
            raise RuntimeError('Frozen teacher changed during training')
        (BASE/'stage-result.json').write_bytes(canonical(result))
        print(json.dumps(result),flush=True)
    finally:
        corpus.close()
        if dev:
            dev.close()


if __name__ == '__main__':
    main()
