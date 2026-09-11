"""Launch the approved paired diagnostic against versioned Runpod assets."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

from audiovae_student.balance_comparison import BalanceComparisonConfig, run_balance_comparison
from audiovae_student.cache import TrainingCrop
from audiovae_student.comparison_data import load_comparison_plan, assert_comparison_disjoint
from audiovae_student.data import load_manifest
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.restart_data import canonical, file_sha
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import CHECKPOINT_SHA256, FrozenAudioVAE2
from audiovae_student.training import _atomic_save

BASE = Path('/workspace/fast-audiovae-convnext-20260909-r6')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PARENT_ROOT = Path('/workspace/fast-audiovae-convnext-20260909-r5')
PARENT_SHA = '955d12a84245765331b8697747e1565a3a90f3f2e1f6d8819e1c4f62d99ac104'
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parent_path = PARENT_ROOT / 'training-runs/representative-reconstruction-v1/latest.pt'
    if file_sha(parent_path) != PARENT_SHA:
        raise ValueError('Completed parent checkpoint changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    plan = load_comparison_plan(BASE / 'data/comparison-v2')
    helper_path = PARENT_ROOT / 'run_representative_stage.py'
    if file_sha(helper_path) != '72c249809abeb889a21c0391c65ec0580abd1c0796d157cdc99dbb4c55683c7a':
        raise ValueError('Original panel-loader source changed')
    spec = importlib.util.spec_from_file_location('original_panel_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    panel_rows, panel_counts, metadata, panel_windows, panel_audit = helper.load_dev_panel(PANEL)
    assert_comparison_disjoint(plan['rows'], panel_rows)
    print(json.dumps({'phase': 'metadata_verified', 'windows': len(plan['windows']),
                      'sources': len(plan['rows']), 'panel_crops': len(panel_windows)}), flush=True)
    teacher = FrozenAudioVAE2.from_files(OLD / 'assets/audio_vae_v2.py', OLD / 'assets/audiovae.pth', device='cuda')
    # Preserve the original qualification warmup instead of selecting a new
    # source-dependent execution path from this experiment's first batch.
    original_rows = load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')
    first = original_rows[0]
    teacher = _target_teacher(teacher, first)
    before = state_fingerprint(teacher.model.state_dict())
    audio_bytes = Path(first.audio_path).read_bytes()
    if hashlib.sha256(audio_bytes).hexdigest() != first.audio_sha256:
        raise ValueError('Teacher warmup input hash differs')
    warmup = read_native_16k(audio_bytes, first).to(teacher.device)
    for _ in range(2):
        teacher.decode(teacher.encode(warmup))
    del warmup
    corpus = SourceCorpus(plan['rows'], teacher, cache_dir=BASE / 'teacher-cache-train',
        input_sample_counts={r.source_id: plan['counts'][r.source_id] for r in plan['rows']},
        max_disk_bytes=3*1024**3, min_free_bytes=4*1024**3, max_memory_utterances=64,
        allow_prepared_source=True)
    dev = None
    try:
        # Independent caches avoid treating unknown FLEURS placeholders as
        # known cross-split speakers. Real identity exclusions were checked above.
        dev = SourceCorpus(panel_rows, teacher, cache_dir=BASE / 'teacher-cache-dev', input_sample_counts=panel_counts,
            max_disk_bytes=1024**3, min_free_bytes=4*1024**3, max_memory_utterances=84,
            allow_prepared_source=True)
        heldout_path = BASE / 'heldout-targets.pt'
        heldout_meta = BASE / 'heldout-targets.json'
        if heldout_path.exists():
            descriptor = json.loads(heldout_meta.read_text())
            if descriptor['sha256'] != file_sha(heldout_path) or descriptor['dev_corpus_identity'] != dev.identity:
                raise ValueError('Pinned common development targets changed')
            saved = torch.load(heldout_path, map_location='cpu', weights_only=True)
            heldout = tuple(TrainingCrop(**fields) for fields in saved['crops'])
            if _crop_identity(heldout) != descriptor['crops']:
                raise ValueError('Pinned development crop hashes changed')
        else:
            prefill = dev.prefetch([r.source_id for r in panel_rows], max_batch_size=8,
                                  max_total_input_samples=1_920_000)
            heldout = tuple(helper.panel_crop(dev.get(w['source_id']), w) for w in panel_windows)
            _atomic_save({'crops': [asdict(c) for c in heldout]}, heldout_path)
            descriptor = {'sha256': file_sha(heldout_path), 'crops': _crop_identity(heldout),
                'dev_corpus_identity': dev.identity, 'qualification': prefill,
                'original_r5_crops_sha256': state_fingerprint(parent['identity']['heldout']),
                'note': 'Same fixed original crops; regenerated targets pinned byte-identically for both new arms.'}
            heldout_meta.write_bytes(canonical(descriptor))
        train_prefill_path = BASE / 'initial-train-prefill.json'
        if not train_prefill_path.exists():
            first_ids = list(dict.fromkeys(w.source_id for w in plan['windows'][:32]))
            qualification = corpus.prefetch(first_ids, max_batch_size=8, max_total_input_samples=1_920_000)
            train_prefill_path.write_bytes(canonical({'corpus_identity': corpus.identity, 'report': qualification}))
        qualification = json.loads(train_prefill_path.read_text())
        if qualification['corpus_identity'] != corpus.identity:
            raise ValueError('Initial training corpus qualification identity changed')
        if before != state_fingerprint(teacher.model.state_dict()):
            raise ValueError('Frozen teacher state changed during preparation')
        identity = {'teacher_checkpoint_sha256': CHECKPOINT_SHA256,
            'teacher_state_sha256': before, 'teacher_batch_qualification': qualification,
            'source_corpus': corpus.identity, 'plan': plan['identity'], 'panel': panel_audit,
            'heldout_targets': descriptor, 'source_archive_sha256': file_sha(BASE / 'source.tgz'),
            'launcher_sha256': file_sha(Path(__file__)), 'panel_loader_sha256': file_sha(helper_path)}
        print(json.dumps({'phase': 'starting_paired_updates', 'updates_per_arm': 500,
                          'teacher_unchanged': True, 'shared_pinned_panel': len(heldout)}), flush=True)
        output = BASE / 'training-runs/balance-v1'
        result = run_balance_comparison(corpus, plan['windows'], plan['rows'], {**plan['counts'], **panel_counts}, plan['ledger'],
            heldout, panel_rows, output, parent=parent, parent_checkpoint_sha256=PARENT_SHA, heldout_corpus=dev,
            data_identity=identity, heldout_metadata=metadata, reserved_rows=plan['reserved'],
            config=BalanceComparisonConfig(), device='cuda', resume_from=output/'latest.pt' if args.resume else None,
            max_updates=args.max_updates, log_dir=OLD/'runs', run_prefix='balance-v1')
        result['teacher_state_unchanged'] = before == state_fingerprint(teacher.model.state_dict())
        result['parent_checkpoint_unchanged'] = file_sha(parent_path) == PARENT_SHA
        (BASE/'stage-result.json').write_bytes(canonical(result))
        print(json.dumps(result), flush=True)
    finally:
        corpus.close()
        if dev:
            dev.close()


if __name__ == '__main__':
    main()
