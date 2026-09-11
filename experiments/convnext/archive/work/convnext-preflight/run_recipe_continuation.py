"""Continue the same trained student with an explicitly versioned data segment."""
import argparse
import json
import os
from pathlib import Path

import torch

from audiovae_student.cache import TrainingCrop
from audiovae_student.comparison_data import load_comparison_plan
from audiovae_student.continuation_data import load_continuation_plan, load_supplemental_catalog
from audiovae_student.data import ManifestRow, load_manifest
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.recipe_v2_continuation import run_recipe_v2_continuation
from audiovae_student.restart_data import file_sha
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import FrozenAudioVAE2


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PARENT = BASE / 'training-runs/decoder-recipe-v2'
OUT = BASE / 'training-runs/decoder-recipe-v2-expressive'
READY_SHA = '7cfab2365224326af965128ca8a5ca375891ded0b9905446b85da6f6fc741a89'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
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
    handoff = json.loads((args.plan / 'handoff.json').read_text())
    if handoff.get('preview_only'):
        raise ValueError('An illustrative preview cannot launch training')
    expected = handoff['parent']['checkpoint_sha256']
    if file_sha(PARENT / 'latest.pt') != expected:
        raise ValueError('Committed parent checkpoint changed')
    payload = torch.load(PARENT / 'latest.pt', map_location='cpu', weights_only=True)
    parent_identity = payload['identity']
    del payload
    plan = load_continuation_plan(args.plan)
    parent_plan = load_comparison_plan(BASE / 'data/recipe-v2/optimization')
    supplement = load_supplemental_catalog(BASE / 'remediation/expressive/ready-v2.json',
                                          expected_sha256=READY_SHA)
    teacher = FrozenAudioVAE2.from_files(OLD / 'assets/audio_vae_v2.py',
                                       OLD / 'assets/audiovae.pth', device='cuda')
    first = load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')[0]
    teacher = _target_teacher(teacher, first)
    before = state_fingerprint(teacher.model.state_dict())
    if before != parent_identity['data']['teacher_state_sha256']:
        raise ValueError('Frozen teacher no longer matches parent')
    raw = Path(first.audio_path).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != first.audio_sha256:
        raise ValueError('Pinned teacher warmup source changed')
    audio = read_native_16k(raw, first).cuda()
    for _ in range(2):
        teacher.decode(teacher.encode(audio))
    del audio
    corpus = SourceCorpus(plan['rows'], teacher, cache_dir=args.plan / 'teacher-cache',
        input_sample_counts=plan['counts'], max_disk_bytes=512 * 1024**2,
        min_free_bytes=4 * 1024**3, max_memory_utterances=64, allow_prepared_source=True)
    try:
        prefill_path = args.plan / 'teacher-prefill.json'
        if not prefill_path.exists():
            qualification = corpus.prefetch(list(dict.fromkeys(w.source_id for w in plan['windows'][:32])),
                max_batch_size=8, max_total_input_samples=1_920_000)
            prefill_path.write_text(json.dumps({'corpus_identity': corpus.identity,
                'report': qualification}, indent=2) + '\n')
        qualification = json.loads(prefill_path.read_text())
        if qualification['corpus_identity'] != corpus.identity:
            raise ValueError('New target corpus changed on resume')
        target_path = BASE / 'heldout-targets.pt'
        descriptor = parent_identity['data']['heldout_targets']
        if file_sha(target_path) != descriptor['sha256']:
            raise ValueError('Original fixed-panel target bytes changed')
        saved = torch.load(target_path, map_location='cpu', weights_only=True)
        heldout = tuple(TrainingCrop(**fields) for fields in saved['crops'])
        if _crop_identity(heldout) != parent_identity['data']['heldout']['crops']:
            raise ValueError('Original fixed-panel latent or waveform target changed')
        panel_rows = [ManifestRow.from_dict(r) for r in parent_identity['data']['heldout']['rows']]
        identity = {**parent_identity['data'], 'source_corpus': corpus.identity,
            'teacher_batch_qualification': qualification,
            'continuation_launcher_sha256': file_sha(Path(__file__)),
            'supplement_ready_sha256': READY_SHA}
        if state_fingerprint(teacher.model.state_dict()) != before:
            raise ValueError('Teacher changed during continuation preparation')
        result = run_recipe_v2_continuation(corpus, plan, heldout, panel_rows, OUT,
            parent_checkpoint=PARENT / 'latest.pt', parent_checkpoint_sha256=expected,
            parent_run_dir=PARENT, parent_plan=parent_plan, data_identity=identity,
            expected_plan_identity_sha256=handoff['expected_plan_identity_sha256'],
            supplement=supplement, device='cuda', resume_from=OUT / 'latest.pt' if args.resume else None,
            max_updates=args.max_updates, checkpoint_interval=100, min_free_bytes=4 * 1024**3)
        if state_fingerprint(teacher.model.state_dict()) != before:
            raise ValueError('Teacher changed during continued training')
        print(json.dumps(result), flush=True)
    finally:
        corpus.close()


if __name__ == '__main__':
    main()
