"""Seal remaining/new audio against a frozen, fully committed parent snapshot."""
import argparse
import json
from pathlib import Path

import torch

from audiovae_student.comparison_data import load_comparison_plan
from audiovae_student.continuation_data import (load_supplemental_catalog,
    plan_fixed_window_continuation, write_continuation_plan)
from audiovae_student.data import load_manifest
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine
from audiovae_student.recipe_v2_pilot import implementation_identity
from audiovae_student.restart_data import digest, file_sha


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
RUN = BASE / 'training-runs/decoder-recipe-v2'
STAGE = BASE / 'remediation/expressive'
READY_SHA = '7cfab2365224326af965128ca8a5ca375891ded0b9905446b85da6f6fc741a89'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha', required=True)
    parser.add_argument('--preview-checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    latest = args.preview_checkpoint or RUN / 'latest.pt'
    if args.preview_checkpoint is not None and not args.destination.name.startswith('preview-'):
        raise ValueError('Preview must use a separate explicitly named preview directory')
    if file_sha(latest) != args.expected_checkpoint_sha or (args.preview_checkpoint is None and (RUN / 'inflight.json').exists()):
        raise ValueError('Parent changed or is not at a committed boundary')
    payload = torch.load(latest, map_location='cpu', weights_only=True)
    identity, state = payload['identity'], payload['engine']
    if identity['implementation'] != implementation_identity():
        raise ValueError('Continuation shadow differs from parent training implementations')
    engine = RecipeV2Engine(StudentDecoder(StudentConfig(**state['model_config'])),
                           recipe=RecipeV2Config(**state['recipe']))
    engine.load_state_dict(state)
    del engine
    plan, calibration = [load_comparison_plan(BASE / 'data/recipe-v2' / name)
                         for name in ('optimization', 'calibration')]
    parent = {'checkpoint_sha256': args.expected_checkpoint_sha,
              'run_identity_sha256': digest(identity),
              'journal_sha256': payload['journal_sha256'],
              'metrics_sha256': payload['metrics_sha256'],
              'step': state['step'], 'batch_size': identity['batch_size']}
    supplement = load_supplemental_catalog(STAGE / 'ready-v2.json', expected_sha256=READY_SHA)
    # Fixed panel, historical reservations and new cohorts remain excluded.
    reservations = list(plan['reserved'])
    language_manifest = BASE / 'remediation/heldout-languages/language-expanded-manifest.jsonl'
    if file_sha(language_manifest) != 'fa33aafe748f612b877d51277f453705495c19e9bc37f0017ba08d502868e152':
        raise ValueError('New language reservation manifest changed')
    reservations += load_manifest(language_manifest)
    event_identity = json.loads((BASE / 'remediation/validation-appendix/ready.json').read_text())
    event_manifest = BASE / 'remediation/validation-appendix/manifest.jsonl'
    if file_sha(event_manifest) != '6d3cf3da2b4803ec810799ed43f67497c72474a6ea06d829982d3651065d2416':
        raise ValueError('New event reservation manifest changed')
    reservations += load_manifest(event_manifest)
    unique = {}
    for row in reservations:
        if row.source_id in unique and row.to_dict() != unique[row.source_id].to_dict():
            raise ValueError('Reservation source identity conflict')
        unique[row.source_id] = row
    new = plan_fixed_window_continuation(plan, payload['sampler'], parent=parent,
        calibration_windows=calibration['windows'], calibration_rows=calibration['rows'],
        calibration_counts=calibration['counts'], supplement=supplement,
        reserved_rows=tuple(unique.values()), target_expressive_fraction=.05, seed=47)
    if file_sha(latest) != args.expected_checkpoint_sha:
        raise ValueError('Parent advanced while sealing the data plan')
    published = write_continuation_plan(new, args.destination)
    receipt = {'parent': parent, 'plan_path': str(args.destination),
               'expected_plan_identity_sha256': published['identity_sha256'],
               'metadata': new['metadata'], 'parent_strict_load_passed': True,
               'teacher_and_objective_changed': False, 'preview_only': args.preview_checkpoint is not None}
    (args.destination / 'handoff.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'parent_step': state['step'], 'plan_path': str(args.destination),
        'expected_plan_identity_sha256': published['identity_sha256'],
        'expected_total_steps': new['metadata']['expected_total_steps'],
        'original_remaining_windows': new['metadata']['original_remaining_windows'],
        'selected_supplement_windows': new['metadata']['selected_supplement_windows'],
        'allocation': new['metadata']['allocation']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
