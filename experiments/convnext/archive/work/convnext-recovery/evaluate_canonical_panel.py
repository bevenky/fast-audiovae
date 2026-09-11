"""Evaluate retained checkpoints on separately corrected encoder inputs; no updates."""
import fcntl,json
from pathlib import Path
from diagnostic_common import load_context,atomic_json,status
from canonical_evaluation import load_canonical_cache,build_canonical_evaluation
from audiovae_student.objective_comparison import state_fingerprint
from evaluation_audit import decorate_evaluation
OUT=Path('/tmp/fast-audiovae-recovery-20260909/canonical-evaluation-v1')
RECEIPT=Path('/tmp/fast-audiovae-recovery-20260909/canonical-panel-v1/receipt.json')
lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
with lock.open('rb') as h:
 fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir(exist_ok=False)
 ctx=load_context();heldout,metadata,receipt=load_canonical_cache(RECEIPT)
 summary={'checkpoints':{},'teacher_contract_sha256':receipt['contract']['identity_sha256'],'parameter_updates':0}
 for name in ['parent','targeted']:
  status('canonical_evaluate_checkpoint',checkpoint=name)
  engine=ctx.engine(name);before=state_fingerprint(engine.state_dict())
  report=build_canonical_evaluation(engine,heldout,metadata,receipt)
  if state_fingerprint(engine.state_dict())!=before:raise RuntimeError('Evaluation mutated engine')
  atomic_json(OUT/(name+'.json'),report)
  summary['checkpoints'][name]={'path':str(OUT/(name+'.json')),'sha256':ctx.expected_hashes[name],
    'step':engine.step,'engine_unchanged':True}
  del engine
 summary.update(ctx.verify_files());atomic_json(OUT/'summary.json',summary)
 status('canonical_evaluation_complete',output=str(OUT))
