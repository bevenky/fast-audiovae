"""Bind the user-approved conditional continuation to its completed evidence."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    base=Path('/tmp/fast-audiovae-recovery-20260909')
    diagnostic_path=base/'corrected-update-diagnostic-v1/corrected-update-diagnostic.json'
    training_path=Path('/dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json')
    canonical_path=base/'canonical-panel-v1/receipt.json'
    out=base/'corrected-training-resolution.json'
    if out.exists():raise FileExistsError('Resolution already exists')
    d=json.loads(diagnostic_path.read_text())
    training=json.loads(training_path.read_text())
    canonical=json.loads(canonical_path.read_text())
    diagnostic_training_path=Path(d['identity']['training_overlay']['receipt_path'])
    if sha(diagnostic_training_path)!=d['identity']['training_overlay']['receipt_sha256']:
        raise ValueError('Fresh32 diagnostic training receipt changed')
    diagnostic_training=json.loads(diagnostic_training_path.read_text())
    generation_path=Path(training['contract']['provenance']['generation_path'])
    generation=json.loads(generation_path.read_text())
    if not d['state_restored'] or not d['original_files']['retained_checkpoint_hashes_unchanged']:
        raise ValueError('Diagnostic preservation failed')
    current=d['native_rates']['current']['relative_change_percent']['quiet_residual_mse']
    quarter=d['native_rates']['quarter']['relative_change_percent']['quiet_residual_mse']
    if not current>0>quarter:raise ValueError('Corrected diagnostic does not support the registered comparison')
    if not generation['complete'] or not generation['teacher_remained_frozen'] or not generation['original_cache_unchanged']:
        raise ValueError('Fresh teacher generation is incomplete or changed original state')
    if generation['pool_counts']!={'targeted_generator':12800}:
        raise ValueError('Wrong continuation pool')
    if generation['encoder_policy'] != 'Full-source singleton cuDNN-disabled FP32 raw_mu reference only; no batched encoder results used':
        raise ValueError('Continuation must use the same independent reference encoder policy')
    if sha(generation_path)!=training['contract']['provenance']['generation_sha256']:
        raise ValueError('Generation evidence changed')
    if canonical['sha256']!=d['identity']['canonical_panel']['cache_sha256']:
        raise ValueError('Continuation panel differs from diagnostic')
    c=training['contract']
    diagnostic_contract=diagnostic_training['contract']
    for key in ('teacher_state_sha256','base_target_cache_sha256','base_data_plan_sha256'):
        if c[key]!=diagnostic_contract[key]:
            raise ValueError('Fresh32 and continuation training differ: '+key)
    diagnostic_runtime=diagnostic_contract['provenance']['runtime']
    if generation['runtime']!=diagnostic_runtime or c['provenance']['runtime']!=diagnostic_runtime:
        raise ValueError('Fresh32 and continuation generation runtimes differ')
    for key in ('torch','cudnn'):
        if d['identity'][key]!=diagnostic_runtime[key]:
            raise ValueError('Corrected diagnostic ran under a different runtime: '+key)
    result={'format_version':1,'training_ready':True,
        'parent_checkpoint_sha256':d['identity']['checkpoint_sha256'],
        'target_cache_sha256':c['base_target_cache_sha256'],'data_plan_sha256':c['base_data_plan_sha256'],
        'requires_fresh_training_pairs':True,'requires_canonical_evaluation':True,
        'training_receipt_sha256':sha(training_path),'training_target_contract_sha256':c['identity_sha256'],
        'training_target_cache_sha256':training['sha256'],
        'canonical_receipt_sha256':sha(canonical_path),'canonical_target_contract_sha256':canonical['contract']['identity_sha256'],
        'canonical_target_cache_sha256':canonical['sha256'],
        'resolution':'Only this fresh reference-target experiment is ready. Historical cache strict failures remain recorded and are not reclassified as passed. Both arms use identical newly regenerated singleton-reference encoder means and full-source teacher decoder targets. The corrected actual one-step test reproduced quiet-error worsening at the current rate and improvement at a quarter rate. This satisfies the user-approved condition for the bounded400-step comparison; it is not a checkpoint promotion or a long-run training restart.',
        'evidence':[{'path':str(p),'sha256':sha(p)} for p in (diagnostic_path,diagnostic_training_path,training_path,generation_path,canonical_path)],
        'diagnostic_quiet_mse_change_percent':{'current':current,'quarter':quarter},
        'limitations':['The single-step teacher-relative peak-excess energy still worsens at both rates.','The old strict numerical screens still contain failures.','No claim of final perceptual equivalence or generalization from one diagnostic batch.']}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'path':str(out),'sha256':sha(out),'training_ready':True}))


if __name__=='__main__':main()
