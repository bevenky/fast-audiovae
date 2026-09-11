"""Read-only CPU metadata audit; output only aggregate statistics and hashes."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path('/dev/shm/fast-audiovae-optimizer-comparison-20260911-v1')
PLAN_SHA = 'c219369ccc09efe9e7b0d7843f778a4d0e38f7d1d6dcac3efa6669e58bf195a7'
CONTROLLER_SHA = 'd5cc30b0cee8124b7e4de292bd339f35468ac3845a8f616f09023829a79b64b8'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())


def protection_record(row, step):
    fractions = tuple(2.**(-power) for power in range(11)) + (0.,)
    return (row.get('step') == step and row.get('q_constraints') == 12
        and row.get('q_constrained_sources') == row.get('q_constraint_gradient_sources') == 6
        and row.get('q_full_primal_verified') == row.get('q_kkt_passed') == row.get('q_caps_passed') == 1
        and row.get('q_normal_full_primal_verified') == row.get('q_normal_kkt_passed') == row.get('q_normal_budget_passed') == 1
        and row.get('q_normal_correction_enabled') == 1 and row.get('startup_anchor_after_passed') == 6
        and row.get('q_accepted_fraction') in fractions
        and row.get('q_base_fraction') == row.get('q_accepted_fraction')
        and row.get('q_normal_solves') in (0,1,2)
        and row.get('q_base_fraction_has_normal_correction') == row.get('q_normal_accepted')
        and (not row.get('q_normal_accepted') or row.get('q_base_fraction') == 1.)
        and row.get('q_normal_accepted_norm',math.inf) <= row.get('q_normal_budget',-1.)
        and row.get('q_canonical_score_forwards') == row.get('startup_anchor_checks') == 6*row.get('q_canonical_score_calls',-1)
        and row.get('q_constraint_gradient_forwards') == 6+row.get('q_normal_gradient_sources',-1)
        and row.get('q_normal_gradient_sources') == 6*row.get('q_normal_solves',-1)
        and all(not isinstance(v,float) or math.isfinite(v) for v in row.values()))


def main():
    assert sha(ROOT/'plan.json') == PLAN_SHA
    assert sha(ROOT/'optimizer_comparison_controller.py') == CONTROLLER_SHA
    sys.path.insert(0, str(ROOT))
    import optimizer_comparison_controller as c
    plan = read(ROOT/'plan.json'); candidates = c.validate_plan(plan)
    decision = read(ROOT/'qualification.json')
    controller = read(ROOT/'qualification-controller-completed.json')
    checks = {'qualification_hash':sha(ROOT/'qualification.json') == (ROOT/'qualification.sha256').read_text().strip(),
              'controller_completed':controller['status'] == 'completed' and len(controller['records']) == 12,
              'plan_manifest_sources':True, 'no_recovery_dispatched':not (ROOT/'recovery-dispatch.json').exists()}
    dispatch = read(ROOT/'qualification-dispatch.json')
    process = c.proc_identity(dispatch['pid'])
    checks['controller_exited'] = process is None or process['state'] == 'Z' or str(process['start_ticks']) != str(dispatch['start_ticks'])
    records=[]; counts=[]; protected_files={}; all_quality=[]
    for candidate in candidates:
        name=candidate['name']; out=ROOT/'qualification'/name
        trial=read(ROOT/'trials'/('qualification-'+name+'.json'))
        exact_control=candidate['method']=='adamw' and candidate['optimizer_kwargs']['matrix_lr']==3e-5
        record=c.inspect_trial(out,trial,control=exact_control,exit_code=0)
        records.append(record)
        done=read(out/'completed.json'); launch=read(out/'launch.json')
        assert done['complete'] and done['status']=='awaiting_comparison' and record['eligible']
        assert not done['pilot_state_resumed'] and not launch['pilot_state_resumed']
        rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
        assert len(rows)==64
        policy_pass=sum(protection_record(row,step) for step,row in enumerate(rows,1))
        assert policy_pass==64
        receipt=read(ROOT/'launches'/('qualification-'+name+'.json'))
        proc=c.proc_identity(receipt['pid'])
        assert proc is None or proc['state']=='Z' or str(proc['start_ticks']) != str(receipt['start_ticks'])
        for path,h in launch['protected'].items():
            if path in protected_files:assert protected_files[path]==h
            protected_files[path]=h
        quality=done['milestone_quality']['64']
        all_quality.append({'trial':name,'method':candidate['method'],'training_probe_total':record['training_probe_total'],
            'aggregate':quality['aggregate'],'regions':quality['regions'],
            'calibration_startup':quality['calibration_startup'],'development_startup':quality['development_startup'],
            'active_rms_ratio':quality.get('active_rms_ratio'),'whistle_rms_ratio':quality.get('whistle_rms_ratio')})
        counts.append({'trial':name,'rows':len(rows),'all_normal_policy_rows_passed':policy_pass,
            'ordinary_sources':record['updates']*12,'nonzero_updates':record['nonzero_updates'],
            'normal_solves':sum(x['q_normal_solves'] for x in rows),
            'normal_accepts':sum(x['q_normal_accepted'] for x in rows),
            'ordinary_seconds':sum(x['q_ordinary_update_seconds'] for x in rows),
            'protection_seconds':sum(x['q_auxiliary_seconds'] for x in rows),
            'optimizer_activity':done['optimizer_activity']})
    checks['all_candidate_records_reproduced']=records==controller['records']==decision['candidates']
    checks['selection_reproduced']=c.select_candidates(plan,records,PLAN_SHA)==decision
    checks['all_protected_files_rehashed']=all(c.sha(p)==h for p,h in protected_files.items())
    checks['all_768_update_certificates']=sum(row['all_normal_policy_rows_passed'] for row in counts)==768
    checks['all_four_methods_eligible']=all(decision['selected'][m]['eligible'] for m in c.METHODS)
    free=shutil.disk_usage(ROOT).free
    checks['first_arm_storage']=free>=c.required_free_bytes(plan,'adamw',2000)
    result={'utc':datetime.now(timezone.utc).isoformat(),'all_pass':all(checks.values()),'checks':checks,
        'qualification_sha256':sha(ROOT/'qualification.json'),'plan_sha256':PLAN_SHA,
        'protected_files_rehashed':len(protected_files),'free_bytes':free,'trials':counts,
        'selected':{m:{'trial':v['name'],'optimizer_kwargs':v['optimizer_kwargs'],'training_probe_total':v['training_probe_total']}
                    for m,v in decision['selected'].items()},'quality':all_quality,
        'no_model_inference':True,'device':'cpu','automatic_promotion':False}
    path=ROOT/'qualification-completion-audit-aggregate.json'
    with path.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('quality','trials')}))
    assert result['all_pass']


if __name__=='__main__':main()
