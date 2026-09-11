"""Pure scalar/receipt tests; no model construction, inference or optimizer step."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import optimizer_comparison_runner as run


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


@pytest.fixture
def sealed_trial(tmp_path):
    config = tmp_path / 'config.json'
    manifest = tmp_path / 'sources.json'
    completed = tmp_path / 'candidate-completed.json'
    decision_path = tmp_path / 'qualification.json'
    write_json(config, {'initializer_state_sha256':'fresh-state', 'source_plan':'fixed-plan'})
    write_json(manifest, {name:'0'*64 for name in (
        'optimizer_comparison_runner.py', 'startup_mixed_optimizer_update.py', 'recovery_optimizers.py')})
    write_json(completed, {'version':run.VERSION, 'complete':True, 'updates':64,
                          'ordinary_unique_sources':768, 'all_preservation_checks_passed':True})
    kwargs = {'matrix_lr':3e-4, 'adam_lr':3e-5}
    decision = {'version':'audiovae2_optimizer_qualification_v1',
        'all_candidates_attempted':True, 'control_parity_passed':True,
        'development_used_for_selection':False,
        'source_manifest_sha256':run.base.sha(manifest), 'config_sha256':run.base.sha(config),
        'selected':{'muon':{'optimizer_kwargs':kwargs, 'eligible':True,
                          'completed_path':str(completed), 'completed_sha256':run.base.sha(completed)}}}
    write_json(decision_path, decision)
    trial = {'updates':2000, 'method':'muon', 'optimizer_kwargs':deepcopy(kwargs),
             'source_manifest':str(manifest), 'qualification_path':str(decision_path),
             'qualification_sha256':run.base.sha(decision_path)}
    return trial, config, decision, decision_path, manifest, completed


def test_numeric_parity_ignores_only_timing_and_allows_new_alias():
    old = {'step':17, 'total':.25, 'q_accepted_fraction':.5,
           'q_actual_norm_over_adam':.4, 'q_gram_condition':None,
           'startup_anchor_after_amplitude_max_excess':-1e-18,
           **{key:1. for key in ('step_seconds','elapsed_seconds','training_update_seconds',
               'validation_seconds','waiting_seconds','q_step_seconds',
               'q_ordinary_update_seconds','q_auxiliary_seconds')}}
    new = {key:(900. if key.endswith('seconds') else value) for key,value in old.items()}
    new['q_actual_norm_over_optimizer'] = old['q_actual_norm_over_adam']
    assert run.numeric_parity(old,new) == {'passed':True, 'compared':6, 'mismatched_fields':[]}


@pytest.mark.parametrize('key,old_value,new_value', [
    ('total',.25,math.nextafter(.25,1.)),
    ('startup_anchor_after_amplitude_max_excess',-1e-18,1e-18),
    ('startup_anchor_after_passed',6,5),
    ('q_accepted_fraction',1.,.5),
    ('q_caps_passed',1,0),
])
def test_numeric_parity_never_tolerates_quality_or_policy_changes(key,old_value,new_value):
    result = run.numeric_parity({'step':1,key:old_value}, {'step':1,key:new_value})
    assert result == {'passed':False, 'compared':2, 'mismatched_fields':[key]}
    assert not run.numeric_parity({'step':1,key:old_value}, {'step':1})['passed']


def test_qualification_requires_exact_selected_configuration(sealed_trial):
    trial, config, _, path, _, _ = sealed_trial
    assert run.validate_recovery_qualification(trial,config) == path
    altered = deepcopy(trial)
    altered['optimizer_kwargs']['matrix_lr'] *= 2
    with pytest.raises(ValueError):
        run.validate_recovery_qualification(altered,config)


@pytest.mark.parametrize('field,value', [
    ('version','foreign-version'),
    ('all_candidates_attempted',False),
    ('control_parity_passed',False),
    ('development_used_for_selection',True),
    ('eligible',False),
])
def test_resealed_ineligible_decision_is_rejected(sealed_trial,field,value):
    trial, config, decision, path, _, _ = sealed_trial
    if field=='eligible':
        decision['selected']['muon'][field] = value
    else:
        decision[field] = value
    write_json(path,decision)
    trial['qualification_sha256'] = run.base.sha(path)
    with pytest.raises(ValueError):
        run.validate_recovery_qualification(trial,config)


@pytest.mark.parametrize('artifact', ['decision','manifest','completed','config'])
def test_any_bound_artifact_byte_change_is_rejected(sealed_trial,artifact):
    trial, config, _, decision_path, manifest, completed = sealed_trial
    path = {'decision':decision_path, 'manifest':manifest, 'completed':completed, 'config':config}[artifact]
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):
        run.validate_recovery_qualification(trial,config)


def test_different_valid_config_cannot_reuse_qualification(sealed_trial,tmp_path):
    trial, config, _, _, _, _ = sealed_trial
    other = tmp_path / 'other-config.json'
    payload = json.loads(config.read_text())
    payload['initializer_state_sha256'] = 'another-fresh-state'
    write_json(other,payload)
    with pytest.raises(ValueError):
        run.validate_recovery_qualification(trial,other)
