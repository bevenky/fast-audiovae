"""Focused acceptance and unchanged quiet-summary capture regressions."""
import copy
import gzip
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import diagnose_progressive_silence_v2 as audit
from joint_recovery_gates_v2 import summarize_regions


def checks():
    boundaries = {name: {'allclose_existing': True, 'exact': True} for name in
                  ('stage1_output', 'stage2_output', 'stage3_output', 'stage4_output',
                   'stage5_output', 'stage6_output', 'pre_tanh', 'waveform')}
    local = {}
    for name in ('stage2_ru1', 'stage2_ru2', 'stage2_ru3', 'stage3_up'):
        value = {'linear_identity': {'teacher_sum': {'passed': True}, 'gap_sum': {'passed': True}}}
        if name != 'stage3_up':
            value['residual_identity'] = copy.deepcopy(value['linear_identity'])
        local[name] = value
    return boundaries, local


def test_hidden_suffix_difference_is_disclosed_without_overriding_valid_group_and_waveform():
    boundaries, local = checks()
    boundaries['stage6_output'] = {'allclose_existing': False, 'exact': False, 'max_abs': 2e-5}
    before = copy.deepcopy((boundaries, local))
    result = audit.restoration_acceptance(boundaries, local)
    assert result['required_contract_passed'] and result['local_accounting_passed']
    assert not result['hidden_boundaries_allclose']
    assert 'stage6_output' in result['failed_hidden_boundaries']
    assert (boundaries, local) == before


@pytest.mark.parametrize('failure', ['group', 'waveform', 'linear_accounting', 'residual_skip'])
def test_local_accounting_group_or_waveform_failure_cannot_be_accepted(failure):
    boundaries, local = checks()
    if failure == 'group': boundaries['stage4_output']['allclose_existing'] = False
    elif failure == 'waveform': boundaries['waveform']['allclose_existing'] = False
    elif failure == 'linear_accounting': local['stage3_up']['linear_identity']['gap_sum']['passed'] = False
    else: local['stage2_ru2']['residual_identity']['gap_sum']['passed'] = False
    assert not audit.restoration_acceptance(boundaries, local)['required_contract_passed']


def test_missing_required_boundary_or_accounting_is_rejected_instead_of_vacuous_pass():
    boundaries, local = checks()
    for missing in ('stage4_output', 'waveform'):
        incomplete = {k: v for k, v in boundaries.items() if k != missing}
        with pytest.raises(ValueError): audit.restoration_acceptance(incomplete, local)
    with pytest.raises(ValueError): audit.restoration_acceptance(boundaries, {})
    del local['stage2_ru1']['residual_identity']['gap_sum']
    with pytest.raises(ValueError): audit.restoration_acceptance(boundaries, local)


def quiet_rows():
    rows = []
    for source, start, samples in [('startup', 0, 960), ('interior', 48000, 13)]:
        rows.append({'window_id': source + ':' + str(start), 'source_id': source,
            'source_start_sample': start, 'source_stop_sample': start + samples,
            'valid_samples': samples, 'is_quiet': True, 'teacher_rms': 5e-6,
            'student_rms': 7e-6, 'residual_limit': 1e-5, 'output_rms_limit': 1e-5,
            'source_reference_exact_zero': source == 'startup', 'failure_category': 'passed',
            'residual_square_sum': samples * 4e-12, 'centered_residual_square_sum': samples * 3e-12,
            'prediction_mean': 3e-6, 'teacher_mean': 2e-6, 'residual_mean': 1e-6})
    return rows


def test_full_quiet_rows_are_captured_once_without_extending_original_summary_schema(tmp_path):
    rows = quiet_rows(); before = copy.deepcopy(rows); calls = []
    expected = summarize_regions(rows)
    def evaluate(monitor, model, teacher, crops, objective, summarize=None):
        calls.append((monitor, model, teacher, crops, objective))
        return {'aggregate': {'quiet_windows': 2}, 'quiet_regions': summarize(rows), 'rows': []}
    output = tmp_path / 'quiet.jsonl.gz'
    result = audit.evaluate_with_quiet_capture(None, None, None, ['fixed'], None, output, evaluate_fn=evaluate)
    assert len(calls) == 1 and calls[0][3] == ['fixed']
    assert result == {'aggregate': {'quiet_windows': 2}, 'quiet_regions': expected, 'rows': []}
    assert rows == before
    with gzip.open(output, 'rt') as handle:
        assert [json.loads(line) for line in handle] == rows
    assert 'prediction_mean' not in result['quiet_regions']['regions']['all_quiet']
    assert 'prediction_mean' in rows[0]  # Extra diagnostics remain in captured rows only.


def test_failed_evaluation_does_not_retry_or_change_the_original_callback(tmp_path):
    original = audit.base.quiet_window_metrics
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('intentional evaluator failure')
    with pytest.raises(RuntimeError, match='intentional evaluator failure'):
        audit.evaluate_with_quiet_capture(None, None, None, [], None,
                                          tmp_path / 'quiet.jsonl.gz', evaluate_fn=evaluate)
    assert calls == [1]
    assert audit.base.quiet_window_metrics is original


def test_missing_or_repeated_quiet_capture_fails_without_repeating_evaluation(tmp_path):
    for count in (0, 2):
        calls = []
        def evaluate(*args, summarize=None):
            calls.append(1)
            for _ in range(count): summarize(quiet_rows())
            return {'aggregate': {}}
        with pytest.raises(RuntimeError):
            audit.evaluate_with_quiet_capture(None, None, None, [], None,
                                              tmp_path / f'quiet-{count}.jsonl.gz', evaluate_fn=evaluate)
        assert calls == [1]
