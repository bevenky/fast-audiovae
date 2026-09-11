"""No-model tests for conservative fixed-panel review decisions."""
from copy import deepcopy
import json
import math
import unittest

from joint_recovery_gates_v1 import ReviewPolicy, classify_review


POLICY = ReviewPolicy(expected_sources=2)


def report(*, mae=.01, mel=.5, correlation=.95, quiet=.0001,
           source_mae=None, source_quiet=None, gains=None,
           quiet_samples=1920, active_samples=1920, near=None,
           quiet_failed=0, near_failed=0):
    source_mae = source_mae or [mae, mae]
    source_quiet = source_quiet or [quiet, quiet]
    gains = gains or [1., 1.]
    rows, by_source = [], {}
    n = quiet_samples + active_samples
    for index, sid in enumerate(('A', 'B')):
        rows.append({'source_id': sid, 'samples': n, 'quiet_samples': quiet_samples,
                     'nonquiet_samples': active_samples, 'quiet_windows': 2,
                     'quiet_failed': quiet_failed, 'quiet_error_sum': source_quiet[index]**2 * quiet_samples,
                     'mae': source_mae[index], 'mel': mel, 'overshoot': 0})
        by_source[sid] = {'valid_samples': n, 'quiet_windows': 2, 'quiet_failed_windows': quiet_failed,
                          'near_silence_windows': 1, 'near_silence_failed_windows': near_failed,
                          'active_teacher_energy': active_samples * .01,
                          'active_student_energy': active_samples * .01 * gains[index]**2}
    value = {'aggregate': {'sources': 2, 'samples': 2*n, 'mae': mae, 'mel': mel,
                           'group_mse': .01, 'quiet_residual_rms_mean': quiet,
                           'nonquiet_cosine_mean': correlation, 'overshoot_samples': 0,
                           'peak_abs_max': .8, 'quiet_windows': 4,
                           'quiet_failed_windows': 2*quiet_failed},
             'rows': rows, 'overview_window_metrics': {'by_source': by_source,
             'near_silence_windows': 2, 'near_silence_failed_windows': 2*near_failed}}
    if near is not None:
        ns = min(quiet_samples, 960)
        nr = [near, near] if isinstance(near, (int, float)) else near
        total_error = sum(r*r*ns for r in nr)
        value['recovery_window_metrics'] = {
            'near_samples': 2*ns, 'near_error_sum': total_error,
            'near_residual_rms': math.sqrt(total_error/(2*ns)) if ns else None,
            'by_source': {sid: {'near_samples': ns, 'near_error_sum': nr[i]**2*ns,
                               'near_residual_rms': nr[i] if ns else None}
                          for i, sid in enumerate(('A', 'B'))}}
    return value


def decision(*reports):
    return classify_review(reports, POLICY)


def keys(result):
    return {(r['metric'], r['scope'], r['source_id']) for r in result['repeated_regressions']}


class ReviewGateTests(unittest.TestCase):
    def test_initial_continue_and_json_finite(self):
        result = decision(report())
        self.assertEqual(result['action'], 'continue')
        self.assertFalse(result['automatic_freezing'])
        json.dumps(result, allow_nan=False)

    def test_single_regression_does_not_pause(self):
        result = decision(report(), report(mae=.011))
        self.assertTrue(result['material_regressions'])
        self.assertEqual(result['action'], 'continue')

    def test_same_source_same_metric_persists_against_initial(self):
        result = decision(report(), report(source_mae=[.011, .01]),
                          report(source_mae=[.011, .01]))
        self.assertEqual(result['action'], 'pause_for_diagnosis')
        self.assertEqual(keys(result), {('mae', 'source', 'A')})

    def test_changing_regressing_source_is_not_persistence(self):
        result = decision(report(), report(source_mae=[.011, .01]),
                          report(source_mae=[.01, .011]))
        self.assertEqual(result['action'], 'continue')
        self.assertEqual(keys(result), set())

    def test_regression_then_recovery_clears(self):
        result = decision(report(), report(mae=.011), report(mae=.0101))
        self.assertEqual(result['action'], 'continue')

    def test_relative_to_previous_regression_is_caught(self):
        result = decision(report(), report(mae=.008), report(mae=.0085), report(mae=.009))
        self.assertIn(('mae', 'aggregate', None), keys(result))

    def test_aggregate_mel_regression_persists(self):
        result = decision(report(), report(mel=.54), report(mel=.54))
        self.assertIn(('mel', 'aggregate', None), keys(result))

    def test_different_metric_does_not_count_as_persistence(self):
        result = decision(report(), report(mel=.54), report(mae=.011))
        self.assertEqual(result['action'], 'continue')

    def test_source_mae_requires_absolute_floor(self):
        result = decision(report(source_mae=[.0001, .01]),
                          report(source_mae=[.00015, .01]), report(source_mae=[.00015, .01]))
        self.assertEqual(result['action'], 'continue')

    def test_quiet_requires_relative_and_absolute_increase(self):
        for before, after in ((1e-5, 1.5e-5), (.001, .00105)):
            with self.subTest(before=before):
                result = decision(report(quiet=before), report(quiet=after), report(quiet=after))
                self.assertEqual(result['action'], 'continue')
        result = decision(report(), report(quiet=.00012), report(quiet=.00012))
        self.assertIn(('quiet_rms', 'aggregate', None), keys(result))

    def test_source_quiet_can_pause_without_aggregate_regression(self):
        result = decision(report(), report(source_quiet=[.00012, .0001]),
                          report(source_quiet=[.00012, .0001]))
        self.assertEqual(keys(result), {('quiet_rms', 'source', 'A')})

    def test_short_source_regions_do_not_trigger_source_gates(self):
        result = decision(report(quiet_samples=959, active_samples=959),
                          report(source_quiet=[.0002, .0001], gains=[1.1, 1/1.1], quiet_samples=959, active_samples=959),
                          report(source_quiet=[.0002, .0001], gains=[1.1, 1/1.1], quiet_samples=959, active_samples=959))
        self.assertEqual(keys(result), set())

    def test_failure_counts_are_only_alerts(self):
        result = decision(report(), report(quiet_failed=1, near_failed=1),
                          report(quiet_failed=2, near_failed=1))
        self.assertEqual(result['action'], 'continue')
        self.assertTrue(any(a['kind'] == 'count_only_alert' for a in result['alerts']))

    def test_level_error_is_distance_from_teacher_not_amplitude_direction(self):
        result = decision(report(gains=[.8, .8]), report(gains=[.9, .9]), report(gains=[.95, .95]))
        self.assertEqual(keys(result), set())
        result = decision(report(), report(gains=[1.1, 1.1]), report(gains=[1.1, 1.1]))
        self.assertIn(('active_level', 'aggregate', None), keys(result))

    def test_zero_student_active_energy_valid_but_regression(self):
        result = decision(report(), report(gains=[0., 0.]), report(gains=[0., 0.]))
        self.assertIn(('active_level', 'aggregate', None), keys(result))
        self.assertTrue(any(a['kind'] == 'active_output_collapsed' for a in result['alerts']))
        self.assertFalse(any(f['kind'] == 'invalid_or_invariant_failure' for f in result['flags']))
        json.dumps(result, allow_nan=False)

    def test_stall_waits_until_four_reviews(self):
        for count in (1, 2, 3, 4):
            result = decision(*(report() for _ in range(count)))
            self.assertEqual(result['action'], 'continue')
        result = decision(*(report() for _ in range(5)))
        self.assertTrue(result['stall']['two_intervals_stalled'])
        self.assertEqual(result['action'], 'pause_for_diagnosis')

    def test_small_real_progress_is_not_stall(self):
        result = decision(*(report(mae=.01 * .997**i) for i in range(5)))
        self.assertEqual(result['action'], 'continue')
        result = decision(*(report(correlation=.95+i*.00011) for i in range(5)))
        self.assertEqual(result['action'], 'continue')

    def test_progress_on_one_of_last_two_intervals_prevents_stall(self):
        result = decision(report(), report(), report(), report(mae=.009), report(mae=.009))
        self.assertEqual(result['action'], 'continue')

    def test_near_silence_guard_and_missing_legacy_field(self):
        result = decision(report(near=1e-5), report(near=3e-5), report(near=3e-5))
        self.assertIn(('near_rms', 'aggregate', None), keys(result))
        self.assertIn(('near_rms', 'source', 'A'), keys(result))
        result = decision(report(), report(near=3e-5), report(near=3e-5))
        self.assertEqual(result['action'], 'continue')

    def test_near_silence_source_persistence_and_minimum_samples(self):
        result = decision(report(near=1e-5), report(near=[3e-5, 1e-5]), report(near=[1e-5, 3e-5]))
        self.assertNotIn(('near_rms', 'source', 'A'), keys(result))
        self.assertNotIn(('near_rms', 'source', 'B'), keys(result))
        result = decision(report(near=1e-5, quiet_samples=400),
                          report(near=3e-5, quiet_samples=400), report(near=3e-5, quiet_samples=400))
        self.assertEqual(result['action'], 'continue')

    def test_invalid_nonfinite_and_invariant_failures_pause_immediately(self):
        cases = []
        r = report(); r['aggregate']['mel'] = float('nan'); cases.append(r)
        r = report(); r['aggregate']['overshoot_samples'] = 1; cases.append(r)
        r = report(); r['aggregate']['peak_abs_max'] = 1.1; cases.append(r)
        r = report(); r['rows'][0]['overshoot'] = 1; cases.append(r)
        r = report(); r['aggregate']['sources'] = 3; cases.append(r)
        r = report(); r['rows'][1]['source_id'] = 'A'; cases.append(r)
        r = report(); r['overview_window_metrics']['by_source']['A']['valid_samples'] -= 1; cases.append(r)
        r = report(near=1e-5); r['recovery_window_metrics']['near_residual_rms'] = 2e-5; cases.append(r)
        for index, value in enumerate(cases):
            with self.subTest(index=index):
                result = decision(value)
                self.assertEqual(result['action'], 'pause_for_diagnosis')
                self.assertEqual(result['flags'][0]['kind'], 'invalid_or_invariant_failure')
                json.dumps(result, allow_nan=False)

    def test_changed_teacher_defined_coverage_or_energy_pauses(self):
        changed = report(quiet_samples=960, active_samples=2880)
        result = decision(report(), changed)
        self.assertEqual(result['flags'][0]['kind'], 'invalid_or_invariant_failure')
        changed = report()
        changed['overview_window_metrics']['by_source']['A']['active_teacher_energy'] *= 1.01
        self.assertEqual(decision(report(), changed)['action'], 'pause_for_diagnosis')

    def test_source_order_is_not_source_identity(self):
        changed = report(); changed['rows'].reverse()
        self.assertEqual(decision(report(), changed)['action'], 'continue')

    def test_no_reports_or_missing_metrics_pauses(self):
        self.assertEqual(decision()['action'], 'pause_for_diagnosis')
        self.assertEqual(decision({})['action'], 'pause_for_diagnosis')

    def test_input_reports_are_not_modified(self):
        history = [report(near=1e-5), report(near=3e-5)]
        original = deepcopy(history)
        classify_review(history, POLICY)
        self.assertEqual(history, original)


if __name__ == '__main__':
    unittest.main()
