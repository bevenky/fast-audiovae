from copy import deepcopy
import unittest

import joint_recovery_gates_v2 as gates
from test_joint_recovery_gates_v1 import report, POLICY


def windows():
    result = []
    for source, start, teacher, zero in [('A', 0, 9e-6, True), ('A', 960, 6e-4, True),
                                        ('B', 1920, 9e-6, True), ('B', 48000, 9e-6, False)]:
        result.append({'window_id': source+':'+str(start), 'source_id': source,
            'source_start_sample': start, 'source_stop_sample': start+960, 'valid_samples': 960,
            'is_quiet': True, 'teacher_rms': teacher, 'student_rms': teacher,
            'residual_limit': max(.02**.5*teacher, 1e-5),
            'output_rms_limit': max(10**.05*teacher, 1e-5), 'source_reference_exact_zero': zero,
            'failure_category': 'passed', 'residual_square_sum': 4e-6**2*960,
            'centered_residual_square_sum': 3e-6**2*960})
    return result


def sample(**kwargs):
    r = report(**kwargs)
    r['quiet_regions'] = gates.summarize_regions(windows())
    return r


def decide(reports, steps=None):
    return gates.classify_review(reports, steps or [5625+500*i for i in range(len(reports))], POLICY)


class ContinuationGates(unittest.TestCase):
    def test_source_zero_teacher_transient_is_separate_from_near_startup(self):
        r = gates.summarize_regions(windows())['regions']
        self.assertEqual(r['near_startup_first20ms']['windows'], 1)
        self.assertEqual(r['source_zero_20to40ms']['windows'], 1)
        self.assertEqual(r['source_zero_after40ms']['windows'], 1)
        self.assertEqual(r['near_after800ms']['windows'], 1)
        self.assertEqual(r['near_silence']['windows'], 3)

    def test_excess_level_uses_original_limits_and_valid_sample_weighting(self):
        w = windows()
        w[0]['student_rms'] = w[0]['output_rms_limit']+2e-6
        w[0]['failure_category'] = 'amplitude_only'
        r = gates.summarize_regions(w)['regions']['all_quiet']
        self.assertEqual(r['amplitude_failed'], 1)
        self.assertAlmostEqual(r['output_limit_excess_rms'], 1e-6, places=14)

    def test_identity_rejects_teacher_or_source_panel_change(self):
        for key, new in [('teacher_rms', 8e-6), ('source_reference_exact_zero', False)]:
            a, b = sample(), sample(mae=.009)
            w = windows(); w[0][key] = new
            b['quiet_regions'] = gates.summarize_regions(w)
            self.assertEqual(decide([a, b])['action'], 'pause_for_diagnosis')

    def test_start_and_improving_two_review_history_continue(self):
        self.assertEqual(decide([sample()])['action'], 'continue')
        r = decide([sample(), sample(mae=.009), sample(mae=.008)])
        self.assertEqual(r['action'], 'continue')
        self.assertEqual(r['step_horizons']['overall_stall'], 1000)

    def test_stall_is_1000_updates_not_four_new_reviews(self):
        self.assertEqual(decide([sample(), sample()])['action'], 'continue')
        self.assertEqual(decide([sample(), sample(), sample()])['action'], 'pause_for_diagnosis')

    def test_repeated_floor_regression_uses_small_absolute_guard(self):
        history = [sample(), sample(mae=.009), sample(mae=.008)]
        for r in history[1:]:
            r['quiet_regions']['regions']['source_zero_after40ms']['residual_rms'] = 5.2e-6
        self.assertEqual(decide(history[:2])['action'], 'continue')
        result = decide(history)
        self.assertEqual(result['action'], 'pause_for_diagnosis')
        self.assertEqual(result['region_repeated_regressions'][0]['region'], 'source_zero_after40ms')

    def test_startup_uses_larger_absolute_guard(self):
        history = [sample(), sample(mae=.009), sample(mae=.008)]
        for r in history[1:]:
            r['quiet_regions']['regions']['near_startup_first20ms']['residual_rms'] = 5.2e-6
        self.assertEqual(decide(history)['action'], 'continue')

    def test_count_only_changes_alert_and_do_not_pause(self):
        history = [sample(), sample(mae=.009), sample(mae=.008)]
        for r in history[1:]:
            v = r['quiet_regions']['regions']['source_zero_after40ms']
            v['failure_categories']['passed'] = 0; v['failure_categories']['amplitude_only'] = 1
            v['failed'] = v['amplitude_failed'] = 1
        result = decide(history)
        self.assertEqual(result['action'], 'continue')

    def test_region_stall_requires_two_full1000_update_segments(self):
        history = [sample(mae=.01-i*.0005) for i in range(5)]
        for r in history:
            v = r['quiet_regions']['regions']['near_after800ms']
            v['failed'] = v['amplitude_failed'] = 1
            v['failure_categories']['passed'] = 0; v['failure_categories']['amplitude_only'] = 1
        self.assertEqual(decide(history[:4])['action'], 'continue')
        result = decide(history)
        self.assertEqual(result['action'], 'pause_for_diagnosis')
        self.assertEqual(result['region_stalls'][0]['optimizer_steps'], [5625, 6625, 7625])

    def test_final375_endpoint_does_not_invent_full_stall_interval(self):
        history = [sample(mae=.01-i*.0005) for i in range(9)]
        history[-1] = deepcopy(history[-2]); history.append(deepcopy(history[-1]))
        result = decide(history, list(range(5625, 10000, 500))+[10000])
        self.assertEqual(result['action'], 'continue')
        self.assertFalse(result['step_horizons']['last_interval_is_full500'])

    def test_missing_review_or_reset_history_is_rejected(self):
        for steps in ([5625, 6625], [6625, 7125], [5625, 5625]):
            self.assertEqual(decide([sample(), sample()], steps)['action'], 'pause_for_diagnosis')


if __name__ == '__main__': unittest.main()
