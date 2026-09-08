"""Tiny synthetic checks of scheduling, complete timing grids and aggregation only.

No numerical model libraries or decoder execution are imported.
"""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('candidate_campaign', Path(__file__).with_name('campaign_v2.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ProtocolChecks(unittest.TestCase):
    def test_actual_short_mimi_lengths(self):
        for length in (53, 57, 60, 63):
            q = c.validation_lengths(length, 170)
            self.assertEqual((q['probe'], q['short'], q['future_cut'], q['other']), (length, 33, 32, 31))
            self.assertTrue(all(0 < n <= length for n in q['boundary_lengths']))
            self.assertTrue(all(0 < n < length for n in q['boundary_cuts']))

    def test_tiny_real_lengths(self):
        for length in (2, 3, 7, 31, 32, 33, 34, 64, 65, 66, 170):
            for other in (1, 2, 30, 31, 53, 170):
                q = c.validation_lengths(length, other)
                self.assertEqual(q['probe'], min(65, length))
                self.assertEqual(q['short'], min(33, q['probe']))
                self.assertEqual(q['future_cut'], min(32, q['probe'] - 1))
                self.assertEqual(q['other'], min(31, other))
                self.assertTrue(0 < q['future_cut'] < q['probe'] <= length)
                self.assertTrue(0 < q['other'] <= other)

    def test_invalid_or_vacuous_probe_rejected(self):
        for length, other in ((0, 31), (1, 31), (65, 0), (-1, 31), (65.0, 31)):
            with self.assertRaises(ValueError):
                c.validation_lengths(length, other)

    def test_frozen_long_length_plan(self):
        q = c.validation_lengths(170, 170)
        self.assertEqual((q['probe'], q['short'], q['future_cut'], q['other']), (65, 33, 32, 31))
        self.assertEqual(q['boundary_lengths'], [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64])
        self.assertEqual(q['boundary_cuts'], [1, 8, 64])

    def test_frozen_pair_schedule(self):
        uids = ['clip' + str(i) for i in range(10)]
        plan = c.schedule(uids)
        self.assertEqual(plan, c.schedule(uids))
        self.assertEqual(len(plan), 50)
        self.assertEqual({(r['repeat'], r['uid']) for r in plan}, {(j, uid) for j in range(5) for uid in uids})
        first = 0
        for block in plan:
            self.assertEqual(set(block['models']), set(c.MODELS))
            x, y = block['models'].index('fast_fp32'), block['models'].index('int8_large')
            self.assertEqual(abs(x - y), 1)
            first += x < y
        self.assertEqual(first, 25)

    def fixture(self):
        rows = []
        for name in c.MODELS:
            for uid, duration, rtf in [('a', 1, .1), ('b', 9, .3)]:
                factor = .8 if name == 'int8_large' else 1
                for repeat in range(5):
                    rows.append({'model': name, 'uid': uid, 'repeat': repeat,
                                 'generated_seconds': duration,
                                 'elapsed_ns': round(1e9 * duration * rtf * factor)})
        return rows

    def test_equal_clip_and_corpus_distinct(self):
        result = c.summarize(self.fixture(), ['a', 'b'])
        self.assertAlmostEqual(result['models']['fast_fp32']['decoder_rtf'], .2)
        self.assertAlmostEqual(result['models']['fast_fp32']['corpus_decoder_rtf'], .28)
        self.assertAlmostEqual(result['paired']['time_reduction_percent'], 20)
        self.assertTrue(result['paired']['confirmed_minimum_10_percent'])
        self.assertFalse(result['paired']['promotion_approved'])

    def test_missing_duplicate_duration_rejected(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):
            c.summarize(rows[:-1], ['a', 'b'])
        with self.assertRaises(ValueError):
            c.summarize(rows + [rows[0]], ['a', 'b'])
        rows = copy.deepcopy(rows)
        next(r for r in rows if r['model'] == 'int8_large')['generated_seconds'] = 2
        with self.assertRaises(ValueError):
            c.summarize(rows, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
