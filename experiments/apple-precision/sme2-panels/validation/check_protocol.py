"""Tiny synthetic checks of scheduling, complete timing grids and aggregation only.

No numerical model libraries or decoder execution are imported.
"""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('candidate_campaign', Path(__file__).with_name('campaign.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ProtocolChecks(unittest.TestCase):
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
