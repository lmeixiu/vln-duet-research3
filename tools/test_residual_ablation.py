"""CPU-only checks for paired evaluation aggregation; no torch dependency."""
import unittest
from evaluate_residual_ablation import compare


def metrics(ids, success):
    return {'aggregate': {'sr': sum(success)/len(success)*100},
            'per_instruction': {'instr_id': ids, 'success': success,
                                'spl': success, 'nav_error': [0 if x else 5 for x in success]}}


class ComparisonTests(unittest.TestCase):
    def test_pairs_by_id_not_list_position(self):
        on = metrics(['a', 'b', 'c'], [1, 0, 1])
        off = metrics(['c', 'b', 'a'], [1, 1, 0])
        pa = [{'instr_id': i, 'trajectory': [[i]]} for i in ['a', 'b', 'c']]
        pb = [{'instr_id': i, 'trajectory': [[i+'changed' if i == 'a' else i]]} for i in ['c', 'a', 'b']]
        summary, rows = compare(on, off, pa, pb)
        self.assertEqual(summary['helped_count'], 1)
        self.assertEqual(summary['hurt_count'], 1)
        self.assertEqual(summary['changed_trajectory_count'], 1)
        self.assertEqual(summary['on_minus_off']['sr'], 0)
        self.assertTrue(rows[0]['helped'])

    def test_reject_mismatched_ids(self):
        with self.assertRaises(ValueError):
            compare(metrics(['a'], [1]), metrics(['b'], [1]), [], [])

    def test_identical_runs(self):
        m = metrics(['a'], [1])
        p = [{'instr_id': 'a', 'trajectory': [['v1'], ['v2']]}]
        summary, _ = compare(m, m, p, p)
        self.assertEqual(summary['helped_count'], 0)
        self.assertEqual(summary['hurt_count'], 0)
        self.assertEqual(summary['changed_trajectory_count'], 0)


if __name__ == '__main__':
    unittest.main()
