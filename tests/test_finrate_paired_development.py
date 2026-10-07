import unittest

from finrate_paired_development import paired_comparison


class PairedDevelopmentTests(unittest.TestCase):
    def setUp(self):
        # A failed generation is a zero, not a removed question.
        self.rows = [
            {'q_id':'a','condition':'embedding','correct':False,'key_point_fraction':0.0,'label':'FAILURE'},
            {'q_id':'b','condition':'embedding','correct':True,'key_point_fraction':1.0,'label':'CORRECT'},
            {'q_id':'a','condition':'raw_agent','correct':True,'key_point_fraction':1.0,'label':'CORRECT'},
            {'q_id':'b','condition':'raw_agent','correct':False,'key_point_fraction':0.5,'label':'PARTIAL'},
        ]

    def test_paired_wins_and_losses_keep_failures_and_question_identity(self):
        row = paired_comparison(list(reversed(self.rows)), ['a','b'], ['embedding','raw_agent'])[0]
        self.assertEqual(row['n'],2)
        self.assertEqual(row['correctness_pairs'],{'both_correct':0,'left_only_correct':1,'right_only_correct':1,'neither_correct':0})
        self.assertEqual(row['key_point_pairs'],{'left_better':1,'right_better':1,'tie':0})
        self.assertEqual(row['right_minus_left_mean_key_point_fraction'],0.25)

    def test_incomplete_duplicate_and_unrequested_rows_are_rejected(self):
        for rows in [self.rows[1:],self.rows+self.rows[:1],self.rows+[{'q_id':'c','condition':'embedding'}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                paired_comparison(rows,['a','b'],['embedding','raw_agent'])


if __name__ == '__main__': unittest.main()
