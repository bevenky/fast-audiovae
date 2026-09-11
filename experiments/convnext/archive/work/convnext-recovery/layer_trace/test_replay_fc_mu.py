import unittest,json
import numpy as np
from replay_fc_mu import channel_summary

class HeadSummaryTests(unittest.TestCase):
    def test_only_last_channel_is_identified(self):
        ref=np.zeros((1,64,146),np.float32);a=ref.copy();a[:,63]=np.arange(146,dtype=np.float32)/100
        r=channel_summary(a,ref)
        self.assertEqual(r['per_channel_max_abs_error'][:63],[0.]*63)
        self.assertAlmostEqual(r['per_channel_max_abs_error'][63],1.45,places=6)
        self.assertEqual(len(r['channel63_error']),146);json.dumps(r,allow_nan=False)
    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):channel_summary(np.zeros((1,63,2)),np.zeros((1,63,2)))

if __name__=='__main__':unittest.main()
