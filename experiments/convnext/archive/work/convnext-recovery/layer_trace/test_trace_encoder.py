"""Offline scalar report and prefix-geometry tests; no model or torch import."""
import json,unittest
import numpy as np
from trace_encoder import comparison,prefix_output_limit

class TraceTests(unittest.TestCase):
    def test_numerical_and_bitwise_separate(self):
        a=np.zeros((1,2,4),np.float32);b=a.copy();b[0,0,0]=-0.
        r=comparison(a,b);self.assertFalse(r['bitwise_equal']);self.assertTrue(r['within_tolerance']);self.assertEqual(r['max_abs_error'],0)
        json.dumps(r,allow_nan=False)
    def test_report_identifies_time_and_channel(self):
        a=np.zeros((1,2,4),np.float32);b=a.copy();a[0,1,2]=.01
        r=comparison(a,b);self.assertFalse(r['within_tolerance']);self.assertEqual(r['failed_samples'],1)
        self.assertEqual(r['max_error_index'],[0,1,2]);self.assertEqual(r['per_time_max_abs'][2],r['max_abs_error'])
        self.assertEqual(r['per_channel_max_abs'][0],0);json.dumps(r,allow_nan=False)
    def test_nonfinite_or_dtype_rejected(self):
        x=np.zeros((1,1,3),np.float32)
        with self.assertRaises(TypeError):comparison(x.astype('float64'),x)
        x[0,0,1]=np.nan
        with self.assertRaises(ValueError):comparison(x,x)
    def test_stride_bound_excludes_missing_right_input(self):
        # s8 kernel16 left8, outputs0..255 use input up to2047.
        self.assertEqual(prefix_output_limit(2048,16,1,8,8,2048),256)
        self.assertEqual(prefix_output_limit(2047,16,1,8,8,2048),255)
    def test_dilated_causal_bound(self):
        self.assertEqual(prefix_output_limit(2048,7,9,1,54,64),64)
        self.assertEqual(prefix_output_limit(12,7,9,1,54,64),12)
    def test_geometry_bad_input(self):
        for args in [(0,7,1,1,6,64),(1,7,0,1,6,64),(1,7,1,1,-1,64)]:
            with self.assertRaises(ValueError):prefix_output_limit(*args)

if __name__=='__main__':unittest.main()
