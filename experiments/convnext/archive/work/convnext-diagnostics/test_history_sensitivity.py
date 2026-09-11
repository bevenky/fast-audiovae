"""Offline helper checks only, no checkpoint or teacher loading."""
import json,unittest
from types import SimpleNamespace
import numpy as np
from diagnose_history_sensitivity import raw_metrics,lag_gain_diagnostic,phase_variance,slope,select_cases,category

class HistoryHelperTests(unittest.TestCase):
    def test_lag_gain_recovers_known_delay(self):
        rng=np.random.default_rng(55);target=rng.normal(size=3000)
        prediction=np.zeros_like(target);prediction[7:]=target[:-7]/1.1
        fit=lag_gain_diagnostic(prediction,target)
        self.assertEqual(fit['fit']['lag_samples'],7)
        self.assertAlmostEqual(fit['fit']['prediction_gain'],1.1,places=12)
        self.assertLess(fit['fit']['mse'],1e-25)
        self.assertGreater(fit['raw_common_grid']['mse'],.1)
        self.assertEqual(fit['common_samples'],2904)

    def test_negative_delay_and_bounded_gain(self):
        rng=np.random.default_rng(99);t=rng.normal(size=2000);p=np.zeros_like(t);p[:-9]=t[9:]*.5
        fit=lag_gain_diagnostic(p,t)
        self.assertEqual(fit['fit']['lag_samples'],-9)
        self.assertEqual(fit['fit']['prediction_gain'],1.25)
        self.assertGreater(fit['fit']['mse'],0)

    def test_zero_ties_and_short_inputs(self):
        for n in (1,7,32,200):
            m=lag_gain_diagnostic(np.zeros(n),np.zeros(n))
            self.assertEqual(m['fit']['lag_samples'],0);self.assertEqual(m['fit']['prediction_gain'],1.)
            self.assertEqual(m['fit']['mse'],0);json.dumps(m,allow_nan=False)

    def test_phase_template_excludes_dc(self):
        p=np.tile([1.,-1.,2.,-2.],20)+3;t=np.zeros_like(p)
        m=phase_variance(p,t,4)
        self.assertAlmostEqual(m['residual_dc'],3.)
        self.assertAlmostEqual(m['phase_mean_variance'],2.5)
        self.assertEqual(m['cycles'],20)
        self.assertFalse(phase_variance(np.zeros(3),np.zeros(3),4)['available'])

    def test_metrics_preserve_raw_primary(self):
        p=np.full(100,2.);t=np.ones(100)
        self.assertEqual(raw_metrics(p,t)['mse'],1.)
        self.assertEqual(lag_gain_diagnostic(p,t)['raw_common_grid']['mse'],1.)
        with self.assertRaises(ValueError):raw_metrics([1,np.nan],[1,2])
        with self.assertRaises(ValueError):raw_metrics([1],[1,2])
        with self.assertRaises(ValueError):lag_gain_diagnostic([1],[1],gain_bounds=(1.1,2.))

    def test_slope_and_serialization(self):
        self.assertAlmostEqual(slope([1,.5,.1,.01,0],[2,1,.2,.02,9],True,True),1.)
        self.assertIsNone(slope([1],[2]))
        json.dumps(raw_metrics(np.zeros(7),np.zeros(7)),allow_nan=False)

    def test_case_selection_real_history_and_natural_priority(self):
        def crop(name,frames,start=0):
            return SimpleNamespace(source_id=name,context_start_frame=start,start_frame=start+29,context_frames=29,
                valid_scored_samples=(frames-29)*1920,latents=np.zeros((1,64,frames)),reference16k=np.zeros((1,1,frames*640)))
        crops=[crop('short',71),crop('speech0',93,10),crop('speech0',100),crop('laugh0',93),crop('whistle0',93),crop('quiet0',93),crop('encoded_silence',150)]
        info={'short':{'condition':'speech'},'speech0':{'condition':'speech'},'laugh0':{'condition':'Laughter'},
              'whistle0':{'condition':'human_whistling_source_description'},'quiet0':{'condition':'Whispering'},'encoded_silence':{'condition':'quiet'}}
        ctx=SimpleNamespace(heldout=crops,probe_crops=[],metadata=info)
        selected,record=select_cases(ctx,4)
        self.assertEqual([v[0] for v in selected],['speech','laughter','whistle','quiet_or_low_level'])
        self.assertEqual(selected[0][1].context_start_frame,0)
        self.assertEqual(record['missing_natural_categories'],[])
        self.assertEqual(category(info['encoded_silence'],crops[-1]),'encoded_control')

if __name__=='__main__':unittest.main()
