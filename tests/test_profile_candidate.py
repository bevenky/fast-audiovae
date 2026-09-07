"""Offline event-parser tests, no imports of ORT or model execution."""
import copy
import unittest
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'cpu_profile_candidate', Path(__file__).resolve().parents[1] / 'benchmarks/profile_candidate.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize_profile = _module.summarize_profile
profile_frame_count = _module.profile_frame_count

def events():
    out=[{'cat':'Session','name':'session_initialization','ph':'X','ts':0,'dur':100}]
    for i in range(5):
        start=1000+i*1000
        out.append({'cat':'Session','name':'model_run','ph':'X','ts':start,'dur':800})
        for name,op,offset,duration in [('first','MatMul',10,100+i),('second','Add',150,20+i)]:
            out.append({'cat':'Node','name':name+'_kernel_time','ph':'X','ts':start+offset,'dur':duration,
                        'pid':12,'tid':3,'args':{'provider':'CPUExecutionProvider','op_name':op,
                        'input_type_shape':[{'float':[1,64,170]}],'output_type_shape':[{'float':[1,64,170]}]}})
        out.append({'cat':'Node','name':'first_fence_before','ph':'X','ts':start,'dur':0,'args':{}})
    return out

class ProfileParserTests(unittest.TestCase):
    def test_explicit_causal_audio_prefix(self):
        self.assertEqual(profile_frame_count(255,None,'audio',True),255)
        self.assertEqual(profile_frame_count(255,170,'audio',True),170)
        self.assertEqual(profile_frame_count(255,255,'other',False),255)
        for requested,kind,causal in [(170,'audio',False),(170,'mimi',True),(256,'audio',True),(0,'audio',True),(-1,'audio',True),(True,'audio',True)]:
            with self.assertRaises(ValueError):profile_frame_count(255,requested,kind,causal)
    def test_only_last_three_runs_and_kernels(self):
        out=summarize_profile(events())
        self.assertEqual(out['selected_kernel_counts'],[2,2,2])
        self.assertEqual(len(out['excluded_warmup_intervals']),2)
        self.assertEqual(len(out['selected_kernel_events']),6)
        mm=out['operators'][0]
        self.assertEqual(mm['op_name'],'MatMul');self.assertEqual(mm['run_duration_us'],[102.,103.,104.])
        self.assertEqual(mm['mean_us_per_model_call'],103.)
    def test_original_arguments_and_shape_variants_preserved(self):
        data=events();data[-3]['args']['input_type_shape']=[{'float':[1,32,340]}]
        out=summarize_profile(data)
        self.assertEqual(len(out['nodes']),3)
        self.assertTrue(any(v['input_type_shape']==[{'float':[1,32,340]}] for v in out['nodes']))
        self.assertTrue(all('args' in v for v in out['selected_kernel_events']))
    def test_missing_run_rejected(self):
        with self.assertRaises(ValueError):summarize_profile([v for v in events() if v.get('ts')!=5000])
    def test_missing_selected_kernels_rejected(self):
        data=[v for v in events() if not (v['cat']=='Node' and 4000<v['ts']<4800)]
        with self.assertRaises(ValueError):summarize_profile(data)
    def test_non_cpu_kernel_rejected_even_during_warmup(self):
        for provider in ('CUDAExecutionProvider',None):
            data=events();data[2]['args']['provider']=provider
            with self.assertRaises(ValueError):summarize_profile(data)
    def test_boundary_crossing_kernel_rejected(self):
        data=events();data.append({'cat':'Node','name':'bad_kernel_time','ph':'X','ts':3799,'dur':2,
                    'args':{'provider':'CPUExecutionProvider','op_name':'Add'}})
        with self.assertRaises(ValueError):summarize_profile(data)
    def test_overlapping_model_intervals_rejected(self):
        data=events();next(v for v in data if v.get('name')=='model_run')['dur']=1100
        with self.assertRaises(ValueError):summarize_profile(data)
    def test_bad_durations_and_missing_op_rejected(self):
        for duration in (-1,float('nan'),'10'):
            data=events();data[2]['dur']=duration
            with self.assertRaises(ValueError):summarize_profile(data)
        data=events();next(v for v in data if v.get('name')=='first_kernel_time' and v['ts']==3010)['args'].pop('op_name')
        with self.assertRaises(ValueError):summarize_profile(data)
    def test_events_do_not_mutate(self):
        data=events();before=copy.deepcopy(data);summarize_profile(data);self.assertEqual(data,before)

if __name__=='__main__':unittest.main()
