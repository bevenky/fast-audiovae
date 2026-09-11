"""Offline natural coverage/grid checks; no checkpoint/model loading."""
import hashlib,json,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from natural_history import MAIN_PATH,MAIN_SHA256,HOP,PREFIXES,SCORE_FRAMES,select_cases,raw_metrics,metrics

def crop(name,condition,frames=64,valid_frames=None,context=0,origin=0,teacher_rms=.1):
    valid_frames=frames if valid_frames is None else valid_frames
    c=SimpleNamespace(source_id=name,start_frame=origin+context,context_start_frame=origin,context_frames=context,
        valid_scored_samples=(valid_frames-context)*HOP,latents=np.zeros((1,64,frames)),
        reference16k=np.zeros((1,1,frames*640)),teacher_audio=np.full((1,1,frames*HOP),teacher_rms))
    return c,{'condition':condition}

def context(items):return SimpleNamespace(heldout=[c for c,_ in items],probe_crops=[],metadata={c.source_id:m for c,m in items})

class NaturalHistoryTests(unittest.TestCase):
    def test_ordinary64_frames_cover_all_four_natural_groups(self):
        items=[crop('speech','speech'),crop('laugh','Laughter'),crop('whistle','human_whistling_source_description'),crop('quiet','Whispering',teacher_rms=.0001),crop('encoded_zero','quiet',frames=150,teacher_rms=0.)]
        selected,report=select_cases(context(items),8)
        self.assertEqual(len(selected),4)
        self.assertEqual(report['missing_requested_categories'],[])
        self.assertEqual(report['excluded_crops']['synthetic'],1)
        self.assertEqual(report['score_frames'],16)
        self.assertFalse(report['synthetic_replacement_allowed'])

    def test56_real_frames_suffice_and_padded_latents_do_not(self):
        items=[crop('exact','speech',frames=56),crop('too_short','speech',frames=64,valid_frames=55),crop('synthetic_probe','speech',frames=150)]
        selected,report=select_cases(context(items))
        self.assertEqual([v[1].source_id for v in selected],['exact'])
        _,_,_,frames=selected[0];anchor=frames-SCORE_FRAMES
        self.assertEqual(anchor,40)
        for prefix in PREFIXES:
            self.assertGreaterEqual(anchor-prefix,0)
            self.assertEqual((frames-(anchor-prefix)-prefix)*HOP,16*HOP)
        self.assertEqual(report['excluded_crops']['insufficient_real_history'],1)

    def test_no_synthetic_substitution_or_missing_category_claim(self):
        selected,report=select_cases(context([crop('encoded_laugh','Laughter',frames=150),crop('yell','Yell')]))
        self.assertEqual([v[0] for v in selected],['other_natural'])
        self.assertEqual(set(report['missing_requested_categories']),{'speech','laughter','whistle','quiet_or_low_level'})

    def test_source_diversity_priority_and_bounded_count(self):
        items=[]
        for kind,condition in [('s','speech'),('l','Laughter'),('w','whistling'),('q','Breathing')]:
            items += [crop(f'{kind}{i}',condition,frames=64,teacher_rms=.0002 if kind=='q' else .1) for i in range(3)]
        selected,report=select_cases(context(items))
        self.assertEqual(len(selected),8);self.assertEqual(len({x[1].source_id for x in selected}),8)
        self.assertEqual(set(report['selected_category_counts'].values()),{2})

    def test_quiet_natural_speech_selection_uses_teacher_not_student(self):
        selected,report=select_cases(context([crop('quiet_speech','speech',teacher_rms=.0001),crop('active_speech','speech',teacher_rms=.1)]))
        self.assertEqual(set(v[0] for v in selected),{'speech','quiet_or_low_level'})

    def test_reused_reports_keep_raw_errors_and_valid_json(self):
        target=np.linspace(-.2,.2,16*HOP);prediction=target+.01
        report=metrics(prediction,target)
        self.assertAlmostEqual(report['raw_primary']['mse'],.0001)
        self.assertEqual(report['raw_primary']['samples'],16*HOP)
        self.assertTrue(report['secondary_lag_gain']['diagnostic_only'])
        json.dumps(report,allow_nan=False)
        self.assertEqual(hashlib.sha256(MAIN_PATH.read_bytes()).hexdigest(),MAIN_SHA256)

if __name__=='__main__':unittest.main()
