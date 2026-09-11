import unittest
from types import SimpleNamespace
import numpy as np
from repair_natural_history import choose,inventory,HOP,PREFIXES,SCORE

def crop(sid,condition,frames=64,valid=None,rms=.1,origin=0):
    valid=frames if valid is None else valid
    c=SimpleNamespace(source_id=sid,start_frame=origin,context_start_frame=origin,context_frames=0,
        valid_scored_samples=valid*HOP,latents=np.zeros((1,64,frames)),reference16k=None,
        teacher_audio=np.full((1,1,frames*HOP),rms,dtype=np.float32))
    return c,{'condition':condition}

def ctx(items):return SimpleNamespace(heldout=[c for c,_ in items],probe_crops=[],metadata={c.source_id:m for c,m in items})
def four():return [crop('speech','speech'),crop('laugh','Laughter'),crop('whistle','human_whistling_source_description'),crop('quiet','Breathing',rms=.0001)]

class RepairTests(unittest.TestCase):
    def test_four_categories_without_any_raw_reference(self):
        selected,report=choose(ctx(four()))
        self.assertEqual(len(selected),4);self.assertFalse(report['reference16k_required_for_history'])
        self.assertEqual(report['missing_categories'],[])
        self.assertTrue(all(v['crop'].reference16k is None for v in selected))
    def test_missing_category_fails_before_models(self):
        with self.assertRaisesRegex(ValueError,'whistle'):choose(ctx(four()[:2]+four()[3:]))
    def test_real_padding_not_counted(self):
        selected,report=choose(ctx(four()+[crop('padding','speech',frames=93,valid=55),crop('encoded_zero','Breathing',frames=150,rms=0)]))
        self.assertEqual(report['excluded']['under56_real_frames'],1)
        self.assertEqual(report['excluded']['synthetic'],1)
        self.assertNotIn('padding',[v['crop'].source_id for v in selected])
        for item in selected:
            for p in PREFIXES:self.assertGreaterEqual(item['frames']-SCORE-p,0)
    def test_quiet_window_has_real_history_and_no_tail_padding(self):
        items=four();c,_=items[3];c.teacher_audio.fill(.1)
        c.teacher_audio[...,40*HOP:56*HOP]=1e-5
        selected,_=choose(ctx(items))
        q=next(v for v in selected if v['category']=='quiet_or_low_level')
        self.assertEqual(q['frames'],56);self.assertLess(q['teacher_selection_rms'],.001)
    def test_no_duplicate_sources_and_at_most8(self):
        items=[]
        for i in range(3):
            items += [crop(f's{i}','speech'),crop(f'l{i}','Laughter'),crop(f'w{i}','whistling'),crop(f'q{i}','Whispering',rms=.0001)]
        selected,report=choose(ctx(items))
        self.assertEqual(len(selected),8);self.assertEqual(len({v['crop'].source_id for v in selected}),8)
        self.assertEqual(set(report['selected_category_counts'].values()),{2})
    def test_short_partial_real_tail_excluded(self):
        items=four();items[0]=crop('speech','speech',frames=56,valid=55)
        with self.assertRaisesRegex(ValueError,'speech'):choose(ctx(items))

if __name__=='__main__':unittest.main()
