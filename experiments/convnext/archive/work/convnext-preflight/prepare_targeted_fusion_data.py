"""Seal matched regular/targeted debug data without inference or training.

Only existing, reviewed prepared audio is read. Source-label plus acoustic activity
is never represented as timestamp-level semantic ground truth. Scored intervals
are unique within each arm including its D warmup and fixed-weight calibration.
The user's debugging exception permits parent-exposed explicit event intervals;
those intervals are tagged, not represented as unseen training data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys

BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9')
RATE=16000
D16=3040
HOP=640
FULL=40960


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def key(w):return (w.source_id,w.start_frame)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(data);tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=BASE/'remediation/corrected-screen/targeted-data.json')
    parser.add_argument('--generator-steps',type=int,default=400)
    parser.add_argument('--warmup-steps',type=int,default=64)
    parser.add_argument('--calibration-batches',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=91032)
    args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite sealed selection')
    sys.path.insert(0,str(BASE/'remediation/fusion-code'))
    from audiovae_student.continuation_data import load_continuation_plan,_check_windows,_IntervalIndex
    from audiovae_student.comparison_data import EVENTS,INDIC22,known_identities
    from audiovae_student.data import validate_manifest
    from audiovae_student.restart_data import digest
    import numpy as np
    import soundfile as sf
    receipt_path=BASE/'remediation/fusion-screen/parent-receipt.json'
    receipt=read(receipt_path);plan=load_continuation_plan(receipt['plan_path'])
    require(receipt['step']==8090 and receipt['cursor']==70080,'Wrong experiment parent')
    require(plan['identity']['identity_sha256']==receipt['plan_identity_sha256'],'Plan identity changed')
    rows={r.source_id:r for r in plan['rows']};counts=plan['counts'];cursor=receipt['cursor']
    oldkeys={key(w) for w in plan['windows'][:cursor]}
    explicit=set(EVENTS)
    # Existing canonical plan already excludes all heldout identities and prior
    # normalization-calibration intervals. Parent-prefix reuse is explicit only.
    events=[w for w in plan['windows'] if w.condition in explicit]
    future=[w for w in plan['windows'][cursor:] if w.valid_input_samples16k==FULL]
    source_verified={};stat_cache={};chosen=set()
    reviewed_path=BASE/'remediation/expressive/fsd-reviewed-selection.json'
    reviewed={str(v['source_id']):v for v in read(reviewed_path)['selected']}
    fresh_path=BASE/'remediation/expressive/fresh-conditions.json';fresh=read(fresh_path)
    labels={w.source_id:sorted(set(fresh.get(w.source_id,[])+[w.condition])) for w in events}
    for row in plan['reserved']:
        if row.source_id in fresh:labels[row.source_id]=fresh[row.source_id]
    # Find source conditions from inherited reviewed labels when absent in fresh.
    for row in plan['reserved']:
        if row.source_id in reviewed:labels[row.source_id]=reviewed[row.source_id]['target_labels']

    @lru_cache(maxsize=16)
    def audio(sid):
        row=rows[sid];raw=Path(row.audio_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==row.audio_sha256,'Source hash changed: '+sid)
        a,sr=sf.read(row.audio_path,dtype='float32',always_2d=True)
        require(sr==RATE and a.shape==(counts[sid],1) and bool(np.isfinite(a).all()),'Bad source PCM: '+sid)
        source_verified[sid]={'audio_sha256':row.audio_sha256,'input_samples16k':len(a),'bytes':len(raw)}
        return a[:,0]

    def metrics(w):
        k=key(w)
        if k in stat_cache:return stat_cache[k]
        a=audio(w.source_id)[w.start_frame*HOP:w.start_frame*HOP+w.valid_input_samples16k]
        starts=np.arange(0,len(a)-D16+1,160,dtype=np.int64)
        if starts[-1]!=len(a)-D16:starts=np.append(starts,len(a)-D16)
        power=np.concatenate(([0.],np.cumsum(a.astype('float64')**2)))
        drms=np.sqrt((power[starts+D16]-power[starts])/D16)
        frame=a[:len(a)//320*320].reshape(-1,320)
        rms=np.sqrt(np.mean(frame.astype('float64')**2,axis=1))
        longest=run=0
        for yes in rms<=.001:
            run=run+1 if yes else 0;longest=max(longest,run)
        # A transition D view contains >=60 ms genuinely low-level and >=60 ms
        # active source audio; direction is kept, with no semantic assumptions.
        transitions=[]
        for off in range(960,len(a)-D16+961,160):
            lo=max(0,off-960);hi=lo+D16
            if hi>len(a):continue
            before=float(np.sqrt(np.mean(a[lo:lo+960].astype('float64')**2)))
            after=float(np.sqrt(np.mean(a[hi-960:hi].astype('float64')**2)))
            if min(before,after)<=.001 and max(before,after)>=.005:
                transitions.append((max(before,after)/(min(before,after)+1e-7),lo,before,after))
        transition=max(transitions,default=None)
        v={'input_rms':float(np.sqrt(np.mean(a.astype('float64')**2))),
           'input_peak':float(np.max(np.abs(a))),
           'complete_20ms_windows':len(rms),'quiet_20ms_windows':int(np.count_nonzero(rms<=.001)),
           'quiet_fraction':float(np.mean(rms<=.001)),'longest_quiet_seconds':longest*.02,
           'active_190ms_start16':int(starts[np.argmax(drms)]),'active_190ms_rms':float(np.max(drms)),
           'quiet_190ms_start16':int(starts[np.argmin(drms)]),'quiet_190ms_rms':float(np.min(drms)),
           'transition':{'start16':int(transition[1]),'before_rms':transition[2],
                         'after_rms':transition[3]} if transition else None}
        stat_cache[k]=v;return v

    def mark(w):
        require(key(w) not in chosen,'Duplicate source window');chosen.add(key(w))

    def entry(w,kind,explicit_d=False):
        v={'window':w.to_dict(),'selection_kind':kind,'prior_parent_exposure':key(w) in oldkeys,
           'discriminator_start_sample48k':None,'source_labels':labels.get(w.source_id,[]),
           'qualification':None}
        if explicit_d:
            m=metrics(w);start=m['active_190ms_start16']
            if kind=='quiet':start=m['quiet_190ms_start16']
            elif kind=='transition':start=m['transition']['start16']
            v['discriminator_start_sample48k']=start*3
            v['qualification']={**m,'semantic_event_verified':False,
                'evidence':'reviewed source-level label plus measured active interval' if kind=='explicit_event'
                    else 'measured input quiet/active energy; not teacher-output qualification',
                'source_label_evidence':{'access_record':rows[w.source_id].access_record,
                    'reviewed_metadata_file':str(reviewed_path) if w.source_id in reviewed else None,
                    'reviewed_source_title':reviewed.get(w.source_id,{}).get('title')}}
            require(0<=start*3<=w.valid_output_samples48k-9120,'Invalid targeted D offset')
        return v

    rng=random.Random(args.seed)
    # Source round-robin within every explicit category prevents a long recording
    # from filling a category before other contributors are represented.
    bycond=defaultdict(lambda:defaultdict(list))
    for w in events:bycond[w.condition][w.source_id].append(w)
    eventqueues={}
    for condition,sources in sorted(bycond.items()):
        ids=sorted(sources);rng.shuffle(ids);out=[]
        for sid in ids:sources[sid].sort(key=lambda w:w.start_frame)
        for ix in range(max(map(len,sources.values()))):
            out.extend(sources[sid][ix] for sid in ids if ix<len(sources[sid]))
        eventqueues[condition]=out
    event_candidates=[]
    while any(eventqueues.values()):
        for condition in sorted(eventqueues):
            if eventqueues[condition]:event_candidates.append(eventqueues[condition].pop(0))
    speech=[w for w in future if w.condition=='speech']
    broad=list(future)
    # Preparatory pools use explicit views as well; all three targeted pools are
    # sampled together first so generator/prep intervals cannot overlap.
    budgets={'targeted_generator':args.generator_steps*args.batch_size,
             'discriminator_warmup':args.warmup_steps*args.batch_size,
             'gradient_calibration':args.calibration_batches*args.batch_size}
    selected_events={};event_cursor=0
    rejected_events=Counter()
    for name,n in budgets.items():
        selected=[];samples=0;goal=math.ceil(n*FULL*.05)
        while samples<goal and event_cursor<len(event_candidates):
            w=event_candidates[event_cursor];event_cursor+=1
            m=metrics(w)
            if m['active_190ms_rms']<.001:
                rejected_events[w.condition]+=1;continue
            mark(w);selected.append(entry(w,'explicit_event',True));samples+=w.valid_input_samples16k
        require(samples>=goal,f'Insufficient explicit activity for {name}: {samples/RATE}s vs {goal/RATE}s')
        selected_events[name]=selected
    # Natural input quiet/transition examples are disjoint full scored windows.
    quiet_candidates=[];transition_candidates=[]
    probe=0
    requested_quiet=sum(math.ceil(n*.025) for n in budgets.values())
    requested_transition=requested_quiet
    for w in speech:
        if key(w) in chosen:continue
        m=metrics(w);probe+=1
        if m['transition'] is not None:transition_candidates.append(w)
        elif m['longest_quiet_seconds']>=1.0 and m['quiet_190ms_rms']<=.001:quiet_candidates.append(w)
        if len(quiet_candidates)>=requested_quiet and len(transition_candidates)>=requested_transition:break
        if probe>=15000:break
    total_edge=requested_quiet+requested_transition
    total_quiet=min(len(quiet_candidates),requested_quiet)
    total_transition=total_edge-total_quiet
    print(json.dumps({'stage':'acoustic_inventory','quiet_candidates':len(quiet_candidates),'transition_candidates':len(transition_candidates),'chosen_quiet':total_quiet,'chosen_transition':total_transition,'scanned':probe}),flush=True)
    require(total_quiet>=16,'Insufficient distinct >=1-second quiet intervals for meaningful screen')
    require(len(transition_candidates)>=total_transition,'Insufficient quiet/transition material at unchanged acoustic thresholds')
    pools={};qcursor=tcursor=0;remaining_quiet=total_quiet;remaining_edge=total_edge
    for name,n in budgets.items():
        selected=selected_events[name]
        edge_count=2*math.ceil(n*.025)
        qn=round(remaining_quiet*edge_count/remaining_edge);tn=edge_count-qn
        remaining_quiet-=qn;remaining_edge-=edge_count
        for kind,candidates,start,count in [('quiet',quiet_candidates,qcursor,qn),('transition',transition_candidates,tcursor,tn)]:
            for w in candidates[start:start+count]:
                mark(w);selected.append(entry(w,kind,True))
        qcursor+=qn;tcursor+=tn
        pools[name]=selected
    # Common ordinary speech is selected after all focused/prep windows are
    # reserved, so it can be exactly shared between regular and targeted arms.
    speech_cursor=0
    for name,n in budgets.items():
        while len(pools[name])<n:
            require(speech_cursor<len(speech),'Insufficient full ordinary speech')
            w=speech[speech_cursor];speech_cursor+=1
            if key(w) in chosen:continue
            mark(w);pools[name].append(entry(w,'ordinary_speech'))
        rng.shuffle(pools[name])
    # Every replacement in regular_generator has exactly the targeted valid
    # scored sample count at the same batch position; ordinary items are shared.
    regular=[];bcursor=0
    for target in pools['targeted_generator']:
        if target['selection_kind']=='ordinary_speech':regular.append(target);continue
        nvalid=target['window']['valid_input_samples16k']
        while bcursor<len(broad):
            w=broad[bcursor];bcursor+=1
            if key(w) not in chosen:break
        else:raise ValueError('Insufficient matched regular replacements')
        w=replace(w,valid_input_samples16k=nvalid,scored_frames=math.ceil(nvalid/HOP))
        mark(w);regular.append(entry(w,'ordinary_replacement'))
    pools['regular_generator']=regular
    # Corpus/provenance checks on the complete union, and per-arm no-replay
    # assertions. Cross-arm reuse is deliberately matched, never counted twice.
    union={}
    for entries in pools.values():
        for e in entries:
            v=e['window']
            from audiovae_student.restart_data import PilotWindow
            w=PilotWindow(**{k:v[k] for k in PilotWindow.__dataclass_fields__})
            union[key(w)]=w
    usedrows={w.source_id:rows[w.source_id] for w in union.values()}
    validate_manifest(usedrows.values(),training_only=True)
    inherited=_IntervalIndex.from_ledger(plan['ledger'])
    _check_windows(union.values(),rows,counts,forbidden=inherited,reserved=plan['reserved'],excluded=plan['excluded'])
    for sid in sorted(usedrows):audio(sid)
    for name in ('regular_generator','targeted_generator'):
        ws=[]
        for pool in (name,'discriminator_warmup','gradient_calibration'):
            for e in pools[pool]:
                v=e['window'];ws.append(PilotWindow(**{k:v[k] for k in PilotWindow.__dataclass_fields__}))
        _check_windows(ws,rows,counts,forbidden=inherited,reserved=plan['reserved'],excluded=plan['excluded'])
    def summary(entries):
        total=sum(e['window']['valid_input_samples16k'] for e in entries)
        bykind=defaultdict(lambda:{'windows':0,'seconds':0.,'sources':set()})
        bycondition=defaultdict(lambda:{'windows':0,'seconds':0.,'sources':set()})
        for e in entries:
            w=e['window']
            for name,d in [(e['selection_kind'],bykind),(w['condition'] or 'unspecified',bycondition)]:
                d[name]['windows']+=1;d[name]['seconds']+=w['valid_input_samples16k']/RATE;d[name]['sources'].add(w['source_id'])
        def compact(d):return {k:{**v,'sources':len(v['sources']),'duration_percent':v['seconds']*RATE/total*100} for k,v in sorted(d.items())}
        language=Counter()
        for e in entries:language[e['window']['language']]+=e['window']['valid_input_samples16k']/RATE
        return {'windows':len(entries),'seconds':total/RATE,'hours':total/RATE/3600,
            'prior_parent_exposed_windows':sum(e['prior_parent_exposure'] for e in entries),
            'prior_parent_exposed_seconds':sum(e['window']['valid_input_samples16k']/RATE for e in entries if e['prior_parent_exposure']),
            'by_selection_kind':compact(bykind),'by_condition':compact(bycondition),'language_seconds':dict(sorted(language.items())),
            'indic_missing':sorted(set(INDIC22)-set(language))}
    report={'format_version':1,'parent_step':receipt['step'],'parent_cursor':cursor,
       'parent_sha256':receipt['checkpoint_sha256'],'parent_receipt_sha256':sha(receipt_path),
       'plan_identity_sha256':receipt['plan_identity_sha256'],'plan_path':receipt['plan_path'],
       'script_sha256':sha(__file__),'seed':args.seed,'batch_size':args.batch_size,
       'generator_steps':args.generator_steps,'warmup_steps':args.warmup_steps,'calibration_batches':args.calibration_batches,
       'discriminator_crop_samples48k':9120,'context_frames':30,
       'policy':{'source_label_is_not_semantic_interval_ground_truth':True,
          'prior_parent_explicit_event_reuse':'Authorized debugging exception; used once within each experiment arm, individually tagged',
          'new_normalization_calibration_data_reuse':False,
          'heldout_or_excluded_identity_reuse':False,
          'shared_causal_context_is_not_rescored':True,
          'regular_vs_targeted_valid_scored_duration_matched_at_every_batch_position':True,
          'ordinary_discriminator_start':None,
          'explicit_discriminator_start_units':'48 kHz samples from beginning of scored interval; no context offset'},
       'pools':pools,'rows':{sid:r.to_dict() for sid,r in sorted(usedrows.items())},
       'counts':{sid:counts[sid] for sid in sorted(usedrows)},'source_verification':source_verified,
       'qualification_rejected_event_windows':dict(rejected_events),
       'candidate_speech_windows_acoustically_scanned':probe,
       'quiet_transition_inventory':{'quiet_candidates':len(quiet_candidates),'transition_candidates':len(transition_candidates),'quiet_selected':total_quiet,'transition_selected':total_transition,'minimum_contiguous_quiet_seconds':1.0,'quiet_rms_max':.001,'transition_active_rms_min':.005,'quota_policy':'Keep 5 percent window allocation; fill unavailable long-quiet quota with qualifying quiet-to-active transitions, no acoustic threshold relaxation'},
       'summary':{name:summary(entries) for name,entries in pools.items()},
       'metadata_hashes':{'fsd-reviewed-selection.json':sha(reviewed_path),'fresh-conditions.json':sha(fresh_path)},
       'addon_validation':[],'addon_validation_note':'No new heldout sources silently created; existing reserved rows require separate label and parent-exposure audit.'}
    require([e['window']['valid_input_samples16k'] for e in regular]==[e['window']['valid_input_samples16k'] for e in pools['targeted_generator']],'Duration mismatch')
    report['identity_sha256']=digest(report)
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic(args.output,report)
    atomic(args.output.with_name('targeted-data-summary.json'),{k:v for k,v in report.items() if k not in ('pools','rows','counts','source_verification')})
    print(json.dumps({'path':str(args.output),'identity_sha256':report['identity_sha256'],'summary':report['summary']},sort_keys=True))

if __name__=='__main__':main()
