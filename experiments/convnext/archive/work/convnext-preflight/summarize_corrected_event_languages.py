"""Summarize every saved natural event/language group without model execution.

Use after the main corrected-screen verifier. This reads JSON reports only;
no tensor/checkpoint loads, inference, training, downloads or benchmarks occur.
A group is a recorded source condition/language, not a verified event timestamp.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

from summarize_fusion_screen import (
    assert_matching_reports, delta, digest, file_sha, load_json, row_map,
    summarize_rows, synthetic,
)

ROOT=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation/corrected-screen')
ARMS=('regular','targeted','targeted_magnitude','targeted_complex')


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root
    output=args.output or root/'event-language-summary.json'
    identity=load_json(root/'identity.json')
    verified=load_json(root/'summary.json')
    require(not verified['pending'],'Main verifier still reports pending arms')
    require(set(verified['arms'])==set(ARMS),'Main verifier has wrong arm set')
    require(verified['experiment_identity_sha256']==digest(identity),'Main summary is for a different experiment')
    parent_step=identity['parent_step'];end_step=parent_step+identity['generator_updates_per_arm']
    require(parent_step==8090 and identity['generator_updates_per_arm']==400,'Unexpected experiment budget')
    parent_path=root/'regular/before.json';parent=load_json(parent_path)
    require(parent['evaluated_step']==parent_step,'Wrong parent evaluation step')
    maps={'parent':row_map(parent)};reports={'parent':parent}
    hashes={'identity.json':file_sha(root/'identity.json'),'summary.json':file_sha(root/'summary.json'),
            'regular/before.json':file_sha(parent_path)}
    for arm in ARMS:
        after_path=root/arm/'after.json';before_path=root/arm/'before.json'
        after=load_json(after_path);before=load_json(before_path)
        assert_matching_reports(parent,before);assert_matching_reports(parent,after)
        require(before['model_identity']['parameter_state_sha256']==parent['model_identity']['parameter_state_sha256'],
                'Arm does not share original parent: '+arm)
        require(before['evaluated_step']==parent_step and after['evaluated_step']==end_step,
                'Wrong before/after step: '+arm)
        # This row field fingerprints full evaluation provenance, including
        # the migration/discriminator variant. Parameter equality is checked
        # above; retain exact equality for every actual metric and target.
        def metric_rows(report):
            return [{k:v for k,v in row.items() if k!='model_state_sha256'}
                    for row in report['rows']]
        require(all(row['model_state_sha256']==before['model_state_sha256']
                    for row in before['rows']),'Inconsistent row provenance: '+arm)
        require(metric_rows(before)==metric_rows(parent),
                'Initial numerical metrics/targets differ for same original student: '+arm)
        maps[arm]=row_map(after);reports[arm]=after
        hashes[arm+'/before.json']=file_sha(before_path);hashes[arm+'/after.json']=file_sha(after_path)
        # Require this exact current report's overall metrics match the separately
        # verified summary, rather than silently analyzing a later replacement.
        summary=summarize_rows((r for r in after['rows'] if not synthetic(r)),after['quiet_config']['quiet_teacher_rms_max'])
        require(summary==verified['arms'][arm]['after']['natural'],'Main verifier natural summary is stale: '+arm)
    threshold=parent['quiet_config']['quiet_teacher_rms_max']
    buckets={'event':defaultdict(list),'language':defaultdict(list)}
    for key,row in maps['parent'].items():
        if synthetic(row):continue
        metadata=row['metadata']
        event=metadata.get('condition') or metadata.get('event') or 'unverified'
        language=metadata.get('language') or 'unverified'
        buckets['event'][str(event)].append(key)
        buckets['language'][str(language)].append(key)
    groups={}
    for family,items in buckets.items():
        groups[family]={}
        for label,keys in sorted(items.items()):
            absolute={name:summarize_rows([mapping[k] for k in keys],threshold) for name,mapping in maps.items()}
            sources=sorted({key[0] for key in keys})
            entry={'crops':len(keys),'sources':len(sources),'source_ids':sources,
                   'scored_seconds_including_overlapping_crops':sum(maps['parent'][key]['samples'] for key in keys)/48000,
                   'parent':absolute['parent'],'arms':{}}
            for arm in ARMS:
                rows=[maps[arm][k] for k in keys]
                entry['arms'][arm]={'absolute':absolute[arm],
                    'vs_parent':delta(absolute[arm],absolute['parent']),
                    'vs_regular':delta(absolute[arm],absolute['regular']),
                    'overshoot_source_ids':sorted({r['source_id'] for r in rows if r['student_overshoot_samples']>0})}
            groups[family][label]=entry
    result={'format_version':1,'experiment_identity_sha256':digest(identity),'parent_step':parent_step,
        'end_step':end_step,'arms':list(ARMS),'natural_crops':sum(not synthetic(r) for r in parent['rows']),
        'natural_sources':len({r['source_id'] for r in parent['rows'] if not synthetic(r)}),
        'groups':groups,'source_report_sha256':hashes,'script_sha256':file_sha(Path(__file__)),
        'interpretation':{'no_model_or_optimizer_execution':True,'no_automatic_winner_selection':True,
            'event_membership':'Single recorded primary condition/event; source-level labels, not verified temporal event occupancy',
            'language_membership':'Language labels exactly as recorded in the common evaluation metadata',
            'missing_categories':'Unlisted event/language groups have no measured cohort; never interpret absence as zero error',
            'small_cohorts':'One-source groups are individual diagnostics, not population estimates',
            'quiet':'Fixed teacher-derived quiet masks; pooled RMS weights valid quiet samples',
            'correlation':'Only nonquiet crops with a defined waveform cosine enter the reported cosine mean',
            'counts':'Crops can overlap; seconds and overshoot samples may count the same physical event more than once',
            'distinct_sources':'Count source IDs, not independently verified people',
            'comparison':'Targeted sampling and discriminator-view placement changed together; regular is not the old broad-emotion mix',
            'statistics':'Descriptive paired outcomes from one trajectory per arm; no test of perceptual equivalence'}}
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps({'output':str(output),'event_groups':len(groups['event']),
        'language_groups':len(groups['language']),'natural_sources':result['natural_sources'],
        'source_summary_sha256':hashes['summary.json']},sort_keys=True))

if __name__=='__main__':main()
