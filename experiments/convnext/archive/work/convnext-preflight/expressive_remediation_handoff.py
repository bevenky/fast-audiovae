import json,hashlib,sys,shutil
from pathlib import Path
R9=Path('/workspace/fast-audiovae-convnext-20260909-r9');ROOT=R9/'remediation/expressive';sys.path.insert(0,str(R9))
from audiovae_student.acquire_expressive import StorageBudget
from audiovae_student.restart_data import canonical
from audiovae_student.data import load_manifest
inv=json.loads((ROOT/'candidate-window-inventory.json').read_text());ready=json.loads((ROOT/'ready.json').read_text());budget=StorageBudget(ROOT,400*1024**2,4*1024**3)
for n,expected in inv['files_sha256'].items():
 if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=expected:raise ValueError('Published v2 changed')
rows=load_manifest(ROOT/'train-candidates-v2.jsonl');dev=load_manifest(ROOT/'fresh-dev-v2.jsonl')
R=E=0
for i,line in enumerate((R9/'data/recipe-v2/optimization/windows.jsonl').open()):
 w=json.loads(line)
 if i>=3000*32:
  n=w['valid_input_samples16k'];R+=n
  if w['condition'] not in ['speech','jnv_nonverbal','jvnv_verbal_and_nonverbal']:E+=n
remaining=R/16000/3600; events=E/16000/3600
explicit=sum(v['scored_hours'] for k,v in inv['condition_totals'].items() if k!='emotional_nonverbal')
# Appending once-only new data while preserving all unconsumed old data is one
# illustrative feasible recipe; root owns actual global budget and schedule.
addition=max(0,(.05*remaining-events)/.95)
generic=max(0,addition-explicit)
recipe={'reference_checkpoint_step':3000,'reference_batch_size':32,'remaining_original_scored_hours':remaining,'remaining_original_explicit_event_hours':events,'target_future_broad_expressive_fraction':.05,'new_explicit_scored_hours_available':explicit,'generic_scored_hours_needed_if_all_original_remaining_preserved':generic,'total_additional_hours_needed':addition,'available_generic_scored_hours':inv['condition_totals']['emotional_nonverbal']['scored_hours'],'feasible':generic<=inv['condition_totals']['emotional_nonverbal']['scored_hours'],'schedule_scope':'Arithmetic capacity only, not a training launch plan. Current trainer globalstep budget, batch composition, adaptive optimizer state, exact prior exposure ledger and teacher caches need explicit compatible continuation. Generic nonverbal is reported separately; this is not 5% verified cry/whistle/action coverage.'}
files={name:{'path':str(ROOT/name),'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in ['train-candidates-v2.jsonl','fresh-dev-v2.jsonl','candidate-windows-v2.jsonl.gz','candidate-window-source-ids.json','candidate-window-inventory.json','emogator-conditions.json','fresh-conditions.json','new-dev-contributor-reservation.json']}
body={'publication_version':2,'state':'verified_staged_not_trained','accepted_files':files,'supersedes':inv['supersedes_manifest_publication'],'validation':'Every accepted canonical manifest loads and round-trips; windows are verified no-repeat and disjoint against complete original optimization/calibration sources and all historical/new heldouts. Original source/prepared SHA and full audio decode validation receipts remain under the stage. No active plan changed.','verified_train_clips':len(rows),'verified_new_dev_clips':len(dev),'scored_window_inventory':{k:inv[k] for k in ['windows','sources','scored_hours','whole_source_hours','condition_totals','context_frames','minimum_valid_input_samples','scored_frames_maximum','window_encoding']},'fresh_source_conditions':ready['fresh_fsd'],'rejected_duplicate_original':ready['fresh_quarantined'],'feasible_continuation_capacity':recipe,'remaining_action_gaps':['Crying_and_sobbing','Giggle','Shout','human_whistling'],'actual_stage_bytes_before_receipt':budget.used,'free_bytes':shutil.disk_usage(ROOT).free,'no_teacher_or_training_changes':True}
budget.write(ROOT/'ready-v2.json',canonical(body));print(json.dumps(body,indent=2))
