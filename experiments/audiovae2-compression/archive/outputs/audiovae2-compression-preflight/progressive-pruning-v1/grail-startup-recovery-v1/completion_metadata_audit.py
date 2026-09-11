import os,json,pathlib,hashlib,math,datetime,torch
torch.set_num_threads(1)
root=pathlib.Path('/dev/shm/fast-audiovae-grail-startup-recovery-20260911-v1'); out=root/'recovery2000'
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
d=read(out/'completed.json'); launch=read(out/'launch.json')
rows=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines() if s]; ids=[sid for r in rows for sid in r['source_ids']]
checks={}
checks['complete']=d.get('complete') is True and d.get('status')=='awaiting_review' and d.get('failure_category') is None
checks['scope']=d.get('version')=='audiovae2_grail_startup_recovery_v1' and d.get('method')=='corrected' and d.get('updates')==d.get('neural_training_target')==2000
checks['preservation_receipts']=d.get('all_preservation_checks_passed') is True and all(d['preserved'].values())
checks['fresh']=d.get('pilot_state_resumed') is False and d.get('automatic_promotion') is False and d.get('initial_parity_passed') is True and launch['installation'].get('old_group_state_loaded') is False and launch['installation'].get('fresh_adam_matches_original') is True
checks['pinned_identity']=launch['runner_source_sha256']=='a1d863ab15423b0c32a954a8bf691302a914fcd568956d096082d704116177c9' and launch['policy_source_sha256']=='9b4a39383b01c81e2d73bfb8b2a90fb054a042326845c63796a85a76014b491d' and launch['initializer_artifact_sha256']=='d2438662f1d0ffe540bf4e11742eb845758784efed69659c849e31a70778714d'
checks['identity_agreement']=all(d[k]==launch[k] for k in ('initial_state_sha256','initializer_artifact_sha256','config_sha256','initial_rng_sha256','source_plan_identity_sha256','runner_source_sha256','policy_source_sha256','installation'))
checks['protected_files_rehashed']=all(pathlib.Path(p).is_file() and sha(pathlib.Path(p))==h for p,h in launch['protected'].items())
checks['source_ledger']=len(ids)==len(set(ids))==24000 and ids==launch['source_ids'] and digest(ids)==launch['source_ids_sha256']==d['ordinary_source_prefix_sha256'] and d['ordinary_unique_sources']==24000
checks['journal_sequence']=len(rows)==2000 and all(r['step']==i and len(r['source_ids'])==12 for i,r in enumerate(rows,1))
samples=sum(c['samples'] for r in rows for c in r['teacher_cache_checks'])
checks['teacher_cache']=all(len(r['teacher_cache_checks'])==12 and all(c['allclose_original_tolerance'] for c in r['teacher_cache_checks']) for r in rows)
checks['sample_ledger']=samples==d['scored_samples']
flags=('q_caps_passed','q_normal_budget_passed','q_full_primal_verified','q_kkt_passed','q_normal_full_primal_verified','q_normal_kkt_passed')
checks['constraints']=all(r.get('q_constraints')==12 and r.get('startup_anchor_after_passed')==6 and all(r.get(k)==1 for k in flags) for r in rows)
checks['movement']=all(r['q_zero_displacement']==0 and r['q_accepted_displacement_norm']>0 for r in rows)
checks['finite_losses']=all(all(math.isfinite(r[k]) for k in ('total','waveform','mel','feature')) for r in rows)
snapshots={}; geometry=None; parameter_ids=None;teacher_pins=None
coeff={'waveform':1.,'mel':0.0006674012905982311,'feature':0.009304078923434964}
for step in (0,1000,2000):
 p=out/f'checkpoint-step{step}.pt'; receipt=read(p.with_suffix('.json')); h=sha(p)
 sc={'hash_receipt':h==receipt['checkpoint_sha256'],'bytes_receipt':p.stat().st_size==receipt['bytes'],'step_receipt':receipt['step']==step and receipt['source_count']==step*12,'optimizer_and_rng_saved':receipt.get('optimizer_and_rng_saved') is True and receipt.get('optimizer_reset_after_start') is False}
 if not all(sc.values()):raise RuntimeError('Checkpoint receipt authentication failed')
 ck=torch.load(p,map_location='cpu',weights_only=True); opt=ck['optimizer']; states=opt['state']; groups=opt['param_groups']; pids=[i for g in groups for i in g['params']]
 sc['step_fields']=ck['format']==d['version'] and ck['step']==ck['cut_updates']==ck['global_updates']==step
 sc['group_tensors']=len(ck['group'])==90 and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in ck['group'].values())
 geo={k:(tuple(v.shape),str(v.dtype)) for k,v in ck['group'].items()}
 if geometry is None:geometry=geo;parameter_ids=pids;teacher_pins=(ck['teacher_source_sha256'],ck['teacher_checkpoint_sha256'])
 sc['geometry_stable']=geo==geometry and pids==parameter_ids and len(pids)==len(set(pids))==90
 sc['identity_and_ledger']=ck['identity']==launch and ck['sources_seen']==ids[:step*12] and ck['selection']==launch['selection'] and (ck['teacher_source_sha256'],ck['teacher_checkpoint_sha256'])==teacher_pins
 sc['optimizer_scope']=not states if step==0 else len(states)==90 and set(states)==set(pids)
 sc['optimizer_counters_finite']=all(float(v['step'])==step and all(bool(torch.isfinite(v[k]).all()) for k in ('exp_avg','exp_avg_sq')) for v in states.values())
 sc['optimizer_recipe']=all(g['lr']==3e-5 and tuple(g['betas'])==(.9,.99) and g['eps']==1e-8 and g['weight_decay']==0 for g in groups) and ck['coefficients']==coeff and ck['accumulation']==12
 sc['no_reset_or_promotion']=all(ck[k] is False for k in ('optimizer_reset_after_start','automatic_next_cut','automatic_promotion'))
 sc['rng_present']=all(k in ck['rng'] for k in ('torch','cuda','python','numpy'))
 sc['scored_samples']=ck['scored_samples']==sum(c['samples'] for r in rows[:step] for c in r['teacher_cache_checks'])
 sc['quality_receipt']=receipt['quality']==read(out/f'development-step{step}.json')['aggregate']
 if step==2000:sc['final_hash_matches']=h==d['last_checkpoint_sha256']
 snapshots[str(step)]={'sha256':h,'bytes':p.stat().st_size,'group_tensors':len(ck['group']),'optimizer_states':len(states),'checks':sc,'all_pass':all(sc.values())}
 del ck
checks['snapshots']=all(v['all_pass'] for v in snapshots.values())
checks['review_receipts']=all(read(out/f'review-step{step}.json')==d['milestone_quality'][str(step)] for step in (250,500,1000,1500,2000))
checks['fixed_window_identity']=len({v['teacher_window_identity_sha256'] for v in d['milestone_quality'].values()})==1
checks['update_totals']=d['update_totals']['ordinary_adam_updates']==d['update_totals']['constrained_updates']==2000
for k,j in (('normal_accepts','q_normal_accepted'),('normal_solves','q_normal_solves'),('constraint_zero_displacements','q_zero_displacement'),('extended_grid_fallbacks','q_extended_grid_fallback'),('canonical_score_forwards','q_canonical_score_forwards'),('constraint_gradient_forwards','q_constraint_gradient_forwards')):
 checks['total_'+k]=d['update_totals'][k]==sum(r[j] for r in rows)
checks['calibration_count']=d['recurring_calibration_sources']==6 and d['anchor_update_participations']==12000
checks['cpu_only_audit']=os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_pass':all(checks.values()),'checks':checks,'checkpoints':snapshots,'protected_files_rehashed':len(launch['protected']),'source_count':len(ids),'unique_sources':len(set(ids)),'scored_samples':samples,'completion_sha256':sha(out/'completed.json'),'initial_state_sha256':launch['initial_state_sha256'],'update_totals':d['update_totals'],'timing':{k:d[k] for k in ('elapsed_seconds','training_update_seconds','ordinary_update_seconds','constraint_auxiliary_seconds','validation_seconds')}}
(root/'completion-audit-aggregate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

