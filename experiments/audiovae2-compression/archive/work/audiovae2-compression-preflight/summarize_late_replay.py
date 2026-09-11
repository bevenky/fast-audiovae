"""Read-only CPU quantification of the completed bounded replay discrepancy."""
import argparse,collections,json,math,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

def exact_tree(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if type(a) is not type(b):return False
 if isinstance(a,dict):return set(a)==set(b) and all(exact_tree(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
 return a==b

def tensor_errors(a,b):
 diff2=0.;base2=0.;count=0;maximum=0.;rows=[]
 for key in a:
  aa,bb=a[key].detach().double(),b[key].detach().double()
  error=aa-bb;ds=float(error.square().sum());bs=float(bb.square().sum());m=float(error.abs().max()) if error.numel() else 0.
  rows.append({'name':key,'max_abs':m,'l2':math.sqrt(ds),'reference_l2':math.sqrt(bs),'relative_l2':math.sqrt(ds/bs) if bs else None,'exact':torch.equal(aa,bb)})
  diff2+=ds;base2+=bs;count+=aa.numel();maximum=max(maximum,m)
 return {'max_abs':maximum,'global_relative_l2':math.sqrt(diff2/base2) if base2 else None,'l2':math.sqrt(diff2),'reference_l2':math.sqrt(base2),'rmse':math.sqrt(diff2/count) if count else None,'elements':count,'tensors':len(rows),'nonexact_tensors':sum(not r['exact'] for r in rows),'worst_tensors_by_relative_l2':sorted(rows,key=lambda r:r['relative_l2'] or 0,reverse=True)[:8],'worst_tensors_by_max_abs':sorted(rows,key=lambda r:r['max_abs'],reverse=True)[:8]}

def effective(group):
 result={}
 for key,value in group.items():
  if key.endswith('weight_v'):
   g=group[key[:-1]+'g'].double();v=value.double();norm=v.square().sum(tuple(range(1,v.ndim)),keepdim=True).sqrt()
   if bool((norm==0).any()):raise ValueError('Zero WN direction prevents effective-weight audit')
   result[key[:-1]+'effective']=v*g/norm
  elif not key.endswith('weight_g'):result[key]=value
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
 complete=json.loads((args.replay/'completed.json').read_text())
 a=torch.load(args.replay/'replayed-step5000.pt',map_location='cpu',weights_only=True,mmap=True);b=torch.load(args.original/'checkpoint-step5000.pt',map_location='cpu',weights_only=True,mmap=True)
 replay=[json.loads(l) for l in (args.replay/'replay.jsonl').read_text().splitlines()];original={r['step']:r for r in [json.loads(l) for l in (args.original/'train.jsonl').read_text().splitlines()]}
 branch={}
 for key in ('total','waveform','mel','feature','gradient_norm'):
  diffs=[]
  for row in replay:
   if key in original[row['step']]:
    av=row['values'][key];bv=original[row['step']][key];diffs.append({'step':row['step'],'absolute_difference':abs(av-bv),'relative_difference':abs(av-bv)/abs(bv) if bv else None,'actual':av,'expected':bv})
  branch[key]={'steps_checked':len(diffs),'exact_steps':sum(r['absolute_difference']==0 for r in diffs),'mean_abs_difference':sum(r['absolute_difference'] for r in diffs)/len(diffs),'maximum':max(diffs,key=lambda r:r['absolute_difference'])}
 states={}
 for moment in ('exp_avg','exp_avg_sq','step'):
  states[moment]=tensor_errors({str(k):v[moment] for k,v in a['optimizer']['state'].items()},{str(k):v[moment] for k,v in b['optimizer']['state'].items()})
 cases={}
 for snap in complete['snapshots']:
  for row in snap['cases']:cases.setdefault(row['source_id'],[]).append({'step':snap['step'],**{k:row[k] for k in ('mae','active_rms_gain','active_projection_gain','active_cosine')}})
 changes=[]
 for sid,rows in cases.items():
  biggest=max(zip(rows,rows[1:]),key=lambda pair:pair[1]['active_rms_gain']-pair[0]['active_rms_gain'])
  changes.append({'source_id':sid,'initial':rows[0],'final':rows[-1],'maximum_gain':max(rows,key=lambda r:r['active_rms_gain']),'minimum_gain':min(rows,key=lambda r:r['active_rms_gain']),'largest25update_gain_rise':{'from_step':biggest[0]['step'],'to_step':biggest[1]['step'],'gain_change':biggest[1]['active_rms_gain']-biggest[0]['active_rms_gain']}})
 positive=[r for r in replay if r['direction']['gradient_dot_actual_update']>0]
 result={'rng_components_exact':{k:exact_tree(a['rng'][k],b['rng'][k]) for k in a['rng']},'source_ledger_exact':exact_tree(a['sources_seen'],b['sources_seen']),'optimizer_parameter_groups_exact':exact_tree(a['optimizer']['param_groups'],b['optimizer']['param_groups']),'execution_identity_exact':exact_tree(a['identity'],b['identity']) and exact_tree(a['resume_identity'],b['resume_identity']),'original_exact_replay_flag':complete['exact_replay'],'exact_replay_not_relaxed':True,'all500_updates_completed':len(replay)==500,'group_raw_parameter_error':tensor_errors(a['group'],b['group']),'group_effective_weight_and_other_parameter_error':tensor_errors(effective(a['group']),effective(b['group'])),'optimizer_moment_errors':states,'original_loss_discrepancies':branch,'case_reference_checks_existing_tolerances':complete['case_reference_checks'],'case_trajectory_summaries':changes,'case_trajectories':cases,'actual_update_direction':{'nonnegative_descent_dot_count':len(positive),'total_steps':len(replay),'positive_dot_steps':[r['step'] for r in positive],'note':'Positive gradient dot actual AdamW update is first-order uphill for that current minibatch; optimizer momentum need not descend every minibatch.'},'teacher_cache_checks':{k:complete[k] for k in ('teacher_cache_sources_checked','all_teacher_cache_bitwise','all_teacher_cache_original_tolerance','teacher_cache_max_abs')},'limits':['These are quantifications of a non-bitwise-exact replay, not acceptance of a relaxed exactness criterion.','Only initial and final diagnostic case scores have saved intermediate-run counterpart reports; new25-step case measurements cannot prove identical historical case values.','Temporal coincidence with a training batch does not isolate that batch as the cause.']}
 args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(args.out),'raw_relative_l2':result['group_raw_parameter_error']['global_relative_l2'],'effective_relative_l2':result['group_effective_weight_and_other_parameter_error']['global_relative_l2'],'raw_max_abs':result['group_raw_parameter_error']['max_abs'],'cases_passed':complete['case_reference_checks']['final']['passed'],'all_teacher_targets_bitwise':complete['all_teacher_cache_bitwise']}))
if __name__=='__main__':main()
