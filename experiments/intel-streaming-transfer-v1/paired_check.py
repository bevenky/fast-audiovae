"""Two resident CPU processes; alternate completed short prefix decodes."""
import json,subprocess,sys,select,time
from pathlib import Path
BASE=Path('/var/tmp/intel-streaming-transfer-v1')
UIDS=('bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994')
processes={};report={'protocol':'Three 1.6-second prefixes, one warmup, three alternating repetitions, separate resident processes; CPU0, one thread, ORT1.29, Session.run only','warmups':[],'rows':[]}
output=Path(sys.argv[1]);packet=int(sys.argv[2])
def recv(p):
 if not select.select([p.stdout],[],[],90)[0]:raise TimeoutError('worker stalled')
 line=p.stdout.readline()
 if not line:raise RuntimeError('worker stopped '+str(p.poll()))
 return json.loads(line)
def send(p,msg):
 p.stdin.write(json.dumps(msg)+'\n');p.stdin.flush();return recv(p)
def save():output.write_text(json.dumps(report,indent=2)+'\n')
try:
 for name in ('baseline','combined'):
  args=[sys.executable,str(BASE/'short_check.py'),'--worker','--output',str(output.with_suffix('.'+name+'.unused'))]
  if name=='combined':args+=['--model','/dev/shm/intel-streaming-transfer-v1-graphs-r2/combined.onnx','--library',str(BASE/'build/libintel_rawhistory.so'),'--library',str(BASE/'build/libintel_phase.so')]
  p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1);processes[name]=p;r=recv(p);assert r['ready'];report[name+'_setup']=r
 for uid in UIDS:
  for name,p in processes.items():r=send(p,dict(uid=uid,packet_ms=packet));r['model']=name;report['warmups'].append(r)
 for rep in range(3):
  for i,uid in enumerate(UIDS):
   order=('baseline','combined') if (rep*3+i)%2==0 else ('combined','baseline');pair={}
   for name in order:pair[name]=send(processes[name],dict(uid=uid,packet_ms=packet))
   a,b=pair['baseline'],pair['combined'];assert a['audio']==b['audio'] and a['state']==b['state'],'Numerical regression'
   report['rows'].append(dict(uid=uid,rep=rep,order=order,**pair));save()
 report['rtf']={name:sum(r[name]['seconds'] for r in report['rows'])/sum(r[name]['duration'] for r in report['rows']) for name in processes}
 report['time_reduction_percent']=100*(1-report['rtf']['combined']/report['rtf']['baseline']);report['all_waveform_and_state_exact']=True;report['packet_ms']=packet;report['complete']=True
finally:
 for p in processes.values():
  if p.poll() is None:
   try:p.stdin.write('{"quit":true}\n');p.stdin.flush();p.wait(timeout=15)
   except BaseException:p.terminate();p.wait(timeout=5)
 report['worker_exit_codes']=[p.returncode for p in processes.values()];save()
assert all(p.returncode==0 for p in processes.values())
print(json.dumps({k:report[k] for k in ('rtf','time_reduction_percent','packet_ms','all_waveform_and_state_exact')}))
