from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
base=Path('/workspace/fast-audiovae-convnext-20260908-r1')
record_path=base/'corpus-indic-44h.launch.json'
audit_path=base/'indic-workers6-migration.json'
history=base/'launch-history';history.mkdir(exist_ok=True)
def write(path,obj):
 part=path.with_suffix(path.suffix+'.tmp');part.write_text(json.dumps(obj,indent=2)+'\n');os.replace(part,path)
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
def canonical(values):return hashlib.sha256(json.dumps(sorted(values,key=lambda v:v['source_id']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def process_command(pid):
 p=Path(f'/proc/{pid}/cmdline')
 return p.read_bytes().rstrip(b'\0').decode().split('\0') if p.exists() else None
def stopped(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return 'absent'
 status=p.joinpath('status').read_text()
 if '\nState:\tZ' in '\n'+status and len(list(p.joinpath('task').iterdir()))==1:
  return 'zombie_without_workers'
 return None
mode=sys.argv[1]
if mode=='pause':
 if audit_path.exists():raise SystemExit('Migration audit already exists; inspect before repeating')
 record=json.loads(record_path.read_text());pid=record['pid'];command=record['command']
 if pid!=884375 or process_command(pid)!=command:raise SystemExit('Original recorded process no longer matches')
 data=base/'data-expanded-indic'
 before=data/'train.jsonl';before_rows=rows(before);progress=json.loads((data/'progress.json').read_text())
 archived=history/f'corpus-indic-44h.{pid}.launch.json'
 archived.write_bytes(record_path.read_bytes())
 snapshot=history/f'corpus-indic-44h.{pid}.before-workers6.jsonl';snapshot.write_bytes(before.read_bytes())
 complete={k:v for k,v in progress['hours_by_language'].items() if v>=2}
 audit={'state':'pausing','previous_pid':pid,'old_launch_record':str(archived),'before_manifest':str(snapshot),'before_rows':len(before_rows),'before_metadata_sha256':canonical(before_rows),'completed_language_hours_before':complete,'started_at':time.time()}
 write(audit_path,audit)
 if process_command(pid)!=command:raise SystemExit('Process changed before signal')
 os.kill(pid,signal.SIGTERM)
 print(json.dumps(audit,indent=2))
elif mode=='status':
 audit=json.loads(audit_path.read_text());pid=audit['previous_pid'];end=time.monotonic()+45
 while time.monotonic()<end and not stopped(pid):time.sleep(1)
 print(json.dumps({'old_process':stopped(pid) or 'still_running','progress':json.loads((base/'data-expanded-indic/progress.json').read_text())['status']}))
elif mode=='resume':
 audit=json.loads(audit_path.read_text());old=json.loads(Path(audit['old_launch_record']).read_text());pid=old['pid']
 if audit['state']!='pausing' or not stopped(pid):raise SystemExit('Old process still has live workers or migration already resumed')
 if json.loads(record_path.read_text())['pid']!=pid:raise SystemExit('Launch record changed externally')
 final_rows=rows(base/'data-expanded-indic/train.jsonl');by_id={x['source_id']:x for x in final_rows}
 before_rows=rows(Path(audit['before_manifest']))
 if len(by_id)!=len(final_rows) or any(by_id.get(x['source_id'])!=x for x in before_rows):raise SystemExit('Original source rows changed or disappeared')
 progress=json.loads((base/'data-expanded-indic/progress.json').read_text())
 if progress['status']!='paused':raise SystemExit('Original process did not publish a clean paused snapshot')
 for lang,hours in audit['completed_language_hours_before'].items():
  if progress['hours_by_language'][lang]!=hours:raise SystemExit('Completed language hours changed')
 command=list(old['command']);index=command.index('--workers')
 if command[index+1]!='3':raise SystemExit('Expected original workers3')
 command[index+1]='6'
 env=os.environ.copy();env['PYTHONPATH']=str(base);env['PYTHONDONTWRITEBYTECODE']='1'
 log=base/'data-expanded-indic/run-workers6.log'
 with log.open('ab',buffering=0) as stream:
  proc=subprocess.Popen(command,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.2);actual=process_command(proc.pid)
 if actual!=command:raise RuntimeError('New process arguments differ')
 s=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()
 btime=next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
 record={'pid':proc.pid,'command':actual,'argv':actual,'workers':6,'started_at':btime+int(s[19])/os.sysconf('SC_CLK_TCK'),'recorded_at':time.time(),'record_source':'actual /proc argv and process start ticks','cwd':str(base),'log':str(log),'previous_pid':pid,'migration_audit':str(audit_path)}
 write(record_path,record)
 audit.update(state='resumed',new_pid=proc.pid,new_workers=6,old_process_state=stopped(pid),paused_rows=len(final_rows),all_previous_rows_preserved=True,paused_metadata_sha256=canonical(final_rows),completed_language_hours_after={k:progress['hours_by_language'][k] for k in audit['completed_language_hours_before']},resumed_at=time.time())
 write(audit_path,audit)
 print(json.dumps(audit,indent=2))
else:raise SystemExit('Expected pause,status,resume')
