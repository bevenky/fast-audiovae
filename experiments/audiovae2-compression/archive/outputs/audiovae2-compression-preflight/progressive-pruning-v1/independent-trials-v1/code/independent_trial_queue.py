"""Run only the sealed, finite GRAIL and quiet trials after the current trial ends.

No model/data values enter the queue status; each child retains its own receipts.
A failed child stops the queue. There is no resume, retry, extension or promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

VERSION='audiovae2_independent_trial_queue_v1'


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def proc_identity(pid):
    path=Path(f'/proc/{pid}/stat')
    try: fields=path.read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:return None
    return {'state':fields[0],'start_ticks':fields[19]}


def active(pid):
    identity=proc_identity(pid)
    return identity is not None and identity['state'] not in ('Z','X')


def verify_completion(path,*,kind,expected_version=None):
    d=json.loads(Path(path).read_text())
    if kind=='initializer':
        passed=(d.get('version')=='audiovae2_grail_hidden_initialization_v1'
            and d.get('variant')=='grail_shared_hidden' and d.get('complete') is True
            and d.get('status')=='evaluated' and d.get('failure') is None
            and d.get('neural_training_updates')==0 and d.get('files_preserved') is True
            and bool(d.get('states_preserved')) and all(d['states_preserved'].values())
            and d.get('teacher_cache_comparisons')==384 and d.get('teacher_cache_all_pass') is True)
        if passed:
            receipt=json.loads((Path(path).parent/'grail-receipt.json').read_text())
            passed=(d.get('candidate_receipt')==receipt
                and sha(Path(path).parent/'grail-native-operators.pt')==receipt.get('operators_sha256'))
    elif kind=='recovery':
        if expected_version not in ('audiovae2_fresh_candidate_recovery_v1','audiovae2_grail_candidate_recovery_v1','audiovae2_quiet_candidate_recovery_v1'):
            raise ValueError('A specific approved recovery version is required')
        passed=(d.get('version')==expected_version
            and d.get('status')=='awaiting_review' and d.get('failure') is None and d.get('step')==2000
            and d.get('source_count')==24000 and d.get('frozen_state_preserved') is True
            and d.get('original_files_preserved') is True and d.get('automatic_next_cut') is False
            and d.get('optimizer_reset_after_start') is False and bool(d.get('last_checkpoint_sha256')))
        if passed:
            receipt=json.loads((Path(path).parent/'checkpoint-step2000.json').read_text())
            passed=(sha(Path(path).parent/'checkpoint-step2000.pt')==d['last_checkpoint_sha256']
                and receipt.get('checkpoint_sha256')==d['last_checkpoint_sha256']
                and receipt.get('step')==2000 and receipt.get('sources_seen')==24000
                and receipt.get('frozen_state_preserved') is True and receipt.get('quality')==d.get('final'))
    else:raise ValueError('Unknown trial kind')
    if not passed:raise RuntimeError('Child completion or preservation checks failed')
    return {'kind':kind,'complete':True,'steps':d.get('step',0),'source_count':d.get('source_count',0),
        'quality':d.get('final') if kind!='initializer' else d['results']['grail']['aggregate'],
        'checkpoint_sha256':d.get('last_checkpoint_sha256'),'completion_sha256':sha(path)}


def authenticate_code(stage):
    for path,expected in stage['protected_code'].items():
        if sha(path)!=expected:raise RuntimeError('Sealed source changed before trial launch')
    if not stage['argv'] or stage['kind'] not in ('initializer','recovery'):
        raise ValueError('Invalid finite trial command')


def resolve_argv(stage,grail_init):
    args=list(stage['argv'])
    if '@GRAIL_ARTIFACT_SHA@' in args or '@GRAIL_STATE_SHA@' in args:
        verify_completion(Path(grail_init)/'completed.json',kind='initializer')
        receipt=json.loads((Path(grail_init)/'grail-receipt.json').read_text())
        artifact=Path(grail_init)/'grail-native-operators.pt'
        if sha(artifact)!=receipt['operators_sha256']:raise RuntimeError('GRAIL artifact changed after initialization')
        replacements={'@GRAIL_ARTIFACT_SHA@':receipt['operators_sha256'],
                      '@GRAIL_STATE_SHA@':receipt['candidate_state_sha256']}
        args=[replacements.get(a,a) for a in args]
    if any(a.startswith('@GRAIL_') for a in args):raise ValueError('Unresolved initializer identity')
    return args


def switch_dashboard(stage,config,state):
    """Only after real child updates; retain every previous event directory."""
    logdir=stage.get('tensorboard')
    if not logdir or not (Path(stage['completion']).parent/'train.jsonl').exists():return False
    initial=json.loads((Path(stage['completion']).parent/'initial-parity.json').read_text())
    if not (initial['quality']['passed'] and initial['rng']['equal'] and initial['optimizer']['equal']
            and initial['candidate_state_exact']):raise RuntimeError('Dashboard candidate has not passed initial parity')
    log=Path(stage['root'])/'tensorboard.log'
    launch_path=Path(stage['root'])/'tensorboard-launch.json'
    if log.exists() or launch_path.exists():raise FileExistsError('Dashboard launch outputs already exist')
    old=state['tensorboard_pid']
    if active(old):
        command=Path(f'/proc/{old}/cmdline').read_bytes().split(b'\0')
        pairs=list(zip(command,command[1:]))
        if (not any(Path(token.decode()).name=='tensorboard' for token in command[:2])
                or (b'--port',b'8888') not in pairs
                or (b'--logdir',state['tensorboard_logdir'].encode()) not in pairs):
            raise RuntimeError('Refusing to replace an unrelated server')
        os.kill(old,signal.SIGTERM)
        for _ in range(50):
            if not active(old):break
            time.sleep(.1)
        if active(old):raise RuntimeError('Previous dashboard did not stop')
    argv=[config['tensorboard_executable'],'--logdir',logdir,'--host','0.0.0.0','--port','8888',
          '--reload_interval','10','--load_fast','false']
    with log.open('xb') as handle:
        p=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(tensorboard_pid=p.pid,tensorboard_logdir=logdir)
    write(launch_path,{'pid':p.pid,'argv':argv,'previous_events_preserved':True})
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);a=parser.parse_args()
    config=json.loads(a.config.read_text());root=a.config.parent
    if (config.get('version')!=VERSION or [s['name'] for s in config['stages']] != ['grail_initialization','grail_recovery','quiet_recovery']
            or [s['kind'] for s in config['stages']] != ['initializer','recovery','recovery']):
        raise ValueError('Only the approved three finite child jobs are allowed')
    lock=root/'queue.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    state={'version':VERSION,'status':'waiting_for_downstream','completed':[],'config_sha256':sha(a.config),
        'tensorboard_pid':config['tensorboard_pid'],'tensorboard_logdir':config['tensorboard_logdir'],
        'queue_pid':os.getpid(),'automatic_extension':False,'automatic_promotion':False}
    write(root/'status.json',state)
    try:
        predecessor=config['predecessor'];pid=predecessor['pid']
        while True:
            identity=proc_identity(pid)
            if identity is None or identity['state'] in ('Z','X'):break
            if identity['start_ticks'] != predecessor['start_ticks']:
                raise RuntimeError('Predecessor process identity changed')
            time.sleep(10)
        state['predecessor']=verify_completion(predecessor['completion'],kind='recovery',expected_version=predecessor['expected_version'])
        for stage in config['stages']:
            authenticate_code(stage);argv=resolve_argv(stage,config['grail_init'])
            stage_root=Path(stage['root']);stage_root.mkdir(parents=True,exist_ok=True)
            if Path(stage['completion']).parent.exists():raise FileExistsError('Child output already exists')
            fd=os.open(stage_root/(stage['name']+'.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
            env=os.environ.copy();env.update(stage['environment'])
            with (stage_root/(stage['name']+'.log')).open('xb') as handle:
                p=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            receipt={'pid':p.pid,'argv':argv,'stage':stage['name'],'started_epoch':time.time(),
                     'protected_code':stage['protected_code'],'config_sha256':state['config_sha256']}
            write(stage_root/(stage['name']+'-launch.json'),receipt)
            state.update(status='running',stage=stage['name'],child_pid=p.pid);write(root/'status.json',state)
            dashboard=False
            while p.poll() is None:
                if not dashboard:
                    try:
                        dashboard=switch_dashboard(stage,config,state)
                    except (OSError,RuntimeError,KeyError,ValueError) as exc:
                        state.setdefault('dashboard_warnings',[]).append({'stage':stage['name'],'error_type':type(exc).__name__})
                        dashboard=True
                    if dashboard:write(root/'status.json',state)
                time.sleep(10)
            if p.returncode!=0:raise RuntimeError('Child process failed; following trials remain pending')
            result=verify_completion(stage['completion'],kind=stage['kind'],expected_version=stage.get('expected_version'));authenticate_code(stage)
            state['completed'].append({'name':stage['name'],**result});write(root/'status.json',state)
        state.update(status='completed',stage=None,child_pid=None);write(root/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',failure_type=type(exc).__name__)
        write(root/'status.json',state)
        raise


if __name__=='__main__':main()
