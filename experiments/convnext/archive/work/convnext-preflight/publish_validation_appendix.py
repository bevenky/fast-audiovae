"""Publish a distinct heldout appendix without altering the original aggregate."""
import hashlib
import argparse
import json
from pathlib import Path
import time
from audiovae_student.monitoring import evaluation_scalars
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

base = Path('/workspace/fast-audiovae-convnext-20260909-r9')
parser = argparse.ArgumentParser()
parser.add_argument('--panel', choices=('events', 'languages'), default='events')
args = parser.parse_args()
folder = base/'remediation'/({'events': 'validation-appendix', 'languages': 'language-validation'}[args.panel])
namespace = {'events': 'appendix', 'languages': 'language_appendix'}[args.panel]
state_file = folder/'dashboard-published.json'
state = json.loads(state_file.read_text()) if state_file.exists() else {}
writer = EventFileWriter(str(base/'remediation/monitoring/tensorboard/decoder-recipe-v2'))
try:
    for path in sorted(folder.glob('evaluation-step*.json')):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in state:
            if state[str(path)] != digest:
                raise ValueError('Previously published appendix evaluation changed')
            continue
        data = json.loads(raw)
        scalars = evaluation_scalars(data)
        values = [Summary.Value(tag=k.replace('quality/', namespace + '/', 1), simple_value=float(v))
                  for k,v in scalars.items() if not k.endswith('reconstruction_and_peak_checks_pass')]
        writer.add_event(Event(wall_time=time.time(), step=data['step'], summary=Summary(value=values)))
        writer.flush()
        state[str(path)] = digest
        temporary = state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
        temporary.replace(state_file)
finally:
    writer.close()
print(json.dumps({'published': state, 'namespace': namespace, 'original_panel_modified': False}))
