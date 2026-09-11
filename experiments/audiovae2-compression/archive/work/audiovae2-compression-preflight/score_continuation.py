"""Repeat the same read-only quiet decomposition at the planned final milestone."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
import settings_screen as screen
import score_reference_mel as scoring

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
directory = root/'current-lowrate-continue1000-v1'
out = root/'continuation-quiet-addendum-v1.json'
assert not out.exists()
base = screen.base
base.policy()
args = SimpleNamespace(base_out=root/'pilot',manifest=root/'pilot-selection-v1.json')
metadata, selection, initial, preflight, manifest, pools = screen.authenticate_inputs(args)
receipt = scoring.read_json(directory/'completed.json')
checkpoint = directory/'checkpoint-step1000.pt'
assert receipt['step']==1000 and receipt['fit_cursor']==3000 and receipt['frozen_state_preserved']
assert base.sha(checkpoint)==receipt['checkpoint_sha256']
payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
assert payload['step']==1000 and payload['fit_cursor']==3000
assert payload['identity']==scoring.read_json(root/'settings-screen-v1/current_lr3e-5/launch.json')
assert payload['sources_seen']==[c['source_id'] for c in pools['fit'][:3000]]
assets = Path('/workspace/fast-audiovae-convnext-20260908-r1/assets')
assert base.sha(assets/'audio_vae_v2.py')==base.SOURCE_SHA256
assert base.sha(assets/'audiovae.pth')==base.CHECKPOINT_SHA256
teacher = base.FrozenAudioVAE2.from_files(assets/'audio_vae_v2.py',assets/'audiovae.pth',device='cuda')
model = base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
model.load_group_state_dict(payload['group'])
frozen = screen.frozen_versions(model)
group_sha = screen.group_digest(model)
with torch.no_grad():
    result = scoring.evaluate_reference(model,teacher,pools['development'],scoring.AuthorMelLoss().cuda(),base.objective())
result['common_reproduction'] = scoring.verify_common(result,scoring.read_json(directory/'development-step1000.json'))
assert frozen==screen.frozen_versions(model) and group_sha==screen.group_digest(model)
assert base.sha(checkpoint)==receipt['checkpoint_sha256']
result.update({'step':1000,'arm':'current_lr3e-5','checkpoint_sha256':receipt['checkpoint_sha256'],
               'script_sha256':base.sha(__file__),'scorer_sha256':base.sha(scoring.__file__),
               'original_preflight':metadata,'complete':True,'training_updates':0})
base.write_json(out,result)
print(json.dumps({'complete':True,'spectral':result['aggregate'],
                  'quiet':result['quiet_diagnostics']['student']['aggregate'],
                  'teacher_replay':result['quiet_diagnostics']['teacher_replay']['aggregate']}))
