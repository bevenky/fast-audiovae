"""CPU tests for authenticated state continuation and unique-source accounting."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/"convnext"))
import resume_settings as resume
from test_continue_settings import make_checkpoint
from test_run_pilot import TinyGroup


def ledger(step=1000, batch_size=3):
    old = [f"old-{i}" for i in range(3000)]
    new = [f"new-{i}" for i in range(27000)]
    cursor = (step-1000)*batch_size
    seen = old+new[:cursor]
    return {"step":step, "fit_cursor":len(seen), "fresh_cursor":cursor, "sources_seen":seen}, old, new


def test_source_cursor_starts_after_all_3000_previous_sources_and_restarts_exactly():
    anchor, old, new = ledger()
    assert resume.validate_source_ledger(anchor, old, new) == 0
    milestone, old, new = ledger(1500)
    assert resume.validate_source_ledger(milestone, old, new) == 1500
    assert new[resume.validate_source_ledger(milestone, old, new)] == "new-1500"
    final, old, new = ledger(5000)
    assert resume.validate_source_ledger(final, old, new) == 12000


@pytest.mark.parametrize("kind", ["reorder", "repeat", "cursor", "step", "original_overlap", "fresh_repeat"])
def test_source_accounting_rejects_repetitions_and_skipped_updates(kind):
    payload, old, new = ledger(1500)
    if kind == "reorder": payload["sources_seen"][3000:3002] = reversed(payload["sources_seen"][3000:3002])
    if kind == "repeat": payload["sources_seen"][-1] = old[0]
    if kind == "cursor": payload["fresh_cursor"] -= 1
    if kind == "step": payload["step"] += 1
    if kind == "original_overlap": new[-1] = old[-1]
    if kind == "fresh_repeat": new[-1] = new[0]
    with pytest.raises(ValueError): resume.validate_source_ledger(payload, old, new)


def test_target_has_a_hard_5000_decision_and_cannot_consume_reserved_sources():
    assert resume.validate_target(1000, 5000, 3) == 12000
    for args in ((1000,10000,3), (1000,5000,4), (1500,1500,3), (1000,1200,3)):
        with pytest.raises(ValueError): resume.validate_target(*args)
    # A changed effective batch uses only its explicit new-run source accounting.
    payload, old, new = ledger(1500, batch_size=2)
    assert resume.validate_source_ledger(payload, old, new, 2) == 1000


def test_missing_shard_waits_without_advancing_or_repeating_the_requested_interval(monkeypatch):
    class Data:
        source_ids = tuple(str(i) for i in range(6))
        calls = []
        def take(self, start, count):
            self.calls.append((start,count))
            if len(self.calls) == 1: raise FileNotFoundError("not sealed")
            return [{"source_id":x} for x in self.source_ids[start:start+count]]
    monkeypatch.setattr(resume.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(resume.base, "event", lambda *args,**kwargs: None)
    data = Data()
    chosen = resume.take_when_ready(data, 3, 3, step=1002, poll_seconds=.001)
    assert [x["source_id"] for x in chosen] == ["3","4","5"]
    assert data.calls == [(3,3),(3,3)]


def test_corrupt_shards_are_not_treated_as_temporary_missing_data():
    class Data:
        source_ids = ["source"]
        def take(self, start, count): raise ValueError("checksum mismatch")
    with pytest.raises(ValueError, match="checksum"):
        resume.take_when_ready(Data(), 0, 1, step=1001)


def test_restored_quality_preserves_exact_counts_and_bounded_float_metrics():
    report = {"mse":.0005, "quiet_windows":2544, "nonquiet_cosine_mean":.944,
              "group_nrmse_floor_applied":False, "optional":None}
    resume.validate_restored_quality(report, report)
    resume.validate_restored_quality({**report,"mse":.0005001}, report)
    for changed in ({**report,"quiet_windows":2543}, {**report,"mse":.00051},
                    {**report,"nonquiet_cosine_mean":float("nan")}, {**report,"optional":0.}):
        with pytest.raises(ValueError): resume.validate_restored_quality(changed, report)


def test_saved_milestone_restores_all_moments_rng_and_next_update_exactly(tmp_path):
    uninterrupted, optimizer, payload = make_checkpoint()
    # The step is represented by the optimizer's saved counter, not reconstructed.
    for state in optimizer.state.values(): state["step"].fill_(1000)
    seen = [f"old-{i}" for i in range(3000)]
    identity = {"learning_rate":3e-5}
    path = tmp_path/"checkpoint-step1000.pt"
    resume.save_checkpoint(path, uninterrupted, optimizer, 1000, identity, seen, {"data":"sealed"})
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["format"] == resume.VERSION and saved["fit_cursor"] == 3000 and saved["fresh_cursor"] == 0
    assert saved["resume_identity"] == {"data":"sealed"} and saved["sources_seen"] == seen
    value = torch.rand(7).sum()
    optimizer.zero_grad(set_to_none=True)
    ((uninterrupted.gain*value-.9)**2).backward(); optimizer.step()
    resumed = TinyGroup()
    restored = resume.continuation.restore_training_state(resumed, saved)
    torch.testing.assert_close(torch.rand(7).sum(), value, rtol=0, atol=0)
    restored.zero_grad(set_to_none=True)
    ((resumed.gain*value-.9)**2).backward(); restored.step()
    torch.testing.assert_close(resumed.gain, uninterrupted.gain, rtol=0, atol=0)
    for key, tensor in optimizer.state[uninterrupted.gain].items():
        torch.testing.assert_close(restored.state[resumed.gain][key], tensor, rtol=0, atol=0)
    torch.testing.assert_close(resumed.frozen_suffix_scale, torch.tensor(1.), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        resume.save_checkpoint(path, resumed, restored, 1001, identity, seen, {"data":"sealed"})


def test_anchor_validation_rejects_changed_recipe_or_false_provenance():
    payload, old, _ = ledger()
    arm = {"arm":"current_lr3e-5", "definition":"current", "learning_rate":3e-5,
           "coefficients":{"waveform":1.,"mel":.002,"feature":.01}, "screen_identity_sha256":"screen"}
    payload.update(format=resume.screen.VERSION, identity=arm)
    receipt = {"step":1000,"fit_cursor":3000,"checkpoint_sha256":resume.ANCHOR_SHA256,
               "arm":arm["arm"],"common_quality":{"mse":.01},"selected_screen_checkpoint_sha256":"original256",
               "optimizer_and_rng_saved":True,"frozen_state_preserved":True}
    launch = {"version":resume.continuation.VERSION, "continuation_sha256":resume.base.sha(resume.continuation.__file__),
              "starting_step":256,"target_step":1000,"starting_source_cursor":768,"manual_arm":arm["arm"],
              "coefficients":arm["coefficients"],"learning_rate":3e-5,"selected_checkpoint_sha256":"original256",
              "screen_identity_sha256":"screen","optimizer_restored":True,"rng_restored":True}
    fitting = [{"source_id":source} for source in old]
    resume.validate_anchor(payload, receipt, receipt, launch, arm, fitting, resume.ANCHOR_SHA256)
    for invalid_launch in ({**launch,"learning_rate":1e-4}, {**launch,"optimizer_restored":False},
                           {**launch,"continuation_sha256":"other"}):
        with pytest.raises(ValueError):
            resume.validate_anchor(payload, receipt, receipt, invalid_launch, arm, fitting, resume.ANCHOR_SHA256)
    with pytest.raises(ValueError):
        resume.validate_anchor(payload, receipt, receipt, launch, arm, fitting, "changed checkpoint")


def test_import_keeps_gpu_visibility_and_avoids_cpu_entry_modules():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="7")
    env["PYTHONPATH"] = os.pathsep.join((str(HERE), str(HERE.parent/"convnext")))
    code = ("import os,sys; import resume_settings; "
            "assert os.environ['CUDA_VISIBLE_DEVICES']=='7'; "
            "assert 'export_preflight' not in sys.modules; "
            "assert 'streaming_preflight' not in sys.modules")
    subprocess.run([sys.executable,"-c",code], env=env, check=True, capture_output=True, text=True)
