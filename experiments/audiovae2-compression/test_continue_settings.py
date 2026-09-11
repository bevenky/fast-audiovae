"""Source-cursor and complete optimizer/RNG restoration checks, CPU only."""
import copy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/"convnext"))
import continue_settings as continuation
from test_run_pilot import TinyGroup


def make_checkpoint():
    model = TinyGroup()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, **continuation.screen.OPTIMIZER)
    for value in (1., 2.):
        optimizer.zero_grad(set_to_none=True)
        ((model.gain*value-.9)**2).backward(); optimizer.step()
    payload = {"group": {key:value.clone() for key,value in model.group_state_dict().items()},
               "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": continuation.screen.rng_state(),
               "step": 2, "fit_cursor": 6, "sources_seen": [str(i) for i in range(6)],
               "identity": {"learning_rate": 3e-5}}
    return model, optimizer, payload


def test_restored_adamw_and_rng_continue_the_uninterrupted_trajectory():
    uninterrupted, original_optimizer, payload = make_checkpoint()
    value = torch.rand(4).sum()
    original_optimizer.zero_grad(set_to_none=True)
    ((uninterrupted.gain*value-.9)**2).backward(); original_optimizer.step()
    resumed = TinyGroup()
    optimizer = continuation.restore_training_state(resumed, payload)
    torch.testing.assert_close(torch.rand(4).sum(), value, rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    ((resumed.gain*value-.9)**2).backward(); optimizer.step()
    torch.testing.assert_close(resumed.gain, uninterrupted.gain, rtol=0, atol=0)
    for key, tensor in original_optimizer.state[uninterrupted.gain].items():
        torch.testing.assert_close(optimizer.state[resumed.gain][key], tensor, rtol=0, atol=0)


def test_source_cursor_requires_the_entire_ordered_prefix():
    fitting = [{"source_id": str(i)} for i in range(3000)]
    payload = {"step":256,"fit_cursor":768,"sources_seen":[str(i) for i in range(768)]}
    assert continuation.validate_source_cursor(payload, fitting) == 768
    assert fitting[continuation.validate_source_cursor(payload, fitting)]["source_id"] == "768"
    for changed in ({**payload,"fit_cursor":767},
                    {**payload,"sources_seen":list(reversed(payload["sources_seen"]))},
                    {**payload,"sources_seen":payload["sources_seen"][:-1]+["0"]}):
        with pytest.raises(ValueError): continuation.validate_source_cursor(changed, fitting)


def test_invalid_moments_are_rejected_before_changing_model_weights():
    _, _, payload = make_checkpoint()
    key = payload["optimizer"]["param_groups"][0]["params"][0]
    payload["optimizer"]["state"][key]["exp_avg"] = torch.zeros(3)
    model = TinyGroup(); initial = model.gain.detach().clone()
    with pytest.raises(ValueError, match="moment"):
        continuation.restore_training_state(model, payload)
    torch.testing.assert_close(model.gain, initial, rtol=0, atol=0)


def test_saved_continuation_retains_resume_state_and_next_source_cursor(tmp_path):
    model, optimizer, payload = make_checkpoint()
    seen = [str(i) for i in range(6)]
    path = tmp_path/"checkpoint-step2.pt"
    continuation.screen.save_checkpoint(path, model, optimizer, 2, payload["identity"], seen)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["step"] == 2 and saved["fit_cursor"] == 6 and saved["sources_seen"] == seen
    assert saved["optimizer"]["state"] and set(saved["rng"]) == {"torch","cuda","python","numpy"}
