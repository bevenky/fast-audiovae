"""Focused CPU contracts for the paired settings experiment."""
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/"convnext"))
import settings_screen as screen
from test_run_pilot import TinyGroup, crop, features


@pytest.fixture
def pipeline(monkeypatch):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    batch = screen.base.batch
    monkeypatch.setattr(screen.base, "batch", lambda crops: batch(crops, device="cpu"))
    observed = []
    def teacher_forward(teacher, z):
        observed.append(z.shape[0])
        return {"group_input": features(z), "group_output": .9*features(z)}
    monkeypatch.setattr(screen.base, "teacher_forward", teacher_forward)
    yield observed
    torch.set_num_threads(old_threads)


@pytest.fixture
def objectives():
    return {"current": screen.base.ReconstructionV2(screen.base.ReconstructionV2Config(fft_sizes=(32,64), mel_bands=(4,8))),
            "author": screen.AuthorMelLoss(screen.AuthorMelConfig(fft_sizes=(32,64), mel_bands=(4,8)))}


def samples():
    return [crop("a",frames=3,valid=4101,amplitude=.08),
            crop("b",frames=5,context=1,valid=7103,amplitude=.2),
            crop("c",frames=4,valid=7019,amplitude=.13)]


def test_import_preserves_gpu_visibility_in_a_fresh_process():
    script = "import os,sys; import settings_screen; assert os.environ['CUDA_VISIBLE_DEVICES']=='7'; assert 'export_preflight' not in sys.modules; assert 'streaming_preflight' not in sys.modules"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="7", PYTHONPATH=os.pathsep.join((str(HERE), str(HERE.parent/"convnext"))))
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_author_keeps_shared_wave_and_feature_objectives_and_centered_counts(pipeline,objectives):
    model = TinyGroup(); crops = samples()
    z,t,valid,spans = screen.base.batch(crops)
    h = model.group_from_input(features(z)); p = model.suffix_from_group(h); ht = .9*features(z)
    current = screen.losses(p,t,h,ht,valid,spans,"current",objectives["current"])
    author = screen.losses(p,t,h,ht,valid,spans,"author",objectives["author"])
    for key in ("waveform","feature"):
        torch.testing.assert_close(author[key],current[key],rtol=2e-6,atol=1e-7)
        g1, = torch.autograd.grad(author[key],model.gain,retain_graph=True)
        g2, = torch.autograd.grad(current[key],model.gain,retain_graph=True)
        torch.testing.assert_close(g1,g2,rtol=2e-6,atol=1e-6)
    totals = screen.counts(crops,"author",objectives["author"])
    assert totals["mel_elements"] == tuple(sum((c["valid_scored_samples"]//(size//4)+1)*bands for c in crops)
        for size,bands in zip((32,64),(4,8)))
    parts = [screen.forward_losses(model,SimpleNamespace(),[c],"author",objectives["author"],totals) for c in crops]
    for key in author:
        torch.testing.assert_close(sum(part[key] for part in parts),author[key],rtol=2e-6,atol=1e-7)


def test_author_singleton_accumulation_matches_pooled_adamw_update(pipeline,objectives):
    actual,reference = TinyGroup(),TinyGroup(); crops=samples()
    coeff={"waveform":1.,"mel":.02,"feature":.5}
    left=torch.optim.AdamW(actual.parameters(),lr=3e-5,**screen.OPTIMIZER)
    right=torch.optim.AdamW(reference.parameters(),lr=3e-5,**screen.OPTIMIZER)
    for optimizer,model in ((left,actual),(right,reference)):
        optimizer.state[model.gain]={"step":torch.tensor(7.),"exp_avg":torch.tensor(.02),"exp_avg_sq":torch.tensor(.03)}
    z,t,valid,spans=screen.base.batch(crops)
    h=reference.group_from_input(features(z))
    values=screen.losses(reference.suffix_from_group(h),t,h,.9*features(z),valid,spans,"author",objectives["author"])
    total=sum(coeff[key]*value for key,value in values.items());total.backward();right.step()
    record=screen.training_update(actual,SimpleNamespace(),crops,"author",objectives["author"],coeff,left)
    assert pipeline==[1,1,1]
    assert record["total"]==pytest.approx(float(total.detach()),rel=2e-6)
    torch.testing.assert_close(actual.gain.grad,reference.gain.grad,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(actual.gain,reference.gain,rtol=0,atol=1e-7)
    for key,value in right.state[reference.gain].items():
        torch.testing.assert_close(left.state[actual.gain][key],value,rtol=2e-6,atol=1e-7)


def test_each_arm_restores_identical_weights_rng_and_empty_optimizer():
    model=TinyGroup(); initial={key:value.clone() for key,value in model.group_state_dict().items()}
    rng=screen.rng_state(); expected=screen.group_digest(model)
    draws=[]
    for _,_,rate in screen.ARMS:
        with torch.no_grad():model.gain.add_(.7)
        optimizer=screen.start_arm(model,initial,rng,rate)
        assert screen.group_digest(model)==expected and not optimizer.state
        draws.append((torch.rand(3),np.random.rand(3),random.random()))
    for draw in draws[1:]:
        assert torch.equal(draw[0],draws[0][0]) and np.array_equal(draw[1],draws[0][1]) and draw[2]==draws[0][2]
    screen.restore_rng(rng)


def test_author_calibration_restores_weights_gradients_and_rng_on_error(pipeline,objectives,tmp_path,monkeypatch):
    model=TinyGroup(); crops=samples()
    current=screen.base.calibration(model,SimpleNamespace(),crops,objectives["current"],tmp_path)
    initial=screen.group_digest(model); gain=model.gain.detach().clone(); model.gain.grad=torch.tensor(.321)
    rng=screen.rng_state()
    original=screen.forward_losses
    def fail_after_update(*args,**kwargs):
        if not torch.equal(model.gain.detach(),gain):
            torch.rand(5);random.random();np.random.rand()
            raise RuntimeError("injected disposable error")
        return original(*args,**kwargs)
    monkeypatch.setattr(screen,"forward_losses",fail_after_update)
    with pytest.raises(RuntimeError,match="injected"):
        screen.calibrate_author(model,SimpleNamespace(),crops,objectives["author"],current)
    assert screen.group_digest(model)==initial
    torch.testing.assert_close(model.gain.grad,torch.tensor(.321),rtol=0,atol=0)
    restored=screen.rng_state()
    assert torch.equal(restored["torch"],rng["torch"]) and restored["numpy"]==rng["numpy"] and restored["python"]==rng["python"]
