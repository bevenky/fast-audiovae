"""Pure data/CPU checks for immutable progressive source extension."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import progressive_extended_data as data
from fresh_training_data import GEOMETRY, PAIR_VERSION, digest, sha


def row(index, prefix="new"):
    sid = f"{prefix}-{index}"
    return {"source_id":sid, "manifest_row":{"source_id":sid,"audio_sha256":digest("audio/"+sid),
        "parent_recording_id":"parent/"+sid,"split":"train"}, "start_frame":0,
        "context_start_frame":0,"context_frames":0,"scored_frames":3,
        "valid_scored_samples":4096,"input_samples16k":6400}


class Original:
    def __init__(self, root):
        self.fit = [{"source_id":f"fit-{i}"} for i in range(3000)]
        rows = [row(i,"old") for i in range(27000)]
        manifest = root/"manifest.json";manifest.write_text('{}\n')
        blocked = {key:set() for key in data.KEYS}
        for i in range(3000):
            source = row(i,"fit")["manifest_row"]
            for key in data.KEYS:blocked[key].add(source[key])
        for key in data.KEYS:blocked[key].add(row(0,"heldout")["manifest_row"][key])
        plan = {"rows":rows,"source_ids":[r["source_id"] for r in rows],
            "original_manifest_sha256":sha(manifest),"blocked_identities":{k:sorted(v) for k,v in blocked.items()},
            "teacher_source_sha256":data.SOURCE_SHA256,"teacher_checkpoint_sha256":data.CHECKPOINT_SHA256}
        plan["identity_sha256"] = digest(plan)
        path=root/"parent-plan.json";path.write_text(json.dumps(plan))
        self.fresh=SimpleNamespace(plan=plan, source_ids=tuple(plan["source_ids"]), identity=plan["identity_sha256"],
            plan_path=path,_original_manifest=manifest)
        self.source_ids=tuple(r["source_id"] for r in self.fit)+self.fresh.source_ids
        self.reads=[];self.checks=0
        self.hashes={path:sha(path),manifest:sha(manifest)}

    def take(self,start,count):
        self.reads.append((start,count))
        return [{"source_id":sid} for sid in self.source_ids[start:start+count]]

    def assert_unchanged(self):
        self.checks+=1
        if any(sha(p)!=value for p,value in self.hashes.items()):raise RuntimeError("Original changed")


def extension(original):
    rows=[row(i) for i in range(30000)]
    result={"version":data.VERSION,"parent_plan_sha256":sha(original.fresh.plan_path),
        "parent_plan_identity_sha256":original.fresh.identity,
        "original_manifest_sha256":sha(original.fresh._original_manifest),
        "starting_checkpoint_sha256":data.PARENT_CHECKPOINT_SHA256,
        "starting_optimizer_step":2000,"target_optimizer_step":5000,
        "appended_start_index":30000,"appended_rows":rows,
        "source_ids_sha256":digest(list(original.source_ids)+[r["source_id"] for r in rows]),
        "authorized_source_interval":[24000,60000],"teacher_source_sha256":data.SOURCE_SHA256,
        "teacher_checkpoint_sha256":data.CHECKPOINT_SHA256}
    result["identity_sha256"]=digest(result)
    return result


def seal(plan):
    plan.pop("identity_sha256",None);plan["identity_sha256"]=digest(plan)
    return plan


@pytest.fixture(scope="module")
def metadata(tmp_path_factory):
    root=tmp_path_factory.mktemp("progressive-extension")
    original=Original(root)
    return original,extension(original)


def provider(tmp_path, metadata, plan=None):
    original, sealed=metadata
    path=tmp_path/"extension.json";path.write_text(json.dumps(sealed if plan is None else plan))
    return data.ExtendedProgressiveData(original,path,tmp_path/"shards")


def test_identity_and_global_prefix_are_preserved(tmp_path,metadata):
    original,plan=metadata;before=copy.deepcopy(original.fresh.plan)
    p=provider(tmp_path,metadata)
    assert p.original is original and p.fresh is original.fresh
    assert p.identity==p.extension.identity==plan["identity_sha256"]
    assert p.source_ids[:30000]==original.source_ids and len(p.source_ids)==60000
    assert p.extension.source_ids==tuple(f"new-{i}" for i in range(30000))
    assert len(p.extension.plan["rows"])==30000
    assert original.fresh.plan==before
    assert data._ExtensionShards._load is data.FreshTrainingData._load
    p.assert_unchanged()


def test_boundary_dispatch_keeps_global_and_extension_local_indices(tmp_path,metadata,monkeypatch):
    p=provider(tmp_path,metadata);calls=[]
    def read(start,count):
        calls.append((start,count));return [{"source_id":sid} for sid in p.extension.source_ids[start:start+count]]
    monkeypatch.setattr(p.extension,"take",read)
    assert [r["source_id"] for r in p.take(29994,12)]==list(p.source_ids[29994:30006])
    assert p.original.reads[-1]==(29994,6) and calls==[(0,6)]
    assert [r["source_id"] for r in p.take(24000,12)]==list(p.source_ids[24000:24012])
    assert [r["source_id"] for r in p.take(59988,12)]==list(p.source_ids[59988:60000])
    assert calls[-1]==(29988,12)
    monkeypatch.setattr(p.extension,"take",lambda a,b:[{"source_id":"wrong"}]*b)
    with pytest.raises(RuntimeError):p.take(30000,12)


@pytest.mark.parametrize("start,count",[(-1,1),(0,0),(59999,2),(True,1),(24000,True),(24000.0,12)])
def test_reject_invalid_global_intervals(tmp_path,metadata,start,count):
    p=provider(tmp_path,metadata)
    with pytest.raises(ValueError):p.take(start,count)


@pytest.mark.parametrize("key",data.KEYS)
@pytest.mark.parametrize("origin",["old","fit","heldout","new"])
def test_reject_each_duplicate_or_reserved_identity(metadata,key,origin):
    original,template=metadata;p=copy.deepcopy(template)
    forbidden=row(0,origin)["manifest_row"][key]
    p["appended_rows"][1]["manifest_row"][key]=forbidden
    if key=="source_id":p["appended_rows"][1]["source_id"]=forbidden
    seal(p)
    with pytest.raises(ValueError,match="identity"):
        data.validate_extension(original,p)


@pytest.mark.parametrize("key,value",[
    ("parent_plan_sha256","changed"),("parent_plan_identity_sha256","changed"),
    ("original_manifest_sha256","changed"),("starting_checkpoint_sha256","changed"),
    ("starting_optimizer_step",1000),("target_optimizer_step",6000),
    ("appended_start_index",27000),("authorized_source_interval",[21000,57000]),
    ("teacher_source_sha256","changed"),("teacher_checkpoint_sha256","changed"),
    ("source_ids_sha256","changed"),
])
def test_reject_resealed_wrong_contract(metadata,key,value):
    original,template=metadata;p=copy.deepcopy(template);p[key]=value;seal(p)
    with pytest.raises(ValueError):data.validate_extension(original,p)


def test_reject_unsealed_edit_short_plan_and_wrong_geometry(metadata):
    original,template=metadata
    p=copy.deepcopy(template);p["target_optimizer_step"]=5001
    with pytest.raises(ValueError):data.validate_extension(original,p)
    p=copy.deepcopy(template);p["appended_rows"].pop();seal(p)
    with pytest.raises(ValueError):data.validate_extension(original,p)
    p=copy.deepcopy(template);p["appended_rows"][0]["context_frames"]=1;seal(p)
    with pytest.raises(ValueError):data.validate_extension(original,p)


def write_shard(p):
    directory=p.extension.shards_dir/"000000-000300";directory.mkdir(parents=True)
    crops=[]
    for r in p.extension.plan["rows"][:300]:
        crop={k:r[k] for k in GEOMETRY}
        crop.update(latents=torch.zeros(1,64,3),teacher_audio=torch.zeros(1,1,5760),
                    reference16k=torch.zeros(1,1,1920),cache_key=digest(r["source_id"]))
        crops.append(crop)
    path=directory/"pairs.pt"
    torch.save({"format":PAIR_VERSION,"plan_identity_sha256":p.identity,"start_index":0,"stop_index":300,"crops":crops},path)
    ids=list(p.extension.source_ids[:300])
    receipt={"format":PAIR_VERSION,"complete":True,"plan_identity_sha256":p.identity,
        "start_index":0,"stop_index":300,"source_ids":ids,"source_ids_sha256":digest(ids),
        "teacher_source_sha256":data.SOURCE_SHA256,"teacher_checkpoint_sha256":data.CHECKPOINT_SHA256,
        "parameter_updates":0,"pairs_path":str(path),"pairs_sha256":sha(path)}
    (directory/"receipt.json").write_text(json.dumps(receipt))
    return path,directory/"receipt.json"


def test_sealed_reader_and_changed_target_bytes(tmp_path,metadata):
    p=provider(tmp_path,metadata)
    with pytest.raises(FileNotFoundError):p.take(30000,12)
    path,_=write_shard(p)
    crops=p.take(30000,12)
    assert [r["source_id"] for r in crops]==list(p.source_ids[30000:30012])
    assert all(r["latents"].shape==(1,64,3) for r in crops)
    p.assert_unchanged()
    with path.open("ab") as f:f.write(b"changed")
    with pytest.raises(RuntimeError):p.assert_unchanged()


def test_old_plan_receipt_cannot_be_relabelled_as_extension(tmp_path,metadata):
    p=provider(tmp_path,metadata);_,path=write_shard(p)
    receipt=json.loads(path.read_text());receipt["plan_identity_sha256"]=p.fresh.identity
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="identity"):p.take(30000,12)


def test_plan_mutation_detected_after_construction(tmp_path,metadata):
    p=provider(tmp_path,metadata)
    with p.plan_path.open("a") as f:f.write(" ")
    with pytest.raises(RuntimeError):p.assert_unchanged()
