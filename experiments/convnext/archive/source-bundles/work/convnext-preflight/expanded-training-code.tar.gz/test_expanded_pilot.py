"""Offline handoff checks. Every acquisition/training subprocess is fake."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PINNED_BOOTSTRAP = "ba999a89e8d87f9da91fd00fa1ddde04398a353a5a1ecc6e23aa36f9bd30605f"


@pytest.fixture
def pipeline(monkeypatch):
    path = Path(__file__).with_name("run_expanded_pilot.py")
    spec = importlib.util.spec_from_file_location("expanded_pilot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "process_matches", lambda _: False)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def option(command, flag):
    return command[command.index(flag) + 1]


@pytest.fixture
def handoff(tmp_path, monkeypatch, pipeline):
    base = tmp_path
    directory = base / "expanded-pilot"
    directory.mkdir()
    config = base / "assembly.json"
    write(config, {
        "format_version": 1,
        "minimum_train_hours": 500,
        "sources": [
            {"name": name, "kind": kind,
             "manifests": [str(base / name / "train.jsonl")],
             "readiness": str(base / name / "ready.json")}
            for name, kind in (("core", "acquire"), ("indic", "indic"),
                               ("expressive", "expressive"))
        ],
        "prior_training_manifests": [str(base / "bootstrap-train.jsonl")],
    })
    write(base / "data-expanded-core/provenance/sources.json",
          {"state": "ready", "fresh_training_hours": 440})
    write(base / "data-expanded-indic/progress.json",
          {"status": "complete", "hours": 44})
    write(base / "data-expressive-originals/prepared/provenance/expressive-complete.json",
          {"state": "complete", "preparation_version": 2, "hours_by": {"split": {"train": 16}}})
    warmup = base / "training-runs/warmup-speech-muon-v1/latest.pt"
    warmup.parent.mkdir(parents=True)
    warmup.write_bytes(b"synthetic checkpoint placeholder; never loaded")
    original_digest = pipeline.digest
    monkeypatch.setattr(pipeline, "digest", lambda path:
                        PINNED_BOOTSTRAP if Path(path) == warmup else original_digest(path))

    h = SimpleNamespace(
        pipeline=pipeline, base=base, directory=directory, config=config,
        warmup=warmup, commands=[], assembly_calls=[], writers=[],
        reports=[{"state": "ready", "verified_train_hours": 500}],
        child_exit=0, child_status="completed", child_step=10000,
    )

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.scalars = []
            h.writers.append(self)

        def add_scalar(self, *args):
            self.scalars.append(args)

        def close(self):
            self.closed = True

    tensorboard = ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = FakeWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)

    def assemble(config_path, output_dir):
        h.assembly_calls.append(Path(config_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("candidate-train.jsonl", "candidate-dev.jsonl",
                     "filler-exclusion.jsonl", "source-manifest.jsonl"):
            (output_dir / name).write_text("{\"fixture\": true}\n")
        report = h.reports[min(len(h.assembly_calls) - 1, len(h.reports) - 1)]
        write(output_dir / "readiness.json", report)
        return report

    assembly = ModuleType("audiovae_student.corpus_assembly")
    assembly.assemble_corpus = assemble
    monkeypatch.setitem(sys.modules, "audiovae_student.corpus_assembly", assembly)

    class FakeProcess:
        def __init__(self, command, **kwargs):
            # Verify a real, open lock FD is passed; do not launch a process.
            assert len(kwargs["pass_fds"]) == 1
            os.fstat(kwargs["pass_fds"][0])
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            assert kwargs["env"]["PYTHONPATH"] == str(base)
            self.command = list(command)
            self.pid = 1000 + len(h.commands)
            self.returncode = None
            self.polls = 0
            h.commands.append((list(command), kwargs))

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            self.returncode = h.child_exit
            if "audiovae_student.source_training" in self.command:
                out = Path(option(self.command, "--output-dir"))
                write(out / "status.json", {"state": h.child_status,
                      "step": h.child_step, "unique_scored_hours_seen": 409.6})
            return self.returncode

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)

    def invoke():
        monkeypatch.setattr(sys, "argv", ["run_expanded_pilot.py", "--base", str(base),
                                         "--assembly-config", str(config)])
        return pipeline.main()

    h.invoke = invoke
    return h


def test_ready_handoff_initializes_verified_bootstrap_once(handoff):
    h = handoff
    assert h.invoke() == 0
    assert len(h.commands) == 1
    command, _ = h.commands[0]
    assert command[1:3] == ["-m", "audiovae_student.source_training"]
    assert option(command, "--initialize-from") == str(h.warmup)
    assert "--resume" not in command
    assert option(command, "--batch-size") == "64"
    assert option(command, "--steps") == "10000"
    assert option(command, "--corpus-audit") == str(h.directory / "corpus/readiness.json")
    assert option(command, "--manifest") == str(h.directory / "corpus/source-manifest.jsonl")
    assert h.pipeline.read_json(h.directory / "status.json")["phase"] == "completed_reconstruction"
    assert h.writers[0].closed


def test_expanded_checkpoint_always_resumes_instead_of_bootstrap(handoff):
    h = handoff
    checkpoint = h.base / "training-runs/warmup-speech-500h-v1/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"expanded exact-resume checkpoint fixture")
    assert h.invoke() == 0
    command, _ = h.commands[0]
    assert option(command, "--resume") == str(checkpoint)
    assert "--initialize-from" not in command


def test_bootstrap_checksum_mismatch_prevents_any_training(handoff, monkeypatch):
    h = handoff
    monkeypatch.setattr(h.pipeline, "digest", lambda _: "wrong")
    with pytest.raises(ValueError, match="verified backup"):
        h.invoke()
    assert not h.commands
    assert h.writers[0].closed


@pytest.mark.parametrize("state", ["pending_sources", "invalid_sources",
                                   "insufficient_coverage", "insufficient_hours"])
def test_nonready_assembly_never_launches_training(handoff, state):
    h = handoff
    h.reports = [{"state": state, "verified_train_hours": 499, "filler_permitted": False}]
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        h.invoke()
    assert not h.commands
    assert h.pipeline.read_json(h.directory / "status.json")["phase"] == "failed"
    assert h.writers[0].closed


def test_dead_unfinished_source_blocks_before_assembly(handoff):
    h = handoff
    write(h.base / "data-expanded-indic/progress.json", {"status": "preparing", "hours": 1})
    with pytest.raises(RuntimeError, match="recorded process is not running"):
        h.invoke()
    assert not h.commands
    assert not h.assembly_calls


def test_new_expressive_source_must_finish_before_handoff(handoff):
    h = handoff
    config = h.pipeline.read_json(h.config)
    config['initial_source_names'] = ['core', 'indic', 'expressive', 'emogator']
    config['sources'].append(dict(name='emogator', kind='expressive', preparation_version=1,
                                 readiness=str(h.base / 'emogator/ready.json'),
                                 launch_record=str(h.base / 'emogator.launch.json')))
    write(h.config, config)
    with pytest.raises(RuntimeError, match='emogator acquisition is not ready'):
        h.invoke()
    assert not h.commands and not h.assembly_calls
    write(h.base / 'emogator/ready.json', dict(state='complete', preparation_version=1,
                                             hours_by={'split': {'train': 15}}))
    assert h.invoke() == 0


def test_preparation_version_must_match_before_handoff(handoff):
    h = handoff
    write(h.base / 'data-expressive-originals/prepared/provenance/expressive-complete.json',
          dict(state='complete', preparation_version=1))
    with pytest.raises(RuntimeError, match='unexpected preparation version'):
        h.invoke()
    assert not h.commands and not h.assembly_calls


def test_failed_source_is_not_treated_as_acquisition_in_progress(handoff):
    h = handoff
    write(h.base / "data-expanded-indic/progress.json", {"status": "failed", "hours": 1})
    with pytest.raises(RuntimeError, match="acquisition reports failed"):
        h.invoke()
    assert not h.commands


def test_bounded_filler_keeps_train_bootstrap_and_evaluation_exclusions(handoff):
    h = handoff
    h.reports = [
        {"state": "insufficient_hours", "verified_train_hours": 490,
         "remaining_train_hours": 10, "filler_permitted": True},
        {"state": "ready", "verified_train_hours": 500.1},
    ]
    assert h.invoke() == 0
    assert len(h.commands) == 2
    fill, _ = h.commands[0]
    assert fill[2] == "audiovae_student.acquire"
    assert option(fill, "--librispeech-other-minutes") == "602"
    assert option(fill, "--exclude-training-manifest") == str(h.directory / "corpus/filler-exclusion.jsonl")
    assert option(fill, "--reserved-manifest") == str(h.directory / "corpus/candidate-dev.jsonl")
    assert option(fill, "--reserved-evaluation-manifest") == str(h.base / "reserved-evaluation.json")
    assert option(fill, "--librispeech-dev-minutes") == "0"
    assert float(option(fill, "--minimum-fresh-hours")) >= 10
    final_config = h.directory / "assembly-with-fill.json"
    assert h.assembly_calls == [h.config, final_config]
    saved = h.pipeline.read_json(final_config)
    assert saved["minimum_train_hours"] == 500
    assert saved["prior_training_manifests"] == [str(h.base / "bootstrap-train.jsonl")]
    assert sum(row["name"] == "fill" for row in saved["sources"]) == 1
    assert all(Path(path).is_absolute() for row in saved["sources"] for path in row["manifests"])
    assert h.commands[1][0][2] == "audiovae_student.source_training"


def test_existing_filler_acquisition_uses_its_resume_contract(handoff):
    h = handoff
    write(h.base / "data-expanded-fill/provenance/acquisition-plan.json", {"fixture": True})
    h.reports = [
        {"state": "insufficient_hours", "verified_train_hours": 490,
         "remaining_train_hours": 10, "filler_permitted": True},
        {"state": "ready", "verified_train_hours": 500},
    ]
    assert h.invoke() == 0
    assert "--resume" in h.commands[0][0]


def test_restart_uses_frozen_with_fill_config_without_reacquisition(handoff):
    h = handoff
    saved = h.directory / "assembly-with-fill.json"
    payload = h.pipeline.read_json(h.config)
    payload["sources"].append({"name": "fill", "kind": "acquire",
                               "manifests": [str(h.base / "data-expanded-fill/train.jsonl")]})
    write(saved, payload)
    assert h.invoke() == 0
    assert h.assembly_calls == [saved]
    assert len(h.commands) == 1
    assert h.commands[0][0][2] == "audiovae_student.source_training"


def test_large_shortfall_stops_instead_of_lowering_gate(handoff):
    h = handoff
    h.reports = [{"state": "insufficient_hours", "verified_train_hours": 479,
                  "remaining_train_hours": 21, "filler_permitted": True}]
    with pytest.raises(RuntimeError, match="20-hour fill bound"):
        h.invoke()
    assert not h.commands


@pytest.mark.parametrize("state,step", [("failed", 10000), ("completed", 9999)])
def test_successful_process_exit_is_not_enough_to_claim_completion(handoff, state, step):
    h = handoff
    h.child_status, h.child_step = state, step
    with pytest.raises(RuntimeError, match="without completing step 10000"):
        h.invoke()
    assert h.pipeline.read_json(h.directory / "status.json")["phase"] == "failed"


def test_nonzero_child_stops_pipeline_and_records_exit(handoff):
    h = handoff
    h.child_exit = 7
    with pytest.raises(RuntimeError, match="exited with status 7"):
        h.invoke()
    assert h.pipeline.read_json(h.directory / "training.launch.json")["returncode"] == 7
    assert h.writers[0].closed


def test_known_active_child_prevents_duplicate_launch(handoff, monkeypatch):
    h = handoff
    write(h.directory / "training.launch.json", {"pid": 42, "command": ["training"]})
    monkeypatch.setattr(h.pipeline, "process_matches", lambda _: True)
    with pytest.raises(RuntimeError, match="do not start a duplicate"):
        h.invoke()
    assert not h.commands


def test_child_inherits_lock_and_is_recorded_before_reporting(tmp_path, pipeline, monkeypatch):
    record = tmp_path / "child.json"
    calls = []
    polls = iter([None, 0])

    def fake_popen(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pid=42, returncode=0, poll=lambda: next(polls))

    def report():
        assert pipeline.read_json(record)["pid"] == 42

    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    with (tmp_path / "pipeline.lock").open("a") as lock:
        pipeline.run_child(["fake-training"], tmp_path, record, tmp_path / "child.log",
                           report, lock_fd=lock.fileno())
        assert calls[0]["pass_fds"] == (lock.fileno(),)
    assert pipeline.read_json(record)["returncode"] == 0


def test_launch_record_write_failure_still_passes_lock_to_live_child(tmp_path, pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda command, **kwargs:
                        calls.append(kwargs) or SimpleNamespace(pid=42))
    monkeypatch.setattr(pipeline, "write_json", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with (tmp_path / "pipeline.lock").open("a") as lock:
        with pytest.raises(OSError, match="disk full"):
            pipeline.run_child(["fake-training"], tmp_path, tmp_path / "child.json",
                               tmp_path / "child.log", lambda: None, lock_fd=lock.fileno())
        assert calls[0]["pass_fds"] == (lock.fileno(),)
