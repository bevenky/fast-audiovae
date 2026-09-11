"""Read-only progress reporting must not fabricate updates or quality values."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_progress as live


def test_partial_trailing_json_and_unterminated_records_do_not_count(tmp_path):
    path = tmp_path/"train.jsonl"
    record = {"step":1500,"unique_sources":4500,"fresh_sources":1500}
    path.write_text(json.dumps(record)+"\n"+'{"step":1501')
    assert live.last_complete_record(path) == record
    path.write_text(json.dumps(record)+"\n"+json.dumps({**record,"step":1501}))
    assert live.last_complete_record(path) == record


def test_malformed_records_are_skipped_without_guessing_a_step(tmp_path):
    path = tmp_path/"train.jsonl"
    path.write_text('broken\n{"step":false,"unique_sources":3}\n{}\n')
    assert live.last_complete_record(path) is None
    assert live.last_complete_record(tmp_path/"missing") is None
    path.write_text('{"step":1500,"unique_sources":4500}\nbroken\n')
    assert live.last_complete_record(path)["step"] == 1500


class Writer:
    def __init__(self): self.scalars=[]; self.text=[]; self.flushes=0
    def add_scalar(self,*args): self.scalars.append(args)
    def add_text(self,*args): self.text.append(args)
    def flush(self): self.flushes += 1


def state():
    return {"step":1500,"target_step":5000,"latest_validation_step":1000,
            "next_validation_step":2000,"prepared_sources":1800,"remaining_to_prepare":10200,
            "fresh_sources_consumed":1500,"waiting":False,"status":"Training","stop":False}


def test_actual_step_percent_is_only_scalar_and_unchanged_state_emits_nothing():
    writer=Writer(); current=state()
    assert live.publish(writer,current)
    assert writer.scalars == [("overview/Training progress to 5000 steps (%)",30.,1500)]
    assert not live.publish(writer,current,current)
    assert writer.flushes == 1
    # Data-readiness changes update status without duplicating the progress point.
    assert live.publish(writer,{**current,"prepared_sources":2100},current)
    assert len(writer.scalars)==1 and len(writer.text)==2
    assert "1000" in writer.text[0][1] and writer.text[0][0] == "Monitor/Live status"


def test_snapshot_reports_ready_data_waiting_and_last_complete_validation(tmp_path):
    (tmp_path/"train.jsonl").write_text(json.dumps({"step":1500,"unique_sources":4500,"fresh_sources":1500})+"\n")
    (tmp_path/"development-step1000.json").write_text('{"aggregate": {}}')
    (tmp_path/"development-step1500.json").write_text('{')
    producer=tmp_path/"producer-progress.json"
    producer.write_text(json.dumps({"sealed_source_prefix":1500,"approved_stop_index":12000}))
    current=live.snapshot(tmp_path,producer,5000)
    assert current["step"]==1500 and current["latest_validation_step"]==1000
    assert current["next_validation_step"]==1500 and current["waiting"]
    assert current["remaining_to_prepare"]==10500
    producer.write_text(json.dumps({"sealed_source_prefix":1800,"approved_stop_index":12000}))
    assert not live.snapshot(tmp_path,producer,5000)["waiting"]


def test_import_does_not_load_any_tensor_or_inference_modules():
    code=("import sys; sys.path.insert(0,"+repr(str(HERE))+"); import live_progress; "
          "assert 'torch' not in sys.modules; assert 'run_pilot' not in sys.modules; "
          "assert 'resume_settings' not in sys.modules; assert 'audiovae_student.teacher' not in sys.modules")
    subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)


def test_tensorboard_multifile_reloads_the_older_training_writer_after_sidecar(tmp_path):
    from tensorboard.backend.event_processing import data_ingester, plugin_event_accumulator
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    active = data_ingester._get_event_file_active_filter(SimpleNamespace(
        reload_multifile=True, reload_multifile_inactive_secs=86400))
    accumulator = plugin_event_accumulator.EventAccumulator(str(tmp_path), event_file_active_filter=active)
    training = EventFileWriter(str(tmp_path), filename_suffix=".training")
    def add(writer, tag, value, step):
        writer.add_event(event_pb2.Event(wall_time=time.time(), step=step,
            summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag,simple_value=value)])))
        writer.flush()
    sidecar = None
    try:
        add(training,"quality/all/correlation",.94,1000); accumulator.Reload()
        sidecar = EventFileWriter(str(tmp_path), filename_suffix=".live-progress")
        add(sidecar,"overview/Training progress",34.,1700); accumulator.Reload()
        # A late validation arrives in the older file with a lower global step.
        add(training,"quality/all/correlation",.96,1500); accumulator.Reload()
        assert [e.step for e in accumulator.Tensors("quality/all/correlation")] == [1000,1500]
        assert [e.step for e in accumulator.Tensors("overview/Training progress")] == [1700]
        add(sidecar,"overview/Training progress",36.,1800); accumulator.Reload()
        add(training,"quality/all/correlation",.97,2000); accumulator.Reload()
        assert [e.step for e in accumulator.Tensors("quality/all/correlation")] == [1000,1500,2000]
        assert [e.step for e in accumulator.Tensors("overview/Training progress")] == [1700,1800]
    finally:
        training.close()
        if sidecar is not None: sidecar.close()
