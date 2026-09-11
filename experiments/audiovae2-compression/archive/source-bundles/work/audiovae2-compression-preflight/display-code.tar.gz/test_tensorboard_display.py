"""Safe display tags must preserve history from all concurrent source writers."""
from pathlib import Path
import sys

import pytest
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import tensorboard_display as display


def scalar(tag,value,step,wall):
    return event_pb2.Event(step=step,wall_time=wall,summary=summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag,simple_value=value)]))


def emit(writer,event): writer.add_event(event); writer.flush()


def output_events(directory):
    return [e for p in directory.glob("events.out.tfevents.*") for e in LegacyEventFileLoader(str(p)).Load()]


def test_safe_labels_keep_meaning_without_parentheses():
    assert display.safe_tag("overview/Active waveform correlation (%)") == "overview/Active waveform correlation - %"
    assert display.safe_tag("overview/Signal peak (% full scale, not a quality score)") == "overview/Signal peak - % full scale, not a quality score"


def test_late_older_writer_history_exact_dedup_and_source_read_only(tmp_path):
    source=tmp_path/"source"; output=tmp_path/"display"
    first=EventFileWriter(str(source),filename_suffix=".training")
    emit(first,scalar("overview/Correlation (%)",94.,1000,100.25))
    original_bytes={p:p.read_bytes() for p in source.iterdir()}
    projection=display.EventProjection(source,output)
    second=None
    try:
        assert projection.poll()==1
        assert all(p.read_bytes()==data for p,data in original_bytes.items())
        second=EventFileWriter(str(source),filename_suffix=".live")
        emit(second,scalar("overview/Progress (%)",34.,1700,101.5))
        emit(second,scalar("overview/Correlation (%)",94.,1000,100.25))
        assert projection.poll()==1  # The exact repeated scalar is deduplicated.
        emit(first,scalar("overview/Correlation (%)",96.,1500,102.75))
        source_bytes={p:p.read_bytes() for p in source.iterdir()}
        assert projection.poll()==1 and projection.poll()==0
        assert all(p.read_bytes()==data for p,data in source_bytes.items())
        receipt=projection.seal_ready()
        assert receipt["ready"] and receipt["overview_scalar_values_written"]==3
        observed=[(v.tag,e.step,e.wall_time,v.simple_value) for e in output_events(output)
                  for v in e.summary.value if v.tag.startswith("overview/")]
        assert observed==[("overview/Correlation - %",1000,100.25,94.),
                          ("overview/Progress - %",1700,101.5,34.),
                          ("overview/Correlation - %",1500,102.75,96.)]
        assert all("(" not in row[0] and ")" not in row[0] for row in observed)
    finally:
        projection.close(); first.close()
        if second is not None: second.close()


def test_one_new_layout_and_selected_text_are_retained(tmp_path):
    source=tmp_path/"source"; output=tmp_path/"display"
    writer=EventFileWriter(str(source))
    text=summary_pb2.Summary.Value(tag="Monitor/Live status/text_summary",
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[b"Step1500"]))
    emit(writer,event_pb2.Event(step=1500,wall_time=20.,summary=summary_pb2.Summary(value=[text])))
    emit(writer,scalar("quality/all/nonquiet_cosine_mean",.95,1500,21.))
    emit(writer,scalar("loss/teacher_waveform",.007,1500,22.))
    emit(writer,display.layout_event())  # Original custom configuration must be ignored.
    projection=display.EventProjection(source,output)
    try:
        assert projection.poll()==3
        assert projection.scalar_values_written==0  # Raw metrics cannot establish overview readiness.
        events=output_events(output)
        values=[(e,v) for e in events for v in e.summary.value]
        assert sum(v.tag==metadata.CONFIG_SUMMARY_TAG for _,v in values)==1
        copied=[(e,v) for e,v in values if v.tag==text.tag]
        assert len(copied)==1 and copied[0][0].step==1500 and copied[0][0].wall_time==20.
        assert copied[0][1]==text
        assert {v.tag for _,v in values if v.WhichOneof("value")=="simple_value"} == {
            "quality/all/nonquiet_cosine_mean", "loss/teacher_waveform"}
    finally:
        projection.close(); writer.close()


def test_display_never_overlaps_the_source_directory(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    for target in (source,source/"display",tmp_path):
        with pytest.raises(ValueError): display.validate_paths(source,target)
