"""Sixteen synthetic milestones exercise display history without inference."""
from pathlib import Path
import sys

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2,summary_pb2,tensor_pb2,types_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import tensorboard_display_v2 as display


def events(path):
    return [event for file in path.glob("events.out.tfevents.*") for event in LegacyEventFileLoader(str(file)).Load()]


def emit(writer,tag,value,step,wall):
    writer.add_event(event_pb2.Event(step=step,wall_time=wall,summary=summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag,simple_value=value)])))
    writer.flush()


def test_16_milestones_have_one_common_key_per_unique_safe_metric_run(tmp_path):
    source=tmp_path/"source"; output=tmp_path/"display"
    writer=EventFileWriter(str(source)); projection=None; second=None
    expected={run:[] for run in display.METRIC_RUNS.values()}
    try:
        for index in range(16):
            for metric,run in display.METRIC_RUNS.items():
                # Exact binary-representable values avoid rounding in this oracle.
                emit(writer,"overview/"+metric,index+.25,index*100,1000.+index)
                expected[run].append((index*100,1000.+index,index+.25))
        original={p:p.read_bytes() for p in source.iterdir()}
        projection=display.MetricProjection(source,output)
        assert projection.poll()==16*11
        report=projection.seal_ready()
        assert report["ready"] and len(report["counts_by_run"])==11
        assert all(count==16 for count in report["counts_by_run"].values())
        assert all(p.read_bytes()==data for p,data in original.items())
        for run in display.METRIC_RUNS.values():
            assert "(" not in run and ")" not in run
            values=[(event,v) for event in events(output/run) for v in event.summary.value]
            assert {v.tag for _,v in values}=={display.COMMON_TAG}
            assert [(event.step,event.wall_time,v.simple_value) for event,v in values]==expected[run]
        # Continue polling the original file even after a second writer appears.
        second=EventFileWriter(str(source),filename_suffix=".later")
        metric,run=next(iter(display.METRIC_RUNS.items()))
        emit(second,"overview/"+metric,15.25,1500,1015.)  # Exact duplicate.
        assert projection.poll()==0
        emit(writer,"overview/"+metric,16.25,1600,1016.)
        assert projection.poll()==1
        values=[(e,v) for e in events(output/run) for v in e.summary.value]
        assert [(e.step,e.wall_time,v.simple_value) for e,v in values][-1]==(1600,1016.,16.25)
    finally:
        writer.close()
        if second is not None: second.close()
        if projection is not None: projection.close()


def test_targets_and_limits_are_explicit_and_distinct():
    labels=list(display.METRIC_RUNS.values())
    assert len(labels)==len(set(labels))==11
    assert [label[:2] for label in labels]==[f"{n:02d}" for n in range(1,12)]
    assert labels[0]=="01 Waveform correlation - target 99%"
    assert labels[8]=="09 Peak level - limit 100% full scale"
    assert labels[9]=="10 Whistling level - teacher match 100%"
    assert labels[10]=="11 Training progress - 5000 steps is 100%"
    assert "not an established practical acceptance threshold" in display.GUIDE
    assert "not a quality measurement" in display.GUIDE


def test_raw_metrics_text_and_only_one_layout_live_in_details(tmp_path):
    source=tmp_path/"source"; output=tmp_path/"display"
    writer=EventFileWriter(str(source))
    emit(writer,"quality/all/mel",.5,1500,50.)
    emit(writer,"loss/teacher_waveform",.125,1600,51.)
    text=summary_pb2.Summary.Value(tag="Monitor/Live status/text_summary",
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[b"Current step1600"]))
    writer.add_event(event_pb2.Event(step=1600,wall_time=52.,summary=summary_pb2.Summary(value=[text])))
    writer.flush()
    projection=display.MetricProjection(source,output)
    try:
        assert projection.poll()==3
        values=[(e,v) for e in events(output/"Details") for v in e.summary.value]
        assert sum(v.tag==metadata.CONFIG_SUMMARY_TAG for _,v in values)==1
        assert {v.tag for _,v in values if v.WhichOneof("value")=="simple_value"}=={"quality/all/mel","loss/teacher_waveform"}
        copied=[(e,v) for e,v in values if v.tag==text.tag]
        assert len(copied)==1 and copied[0][0].step==1600 and copied[0][0].wall_time==52.
        assert copied[0][1]==text
        assert all(count==0 for count in projection.counts.values())
    finally:
        projection.close(); writer.close()
