"""Read-only metric-per-run projection for distinct chart colors and targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.plugins.custom_scalar import layout_pb2, metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tensorboard_display import RAW_LOSS_TAGS, TEXT_PREFIXES, event_key, validate_paths


METRIC_RUNS = {
    "Active waveform correlation (%)": "01 Waveform correlation - target 99%",
    "Correlation target only (99%)": "02 Correlation target - 99%",
    "Waveform MAE reduction vs step 0 (%)": "03 Waveform error reduction - ideal 100%",
    "Mel error reduction vs step 0 (%)": "04 Mel error reduction - ideal 100%",
    "Stage 2-4 output MSE reduction vs step 0 (%)": "05 Stage 2-4 error reduction - ideal 100%",
    "Quiet RMS error reduction vs step 0 (%)": "06 Quiet RMS error reduction - ideal 100%",
    "Quiet windows passing (%)": "07 Quiet windows passing - target 100%",
    "Near-silence windows passing (%)": "08 Near-silence passing - target 100%",
    "Signal peak (% full scale, not a quality score)": "09 Peak level - limit 100% full scale",
    "Whistling level (% of teacher, 100 is matched)": "10 Whistling level - teacher match 100%",
    "Training progress to 5000 steps (%)": "11 Training progress - 5000 steps is 100%",
}
COMMON_TAG = "overview/Percent"
GUIDE = (
    "Each colored run is one metric from the same student and validation panel. "
    "The run labels state the target, ideal or limit. "
    "Waveform correlation has a 99% reconstruction target; this is not perceptual-quality equivalence. "
    "Higher error reduction is better, and 100% means zero error against the teacher. "
    "That is a mathematical ideal, not an established practical acceptance threshold. "
    "Quiet and near-silence pass rates target 100% under the existing checks. "
    "Peak level must stay within 100% full scale; increasing it is not a quality goal. "
    "Whistling level should match 100% of the teacher; matching volume does not establish waveform fidelity. "
    "Training progress is work completed, not a quality measurement. "
    "Raw quality and loss values plus the latest status remain in the Details run."
)


def configuration_events():
    layout = layout_pb2.Layout(category=[layout_pb2.Category(title="Progress", chart=[
        layout_pb2.Chart(title="All quality metrics", multiline=layout_pb2.MultilineChartContent(tag=[r"^overview/.*"]))])])
    config = summary_pb2.Summary.Value(tag=metadata.CONFIG_SUMMARY_TAG,
        metadata=metadata.create_summary_metadata(),
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[layout.SerializeToString()]))
    guide = summary_pb2.Summary.Value(tag="Guide/Metric targets/text_summary",
        metadata=summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text")),
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[GUIDE.encode()]))
    return event_pb2.Event(wall_time=time.time(),step=0,summary=summary_pb2.Summary(value=[config,guide]))


class MetricProjection:
    def __init__(self, source_dir, display_dir):
        self.source,self.display = validate_paths(source_dir,display_dir)
        if len(set(METRIC_RUNS.values())) != len(METRIC_RUNS): raise ValueError("Metric run names must be unique")
        if any(any(char in run for char in "()/\\") for run in METRIC_RUNS.values()):
            raise ValueError("Metric run names must be safe single directory names")
        self.writers = {}
        self._writer("Details").add_event(configuration_events())
        self.loaders = {}
        self.seen = set()
        self.counts = {run:0 for run in METRIC_RUNS.values()}
        self.details_written = 0
        self.flush()

    def _writer(self,run):
        if run not in self.writers:
            self.writers[run] = EventFileWriter(str(self.display/run),filename_suffix=".metric-display")
        return self.writers[run]

    def poll(self):
        added=0
        for path in sorted(self.source.glob("events.out.tfevents.*")):
            if not path.is_file(): continue
            if str(path) not in self.loaders: self.loaders[str(path)] = LegacyEventFileLoader(str(path))
            for event in self.loaders[str(path)].Load():
                if not event.HasField("summary"): continue
                destinations = {}
                for value in event.summary.value:
                    kind=value.WhichOneof("value")
                    overview=value.tag.startswith("overview/") and kind=="simple_value"
                    raw=kind=="simple_value" and (value.tag.startswith("quality/all/") or value.tag in RAW_LOSS_TAGS)
                    text=kind=="tensor" and value.tag.startswith(TEXT_PREFIXES)
                    if not (overview or raw or text): continue
                    key=event_key(value,event.step,event.wall_time)
                    if key in self.seen: continue
                    copied=summary_pb2.Summary.Value(); copied.CopyFrom(value)
                    run="Details"
                    if overview:
                        name=value.tag.removeprefix("overview/")
                        if name not in METRIC_RUNS: raise ValueError("Unmapped overview metric: "+name)
                        run=METRIC_RUNS[name]; copied.tag=COMMON_TAG
                        self.counts[run] += 1
                    else:
                        self.details_written += 1
                    destinations.setdefault(run,[]).append(copied)
                    self.seen.add(key); added += 1
                for run,values in destinations.items():
                    self._writer(run).add_event(event_pb2.Event(step=event.step,wall_time=event.wall_time,
                        summary=summary_pb2.Summary(value=values)))
        self.flush()
        return added

    def flush(self):
        for writer in self.writers.values(): writer.flush()

    def seal_ready(self):
        missing=[run for run,count in self.counts.items() if count==0]
        if missing: raise RuntimeError("Incomplete initial metric history: "+", ".join(missing))
        self.flush()
        report={"ready":True,"source_dir":str(self.source),"display_dir":str(self.display),
                "metric_runs":METRIC_RUNS,"common_tag":COMMON_TAG,"counts_by_run":self.counts,
                "details_values_written":self.details_written,"source_files":sorted(self.loaders),
                "single_layout_run":"Details","source_read_only":True,
                "preserved_fields":["scalar value","event step","event wall_time"]}
        temporary=self.display/"projection-ready.json.tmp"
        temporary.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        temporary.replace(self.display/"projection-ready.json")
        return report

    def close(self):
        for writer in self.writers.values(): writer.flush(); writer.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-logdir",type=Path,required=True)
    parser.add_argument("--display-logdir",type=Path,required=True)
    parser.add_argument("--poll-seconds",type=float,default=5.)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if args.poll_seconds<=0: parser.error("Polling interval must be positive")
    projection=MetricProjection(args.source_logdir,args.display_logdir)
    try:
        projection.poll()
        print(json.dumps(projection.seal_ready()),flush=True)
        if args.once: return
        while True:
            time.sleep(args.poll_seconds)
            if projection.poll(): projection.seal_ready()
    finally:
        projection.close()


if __name__=="__main__": main()
