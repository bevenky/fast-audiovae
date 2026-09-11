"""Project overview events into a separate display run with safe chart labels.

Every source file retains its own read cursor so late training-writer events
remain visible. Source event files and authenticated training code are read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import time

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.plugins.custom_scalar import layout_pb2, metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter


TEXT_PREFIXES = ("Monitor/Latest raw metrics", "Monitor/Live status", "Guide/")
RAW_LOSS_TAGS = {"loss/total", "loss/teacher_waveform", "loss/teacher_mel", "loss/teacher_group_mse"}


def safe_tag(tag):
    return re.sub(r"\s+", " ", tag.replace("(", " - ").replace(")", "")).strip()


def validate_paths(source_dir, display_dir):
    source, display = Path(source_dir).resolve(), Path(display_dir).resolve()
    if source == display or source in display.parents or display in source.parents:
        raise ValueError("Source and display directories must be separate, non-nested paths")
    if not source.is_dir(): raise FileNotFoundError("Source event directory does not exist")
    if display.exists() and any(display.iterdir()):
        raise FileExistsError("Use a new empty display directory to avoid duplicate projection history")
    return source, display


def layout_event():
    layout = layout_pb2.Layout(category=[layout_pb2.Category(title="Progress", chart=[
        layout_pb2.Chart(title="All quality metrics", multiline=layout_pb2.MultilineChartContent(tag=[r"^overview/.*"]))])])
    tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING, string_val=[layout.SerializeToString()])
    summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=metadata.CONFIG_SUMMARY_TAG,
        metadata=metadata.create_summary_metadata(), tensor=tensor)])
    return event_pb2.Event(wall_time=time.time(), step=0, summary=summary)


def event_key(value, step, wall_time):
    kind = value.WhichOneof("value")
    payload = struct.pack("!d", value.simple_value) if kind == "simple_value" else value.tensor.SerializeToString()
    return value.tag, step, struct.pack("!d", wall_time), kind, payload


class EventProjection:
    def __init__(self, source_dir, display_dir):
        self.source, self.display = validate_paths(source_dir, display_dir)
        self.writer = EventFileWriter(str(self.display), filename_suffix=".safe-display")
        self.writer.add_event(layout_event())
        self.writer.flush()
        self.loaders = {}
        self.seen = set()
        self.tag_origins = {}
        self.values_written = 0
        self.scalar_values_written = 0

    def poll(self):
        """Load newly appended events from all old and newly discovered writers."""
        added = 0
        for path in sorted(self.source.glob("events.out.tfevents.*")):
            if not path.is_file(): continue
            loader = self.loaders.setdefault(str(path), None)
            if loader is None:
                loader = self.loaders[str(path)] = LegacyEventFileLoader(str(path))
            for original in loader.Load():
                if not original.HasField("summary"): continue
                values = []
                for value in original.summary.value:
                    kind = value.WhichOneof("value")
                    overview = value.tag.startswith("overview/") and kind == "simple_value"
                    scalar = kind == "simple_value" and (overview or value.tag.startswith("quality/all/") or value.tag in RAW_LOSS_TAGS)
                    text = value.tag.startswith(TEXT_PREFIXES) and kind == "tensor"
                    if not (scalar or text): continue
                    key = event_key(value, original.step, original.wall_time)
                    if key in self.seen: continue
                    copied = summary_pb2.Summary.Value()
                    copied.CopyFrom(value)
                    if overview:
                        copied.tag = safe_tag(value.tag)
                        origin = self.tag_origins.setdefault(copied.tag, value.tag)
                        if origin != value.tag: raise ValueError("Safe chart label would merge distinct source tags")
                    values.append(copied)
                    self.seen.add(key)
                    added += 1
                    self.scalar_values_written += int(overview)
                if values:
                    copied_event = event_pb2.Event(wall_time=original.wall_time, step=original.step,
                                                  summary=summary_pb2.Summary(value=values))
                    self.writer.add_event(copied_event)
        self.values_written += added
        self.writer.flush()
        return added

    def seal_ready(self):
        self.writer.flush()
        receipt = {"ready":True, "source_dir":str(self.source), "display_dir":str(self.display),
                   "source_files":sorted(self.loaders), "summary_values_written":self.values_written,
                   "overview_scalar_values_written":self.scalar_values_written,
                   "tag_mapping":{old:new for new,old in self.tag_origins.items()},
                   "preserved_fields":["scalar value", "event step", "event wall_time"],
                   "source_read_only":True, "source_files_polled_independently":True}
        temporary = self.display/"projection-ready.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        temporary.replace(self.display/"projection-ready.json")
        return receipt

    def close(self):
        self.writer.flush(); self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-logdir", "--source-dir", dest="source_dir", type=Path, required=True)
    parser.add_argument("--display-logdir", "--display-dir", dest="display_dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0: parser.error("Polling interval must be positive")
    projection = EventProjection(args.source_dir, args.display_dir)
    try:
        projection.poll()
        if projection.scalar_values_written == 0:
            raise RuntimeError("No overview scalar history found; the display is not ready")
        print(json.dumps(projection.seal_ready()), flush=True)
        if args.once: return
        while True:
            time.sleep(args.poll_seconds)
            if projection.poll(): projection.seal_ready()
    finally:
        projection.close()


if __name__ == "__main__": main()
