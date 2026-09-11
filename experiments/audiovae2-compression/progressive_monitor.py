"""Fresh, observation-only TensorBoard monitoring for one pruning cut."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

VERSION = "audiovae2_progressive_monitor_v1"
COMMON_TAG = "overview/Percent"
REGIONS = (
    "all_quiet", "near_silence", "near_startup_first20ms",
    "source_zero_20to40ms", "source_zero_after40ms", "near_after800ms",
    "quiet_nonzero_reference",
)
ERRORS = {
    "mae": "03 Waveform error reduction from cut start",
    "mel": "04 Mel error reduction from cut start",
    "group_mse": "05 Group error reduction from cut start",
    "quiet_residual_rms_mean": "06 Quiet error reduction from cut start",
}
PASSES = {
    "all_quiet": "07 Quiet windows passing - target 100%",
    "near_silence": "08 Near-silence passing - target 100%",
    "near_startup_first20ms": "09 Startup near-silence passing - target 100%",
    "source_zero_after40ms": "10 Sustained source silence passing - target 100%",
    "near_after800ms": "11 Interior near-silence passing - target 100%",
}


def _finite(value):
    if isinstance(value, dict):
        for child in value.values(): _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite monitoring value")


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric metric: " + name)
    if minimum is not None and value < minimum:
        raise ValueError("Negative metric: " + name)
    return value


def _panel(report):
    aggregate, quiet = report["aggregate"], report["quiet_regions"]
    regions = quiet["regions"]
    if set(regions) != set(REGIONS): raise ValueError("Expected all seven quiet regions")
    sources = [row["source_id"] for row in report["rows"]]
    if len(sources) != aggregate["sources"] or len(set(sources)) != len(sources):
        raise ValueError("Validation source accounting differs")
    signature = {"sources": sources, "samples": aggregate["samples"],
                 "quiet_identity": quiet["window_identity_sha256"], "regions": {}}
    for name, row in regions.items():
        count, failed = row["windows"], row["failed"]
        if type(count) is not int or type(failed) is not int or not 0 <= failed <= count:
            raise ValueError("Invalid quiet-window count")
        cats = row["failure_categories"]
        if (set(cats) != {"passed", "amplitude_only", "residual_only", "both"}
                or any(type(v) is not int or v < 0 for v in cats.values())
                or sum(cats.values()) != count or cats["passed"] != count-failed
                or row["amplitude_failed"] != cats["amplitude_only"]+cats["both"]):
            raise ValueError("Quiet categories disagree")
        signature["regions"][name] = {key: row[key] for key in (
            "windows", "samples", "window_ids_sha256", "teacher_rms")}
    if (regions["all_quiet"]["windows"] != aggregate["quiet_windows"]
            or regions["all_quiet"]["failed"] != aggregate["quiet_failed_windows"]):
        raise ValueError("Quiet overview differs from original metric")
    return signature


class ProgressiveMonitor:
    """One fixed pruned baseline and fresh metric runs, without a projector."""

    def __init__(self, logdir, *, cut_name="cut1_384x256", widths=(384, 256),
                 total_updates=1000, correlation_target=.99, writer_factory=None):
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", cut_name)
                or len(widths) != 2 or any(type(w) is not int or w <= 0 for w in widths)
                or type(total_updates) is not int or total_updates != 1000
                or not 0 < _number(correlation_target, "correlation target") <= 1):
            raise ValueError("Invalid cut monitoring configuration")
        self.path = Path(logdir).resolve()
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError("Use a fresh TensorBoard directory for this pruning cut")
        self.path.mkdir(parents=True, exist_ok=True)
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            writer_factory = lambda path: SummaryWriter(str(path), flush_secs=10)
        self.factory, self.writers = writer_factory, {}
        self.cut_name, self.widths = cut_name, tuple(widths)
        self.total_updates, self.target = total_updates, correlation_target
        self.baseline, self.signature = None, None
        self.last_training_step, self.last_validation_step = -1, -1
        self.closed = False
        self.details = self._writer("Details")
        self.details.add_custom_scalars({"Pruning " + cut_name: {
            "Measured quality and training progress": ["Multiline", [r"^overview/Percent$"]]}})
        self.details.add_text("Guide/Reading this cut",
            f"Cut **{cut_name}**, stage widths **{widths[0]}/{widths[1]}**. "
            "Each colored run represents one metric from the same student. "
            f"The {100*self.target:g}% cosine line is a reconstruction milestone, not perceptual accuracy. "
            "Error reduction uses this cut's actual pruned step0, never the exact-copy teacher control. "
            "It is a relative improvement, not fidelity; negative reductions remain visible. "
            "An error with zero baseline has no reduction percentage and remains available as a raw value. "
            "Quality is measured at step0,500,1000; training losses update every optimizer step in Details. "
            "Progress counts work completed and is not a quality score. Peak100% is full scale, not an improvement target. "
            "Startup near-silence covers the first20ms, sustained source silence covers exact-zero prepared input after40ms, "
            "and interior near-silence starts after800ms. These subsets overlap. The separate teacher20–40ms transient "
            "and all seven regions retain their exact residual, level, and failure counts in Details. "
            "No later pruning cut starts automatically.", 0)
        self._json("monitor.json", {"version": VERSION, "cut_name": cut_name,
            "widths": list(widths), "total_updates": total_updates,
            "correlation_milestone": correlation_target, "automatic_next_cut": False,
            "baseline": "Actual pruned step0 of this cut", "quality_steps": [0, 500, 1000]})
        self.flush()

    def _writer(self, label):
        run = self.cut_name + " " + label
        if run not in self.writers: self.writers[run] = self.factory(self.path/run)
        return self.writers[run]

    def _json(self, name, value):
        path = self.path/name
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        temporary.replace(path)

    def _step(self, step):
        if self.closed or type(step) is not int or not 0 <= step <= self.total_updates:
            raise ValueError("Invalid cut-local optimizer step or closed monitor")

    def _overview(self, label, value, step):
        if value is not None: self._writer(label).add_scalar(COMMON_TAG, _number(value, label), step)

    def log_validation(self, report, step):
        self._step(step); _finite(report)
        if step not in (0, 500, 1000): raise ValueError("Quality logging expects the fixed0/500/1000 milestones")
        if step <= self.last_validation_step: raise ValueError("Validation steps must increase")
        if self.baseline is None and step != 0: raise ValueError("First validation must be pruned step0")
        aggregate = report["aggregate"]
        signature = _panel(report)
        for key in ERRORS: _number(aggregate[key], key, 0)
        cosine = aggregate.get("nonquiet_cosine_mean")
        if cosine is not None and not -1.000001 <= _number(cosine, "cosine") <= 1.000001:
            raise ValueError("Invalid waveform cosine")
        _number(aggregate["peak_abs_max"], "peak", 0)
        if self.baseline is None:
            if all(aggregate[key] == 0 for key in ("mae", "mel", "group_mse")):
                raise ValueError("Keep the exact-copy teacher control separate from the pruned baseline")
            self.baseline, self.signature = dict(aggregate), signature
            self._json("pruned-step0-baseline.json", {"cut_name": self.cut_name,
                "widths": self.widths, "step": 0, "aggregate": self.baseline, "panel": signature})
        elif signature != self.signature:
            raise ValueError("The fixed validation panel or teacher changed")
        self._overview(f"01 Active waveform cosine - milestone {100*self.target:g}%", None if cosine is None else 100*cosine, step)
        self._overview(f"02 Cosine milestone - {100*self.target:g}%", 100*self.target, step)
        omitted = []
        for key, label in ERRORS.items():
            origin = self.baseline[key]
            if origin > 0: self._overview(label, 100*(1-aggregate[key]/origin), step)
            else: omitted.append(key)
        for name, label in PASSES.items():
            row = report["quiet_regions"]["regions"][name]
            if row["windows"]: self._overview(label, 100*(1-row["failed"]/row["windows"]), step)
        self._overview("12 Signal peak - limit 100% full scale", 100*aggregate["peak_abs_max"], step)
        self._overview("13 Training progress - target 100%", 100*step/self.total_updates, step)
        for key, value in aggregate.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.details.add_scalar("quality/"+key, value, step)
        for name, row in report["quiet_regions"]["regions"].items():
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.details.add_scalar("quiet/"+name+"/"+key, value, step)
            for key, count in row["failure_categories"].items():
                self.details.add_scalar("quiet/"+name+"/"+key, count, step)
        raw = {"step": step, "cut_name": self.cut_name, "widths": self.widths,
               "aggregate": aggregate, "quiet_regions": report["quiet_regions"],
               "omitted_zero_baseline_reductions": omitted}
        self._json("latest-validation.json", raw)
        self.details.add_text("Monitor/Latest exact raw values", "```json\n"+json.dumps(raw, indent=2)+"\n```", step)
        self.last_validation_step = step
        self.flush()

    def log_training(self, record, step):
        self._step(step); _finite(record)
        if self.baseline is None or step < 1 or step <= self.last_training_step:
            raise ValueError("Training needs a baseline and increasing positive steps")
        names = {"total": "train/total", "waveform": "loss/teacher_waveform", "mel": "loss/teacher_mel",
                 "feature": "loss/group_feature", "step_seconds": "performance/step_seconds",
                 "elapsed_seconds": "performance/elapsed_seconds", "unique_sources": "data/unique_sources",
                 "audio_hours": "data/scored_audio_hours", "gradient_norm": "train/gradient_norm"}
        for key, tag in names.items():
            if record.get(key) is not None: self.details.add_scalar(tag, _number(record[key], key), step)
        self._overview("13 Training progress - target 100%", 100*step/self.total_updates, step)
        if step == 1 or step % 25 == 0 or step == self.total_updates:
            self.details.add_text("Monitor/Training status",
                f"Cut **{self.cut_name}**, update **{step}/{self.total_updates}**. "
                f"Distinct sources **{record.get('unique_sources', 'unavailable')}**. "
                f"Latest measured quality is step **{self.last_validation_step}**; values between reviews are not new measurements.", step)
            self.flush()
        self.last_training_step = step

    def log_waiting(self, message, step):
        self._step(step)
        self.details.add_text("Monitor/Training status", str(message), step)
        self.flush()

    def flush(self):
        for writer in self.writers.values(): writer.flush()

    def close(self):
        if not self.closed:
            for writer in self.writers.values(): writer.flush(); writer.close()
            self.closed = True
