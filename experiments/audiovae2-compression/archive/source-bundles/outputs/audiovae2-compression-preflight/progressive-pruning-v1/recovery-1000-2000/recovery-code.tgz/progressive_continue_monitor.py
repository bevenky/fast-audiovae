"""Replay saved cut1 logs once, then observe the unchanged-width continuation.

No audio, model, optimizer, random generator or source iterator is accessed.
The caller authenticates the parent checkpoint and checks the resumed evaluation
against its saved metrics. Replaying logs records zero new training exposure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import progressive_monitor as original

VERSION = "audiovae2_progressive_continue_monitor_v1"
QUALITY_STEPS = (0, 500, 1000, 1500, 2000)
COMMON_TAG = original.COMMON_TAG
REGIONS = original.REGIONS


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_report(report, signature=None):
    original._finite(report)
    panel = original._panel(report)
    if signature is not None and panel != signature:
        raise ValueError("The fixed validation panel or teacher changed")
    aggregate = report["aggregate"]
    for key in original.ERRORS:
        original._number(aggregate[key], key, 0)
    cosine = aggregate.get("nonquiet_cosine_mean")
    if cosine is not None and not -1.000001 <= original._number(cosine, "cosine") <= 1.000001:
        raise ValueError("Invalid waveform cosine")
    original._number(aggregate["peak_abs_max"], "peak", 0)
    extras = report.get("overview_window_metrics", {}).get("by_source", {})
    if extras and set(extras) != set(panel["sources"]):
        raise ValueError("Active-window source identities differ from validation")
    for source, row in extras.items():
        for key in ("active_teacher_energy", "active_student_energy"):
            original._number(row[key], source + "/" + key, 0)
    return panel


def load_history(parent_dir):
    """Read the completed0..1000 history without interpreting it as new work."""
    parent = Path(parent_dir).resolve()
    reports, identities = {}, {}
    for step in QUALITY_STEPS[:3]:
        path = parent / f"development-step{step}.json"
        reports[step] = json.loads(path.read_text())
        identities[str(path)] = _sha(path)
    signature = _validate_report(reports[0])
    if all(reports[0]["aggregate"][key] == 0 for key in ("mae", "mel", "group_mse")):
        raise ValueError("Keep the exact-copy teacher control separate from the pruned baseline")
    for step in (500, 1000):
        _validate_report(reports[step], signature)
    path = parent / "train.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities[str(path)] = _sha(path)
    if len(records) != 1000 or [r.get("step") for r in records] != list(range(1, 1001)):
        raise ValueError("Expected exactly the saved1000 consecutive training records")
    for record in records:
        original._finite(record)
        for key in ("total", "waveform", "mel", "feature"):
            original._number(record[key], key)
    control = parent / "full-width-copy.json"
    if control.exists():
        identities[str(control)] = _sha(control)
    return reports, records, {
        "parent_dir": str(parent), "files_sha256": identities,
        "replayed_quality_steps": [0, 500, 1000], "replayed_training_records": 1000,
        "new_training_updates_from_replay": 0, "new_audio_samples_from_replay": 0,
        "baseline": "original pruned step0, not the separate full-width-copy control",
        "prior_unique_sources": records[-1].get("unique_sources"),
        "prior_scored_audio_hours": records[-1].get("audio_hours"),
    }


class ContinuationMonitor(original.ProgressiveMonitor):
    """Simple API used by progressive_continue.py.

    Constructor replays parent_dir's saved quality and train.jsonl once. Then
    call log_training(record, step) at1001..2000 and log_validation(report, step)
    at1500/2000. A single log_validation(report,1000) verifies panel identity and
    records the caller-checked resumed observation without duplicating curves.
    It does not replace the runner's numerical checkpoint-parity assertion.
    """

    def __init__(self, logdir, *, parent_dir, source_metadata=None,
                 cut_name="cut1_384x256", widths=(384, 256), total_updates=2000,
                 correlation_target=.99, writer_factory=None):
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", cut_name) or tuple(widths) != (384, 256)
                or type(total_updates) is not int or total_updates != 2000
                or not 0 < original._number(correlation_target, "correlation target") <= 1):
            raise ValueError("Expected unchanged cut1 widths384/256 and target2000")
        self.path = Path(logdir).resolve()
        parent = Path(parent_dir).resolve()
        if self.path.is_relative_to(parent) or parent.is_relative_to(self.path):
            raise ValueError("Use a separate directory from the retained original cut")
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError("Use a fresh continuation TensorBoard directory")
        reports, records, history = load_history(parent)
        self.path.mkdir(parents=True, exist_ok=True)
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            writer_factory = lambda path: SummaryWriter(str(path), flush_secs=10)
        self.factory, self.writers = writer_factory, {}
        self.cut_name, self.widths = cut_name, tuple(widths)
        self.total_updates, self.target = total_updates, correlation_target
        self.baseline, self.signature = None, None
        self.last_training_step, self.last_validation_step = 0, -1
        self.closed, self.resume_checked = False, False
        self.source_metadata = copy.deepcopy(source_metadata or {})
        self.history_identity = copy.deepcopy(history)
        self.details = self._writer("Details")
        try:
            self.details.add_custom_scalars({"Pruning " + cut_name: {
                "Measured quality and training progress": ["Multiline", [r"^overview/Percent$"]]}})
            self.details.add_text("Guide/Reading this continuation",
                "The SAME cut1 student continues at widths384/256. Saved0..1000 logs are replayed once, "
                "without training, model evaluation or audio reuse. Progress uses2000 updates across the entire chart; "
                "step1000 is50%. Error reductions keep the original pruned step0 baseline, never the exact-copy teacher control. "
                "They are relative error reductions, not perceptual fidelity; negative changes remain visible. "
                "Quality is measured only at0,500,1000,1500,2000. Raw losses are logged every optimizer update. "
                "The cosine milestone applies only to correlation. Peak100% is full scale. Active and whistling RMS "
                "levels target100% of the teacher, with both lower and higher levels potentially wrong; "
                "these are pooled RMS ratios, not least-squares projection gains. "
                "All seven overlapping quiet groups retain their measured levels, errors and failure categories. "
                "No later pruning cut starts automatically.", 0)
            self._json("monitor.json", {"version": VERSION, "cut_name": cut_name,
                "widths": list(widths), "total_updates": total_updates,
                "correlation_milestone": correlation_target, "automatic_next_cut": False,
                "quality_steps": list(QUALITY_STEPS), "history": history,
                "baseline": "Original pruned step0 of this cut"})
            self._replaying = True
            self.log_validation(reports[0], 0)
            for record in records:
                self.log_training(record, record["step"])
                if record["step"] in (500, 1000):
                    self.log_validation(reports[record["step"]], record["step"])
            self._replaying = False
            self._json("history-replay.json", {**history, "complete": True})
            self.assert_history_unchanged()
            self.details.add_text("Monitor/Training status",
                "Saved history replay complete; same-width recovery1000 to2000 is ready. "
                "The replay performed zero training updates and consumed no additional audio.", 1000)
            self.flush()
        except BaseException:
            self.close()
            raise

    def assert_history_unchanged(self):
        if any(_sha(Path(path)) != expected for path, expected in self.history_identity["files_sha256"].items()):
            raise RuntimeError("Retained original monitoring history changed")

    def log_training(self, record, step):
        self._step(step)
        if step != self.last_training_step + 1 or record.get("step", step) != step:
            raise ValueError("Training log steps must be consecutive without replaying an update")
        if not self._replaying and step < 1001:
            raise ValueError("Saved training history has already been replayed")
        return super().log_training(record, step)

    def log_validation(self, report, step):
        self._step(step)
        signature = _validate_report(report, self.signature)
        if (not self._replaying and step == 1000 and self.last_validation_step == 1000
                and not self.resume_checked):
            self._json("resumed-step1000-observation.json", {
                "step": 1000, "aggregate": report["aggregate"], "chart_point_replaced": False,
                "saved_metric_parity": "checked independently by the continuation runner"})
            self.resume_checked = True
            return
        if step not in QUALITY_STEPS or step <= self.last_validation_step:
            raise ValueError("Validation expects increasing fixed0/500/1000/1500/2000 milestones")
        expected = 0 if self.last_validation_step == -1 else self.last_validation_step + 500
        if step != expected or (step > 0 and step > self.last_training_step):
            raise ValueError("A quality milestone cannot be skipped or precede its training records")
        if self.baseline is None and step != 0:
            raise ValueError("The original pruned step0 baseline must be replayed first")
        if not self._replaying and step not in (1500, 2000):
            raise ValueError("Historical validation has already been replayed")
        aggregate = report["aggregate"]
        if self.baseline is None:
            self.baseline, self.signature = dict(aggregate), signature
            self._json("pruned-step0-baseline.json", {"cut_name": self.cut_name,
                "widths": self.widths, "step": 0, "aggregate": self.baseline, "panel": signature})
        cosine = aggregate.get("nonquiet_cosine_mean")
        self._overview(f"01 Active waveform cosine - milestone {100*self.target:g}%", None if cosine is None else 100*cosine, step)
        self._overview(f"02 Cosine milestone - {100*self.target:g}%", 100*self.target, step)
        omitted = []
        for key, label in original.ERRORS.items():
            origin = self.baseline[key]
            if origin > 0:
                self._overview(label, 100*(1-aggregate[key]/origin), step)
            else:
                omitted.append(key)
        for name, label in original.PASSES.items():
            row = report["quiet_regions"]["regions"][name]
            if row["windows"]:
                self._overview(label, 100*(1-row["failed"]/row["windows"]), step)
        self._overview("12 Signal peak - limit 100% full scale", 100*aggregate["peak_abs_max"], step)
        # Training already emitted this same progress point at the review step.
        if step == 0:
            self._overview("13 Training progress - target 100%", 0., step)
        ratios = {}
        rows = report.get("overview_window_metrics", {}).get("by_source", {})
        for name, label in (("active", "14 Active RMS level - teacher matched 100%"),
                            ("whistle", "15 Whistling RMS level - teacher matched 100%")):
            chosen = [row for source, row in rows.items() if name == "active" or
                      "human_whistling_source_description" in self.source_metadata.get(source, {}).get("verified_source_labels", [])]
            te = sum(row["active_teacher_energy"] for row in chosen)
            pe = sum(row["active_student_energy"] for row in chosen)
            value = math.sqrt(pe/te) if te > 0 else None
            ratios[name + "_rms_ratio"] = value
            self._overview(label, 100*value if value is not None else None, step)
            if value is not None:
                self.details.add_scalar("quality/"+name+"_rms_ratio", value, step)
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
               "amplitude_ratios": ratios, "omitted_zero_baseline_reductions": omitted}
        self._json("latest-validation.json", raw)
        self.details.add_text("Monitor/Latest exact raw values", "```json\n"+json.dumps(raw, indent=2)+"\n```", step)
        self.last_validation_step = step
        self.flush()
