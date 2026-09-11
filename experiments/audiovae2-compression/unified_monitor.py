"""One TensorBoard overview with explicit units and unmodified raw metrics."""
from __future__ import annotations

import math

import run_pilot as base


CHART = {
    'Progress': {'All quality metrics': ['Multiline', [r'^overview/.*']]},
}


def window_summary(windows):
    quiet = [w for w in windows if w['is_quiet']]
    near = [w for w in quiet if w['teacher_rms'] <= 1e-5]
    active = [w for w in windows if not w['is_quiet']]
    def power(key):
        return sum(w['valid_samples']*w[key]**2 for w in active)
    return {'quiet_windows':len(quiet), 'quiet_failed_windows':sum(w['passed'] is False for w in quiet),
            'near_silence_windows':len(near), 'near_silence_failed_windows':sum(w['passed'] is False for w in near),
            'active_teacher_energy':power('teacher_rms'), 'active_student_energy':power('student_rms'),
            'valid_samples':sum(w['valid_samples'] for w in windows)}


class UnifiedMonitor:
    def __init__(self, writer, baseline_aggregate, source_metadata):
        self.writer = writer
        self.baseline = dict(baseline_aggregate)
        self.source_metadata = source_metadata

    def initialize_layout(self):
        self.writer.add_custom_scalars(CHART)
        self.writer.add_text('Guide/How to read the overview',
            '**Active waveform correlation (%)** is 100 times the teacher-active waveform cosine. '
            'The 99% line applies only to correlation, not perceptual accuracy or every trace. '
            'Error reductions use the fixed untrained step-0 checkpoint; negative values remain visible. '
            'Quiet pass rates use the existing residual and amplitude checks. '
            'Signal peak and whistling level are descriptive amplitude ratios, not higher-is-better quality scores. '
            'Whistling level should approach 100% of the teacher. '
            'All quality points are measured validation results, without interpolation or guessed near-silence values. '
            'Raw values remain in Scalars under quality/all and in Text under Monitor/Latest raw metrics.', 0)

    def evaluate(self, model, teacher, crops, spectral):
        """Observe already-computed windows without changing inference or reductions.

        Evaluation is synchronous and singleton. The temporary observer is
        restored even on failure, and every record is checked against its source.
        """
        original = base.quiet_window_metrics
        if getattr(original, '_overview_observer', False):
            raise RuntimeError('Nested overview evaluation is unsupported')
        captured = []
        def observer(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(window_summary(result['windows']))
            return result
        observer._overview_observer = True
        base.quiet_window_metrics = observer
        try:
            report = base.evaluate(model, teacher, crops, spectral, batch_size=1)
        finally:
            base.quiet_window_metrics = original
        if len(captured) != len(report['rows']):
            raise RuntimeError('Quiet observation count differs from validation source count')
        summaries = {}
        for row, summary in zip(report['rows'], captured):
            if (row['samples'] != summary['valid_samples'] or row['quiet_windows'] != summary['quiet_windows']
                    or row['quiet_failed'] != summary['quiet_failed_windows']):
                raise RuntimeError('Quiet observation is misaligned with source scoring')
            summaries[row['source_id']] = summary
        report['overview_window_metrics'] = {
            'near_silence_windows':sum(x['near_silence_windows'] for x in captured),
            'near_silence_failed_windows':sum(x['near_silence_failed_windows'] for x in captured),
            'by_source':summaries,
        }
        return report

    def _scalar(self, name, value, step):
        if value is not None and math.isfinite(value):
            self.writer.add_scalar('overview/'+name, value, step)

    def log_validation(self, report, step):
        a = report['aggregate']
        c = a.get('nonquiet_cosine_mean')
        self._scalar('Active waveform correlation (%)', None if c is None else 100*c, step)
        self._scalar('Correlation target only (99%)', 99., step)
        for key, name in (
            ('mae','Waveform MAE reduction vs step 0 (%)'),
            ('mel','Mel error reduction vs step 0 (%)'),
            ('group_mse','Stage 2-4 output MSE reduction vs step 0 (%)'),
            ('quiet_residual_rms_mean','Quiet RMS error reduction vs step 0 (%)'),
        ):
            origin, now = self.baseline.get(key), a.get(key)
            value = 100*(1-now/origin) if origin is not None and origin > 0 and now is not None else None
            self._scalar(name, value, step)
        count, failed = a.get('quiet_windows',0), a.get('quiet_failed_windows',0)
        if count: self._scalar('Quiet windows passing (%)', 100*(1-failed/count), step)
        extra = report.get('overview_window_metrics',{})
        near_count, near_failed = extra.get('near_silence_windows',0), extra.get('near_silence_failed_windows',0)
        if near_count:
            self._scalar('Near-silence windows passing (%)', 100*(1-near_failed/near_count), step)
        peak = a.get('peak_abs_max')
        self._scalar('Signal peak (% full scale, not a quality score)', None if peak is None else 100*peak, step)
        teacher_energy = student_energy = 0.
        for source_id, row in extra.get('by_source',{}).items():
            labels = self.source_metadata.get(source_id,{}).get('verified_source_labels',[])
            if 'human_whistling_source_description' in labels:
                teacher_energy += row['active_teacher_energy']
                student_energy += row['active_student_energy']
        gain = 100*math.sqrt(student_energy/teacher_energy) if teacher_energy > 0 else None
        self._scalar('Whistling level (% of teacher, 100 is matched)', gain, step)
        rows = [('| Metric | Raw value |'), ('|---|---:|'),
                f'| Validation step | {step} |']
        for key in ('nonquiet_cosine_mean','mae','mel','group_mse','waveform_nrmse',
                    'quiet_residual_rms_mean','quiet_failed_windows','quiet_windows',
                    'peak_abs_max','overshoot_samples'):
            value = a.get(key)
            rows.append(f'| {key} | {value if value is not None else "unavailable"} |')
        rows.append(f'| Near-silence failed / measured | {str(near_failed)+" / "+str(near_count) if near_count else "unmeasured at this step"} |')
        rows.append(f'| Active whistling level / teacher (%) | {gain if gain is not None else "unmeasured at this step"} |')
        self.writer.add_text('Monitor/Latest raw metrics','\n'.join(rows),step)
        self.writer.flush()

    def log_training(self, record, step):
        self.writer.add_text('Monitor/Training status',
            f'Update **{step}**; distinct sources **{record.get("unique_sources", "unavailable")}**; '
            f'scored audio hours **{record.get("audio_hours", "unavailable")}**. '
            'Quality overview updates at the fixed validation milestones, not on every optimizer step.', step)

    def log_waiting(self, message, step):
        self.writer.add_text('Monitor/Training status',message,step)
        self.writer.flush()
