"""Plot completed diagnostic reports and saved waveforms without model execution."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'results'
variants = [
    ('baseline', 'Current\nstudent', '#2563eb'),
    ('teacher_up4', 'Exact teacher\nfeatures (oracle)', '#9333ea'),
    ('native_fit', 'Fitted\nupsampler', '#ea580c'),
    ('native_fit_original_residuals', 'Fit + original\nresidual units', '#059669'),
]
reports = {key: json.loads((root / f'trained4625-{key}.json').read_text()) for key, _, _ in variants}
snippets = np.load(root / 'waveform-snippets.npz')
index = {r['key']: r for r in json.loads((root / 'waveform-snippets.json').read_text())}
fig, ax = plt.subplots(2, 2, figsize=(13.6, 8.5), layout='constrained')
fig.suptitle('Correct teacher features can upset an adapted downstream decoder', fontsize=16, weight='bold')
labels = [label for _, label, _ in variants]
colors = [color for _, _, color in variants]
gains, errors = [], []
for key, _, _ in variants:
    report = reports[key]['quality']
    rows = report['overview_window_metrics']['by_source'].values()
    teacher = sum(r['active_teacher_energy'] for r in rows)
    student = sum(r['active_student_energy'] for r in rows)
    gains.append(100 * math.sqrt(student / teacher))
    errors.append(report['aggregate']['mae'])
for axis, values, title, ylabel in (
    (ax[0, 0], gains, 'Active audio level across 96 recordings', 'RMS level (% of teacher)'),
    (ax[0, 1], errors, 'Final waveform error across 96 recordings', 'Mean absolute error'),
):
    bars = axis.bar(range(4), values, color=colors, width=.62)
    axis.set_xticks(range(4), labels, fontsize=9)
    axis.set_title(title, fontsize=12)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(values) * 1.2)
    axis.bar_label(bars, labels=[f'{v:.1f}%' if values is gains else f'{v:.5f}' for v in values], fontsize=10, padding=4)
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)
ax[0, 0].axhline(100, color='#111827', linestyle='--', linewidth=1)

for axis, snippet_key, title, residual in (
    (ax[1, 0], 'source1_teacher_peak', 'Kannada: 12 ms around the teacher peak', False),
    (ax[1, 1], 'source0_first_near_silence', 'Spanish: interior near-silence at 10.24 s', True),
):
    teacher = snippets[snippet_key + '_teacher']
    if residual:
        a, b = 0, len(teacher)
    else:
        center = int(np.argmax(np.abs(teacher)))
        a, b = max(0, center-288), min(len(teacher), center+288)
    times = (np.arange(a, b) + index[snippet_key]['absolute_source_samples'][0]) / 48.0
    if not residual:
        axis.plot(times, teacher[a:b], color='#111827', linewidth=1.1, label='Teacher', zorder=5)
    else:
        axis.axhline(0, color='#111827', linewidth=1, label='Teacher (zero residual)')
    for key, label, color in variants:
        y = snippets[snippet_key + '_trained4625_' + key]
        if residual:
            y = y-teacher
        axis.plot(times, y[a:b], color=color, linewidth=.9, alpha=.9, label=label.replace('\n', ' '))
    axis.set_title(title, fontsize=12)
    axis.set_xlabel('Absolute source time (ms)')
    axis.set_ylabel('Residual amplitude' if residual else 'Waveform amplitude')
    axis.grid(alpha=.2)
    axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
handles, legend_labels = ax[1, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='outside lower center', ncol=3, fontsize=9, frameon=False)
fig.savefig(root / 'reconstruction-diagnostics.png', dpi=160)
fig.savefig(root / 'reconstruction-diagnostics.svg')
print(root / 'reconstruction-diagnostics.png')
