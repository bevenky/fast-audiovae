"""Plot saved diagnostic waveforms only; does not load or execute a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
records = json.loads((args.directory / 'waveform-snippets.json').read_text())
data = np.load(args.directory / 'waveform-snippets.npz')
cases = [
    ('es_419:train:13461728374156750135.wav', 'first_near_silence', 'Spanish near-silence', None),
    ('kn_in:train:15096009457771558023.wav', 'scored_start', 'Kannada scored start', 60),
    ('freesound:428921', 'teacher_peak', 'Whistling near teacher peak', 12),
]
colors = ['#2175b5', '#de7328', '#b64265']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for row, (source, region, label, duration_ms) in enumerate(cases):
    record = next(r for r in records if r['source_id'] == source and r['interval'] == region)
    key = record['key']
    target = data[key + '_teacher']
    left, right = 0, len(target)
    if duration_ms is not None:
        count = min(len(target), round(duration_ms * record['sample_rate'] / 1000))
        left = max(0, (len(target) - count) // 2) if region == 'teacher_peak' else 0
        right = left + count
    time_ms = (np.arange(left, right) + record['absolute_source_samples'][0]) / record['sample_rate'] * 1000
    for col, variants in enumerate([
        [('teacher_ablate_stage2_ru1', 'Remove stage 2 mixer contribution'),
         ('teacher_ablate_stage3_up', 'Remove stage 3 upsampler contribution'),
         ('teacher_ablate_stage4_up', 'Remove stage 4 upsampler contribution')],
        [('plain', 'Plain sliced initialization'), ('oracle_first', 'Exact first contribution restored'),
         ('fitted', 'First contribution fitted from retained channels')],
    ]):
        axis = axes[row, col]
        for (variant, name), color in zip(variants, colors):
            values = data[key + '_' + variant][left:right]
            assert len(values) == len(time_ms) and np.isfinite(values).all()
            axis.plot(time_ms, values, label=name, color=color, linewidth=.85, alpha=.85)
        axis.plot(time_ms, target[left:right], label='Original teacher', color='#171717', linewidth=1.0, zorder=5)
        axis.set_title(label, loc='left', fontsize=11)
        axis.set_xlabel('Time in original recording (ms)')
        axis.set_ylabel('Waveform amplitude')
        axis.grid(alpha=.16)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        if row == 0:
            handles, names = axis.get_legend_handles_labels()
            fig.legend(handles, names, loc='upper left', bbox_to_anchor=(.075 if col == 0 else .555, .935),
                       fontsize=9, frameon=False)
fig.suptitle('What removed channel contributions do to actual waveforms', fontsize=16, y=.995)
fig.text(.255, .955, 'Interventions in the original teacher', ha='center', fontsize=12, fontweight='bold')
fig.text(.745, .955, 'Interventions in the untrained pruned student', ha='center', fontsize=12, fontweight='bold')
fig.text(.5, .012, 'These are initialization diagnostics, not the trained 5,000-step model. Each panel has its own amplitude scale.',
         ha='center', color='#444444', fontsize=10)
fig.subplots_adjust(top=.805, bottom=.085, hspace=.55, wspace=.25)
for extension in ('png', 'svg'):
    fig.savefig(args.directory / ('channel-waveforms.' + extension), dpi=180, facecolor='white')
plt.close(fig)
