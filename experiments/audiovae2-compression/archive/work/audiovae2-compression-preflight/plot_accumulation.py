"""Plot saved matched-exposure diagnostics without any model execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
labels = {
    'es_419:train:13461728374156750135.wav': 'Spanish speech',
    'kn_in:train:15096009457771558023.wav': 'Kannada speech',
    'kashmiri:1970324837177077_chunk_1.flac': 'Kashmiri speech',
    'freesound:428921': 'Whistling',
}
data = {n: json.loads((args.directory / f'accumulation{n}' / 'case-trajectories.json').read_text()) for n in (3, 12)}
expected = list(range(0, 1501, 60))
for rows in data.values():
    assert [r['sources_consumed'] for r in rows] == expected

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for axis, (source, label) in zip(axes.flat, labels.items()):
    for n, color in ((3, '#d66022'), (12, '#246bc3')):
        values = [next(c['active_rms_gain'] for c in r['cases'] if c['source_id'] == source) * 100 for r in data[n]]
        axis.plot(expected, values, marker='o', markersize=2.5, linewidth=1.6,
                  color=color, label=f'{n} recordings per update')
    axis.axhline(100, color='#555555', linestyle='--', linewidth=1, label='Teacher level')
    axis.set_title(label, loc='left', fontweight='bold')
    axis.grid(alpha=.18)
    axis.set_xticks([0, 300, 600, 900, 1200, 1500])
    axis.set_xlim(0, 1500)
for axis in axes[:, 0]:
    axis.set_ylabel('Active RMS level (% of teacher)')
for axis in axes[-1, :]:
    axis.set_xlabel('Additional training recordings consumed')
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .94), frameon=False)
fig.suptitle('AudioVAE2 student: matched-exposure accumulation comparison', y=.99, fontsize=15)
fig.text(.5, .012, 'Same initial checkpoint and 1,500 recordings. Matching level does not imply matching waveform quality.',
         ha='center', fontsize=10, color='#444444')
fig.tight_layout(rect=(0, .035, 1, .90))
for extension in ('png', 'svg'):
    fig.savefig(args.directory / f'gain-comparison.{extension}', dpi=170, facecolor='white')
plt.close(fig)
