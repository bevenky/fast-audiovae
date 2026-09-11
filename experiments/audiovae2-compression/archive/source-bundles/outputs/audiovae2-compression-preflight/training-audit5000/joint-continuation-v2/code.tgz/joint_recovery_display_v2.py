"""Continuation dashboard: preserved overview plus separate silence checks."""
import tensorboard_display_v2 as display


def configure():
    del display.METRIC_RUNS['Training progress to 5000 steps (%)']
    display.METRIC_RUNS.update({
        'Training progress to 10000 steps (%)': '11 Training progress - 10000 steps is 100%',
        'Startup near-silence passing (%)': '12 Startup near-silence passing - target 100%',
        'Sustained source silence passing (%)': '13 Sustained source silence passing - target 100%',
        'Interior near-silence passing (%)': '14 Interior near-silence passing - target 100%',
    })
    display.GUIDE += (
        ' Startup near-silence measures the first20ms of teacher-near-silent source starts. '
        'Sustained source silence measures exact-zero prepared input windows after40ms. '
        'Interior near-silence measures teacher-near-silent windows after800ms of source time. '
        'These overlap and must not be added together. Original teacher fidelity limits remain unchanged. '
        'Raw residuals and output-limit excess for each group, including the separate teacher20–40ms '
        'startup transient, appear under Details. All steps are lifetime optimizer steps.'
    )


if __name__ == '__main__':
    configure()
    display.main()
