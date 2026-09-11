"""Reuse the verified color-per-metric display for the bounded recovery run."""
import tensorboard_display_v2 as display

if __name__ == '__main__':
    del display.METRIC_RUNS['Training progress to 5000 steps (%)']
    display.METRIC_RUNS['Recovery progress to 1000 updates (%)'] = '11 Recovery progress - 1000 updates is 100%'
    display.main()
