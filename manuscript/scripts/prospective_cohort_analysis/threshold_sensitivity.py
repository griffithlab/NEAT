"""Sweep interim-model probability thresholds without fitting or threshold selection."""

from common import arguments, load_data, write_manifest
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'threshold_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'


def main():
    global OUT
    args = arguments(__doc__, "threshold_sensitivity")
    OUT = args.output_dir
    all_data = load_data(args.input)
    data = all_data if args.cohort == "all_samples" else all_data.loc[~all_data.is_calibration]
    scores = data.interim_accept_probability.to_numpy(dtype=float)
    if not (np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError('Invalid Accept probabilities')
    positive = scores[data.Evaluation.eq('Accept')]
    negative = scores[data.Evaluation.eq('Reject')]
    if not len(positive) or not len(negative):
        raise ValueError('Both reference classes are required')
    # Include exact scores and the immediately following floats to preserve
    # strict inequalities: Reject p < t; Accept p > t.
    thresholds = np.unique(np.r_[0., 1., .30, .55, scores,
                                np.nextafter(scores[scores < 1], 1)])
    fnr = np.array([(positive < t).mean() for t in thresholds]) * 100
    far = np.array([(negative > t).mean() for t in thresholds]) * 100
    baseline_fnr = (positive < .30).mean() * 100
    baseline_far = (negative > .55).mean() * 100
    pd.DataFrame({"threshold": thresholds, "true_accept_predicted_reject_pct": fnr,
                  "true_reject_predicted_accept_pct": far}).to_csv(OUT / f"threshold_error_rates_{args.cohort}.csv", index=False)
    data[["patient_id", "ID", "Evaluation"]].to_csv(OUT / f"cohort_membership_{args.cohort}.csv", index=False)
    write_manifest(args, "threshold_sensitivity", all_data,
                   filename=f"run_manifest_{args.cohort}.json",
                   cohort=args.cohort, evaluated_rows=len(data), evaluated_patients=data.patient_id.nunique(),
                   expert_accept_count=len(positive), expert_reject_count=len(negative),
                   reject_threshold=.30, accept_threshold=.55,
                   baseline_false_negative_pct=float(baseline_fnr), baseline_false_accept_pct=float(baseline_far),
                   definition="Separate strict sweeps: fraction of expert Accept with p < t; fraction of expert Reject with p > t")
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 10, 'axes.linewidth': .8,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9}):
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        red, blue = '#b2182b', '#2166ac'
        ax.step(thresholds, fnr, where='post', color=red, lw=1.8,
                label='True Accept → predicted Reject')
        ax.step(thresholds, far, where='post', color=blue, lw=1.8,
                label='True Reject → predicted Accept')
        for t, color, value in [(.30, red, baseline_fnr), (.55, blue, baseline_far)]:
            ax.axvline(t, color=color, lw=1, ls=(0, (2, 3)), alpha=.85)
            ax.plot(t, value, 'o', color=color, ms=5, zorder=4, clip_on=False)
        ax.annotate(f'Reject threshold = 0.30\nFalse-negative rate = {baseline_fnr:.1f}%',
                    xy=(.30, baseline_fnr), xytext=(.07, 22), fontsize=9,
                    arrowprops={'arrowstyle': '-', 'color': red, 'lw': 1},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
        ax.annotate(f'Accept threshold = 0.55\nFalse-accept rate = {baseline_far:.1f}%',
                    xy=(.55, baseline_far), xytext=(.43, 69), fontsize=9,
                    arrowprops={'arrowstyle': '-', 'color': blue, 'lw': 1},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
        ax.set(xlim=(0, 1), ylim=(-2, 102), xlabel='Decision threshold (probability)',
               ylabel='Fraction of the respective expert class (%)')
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', width=.8)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), frameon=False, fontsize=9)
        fig.text(.5, .97, 'Prospective threshold sensitivity', ha='center', va='top', fontsize=12)
        fig.text(.5, .025,
                 f'{len(data):,} candidates across {data.patient_id.nunique()} samples; each decision boundary is swept separately.\n'
                 f'Denominators: {len(positive)} expert-Accept and {len(negative)} expert-Reject candidates.',
                 ha='center', va='bottom', fontsize=8)
        fig.subplots_adjust(left=.12, right=.98, top=.90, bottom=.30)
        OUT.mkdir(exist_ok=True)
        path = OUT / f'error_curves_{args.cohort}.png'
        fig.savefig(path, dpi=600, facecolor='white')
        plt.close(fig)
    print(f'Saved {path} (4320 × 3180 pixels, 600 dpi)')
    print(f'Baseline FNR: {baseline_fnr:.1f}%; false-accept rate: {baseline_far:.1f}%')


if __name__ == '__main__':
    main()
