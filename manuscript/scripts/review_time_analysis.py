"""Save the pre/post-prediction ITB review-time ECDF and print statistical tests."""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "neat-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu

# Data lives next to this script in the repo: manuscript/data/review_time_analysis_data/
# Resolving from __file__ works for any clone path and does not depend on the shell cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "data" / "review_time_analysis_data"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=_SCRIPT_DIR.parent / "results" / "review_time_analysis")
    args = parser.parse_args()
    plot(args.data_dir, args.output_dir)


def plot(data_dir, output_dir):
    pre_merged = pd.read_csv(data_dir / "pre_merged.csv")
    post_merged = pd.read_csv(data_dir / "post_merged.csv")

    pre_times = pd.to_numeric(pre_merged['ITB_review_time_seconds'], errors='raise').dropna()
    post_times = pd.to_numeric(post_merged['ITB_review_time_seconds'], errors='raise').dropna()

    if any(times.empty or not times.between(0, float('inf'), inclusive='left').all()
           for times in [pre_times, post_times]):
        raise ValueError('Require finite, nonnegative review times in both cohorts.')
    print(f'Review times: pre n={len(pre_times)}, post n={len(post_times)}')

    ## Mann-Whitney U test
    u_stat, u_p = mannwhitneyu(pre_times, post_times, alternative='two-sided')
    print(f"Mann-Whitney U: p={u_p}")


    ## ECDF plot

    ecdf_df = pd.DataFrame({
        'time_seconds': pd.concat([pre_times, post_times], ignore_index=True),
        'Group': (['Pre-predictions'] * len(pre_times)) + (['Post-predictions'] * len(post_times))
    })

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.ecdfplot(data=ecdf_df, x='time_seconds', hue='Group', linewidth=2, ax=ax)

    ax.set_xlabel('ITB review time (seconds)', fontsize=25)
    ax.set_ylabel('ECDF', fontsize=25)
    #ax.set_title('ECDF of ITB review time: Pre vs Post', fontsize=25)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(None)
        plt.setp(leg.get_texts(), fontsize=18)
        # Move the legend box slightly lower by adjusting bbox_to_anchor and loc
        leg.set_bbox_to_anchor((0.78, 0.95))  # (x, y) - decrease y to move lower
        leg.set_loc('upper left')

    ks_stat, ks_p = ks_2samp(pre_times, post_times)
    print(f'KS test: statistic={ks_stat}, p={ks_p}')
    ax.text(
        0.98, 0.02,
        f'KS test: p = {ks_p:.3f}',
        fontsize=20,
        ha='right',
        va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7)
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "itb_review_time_ecdf_plot.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
