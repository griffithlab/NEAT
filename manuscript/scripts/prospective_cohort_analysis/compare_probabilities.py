"""Save probability-comparison and three-class performance PNGs from one CSV."""

from common import arguments, load_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

KEYS = ["patient_id", "ID"]
CLASS_LABELS = ["Accept", "Reject", "Review"]


def evaluate_models(joined, output_dir):
    """Notebook-style three-class evaluation on all matched reference records."""
    records = joined[KEYS + ["Evaluation", "is_calibration"]].copy()
    records["interim_prediction"] = joined.interim_prediction
    records["final_prediction"] = joined.final_prediction
    for name, prediction in [
        ("Interim model", records.interim_prediction),
        ("Final model", records.final_prediction),
    ]:
        if not prediction.isin(CLASS_LABELS).all():
            raise ValueError(f"Unexpected predicted labels for {name}")
        slug = name.lower().replace(" ", "_")
        cm = confusion_matrix(joined.Evaluation, prediction, labels=CLASS_LABELS)
        accuracy = float(np.trace(cm) / cm.sum())
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nn = {len(joined)}; accuracy = {accuracy:.3f}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{slug}_confusion_matrix.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    render_model_tables(records, output_dir)


def render_model_tables(records, output_dir):
    """Separate model figures with identical notebook-defined cohort exclusions."""
    columns = ["Class", "Sensitivity", "Specificity", "Precision", "F1"]
    cohorts = [
        ("calibration_independent", "Calibration-independent", records.loc[~records.is_calibration]),
        ("all_samples", "All samples", records),
    ]
    for name, prediction in [("Interim model", "interim_prediction"), ("Final model", "final_prediction")]:
        slug = name.lower().replace(" ", "_")
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
        for ax, (cohort, label, selected) in zip(axes, cohorts):
            cm = confusion_matrix(selected.Evaluation, selected[prediction], labels=CLASS_LABELS)
            assert cm.sum() == len(selected)
            rows = []
            for i, cls in enumerate(CLASS_LABELS):
                tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
                fp = int(cm[:, i].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
                divide = lambda a, b: a / b if b else 0.0
                rows.append({"Class": cls, "Sensitivity": divide(tp, tp + fn),
                             "Specificity": divide(tn, tn + fp), "Precision": divide(tp, tp + fp),
                             "F1": divide(2 * tp, 2 * tp + fp + fn)})
            table = pd.DataFrame(rows, columns=columns)
            for col in columns[1:]:
                table[col] = table[col].map(lambda v: f"{v:.3f}")
            ax.axis("off")
            ax.set_title(f"{label}: {selected.patient_id.nunique()} samples, {len(selected)} records", fontsize=12)
            artist = ax.table(cellText=table.values, colLabels=columns,
                              cellLoc="center", loc="center")
            artist.auto_set_font_size(False); artist.set_fontsize(11); artist.scale(1, 1.8)
            for (row, col), cell in artist.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#dce8f2"); cell.set_text_props(weight="bold")
        fig.suptitle(f"{name}: one-versus-rest performance", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output_dir / f"{slug}_one_vs_rest_metrics.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def main():
    args = arguments(__doc__, "probability_comparison")
    output_dir = args.output_dir
    joined = load_data(args.input)
    x = joined.interim_accept_probability.to_numpy(dtype=float)
    y = joined.final_accept_probability.to_numpy(dtype=float)
    colors = {"Accept": "green", "Review": "orange", "Reject": "red"}
    if joined.Evaluation.isna().any() or not set(joined.Evaluation).issubset(colors):
        raise ValueError("Unexpected manuscript Evaluation labels")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Correlation undefined for constant probabilities")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    evaluate_models(joined, output_dir)
    fig, ax = plt.subplots(figsize=(6.5, 6.3))
    for label, color in colors.items():
        selected = joined.Evaluation.eq(label)
        ax.scatter(x[selected], y[selected], c=color, label=f"{label} (n={selected.sum()})",
                   s=22, alpha=0.65, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    for threshold, label in [(0.3, "Reject threshold"), (0.55, "Accept threshold")]:
        ax.axvline(threshold, color="black", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(threshold, color="black", linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(label, xy=(threshold, 0.98),
                    xycoords=ax.get_xaxis_transform(), xytext=(3, 0),
                    textcoords="offset points", rotation=90,
                    rotation_mode="anchor", ha="right", va="top", fontsize=7)
        ax.annotate(label, xy=(0.98, threshold),
                    xycoords=ax.get_yaxis_transform(), xytext=(0, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=7)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel="Interim model Accept probability", ylabel="Final model Accept probability",
           title="Final versus Interim model predictions")
    ax.set_aspect("equal")
    ax.set_title("Final versus Interim model predictions", pad=10)
    ax.text(0.03, 0.97, f"n = {len(joined)}\nPearson r = {pearson.statistic:.3f}\nSpearman ρ = {spearman.statistic:.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=7, title="True ITB evaluation", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.04, "Reject: probability < 0.30; Review: 0.30–0.55; Accept: probability > 0.55", ha="center", fontsize=8)
    fig.text(0.5, 0.015, "Source: unified prospective input; all matched rows", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "probability_correlation.png", dpi=300)
    plt.close(fig)
    print(f"Saved PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
