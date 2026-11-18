"""Save ROC/PR and confusion-matrix PNGs from the unified prospective CSV."""

from common import arguments, load_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)

CLASSES = ["Accept", "Reject", "Review"]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments(__doc__, "binary_roc_pr_curves")
    data = load_data(args.input)
    for cohort, selected in [("all_samples", data),
                             ("excluding_calibration", data.loc[~data.is_calibration])]:
        for model, prediction, probability in [
            ("interim", "interim_prediction", "interim_accept_probability"),
            ("final", "final_prediction", "final_accept_probability")]:
            folder = args.output_dir / cohort / model
            folder.mkdir(parents=True, exist_ok=True)
            cm = confusion_matrix(selected.Evaluation, selected[prediction], labels=CLASSES)
            accuracy = float(np.trace(cm) / cm.sum())
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{model.capitalize()} model\nn = {len(selected)}; accuracy = {accuracy:.3f}")
            save(fig, folder / "confusion_matrix")
            for rule in ["true_binary", "notebook_filtered"]:
                mask = selected.Evaluation.isin(["Accept", "Reject"])
                if rule == "notebook_filtered":
                    mask &= selected[prediction].isin(["Accept", "Reject"])
                binary = selected.loc[mask]
                y = binary.Evaluation.eq("Accept").astype(int)
                score = binary[probability]
                if y.nunique() != 2:
                    raise ValueError("ROC/PR evaluation requires both classes")
                fpr, tpr, _ = roc_curve(y, score)
                precision, recall, _ = precision_recall_curve(y, score)
                auroc = float(roc_auc_score(y, score))
                ap = float(average_precision_score(y, score))
                for kind in ["roc", "pr", "roc_pr"]:
                    fig, axes = plt.subplots(1, 2 if kind == "roc_pr" else 1,
                                             figsize=(10, 4.5) if kind == "roc_pr" else (5.5, 4.5))
                    axes = np.atleast_1d(axes)
                    for ax, curve in zip(axes, ["roc", "pr"] if kind == "roc_pr" else [kind]):
                        if curve == "roc":
                            ax.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.3f}")
                            ax.set(xlabel="False positive rate", ylabel="True positive rate")
                        else:
                            ax.plot(recall, precision, linewidth=2, label=f"Average precision = {ap:.3f}")
                            ax.set(xlabel="Recall", ylabel="Precision")
                        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
                        ax.legend(loc="lower right" if curve == "roc" else "lower left", fontsize=9)
                        ax.spines[["top", "right"]].set_visible(False)
                    fig.suptitle(f"{model.capitalize()} model: Accept versus Reject (n = {len(binary)})")
                    save(fig, folder / f"{kind}_{rule}")
    print(f"Saved PNG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
