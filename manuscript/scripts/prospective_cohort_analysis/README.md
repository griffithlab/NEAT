# Prospective cohort: interim versus final models

All analyses read the single supplied table:
`manuscript/data/prospective_cohort/prospective_model_comparison.csv`.
It contains saved predictions and true ITB evaluations for 706 candidates from
12 patients. The four threshold-calibration patients contribute 170 candidates;
excluding them leaves 536 candidates from eight patients.

The CSV preserves all original prospective reference features and annotations.
Its original `python_Accept_pred_prob` and `python_Evaluation_pred` columns are
renamed `interim_accept_probability` and `interim_prediction`; the final-model
probability, final-model label, and calibration flag are appended. It contains
29 columns, with the original reference row and column order retained.

No model training, imputation, feature matrix, or saved model is required to
reproduce these figures. Each script resolves default paths from its own
location, so execution is independent of the current working directory.

## Environment and one-command reproduction

The commands below assume the NEAT root. Use the shared Python 3.13 manuscript
environment and its single requirements file:

```bash
conda activate neoantigen_ml
python -m pip install -r manuscript/requirements.txt
python manuscript/scripts/prospective_cohort_analysis/run_all.py
```

See [`manuscript/README.md`](../../README.md) for fresh environment setup. The
existing `neoantigen_ml` environment matches the pinned versions. All output
goes under `manuscript/results/prospective_cohort/`.
Rerunning replaces the generated figures and tables in that output directory.
To use custom paths:

```bash
python manuscript/scripts/prospective_cohort_analysis/run_all.py \
  --input path/to/prospective_model_comparison.csv \
  --output-dir path/to/results
```

The input CSV remains unchanged. The probability-comparison and ROC/PR workflows
generate PNG figures only. The threshold analysis also saves its curve data and
cohort-specific run manifests.

## Individual analyses

```bash
python manuscript/scripts/prospective_cohort_analysis/compare_probabilities.py
python manuscript/scripts/prospective_cohort_analysis/evaluate_models.py
python manuscript/scripts/prospective_cohort_analysis/threshold_sensitivity.py
```

Each accepts `--input` and `--output-dir`; use `--help` for details.

### Probability comparison and per-class tables

`compare_probabilities.py` saves five PNGs to `probability_comparison/`:

- `probability_correlation.png`: interim versus final Accept probabilities,
  colored by true ITB evaluation, with Pearson and Spearman correlations.
- `interim_model_confusion_matrix.png` and `final_model_confusion_matrix.png`:
  confusion matrices for all samples, including calibration patients.
- `interim_model_one_vs_rest_metrics.png` and `final_model_one_vs_rest_metrics.png`:
  sensitivity, specificity, precision, and F1 tables, showing all samples and
  the cohort excluding calibration patients in separate panels.

### Binary ROC/PR curves and confusion matrices

`evaluate_models.py` writes to `binary_roc_pr_curves/`, organized by cohort
(`all_samples`, `excluding_calibration`) and model (`interim`, `final`).
Each of the four folders contains only seven PNG figures: a three-class
confusion matrix and separate/combined ROC and PR plots for both filters below.
Run it from the NEAT root:

```bash
python manuscript/scripts/prospective_cohort_analysis/evaluate_models.py
```

Both historical binary filters are preserved:

| Suffix | Evaluation population |
| --- | --- |
| `true_binary` | All true Accept/Reject candidates, including candidates predicted Review. The same population is used for both models. |
| `notebook_filtered` | True Accept/Reject candidates also predicted Accept/Reject. This produces a model-specific subset. |

`true_binary` is the population used by the interim/final ROC/PR figures in
`Review/new_figures_tables`. It has 618 candidates for all patients and 474
after excluding calibration patients. True Review cases remain in the
three-class confusion matrices, but are excluded from the binary curves.
The reported PR summary is average precision, matching the original code.
AUROC and average precision are displayed directly on the plots.

### Interim-model threshold sensitivity

`threshold_sensitivity.py` reproduces the interim probability threshold sweep
in `threshold_sensitivity/error_curves_all_samples.png`, with all 706 candidates.
The calibration-excluded figure is `error_curves_excluding_calibration.png` in
the same directory. Curve data, cohort membership, and run manifests also use
`all_samples` or `excluding_calibration` filename suffixes.

The two boundaries are swept separately:

- Red curve: fraction of true expert Accept candidates with probability **< t**,
  which would be assigned Reject at rejection threshold t.
- Blue curve: fraction of true expert Reject candidates with probability **> t**,
  which would be assigned Accept at acceptance threshold t.

These are percentages of the respective expert class, rather than percentages
of the full cohort. True Review cases are excluded from the denominators.
The red curve does not count true Accept candidates assigned Review as false
Rejects. The 0.30 and 0.55 annotations display the existing decision boundaries;
the script performs no threshold selection on this cohort. Exact scores and
adjacent floating-point values preserve the original strict inequalities.

To produce the additional calibration-excluded threshold plot:

```bash
python manuscript/scripts/prospective_cohort_analysis/threshold_sensitivity.py \
  --cohort excluding_calibration
```

`run_all.py` generates both threshold cohorts automatically in the shared
`threshold_sensitivity/` directory.

## Figure locations

Paths below are relative to `manuscript/results/prospective_cohort/`:

| Existing figure in `Review/new_figures_tables` | Reproduced output |
| --- | --- |
| `probability_correlation.png` | `probability_comparison/probability_correlation.png` |
| `interim_model_confusion_matrix.png` | `probability_comparison/interim_model_confusion_matrix.png` |
| `final_model_confusion_matrix.png` | `probability_comparison/final_model_confusion_matrix.png` |
| `rf_ds_external_eval_conf_matrix_no_cal.png` | `binary_roc_pr_curves/excluding_calibration/interim/confusion_matrix.png` (same counts; consistent model labels/style) |
| `interim_model_roc_pr_true_binary.png` | `binary_roc_pr_curves/all_samples/interim/roc_pr_true_binary.png` |
| `final_model_roc_pr_true_binary.png` | `binary_roc_pr_curves/all_samples/final/roc_pr_true_binary.png` |
| `error_curves_only.png` | `threshold_sensitivity/error_curves_all_samples.png` |

Confusion matrices and ROC/PR figures are exported at 600 dpi; the probability
scatter is 300 dpi, matching the source settings. Per-class performance tables
are included as PNGs in `probability_comparison/`.

## Ready input and validation

Use the supplied unified CSV directly. Data preparation is maintained outside
NEAT in the original analysis working directory; no preparation step or original
prediction tables are required to reproduce these analyses. See the data README
for provenance, column definitions, and the four calibration patient IDs.

All original reference feature/annotation columns and their missing values are
retained. Finite-value validation applies to the two prediction probabilities.

The analysis loader validates unique keys, finite probabilities in [0, 1],
recognized true/predicted classes, patient-consistent calibration flags, and
agreement of final predictions with the 0.30/0.55 rule. It uses round-trip float
parsing to preserve exported probability values for the exact threshold sweep.

## Validation against the historical analysis

The supplied table preserves both probabilities and true evaluations exactly
for all 706 matched candidates. Running the full workflow reproduced all eight
historical ROC/PR summary rows, all four confusion matrices, eight binary cohort
membership lists, sixteen ROC/PR coordinate tables, and four per-class performance
tables. Numeric comparisons used a tolerance of 1e-12. The interim threshold
grid matched exactly, and both complete error curves matched within 1e-12.
Strict boundary behavior (0.30 and 0.55 both map to Review) and rejection of
invalid labels/probabilities, duplicate keys, and invalid calibration flags were
also checked. Representative output figures were visually inspected.

Useful reproduction targets are Pearson r = 0.9481727288833885, Spearman rho =
0.9421784903762999, all-sample interim accuracy = 0.6572237960339944, and final
accuracy = 0.6968838526912181. The all-sample interim threshold figure shows
0.0% expert-Accept assigned Reject at 0.30 and 11.8% expert-Reject assigned
Accept at 0.55. Different plotting library versions can change raster rendering
while leaving these numeric results unchanged.

The entire workflow was subsequently rerun in the shared `neoantigen_ml`
environment (Python 3.13.2 and the exact `manuscript/requirements.txt` versions).
The correlations, all eight ROC/PR summary rows, and all four per-class
performance tables matched the historical references within 1e-12. The
threshold manifests record the shared package versions.

The ROC/PR script was subsequently simplified to PNG-only output. All 28
regenerated figures matched the previous PNG pixel data exactly.
