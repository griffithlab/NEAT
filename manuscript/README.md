# Reproduce the manuscript analyses

Run the following commands from the NEAT repository root.

## Setup

```bash
conda create -n neoantigen_ml python=3.13 pip
conda activate neoantigen_ml
python -m pip install -r manuscript/requirements.txt
```

All manuscript workflows use this requirements file. The supplied datasets
already contain the inputs needed for analysis; no private source directories
or patient-ID mapping file are required. Patient IDs use anonymized `pt_XX`
values. Scripts resolve default paths relative to their own locations and save
plots without opening a graphical window.

## Reproduce plots

```bash
python manuscript/scripts/prospective_cohort_analysis/run_all.py
python manuscript/scripts/review_time_analysis.py
```

Outputs:

- `manuscript/results/prospective_cohort/`: 35 PNGs covering probability
  comparisons, confusion matrices, ROC/PR curves, and both threshold cohorts.
  Threshold curve data and cohort membership are also saved.
- `manuscript/results/review_time_analysis/itb_review_time_ecdf_plot.png`:
  pre/post-prediction review times. Statistical test results print to the console.

See the [prospective analysis instructions](scripts/prospective_cohort_analysis/README.md)
for individual plots and cohort definitions.

## Reproduce training analyses

For four models, two datasets, and 100 repeated peptide/patient splits, follow
[these commands](scripts/model_development_100iter/README_model_development_CIs.md).

For the original two interim random forests on the supplied fixed split:

```bash
python manuscript/scripts/ml_randomforest_interim_model.py --jobs 2
```

This saves models, parameters, feature importance, and 12 PNGs under
`manuscript/results/randomforest/`. It leaves the supplied model snapshots
unchanged. The full search uses 7,200 candidates and three folds per model and
can take a long time. This fixed split shares patients between training and
testing; use the repeated-development workflow for patient-aware evaluation.
Use `--help` for smaller grids or custom output paths. Smaller grids validate
execution but do not reproduce the full manuscript training results.

## Demonstration prediction

```bash
python manuscript/scripts/predict_final_model.py
```

Uses the supplied final model and demo TSVs, and writes an aggregated TSV under
`manuscript/data/manuscript_prediction_results/`. For another sample, pass
`--sample NAME --data-dir /path/to/inputs --output-dir /path/to/results`.

The historical final bundle emits a scikit-learn version warning in the shared
environment. See the [final-model instructions](manuscript_final_model/README.md)
for details about the saved bundle.
