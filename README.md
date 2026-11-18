# NEAT - Neoantigen Evaluation & Automated Triage

**Machine learning models for neoantigen candidate classification in personalized cancer vaccine design**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![pVACtools](https://img.shields.io/badge/pVACtools-models-teal)](https://pvactools.readthedocs.io/)

NEAT contains models and analysis code for classifying neoantigen candidates as
**Accept**, **Review**, or **Reject**, using expert Immunogenomics Tumor Board
(ITB) evaluations and genomic, expression, and MHC binding features.

The repository has two workflows with separate dependency files and model artifacts:

| Section | Purpose | Instructions | Dependencies |
| --- | --- | --- | --- |
| Manuscript | Reproduce figures, repeated model-development analyses, and a demonstration prediction | [Manuscript README](manuscript/README.md) | [manuscript/requirements.txt](manuscript/requirements.txt) |
| Model development | Encode data, retrain the revised pVACtools 7.2 model, and predict new cases | [Scripts README](model_development/scripts/README.md) | [model_development/requirements.txt](model_development/requirements.txt) |

Run the commands below from the **NEAT repository root**. Use separate Python
environments for the two workflows.

## Repository structure

```text
NEAT/
├── README.md
├── LICENSE
├── manuscript/
│   ├── README.md
│   ├── VALIDATION.md
│   ├── requirements.txt
│   ├── manuscript_final_model/          # Final bundle and best parameters
│   ├── manuscript_interim_model/        # Historical forest, imputer, and encoders
│   ├── data/
│   │   ├── interim_model_training_testing_data/
│   │   ├── missing_data_handeling/      # Strategy 1 and Strategy 2 datasets
│   │   ├── prospective_cohort/          # Ready comparison CSV and provenance
│   │   ├── review_time_analysis_data/
│   │   ├── predict_new_case_data/       # Demo pVACseq TSVs
│   │   └── manuscript_prediction_results/
│   ├── scripts/
│   │   ├── model_development_100iter/
│   │   ├── prospective_cohort_analysis/
│   │   ├── ml_randomforest_interim_model.py
│   │   ├── predict_final_model.py
│   │   └── review_time_analysis.py
│   └── results/                        # Generated analysis outputs
└── model_development/
    ├── requirements.txt
    ├── data/                           # Pre-encoding and encoded tables
    ├── scripts/
    │   ├── README.md
    │   ├── encode_missing.py
    │   ├── train_model.py
    │   └── predict.py                  # 7.2 model with missingness encoding
    └── model/
        ├── pvactools7.2_model/          # Revised model and saved evaluations
        │   ├── random_forest_ds_pvactools7.2.joblib
        │   ├── label_encoders_numpy126.pkl
        │   ├── cv_results.csv
        │   ├── oob_performance.csv
        │   ├── prospective_performance.csv
        │   └── prospective_predictions.csv
        └── pvactools7.0_model/          # Legacy model, imputer, and encoders
            ├── README.md
            ├── rf_downsample_model_numpy126.pkl
            ├── trained_imputer_numpy126.joblib
            ├── label_encoders_numpy126.pkl
            └── archived/               # Previous training/imputation scripts and data
```

## Manuscript analyses

For a fresh environment:

```bash
conda create -n neoantigen_ml python=3.13 pip
conda activate neoantigen_ml
python -m pip install -r manuscript/requirements.txt
```

### Reproduce figures

```bash
python manuscript/scripts/prospective_cohort_analysis/run_all.py
python manuscript/scripts/review_time_analysis.py
```

The prospective workflow reads the ready
[prospective_model_comparison.csv](manuscript/data/prospective_cohort/prospective_model_comparison.csv).
It generates probability comparisons, ROC/PR curves, confusion matrices, and
interim-model threshold sensitivity with and without calibration patients.
Data preparation and original prediction tables are not required.

Outputs are saved under `manuscript/results/prospective_cohort/` (35 PNGs plus
threshold data/manifests) and `manuscript/results/review_time_analysis/` (one ECDF
PNG). See the [prospective instructions](manuscript/scripts/prospective_cohort_analysis/README.md)
for individual analyses.

### Reproduce model development

The [100-iteration workflow](manuscript/scripts/model_development_100iter/README_model_development_CIs.md)
compares four models, peptide and patient splits, and two preprocessing
strategies. It reads the supplied datasets in
`manuscript/data/missing_data_handeling/` and saves empirical 95% intervals and
the combined performance table under `manuscript/results/model_development_CIs/`.

For the two original interim random forests on the fixed training/test split:

```bash
python manuscript/scripts/ml_randomforest_interim_model.py --jobs 2
```

This reads `manuscript/data/interim_model_training_testing_data/` and writes
models, tuning results, and plots under `manuscript/results/randomforest/`.
The fixed split shares patients between training and testing. Full training
searches can take substantial time; see the [manuscript instructions](manuscript/README.md)
for details.

### Predict a new case

```bash
python manuscript/scripts/predict_final_model.py
```

The demo reads three `Test.*.tsv` files from
`manuscript/data/predict_new_case_data/`, adds Strategy 2 missingness indicators,
fills missing predictors with `-1`, and uses
[manuscript_final_model/model_bundle.joblib](manuscript/manuscript_final_model/model_bundle.joblib).
It writes the aggregated prediction TSV under
`manuscript/data/manuscript_prediction_results/`.

For another sample:

```bash
python manuscript/scripts/predict_final_model.py \
  --sample SAMPLE \
  --data-dir /path/to/pvacseq_tables \
  --output-dir /path/to/predictions
```

Supply `<sample>.MHC_I.all_epitopes.aggregated.tsv`,
`<sample>.MHC_I.all_epitopes.tsv`, and
`<sample>.MHC_II.all_epitopes.aggregated.tsv`.
See the [validation notes](manuscript/VALIDATION.md) for checks, training limits,
and the historical final bundle's version warning.

## pVACtools 7.2 model development

In a separate environment, install the model-development dependencies and run:

```bash
python -m pip install -r model_development/requirements.txt
python model_development/scripts/encode_missing.py
python model_development/scripts/train_model.py \
  --input model_development/data/merged_df_encoded.csv \
  --output-dir model_development/model/pvactools7.2_retraining \
  --jobs 2
```

`encode_missing.py` reads `model_development/data/merged_df_pre_imputation.csv`
and writes `merged_df_encoded.csv` in the same directory, adding WT/MT/RNA
missingness indicators and filling supported missing measurements with `-1`.

`train_model.py` tunes a balanced random forest by three-fold row-level
cross-validation, then fits all eligible rows. It saves `model_bundle.joblib`,
best parameters, CV results, OOB diagnostics, and feature importance. Use the
explicit input path above because its built-in input default is outdated. The
output directory must be empty; retraining does not replace the saved 7.2 model
automatically or evaluate the prospective cohort.

The existing
[pvactools7.2_model directory](model_development/model/pvactools7.2_model/)
contains `random_forest_ds_pvactools7.2.joblib` and saved CV, OOB, and prospective
evaluation outputs. See the [scripts README](model_development/scripts/README.md)
for inputs, outputs, and usage of each script.

### Predict with the pVACtools 7.2 model

```bash
python model_development/scripts/predict.py
```

This reads the bundled manuscript demo TSVs and writes
`model_development/data/Test.MHC_I.all_epitopes.aggregated.ML_predict.tsv`.
It applies `pvactools7.2_model/label_encoders_numpy126.pkl`, adds WT/MT/RNA
missingness indicators, replaces missing predictors with `-1`, and scores using
`random_forest_ds_pvactools7.2.joblib`. Use `--sample`, `--data-dir`, and
`--output-dir` for another case. The output TSV format is preserved.

### Legacy pVACtools 7.0 artifacts

The previous model, imputer, and encoders remain in
[pvactools7.0_model/](model_development/model/pvactools7.0_model/README.md).
Previous `impute_missing.py` and `old_train.py` scripts are retained under
`pvactools7.0_model/archived/` as historical references. The current predictor
uses missingness encoding and the 7.2 bundle.

For pVACseq/pVACview integration, consult the
[pVACtools documentation](https://pvactools.readthedocs.io/).

## Dataset

The development analysis uses 1,237 eligible candidates from 21 patients
(from 1,324 rows before excluding development cases labeled Review). The
prospective cohort contains 706 candidates from 12 patients, for 1,943 analysis
candidates across 33 patients.

The cohorts span eight cancer types from three clinical trials at Washington
University School of Medicine:

| Trial | Cancer type | Patients |
| --- | --- | --- |
| NCT05111353 | Pancreatic cancer | 14 |
| NCT03606967 | Metastatic TNBC | 11 |
| NCT05741242 | Basket trial (multiple types) | 8 |

The supplied feature tables use anonymized `pt_XX` patient IDs. They include ITB
labels and genomic, expression, and MHC prediction features; Strategy 2 adds
three missingness indicators to the 72 predictors. The patient-ID mapping is
not included and is not required to run the manuscript analyses. Clinical
genomic source data are not included due to IRB restrictions.

## Citation

If you use pVACml or the associated model in your work, please cite:

```
Yao J, Singhal K, Kiwala S, et al.
Automating immunogenomic tumor board decision-making for neoantigen cancer vaccine design.
[Journal] (2025). DOI: [pending]
```

---

## Related resources

- [pVACtools documentation](https://pvactools.readthedocs.io/)
- [pVACtools ML model artifacts on GitHub](https://github.com/griffithlab/pVACtools/tree/master/pvactools/supporting_files/ml_model_artifacts)
- [pVACview interface](https://pvactools.readthedocs.io/en/latest/pvacview.html)
- [ImmunoNX pipeline](https://github.com/griffithlab/ImmunoNX_protocol)

---

## Questions and contributions

For questions about the model or codebase, please [open an issue](https://github.com/jyao36/pVACml/issues). For questions related to the pVACtools integration, see the [pVACtools GitHub](https://github.com/griffithlab/pVACtools).
