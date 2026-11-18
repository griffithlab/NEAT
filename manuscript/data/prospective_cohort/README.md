# Unified prospective model comparison input

`prospective_model_comparison.csv` is the single input for all scripts in
`manuscript/scripts/prospective_cohort_analysis/`. Each row is a prospective
candidate, uniquely identified by `(patient_id, ID)`.

Patient IDs use the anonymized `pt_XX` values assigned by the workspace's
`data/meta_manual.csv` mapping. The mapping itself remains outside NEAT.

The table retains all 26 columns from the original `prospective_test_set.csv`
in their original order, renames its two prediction columns as described below,
and appends `final_accept_probability`, `final_prediction`, and `is_calibration`.
The resulting CSV has 29 columns. Original feature and annotation columns such
as `TSL`, `DNA VAF`, `RNA Expr`, `Allele`, `Gene`, `Best Peptide`, `Best Transcript`,
`Tier`, and `AA Change` are retained alongside the analysis columns below.

| Column | Meaning |
| --- | --- |
| `patient_id` | Patient/sample identifier shared across both models. |
| `ID` | Candidate identifier; the combination with patient_id is unique. |
| `Evaluation` | True expert ITB label: Accept, Reject, or Review. |
| `interim_accept_probability` | Saved manuscript/interim model probability of Accept. |
| `interim_prediction` | Saved manuscript/interim three-class prediction. |
| `final_accept_probability` | Saved final model probability of Accept. |
| `final_prediction` | Final three-class prediction: Reject if p < 0.30; Accept if p > 0.55; Review otherwise, including both boundaries. |
| `is_calibration` | True if the patient was included in threshold calibration, False otherwise. |

The interim labels are preserved directly from `python_Evaluation_pred`.
The final labels use the threshold rule from the historical comparison script;
they represent the three-class manuscript evaluation, rather than the binary
classifier's default `predict()` output. The original prospective features and
annotations are retained for reference; the analyses use the saved predictions,
true evaluations, keys, and calibration flag without fitting a model.

## Cohort

- 706 candidates from 12 patients: 142 expert Accept, 476 Reject, 88 Review.
- 170 candidates from four calibration patients.
- 536 candidates from eight other patients: 100 Accept, 374 Reject, 62 Review.

The historical calibration patients are `pt_33`, `pt_34`,
`pt_35`, and `pt_36`. All analysis filtering uses the CSV flag;
the IDs are needed only for rebuilding the historical table.

## Provenance

The historical interim reference is
`Review/additional_analysis/imputation_test/train_on_all/predict_on_prospective_set/prospective_test_set.csv`
in the analysis working directory outside NEAT. Its true labels, probabilities,
and predicted labels match the supplied table on all 706 candidates after
applying the patient-ID mapping. The final probabilities come from
`Review/additional_analysis/imputation_test/train_on_all/predict_on_prospective_set/prospective_predictions.csv`.
That file's `Evaluation` column contains Pending placeholders and is not used
as truth. The final model used Strategy 2 encoding and was trained on all
1,237 eligible development observations. Its saved prediction manifest records
`max_features=67`, `n_estimators=101`, and 75 predictors.

`prospective_model_comparison.manifest.json` records hashes for both original
sources and the exported table, cohort counts, and label definitions. The
preparation script, `prepare_input.py`, lives alongside those historical sources
outside NEAT. It joins matching unique keys and applies the patient-ID mapping.
No rows are removed during table construction. NEAT includes the ready CSV;
reproducing the analyses requires no preparation step or original source files.
