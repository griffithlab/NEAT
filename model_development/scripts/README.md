# pVACtools 7.2 model workflow

The current encoding and training scripts support the pVACtools 7.2 model
workflow. The saved model is
`model_development/model/pvactools7.2_model/random_forest_ds_pvactools7.2.joblib`.
`predict.py` still uses the legacy pVACtools 7.0 workflow.

Run commands below from the **NEAT repository root**.

| Script | What it does | Input | Generated files |
| --- | --- | --- | --- |
| `encode_missing.py` | Adds `WT_missing`, `MT_missing`, and `RNA_missing`; fills supported missing measurements with `-1`. Fits no imputer. | `model_development/data/merged_df_pre_imputation.csv` | `model_development/data/merged_df_encoded.csv` |
| `train_model.py` | Excludes Review, treats Pending as Reject, integer-encodes categories, tunes a balanced random forest by 3-fold row-level CV/AUROC, then fits all eligible rows. | Encoded CSV supplied with `--input` below | `model_bundle.joblib`, `best_params.json`, CV results/membership, OOB metrics/predictions, feature importance, settings, and completion record in `--output-dir` |
| `predict.py` **(legacy 7.0)** | Merges three pVACseq TSVs, applies saved encoders/imputer, and predicts Accept/Review/Reject. Does not use the 7.2 model or missingness indicators. | Demo `Test.*.tsv` files in `manuscript/data/predict_new_case_data/`; model, imputer, and encoders in `model_development/model/pvactools7.0_model/` | `model_development/data/Test.MHC_I.all_epitopes.aggregated.ML_predict.tsv` |

## Encode and train

```bash
python -m pip install -r model_development/requirements.txt
python model_development/scripts/encode_missing.py
python model_development/scripts/train_model.py \
  --input model_development/data/merged_df_encoded.csv \
  --output-dir model_development/model/pvactools7.2_retraining \
  --jobs 2
```

Use the explicit training input above because the script's built-in input path
is outdated. The output directory must be empty. Training writes a new bundle;
it does not replace the saved 7.2 model automatically. The full grid has 3,750
candidates and 11,250 CV fits. OOB metrics are training diagnostics; the script
does not evaluate the prospective cohort or generate plots.

`encode_missing.py` accepts `--input` and `--output`. Use `--help` on either
encoding or training script for options.

## Legacy prediction usage

```bash
python model_development/scripts/predict.py
```

This runs the 7.0 demo only. Its sample, model directory, artifact version, and
output directory are configured at the bottom of `predict.py`; it has no CLI
options for those settings.
