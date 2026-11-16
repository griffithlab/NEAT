# Final manuscript model

- `model_bundle.joblib`: trained model, feature order, categorical mappings, and training settings.
- `best_params.json`: selected parameters (`max_features=67`, `n_estimators=101`).

From the NEAT repository root, using the shared manuscript environment:

```bash
python -m pip install -r manuscript/requirements.txt
python manuscript/scripts/predict_final_model.py
```

The prediction script reads the three sample TSVs under
`manuscript/data/predict_new_case_data/` and saves the aggregated prediction TSV
under `manuscript/data/manuscript_prediction_results/`.

For another sample, use `--sample NAME --data-dir /path/to/inputs`.
Use `--output-dir /path/to/results` to change the destination.

The supplied historical bundle was saved with scikit-learn 1.7.2 and emits a
version warning in the shared environment (1.6.1). It has not been retrained or
converted. See the manuscript validation notes for the checks performed.
