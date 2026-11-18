# Historical interim model artifacts

- `rf_downsample_model.pkl`: saved interim balanced random forest.
- `trained_imputer.joblib`: fitted preprocessing imputer used with that model.
- `label_encoders.pkl`: saved categorical encoders for the interim workflow.

These are retained reference artifacts. The prospective plots use saved
predictions in the ready comparison CSV. New-case predictions use
`../manuscript_final_model/model_bundle.joblib` through `predict_final_model.py`.
Retraining the interim forests writes new models under `../results/randomforest/`
and does not overwrite these snapshots.
