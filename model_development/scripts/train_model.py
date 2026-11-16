"""Tune a balanced random forest by row-level CV, then fit all eligible rows.

No outer holdout or statistical imputation. Default grid is exactly the supplied
range expressions (tree counts end at 4901, not 5000).
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

HERE = Path(__file__).resolve().parent


def write_json(value, path):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "encode_data/output/merged_df_encoded.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=918)
    parser.add_argument("--max-features", type=int, nargs="+", help="Optional smaller verification grid")
    parser.add_argument("--trees", type=int, nargs="+", help="Optional smaller verification grid")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir.")
    data = pd.read_csv(args.input)
    input_rows = len(data)
    data = data.loc[data.Evaluation != "Review"].copy().reset_index(drop=True)
    y = data.Evaluation.replace("Pending", "Reject").map({"Accept": 1, "Reject": 0})
    if y.isna().any() or data.patient_id.isna().any():
        raise ValueError("Unknown labels or missing patient IDs")
    X = data.drop(columns=["ID", "patient_id", "Evaluation"]).copy()
    if X.isna().any().any() or X.shape[1] != 75:
        raise ValueError("Expected 75 encoded predictors without nulls")
    maps = {}
    for c in X.select_dtypes(include=["object", "category"]):
        maps[c] = {value: i for i, value in enumerate(sorted(X[c].unique()))}
        X[c] = X[c].map(maps[c])
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite predictors")
    grid = {"max_features": args.max_features or list(range(1, X.shape[1] + 1)),
            "n_estimators": args.trees or list(range(1, 5001, 100))}
    if min(grid["max_features"]) < 1 or max(grid["max_features"]) > X.shape[1] or min(grid["n_estimators"]) < 1:
        raise ValueError("Invalid grid")
    settings = {"input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "input_rows": input_rows, "training_rows": len(data), "excluded_review_rows": input_rows-len(data),
                "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
                "feature_count": X.shape[1], "feature_order": list(X.columns), "category_maps": maps,
                "label_map": {"Accept": 1, "Reject": 0, "Pending": 0},
                "seed": args.seed, "grid": grid, "cv_folds": 3, "scoring": "roc_auc", "outer_holdout": False,
                "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "joblib"]}}
    write_json(settings, out / "settings.json")
    cv = StratifiedKFold(3, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X, y))
    membership = data[["ID", "patient_id"]].copy()
    membership["label"] = y
    membership["validation_fold"] = -1
    for i, (_, validation) in enumerate(folds):
        membership.loc[validation, "validation_fold"] = i
    membership.to_csv(out / "cv_membership.csv", index=False)
    candidates = len(grid["max_features"]) * len(grid["n_estimators"])
    print(f"Training rows: {len(X)}; predictors: {X.shape[1]}; candidates: {candidates}; CV fits: {candidates*3}", flush=True)
    search = GridSearchCV(BalancedRandomForestClassifier(random_state=args.seed, n_jobs=1,
                          replacement=True, bootstrap=True), grid, scoring="roc_auc", cv=folds,
                          n_jobs=args.jobs, pre_dispatch=args.jobs, refit=False, error_score="raise", verbose=2)
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(out / "cv_results.csv", index=False)
    write_json(search.best_params_, out / "best_params.json")
    model = BalancedRandomForestClassifier(**search.best_params_, random_state=args.seed,
                                           n_jobs=args.jobs, replacement=True, bootstrap=True, oob_score=True)
    model.fit(X, y)
    oob = model.oob_decision_function_
    valid = np.isfinite(oob).all(axis=1) & (oob.sum(axis=1) > 0)
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    prob = oob[valid, positive]
    labels = y.to_numpy()[valid]
    two_classes = len(np.unique(labels)) == 2
    metrics = {"best_cv_AUROC": float(search.best_score_),
               "OOB_AUROC": float(roc_auc_score(labels, prob)) if two_classes else None,
               "OOB_AUPR": float(average_precision_score(labels, prob)) if two_classes else None,
               "OOB_accuracy_valid_rows": float(accuracy_score(labels, model.classes_[oob[valid].argmax(axis=1)])) if valid.any() else None,
               "OOB_valid_rows": int(valid.sum()), "OOB_excluded_rows": int((~valid).sum()),
               "training_rows": len(X),
               "interpretation": "Library row-level OOB diagnostic after CV selection; not held-out performance. Balanced resampling with replacement can make OOB optimistic."}
    write_json(metrics, out / "oob_metrics.json")
    predictions = membership.drop(columns="validation_fold")
    predictions["valid_oob"] = valid
    predictions["accept_probability"] = np.where(valid, oob[:, positive], np.nan)
    predictions.to_csv(out / "oob_predictions.csv", index=False)
    pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    artifact = {"model": model, "feature_order": list(X.columns), "category_maps": maps,
                "label_map": settings["label_map"], "settings": settings}
    temporary = out / "model.tmp"
    joblib.dump(artifact, temporary)
    temporary.replace(out / "model_bundle.joblib")
    loaded = joblib.load(out / "model_bundle.joblib")
    assert np.array_equal(model.predict_proba(X.iloc[:10]), loaded["model"].predict_proba(X.iloc[:10]))
    write_json({"status": "complete", "model_reload_verified": True, "metrics": metrics}, out / "completed.json")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
