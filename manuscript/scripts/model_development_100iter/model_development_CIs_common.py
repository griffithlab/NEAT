"""Shared implementation of the manuscript's repeated-holdout experiments.

The two strategy entry points differ only in input and categorical encoding.
Preprocessing is conditional on the supplied, already processed dataset.
Intervals measure variability across splits, not population confidence intervals.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MANUSCRIPT = Path(__file__).resolve().parents[2]
MODELS = {
    "logistic_class_weighting": ("Logistic regression", "Class weighting"),
    "logistic_downsample": ("Logistic regression", "Down-sampling"),
    "rf_class_weighting": ("Random forest", "Class weighting"),
    "rf_downsample": ("Random forest", "Down-sampling"),
}
STRATEGIES = {
    1: "Strategy 1: Median and iterative imputation",
    2: "Strategy 2: Fixed-value encoding with missingness indicators",
}
INPUTS = {1: "Strategy1_df_imputed_python.csv", 2: "Strategy2_merged_df_encoded.csv"}
METRICS = ["OOB_AUROC", "Accuracy", "Sensitivity", "Specificity", "F1", "AUROC", "AUPR"]
TABLE_COLUMNS = ["Imputation strategy", "Model family", "Class imbalance strategy",
                 "Train/Test split strategy", *METRICS]


def write_json(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")
    temporary.replace(path)


def write_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_data(source, strategy):
    data = pd.read_csv(source)
    required = {"ID", "patient_id", "Evaluation"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(data.columns))}")
    # Preserve the original row order: it determines the seeded row holdouts.
    data = data.loc[data.Evaluation != "Review"].copy().reset_index(drop=True)
    data["Evaluation"] = data.Evaluation.replace("Pending", "Reject").map(
        {"Accept": 1, "Reject": 0})
    if data[["patient_id", "Evaluation"]].isna().any().any():
        raise ValueError("Missing patient IDs or unrecognized labels.")
    X = data.drop(columns=["ID", "patient_id", "Evaluation"]).copy()
    category_maps = {}
    if strategy == 2:
        # Reproduce the original fixed integer encoding, including its order.
        for column in X.select_dtypes(include=["object", "category"]):
            if X[column].isna().any():
                raise ValueError(f"Null categorical predictor: {column}")
            mapping = {value: index for index, value in enumerate(sorted(X[column].unique()))}
            category_maps[column] = mapping
            X[column] = X[column].map(mapping)
    if X.empty or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Predictors must be numeric after the strategy's encoding.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Predictors contain nulls or infinities; provide the processed input CSV.")
    if data.Evaluation.nunique() != 2:
        raise ValueError("Both Accept and Reject/Pending observations are required.")
    return data, X, category_maps


def build_estimator(model, seed, args):
    if model.startswith("logistic"):
        steps = [("scaler", StandardScaler())]
        if model == "logistic_downsample":
            from imblearn.pipeline import Pipeline as SamplingPipeline
            from imblearn.under_sampling import RandomUnderSampler
            steps.append(("undersample", RandomUnderSampler(
                random_state=seed, sampling_strategy="auto", replacement=False)))
            pipeline = SamplingPipeline
            weight = None
        else:
            pipeline = Pipeline
            weight = "balanced"
        steps.append(("classifier", LogisticRegression(
            random_state=seed, max_iter=1000, class_weight=weight)))
        return pipeline(steps), {
            "classifier__C": args.c_values,
            "classifier__penalty": args.penalties,
            "classifier__solver": ["liblinear"],
        }, 5
    grid = {"max_features": args.max_features, "n_estimators": args.trees}
    if model == "rf_class_weighting":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1,
            class_weight="balanced_subsample", oob_score=True, bootstrap=True)
    else:
        from imblearn.ensemble import BalancedRandomForestClassifier
        estimator = BalancedRandomForestClassifier(random_state=seed, n_jobs=1,
            replacement=True, bootstrap=True, sampling_strategy="all", oob_score=False)
    return estimator, grid, 3


def split_data(data, X, approach, seed, cv_folds):
    y = data.Evaluation
    if approach == "patient":
        a, b = train_test_split(sorted(data.patient_id.unique()),
                                test_size=0.25, random_state=seed)
        train = np.flatnonzero(data.patient_id.isin(a))
        test = np.flatnonzero(data.patient_id.isin(b))
        splitter = StratifiedGroupKFold(cv_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(X.iloc[train], y.iloc[train],
                                    groups=data.patient_id.iloc[train]))
    else:
        # "Peptide split" in the table is an unstratified random row split.
        train, test = train_test_split(np.arange(len(data)),
                                      test_size=0.25, random_state=seed)
        splitter = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(X.iloc[train], y.iloc[train]))
    return train, test, folds


def save_oob(model, data, train, checkpoint):
    decision = np.asarray(model.oob_decision_function_)
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1 or decision.shape != (len(train), 2):
        raise ValueError("Expected binary OOB probabilities with Accept=1.")
    valid = np.isfinite(decision).all(axis=1) & (decision.sum(axis=1) > 0)
    probability = decision[:, positive[0]].copy()
    probability[~valid] = np.nan
    labels = data.Evaluation.iloc[train].to_numpy()
    scoreable = np.unique(labels[valid]).size == 2
    score = float(roc_auc_score(labels[valid], probability[valid])) if scoreable else None
    predictions = data.iloc[train][["ID", "patient_id", "Evaluation"]].copy()
    predictions.insert(0, "row_index", train)
    predictions["oob_accept_probability"] = probability
    predictions["has_oob_prediction"] = valid
    write_csv(predictions, checkpoint / "oob_predictions.csv")
    return {"OOB_AUROC": score, "OOB_valid_rows": int(valid.sum()),
            "OOB_total_rows": len(train), "OOB_coverage": float(valid.mean())}


def summarize(results, out, model):
    frame = pd.DataFrame(results)
    write_csv(frame, out / "per_repeat_metrics.csv")
    valid = frame.loc[frame.status == "ok"]
    metrics = METRICS if model.startswith("rf") else METRICS[1:]
    rows = []
    for metric in metrics:
        values = valid[metric].dropna() if metric in valid else pd.Series(dtype=float)
        rows.append({"metric": metric, "valid_repeats": len(values),
                     "completed_repeats": len(frame), "mean": values.mean(),
                     "median": values.median(), "SD": values.std(),
                     "split_interval_2.5pct": values.quantile(0.025),
                     "split_interval_97.5pct": values.quantile(0.975)})
    write_csv(pd.DataFrame(rows), out / "summary.csv")


def experiment_settings(strategy, source, X, category_maps, model, approach, args):
    estimator, grid, cv_folds = build_estimator(model, args.seed, args)
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]
    if model.endswith("downsample"):
        packages.append("imbalanced-learn")
    # Repeats and worker count may change when resuming; scientific settings may not.
    return {
        "schema_version": 1, "strategy": strategy, "model": model,
        "approach": approach, "grid": grid, "seed": args.seed,
        "test_size": 0.25, "cv_folds": cv_folds, "scoring": "roc_auc",
        "estimator": repr(estimator), "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "features": list(X.columns), "category_maps": category_maps,
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "labels": "exclude Review; Pending becomes Reject; Accept=1, Reject=0",
        "preprocessing": ("existing median/iterative-imputed CSV" if strategy == 1 else
                          "existing sentinel/indicator CSV; sorted categorical integer encoding"),
        "interval": "mean and empirical 2.5th-97.5th percentiles across holdouts",
        "oob_definition": "library row-level OOB; finite, nonzero-vote outer-training rows only",
    }


def check_settings(out, settings):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "settings.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Settings, code, packages, or input differ in {out}. Use a new --output-dir.")
    elif any(out.iterdir()):
        raise ValueError(f"Files without settings in {out}. Use a new --output-dir.")
    else:
        write_json(settings, path)


def run_experiment(data, X, model, approach, args, out):
    y = data.Evaluation
    results = []
    for repeat in range(args.repeats):
        checkpoint = out / f"repeat_{repeat + 1:03d}"
        checkpoint.mkdir(exist_ok=True)
        completed = checkpoint / "completed.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            if record["repeat"] != repeat + 1 or record["seed"] != args.seed + repeat:
                raise ValueError(f"Checkpoint index/seed mismatch in {completed}")
            results.append(record)
            print(f"{model}/{approach} {repeat + 1}/{args.repeats}: loaded checkpoint", flush=True)
            continue
        started = perf_counter()
        seed = args.seed + repeat
        estimator, grid, cv_folds = build_estimator(model, seed, args)
        train, test, folds = split_data(data, X, approach, seed, cv_folds)
        record = {
            "repeat": repeat + 1, "seed": seed,
            "train_rows": len(train), "test_rows": len(test),
            "train_patients": data.patient_id.iloc[train].nunique(),
            "test_patients": data.patient_id.iloc[test].nunique(),
            "test_patient_ids": json.dumps(sorted(data.patient_id.iloc[test].unique())),
            "train_accept_prevalence": y.iloc[train].mean(),
            "test_accept_prevalence": y.iloc[test].mean(),
        }
        membership = data[["ID", "patient_id", "Evaluation"]].copy()
        membership.insert(0, "row_index", data.index)
        membership["split"] = np.where(data.index.isin(train), "train", "test")
        membership["validation_fold"] = -1
        for fold, (_, validation) in enumerate(folds):
            membership.loc[train[validation], "validation_fold"] = fold
        write_csv(membership, checkpoint / "split_membership.csv")
        valid = y.iloc[train].nunique() == y.iloc[test].nunique() == 2 and all(
            y.iloc[train].iloc[a].nunique() == y.iloc[train].iloc[b].nunique() == 2
            for a, b in folds)
        if not valid:
            record["status"] = "invalid: a partition contains only one class"
        else:
            search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=folds,
                n_jobs=args.jobs, refit=True, verbose=0, error_score="raise")
            search.fit(X.iloc[train], y.iloc[train])
            write_csv(pd.DataFrame(search.cv_results_), checkpoint / "cv_results.csv")
            write_json(search.best_params_, checkpoint / "best_params.json")
            selected = search.best_estimator_
            if model == "rf_downsample":
                # Match the originals: enable OOB only after hyperparameter selection.
                selected = clone(selected).set_params(oob_score=True)
                selected.fit(X.iloc[train], y.iloc[train])
            if model.startswith("rf"):
                record.update(save_oob(selected, data, train, checkpoint))
            pred = selected.predict(X.iloc[test])
            prob = selected.predict_proba(X.iloc[test])[:, 1]
            tn, fp, fn, tp = confusion_matrix(y.iloc[test], pred, labels=[0, 1]).ravel()
            record.update({
                "status": "ok", **search.best_params_, "CV_AUROC": search.best_score_,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "Accuracy": accuracy_score(y.iloc[test], pred),
                "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                "F1": f1_score(y.iloc[test], pred, zero_division=0),
                "AUROC": roc_auc_score(y.iloc[test], prob),
                "AUPR": average_precision_score(y.iloc[test], prob),
            })
            predictions = data.iloc[test][["ID", "patient_id", "Evaluation"]].copy()
            predictions.insert(0, "row_index", test)
            predictions["prediction"] = pred
            predictions["accept_probability"] = prob
            write_csv(predictions, checkpoint / "test_predictions.csv")
        record["seconds"] = perf_counter() - started
        # The last write commits a repeat. Interrupted repeats are recomputed.
        write_json(record, completed)
        results.append(record)
        summarize(results, out, model)
        print(f"{model}/{approach} {repeat + 1}/{args.repeats}: "
              f"{record['status']}; {record['seconds']:.1f}s", flush=True)
    summarize(results, out, model)


def collect_table(root, strategy, repeats, models=None, splits=None):
    """Recompute the table from complete checkpoints; never use stale summaries."""
    rows = []
    baseline = None
    for model in models or MODELS:
        for approach in splits or ["row", "patient"]:
            folder = root / model / approach
            settings = json.loads((folder / "settings.json").read_text())
            if (settings["strategy"], settings["model"], settings["approach"]) != (strategy, model, approach):
                raise ValueError(f"Mismatched experiment in {folder}")
            common = {k: settings[k] for k in ["data_sha256", "features", "category_maps", "seed",
                      "test_size", "strategy", "implementation_sha256"]}
            if baseline is None:
                baseline = common
            elif common != baseline:
                raise ValueError(f"Incompatible input/settings across experiments: {folder}")
            records = []
            for index in range(1, repeats + 1):
                path = folder / f"repeat_{index:03d}" / "completed.json"
                record = json.loads(path.read_text())
                if record["status"] != "ok" or record["repeat"] != index or record["seed"] != settings["seed"] + index - 1:
                    raise ValueError(f"Invalid repetition: {path}")
                records.append(record)
            family, imbalance = MODELS[model]
            row = {"Imputation strategy": STRATEGIES[strategy], "Model family": family,
                   "Class imbalance strategy": imbalance,
                   "Train/Test split strategy": "Peptide split" if approach == "row" else "Patient split"}
            for metric in METRICS:
                if metric == "OOB_AUROC" and model.startswith("logistic"):
                    row[metric] = "N/A"
                    continue
                values = np.array([record.get(metric) for record in records], dtype=float)
                if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError(f"Invalid {metric} values in {folder}; inspect checkpoints.")
                lower, upper = np.quantile(values, [0.025, 0.975])
                row[metric] = f"{values.mean():.2f} ({lower:.2f}, {upper:.2f})"
            rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def main(strategy):
    parser = argparse.ArgumentParser(description=f"Strategy {strategy}: four models, two splits, empirical 95% intervals.")
    parser.add_argument("--input", type=Path, default=MANUSCRIPT / "data" / "missing_data_handeling" / INPUTS[strategy])
    parser.add_argument("--output-dir", type=Path, default=MANUSCRIPT / "results" / "model_development_CIs" / f"strategy{strategy}")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=918)
    parser.add_argument("--jobs", type=int, default=2, help="Parallel CV workers; models/splits run sequentially.")
    parser.add_argument("--models", choices=list(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--splits", choices=["row", "patient"], nargs="+", default=["row", "patient"])
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1, 10, 100])
    parser.add_argument("--penalties", choices=["l1", "l2"], nargs="+", default=["l1", "l2"])
    parser.add_argument("--trees", type=int, nargs="+", default=[100, 300, 500, 1000])
    parser.add_argument("--max-features", type=int, nargs="+", default=[8, 18, 36, 54, 72])
    parser.add_argument("--summary-only", action="store_true", help="Rebuild tables from existing checkpoints without fitting.")
    args = parser.parse_args()
    if args.repeats < 1 or args.jobs == 0 or not 0 <= args.seed <= 2**32 - args.repeats:
        parser.error("Require positive repeats, nonzero jobs, and seeds within the uint32 range.")
    if any(not np.isfinite(c) or c <= 0 for c in args.c_values) or min(args.trees) < 1:
        parser.error("C values must be finite and positive; tree counts must be positive.")
    data, X, maps = load_data(args.input, strategy)
    if any(model.startswith("rf") for model in args.models) and not 1 <= min(args.max_features) <= max(args.max_features) <= X.shape[1]:
        parser.error(f"max-features must be between 1 and {X.shape[1]}.")
    # Check every selected experiment before doing any expensive fitting.
    for model in args.models:
        for approach in args.splits:
            out = args.output_dir / model / approach
            settings = experiment_settings(strategy, args.input, X, maps, model, approach, args)
            if args.summary_only and not (out / "settings.json").exists():
                raise FileNotFoundError(f"Missing experiment settings in {out}")
            check_settings(out, settings)
    for model in args.models:
        for approach in args.splits:
            if not args.summary_only:
                run_experiment(data, X, model, approach, args, args.output_dir / model / approach)
    table = collect_table(args.output_dir, strategy, args.repeats, args.models, args.splits)
    complete = set(args.models) == set(MODELS) and set(args.splits) == {"row", "patient"}
    name = "all_models_summary.csv" if complete else "selected_models_summary.csv"
    write_csv(table, args.output_dir / name)
    write_json({"strategy": strategy, "repeats_per_experiment": args.repeats,
                "models": args.models, "splits": args.splits,
                "table": name, "interval": "mean (empirical 2.5th percentile, 97.5th percentile)"},
               args.output_dir / name.replace(".csv", "_metadata.json"))
    print(f"Saved {len(table)} rows to {args.output_dir / name}", flush=True)
