"""Predict a new case using the final model and Strategy 2 missingness encoding.

Run from NEAT: python manuscript/scripts/predict_final_model.py
Inputs: manuscript/data/predict_new_case_data/<sample>.*.tsv
Model: manuscript/manuscript_final_model/model_bundle.joblib
Output: manuscript/data/manuscript_prediction_results/<sample>.*.ML_predict.tsv
"""

import argparse
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

MANUSCRIPT = Path(__file__).resolve().parents[1]
DATA_DIR = MANUSCRIPT / "data" / "predict_new_case_data"
ARTIFACTS_DIR = MANUSCRIPT / "manuscript_final_model"
OUTPUT_DIR = MANUSCRIPT / "data" / "manuscript_prediction_results"
SAMPLE_NAME = "Test"
ML_THRESHOLD_ACCEPT = 0.55
ML_THRESHOLD_REJECT = 0.30
INDICATORS = ["WT_missing", "MT_missing", "RNA_missing"]


def read_columns(path, columns):
    """Read required features, normalizing legacy predictor column names."""
    aliases = {f"{method} {side} Score": f"{method} {side} IC50 Score"
               for method in ["NetMHCpanEL", "MHCflurry", "MHCnuggetsI", "NetMHC",
                              "NetMHCcons", "NetMHCpan", "PickPocket", "SMM", "SMMPMBEC"]
               for side in ["WT", "MT"]}
    aliases.update({f"NetMHCpanEL {side} Presentation Score": f"NetMHCpanEL {side} IC50 Score"
                    for side in ["WT", "MT"]})
    frame = pd.read_csv(path, sep="\t", usecols=lambda c: aliases.get(c, c) in columns,
                        na_values=["", "NA", "NaN", "nan", "N/A", "n/a", "null", "None", "none", "<NA>"],
                        keep_default_na=False).rename(columns=aliases)
    if not frame.columns.is_unique:
        raise ValueError(f"Duplicate predictor names after normalization in {path.name}.")
    return frame.reindex(columns=columns)


def merge_and_prepare_data(class1_aggregated_path, class1_all_epitopes_path,
                           class2_aggregated_path, features):
    """Merge the pVACseq tables and retain the final model's raw feature set."""
    aggregate_columns = ["ID", "Index", "Best Peptide", "Allele", "Evaluation", "IC50 MT", "IC50 WT",
        "%ile MT", "%ile WT", "Allele Expr", "DNA VAF", "TSL", "Num Passing Peptides", "Pos",
        "Prob Pos", "RNA Depth", "RNA Expr", "RNA VAF", "Ref Match"]
    scores = ["IC50 MT", "IC50 WT", "%ile MT", "%ile WT"]
    class1 = read_columns(class1_aggregated_path, aggregate_columns).rename(
        columns={c: f"{c} class1" for c in scores})
    class2 = read_columns(class2_aggregated_path, ["ID", *scores]).rename(
        columns={c: f"{c} class2" for c in scores})
    raw_features = [c for c in features if c not in INDICATORS]
    aggregate_features = set(class1.columns) | set(class2.columns) | {"Prob match"}
    epitope_columns = ["Index", "MT Epitope Seq", "HLA Allele"] + [
        c for c in raw_features if c not in aggregate_features]
    epitopes = read_columns(class1_all_epitopes_path, epitope_columns)
    merged = class1.merge(class2, on="ID").merge(epitopes,
        left_on=["Index", "Best Peptide", "Allele"],
        right_on=["Index", "MT Epitope Seq", "HLA Allele"])
    if merged.empty:
        raise ValueError("No candidates matched across the three input tables.")
    merged["Pos"] = pd.to_numeric(merged.Pos.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    merged["Prob Pos"] = merged["Prob Pos"].fillna("0").astype(str).str.split(",").map(
        lambda parts: int(float(parts[0])) if parts[0].replace(".", "", 1).isdigit() else 0)
    merged["Prob match"] = merged.Pos.notna() & merged.Pos.eq(merged["Prob Pos"])
    merged["Gene of Interest"] = merged["Gene of Interest"].fillna(False)
    merged["TSL"] = pd.to_numeric(merged.TSL, errors="coerce").fillna(6).astype(int)
    return merged[["ID", "Evaluation", *raw_features]]


def encode_missing_data(data, bundle):
    """Use the training indicator groups and -1 for missing predictors; fit nothing."""
    features = bundle["feature_order"]
    X = data[[c for c in features if c not in INDICATORS]].copy()
    groups = {
        "WT_missing": [c for c in X if "WT" in c.split() or re.fullmatch(r"(?:Corresponding|Median) Fold Change", c)],
        "MT_missing": [c for c in X if re.fullmatch(r"(?:NetMHC|SMM|SMMPMBEC) MT (?:IC50 Score|Percentile)", c)],
        "RNA_missing": [c for c in X if re.fullmatch(r"RNA (?:VAF|Depth)|Allele Expr", c)],
    }
    if [len(group) for group in groups.values()] != [29, 6, 3]:
        raise ValueError("The final model does not have the expected Strategy 2 feature groups.")
    for indicator, columns in groups.items():
        X[indicator] = X[columns].isna().any(axis=1).astype(int)
    for column in ["Ref Match", "Prob match", "Gene of Interest"]:
        encoded = X[column].map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
        if (X[column].notna() & encoded.isna()).any():
            raise ValueError(f"Unknown boolean values in {column}.")
        X[column] = encoded
    for column, mapping in bundle["category_maps"].items():
        unknown = set(X[column].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown categories in {column}: {sorted(map(str, unknown))}")
        X[column] = X[column].map(mapping)
    # New cases can lack additional predictors; retain the original indicator
    # definitions and encode all remaining missing predictor values as -1.
    X = X[features].apply(pd.to_numeric, errors="raise").fillna(-1)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Predictors contain infinities.")
    return X


def make_ml_predictions(data, bundle, ml_threshold_accept=0.55, ml_threshold_reject=0.30):
    X = encode_missing_data(data, bundle)
    model = bundle["model"]
    model.set_params(n_jobs=1)
    result = data.copy()
    result["Accept_pred_prob"] = model.predict_proba(X)[:, list(model.classes_).index(1)]
    if not np.isfinite(result.Accept_pred_prob).all() or not result.Accept_pred_prob.between(0, 1).all():
        raise ValueError("The model returned invalid Accept probabilities.")
    # Preserve the previous script's threshold boundaries and TSV labeling.
    result["Evaluation_pred"] = np.select(
        [result.Accept_pred_prob.ge(ml_threshold_accept), result.Accept_pred_prob.gt(ml_threshold_reject)],
        ["Accept", "Review"], default="Reject")
    return result


def create_final_output(
    post_imputed_data: pd.DataFrame,
    original_agg_file_path: Path,
    output_dir: Path,
    sample_name: str,
) -> Path:
    """Merge predictions back into the original class I aggregated file."""
    print("Writing final prediction file...")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_df = pd.read_csv(
        original_agg_file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    original_columns = list(orig_df.columns)
    cols_to_preserve = [c for c in original_columns if c != "Evaluation"]

    base_df = orig_df.drop(columns=["Evaluation"])
    merged_df = base_df.merge(
        post_imputed_data[["ID", "Evaluation_pred", "Accept_pred_prob"]],
        on="ID",
        how="left",
    ).rename(columns={"Evaluation_pred": "Evaluation"})
    merged_df["Evaluation"] = merged_df["Evaluation"].fillna("Pending")

    merged_df["ML Prediction (score)"] = merged_df.apply(
        lambda row: "NA"
        if (
            pd.isna(row["Evaluation"])
            or row["Evaluation"] == "Pending"
            or pd.isna(row["Accept_pred_prob"])
        )
        else f"{row['Evaluation']} ({round(row['Accept_pred_prob'], 2)})",
        axis=1,
    )
    merged_df.loc[merged_df["Evaluation"] == "Review", "Evaluation"] = "Pending"
    final_df = merged_df.drop(columns=["Accept_pred_prob"])

    final_df[cols_to_preserve] = orig_df[cols_to_preserve]

    output_file = output_dir / f"{sample_name}.MHC_I.all_epitopes.aggregated.ML_predict.tsv"
    final_df.to_csv(output_file, float_format="%.3f", sep="\t", index=False, na_rep="NA")
    print(f"ML predictions saved to: {output_file}")
    return output_file


def run_predictions(sample_name=SAMPLE_NAME, artifacts_dir=ARTIFACTS_DIR, output_dir=OUTPUT_DIR,
                    ml_threshold_accept=ML_THRESHOLD_ACCEPT, ml_threshold_reject=ML_THRESHOLD_REJECT,
                    data_dir=DATA_DIR):
    """Read pVACseq tables, encode missingness, predict, and write the original TSV format."""
    if not 0 <= ml_threshold_reject <= ml_threshold_accept <= 1:
        raise ValueError("Require 0 <= reject threshold <= accept threshold <= 1.")
    bundle = joblib.load(Path(artifacts_dir) / "model_bundle.joblib")
    data_dir = Path(data_dir)
    merged = merge_and_prepare_data(
        data_dir / f"{sample_name}.MHC_I.all_epitopes.aggregated.tsv",
        data_dir / f"{sample_name}.MHC_I.all_epitopes.tsv",
        data_dir / f"{sample_name}.MHC_II.all_epitopes.aggregated.tsv", bundle["feature_order"])
    predictions = make_ml_predictions(merged, bundle, ml_threshold_accept, ml_threshold_reject)
    return create_final_output(predictions,
        data_dir / f"{sample_name}.MHC_I.all_epitopes.aggregated.tsv", Path(output_dir), sample_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=SAMPLE_NAME)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_predictions(args.sample, args.model_dir, args.output_dir, data_dir=args.data_dir)
