"""Input contract and reproducibility metadata for prospective cohort analyses."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "neat-prospective-matplotlib"))

import numpy as np
import pandas as pd

MANUSCRIPT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = MANUSCRIPT / "data" / "prospective_cohort" / "prospective_model_comparison.csv"
DEFAULT_OUTPUT = MANUSCRIPT / "results" / "prospective_cohort"
KEYS = ["patient_id", "ID"]
CLASSES = ["Accept", "Reject", "Review"]
COLUMNS = [*KEYS, "Evaluation", "interim_accept_probability", "interim_prediction",
           "final_accept_probability", "final_prediction", "is_calibration"]


def final_labels(probabilities):
    """Preserve historical strict thresholds, with both boundaries in Review."""
    return np.select([probabilities.lt(.30), probabilities.gt(.55)],
                     ["Reject", "Accept"], default="Review")


def validate(data):
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    # Keep the original prospective features/annotations alongside analysis fields.
    data = data.copy()
    if data.empty or data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Require nonempty data with unique, nonnull (patient_id, ID) keys.")
    for column in ["Evaluation", "interim_prediction", "final_prediction"]:
        if not data[column].isin(CLASSES).all():
            raise ValueError(f"Unknown or missing labels in {column}; expected Accept/Reject/Review.")
    for column in ["interim_accept_probability", "final_accept_probability"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all() or not data[column].between(0, 1).all():
            raise ValueError(f"Invalid probabilities in {column}.")
    boolean = data.is_calibration.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if boolean.isna().any():
        raise ValueError("is_calibration must contain only true/false or 1/0.")
    data["is_calibration"] = boolean.astype(bool)
    if data.groupby("patient_id").is_calibration.nunique().gt(1).any():
        raise ValueError("Calibration status must be constant within each patient.")
    if not (data.final_prediction == final_labels(data.final_accept_probability)).all():
        raise ValueError("final_prediction disagrees with the historical 0.30/0.55 threshold rule.")
    return data


def load_data(path):
    # Preserve exported doubles exactly for strict threshold/nextafter sweeps.
    return validate(pd.read_csv(path, dtype={"patient_id": str, "ID": str}, float_precision="round_trip"))


def arguments(description, analysis):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / analysis)
    if analysis == "threshold_sensitivity":
        parser.add_argument("--cohort", choices=["all_samples", "excluding_calibration"],
                            default="all_samples", help="Historical threshold figure uses all_samples.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def write_manifest(args, analysis, data, filename="run_manifest.json", **details):
    value = {
        "analysis": analysis, "input": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(data), "patients": data.patient_id.nunique(),
        "calibration_rows": int(data.is_calibration.sum()),
        "calibration_patients": sorted(data.loc[data.is_calibration, "patient_id"].unique().tolist()),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]},
        "no_model_fitting_or_threshold_selection": True, **details,
    }
    (args.output_dir / filename).write_text(json.dumps(value, indent=2) + "\n")
