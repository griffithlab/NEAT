"""Encode Strategy 2 missingness indicators and -1 values without fitting."""

import argparse
import csv
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1] / "data"


def missing(value):
    return (value or "").strip().lower() in {"", "na", "n/a", "nan", "null", "none", "<na>"}


def encode(columns, rows):
    groups = {
        "WT_missing": [c for c in columns if "WT" in c.split() or re.fullmatch(r"(?:Corresponding|Median) Fold Change", c)],
        "MT_missing": [c for c in columns if re.fullmatch(r"(?:NetMHC|SMM|SMMPMBEC) MT (?:IC50 Score|Percentile)", c)],
        "RNA_missing": [c for c in columns if re.fullmatch(r"RNA (?:VAF|Depth)|Allele Expr", c)],
    }
    if not rows or len(columns) != 75 or len(set(columns)) != 75 or any(c in columns for c in groups):
        raise ValueError("Expected a nonempty, pre-encoding table with 75 unique columns.")
    if [len(group) for group in groups.values()] != [29, 6, 3] or columns.count("Pos") != 1:
        raise ValueError("Unexpected WT/MT/RNA/Pos feature groups.")
    affected = [c for group in groups.values() for c in group] + ["Pos"]
    if len(set(affected)) != 39:
        raise ValueError("Missingness groups overlap.")
    for column in affected:
        observed = [float(row[column]) for row in rows if not missing(row[column])]
        if not observed or not all(value > -1 for value in observed):
            raise ValueError(f"Missing observations or -1 sentinel collision in {column}.")
    if any(sum(missing(row[c]) for c in groups["RNA_missing"]) not in (0, 3) for row in rows):
        raise ValueError("Expected the three RNA measurements to be missing together.")
    encoded = []
    for row in rows:
        result = dict(row)
        result.update({name: int(any(missing(row[c]) for c in group)) for name, group in groups.items()})
        for column in affected:
            if missing(row[column]):
                result[column] = "-1"
        if any(missing(str(value)) for value in result.values()):
            raise ValueError("Missing values remain outside the supported feature groups.")
        encoded.append(result)
    return columns + list(groups), encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "merged_df_pre_imputation.csv")
    parser.add_argument("--output", type=Path, default=DATA / "merged_df_encoded.csv")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input.")
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    columns, rows = encode(columns, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows, {len(columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
