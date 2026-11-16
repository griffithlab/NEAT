"""Generate all prospective-cohort figures/tables from the single input CSV."""

import argparse
from pathlib import Path
import subprocess
import sys

from common import DEFAULT_INPUT, DEFAULT_OUTPUT, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    load_data(args.input)
    scripts = Path(__file__).resolve().parent
    for script, folder, options in [
        ("compare_probabilities.py", "probability_comparison", []),
        ("evaluate_models.py", "binary_roc_pr_curves", []),
        ("threshold_sensitivity.py", "threshold_sensitivity", []),
        ("threshold_sensitivity.py", "threshold_sensitivity",
         ["--cohort", "excluding_calibration"]),
    ]:
        print(f"Running {script} -> {folder}", flush=True)
        subprocess.run([sys.executable, str(scripts / script), "--input", str(args.input),
                        "--output-dir", str(args.output_dir / folder), *options], check=True)
    print(f"Saved prospective cohort analyses to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
