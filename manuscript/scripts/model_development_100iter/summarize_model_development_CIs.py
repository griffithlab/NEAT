"""Combine both strategies into the manuscript's 16-row performance table.

Requires completed, valid checkpoints for every selected repeat in all 16
experiments. Recomputes means and percentiles without retraining any models.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from model_development_CIs_common import (MANUSCRIPT, MODELS, collect_table,
                                         write_csv, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = MANUSCRIPT / "results" / "model_development_CIs"
    parser.add_argument("--strategy1-dir", type=Path, default=base / "strategy1")
    parser.add_argument("--strategy2-dir", type=Path, default=base / "strategy2")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        help="Default: manuscript/results/model_development_CIs/model_development_all_comb_<repeats>iter.csv")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    output = args.output or base / f"model_development_all_comb_{args.repeats}iter.csv"
    provenance = {}
    for model in MODELS:
        for split in ["row", "patient"]:
            pair = []
            for strategy, root in [(1, args.strategy1_dir), (2, args.strategy2_dir)]:
                path = root / model / split / "settings.json"
                settings = json.loads(path.read_text())
                provenance[f"strategy{strategy}/{model}/{split}"] = settings
                pair.append({key: settings[key] for key in ["seed", "test_size", "cv_folds", "grid"]})
            if pair[0] != pair[1]:
                raise ValueError(f"The strategies use different split/tuning settings for {model}/{split}.")
    # Validate all checkpoints before writing either final artifact.
    frames = [collect_table(root, strategy, args.repeats) for strategy, root in
              [(1, args.strategy1_dir), (2, args.strategy2_dir)]]
    table = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(table, output)
    write_json({"repeats_per_experiment": args.repeats, "rows": len(table),
                "interval": "mean (empirical 2.5th percentile, 97.5th percentile)",
                "experiments": provenance}, output.with_suffix(".json"))
    print(f"Saved {len(table)} rows to {output}")


if __name__ == "__main__":
    main()
