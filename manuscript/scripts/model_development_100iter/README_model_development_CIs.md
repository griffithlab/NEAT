# Reproduce model development analyses

Run these commands from the NEAT repository root, with your Python environment activated.

Install dependencies:

```bash
python -m pip install -r manuscript/requirements.txt
```

Run Strategy 1, then Strategy 2, then generate the combined results table:

```bash
python manuscript/scripts/model_development_100iter/model_development_strategy1_4models_CIs.py --jobs 2
python manuscript/scripts/model_development_100iter/model_development_strategy2_4models_CIs.py --jobs 2
python manuscript/scripts/model_development_100iter/summarize_model_development_CIs.py
```

Each strategy runs all four models with peptide and patient splits for 100 repetitions.
Input files are loaded automatically from `manuscript/data/missing_data_handeling/`.

Results are saved under `manuscript/results/model_development_CIs/`. The combined table is
`model_development_all_comb_100iter.csv` in that directory.

Rerun the same command to resume an interrupted run. Adjust `--jobs` to change the number of workers.
