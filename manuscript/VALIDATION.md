# Manuscript reproducibility check

Checked November 16, 2026, using Python 3.13.2 and all eight exact pins in
`requirements.txt`. Package dependency checks passed. Runs used an isolated
copy containing only `manuscript/`, with a space in its path, launched from an
unrelated working directory.

| Workflow | Check performed | Result |
| --- | --- | --- |
| Prospective cohort | Full `run_all.py`, both calibration cohorts | 35 PNGs generated; reference correlations, accuracies, membership, and every threshold curve point verified |
| Review time | Full data: 187 pre / 129 post observations | ECDF PNG saved; Mann–Whitney p = 0.124000 and KS p = 0.002945 |
| Demo prediction | All three supplied TSVs and final bundle | Output matches the existing 23-row, 36-column TSV byte for byte |
| Repeated development | Both strategies, all four models and both split types, two repetitions; RF 101 trees / 8 features; logistic C=1 / L2 | All 32 checkpoints completed; metrics recomputed from predictions; patient separation verified in outer and inner folds |
| Aggregation / resume | Combined table, checkpoint resume, summary-only mode | 16-row table generated; existing checkpoints reused |
| Interim random forests | Both approaches, 3-fold CV, 101 trees / 8 features | 12 PNGs generated; all four saved model files are fitted and match independent fits |
| Entry points | `--help` on all ten commands; parse all twelve Python files | Passed |

All 48 generated PNGs decoded correctly and contained plotted content.
Representative figures were visually checked. The 12 interim figures above
come from a reduced-grid execution check and are not full-search results.

## Patient IDs and input integrity

- All 12 CSV/TSV datasets were checked. Every `patient_id` value uses an
  anonymized ID from the workspace mapping. No mapped original patient IDs
  were found in the manuscript text/data files or serialized model artifacts.
- For the four remaining datasets changed during anonymization, every cell
  was compared with the original backup after applying the mapping. All matched.
- The patient-ID mapping remains outside NEAT and is not required to run any
  manuscript script.
- Input data, supplied model artifacts, and best parameters remained unchanged.

## Limits

The full 100-repetition analyses and full interim search were not rerun in this
check. Reduced grids test execution and data handling, not the published training
numbers. Historical package versions and patient-ID ordering can affect the
exact results of retraining. The full commands remain documented in the README.

The final bundle was saved with scikit-learn 1.7.2 and emits a version warning
when loaded by the shared 1.6.1 environment. Its demo output matches the supplied
result, but that check does not establish general cross-version compatibility.
The model was not replaced or retrained. Prospective and review-time plots do
not load this bundle and are unaffected by this warning.

Dependencies were verified in the existing environment; installation into a new
environment or on another operating system was not tested. Changes are in the
working tree; new scripts and datasets must be committed before sharing a clone.
