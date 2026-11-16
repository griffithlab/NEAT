"""Reproduce four models and repeated-split empirical intervals for Strategy 1.

Run with --help for input, output, model selection, and tuning options.
See README_model_development_CIs.md for the complete reviewer workflow.
"""

from model_development_CIs_common import main


if __name__ == "__main__":
    main(strategy=1)
