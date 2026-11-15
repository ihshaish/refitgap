"""refitgap: thresholded forgetting, split into what the scores lost and what the thresholds lost.

Stored per-period scores and labels in; cross-fitted levels, the split
deficit = attainable gap + change in mismatch, and seed-paired contrasts out.
compare() handles two scorers on the same records, method_contrast() the paired lead
of one method over another. API.txt in this directory has the details. numpy and scipy
only; figures need matplotlib (extra "plots"). Command line: refitgap.
"""
from .fitters import GRID, f1_vec, fit_exact, fit_grid, fit_shared_exact, macro_f1, shared_objective
from .crossfit import (SPLITS, SPLIT_BASE, Evaluation, array_hash, decompose, evaluate, fit_validation_rule,
                       partitions)
from .inference import (ALPHA, MARGIN, classify, classify_unadjusted, family, family_ni, holm, paired_contrast,
                        paired_stats, paired_vector, dropped_seeds)
from .records import align_to, drop_unlabelled
from .runs import evaluate_npz, load_run
from .analysis import analyse, make_run, run_from_arrays, run_from_npz, save_arrays, write_tables
from .twoscorer import compare, method_contrast
from .identification import f1_bounds_from_counts, identification_example

__version__ = "0.1.0"

__all__ = [
    "GRID", "f1_vec", "fit_exact", "fit_grid", "fit_shared_exact", "macro_f1", "shared_objective", "SPLITS",
    "SPLIT_BASE", "Evaluation", "array_hash", "decompose", "evaluate", "fit_validation_rule", "partitions",
    "ALPHA", "MARGIN", "classify", "classify_unadjusted", "family", "family_ni", "holm", "paired_contrast",
    "paired_stats", "paired_vector", "dropped_seeds", "align_to", "drop_unlabelled", "evaluate_npz",
    "load_run", "analyse", "make_run", "run_from_arrays", "run_from_npz", "save_arrays", "write_tables",
    "compare", "method_contrast", "f1_bounds_from_counts", "identification_example", "__version__",
]
