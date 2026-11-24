"""refitgap: thresholded forgetting, split into what the scores lost and what the thresholds lost.

Stored per-period scores and labels in; cross-fitted levels, deficit = attainable gap +
change in mismatch, and seed-paired contrasts out. compare(): two scorers on the same
records. method_contrast(): one method's lead over another, at the carried rule and after
both refit. rule_lead(): that lead under the previous period's rule or any fixed rule, beside
the refitted lead on the same halves. Details in API.txt next to this file. numpy and scipy
only; figures need matplotlib (extra "plots"). Command line: refitgap.
"""
from .fitters import GRID, f1_vec, fit_exact, fit_grid, fit_shared_exact, macro_f1, shared_objective
from .crossfit import (SPLITS, SPLIT_BASE, LIMITED_BASE, PREVIOUS_BASE, PREVIOUS_N, Evaluation, array_hash,
                       decompose, evaluate, fit_validation_rule, limited_draws, partitions, previous_draws)
from .inference import (ALPHA, MARGIN, classify, classify_unadjusted, family, family_ni, holm, paired_contrast,
                        paired_stats, paired_vector, dropped_seeds)
from .records import align_to, drop_unlabelled
from .runs import evaluate_npz, load_run
from .analysis import analyse, make_run, run_from_arrays, run_from_npz, save_arrays, write_tables
from .twoscorer import compare, method_contrast, rule_lead
from .identification import f1_bounds_from_counts, identification_example

__version__ = "0.2.2"

__all__ = [
    "GRID", "f1_vec", "fit_exact", "fit_grid", "fit_shared_exact", "macro_f1", "shared_objective", "SPLITS",
    "SPLIT_BASE", "LIMITED_BASE", "PREVIOUS_BASE", "PREVIOUS_N", "Evaluation", "array_hash", "decompose",
    "evaluate", "fit_validation_rule", "limited_draws", "partitions", "previous_draws",
    "ALPHA", "MARGIN", "classify", "classify_unadjusted", "family", "family_ni", "holm", "paired_contrast",
    "paired_stats", "paired_vector", "dropped_seeds", "align_to", "drop_unlabelled", "evaluate_npz",
    "load_run", "analyse", "make_run", "run_from_arrays", "run_from_npz", "save_arrays", "write_tables",
    "compare", "method_contrast", "rule_lead", "f1_bounds_from_counts", "identification_example", "__version__",
]
