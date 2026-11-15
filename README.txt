refitgap
========

Train a multi-label classifier period by period, score the final model on an old period, and
its macro-F1 there drops. Part of the drop sits in the scores, which now separate the labels
less well, and part in the thresholds, carried from period one and now in the wrong place.
refitgap says how much of each, from stored scores alone:

    deficit B = attainable gap A + change in mismatch dR

The attainable gap compares the two models after each refits its thresholds on equal terms
(split-half cross-fitting, same partitions); the change in mismatch is what each loses by
keeping the carried rule. Same trick for two methods: replay's lead over sequential training
at the inherited thresholds, after both refit, and the change. This is the code of
"Decomposing Backward Transfer in Continual Multi-Label Text Classification" (Ihshaish,
Mayhew and Aydin, under review), with its plans, stored results and scripts.

Install
-------

    pip install refitgap                 # Python 3.10+, numpy and scipy, nothing else
    pip install "refitgap[plots,test]"   # matplotlib for figures, pytest etc. for the tests

(or pip install . from the clone). The entry points below read files in this repository, so
clone it too. On the command line, refitgap evaluate | report | example runs the same
evaluator on stored arrays (refitgap --help; API.txt, "Command line").

Reproduce Figure 2
------------------

    python examples/figure2/reproduce_figure2.py                # a few minutes on 4 cores
    python examples/figure2/reproduce_figure2.py --from-levels  # about a second

All five paired comparisons of Figure 2 (supplement Table C.1) over periods 2 to 7: lead at
the inherited thresholds, after both refit, and the change. Paired t intervals over six
DistilBERT pairs (from stored per-pair levels: their 30 score files, about 60 MB, do not ship),
percentile bootstrap over ten BiGRU pairs (from raw scores in paper/results/, or stored levels
with --from-levels). Every lead, mean and bound must match examples/figure2/expected.json to
1e-10.

    python examples/two_scorer/table1.py

Table 1: two scorers with equal AUROC (6/11), average precision (5/18) and deficit (1/26),
one all change in mismatch, the other all attainable gap. Exact fractions, no luck involved!

Compare your own models
-----------------------

    python examples/your_models/compare_your_models.py [DATA_DIR] --a mine --b baseline

DATA_DIR holds three kinds of CSV (examples/your_models/data/ is a tiny synthetic set):

    labels.csv                    period, record_id, one 0/1 column per label
    scores_<method>_seed<k>.csv   period, record_id, one score column per label (the final
                                  model of <method>, trained under seed k)
    thresholds.csv                method, seed, the carried rule (one threshold per label)

Out come a's leads over b at the carried rule and after both refit, and the change (refitted
lead minus carried lead), per seed and with paired t intervals. In Python: refitgap.compare()
and method_contrast(); API.txt has the lot.

Interface
---------

In: aligned scores and labels, the carried rule, and a seed for the common partitions.
- Scores: one (records, labels) array per period and model, probabilities in [0, 1]; logits go
  through a sigmoid first, or declare score_scale="other" with thresholds on the same scale.
  Labels: 0/1, same shape.
- Alignment: rows match by record id, columns by label name. A shape mismatch, a duplicate or
  missing id, a different label set, or different labels for the two models raises. Nothing
  is truncated or dropped quietly.
- Ties: a score equal to its threshold is a positive. A refitted threshold is an observed
  score; of two equally good cutpoints the higher wins.
- A label with no positive in a period has F1 0 there and stays in the macro mean.
- Partitions: partition b of period id j under seed s is the b-th permutation of numpy
  default_rng(7000 + 100*s + j); first half fits, second scores, then the reverse. 200 in the
  paper. Both models of a pair share them, so give both the same seed and period ids. Period
  ids are integers in [0, 100); the paper's periods 2 to 7 are ids 1 to 6.
- Out: carried level U, refitted level V, deficit, attainable gap and change in mismatch, per
  period and averaged; for two methods, leads per seed with means and 95% intervals. Positive
  means the first-named model scores higher. Numbers only: refitgap recommends nothing, and
  the attainable gap describes the scores, not what the representation kept.

Repository layout
-----------------

    src/refitgap/   the evaluator; src/refitgap/API.txt is the reference
    examples/       Table 1, Figure 2, your own models, walkthrough.ipynb (seconds, the lot)
    paper/          plans, scripts, stored results, checks; README.txt maps tables to scripts,
                    DATA.txt the corpora (public, not redistributed, bar a 3200-record ASRS sample)
    paper/threshold_policy_survey.txt   how 34 sampled studies report their thresholds
    tests/          python -m pytest (fast, a minute or two); -m slow adds the full Figure 2
                    recomputation from raw scores (minutes)

MIT licence. Please cite the paper and the software (CITATION.cff has both). Have fun with it!
