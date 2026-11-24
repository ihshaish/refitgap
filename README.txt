refitgap
========

A multi-label classifier is trained period by period, and its macro-F1 on old
periods drops. Is the model worse, or are the thresholds from period one just
in the wrong place now? refitgap answers from stored scores and labels. No
retraining.

It splits the drop into two parts:

    deficit B  =  attainable gap A  +  change in mismatch dR

A is what is left after the old and the new model both refit their thresholds
on the same labelled records. dR is what each loses by keeping the inherited
thresholds. The same refit compares two training methods: the lead at the
inherited thresholds, the lead after both refit, and the difference.

On ASRS, replay's lead over sequential training is gone once both models are
refitted. On MultiEURLEX it stays. That is Figure 2 of the paper, "Decomposing
Backward Transfer in Continual Multi-Label Text Classification" (Ihshaish,
Mayhew and Aydin, under review). The figure is in docs/paired_refit.png.


Install
-------

    pip install refitgap                 # Python 3.10+, needs numpy and scipy
    pip install "refitgap[plots,test]"   # adds matplotlib and pytest

The examples below read files from this repository, so clone it as well.


Thirty seconds
--------------

    $ python examples/quickstart.py
    mine minus baseline, macro-F1, 4 seeds, 95% t intervals
      at the carried rule  +0.2093  [+0.1373, +0.2814]
      after both refits    +0.0648  [+0.0450, +0.0847]
      change               -0.1445  [-0.2118, -0.0772]

Two synthetic methods, four seeds. At the carried thresholds "mine" leads by
0.21. Refit both and 0.06 is left. The script is twenty lines around two calls:

    import refitgap as rg

    # one evaluation per method and seed
    ev = rg.evaluate(scores, labels, seed, period_ids, rules={"carried": thresholds})

    # then the paired contrast over seeds
    mc = rg.method_contrast(evals["mine"], evals["baseline"], "carried",
                            names=("mine", "baseline"), same_rule=False)

scores and labels are lists with one (records, labels) array per period.


Your own models
---------------

    python examples/your_models/compare_your_models.py DATA_DIR --a mine --b baseline

DATA_DIR holds three kinds of CSV:

    labels.csv                   period, record_id, one 0/1 column per label
    scores_<method>_seed<k>.csv  period, record_id, one score column per label
    thresholds.csv               method, seed, one threshold per label

examples/your_models/data/ is a small synthetic set in this layout. From the
shell, "refitgap evaluate" and "refitgap report" do the same on .npz arrays
(refitgap --help).

Short of labels? evaluate(..., refit_n=500) refits on 500 labelled records.
evaluate(..., previous=...) scores thresholds fitted on the previous period.
rule_lead() sets a lead under either beside the refitted lead, and
per_label=True breaks a lead down by label.


Conventions
-----------

- Scores are probabilities in [0, 1]. Put logits through a sigmoid first.
- Rows are matched by record id and columns by label name. Anything that does
  not line up raises an error. Nothing is dropped quietly.
- A score equal to its threshold counts as positive. Between two equally good
  cuts the higher one wins.
- A label with no positive in a period has F1 0 there and stays in the mean.
- Thresholds are fitted on one random half of a period's records and scored on
  the other, then the halves swap. The paper averages 200 such splits. Both
  models of a pair get the same splits, so give them the same seed and period
  ids.
- Positive numbers mean the model named first scores higher.

src/refitgap/API.txt is the full reference.


Reproducing the paper
---------------------

    python examples/two_scorer/table1.py                         # Table 1, exact fractions
    python examples/figure2/reproduce_figure2.py                 # Figure 2, about a minute
    python examples/figure2/reproduce_figure2.py --from-levels   # the same in a second
    python examples/asrs_distilbert/recompute.py                 # ASRS with DistilBERT from scores, about 3 minutes

The Figure 2 script recomputes the five ASRS and MultiEURLEX rows and checks
them against stored values to 1e-10. The BiGRU rows come from raw scores in
this repository. The DistilBERT rows come from stored levels per pair. The
PubMed rows are stored in paper/results/pubmed_paired_refit/.

The paper's main result can also be recomputed from scores.
examples/asrs_distilbert/ holds the scores, labels and inherited thresholds of
the six ASRS DistilBERT pairs (6 MB, no text). Its script recomputes replay's
lead over sequential training at the inherited thresholds and after both
refits, and the decomposition of the sequential runs, and checks them against
the stored values to 1e-10. The MultiEURLEX and PubMed scores for DistilBERT
are not in this repository.

paper/README.txt maps every table and figure to its script and result files.
paper/plans/ holds the analysis plans. paper/checks/cold_install_0.2.0.txt
records a clean-environment run of the first three commands above.


Layout
------

    src/refitgap/   the evaluator
    examples/       quickstart, Table 1, Figure 2, ASRS from scores, your own models, a notebook
    paper/          plans, scripts, stored results and checks for the paper
    tests/          python -m pytest
    docs/           the figure above

MIT licence. If you use it, please cite the paper and the software
(CITATION.cff).
