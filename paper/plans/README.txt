Analysis plans
==============

Each plan was written down, dated, before the runs it governs; where it was not, the note
says so. A note gives the aim, design, analysis, predictions, date and any departure that
changes a reading. The full dated originals are in a private archive, available to reviewers
on request.

Periods count from 1 to 8 as in the paper, except in the repair notes, which keep their
originals' 0 to 7.

study-design.txt                  2 Aug 2026   streams, learners, inherited thresholds, estimator
vocabulary-stream.txt             8 Aug 2026   the vocabulary stream
reweighting-stream.txt            8 Aug 2026   the reweighted stream
factorial-remedies-cumulative.txt 25 Aug 2026  both drifts crossed; prior-odds, cumulative baselines
arxiv-controlled-study.txt        28 Aug 2026  arXiv study, stopped at criterion (+0.0087 < 0.010)
training-regimes.txt              29 Aug 2026  epochs by embedding learning rate
repeated-constructions.txt        21 Sep 2026  eight more construction seeds per drift
benchmark-decomposition.txt       11 Oct 2026  B = A + Delta R on the public benchmark streams
distilbert-decomposition.txt      13 Oct 2026  the DistilBERT decomposition on ASRS
multieurlex-decomposition.txt     22 Oct 2026  the MultiEURLEX stream and its decomposition
bigru-paired-refit.txt            22 Oct 2026  BiGRU replay vs sequential training, both refits
method-comparison.txt             26 Oct 2026  five arms, two constructed pairs, five predictions
method-comparison-reproduction-check.txt
                                  28 Oct 2026  four of 40 runs beyond 1e-3; check waived
shared-rule-cost.txt              29 Oct 2026  what one rule across periods costs each arm
shallower-repairs.txt             29 Oct 2026  four repair depths on replay's final buffer
bigru-repair-vocabulary.txt       31 Oct 2026  full-model repair on new seeds, vocabulary stream
bigru-repair-multieurlex.txt      1 Nov 2026   full-model repair on MultiEURLEX; dating a lead
distilbert-paired-refit.txt       9 Nov 2026   DistilBERT on ASRS and MultiEURLEX, both refits
forward-comparison.txt            9 Nov 2026   stale models scored forward, by lag
distilbert-matched-updates.txt    10 Nov 2026  new ASRS pairs; replay vs extra updates on current data
distilbert-repair.txt             13 Nov 2026  full-model repair of the final DistilBERT
pubmed-paired-refit.txt           16 Nov 2026  four arms on a PubMed stream, fixed before its runs
pubmed-lwf-selection.txt          16 Nov 2026  LwF setting chosen in development, before confirmation
carried-rules.txt                 19 Nov 2026  carried rules against the refitted lead; records
                                               and labels resampled; LwF against replay.
                                               Designed after the results

The common-threshold check (22 Nov 2026, designed after the results) is defined in
../results/carried_rules/common_thresholds/DEFINITION.txt.
