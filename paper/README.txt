Paper materials
===============

Plans, scripts, stored results, checks and data notes. Run scripts from this folder
(python scripts/NAME.py); paths below are relative to it. "Needs" says what a rerun wants
beyond the repository: "-" nothing, "scores" score files kept outside it (the scripts take
their location as an argument), "ckpt" the trained checkpoints and the streams, "corpora" the
raw data (DATA.txt). Stored files call sequential training "naive", C1 to C4 the embedding
conditions, lam the embedding learning-rate multiplier and replaycur extra updates on
current data.

Paper item              Script (scripts/)                    Stored result (results/)                 Needs
---------------------   ----------------------------------   --------------------------------------   -------
Table 1                 ../examples/two_scorer/table1.py     (exact fractions)                        -
Figure 2, Table C.1     figure_paired_refit.py (drawing)     the files below                          -
  ASRS, MultiEURLEX     ../examples/figure2/                 paired_refit/, ../examples/figure2/      -
  BiGRU rows            paired_refit_bigru.py                paired_refit/paired_replay*.json         -
  DistilBERT rows       paired_refit_distilbert.py           expected/paired_refit_distilbert.json    scores
    ASRS row            ../examples/asrs_distilbert/         (scores and labels shipped there)        -
  PubMed rows           paired_refit_pubmed.py               pubmed_paired_refit/                     scores
  BiGRU LwF rows        carried_rules_lwf.py                 carried_rules/item3.json                 scores
Table C.3               ../examples/figure2/                 ../examples/figure2/distilbert_levels.csv -
Table C.4               decomposition_distilbert.py          distilbert_decomposition/                -
  paired refit's runs   ../examples/asrs_distilbert/         (scores and labels shipped there)        -
Table C.6               paired_refit_pubmed.py               pubmed_paired_refit/                     scores
App. C.2, both arms     carried_rules_common_thresholds.py   carried_rules/common_thresholds/         scores
  at one arm's rule
Figure 3, Table C.7     figure_carried_lead.py (drawing),    carried_rules/SUMMARY.json, item1.json,  scores
                        carried_rules_items1_2b.py           report_tables.txt
Table C.8, App. C.5     carried_rules_items1_2b.py,          carried_rules/item2b*, item2a_*          scores
                        carried_rules_records.py (hours)
Table C.9               carried_rules_constructed.py         carried_rules/item4.json                 scores
Table D.1               decomposition_benchmarks.py          benchmarks/, asrs_tuned/                 -
                        decomposition_multieurlex.py         multieurlex_bigru/eurlex_bridge*         -
                        decomposition_multieurlex_periods.py multieurlex_bigru/eurlex_per_period_ci   -
Table D.2               decomposition_asrs.py                asrs_inherited_rule/bwt_bridge*          ckpt
Table D.3               bwt_statistics.py                    asrs_inherited_rule/stats_addenda.json   corpora
Table D.4, App. D.1     per_label_mismatch.py                benchmarks/perlabel_dR*.parquet          scores
  thresholds by label
Figure 4                no drawing script; its rows come from Tables D.1, D.2 and E.2
App. A.1                constructed_drifts.py                constructed_drift_*, exact_fitter/       ckpt
App. A.2, A.3, A.6,     asrs_supplementary.py (subcommands)  asrs_supplementary/*.json                scores
  App. D.3
App. A.3                simulation_finite_labels.py          asrs_supplementary/simulation_Vn.json    -
                        simulation_regret.py                 paired_refit/simulation_regret.json      - (hours)
App. B.1                asrs_profile.py,                     benchmarks/prior_movement_streams.csv    corpora
                        prior_movement_streams.py
App. D.2, trainable     trainable_embedding_gap.py           oracle_trainable.parquet                 ckpt
  embeddings
App. D.4                threshold_update.py                  asrs_inherited_rule/threshold_update*,   -
                                                             capture_deployed*
                        threshold_update_sample_size.py      calib_experiment_v2.parquet              ckpt
                        threshold_update_lagged.py           calib_lagged.parquet                     ckpt
                        operational_counts.py                asrs_inherited_rule/operational_counts*  -
                        prior_movement_valsplit.py           prior_movement_valsplit_grid/            ckpt
                          (FITTER=grid)
                        score_inherited_rule.py              asrs_inherited_rule/scores, deployed_*   ckpt
App. D, E AUROC         auroc_prior_shift.py                 auroc.parquet, auroc_priorshift.csv      ckpt
Table E.1               constructed_drift_components.py      asrs_inherited_rule/dissoc_components    -
                        constructed_drift_levels.py and its  calib_intervention_v2, calib_splits200*, ckpt
                        _control, _prior_200, _prior40_200,  calib_control_primary .parquet
                        _vocabulary_200 companions
App. E.1                check_vocabulary_pairing.py          asrs_supplementary/pairing_check.json    corpora
                        vocabulary_marginal.py               asrs_supplementary/vocab_marginal.json   corpora
App. E.2                construction_seeds.py,               constructions/construction_seeds*        ckpt (- for
                        construction_matched_pair.py                                                    the second)
App. E.3                construction_crossed_terms.py        constructions/crossed_*                  ckpt
App. E.4                factorial_recompute.py               paired_refit/factorial_recompute.json    -
Table E.2               training_regime_factorial.py         training_regime/                         scores
Table G.2               repair_vocabulary.py,                expected/repair_vocabulary_inputs.json,  scores
                        repair_multieurlex.py                expected/repair_multieurlex.json

Training and data, which need the corpora: build_*.py (streams), train_sequential.py,
train_methods.py (--strategy sequential|replay|ewc), train_distilbert.py, train_embedding.py,
store_asrs_scores.py; fetch_data.py and check_data_layout.py help with the downloads.
Imported by the others: exact_fitters.py and decomposition.py; carried_rules_core.py (by the
carried_rules_*.py scripts); figstyle.py (by the two figure scripts); prior_movement_asrs.py
(by prior_movement_valsplit.py). carried_rules_summary.py, carried_rules_tables.py and
carried_rules_csvs.py rebuild SUMMARY.json, report_tables.txt and the CSVs in
results/carried_rules/ from the item outputs; the record-level CSVs of items 1, 3 and 4 (about
290 MB) are not stored.

benchmark_vs_asrs.py and benchmark_gap_minus_mismatch.py (asrs_supplementary/benchmark_*.json)
set each benchmark stream against ASRS; the current supplement does not report them.

Stored without a script: results/exact_fitter/, e1_results.parquet, prior_only_baseline.csv and
constructions/training_rows/ come from the training harness and an exact-fitter rescoring run.

results/forward_comparison/ holds the forward comparison by lag (plans/forward-comparison.txt):
forward_{primary,eurlex}_{naive,replay}.csv for DistilBERT on ASRS and MultiEURLEX, and
forward_bigru.csv for the BiGRU on the vocabulary stream, MultiEURLEX and arXiv cs
subcategories. The supplement reports it in summary only (Appendix H).

The method comparison (Tables F.1 and F.2), the shared-rule cost (Appendix F.3), the timing of
a lead (Table F.3) and the DistilBERT repair (Appendix G.3) rest on runs too large to ship;
their plans are in plans/, their outputs are not.

    plans/      the analysis plans, one short note each; plans/README.txt is the index
    checks/     the same-fold scoring and record order checks with their outputs, and the
                cold install of the 0.2.0 release (cold_install_0.2.0.txt)
    streams/    per-stream metadata and a public-domain ASRS sample
    threshold_policy_survey.txt   our coding of the 34 studies on threshold reporting (App. B.2)
    requirements.txt              the analysis dependencies (training wants torch etc.)
