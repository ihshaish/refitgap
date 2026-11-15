Paper materials
===============

Plans, scripts, stored results, two checks and the data notes. Run scripts from this folder
(python scripts/NAME.py); paths below are relative to it. "Needs" says what a rerun wants
beyond the repository: "-" means nothing, "scores" score files we keep outside it (three
scripts take their location as an argument), "ckpt" the trained checkpoints and the streams,
"corpora" the raw data (DATA.txt). Stored files call sequential training "naive",
C1 to C4 the embedding conditions and lam the embedding learning-rate multiplier.

Paper item              Script (scripts/)                    Stored result (results/)                 Needs
---------------------   ----------------------------------   --------------------------------------   -------
Figure 2, Table C.1     ../examples/figure2/                 paired_refit/, ../examples/figure2/      -
  BiGRU rows            paired_refit_bigru.py                paired_refit/paired_replay*.json         -
  DistilBERT rows       paired_refit_distilbert.py           expected/paired_refit_distilbert.json    scores
Table 1                 ../examples/two_scorer/table1.py     (exact fractions)                        -
Table D.1               decomposition_benchmarks.py          benchmarks/, asrs_tuned/                 -
                        decomposition_multieurlex.py         multieurlex_bigru/eurlex_bridge*         -
                        decomposition_multieurlex_periods.py multieurlex_bigru/eurlex_per_period_ci   -
Table D.2               benchmark_vs_asrs.py,                asrs_supplementary/benchmark_*.json      -
                        benchmark_gap_minus_mismatch.py
Table D.3, Figure 3     decomposition_asrs.py                asrs_inherited_rule/bwt_bridge*          ckpt
Table D.4               bwt_statistics.py                    asrs_inherited_rule/stats_addenda.json   corpora
App. D, labels          per_label_mismatch.py               benchmarks/perlabel_dR*.parquet          scores
Table C.4               decomposition_distilbert.py          distilbert_decomposition/                -
App. A known cases,     asrs_supplementary.py (subcommands)  asrs_supplementary/*.json                scores
  budgets, records; App. D sensitivity
App. A budgets          simulation_finite_labels.py          asrs_supplementary/simulation_Vn.json    -
                        simulation_regret.py                 paired_refit/simulation_regret.json      - (hours)
App. A threshold fits   constructed_drifts.py                constructed_drift_*, exact_fitter/       ckpt
App. B, ASRS            asrs_profile.py, prior_movement_streams.py   benchmarks/prior_movement_streams.csv   corpora
App. D threshold update threshold_update.py                  asrs_inherited_rule/threshold_update*,   -
                                                             capture_deployed*
                        threshold_update_sample_size.py      calib_experiment_v2.parquet              ckpt
                        threshold_update_lagged.py           calib_lagged.parquet                     ckpt
                        operational_counts.py                asrs_inherited_rule/operational_counts*  -
                        prior_movement_valsplit.py (FITTER=grid)  prior_movement_valsplit_grid/       ckpt
                        trainable_embedding_gap.py           oracle_trainable.parquet                 ckpt
                        score_inherited_rule.py              asrs_inherited_rule/scores, deployed_*   ckpt
App. D, E AUROC         auroc_prior_shift.py                 auroc.parquet, auroc_priorshift.csv      ckpt
Table E.1               constructed_drift_components.py      asrs_inherited_rule/dissoc_components    -
                        constructed_drift_levels.py and its  calib_intervention_v2, calib_splits200*, ckpt
                        _control, _prior_200, _prior40_200,  calib_control_primary .parquet
                        _vocabulary_200 companions
App. E streams          check_vocabulary_pairing.py          asrs_supplementary/pairing_check.json    corpora
                        vocabulary_marginal.py               asrs_supplementary/vocab_marginal.json   corpora
App. E repeated         construction_seeds.py,               constructions/construction_seeds*        ckpt (- for
  constructions         construction_matched_pair.py                                                    the second)
App. E crossed terms    construction_crossed_terms.py        constructions/crossed_*                  ckpt
App. E both drifts      factorial_recompute.py               paired_refit/factorial_recompute.json    -
Table E.2               training_regime_factorial.py         training_regime/                         scores
Table G.2, Table 3      repair_vocabulary.py,                expected/repair_vocabulary_inputs.json,  scores
                        repair_multieurlex.py                expected/repair_multieurlex.json

Behind all of these, needing corpora and training: build_*.py (streams), train_sequential.py,
train_methods.py (--strategy sequential|replay|ewc), train_distilbert.py, train_embedding.py,
store_asrs_scores.py; fetch_data.py and check_data_layout.py help with the downloads;
exact_fitters.py and decomposition.py are imported by the others. prior_movement_asrs.py
holds functions that prior_movement_valsplit.py reuses.

No script here, numbers stored or not at all: results/exact_fitter/ and a few loose files
(e1_results.parquet, prior_only_baseline.csv, constructions/training_rows/) come from the
training harness and an exact-fitter rescoring run in our working tree. The shared-rule cost
and the method comparison (Tables F.1 to F.3), the forward comparison (Table H.1), dating a
lead (Table F.4) and the DistilBERT repair rest on runs too large to ship; their plans are in
plans/, their outputs are not.

    plans/      the analysis plans, one short note each; plans/README.txt is the index
    checks/     the same-fold scoring check and the record order check, with their outputs
    streams/    per-stream metadata and a public-domain ASRS sample
    threshold_policy_survey.txt   our coding of the 34 studies on threshold reporting
    requirements.txt              the analysis dependencies (training wants torch etc.)
