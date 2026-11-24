ASRS DistilBERT paired refit from scores: python examples/asrs_distilbert/recompute.py [--workers N]
scores/primary_distilbert_{replay,naive}_s<k>_scores.npz, k = 0..5: one file per run, naive = sequential training,
same k = same initialisation (pair). Each holds P_7_j (final model on period j's test records), P_j_j (model j on
its own period, for the decomposition), Y_j (0/1 labels, identical across runs), thresholds (inherited, float32),
labels (names), seed, arm; j = 0..7, scores float16 as stored, rows in the original record order. No text.
Label (column) order: ATC Issue, Aircraft Equipment Problem, Conflict, Deviation - Altitude, Deviation - Track /
Heading, Deviation / Discrepancy - Procedural, Flight Deck / Cabin / Aircraft Event, Ground Event / Encounter,
Inflight Event / Encounter. Periods are 0-indexed in the files; the paper's periods 2 to 7 are ids 1-6.
Leads are replay minus sequential (positive = replay higher); change = refitted lead - inherited lead. Expected:
inherited +0.0109 [+0.0089, +0.0129], refitted -0.0004 [-0.0030, +0.0022], change -0.0113 [-0.0138, -0.0088];
B +0.0168, dR +0.0187, A -0.0020 (Table C.4); "match" and exit 0 at tolerance 1e-10. About 3 minutes on 2 cores.
