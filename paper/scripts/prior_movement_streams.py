#!/usr/bin/env python3
"""Label-prior movement per stream: mean absolute change in per-label prevalence and log-odds
between the first and last period, on the test split and the training pool, and per label.
Reads streams/stream_*.parquet; writes results/benchmarks/prior_movement_streams.csv.
Backs the ASRS label priors of Appendix B (5% to 18%) and the prior comparison in Appendix D.
Needs the stream files built by scripts/build_*.py from the corpora (not shipped)."""
import numpy as np, pandas as pd
rows = []
lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for s in ["primary", "huffpost", "amazon", "arxivfine", "arxiv"]:
    st = pd.read_parquet(f"streams/stream_{s}.parquet"); labels = sorted({l for x in st.labels for l in x.split(";")})
    for split in ("test", "train_pool"):
        d = st[st.split == split]
        pv = np.array(
            [
                [d[d.period == p].labels.str.split(";").apply(lambda L: l in L).mean() for l in labels]
                for p in sorted(d.period.unique())
            ]
        )
        rows.append(
            dict(
                stream=s,
                split=split,
                labels=len(labels),
                periods=len(pv),
                mean_abs_dprior=float(np.abs(pv[-1] - pv[0]).mean()),
                mean_abs_dlogodds=float(np.abs(lo(pv[-1]) - lo(pv[0])).mean()),
                mean_prior=float(pv.mean()),
            )
        )
        if split == "train_pool":
            for k, l in enumerate(labels):
                rows.append(
                    dict(
                        stream=s,
                        split="train_pool_label",
                        label=l,
                        prev_first=float(pv[0, k]),
                        prev_last=float(pv[-1, k]),
                        dlogodds=float(lo(pv[-1, k]) - lo(pv[0, k])),
                    )
                )
t = pd.DataFrame(rows); t.to_csv("results/benchmarks/prior_movement_streams.csv", index=False)
print(t[t.split == "test"][["stream", "mean_abs_dlogodds"]].to_string(index=False))
print(
    t[(t.split == "train_pool_label") & (t.stream == "primary")][["label", "prev_first", "prev_last", "dlogodds"]]
    .round(3)
    .to_string(index=False)
)
