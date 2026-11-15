#!/usr/bin/env python3
"""Crossed-environment terms per repeated prior-reweighting construction: control and intervention final
models each scored on both streams' test records, V_ab by exact split-half (partitions shared by the four
cells of a period), with A_net, A0, A1, E0, E1 and J. python scripts/construction_crossed_terms.py [SEEDS]
Writes results/constructions/crossed_<seed>.parquet and crossed_terms.json. Backs Appendix E,
"Crossed-environment terms". Needs the construction streams and checkpoints (not shipped)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts")
from train_sequential import KNOBS, Vocab, BiGRU, evaluate, load_vectors
from constructed_drifts import fit_exact
from sklearn.metrics import f1_score

OUT = Path("results/constructions")
SEEDS_C = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(20260921, 20260929))
SPLITS = 200
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vecs = load_vectors("glove"); toks = list(vecs); vocab = Vocab(toks, grow=False)


def mf1(P, Y, th): return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def load_stream(name):
    df = pd.read_parquet(f"streams/stream_{name}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")}); li = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"): Y[r, li[l]] = 1
    N = int(df.period.max()) + 1
    tests = {j: (df[(df.period == j) & (df.split == "test")], None) for j in range(N)}
    codes = {j: [vocab.encode(t, False) for t in tests[j][0].text] for j in range(N)}
    Ys = {j: Y[tests[j][0].index.to_numpy()] for j in range(N)}
    return N, len(labels), codes, Ys


def load_model(name, seed, K):
    rng = np.random.default_rng(seed)
    mat = np.vstack(
        [np.zeros(KNOBS["emb_dim"]), rng.normal(0, 0.1, KNOBS["emb_dim"]), np.stack([vecs[t] for t in toks])]
    ).astype(np.float32)
    m = BiGRU(torch.tensor(mat), freeze=True, n_labels=K).to(device)
    sd = torch.load(f"checkpoints/{name}_C2_s{seed}_v2/period7.pt", map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); m.eval(); return m


def V(P, Y, rng):
    v = []
    for _ in range(SPLITS):
        perm = rng.permutation(len(Y)); h = len(Y) // 2; a, b = perm[:h], perm[h:]
        v.append(0.5 * (mf1(P[b], Y[b], fit_exact(P[a], Y[a])) + mf1(P[a], Y[a], fit_exact(P[b], Y[b]))))
    return float(np.mean(v))


res = {}
for S in SEEDS_C:
    ctrl, arm = f"asrsprior{S}_00", f"asrsprior{S}_100"
    try:
        N, K, c_codes, c_Y = load_stream(ctrl); _, _, a_codes, a_Y = load_stream(arm)
    except FileNotFoundError:
        continue
    rows = []
    for seed in (0, 1):
        if not (
            Path(f"checkpoints/{ctrl}_C2_s{seed}_v2/period7.pt").exists()
            and Path(f"checkpoints/{arm}_C2_s{seed}_v2/period7.pt").exists()
        ):
            continue
        mc, ma = load_model(ctrl, seed, K), load_model(arm, seed, K)
        for j in range(N):
            cells = {}
            for mname, m in (("0", mc), ("1", ma)):
                for ename, (codes, Y) in (("0", (c_codes[j], c_Y[j])), ("1", (a_codes[j], a_Y[j]))):
                    _, _, P, _, _ = evaluate(m, codes, Y, np.full(K, .5), device)
                    cells[mname + ename] = V(P.astype(np.float64), Y, np.random.default_rng(7000 + 100 * seed + j))
            rows.append(dict(seed=seed, j=j, **{f"V{k}": v for k, v in cells.items()}))
        del mc, ma
    if not rows:
        continue
    t = pd.DataFrame(rows); m = t.groupby("seed")[["V00", "V01", "V10", "V11"]].mean().mean()
    res[str(S)] = dict(
        n_seeds=int(t.seed.nunique()),
        V00=m.V00,
        V01=m.V01,
        V10=m.V10,
        V11=m.V11,
        A_net=m.V00 - m.V11,
        A1_arm_env=m.V01 - m.V11,
        A0_ctrl_env=m.V00 - m.V10,
        E0=m.V00 - m.V01,
        E1=m.V10 - m.V11,
        J=(m.V00 - m.V01) - (m.V10 - m.V11),
    )
    print(S, {k: round(v, 4) for k, v in res[str(S)].items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).assign(construction=S).to_parquet(OUT / f"crossed_{S}.parquet", index=False)
(OUT / "crossed_terms.json").write_text(json.dumps(res, indent=1)); print("wrote", OUT / "crossed_terms.json")
