#!/usr/bin/env python3
"""Per-label account of the change in mismatch on ASRS and the benchmark streams (restricted range): for the
final and the contemporary model, F1 at theta_1 and at the in-sample optimum, its distance from theta_1,
mean scores on positives and negatives and prevalences; in-sample optima, so r bounds the mismatch from above.
Reads results/e1r/primary_*_scores.npz and results/benchmarks/scores/; writes
results/benchmarks/perlabel_dR{,_wide}.parquet. Backs the label-by-label comparison of thresholds in Appendix D.
Needs the ASRS score files results/e1r/ (not shipped)."""
import glob, re
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from constructed_drifts import fit_exact

def f1k(p, y, t): return f1_score(y, (p >= t).astype(int), zero_division=0)

rows = []
for stream in ['primary', 'huffpost', 'amazon', 'arxivfine', 'arxiv']:
    files = sorted(glob.glob(f'results/e1r/{stream}_C2_e1r_s*_scores.npz')) if stream == 'primary' \
        else sorted(glob.glob(f'results/benchmarks/scores/{stream}_C2_s*_scores.npz'))
    for f in files:
        seed = int(re.search(r'_s(\d+)_scores', f).group(1)); z = np.load(f)
        th1 = z['thresholds'].astype(np.float64)
        N = max(int(k.split('_')[1]) for k in z.files if k.startswith('P_')) + 1
        prev1 = z['Y_1'].astype(int).mean(0) if 'Y_1' in z.files else z['Y_0'].astype(int).mean(0)
        for j in range(1, N - 1):
            Y = z[f'Y_{j}'].astype(int)
            for model, P in (
                ('final', z[f'P_{N-1}_{j}'].astype(np.float64)),
                ('contemporary', z[f'P_{j}_{j}'].astype(np.float64)),
            ):
                ts = fit_exact(P, Y)
                for k in range(Y.shape[1]):
                    y = Y[:, k]
                    if y.sum() == 0: continue
                    u = f1k(P[:, k], y, th1[k]); v = f1k(P[:, k], y, ts[k])
                    rows.append(
                        dict(
                            stream=stream,
                            seed=seed,
                            j=j,
                            model=model,
                            label=k,
                            prev_j=y.mean(),
                            prev_1=prev1[k],
                            theta1=th1[k],
                            theta_star=ts[k],
                            dist=abs(ts[k] - th1[k]),
                            u=u,
                            v=v,
                            r=v - u,
                            mean_pos=P[y == 1, k].mean(),
                            mean_neg=P[y == 0, k].mean(),
                            frac_above=(P[:, k] >= th1[k]).mean(),
                            frac_above_star=(P[:, k] >= ts[k]).mean(),
                        )
                    )
        print(f'  {stream} seed {seed} done', flush=True)
t = pd.DataFrame(rows)
t.to_parquet('results/benchmarks/perlabel_dR.parquet', index=False)

w = t.pivot_table(
    index=['stream', 'seed', 'j', 'label'],
    columns='model',
    values=['r', 'dist', 'mean_pos', 'mean_neg', 'frac_above', 'theta_star', 'u', 'v'],
).reset_index()
w.columns = ['_'.join(c).strip('_') for c in w.columns]
w['dR'] = w.r_final - w.r_contemporary
w['ddist'] = w.dist_final - w.dist_contemporary
w['dpos'] = w.mean_pos_final - w.mean_pos_contemporary
w['dneg'] = w.mean_neg_final - w.mean_neg_contemporary
w['dfrac'] = w.frac_above_final - w.frac_above_contemporary
w['dtheta_star'] = w.theta_star_final - w.theta_star_contemporary
w.to_parquet('results/benchmarks/perlabel_dR_wide.parquet', index=False)
pd.set_option('display.width', 250)
print('\nPer stream, mean over seeds, periods and labels (restricted range); in-sample optima:')
print(
    w.groupby('stream')[
        [
            'r_contemporary',
            'r_final',
            'dR',
            'dist_contemporary',
            'dist_final',
            'ddist',
            'dpos',
            'dneg',
            'dfrac',
            'dtheta_star',
        ]
    ]
    .mean()
    .round(4)
)
print('\nShare of labels with dR > 0 and mean |dR| per stream:')
print(
    w.groupby('stream')
    .apply(lambda d: pd.Series(dict(share_dR_pos=(d.dR > 0).mean(), mean_abs_dR=d.dR.abs().mean(), sd_dR=d.dR.std())))
    .round(3)
)
print('\nASRS per label (mean over seeds and periods):')
print(
    w[w.stream == 'primary']
    .groupby('label')[
        ['theta_star_contemporary', 'theta_star_final', 'r_contemporary', 'r_final', 'dR', 'dpos', 'dneg', 'dfrac']
    ]
    .mean()
    .round(4)
)
