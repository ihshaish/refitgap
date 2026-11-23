#!/usr/bin/env python3
"""Figure 3: replay's lead over sequential training under a carried rule set,
minus its lead after both models are refitted; DistilBERT, periods 2 to 7,
on three streams (exact values in supplement Table C.7).

    python scripts/figure_carried_lead.py [OUTDIR]      (run from paper/)

Layout. Full text width, three panels side by side (ASRS, MultiEURLEX,
PubMed, with the number of pairs under each name) sharing four rows and one
x range. x is the lead under the carried rule minus the lead after both
refits, in macro-F1, positive where the carried rule gives replay the larger
lead. Rows, top to bottom: inherited thresholds, cut of 0.5, own validation,
previous period's rule. A point is the mean over pairs and the capped bar
through it the 95% paired t interval. The first three rows (filled circle)
are set against the full refit; the previous period's rule (open circle,
below a thin rule) against the refit on 500 records. A dotted hairline marks
zero. Ink only, no margin band and no numbers on the artwork; frame,
gridlines, face and sizes are those of Figure 2.

Data: paper/results/carried_rules/SUMMARY.json, item1.comparisons
["<stream>: replay minus sequential training"].rules:
- theta1, cut0.5, own_val: rules[r]["G_minus_G_V"].{mean, t95, per_pair};
- prev_period: rules["prev_period"]["G_minus_G_V500"].{mean, t95, per_pair}.
The interval is recomputed from the per-pair values and checked against the
stored one.

Checks. Every drawn mean and interval end equals, to four decimals, the line
printed in paper/results/carried_rules/report_tables.txt, item 1. No label
overlaps another or reaches the panel, and nothing is drawn outside the
figure box.

Output: OUTDIR/fig_carried_lead.pdf (default OUTDIR: paper/figures/).
"""
import json
import os
import re
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER, "figures")
SUMMARY = os.path.join(PAPER, "results", "carried_rules", "SUMMARY.json")
REPORT = os.path.join(PAPER, "results", "carried_rules", "report_tables.txt")

STREAMS = ["ASRS", "MultiEURLEX", "PubMed"]
# rule key, field, row label, start of the printed row, which bracketed value on that row
ROWS = [
    ("theta1", "G_minus_G_V", "inherited thresholds", "a inherited thresholds", 1),
    ("cut0.5", "G_minus_G_V", "cut of 0.5", "b cut of 0.5", 1),
    ("own_val", "G_minus_G_V", "own validation", "c own validation", 1),
    ("prev_period", "G_minus_G_V500", "previous period's rule", "G_d - G_V500", 0),
]


def f4(x):
    s = f"{x:+.4f}"
    return "+0.0000" if s == "-0.0000" else s


# ------------------------------------------------------------------ sources
comps = json.load(open(SUMMARY))["item1"]["comparisons"]
DATA, NPAIRS = {}, {}
for s in STREAMS:
    c = comps[f"{s}: replay minus sequential training"]
    assert c["arms"] == ["replay", "sequential training"], c["arms"]
    NPAIRS[s] = c["n_pairs"]
    for key, field, *_ in ROWS:
        e = c["rules"][key][field]
        v = np.array(e["per_pair"])
        assert len(v) == e["n"] == c["n_pairs"]
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        assert np.allclose([v.mean(), v.mean() - h, v.mean() + h], [e["mean"], *e["t95"]],
                           atol=1e-12), (s, key)
        DATA[s, key] = (e["mean"], *e["t95"])

# ------------------------------------------------------------------ check against printed values
rep = open(REPORT).read()
rep = rep[rep.index("ITEM 1"):rep.index("ITEM 2a")]
VAL = r"([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]"
for s in STREAMS:
    m = re.search(rf"^{s}: replay minus sequential training \((\d+) pairs\)", rep, re.M)
    assert int(m.group(1)) == NPAIRS[s], s
    blk = rep[m.end():]
    blk = blk[blk.index("\n") + 1:]             # rows start on the line after the heading
    nxt = re.search(r"^\S", blk, re.M)          # the next comparison starts at the margin
    blk = blk[:nxt.start()] if nxt else blk
    for key, field, _lab, name, which in ROWS:
        line = [l for l in blk.splitlines() if l.strip().startswith(name)]
        assert len(line) == 1, (s, name)
        want = tuple(f4(float(x)) for x in re.findall(VAL, line[0])[which])
        drawn = tuple(f4(x) for x in DATA[s, key])
        assert drawn == want, ("drawn values differ from report_tables.txt", s, key, drawn, want)
print(f"check passed: {len(DATA)} cells x (mean + 2 interval ends) equal report_tables.txt item 1 to 4 dp")

# ------------------------------------------------------------------ layout
F.apply("cm")
FS_ROW, FS_HEAD, FS_TICK = 10.0, 10.5, 10.0          # as in Figure 2
plt.rcParams.update({"xtick.labelsize": FS_TICK})
W, H_CM = F.W2, 6.0
fig = plt.figure(figsize=(W, H_CM / 2.54))
L, R, BOTTOM, TOP = 0.272, 0.988, 0.235, 0.832
GAP = 0.016
PW = (R - L - 2 * GAP) / 3
Y = [0.0, 1.0, 2.0, 3.45]                             # the last row stands apart
Y_RULE = 2.72
CAP_H = 0.34      # caps taller than the marker, so a narrow interval still shows its ends
MS = 3.8
allv = [v for d in DATA.values() for v in d[1:]] + [0.0]
XLIM = (min(allv) - 0.0015, max(allv) + 0.0015)
X_PER_PT = (XLIM[1] - XLIM[0]) / (PW * W * 72)        # data units per point

axes, texts = [], []
for k, s in enumerate(STREAMS):
    ax = fig.add_axes([L + k * (PW + GAP), BOTTOM, PW, TOP - BOTTOM])
    axes.append(ax)
    F.refline(ax, 0.0, "zero")
    ax.axhline(Y_RULE, color=F.GRID, lw=0.6, zorder=1)
    for yi, (key, *_r) in zip(Y, ROWS):
        m, lo, hi = DATA[s, key]
        is_open = key == "prev_period"
        F.point_whisker(ax, yi, m, lo, hi, F.INK, orient="h", ms=MS, lw=1.0, marker="o",
                        open_marker=is_open, mew=0.9)
        for xv in (lo, hi):
            # a cap that would fall on the marker's face goes beneath the marker
            inside = abs(xv - m) < (MS / 2 + 1.0) * X_PER_PT
            ax.plot([xv, xv], [yi - CAP_H / 2, yi + CAP_H / 2], color=F.INK, lw=1.0,
                    zorder=5.5 if inside else 7, solid_capstyle="butt")
    ax.set_xlim(*XLIM)
    ax.set_ylim(Y[-1] + 0.62, Y[0] - 0.62)
    ax.set_xticks([0.0, 0.01])
    ax.set_xticks([-0.005, 0.005, 0.015], minor=True)
    ax.tick_params(axis="x", which="minor", length=1.8)
    ax.tick_params(axis="x", pad=2)
    F.trim_ticks(ax.xaxis)
    ax.set_yticks([])
    F.frame(ax, grid="x")
    xc = L + k * (PW + GAP) + PW / 2
    texts.append(fig.text(xc, TOP + 0.090, s, ha="center", va="baseline", fontsize=FS_HEAD,
                          color=F.INK))
    texts.append(fig.text(xc, TOP + 0.022, f"{NPAIRS[s]} pairs", ha="center", va="baseline",
                          fontsize=FS_TICK, color=F.INK))
tr = axes[0].get_yaxis_transform()
XLAB = -(L - 0.006) / PW
row_texts = []
for yi, (_k, _f, lab, *_r) in zip(Y, ROWS):
    row_texts.append(axes[0].text(XLAB, yi, lab, transform=tr, ha="left", va="center",
                                  fontsize=FS_ROW, color=F.INK, clip_on=False))
xlab = fig.text(0.5, 0.030, "Lead under the carried rule minus lead after both refits (macro-F1)",
                ha="center", va="baseline", fontsize=FS_ROW, color=F.INK)
texts += row_texts + [xlab]

# ------------------------------------------------------------------ layout checks
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_bbs = [(t.get_text(), t.get_window_extent(_r)) for t in texts]
for ax in axes:
    _bbs += [(t.get_text(), t.get_window_extent(_r)) for t in ax.get_xticklabels() if t.get_text()]
for i in range(len(_bbs)):
    for j in range(i + 1, len(_bbs)):
        assert not _bbs[i][1].overlaps(_bbs[j][1]), ("labels overlap", _bbs[i][0], _bbs[j][0])
_x_panel = axes[0].get_window_extent(_r).x0
for t in row_texts:
    assert t.get_window_extent(_r).x1 < _x_panel - 3, ("label reaches the panel", t.get_text())
F.check_fits(fig)
os.makedirs(OUTDIR, exist_ok=True)
F.save(fig, os.path.join(OUTDIR, "fig_carried_lead"))
print(f"written {os.path.join(OUTDIR, 'fig_carried_lead.pdf')} ({W * 2.54:.1f} x {H_CM:.1f} cm)")
for s in STREAMS:
    for key, field, lab, *_r in ROWS:
        m, lo, hi = DATA[s, key]
        print(f"{s:12s} n={NPAIRS[s]}  {lab:24s} {field:15s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
