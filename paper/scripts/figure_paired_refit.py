#!/usr/bin/env python3
"""Figure 2: the lead of one arm over another at the inherited thresholds and
after both models are refitted, for eight comparisons on three streams
(exact values in supplement Table C.1).

    python scripts/figure_paired_refit.py [OUTDIR]      (run from paper/)
    python paper/scripts/figure_paired_refit.py --png docs/paired_refit.png
                                                        (the README image; no PDF)

Layout. Eight rows in three stream groups (ASRS, MultiEURLEX, PubMed) on one
horizontal axis, the lead in macro-F1 of the first named arm over the second,
positive where the first arm leads. In each row an open circle, a little
above the row line, is the lead at the inherited thresholds, and a filled
orange diamond, a little below, the lead after both models are refitted;
each carries its capped 95% interval. A thin grey connector joins the two
means and shows no uncertainty of the change. A pale band marks the 0.005
margin either side of zero and a dotted hairline marks zero. Fill and shape
carry the marker distinction, so the figure reads in greyscale. Where an
interval is narrower than its marker the caps are drawn beneath the marker.

Data (paths relative to paper/results/):
- DistilBERT, ASRS and MultiEURLEX (six pairs, paired t interval over seeds):
  expected/paired_refit_distilbert.json, contrasts[name].terms
  {lead_at_carried_rule, lead_after_refit}.{stored_mean, stored_t95,
  per_pair}. The interval is recomputed from the per-pair leads and checked
  against the stored one.
- BiGRU, ASRS and MultiEURLEX (ten pairs, percentile bootstrap):
  paired_refit/paired_replay.json and paired_replay_eurlex.json,
  summary.{G_U, G_V}.{mean, ci}.
- DistilBERT, PubMed (eight pairs, paired t interval over initialisations):
  pubmed_paired_refit/pubmed_paired_refit.json, comparisons[name]
  .ranges["periods 2-7"].{G_U, G_V}.{mean, t95, per_pair}; the interval is
  recomputed and checked in the same way.

Checks. Every drawn mean and interval end, and the change between the two
means, equals to four decimals the values in examples/figure2/expected.json
for the five ASRS and MultiEURLEX rows, and the lines printed in
pubmed_paired_refit/pubmed_paired_refit.out for the three PubMed rows. No
label overlaps another or reaches into the panel, and nothing is drawn
outside the figure box.

Output: OUTDIR/fig_paired_refit.pdf (default OUTDIR: paper/figures/). With --png FILE,
only FILE: the same figure 1600 px wide on white, tagged 200 dpi.
"""
import json
import os
import re
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
RESULTS = os.path.join(PAPER, "results")
ARGS = sys.argv[1:]
PNG = None
if "--png" in ARGS:
    i = ARGS.index("--png")
    PNG = ARGS[i + 1]
    del ARGS[i:i + 2]
OUTDIR = ARGS[0] if ARGS else os.path.join(PAPER, "figures")
PNG_WIDTH_PX, PNG_DPI_TAG = 1600, 200
DISTILBERT_JSON = os.path.join(RESULTS, "expected", "paired_refit_distilbert.json")
EXPECTED_JSON = os.path.join(os.path.dirname(PAPER), "examples", "figure2", "expected.json")
PUBMED_JSON = os.path.join(RESULTS, "pubmed_paired_refit", "pubmed_paired_refit.json")
PUBMED_OUT = os.path.join(RESULTS, "pubmed_paired_refit", "pubmed_paired_refit.out")


def f4(x):
    s = f"{x:+.4f}"
    return "+0.0000" if s == "-0.0000" else s


def t_interval_matches(v, mean, lo, hi):
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return np.allclose([v.mean(), v.mean() - h, v.mean() + h], [mean, lo, hi], atol=1e-12)


# ------------------------------------------------------------------ sources
DB = json.load(open(DISTILBERT_JSON))["contrasts"]


def distilbert(name, direction):
    """(inherited, refitted), each (mean, lo, hi, n): paired t over seeds."""
    c = DB[name]
    assert c["direction"].startswith(direction), (name, c["direction"])
    out = []
    for term in ("lead_at_carried_rule", "lead_after_refit"):
        r = c["terms"][term]
        v = np.array(r["per_pair"])
        assert len(v) == 6, (name, term)
        assert t_interval_matches(v, r["stored_mean"], *r["stored_t95"]), (name, term)
        out.append((r["stored_mean"], *r["stored_t95"], len(v)))
    return out


def bigru(fname):
    """(inherited, refitted), each (mean, lo, hi, n): percentile bootstrap over pairs."""
    d = json.load(open(os.path.join(RESULTS, "paired_refit", fname)))
    assert d["n_pairs"] == 10
    s = d["summary"]
    return [(s[k]["mean"], *s[k]["ci"], d["n_pairs"]) for k in ("G_U", "G_V")]


def pubmed(comparison):
    """(inherited, refitted), each (mean, lo, hi, n): paired t over initialisations."""
    c = PUBMED["comparisons"][comparison]
    assert c["primary_range"] == "periods 2-7" and c["margin"] == 0.005
    r = c["ranges"]["periods 2-7"]
    assert r["periods_1_indexed"] == [2, 3, 4, 5, 6, 7] and r["n_pairs"] == 8
    out = []
    for k in ("G_U", "G_V"):
        v = np.array(r[k]["per_pair"])
        assert len(v) == 8
        assert t_interval_matches(v, r[k]["mean"], *r[k]["t95"]), (comparison, k)
        out.append((r[k]["mean"], *r[k]["t95"], len(v)))
    return out


PUBMED = json.load(open(PUBMED_JSON))
assert PUBMED["stream"] == "pubmed"

# row id -> (interval kind, (inherited, refitted)), in the order of Table C.1
DATA = {
    "db_asrs": ("t", distilbert("primary", "replay minus sequential")),
    "db_meur": ("t", distilbert("eurlex", "replay minus sequential")),
    "db_meur_ext": ("t", distilbert("eurlex_replay_vs_replaycur", "replay minus replaycur")),
    "gru_asrs": ("boot", bigru("paired_replay.json")),
    "gru_meur": ("boot", bigru("paired_replay_eurlex.json")),
}
EXPECTED_KEY = {"db_asrs": "distilbert_asrs_sequential",
                "db_meur": "distilbert_multieurlex_sequential",
                "db_meur_ext": "distilbert_multieurlex_extra_updates",
                "gru_asrs": "bigru_asrs_sequential",
                "gru_meur": "bigru_multieurlex_sequential"}
PM_ROWS = {"pm_seq": "replay minus sequential", "pm_lwf": "lwf minus sequential",
           "pm_ext": "replay minus extra updates"}

# ------------------------------------------------------------------ checks against printed values
expected = json.load(open(EXPECTED_JSON))
for rid, (kind, (inh, ref)) in DATA.items():
    e = expected[EXPECTED_KEY[rid]]
    assert e["pairs"] == inh[3] == ref[3], rid
    drawn = (f4(inh[0]), f4(ref[0]), f4(inh[1]), f4(inh[2]), f4(ref[1]), f4(ref[2]))
    want = (f4(e["inherited"]["mean"]), f4(e["refitted"]["mean"]), *map(f4, e["inherited"]["ci"]),
            *map(f4, e["refitted"]["ci"]))
    assert drawn == want, ("drawn values differ from expected.json", rid, drawn, want)
    assert f4(ref[0] - inh[0]) == f4(e["change"]["mean"]), ("change", rid)
print("check passed: 5 rows x (2 means + 4 interval ends + change) equal "
      "examples/figure2/expected.json to 4 dp")

out_txt = open(PUBMED_OUT).read()
NUMP = r"([+-]\d\.\d{4})"
for rid, comp in PM_ROWS.items():
    DATA[rid] = ("t", pubmed(comp))
    sec = out_txt[out_txt.index("=== " + comp + ","):]
    sec = sec[sec.index("periods 2-7 (primary)"):]
    for k, (m, lo, hi, _n) in zip(("G_U", "G_V"), DATA[rid][1]):
        g = re.search(r"^  " + k + r"\s+" + NUMP + r"\s+t95 \[" + NUMP + ", " + NUMP + r"\]", sec, re.M)
        want = tuple(f4(float(x)) for x in g.groups())
        assert (f4(m), f4(lo), f4(hi)) == want, ("drawn values differ from the printed .out", rid, k, want)
print("check passed: 3 PubMed rows x (2 means + 4 interval ends) equal pubmed_paired_refit.out to 4 dp")

# ------------------------------------------------------------------ layout
LAYOUT = [
    ("head", "ASRS"),
    ("row", "db_asrs", "DistilBERT: replay over\nsequential training"),
    ("row", "gru_asrs", "BiGRU: replay over\nsequential training"),
    ("head", "MultiEURLEX"),
    ("row", "db_meur", "DistilBERT: replay over\nsequential training"),
    ("row", "db_meur_ext", "DistilBERT: replay over extra\nupdates on current data"),
    ("row", "gru_meur", "BiGRU: replay over\nsequential training"),
    ("head", "PubMed"),
    ("row", "pm_seq", "DistilBERT: replay over\nsequential training"),
    ("row", "pm_lwf", "DistilBERT: LwF over\nsequential training"),
    ("row", "pm_ext", "DistilBERT: replay over extra\nupdates on current data"),
]
assert {i[1] for i in LAYOUT if i[0] == "row"} == set(DATA)

F.apply("cm")
FS_ROW, FS_HEAD, FS_TICK, FS_KEY = 10.0, 10.5, 10.0, 10.0
plt.rcParams.update({"xtick.labelsize": FS_TICK})
y, ids, labels, heads = [], [], [], []
cur = 0.0
for k, item in enumerate(LAYOUT):
    if item[0] == "head":
        if k:
            cur -= 0.25
        heads.append((item[1], cur - 0.42))
        cur -= 0.80
    else:
        y.append(cur - 0.50)
        ids.append(item[1])
        labels.append(item[2])
        cur -= 1.0
Y_TOP, Y_BOT = 0.0, cur - 0.10

W = F.W2
# Height scales with the number of rows: the row pitch and the space above and
# below the panel are those of a reference layout 7.2 cm tall whose panel runs
# from 0.185 to 0.925 of the height and spans 6.95 row units.
H0_CM, B0, T0, UNITS0 = 7.2, 0.185, 0.925, 6.95
H_CM = round(H0_CM * (1 - (T0 - B0)) + (Y_TOP - Y_BOT) * H0_CM * (T0 - B0) / UNITS0, 1)
fig = plt.figure(figsize=(W, H_CM / 2.54))
L, R = 0.450, 0.985
BOTTOM, TOP = B0 * H0_CM / H_CM, 1 - (1 - T0) * H0_CM / H_CM
ax = fig.add_axes([L, BOTTOM, R - L, TOP - BOTTOM])
MARGIN = 0.005
DY = 0.21                     # open marker above the row line, filled below
assert 2 * DY < 0.75 * (1.0 - 2 * DY)   # a row's two markers are nearer each other than the next row
CAP_H = 0.22
STY = {"inh": dict(marker="o", ms=4.4, open_marker=True, dy=+DY),
       "ref": dict(marker="D", ms=4.0, open_marker=False, dy=-DY)}
COL = {"inh": F.INK, "ref": F.A_COL}     # refitted lead: the attainable-gap orange
CONNECTOR = dict(color=F.GREY, lw=0.6)   # thin neutral grey; it carries no quantity colour

ax.axvspan(-MARGIN, MARGIN, facecolor="#e9e9e9", edgecolor="none", zorder=0)
F.refline(ax, 0.0, "zero")
allv = [-MARGIN, MARGIN]
for rid in ids:
    for e in DATA[rid][1]:
        allv += [e[1], e[2]]
XLIM = (min(allv) - 0.002, max(allv) + 0.0015)
ax.set_xlim(*XLIM)
X_PER_PT = (XLIM[1] - XLIM[0]) / ((R - L) * W * 72)     # data units per point
for yi, rid in zip(y, ids):
    inh, ref = DATA[rid][1]
    ax.plot([inh[0], ref[0]], [yi + DY, yi - DY], zorder=3, **CONNECTOR,
            solid_capstyle="butt")
    for key, (m, lo, hi, _n) in (("inh", inh), ("ref", ref)):
        st = STY[key]
        yy = yi + st["dy"]
        F.point_whisker(ax, yy, m, lo, hi, COL[key], orient="h", ms=st["ms"], lw=1.0,
                        marker=st["marker"], open_marker=st["open_marker"], mew=0.9)
        for xv in (lo, hi):
            # a cap that would fall on the marker's face goes beneath the marker
            inside = abs(xv - m) < (st["ms"] / 2 + 1.0) * X_PER_PT
            ax.plot([xv, xv], [yy - CAP_H / 2, yy + CAP_H / 2], color=COL[key], lw=1.0,
                    zorder=5.5 if inside else 7, solid_capstyle="butt")
ax.set_ylim(Y_BOT, Y_TOP)
ax.set_xlim(*XLIM)
ax.set_xticks([0.0, 0.01, 0.02, 0.03])
ax.set_xticks([-0.005, 0.005, 0.015, 0.025], minor=True)
ax.tick_params(axis="x", which="minor", length=1.8)
ax.tick_params(axis="x", pad=2)
F.trim_ticks(ax.xaxis)
ax.set_yticks([])
F.frame(ax, grid="x")
ax.set_xlabel("Lead in macro-F1", fontsize=FS_ROW, labelpad=3)
# the margin band is labelled directly, above the panel
band_lab = ax.text(0.0, 1.015, "$\\pm 0.005$ margin", transform=ax.get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=FS_KEY, color=F.INK, clip_on=False)
# group and row labels, left of the panel
tr = ax.get_yaxis_transform()
XLAB = -(L - 0.004) / (R - L)
XROW = XLAB + 0.035
texts = [band_lab]
for gname, yh in heads:
    texts.append(ax.text(XLAB, yh, gname, transform=tr, ha="left", va="center", fontsize=FS_HEAD,
                         color=F.INK, clip_on=False))
for yi, lab in zip(y, labels):
    texts.append(ax.text(XROW, yi, lab, transform=tr, ha="left", va="center", fontsize=FS_ROW,
                         color=F.INK, clip_on=False, linespacing=1.0))
# key for the two markers, bottom left, beside the axis
for yk, key, text in ((0.118, "inh", "inherited thresholds"), (0.040, "ref", "both models refitted")):
    yk = yk * H0_CM / H_CM
    st = STY[key]
    fig.add_artist(Line2D([0.012, 0.062], [yk, yk], color=COL[key], lw=1.0, transform=fig.transFigure))
    fig.add_artist(Line2D([0.037], [yk], marker=st["marker"], ms=st["ms"], mec=COL[key], mew=0.9,
                          mfc="white" if st["open_marker"] else COL[key], ls="",
                          transform=fig.transFigure))
    texts.append(fig.text(0.076, yk, text, fontsize=FS_KEY, color=F.INK, ha="left", va="center"))

# ------------------------------------------------------------------ layout checks
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_bbs = [(t.get_text(), t.get_window_extent(_r)) for t in texts]
for i in range(len(_bbs)):
    for j in range(i + 1, len(_bbs)):
        assert not _bbs[i][1].overlaps(_bbs[j][1]), ("labels overlap", _bbs[i][0], _bbs[j][0])
_x_panel = ax.get_window_extent(_r).x0
for name, bb in _bbs[1:]:
    assert bb.x1 < _x_panel, ("label reaches into the panel", name)
F.check_fits(fig)
if PNG:
    os.makedirs(os.path.dirname(os.path.abspath(PNG)), exist_ok=True)
    fig.savefig(PNG, dpi=PNG_WIDTH_PX / W, facecolor="white",
                metadata={"Software": None}, pil_kwargs={"dpi": (PNG_DPI_TAG, PNG_DPI_TAG)})
    plt.close(fig)
    print(f"written {PNG} ({PNG_WIDTH_PX} px wide)")
else:
    os.makedirs(OUTDIR, exist_ok=True)
    F.save(fig, os.path.join(OUTDIR, "fig_paired_refit"))
    print(f"written {os.path.join(OUTDIR, 'fig_paired_refit.pdf')} ({W * 2.54:.1f} x {H_CM:.1f} cm)")
for rid, lab in zip(ids, labels):
    kind, (inh, ref) = DATA[rid]
    print(f"{rid:12s} {kind:4s} n={inh[3]:2d}  inherited {inh[0]:+.4f} [{inh[1]:+.4f}, {inh[2]:+.4f}]"
          f"  refitted {ref[0]:+.4f} [{ref[1]:+.4f}, {ref[2]:+.4f}]")
