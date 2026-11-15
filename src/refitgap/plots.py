"""The three figures of a report, from analysis.analyse() tables: deficit decomposition
(A and R per period), period profile (U, V_sep, V_shared) and method comparison (leads with
intervals and the margin). PDF and SVG; needs matplotlib (extra "plots")."""
import io
from pathlib import Path

import numpy as np

A_COL = "#D95319"        # attainable term A (as in the paper)
R_COL = "#0072BD"        # mismatch terms R, dR, C, M (as in the paper)
INK = "#222222"
MUTED = "#555555"
GREY = "#7a7a7a"
FIGURES = ("decomposition", "profile", "comparison")


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:                       # pragma: no cover
        raise ImportError("figures need matplotlib: pip install 'refitgap[plots]'") from e
    # fixed salt for the ids matplotlib writes into SVG, so a figure is the same byte for byte
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
                         "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "refitgap", "pdf.fonttype": 42,
                         "lines.linewidth": 1.2})
    return plt


def _is_mean(p):
    return str(p).startswith("mean")


def _note(fig, text):
    import textwrap
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(text, 95)), ha="left", va="top", fontsize=7, color=MUTED)


def _jitter(n, width=0.12):
    return np.zeros(1) if n == 1 else np.linspace(-width, width, n)


def figure_decomposition(result):
    """Signed A and dR per period and their mean over the selected periods, against the
    per-period reference. One panel per method that has a reference."""
    plt = _plt()
    rows = result["decomposition"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    if not methods:
        fig, ax = plt.subplots(figsize=(4.2, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No per-period reference scores were given,\nso B = A + dR is not computed.",
                ha="center", va="center", color=MUTED)
        return fig
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods) + 0.6, 2.9), squeeze=False)
    conv = rows[0]["U_convention"]
    for ax, m in zip(axes[0], methods):
        mr = [r for r in rows if r["method"] == m]
        cats = list(dict.fromkeys(r["period"] for r in mr))
        xs = np.arange(len(cats), dtype=float)
        seeds = sorted({r["seed"] for r in mr})
        for term, col, mk, ls, off in (("A", A_COL, "s", "-", -0.17), ("dR", R_COL, "o", "--", 0.17)):
            means = []
            for i, c in enumerate(cats):
                v = np.array([r[term] for r in mr if r["period"] == c])
                if len(seeds) > 1:
                    ax.plot(xs[i] + off + _jitter(len(v), 0.06), v, mk, ms=2.8, mfc="white", mec=col, mew=0.8,
                            alpha=0.9, zorder=2)
                means.append(v.mean())
            per = [i for i, c in enumerate(cats) if not _is_mean(c)]
            ax.plot(xs[per] + off, np.array(means)[per], ls=ls, color=col, lw=0.9, zorder=3)
            ax.plot(xs + off, means, mk, ms=6, color=col, mec="white", mew=0.6, zorder=4,
                    label="A, attainable gap" if term == "A" else "dR, change in mismatch")
        ax.axhline(0, color=MUTED, lw=0.7, ls=":")
        if any(_is_mean(c) for c in cats):
            ax.axvline(len(cats) - 1.5, color=GREY, lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["mean" if _is_mean(c) else str(c) for c in cats])
        ax.set_xlabel("period id")
        ax.set_ylabel("reference minus final\nmacro-F1 difference (0 to 1 scale)")
        ax.set_title(f"{m}: B = A + dR, rule '{rows[0]['rule']}', {conv}", loc="left")
        n = len(seeds)
    axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout()
    _note(fig, f"{'Small hollow markers: single runs. ' if n > 1 else ''}Filled markers: mean over {n} "
               f"run{'s' if n > 1 else ''}. Positive where the final model is below the reference. Descriptive.")
    return fig


def figure_profile(result):
    """U at the fixed rule, V_sep and V_shared per period for each method's final model."""
    plt = _plt()
    rule = result["settings"]["rule"]
    rows = [r for r in result["levels"] if r["model"] == "final" and r["rule"] == rule and not _is_mean(r["period"])]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods) + 0.6, 2.9), squeeze=False, sharey=True)
    series = (("V_sep", "V_sep, own-period rules", INK, "o", "-"),
              ("V_shared", "V_shared, one shared rule", GREY, "s", "--"),
              ("U_full", f"U_full, fixed rule '{rule}'", R_COL, "^", ":"))
    seen = []
    for ax, m in zip(axes[0], methods):
        mr = [r for r in rows if r["method"] == m]
        per = list(dict.fromkeys(r["period"] for r in mr))
        xs = np.arange(len(per), dtype=float)
        nseed = len({r["seed"] for r in mr})
        for key, lab, col, mk, ls in series:
            seen.extend(r[key] for r in mr if r[key] is not None)
            vals = [[r[key] for r in mr if r["period"] == p] for p in per]
            if any(v is None for vv in vals for v in vv):
                continue
            if nseed > 1:
                for i, vv in enumerate(vals):
                    ax.plot(xs[i] + _jitter(len(vv), 0.08), vv, mk, ms=2.6, mfc="white", mec=col, mew=0.7, zorder=2)
            ax.plot(xs, [np.mean(v) for v in vals], ls=ls, marker=mk, color=col, ms=5, label=lab, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p in per])
        ax.set_xlabel("period id")
        ax.set_title(f"{m}: final model", loc="left")
    # the axis spans the data (at least 0.1 wide, clipped to 0..1) so the gaps between the
    # three levels stay visible; the note states the scale
    lo, hi = float(min(seen)), float(max(seen))
    pad = max(0.02, 0.15 * (hi - lo), (0.1 - (hi - lo)) / 2)
    y0, y1 = max(0.0, lo - pad), min(1.0, hi + pad)
    for ax in axes[0]:
        ax.set_ylim(y0, y1)
    axes[0][0].set_ylabel("macro-F1 (0 to 1 scale)")
    axes[0][0].legend(frameon=False, loc="best", fontsize=7)
    fig.tight_layout()
    _note(fig, f"{'Small hollow markers: single runs. ' if nseed > 1 else ''}Lines: mean over {nseed} "
               f"run{'s' if nseed > 1 else ''}. The axis spans the data, from {y0:.2f} to {y1:.2f} of the "
               f"0 to 1 scale. Descriptive.")
    return fig


def figure_comparison(result):
    """The paired lead at the fixed rule and after refit, one row per term and comparison,
    with its marginal 95% t interval (two or more seeds) and the margin drawn."""
    plt = _plt()
    rows = result["contrasts"]
    margin = result["settings"]["margin"]
    if not rows:
        fig, ax = plt.subplots(figsize=(4.2, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No comparison: give a second method or per-period reference scores.", ha="center",
                va="center", color=MUTED)
        return fig
    style = {"at the rule": (INK, "D"), "after own-period refit": (A_COL, "s"), "after one shared refit": (GREY, "o")}
    fig, ax = plt.subplots(figsize=(5.4, 0.5 * len(rows) + 1.3))
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, r in zip(ys, rows):
        col, mk = style.get(r["term"], (INK, "D"))
        if r["ci95_low"] is not None:
            ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=col, lw=1.3, zorder=2)
        ax.plot(r["mean"], y, mk, color=col, ms=6.5, mec="white", mew=0.6, zorder=4)
        vals = [float(v) for v in str(r.get("values") or "").split()]
        if len(vals) > 1:
            ax.plot(vals, y + 0.22 + _jitter(len(vals), 0.05), "s", ms=2.6, mfc="white", mec=col, mew=0.7, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    for s in (-margin, margin):
        ax.axvline(s, color=MUTED, lw=0.7, ls="--")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, -2 * margin), max(hi, 2 * margin))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['comparison']}\n{r['term']}" for r in rows], fontsize=7.5)
    ax.set_xlabel("lead in macro-F1 (difference on the 0 to 1 scale)")
    n = max(r["n"] for r in rows)
    ax.set_title("paired leads, positive where the first named side is ahead", loc="left", fontsize=8)
    fig.tight_layout()
    note = ("Diamonds, squares and circles: means. Bars: marginal 95% t intervals over seeds. Small hollow squares: "
            "single runs." if n > 1 else "One run: no interval is computed.")
    _note(fig, note + f" Dashed lines: margin {margin:g} either side of zero (dotted). Descriptive unless a "
                      "planned family is named.")
    return fig


def build(result):
    """{name: matplotlib Figure} for the three default figures."""
    return {"decomposition": figure_decomposition(result), "profile": figure_profile(result),
            "comparison": figure_comparison(result)}


def save_figures(result, outdir, formats=("pdf", "svg")):
    """Write fig_<name>.<fmt> for the three figures; returns {name: {fmt: path}}."""
    plt = _plt()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in build(result).items():
        paths[name] = {}
        for fmt in formats:
            p = outdir / f"fig_{name}.{fmt}"
            fig.savefig(p, format=fmt, bbox_inches="tight", metadata={"Date": None} if fmt == "svg" else
                        {"CreationDate": None})
            paths[name][fmt] = p
        plt.close(fig)
    return paths


def svg_strings(result):
    """{name: inline SVG markup} for the HTML report (no XML prolog, no script)."""
    plt = _plt()
    out = {}
    for name, fig in build(result).items():
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        s = buf.getvalue()
        out[name] = s[s.index("<svg"):]
    return out
