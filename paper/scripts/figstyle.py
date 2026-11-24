"""Shared figure style for the paper. Every figure script imports its colours,
reference lines, interval marks, sizes and widths from here and sets no
rcParams of its own.

Conventions:
- colour names a quantity, not a series: orange for the attainable term A,
  blue for the operating-point family (R, Delta R, C, M), grey for
  threshold-free ranking; everything else is ink and greys;
- methods, conditions and streams are told apart by line style, marker and
  a direct label, so every figure reads in greyscale and under colour-vision
  deficiency (worst adjacent deltaE 9.6 for deuteranopia, 18.4 for normal
  vision);
- sequential magnitudes use single-hue Blues; diverging quantities use
  blue-white-dark grey with a neutral midpoint;
- thin marks (0.8 to 1.2 pt), dark-grey axes, outward ticks; data panels
  take frame(): a full thin frame and faint gridlines on the value axis;
  direct labels rather than legend boxes; text in ink, never in a series
  colour;
- every figure is saved as a vector PDF.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#222222"       # text, mean marks of series that are not components
MUTED = "#555555"     # axes, ticks, in-axes notes (dark grey)
GRID = "#dddddd"      # hairline separators and faint gridlines
PALE = "#cbcbcb"      # neutral fill for areas that carry no quantity

# Ink, one blue and greys for distinctions that carry no quantity. Every pair
# separates in greyscale print and under deuteranopia, and no figure relies on
# hue alone: line style and marker carry the same distinction.
DARK = "#3f3f3f"        # second series: dark ink
MID = "#808080"         # the final model's levels in the schematic (distinct from ink in print)
RED  = DARK             # older name for DARK, kept for scripts that import it
GREY = "#7a7a7a"
LGREY = "#adadad"

# Quantity colours, fixed across all figures: the attainable term A is
# orange; the operating-point family (R, Delta R and its shared-rule parts C
# and M) is blue; threshold-free ranking is grey. Methods (sequential
# training, replay, extra current-data updates, LwF, EWC, the repair arms) are
# told apart by line style, marker and a direct label (METHOD_STYLE below),
# never by a hue that also names a component.
A_COL = "#D95319"       # attainable term A (orange)
R_COL = "#0072BD"       # operating-point family R, Delta R, C, M (blue)
RANK_COL = "#8c8c8c"    # threshold-free ranking (grey: neither A nor R)
BLUE = R_COL            # older name for the same role

# One size for every in-axes note and value label; titles stay 8, ticks 7 to
# 7.5, direct series labels 7.5.
ANN = 6.5

# Further conventions:
# - raw data: small open markers with alpha and thin edges (ms 2 to 2.6,
#   mew 0.6);
# - mean and 95% interval: the whisker is drawn through the point estimate
#   (point_whisker), with raw units as small faint open markers behind; a
#   half-violin where the distribution itself is the point;
# - reference lines (zero, no change, bounds): dotted hairline in GRID or
#   MUTED;
# - line weights at most 1.2; a legend never overlaps data;
# - where identity is the point of a panel, a few series may use R_COL, ink
#   and greys with distinct small markers, and streams use the CORPUS ramp;
# - in-axes labels are INK or MUTED; no text touches a marker or data point.

# Methods: one line style and marker per arm, the same in every figure, and a
# direct label on the artwork. Colour is never part of a method's identity.
METHOD_STYLE = {
    "naive":     dict(ls=(0, (1, 1.5)), marker="o", name="sequential training"),
    "replay":    dict(ls="-",           marker="D", name="replay"),
    "replaycur": dict(ls=(0, (4, 2)),   marker="^", name="extra current-data updates"),
    "lwf":       dict(ls=(0, (4, 1.5, 1, 1.5)), marker="v", name="LwF"),
    "ewc":       dict(ls=(0, (2, 1.5)), marker="p", name="EWC"),
    "r_full":    dict(ls="-",           marker="s", name="full-model repair"),
}

# Reference lines: zero is a dotted hairline; the benchmarking margin a
# wide-dashed grey line, labelled where it is drawn.
ZERO_LS = (0, (1, 2))
MARGIN_LS = (0, (5, 2))

def refline(ax, value, kind, orient="v"):
    """Zero or margin reference line, the same in every figure."""
    kw = (dict(color=MUTED, lw=0.6, ls=ZERO_LS) if kind == "zero"
          else dict(color=GREY, lw=0.9, ls=MARGIN_LS))
    (ax.axvline if orient == "v" else ax.axhline)(value, zorder=1, **kw)

def frame(ax, grid="x", lw=0.6, grid_lw=0.4, edge=None):
    """A complete thin frame on all four sides of a data panel and faint
    gridlines at the major ticks of the value axis (grid="x" for horizontal
    layouts, "y" for vertical ones), behind the data. Call after any spine
    has been hidden; ticks stay where the script put them."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_edgecolor(edge or MUTED)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid, which="major", color=GRID, linewidth=grid_lw, zorder=0.5)


def point_whisker(ax, pos, mean, lo, hi, color, orient="v", ms=3.0,
                  lw=1.0, marker="D", open_marker=False, mew=0.9):
    """Point estimate with its interval drawn through the marker."""
    kw = dict(color=color, lw=lw, solid_capstyle="butt", zorder=5)
    if orient == "v":
        ax.plot([pos, pos], [lo, hi], **kw)
        x, y = pos, mean
    else:
        ax.plot([lo, hi], [pos, pos], **kw)
        x, y = mean, pos
    mkw = dict(marker=marker, ms=ms, mec=color, mew=mew, zorder=6, ls="")
    ax.plot([x], [y], mfc="white" if open_marker else color, **mkw)

def mean_interval(ax, pos, mean, lo, hi, color, orient="v", tick=None,
                  lw=1.0):
    """Slim 95% interval with a short perpendicular tick at the mean, drawn
    at `pos` beside the raw points. `tick` is the tick half-width in data
    units (default 1.5% of the axis span on the pos axis)."""
    if orient == "v":
        if tick is None:
            x0, x1 = ax.get_xlim(); tick = 0.015 * (x1 - x0)
        ax.plot([pos, pos], [lo, hi], color=color, lw=lw,
                solid_capstyle="butt", zorder=5)
        ax.plot([pos - tick, pos + tick], [mean, mean], color=color,
                lw=lw * 1.7, solid_capstyle="butt", zorder=6)
    else:
        if tick is None:
            y0, y1 = ax.get_ylim(); tick = 0.015 * (y1 - y0)
        ax.plot([lo, hi], [pos, pos], color=color, lw=lw,
                solid_capstyle="butt", zorder=5)
        ax.plot([mean, mean], [pos - tick, pos + tick], color=color,
                lw=lw * 1.7, solid_capstyle="butt", zorder=6)

# fixed identity of the embedding conditions
COND_COLORS = {
    "C1": BLUE,        # frozen-domain
    "C2": INK,         # frozen-GloVe
    "C3": GREY,        # tuned-domain
    "C4": RED,         # tuned-GloVe
    "C5": LGREY,       # random-trainable
    "C6": "#5a5a5a",   # frozen-random
    "scratch": "#c8c8c8",
}
COND_LS = {"C1": "-", "C2": "--", "C3": "-", "C4": "--", "C5": ":", "C6": "-.",
           "scratch": ":"}
COND_MARK = {"C1": "d", "C2": "x", "C3": ">", "C4": "<", "C5": "p", "C6": "h",
             "scratch": "1"}
COND_NAMES = {
    "C1": "frozen domain", "C2": "frozen GloVe", "C3": "fine-tuned domain",
    "C4": "fine-tuned GloVe", "C5": "random trainable", "C6": "frozen random",
    "scratch": "per-period scratch",
}

# the two metrics, used in every paired plot: blue solid, ink dashed
M_THR, M_FREE = BLUE, INK
LS_THR, LS_FREE = "-", (0, (4, 2))

# five corpora: blue and dark ink for the aviation and news poles, the rest a
# grey ramp, so the figure survives greyscale printing (the fourth step is
# #9a9a9a, not RANK_COL, so the ranking grey never names a corpus)
CORPUS = [BLUE, DARK, "#6e6e6e", "#9a9a9a", "#c0c0c0"]

SEQ_CMAP = "Blues"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "bwv", ["#0072B2", "#f7f7f7", "#3f3f3f"])  # neutral midpoint

def apply(face="stix"):
    """Set the shared rcParams. face="cm" sets Computer Modern (matplotlib's
    bundled cmr10 with mathtext "cm"), the face of the manuscript's Latin
    Modern body, so a figure's text matches the body type. cmr10 has no
    Unicode minus, en dash or Greek capitals: write those in mathtext ($-$,
    $\\Delta$) or through mt() below; tick labels use mathtext automatically.
    The default "stix" sets STIX."""
    _apply_base()
    if face == "cm":
        plt.rcParams.update({
            "font.serif": ["cmr10"], "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        })


def mt(s):
    """Text for the cm face: Unicode minus and en dash become mathtext."""
    return s.replace("−", "$-$").replace("–", "$-$")


def _apply_base():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # sizes are points at final size: figures are drawn at the width they
        # are printed (W1 single column, W2 full width), labels 9 pt and tick
        # labels 8.5 pt (8 to 10 pt at final size)
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.35,
        "axes.axisbelow": True,
        "lines.linewidth": 1.0, "lines.markersize": 3.0,
        "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })

def _cm_slot_to_unicode():
    """(font, slot) -> Unicode for matplotlib's bundled Computer Modern fonts,
    from matplotlib's own mathtext tables. These fonts keep TeX's slot layout
    (the minus sign of cmsy10 sits at 0xA1, the math full stop of cmmi10 at
    0x3A), and the PDF backend writes the slot number as the character's
    Unicode value, so text extracted from the PDF would read a full stop as a
    colon, a minus sign as an inverted exclamation mark, and so on."""
    from matplotlib import _mathtext_data as md
    table = {}
    for name, (font, slot) in md.latex_to_bakoma.items():
        uni = ord(name) if len(name) == 1 else md.tex2uni.get(name[1:])
        if uni is None or uni > 0xFFFF:
            continue
        # where two names share a glyph, the single-character name decides
        if (font, slot) not in table or len(name) == 1:
            table[font, slot] = uni
    return table


def _pdf_text_layer():
    """Context manager: while it is open, PDFs saved by matplotlib carry a
    ToUnicode map for the Computer Modern fonts that names the glyph drawn
    (_cm_slot_to_unicode), so text copied or extracted from the figure reads
    as it is printed. Nothing that is drawn changes."""
    import contextlib
    import re as _re
    from matplotlib.backends.backend_pdf import PdfFile

    table = _cm_slot_to_unicode()
    embed, output = PdfFile.embedTTF, PdfFile.outputStream
    state = {"font": None}

    def embedTTF(self, filename, characters):
        state["font"] = _os.path.splitext(_os.path.basename(filename))[0].lower()
        try:
            return embed(self, filename, characters)
        finally:
            state["font"] = None

    def _range(m):
        start = int(m.group(1), 16)
        dest = [int(h, 16) for h in _re.findall(rb"<([0-9a-fA-F]{4})>", m.group(3))]
        dest = [table.get((state["font"], start + i), u) for i, u in enumerate(dest)]
        return b"<%s> <%s> [%s]" % (m.group(1), m.group(2), b" ".join(b"<%04x>" % u for u in dest))

    def outputStream(self, ref, data, extra=None):
        if state["font"] is not None and isinstance(data, bytes) and b"beginbfrange" in data:
            data = _re.sub(rb"<([0-9a-fA-F]{4})> <([0-9a-fA-F]{4})> \[([^\]]*)\]", _range, data)
        return output(self, ref, data, extra=extra)

    @contextlib.contextmanager
    def patched():
        PdfFile.embedTTF, PdfFile.outputStream = embedTTF, outputStream
        try:
            yield
        finally:
            PdfFile.embedTTF, PdfFile.outputStream = embed, output
    return patched()


def save(fig, path_stem, png=False):
    """Save the figure as a vector PDF (and, if asked, a 400 dpi PNG) and
    close it."""
    with _pdf_text_layer():
        fig.savefig(f"{path_stem}.pdf")
    if png:
        fig.savefig(f"{path_stem}.png", dpi=400)
    plt.close(fig)

# Final widths. The manuscript is set with elsarticle [preprint,12pt], whose
# text block is 390 pt = 13.7 cm; a single column is taken as 6.8 cm. Article
# widths (17.5 / 8.5 cm) are available with
#   FIGSTYLE_LAYOUT=article python scripts/<figure script>.py
# Every script draws at W1 or W2, so labels keep their point size on the page.
import os as _os
CM = 1 / 2.54
LAYOUT = _os.environ.get("FIGSTYLE_LAYOUT", "preprint")
WIDTHS_CM = {"preprint": (6.8, 13.7), "article": (8.5, 17.5)}
W1, W2 = (w * CM for w in WIDTHS_CM[LAYOUT])
# The supplement is article 11pt on A4 with 2.4 cm margins: its text block is
# 21.0 - 4.8 = 16.2 cm. Supplement figures included at \textwidth are drawn
# at this width so they print at scale 1.
WS = 16.2 * CM


def check_fits(fig, tol_cm=0.05):
    """The saved PDF uses bbox="tight", so any artist outside the figure box
    widens the page, and LaTeX then shrinks every font when the figure is
    included at its drawn width. Assert that nothing lies outside the drawn
    W x H (within tol_cm), so the page size equals the drawn size. With
    FIGSTYLE_DRAFT set, an overrun is printed instead."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
    w, h = fig.get_size_inches()
    tol = tol_cm / 2.54
    ok = bb.x0 >= -tol and bb.y0 >= -tol and bb.x1 <= w + tol and bb.y1 <= h + tol
    if not ok and _os.environ.get("FIGSTYLE_DRAFT"):     # draft renders only
        print("DRAFT: artwork outside the figure box (cm)",
              [round(v * 2.54, 2) for v in (bb.x0, bb.y0, bb.x1, bb.y1)])
        return
    assert bb.x0 >= -tol and bb.y0 >= -tol and bb.x1 <= w + tol and bb.y1 <= h + tol, (
        "artwork outside the figure box (cm)",
        [round(v * 2.54, 2) for v in (bb.x0, bb.y0, bb.x1, bb.y1)],
        (round(w * 2.54, 2), round(h * 2.54, 2)))


def trim_ticks(axis):
    """Tick labels without trailing zeros ("0", "0.01", "$-0.02$")."""
    from matplotlib.ticker import FuncFormatter
    def _f(v, _pos):
        txt = f"{v:.4f}".rstrip("0").rstrip(".")
        txt = "0" if txt in ("-0", "") else txt
        return f"${txt}$"
    axis.set_major_formatter(FuncFormatter(_f))
