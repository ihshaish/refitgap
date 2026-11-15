"""analysis.analyse() result -> one self-contained HTML page (inline SVG figures, tables,
one sentence per comparison; no network, no script)."""
import html
from pathlib import Path

from .analysis import CONTRAST_COLUMNS, DECOMP_COLUMNS, LEVEL_COLUMNS, VOCABULARY_NOTE

CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#222;background:#fff;max-width:72rem;margin:0 auto;
padding:1rem 16px 3rem;line-height:1.45}
h1{font-size:1.4rem;margin-bottom:.2rem}h2{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ccc}
table{border-collapse:collapse;font-family:Menlo,Consolas,monospace;font-size:.72rem;margin:.5rem 0}
th,td{border-bottom:1px solid #e3e3e3;padding:2px 6px;text-align:right;white-space:nowrap}
th{background:#f4f4f4;position:sticky;top:0}td.t,th.t{text-align:left}
.scroll{overflow-x:auto;max-height:28rem;overflow-y:auto}
.note{color:#555;font-size:.9rem}figure{margin:1rem 0}figure svg{max-width:100%;height:auto}
figcaption{color:#555;font-size:.85rem}
"""

TEXT_COLS = {"method", "model", "period", "rule", "direction", "sample_unit", "interval_type", "family", "adjustment",
             "status", "U_convention", "comparison", "term", "policy", "periods", "seeds", "values", "class"}


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4f}"
    return str(v)


def _table(rows, cols):
    if not rows:
        return "<p class='note'>No rows.</p>"
    head = "".join(f"<th class='{'t' if c in TEXT_COLS else ''}'>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td class='{'t' if c in TEXT_COLS else ''}'>{html.escape(_fmt(r.get(c)))}</td>"
                                     for c in cols) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _kv(d):
    rows = "".join(f"<tr><td class='t'>{html.escape(str(k))}</td><td class='t'>{html.escape(str(v))}</td></tr>"
                   for k, v in d.items())
    return f"<div class='scroll'><table><tbody>{rows}</tbody></table></div>"


CAPTIONS = {
    "decomposition": "Deficit decomposition against the per-period reference: A (orange squares, solid line) and "
                     "dR, the change in mismatch (blue circles, dashed line), per period and their "
                     "mean, macro-F1 on the 0 to 1 scale, positive where the final model is below the reference. "
                     "B = A + dR holds in every row.",
    "profile": "Period profile of each final model: V_sep under own-period rules (black circles, solid), V_shared "
               "under one shared rule (grey squares, dashed) and U_full at the fixed rule (blue triangles, dotted), "
               "macro-F1 on the 0 to 1 scale; the axis spans the data and its note gives the range.",
    "comparison": "Paired leads at the fixed rule (black diamonds), after own-period refit (orange squares) and after "
                  "one shared refit (grey circles), with marginal 95% t intervals over seeds where there are two or "
                  "more. The dashed lines mark the margin, the dotted line zero.",
}


def render(result, svgs=None, title="refitgap report"):
    """The HTML page as a string. svgs: {name: svg markup} (plots.svg_strings); None tries
    to draw them and falls back to a note when matplotlib is unavailable."""
    if svgs is None:
        try:
            from .plots import svg_strings
            svgs = svg_strings(result)
        except ImportError:
            svgs = {}
    st = result["settings"]
    settings = {
        "refitgap version": st.get("refitgap_version"),
        "methods and inputs": "; ".join(f"{k}: {', '.join(v)}" for k, v in st["methods"].items()),
        "seeds": " ".join(map(str, st["seeds"])),
        "partitions (splits)": st["splits"], "partition rule": st["split_rule"] + f" (split_base {st['split_base']})",
        "fixed rule of the decomposition": st["rule"],
        "rule vectors": "; ".join(f"{src}: " + ", ".join(f"{k} = {[round(x, 6) for x in v]}" for k, v in r.items())
                                  for src, r in st["rules"].items()),
        "evaluated period ids": st["period_ids"], "periods averaged": st["mean_over"],
        "shared periods": st["shared_periods"], "shared weights": st["shared_weights"] or "uniform",
        "weights of the average": st["mean_weights"] or "uniform",
        "U convention": ("U_full: the fixed-rule level on the whole period (decomposition B = A + dR as in Algorithm 1 "
                         "of the paper)" if st["U_convention"] == "full" else
                         "U_half: the fixed-rule level on the held-out halves (R = C + M exact per partition)"),
        "margin": st["margin"], "family": st["family"] or "none (every row descriptive)",
    }
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             "<p class='note'>Evaluation of thresholded macro-F1 from stored per-period scores and labels. "
             "Every quantity is a function of the scores, the labels, the rules, the seed and the period ids. The "
             "terms describe what thresholded predictions built from these scores reach; they do not identify what "
             "the model's internal features retain, and the report proposes no action.</p>",
             "<h2>Inputs and validation</h2>",
             _table(result["validation"], ["check", "result", "detail"]),
             "<h2>Settings</h2>", _kv(settings),
             "<h2>Interpretation</h2>"]
    if result["interpretation"]:
        parts.append("<ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in result["interpretation"]) + "</ul>")
    else:
        parts.append("<p class='note'>No comparison was computed.</p>")
    parts.append(f"<p class='note'>Rows are labelled descriptive unless a planned family was named. "
                 f"{html.escape(VOCABULARY_NOTE)}</p>")
    parts.append("<h2>Figures</h2>")
    if svgs:
        for name in ("decomposition", "profile", "comparison"):
            if name in svgs:
                parts.append(f"<figure>{svgs[name]}<figcaption>{html.escape(CAPTIONS[name])}</figcaption></figure>")
    else:
        parts.append("<p class='note'>Figures were not drawn: matplotlib is not installed "
                     "(pip install 'refitgap[plots]'). The tables below hold every number.</p>")
    parts += [
        "<h2>Decomposition B = A + dR (reference minus final)</h2>",
        _table(result["decomposition"], DECOMP_COLUMNS),
        "<h2>Paired contrasts</h2>",
        _table(result["contrasts"], CONTRAST_COLUMNS),
        "<h2>Per-period levels</h2>",
        _table(result["levels"], LEVEL_COLUMNS),
        "<p class='note'>V_sep: own-period rules; V_shared: one shared rule; U_half, U_full: the fixed "
        "rule on the held-out halves and on the whole period; C = V_sep - V_shared; M = V_shared - U_half; "
        "R_half = V_sep - U_half = C + M; R_full = V_sep - U_full. In the decomposition, B = U_ref - U_final, "
        "A = V_ref - V_final and dR = R_final - R_ref at the chosen U convention. Full-precision values are "
        "in the CSV files.</p>",
        "</body></html>",
    ]
    return "\n".join(parts)


def write_report(result, path, svgs=None, title="refitgap report"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result, svgs, title), encoding="utf-8")
    return path
