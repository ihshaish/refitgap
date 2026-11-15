#!/usr/bin/env python3
"""Profile the ASRS exports year by year: counts, lengths, ALL-CAPS share, label priors, adjacent-year JSD, term trends.

Reads the CSV exports in --data-dir; writes per_year_stats.csv, anomaly_values_by_year.csv, label_priors_top14.csv,
jsd_matrix.csv, decaying/emerging_terms.csv, fig_profile_*.pdf/.png and profile_report.md to --out-dir.
Backs the ASRS profile (Appendix B, "ASRS"); build_streams.py imports its label mapping.
Needs the ASRS export (e.g. ../data_asrs), which the repository does not ship."""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Expected column indices in the DBOnline export (126-column layout, two header
# rows: section / field). Located by name first; these are the fallbacks.
FALLBACK_COLS = {"anomaly": 112, "primary_problem": 119, "narrative": 120}
HEADER_NAMES = {
    "anomaly": ("Events", "Anomaly"),
    "primary_problem": ("Assessments", "Primary Problem"),
    "narrative": ("Report 1", "Narrative"),
}

# Prefix mapping of exported Anomaly values to the 14 top-level categories of the
# post-June-2009 taxonomy; unmatched values go to 'UNMAPPED' and are reported per year.
TOP14_RULES = [
    ("Aircraft Equipment", "Aircraft Equipment Problem"),
    ("ATC Issue", "ATC Issue"),
    ("Conflict", "Conflict"),
    ("Deviation - Altitude", "Deviation - Altitude"),
    ("Deviation - Speed", "Deviation - Speed"),
    ("Deviation - Track", "Deviation - Track / Heading"),
    ("Deviation / Discrepancy - Procedural", "Deviation / Discrepancy - Procedural"),
    ("Flight Deck / Cabin", "Flight Deck / Cabin / Aircraft Event"),
    ("Ground Event", "Ground Event / Encounter"),
    ("Ground Excursion", "Ground Excursion"),
    ("Ground Incursion", "Ground Incursion"),
    ("Inflight Event", "Inflight Event / Encounter"),
    ("No Specific Anomaly", "No Specific Anomaly Occurred"),
    ("Airspace Violation", "Airspace Violation"),          # exported as e.g. "Airspace Violation All Types"
    ("Other Airspace Violation", "Airspace Violation"),    # legacy-mapped variant seen in 1988
]
OTHER_UNSPECIFIED = "Other Unspecified"  # mapping residue; tracked separately

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
YEAR_RE = re.compile(r"(19[89]\d|20[0-2]\d)")


def map_top14(value):
    v = value.strip()
    if not v:
        return None
    if v.startswith(OTHER_UNSPECIFIED):
        return OTHER_UNSPECIFIED
    for prefix, cat in TOP14_RULES:
        if v.startswith(prefix):
            return cat
    return "UNMAPPED"


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_year_csv(path):
    """Parse one DBOnline export; returns (columns dict, list of row dicts)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            sections = next(reader)
            fields = next(reader)
        except StopIteration:
            return None
        header = [(s.strip(), t.strip()) for s, t in zip(sections, fields)]
        cols = {}
        for key, (sec, fld) in HEADER_NAMES.items():
            hits = [i for i, (s, t) in enumerate(header) if s == sec and t == fld]
            cols[key] = hits[0] if hits else FALLBACK_COLS[key]
        need = max(cols.values())
        records = []
        for row in reader:
            if len(row) <= need or not row[0].strip().isdigit():
                continue  # blank separator rows / stray lines
            records.append({
                "acn": row[0].strip(),
                "anomaly": row[cols["anomaly"]],
                "primary_problem": row[cols["primary_problem"]].strip(),
                "narrative": row[cols["narrative"]],
            })
    return cols, records


def js_divergence(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def theil_sen_slope(ys):
    """Median pairwise slope — robust trend for token log-frequency series."""
    slopes = [(ys[j] - ys[i]) / (j - i) for i in range(len(ys)) for j in range(i + 1, len(ys))]
    return float(np.median(slopes)) if slopes else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="profile_out")
    ap.add_argument("--min-year", type=int, default=1988)
    ap.add_argument("--max-year", type=int, default=2021)
    ap.add_argument("--vocab-min-count", type=int, default=50,
                    help="min total count for a token to enter the JSD/trend vocabulary")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exports are partial-year files that never span two calendar years:
    # pool every file into the year in its name.
    files = defaultdict(list)
    for p in sorted(data_dir.glob("*.csv")):
        m = YEAR_RE.search(p.name)
        if m and args.min_year <= int(m.group(1)) <= args.max_year:
            files[int(m.group(1))].append(p)
    if not files:
        sys.exit(f"no yearly CSVs found in {data_dir}")
    n_files = sum(len(v) for v in files.values())
    print(f"found {n_files} files covering {len(files)} years: {min(files)}–{max(files)}")
    known_gaps = {2012: "H2 missing", 2020: "entire year missing", 2021: "Oct-Dec missing"}
    for y in range(min(files), max(files) + 1):
        if y not in files:
            print(f"  WARNING: no files for {y}" + (f" ({known_gaps[y]})" if y in known_gaps else ""))

    per_year, token_dists, anomaly_rows, prior_rows = [], {}, [], []
    seen_acn, dup_acn = set(), 0
    for year, paths in sorted(files.items()):
        recs, cols = [], None
        for path in paths:
            parsed = read_year_csv(path)
            if not parsed:
                print(f"  {year}: {path.name} unreadable, skipped"); continue
            cols, part = parsed
            for rec in part:  # exact-ACN dedup across all files (overlapping exports)
                if rec["acn"] in seen_acn:
                    dup_acn += 1
                    continue
                seen_acn.add(rec["acn"])
                recs.append(rec)
        if not recs:
            continue
        n = len(recs)
        lengths, caps, tokens = [], 0, Counter()
        label_counts, top14_counts, pp_counts, nlabels = Counter(), Counter(), Counter(), []
        empty_narr = 0
        for rec in recs:
            narr = rec["narrative"].strip()
            if not narr:
                empty_narr += 1
            else:
                toks = tokenize(narr)
                lengths.append(len(toks))
                tokens.update(toks)
                letters = [c for c in narr if c.isalpha()]
                if letters and sum(c.isupper() for c in letters) / len(letters) > 0.9:
                    caps += 1
            labs = [v.strip() for v in rec["anomaly"].split(";") if v.strip()]
            nlabels.append(len(labs))
            label_counts.update(labs)
            top14_counts.update({t for t in (map_top14(v) for v in labs) if t})
            if rec["primary_problem"]:
                pp_counts[rec["primary_problem"]] += 1
        per_year.append({
            "year": year, "n_records": n, "n_empty_narrative": empty_narr,
            "narrative_tokens_median": float(np.median(lengths)) if lengths else 0,
            "narrative_tokens_mean": float(np.mean(lengths)) if lengths else 0,
            "allcaps_fraction": caps / max(1, n - empty_narr),
            "labels_per_report_mean": float(np.mean(nlabels)) if nlabels else 0,
            "other_unspecified_share": top14_counts[OTHER_UNSPECIFIED] / max(1, n),
            "unmapped_share": top14_counts["UNMAPPED"] / max(1, n),
            "anomaly_col": cols["anomaly"], "narrative_col": cols["narrative"],
        })
        token_dists[year] = tokens
        for v, c in label_counts.most_common():
            anomaly_rows.append({"year": year, "anomaly_value": v, "count": c,
                                 "top14": map_top14(v)})
        for cat, c in sorted(top14_counts.items()):
            prior_rows.append({"year": year, "top14": cat, "count": c,
                               "prior": c / max(1, n)})
        print(f"  {year}: {n} records, caps={caps / max(1, n - empty_narr):.2f}, "
              f"other-unspec={top14_counts[OTHER_UNSPECIFIED] / max(1, n):.2f}")

    if dup_acn:
        print(f"  note: {dup_acn} duplicate-ACN records dropped across files")
    stats = pd.DataFrame(per_year).set_index("year")
    stats.to_csv(out_dir / "per_year_stats.csv")
    pd.DataFrame(anomaly_rows).to_csv(out_dir / "anomaly_values_by_year.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / "label_priors_top14.csv", index=False)

    # ---- shared vocabulary, JSD matrix, birth/death, trends
    years = sorted(token_dists)
    total = Counter()
    for c in token_dists.values():
        total.update(c)
    vocab = sorted(w for w, c in total.items() if c >= args.vocab_min_count)
    widx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(years), len(vocab)))
    for yi, y in enumerate(years):
        for w, c in token_dists[y].items():
            if w in widx:
                mat[yi, widx[w]] = c
    row_tot = mat.sum(axis=1, keepdims=True)
    dist = mat / np.maximum(row_tot, 1)

    jsd = np.zeros((len(years), len(years)))
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            jsd[i, j] = jsd[j, i] = js_divergence(dist[i], dist[j])
    pd.DataFrame(jsd, index=years, columns=years).to_csv(out_dir / "jsd_matrix.csv")

    if len(years) >= 4:
        # token trends on log relative frequency (Laplace-smoothed), Theil-Sen
        logf = np.log10(dist + 1e-8)
        trends = np.apply_along_axis(theil_sen_slope, 0, logf)
        present_first = dist[: max(2, len(years) // 4)].mean(axis=0)
        present_last = dist[-max(2, len(years) // 4):].mean(axis=0)
        rows = [{"token": vocab[k], "slope": trends[k],
                 "freq_early": present_first[k], "freq_late": present_last[k],
                 "total_count": int(total[vocab[k]])} for k in range(len(vocab))]
        tdf = pd.DataFrame(rows)
        tdf.nsmallest(60, "slope").to_csv(out_dir / "decaying_terms.csv", index=False)
        tdf.nlargest(60, "slope").to_csv(out_dir / "emerging_terms.csv", index=False)

    # ---- figures (paper style: serif, small, PDF+PNG)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if len(years) >= 2:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(jsd, cmap="viridis", origin="lower")
        step = max(1, len(years) // 8)
        ax.set_xticks(range(0, len(years), step), [str(y) for y in years[::step]], rotation=90)
        ax.set_yticks(range(0, len(years), step), [str(y) for y in years[::step]])
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jensen–Shannon divergence (bits)")
        save(fig, "fig_profile_jsd")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    axes[0].plot(stats.index, stats["n_records"], color="k")
    axes[0].set_title("records / year")
    axes[1].plot(stats.index, stats["allcaps_fraction"], color="k")
    axes[1].axvline(2009, ls=":", lw=0.8, color="0.4")
    axes[1].set_title("ALL-CAPS narrative fraction")
    axes[2].plot(stats.index, stats["other_unspecified_share"], color="k")
    axes[2].axvline(2009, ls=":", lw=0.8, color="0.4")
    axes[2].set_title("'Other Unspecified' share")
    save(fig, "fig_profile_breaks")

    pri = pd.DataFrame(prior_rows)
    if not pri.empty:
        piv = pri.pivot_table(index="year", columns="top14", values="prior").fillna(0)
        keep = [c for c in piv.columns if c not in (OTHER_UNSPECIFIED, "UNMAPPED")]
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        for c in keep:
            ax.plot(piv.index, piv[c], lw=0.9, label=c[:28])
        ax.axvline(2009, ls=":", lw=0.8, color="0.4")
        ax.set_ylabel("label prior")
        ax.legend(fontsize=5.2, ncol=2, frameon=False)
        save(fig, "fig_profile_priors")

    # ---- report
    lines = ["# ASRS export profile\n",
             f"Years covered: {min(years)}–{max(years)} ({len(years)} files). "
             f"Vocabulary for JSD/trends: {len(vocab)} tokens (min count {args.vocab_min_count}).\n"]
    if 2008 in years and 2010 in years:
        i08, i10 = years.index(2008), years.index(2010)
        adj = [jsd[k, k + 1] for k in range(len(years) - 1)
               if not (years[k] <= 2009 <= years[k + 1])]
        lines.append("## 2009-break positive control\n")
        lines.append(f"- JSD(2008, 2010) across the break: **{jsd[i08, i10]:.4f}** vs "
                     f"median adjacent-year JSD elsewhere {np.median(adj):.4f}\n")
        lines.append(f"- ALL-CAPS fraction: 2008 = {stats.loc[2008, 'allcaps_fraction']:.2f}, "
                     f"2010 = {stats.loc[2010, 'allcaps_fraction']:.2f}\n")
        lines.append(f"- Other-Unspecified share: 2008 = {stats.loc[2008, 'other_unspecified_share']:.2f}, "
                     f"2010 = {stats.loc[2010, 'other_unspecified_share']:.2f}\n")
    lines.append("\n## Per-year table\n\n" + stats.to_markdown() + "\n")
    (out_dir / "profile_report.md").write_text("\n".join(lines))
    print(f"\nwrote outputs to {out_dir}/ (see profile_report.md)")


if __name__ == "__main__":
    main()
