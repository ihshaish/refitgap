"""The command line, the CSV and JSON tables, the figures, the HTML report and the tiny example.

All of them read analysis.analyse(). The tests check that the CLI's CSV numbers equal the API's,
that every row carries its direction, sample unit, interval type, family, adjustment, margin and
status, and that no output carries a recommendation or a diagnosis. Seconds to run.
"""
import csv
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import refitgap as rg
from refitgap import analysis, cli, crossfit, inference, tiny

from conftest import REPO

EXACT = 1e-12
BANNED = ("retrain", "recommend", "representation loss", "lost representation", "successful", "success",
          "—", "–", "→")


# ------------------------------------------------------------------ helpers ---
def _read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _synthetic_runs(seeds=(0, 1, 2), N=3, K=3, n=(40, 37, 52)):
    """Two methods on the same records per seed, each with a per-period reference."""
    a, b = [], []
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        Ys = [(rng.random((n[j], K)) < [0.3, 0.2, 0.4]).astype(int) for j in range(N)]
        sc = lambda Y, sep, off: np.round(1 / (1 + np.exp(-(sep * Y + rng.normal(0, 1, Y.shape) + off))), 2)
        ref = [sc(Y, 2.5, -1.0) for Y in Ys]
        fa = [sc(Y, 2.5 - 0.4 * (N - 1 - j), -1.0 + 0.4 * (N - 1 - j)) for j, Y in enumerate(Ys)]
        fb = [sc(Y, 2.5 - 0.1 * (N - 1 - j), -1.0 + 0.1 * (N - 1 - j)) for j, Y in enumerate(Ys)]
        th = np.array([0.5, 0.45, 0.55])
        a.append(analysis.make_run(fa, Ys, s, {"inherited": th}, ref, source=f"a{s}"))
        b.append(analysis.make_run(fb, Ys, s, {"inherited": th}, ref, source=f"b{s}"))
    return a, b


def _write_arrays(tmp, runs, tag):
    paths = []
    for r in runs:
        p = tmp / f"{tag}_s{r['seed']}.npz"
        analysis.save_arrays(p, r)
        paths.append(str(p))
    return paths


def _scan(text, where, strip=()):
    for s in strip:                      # temporary paths carry the test's own name
        text = text.replace(str(s), "").replace(str(s).lower(), "")
    low = text.lower()
    for w in BANNED:
        assert w not in low, f"{w!r} in {where}"


# --------------------------------------------------------------- tiny example ---
def _parse(out):
    """{(section, model, period): {quantity: value}} from the example's printed lines."""
    got = {}
    for line in out.splitlines():
        m = re.match(r"\s+(final|reference) period (\S+): (.*)", line)
        if m:
            got[(m.group(1), m.group(2))] = {k: float(v) for k, v in re.findall(r"(\S+) (-?[\d.e+-]+)", m.group(3))}
        m = re.match(r"\s+period (\S+): (B .*)", line)
        if m:
            got[("decomposition", m.group(1))] = {
                k: float(v) for k, v in re.findall(r"(\S+) (-?[\d.e+-]+)", m.group(2))
            }
    return got


def test_tiny_example_printed_values(capsys):
    # the halves the README works with: seed 0, periods 0 and 1, one partition
    assert list(crossfit.partitions(4, 0, 0, 1)[0]) == [1, 2, 0, 3]
    assert list(crossfit.partitions(4, 0, 1, 1)[0]) == [3, 0, 1, 2]
    assert cli.main(["example"]) == 0
    got = _parse(capsys.readouterr().out)
    res = tiny.tiny_result()
    exact = {}
    for r in res["levels"]:
        exact[(r["model"], "mean" if str(r["period"]).startswith("mean") else str(r["period"]))] = r
    for r in res["decomposition"]:
        exact[("decomposition", "mean" if str(r["period"]).startswith("mean") else str(r["period"]))] = r
    n = 0
    for section, per in tiny.HAND.items():
        for p, vals in per.items():
            row = got[(section, str(p))]
            for k, want in vals.items():
                assert abs(row[k] - want) < 5e-5, (section, p, k, row[k], want)          # printed to 4 decimals
                assert abs(exact[(section, str(p))][k] - want) < EXACT, (section, p, k)
                n += 1
    assert n == 39
    # the fitted own-period thresholds the README lists (final model, period 0, label b)
    assert rg.fit_exact(tiny.F0[[1, 2]][:, [1]], tiny.Y0[[1, 2]][:, [1]])[0] == 0.8
    assert rg.fit_exact(tiny.F0[[0, 3]][:, [1]], tiny.Y0[[0, 3]][:, [1]])[0] == 0.3
    assert rg.fit_shared_exact([tiny.F0[[0, 3]][:, [1]], tiny.F1[[1, 2]][:, [1]]],
                               [tiny.Y0[[0, 3]][:, [1]], tiny.Y1[[1, 2]][:, [1]]])[0] == 0.3


def test_tiny_example_writes_outputs(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert cli.main(["example", "--out", str(tmp_path)]) == 0
    for f in ("levels.csv", "decomposition.csv", "contrasts.csv", "result.json", "report.html"):
        assert (tmp_path / f).stat().st_size > 0
    for name in ("decomposition", "profile", "comparison"):
        for fmt in ("pdf", "svg"):
            assert (tmp_path / f"fig_{name}.{fmt}").stat().st_size > 0
    d = _read_csv(tmp_path / "decomposition.csv")
    assert [r["period"] for r in d] == ["0", "1", "mean 0,1"]
    assert abs(float(d[2]["B"]) - 3 / 8) < EXACT and d[2]["U_convention"] == "U_full"


# ------------------------------------------------------- CLI equals the API ---
def test_cli_csv_equals_api_arrays(tmp_path, capsys):
    a, b = _synthetic_runs()
    pa, pb = _write_arrays(tmp_path, a, "a"), _write_arrays(tmp_path, b, "b")
    out = tmp_path / "out"
    assert cli.main(["evaluate", "--arrays", *pa, "--arrays-b", *pb, "--splits", "6", "--mean-over", "0", "1",
                     "--names", "A", "B", "--out", str(out)]) == 0
    lev, dec, con = (_read_csv(out / f"{t}.csv") for t in ("levels", "decomposition", "contrasts"))
    ev = {}
    for side, runs in (("A", a), ("B", b)):
        for r in runs:
            ev[(side, r["seed"], "final")] = crossfit.evaluate(r["final"], r["labels"], r["seed"], r["period_ids"],
                                                               rules=r["rules"], splits=6)
            ev[(side, r["seed"], "reference")] = crossfit.evaluate(r["reference"], r["labels"], r["seed"],
                                                                   r["period_ids"], rules=r["rules"], splits=6,
                                                                   single_model=False)
    checked = 0
    for row in lev:
        e = ev[(row["method"], int(row["seed"]), row["model"])]
        if row["period"].startswith("mean"):
            sm = e.summary([0, 1])
            want = dict(V_sep=sm["V_sep"], U_half=sm["rules"]["inherited"]["U_half"],
                        U_full=sm["rules"]["inherited"]["U_full"], R_half=sm["rules"]["inherited"]["R"])
            if row["model"] == "final":
                want.update(V_shared=sm["V_shared"], C=sm["C"], M=sm["rules"]["inherited"]["M"])
        else:
            pp = e.per_period()[int(row["period"])]
            want = dict(V_sep=pp["V_sep"], U_half=pp["U_half"]["inherited"], U_full=pp["U_full"]["inherited"])
            if row["model"] == "final":
                want.update(V_shared=pp["V_shared"])
        for k, v in want.items():
            assert float(row[k]) == v, (row, k)
            checked += 1
        if row["model"] == "reference":
            assert row["V_shared"] == "" and row["C"] == "" and row["M"] == ""
    assert checked > 100
    for row in dec:
        side, s = row["method"], int(row["seed"])
        per = [0, 1] if row["period"].startswith("mean") else [int(row["period"])]
        d = crossfit.decompose(ev[(side, s, "reference")], ev[(side, s, "final")], "inherited", periods=per)
        for k in ("B", "A", "dR", "U_ref", "U_final", "V_ref", "V_final"):
            assert float(row[k]) == d[k]
    U = lambda side, s, m: ev[(side, s, m)].summary([0, 1])["rules"]["inherited"]["U_full"]
    V = lambda side, s, m, k: ev[(side, s, m)].summary([0, 1])[k]
    seeds = [0, 1, 2]
    want = {("A minus B", "at the rule"): [U("A", s, "final") - U("B", s, "final") for s in seeds],
            ("A minus B", "after own-period refit"): [V("A", s, "final", "V_sep") - V("B", s, "final", "V_sep")
                                                       for s in seeds],
            ("A minus B", "after one shared refit"): [V("A", s, "final", "V_shared") - V("B", s, "final", "V_shared")
                                                       for s in seeds],
            ("reference minus final (A)", "at the rule"): [U("A", s, "reference") - U("A", s, "final") for s in seeds],
            ("reference minus final (A)", "after own-period refit"):
                [V("A", s, "reference", "V_sep") - V("A", s, "final", "V_sep") for s in seeds]}
    rows = {(r["comparison"], r["term"]): r for r in con}
    for key, x in want.items():
        st = inference.paired_stats(np.array(x))
        r = rows[key]
        assert [float(v) for v in r["values"].split()] == x
        assert float(r["mean"]) == st["mean"]
        assert float(r["ci95_low"]) == st["ci95"][0] and float(r["ci95_high"]) == st["ci95"][1]
        assert float(r["p_t_unadjusted"]) == st["p_t"]
    assert len(con) == 7                     # 3 method terms, 2 + 2 reference terms


def test_cli_csv_equals_api_stored_run(tmp_path, capsys):
    """The stored-run route (P_<i>_<j> files) equals runs.evaluate_npz."""
    rng = np.random.default_rng(9)
    N, K = 3, 2
    Ys = [(rng.random((30 + 5 * j, K)) < 0.35).astype(int) for j in range(N)]
    z = {f"Y_{j}": Ys[j] for j in range(N)}
    for i in range(N):
        for j in range(N):
            z[f"P_{i}_{j}"] = 1 / (1 + np.exp(-(2 * Ys[j] + rng.normal(0, 1, Ys[j].shape) - 0.3 * abs(i - j))))
    z["thresholds"] = np.array([0.45, 0.6])
    p = tmp_path / "run_s4_scores.npz"
    np.savez(p, **z)
    out = tmp_path / "out"
    assert cli.main(["evaluate", "--npz", str(p), "--seeds", "4", "--splits", "5", "--u-convention", "half",
                     "--out", str(out)]) == 0
    fin = rg.evaluate_npz(str(p), 4, splits=5)
    ref = rg.evaluate_npz(str(p), 4, splits=5, model="diagonal")
    lev = [r for r in _read_csv(out / "levels.csv") if r["model"] == "final" and not r["period"].startswith("mean")]
    for r in lev:
        pp = fin.per_period()[int(r["period"])]
        assert float(r["V_sep"]) == pp["V_sep"] and float(r["V_shared"]) == pp["V_shared"]
        assert float(r["U_half"]) == pp["U_half"]["inherited"] and float(r["U_full"]) == pp["U_full"]["inherited"]
    dec = _read_csv(out / "decomposition.csv")
    d = crossfit.decompose(ref, fin, "inherited", halves=True)
    assert dec[-1]["U_convention"] == "U_half" and float(dec[-1]["B"]) == d["B"] and float(dec[-1]["A"]) == d["A"]


# --------------------------------------------------- row metadata and status ---
def test_rows_carry_metadata_and_status(tmp_path, capsys):
    a, b = _synthetic_runs()
    res = analysis.analyse(a, b, splits=4)
    meta = ("direction", "sample_unit", "interval_type", "family", "adjustment", "margin", "status")
    for table in ("levels", "decomposition", "contrasts"):
        assert res[table]
        for r in res[table]:
            assert all(r.get(k) not in (None, "") for k in meta), (table, r)
            assert r["status"] == "descriptive" and r["family"] == "none"
    assert all(r["class"] is None for r in res["contrasts"])
    fam = analysis.analyse(a, b, splits=4, family="my-planned-family")
    for r in fam["contrasts"]:
        assert r["status"] == "planned family my-planned-family"
        assert r["class"] in ("material", "small non-zero", "equivalent", "unresolved")
        assert r["adjustment"].startswith("Holm over")
    assert all(r["status"] == "descriptive" for r in fam["levels"] + fam["decomposition"])
    # the family's classes equal inference.family_ni on the same per-seed vectors
    rows = [r for r in fam["contrasts"] if r["comparison"] == "a minus b"]
    ref = inference.family_ni({r["term"]: np.array([float(v) for v in r["values"].split()]) for r in rows})
    for r in rows:
        assert r["class"] == ref[r["term"]]["class"] and r["p_t_holm"] == ref[r["term"]]["p_holm"]
    out = tmp_path / "t"
    analysis.write_tables(fam, out)
    for t in ("levels", "decomposition", "contrasts"):
        for row in _read_csv(out / f"{t}.csv"):
            assert not any("confirmatory" in v.lower() for v in row.values())


def test_one_run_has_no_interval():
    res = tiny.tiny_result()
    for r in res["contrasts"]:
        assert r["n"] == 1 and r["ci95_low"] is None and r["interval_type"].startswith("none")
    assert res["interpretation"][0].endswith("with one run no interval is computed.")


def test_two_u_conventions():
    """evaluate() returns both levels; the decomposition reads the one asked for."""
    a, _ = _synthetic_runs(seeds=(0,))
    full = analysis.analyse(a, splits=5, u_convention="full")
    half = analysis.analyse(a, splits=5, u_convention="half")
    ev_f = full["evaluations"][("a", 0, "final")]
    sm = ev_f.summary()
    df, dh = full["decomposition"][-1], half["decomposition"][-1]
    assert df["U_final"] == sm["rules"]["inherited"]["U_full"] and dh["U_final"] == sm["rules"]["inherited"]["U_half"]
    assert df["U_final"] != dh["U_final"]
    assert abs(df["B"] - df["A"] - df["dR"]) < EXACT and abs(dh["B"] - dh["A"] - dh["dR"]) < EXACT
    with pytest.raises(ValueError):
        analysis.analyse(a, splits=2, u_convention="both")


def test_validation_refuses_bad_pairing():
    a, b = _synthetic_runs(seeds=(0, 1))
    with pytest.raises(ValueError, match="different seeds"):
        analysis.analyse(a, b[:1], splits=2)
    b[0] = dict(b[0], labels=[Y[::-1] for Y in b[0]["labels"]], final=[P[::-1] for P in b[0]["final"]])
    with pytest.raises(ValueError, match="record order"):
        analysis.analyse(a, b, splits=2)
    names = [dict(r, label_names=["x", "y", "z"]) for r in a]
    names[1] = dict(names[1], label_names=["y", "x", "z"])
    with pytest.raises(ValueError, match="label names"):
        analysis.analyse(names, splits=2)
    checks = analysis.analyse(a, splits=2)["validation"]
    assert [c["check"] for c in checks] == ["period ids", "label order", "alignment", "pairing"]


# ------------------------------------------------------- figures and report ---
def test_figures_export(tmp_path):
    pytest.importorskip("matplotlib")
    from refitgap import plots
    a, b = _synthetic_runs()
    paths = plots.save_figures(analysis.analyse(a, b, splits=3), tmp_path)
    assert sorted(paths) == ["comparison", "decomposition", "profile"]
    for d in paths.values():
        assert d["pdf"].read_bytes()[:4] == b"%PDF" and "<svg" in d["svg"].read_text()


def test_report_is_self_contained(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    a, b = _synthetic_runs()
    pa, pb = _write_arrays(tmp_path, a, "a"), _write_arrays(tmp_path, b, "b")
    out = tmp_path / "rep"
    assert cli.main(["report", "--arrays", *pa, "--arrays-b", *pb, "--splits", "3", "--out", str(out)]) == 0
    page = (out / "report.html").read_text()
    assert "<script" not in page.lower() and "javascript:" not in page.lower()
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", page))
    assert all(u.startswith(("http://www.w3.org/", "http://purl.org/dc/", "http://creativecommons.org/",
                             "https://matplotlib.org")) for u in urls), urls
    assert not re.search(r"<link[^>]+href", page) and "@import" not in page
    assert page.count("<svg") == 3
    import html
    assert html.escape(analysis.VOCABULARY_NOTE) in page and "confer no confirmatory status" in page
    assert "descriptive" in page and "after own-period refit the lead goes from" in page


def test_report_without_figures():
    from refitgap.report import render
    page = render(tiny.tiny_result(), svgs={})
    assert "Figures were not drawn" in page and "<svg" not in page


def test_core_does_not_import_matplotlib():
    code = ("import sys, refitgap, refitgap.cli, refitgap.report, refitgap.tiny as t;"
            "refitgap.report.render(t.tiny_result(), svgs={});"
            "print('matplotlib' in sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "False"


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == f"refitgap {rg.__version__}"
    assert rg.__version__ == "0.2.1"
    if (REPO / "pyproject.toml").exists():          # one version source: pyproject reads refitgap.__version__
        text = (REPO / "pyproject.toml").read_text()
        assert 'dynamic = ["version"]' in text and 'attr = "refitgap.__version__"' in text
        assert "\nversion = " not in text.split("[tool.setuptools.dynamic]")[0]
    if (REPO / "CITATION.cff").exists():
        assert f"version: {rg.__version__}" in (REPO / "CITATION.cff").read_text()


# ------------------------------------- no recommendation or diagnosis in outputs ---
def test_no_recommendation_or_diagnosis_in_new_outputs(tmp_path, capsys):
    """test_hand_checks.py::test_no_retrain_recommendation_in_outputs, extended to the command
    line, the CSV and JSON tables, the figures and the HTML report."""
    src = Path(rg.__file__).resolve().parent
    for name in ("analysis.py", "cli.py", "plots.py", "report.py", "tiny.py", "__main__.py"):
        _scan((src / name).read_text(), name)
    has_mpl = importlib.util.find_spec("matplotlib") is not None
    a, b = _synthetic_runs()
    pa, pb = _write_arrays(tmp_path, a, "a"), _write_arrays(tmp_path, b, "b")
    out = tmp_path / "o"
    cli.main(["report" if has_mpl else "evaluate", "--arrays", *pa, "--arrays-b", *pb, "--splits", "3",
              "--family", "fam", "--out", str(out)])
    cli.main(["example", "--out", str(tmp_path / "tiny")] if has_mpl else ["example"])
    _scan(capsys.readouterr().out, "stdout", [tmp_path, Path(os.path.realpath(tmp_path))])
    files = [p for d in (out, tmp_path / "tiny") if d.exists() for p in d.iterdir()
             if p.suffix in (".csv", ".json", ".html", ".svg")]
    assert files
    for p in files:
        _scan(p.read_text(), p.name, [tmp_path, Path(os.path.realpath(tmp_path))])


# ------------------------------------------------------------- edge cases ---
def test_per_period_reference_fits_no_shared_rule():
    """single_model=False: no shared rule across different models' scores is fitted or exposed;
    every other level is unchanged against the same scores evaluated as one model."""
    a, _ = _synthetic_runs(seeds=(0,))
    r = a[0]
    ref = crossfit.evaluate(r["reference"], r["labels"], 0, rules=r["rules"], splits=4, single_model=False,
                            shared_periods=[0, 1])
    one = crossfit.evaluate(r["reference"], r["labels"], 0, rules=r["rules"], splits=4, shared_periods=[0, 1])
    assert np.isnan(ref.shared).all()
    pp = ref.per_period()
    assert all(pp[j]["V_shared"] is None for j in ref.period_ids)
    sm = ref.summary([0, 1])
    assert sm["V_shared"] is None and sm["C"] is None and "M" not in sm["rules"]["inherited"]
    assert np.array_equal(ref.sep, one.sep) and np.array_equal(ref.U_half["inherited"], one.U_half["inherited"])
    assert sm["V_sep"] == one.summary([0, 1])["V_sep"]


def test_no_positive_label_at_large_magnitude():
    """The no-positive cutpoint is max + 1e-9 (as decomposition.py) wherever that lies above the
    top score. At 1e9 the 1e-9 is lost in float64, so the cutpoint is the next float above the
    top score and nothing is predicted; F1 is 0 either way."""
    P = np.array([[1e9, 0.9], [2e9, 0.1], [3e9, 0.5]])
    Y = np.array([[0, 1], [0, 0], [0, 1]])
    th = rg.fit_exact(P, Y)
    assert th[0] == np.nextafter(3e9, np.inf) and (P[:, 0] >= th[0]).sum() == 0
    assert rg.fit_shared_exact([P], [Y])[0] == np.nextafter(3e9, np.inf)
    assert rg.f1_vec(P, Y, th)[0] == 0.0
    small = rg.fit_exact(P / 1e9, Y)                # moderate magnitude: max + 1e-9, as before
    assert small[0] == 3.0 + 1e-9 and (P[:, 0] / 1e9 >= small[0]).sum() == 0
    # one record would have been predicted
    assert rg.fit_exact(np.array([[2e8], [1e8]]), np.array([[0], [0]]))[0] > 2e8


def test_no_positive_cutpoint_unchanged_on_probability_scale():
    """For scores in [0, 1] the cutpoint is exactly max + 1e-9 (the ported expression), bit for bit."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        P = rng.random((int(rng.integers(2, 30)), 3))
        P[rng.random(P.shape) < 0.2] = 1.0                       # include the top of the scale
        Y = np.zeros(P.shape, int)
        th = rg.fit_exact(P, Y)
        assert np.array_equal(th, P.max(0) + 1e-9)
        assert rg.fit_shared_exact([P, P[::-1]], [Y, Y])[0] == float(P[:, 0].max()) + 1e-9


def test_signflip_above_the_exact_limit_is_monte_carlo():
    """Above SIGNFLIP_MAX_N seeds the sign-flip p values are Monte Carlo estimates with a fixed
    seed and SIGNFLIP_MC_DRAWS draws, labelled in the output; nothing fails."""
    n = inference.SIGNFLIP_MAX_N + 5
    x = np.linspace(-0.01, 0.02, n)
    p = inference.signflip_p(x)
    assert p == inference.signflip_p(x)                      # fixed seed: reproducible
    assert 1 / (1 + inference.SIGNFLIP_MC_DRAWS) <= p <= 1
    assert inference.SIGNFLIP_MC_DRAWS == 100_000
    st = inference.paired_stats(x)
    assert "Monte Carlo" in st["p_signflip_method"] and "100000" in st["p_signflip_method"]
    fam = inference.family_ni({"a": x})["a"]
    assert "Monte Carlo" in fam["p_ni_signflip_method"] and "Monte Carlo" in fam["p_inf_signflip_method"]
    c = inference.paired_contrast({s: v for s, v in enumerate(x)}, {s: 0.0 for s in range(n)})
    assert c["n"] == n and "p_signflip_method" in c
    # agreement with the exact p in distribution: a strong effect and a null-like vector
    rng = np.random.default_rng(4)
    strong = 0.01 + rng.normal(0, 0.002, n)
    assert inference.signflip_p(strong) == 1 / (1 + inference.SIGNFLIP_MC_DRAWS)   # exact p = 2/2^25
    null = rng.normal(0, 1, n)
    # exact two-sided p by the normal approximation of the sign-flip distribution is close to the t p
    assert abs(inference.signflip_p(null) - inference.paired_stats(null)["p_t"]) < 0.1
    # the MC p agrees with full enumeration where both are feasible (n = 20 forced through the MC path)
    y = rng.normal(0.3, 1, 20)
    exact = inference._signflip_p_enumerated(y)
    old = inference.SIGNFLIP_MAX_N
    try:
        inference.SIGNFLIP_MAX_N = 19
        mc = inference.signflip_p(y)
    finally:
        inference.SIGNFLIP_MAX_N = old
    assert abs(mc - exact) < 4 * np.sqrt(exact * (1 - exact) / inference.SIGNFLIP_MC_DRAWS) + 1e-5
    y = x[:8]                                        # within the limit: the copied function's value
    assert inference.signflip_p(y) == inference._signflip_p_enumerated(y)
    assert inference.ni_signflip_p(y) == inference._ni_signflip_p_enumerated(y)
    assert "p_signflip_method" not in inference.paired_stats(y)    # exact outputs carry no new key


def test_csv_cells_plain_float_repr():
    """numpy >= 2 reprs np.float64 as 'np.float64(x)'; CSV cells must hold the plain float repr."""
    from refitgap.analysis import _cell
    assert _cell(np.float64(-0.1684720937325767)) == "-0.1684720937325767"
    assert _cell(np.float32(0.5)) == "0.5" and _cell(0.25) == "0.25" and _cell(None) == "" and _cell(3) == "3"
