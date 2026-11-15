"""The two entry points and the Table 1 example, run as a referee or a colleague would (seconds).

Figure 2: all five comparisons from stored levels against expected.json, plus one BiGRU pair
per stream recomputed from the raw scores on one period (the full raw run is in
test_slow_examples.py). Your own models: the synthetic CSV example, and loud failures on
mismatched inputs.
"""
import importlib.util
import json
import shutil
import sys

import numpy as np
import pytest

import refitgap as rg
from refitgap.runs import load_run

from conftest import REPO

EXAMPLES = REPO / "examples"
RESULTS = REPO / "paper" / "results"
pytestmark = pytest.mark.skipif(not (EXAMPLES / "figure2" / "expected.json").exists(),
                                reason="needs a clone of the repository (examples/ and paper/results/)")


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def test_table1_example():
    assert _module(EXAMPLES / "two_scorer" / "table1.py").main() == 0


def test_figure2_from_stored_levels(capsys):
    fig2 = _module(EXAMPLES / "figure2" / "reproduce_figure2.py")
    assert fig2.main(["--from-levels"]) == 0
    out = capsys.readouterr().out
    for cell in ("+0.0109", "-0.0113", "+0.0241", "+0.0223", "+0.0217", "+0.0198", "-0.0019",
                 "+0.0086", "+0.0034", "-0.0052", "+0.0305", "+0.0243", "-0.0062"):
        assert cell in out


@pytest.mark.parametrize("folder,stream,stored", [("asrs_bigru_paired", "primary", "paired_replay.json"),
                                                  ("multieurlex_bigru", "eurlex", "paired_replay_eurlex.json")])
def test_figure2_one_bigru_pair_from_raw_scores(folder, stream, stored):
    """Seed 0, period 2 (id 1): compare() on the raw scores gives the stored levels to 1e-12."""
    f = lambda arm: RESULTS / folder / f"{stream}_C2_{arm}_lam1_s0_scores.npz"   # noqa: E731
    rep, seq = load_run(f("replay"), periods=[1]), load_run(f("naive"), periods=[1])
    r = rg.compare(rep["Ps"], seq["Ps"], rep["Ys"], rep["thresholds"], seed=0, splits=200, period_ids=[1],
                   labels_final=seq["Ys"])
    st = json.loads((RESULTS / "paired_refit" / stored).read_text())["pairs"]["0"]["per_period"]["1"]
    got = [r["carried"]["ref"], r["carried"]["final"], r["refitted"]["ref"], r["refitted"]["final"]]
    assert np.max(np.abs(np.array(got) - [st["U_rep"], st["U_seq"], st["V_rep"], st["V_seq"]])) <= 1e-12


def test_your_models_example(capsys):
    mod = _module(EXAMPLES / "your_models" / "compare_your_models.py")
    assert mod.main([str(EXAMPLES / "your_models" / "data"), "--splits", "20"]) == 0
    out = capsys.readouterr().out
    assert "mine against baseline: 4 labels, periods [1, 2, 3], seeds [0, 1, 2, 3]" in out


@pytest.fixture
def data_copy(tmp_path):
    d = tmp_path / "data"
    shutil.copytree(EXAMPLES / "your_models" / "data", d)
    return d


def _edit(path, fn):
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(fn(lines)))


@pytest.mark.parametrize("damage,message", [
    (lambda d: _edit(d / "scores_mine_seed1.csv", lambda ls: ls[:-1]), "record ids"),
    (lambda d: _edit(d / "scores_mine_seed1.csv", lambda ls: ls + [ls[-1]]), "duplicate"),
    (lambda d: _edit(d / "labels.csv", lambda ls: [ls[0].replace("crew", "cabin")] + ls[1:]), "label columns"),
    (lambda d: _edit(d / "thresholds.csv", lambda ls: [l for l in ls if not l.startswith("baseline,3")]),
     "seeds differ"),
    (lambda d: _edit(d / "scores_baseline_seed0.csv", lambda ls: ls + ["9,p9-000,0.1,0.2,0.3,0.4\n"]), "no labels"),
])
def test_your_models_rejects_mismatched_inputs(data_copy, damage, message):
    mod = _module(EXAMPLES / "your_models" / "compare_your_models.py")
    damage(data_copy)
    with pytest.raises(ValueError, match=message):
        mod.main([str(data_copy), "--splits", "2"])


def test_compare_rejects_shape_and_id_mismatch():
    rng = np.random.default_rng(0)
    Y = (rng.random((20, 3)) < 0.3).astype(int)
    P = rng.random((20, 3))
    with pytest.raises(ValueError, match="shape"):
        rg.compare(P, P[:19], Y, [0.5] * 3, seed=0, splits=2)
    with pytest.raises(ValueError, match="differ"):
        ids = list(range(20))
        rg.compare(
            P, P, Y, [0.5] * 3, seed=0, splits=2, record_ids={"labels": ids, "ref": ids, "final": ids[:-1] + [99]}
        )
    with pytest.raises(ValueError, match="duplicate"):
        rg.compare(P, P, Y, [0.5] * 3, seed=0, splits=2, record_ids=[0] * 20)
