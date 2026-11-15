"""family_at(members, margin) is family(members, margin)."""
import numpy as np
import pytest

from refitgap import inference

def _members():
    rng = np.random.default_rng(3)
    return {"a": rng.normal(0.0004, 0.002, 6), "b": rng.normal(0.011, 0.002, 6)}

def test_family_at_default_matches_family():
    m = _members()
    f, g = inference.family(m), inference.family_at(m, inference.MARGIN)
    for k in m:
        assert f[k]["class"] == g[k]["class"]
        assert f[k]["p_tost_holm"] == pytest.approx(g[k]["p_tost_holm"], abs=1e-15)

def test_family_at_moves_tost_and_restores_margin():
    m = _members()
    lo, hi = inference.family_at(m, 0.003), inference.family_at(m, 0.010)
    assert lo["a"]["p_tost_holm"] != hi["a"]["p_tost_holm"]
    assert inference.MARGIN == 0.005
