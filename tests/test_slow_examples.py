"""Slow: the full Figure 2 recomputation from raw BiGRU scores (minutes of CPU, data in the repository).

    python -m pytest -m slow tests/test_slow_examples.py
"""
import importlib.util
import os
import sys

import pytest

from conftest import REPO

pytestmark = pytest.mark.slow


def test_figure2_from_raw_scores():
    p = REPO / "examples" / "figure2" / "reproduce_figure2.py"
    if not p.exists():
        pytest.skip(f"{p} absent (tests run outside the repository)")
    spec = importlib.util.spec_from_file_location("reproduce_figure2", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reproduce_figure2"] = mod
    spec.loader.exec_module(mod)
    assert mod.main(["--workers", os.environ.get("REFITGAP_WORKERS", "4")]) == 0
