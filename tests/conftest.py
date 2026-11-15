"""Shared fixtures and paths. REPO is the repository root."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEC_SHA16 = "9eb637182d987d7e"          # paper/scripts/decomposition.py (numerics as the paper ran them)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: recomputes stored paper numbers from score files (minutes to hours)")


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


@pytest.fixture(scope="session")
def DEC():
    """paper/scripts/decomposition.py, the standalone decomposition the method comparison used."""
    p = REPO / "paper" / "scripts" / "decomposition.py"
    if not p.exists():
        pytest.skip(f"{p} absent (tests run outside the repository)")
    assert sha16(p) == DEC_SHA16, "decomposition.py changed"
    spec = importlib.util.spec_from_file_location("decomposition", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
