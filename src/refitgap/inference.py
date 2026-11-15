"""Seed-paired contrasts: t intervals, TOST, sign-flip tests, Holm families and outcome classes.

A contrast is a per-seed vector x (method minus comparator). family() Holm-adjusts the t and
TOST p values across its members and assigns, at `margin` (default MARGIN = 0.005):
    equivalent      Holm TOST p < ALPHA
    material        Holm t p < ALPHA and |mean| >= margin
    small non-zero  Holm t p < ALPHA and |mean| <  margin
    unresolved      otherwise, or when the sign-flip p >= ALPHA overrules the two above
Intervals are marginal (unadjusted). Sign-flip tests are exact up to SIGNFLIP_MAX_N seeds and
Monte Carlo above (the result then says so). Non-finite values raise ValueError.
"""
import inspect
import itertools

import numpy as np
from scipy import stats

MARGIN, ALPHA = 0.005, 0.05
BCA_N, BCA_SEED = 10_000, 20261029
SIGNFLIP_MAX_N = 20                      # exact enumeration up to here (2^20 patterns)
SIGNFLIP_MC_DRAWS = 100_000
SIGNFLIP_MC_SEED = 20261114
_SIGNFLIP_MC_CHUNK = 10_000

# scipy renamed bootstrap's random_state= to rng=
try:
    _BOOT_RNG_KW = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"
except (TypeError, ValueError):
    _BOOT_RNG_KW = "random_state"


# ------------------------------------------------------------------ sign-flip tests ---
def _signflip_p_enumerated(x):
    x = np.asarray(x, float)
    n, obs = len(x), abs(x.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    means = np.abs((signs * x).mean(axis=1))
    return float(np.mean(means >= obs - 1e-12))


def _ni_signflip_p_enumerated(x, margin=MARGIN):
    y = np.asarray(x, float) + margin
    obs = y.mean()
    if np.allclose(y, 0):
        return 1.0
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(y))))
    return float(np.mean((signs * np.abs(y)).mean(1) >= obs - 1e-12)) if obs > 0 else 1.0


def signflip_method(n):
    """n seeds -> how the sign-flip p is computed, as text."""
    if n <= SIGNFLIP_MAX_N:
        return f"exact (all 2^{n} sign patterns)"
    return (f"Monte Carlo ({SIGNFLIP_MC_DRAWS} random sign patterns from numpy default_rng({SIGNFLIP_MC_SEED}); "
            f"p = (1 + count) / (1 + {SIGNFLIP_MC_DRAWS}), the observed pattern counted once)")


def _mc_sign_chunks(n):
    rng = np.random.default_rng(SIGNFLIP_MC_SEED)
    left = SIGNFLIP_MC_DRAWS
    while left > 0:
        b = min(_SIGNFLIP_MC_CHUNK, left)
        yield rng.integers(0, 2, size=(b, n)).astype(float) * 2.0 - 1.0
        left -= b


def signflip_p(x):
    """Per-seed vector -> two-sided sign-flip p of the mean."""
    x = np.asarray(x, float)
    n = len(x)
    if n <= SIGNFLIP_MAX_N:
        return _signflip_p_enumerated(x)
    obs = abs(x.mean())
    count = sum(int(np.sum(np.abs((s * x).mean(axis=1)) >= obs - 1e-12)) for s in _mc_sign_chunks(n))
    return float((1 + count) / (1 + SIGNFLIP_MC_DRAWS))


def ni_signflip_p(x, margin=MARGIN):
    """Per-seed vector -> one-sided sign-flip p of H0 'symmetric about -margin'."""
    x = np.asarray(x, float)
    n = len(x)
    if n <= SIGNFLIP_MAX_N:
        return _ni_signflip_p_enumerated(x, margin)
    y = x + margin
    obs = y.mean()
    if np.allclose(y, 0) or not obs > 0:
        return 1.0
    ay = np.abs(y)
    count = sum(int(np.sum((s * ay).mean(1) >= obs - 1e-12)) for s in _mc_sign_chunks(n))
    return float((1 + count) / (1 + SIGNFLIP_MC_DRAWS))


def ni_p(x, margin=MARGIN):
    """Per-seed vector -> one-sided paired t p for non-inferiority (H0: mean <= -margin)."""
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se == 0:
        return 0.0 if m > -margin else 1.0
    return float(1 - stats.t.cdf((m + margin) / se, n - 1))


# ------------------------------------------------------------------ input checks ---
def _check_finite(x, what):
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"{what}: a per-seed vector (1-d) is expected, got shape {x.shape}")
    bad = np.flatnonzero(~np.isfinite(x))
    if bad.size:
        raise ValueError(
            f"{what}: non-finite value {float(x[bad[0]])!r} at position {int(bad[0])} ({bad.size} non-finite in "
            "all); a failed run must be dropped (with the seed set adjusted openly) or re-run, not "
            "stored as NaN or inf"
        )
    return x


def _check_margin(margin):
    m = float(margin)
    if not np.isfinite(m) or m < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin!r}")
    return m


# ------------------------------------------------------------------ one contrast ---
def bca_ci(x):
    """Per-seed vector -> BCa 95 % interval of the mean, or None (fewer than 3 values, constant,
    or BCa undefined). Sensitivity only; it never decides a class."""
    x = np.asarray(x, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    try:
        r = stats.bootstrap((x,), np.mean, method="BCa", n_resamples=BCA_N,
                            **{_BOOT_RNG_KW: np.random.default_rng(BCA_SEED)})
    except (ValueError, ArithmeticError):
        return None
    return [float(r.confidence_interval.low), float(r.confidence_interval.high)]


def paired_stats(x, margin=MARGIN, name="x"):
    """Per-seed vector -> dict: mean, sd, se, 95 % and 90 % t intervals, t p, TOST p at +-margin,
    sign-flip p, BCa interval. All values equal: p_t is 1 for mean 0 and 0 otherwise."""
    x = _check_finite(x, name)
    margin = _check_margin(margin)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return dict(n=n, mean=m)
    se = float(x.std(ddof=1) / np.sqrt(n))
    t95 = stats.t.ppf(0.975, n - 1) * se
    t90 = stats.t.ppf(0.95, n - 1) * se
    p = float(stats.ttest_1samp(x, 0.0).pvalue) if se > 0 else 0.0
    if se > 0:
        p_lo = float(1 - stats.t.cdf((m + margin) / se, n - 1))
        p_hi = float(stats.t.cdf((m - margin) / se, n - 1))
    else:
        p_lo = p_hi = 0.0 if abs(m) < margin else 1.0
    p_tost = max(p_lo, p_hi)
    st = dict(n=n, mean=m, se=se, sd=float(x.std(ddof=1)), ci95=[m - t95, m + t95], ci90=[m - t90, m + t90],
              p_t=p, p_tost=p_tost, p_lo=p_lo, p_hi=p_hi, tost_pass=bool(p_tost < ALPHA),
              p_signflip=signflip_p(x), values=x.tolist(), bca95=bca_ci(x))
    if se == 0 and m == 0:
        st["p_t"] = 1.0
    if n > SIGNFLIP_MAX_N:
        st["p_signflip_method"] = signflip_method(n)
    return st


def _holm_core(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def holm(ps):
    """p values in [0, 1] -> Holm step-down adjusted p values (a NaN raises)."""
    ps = np.asarray(ps, float)
    if ps.ndim != 1:
        raise ValueError(f"holm: a 1-d sequence of p values is expected, got shape {ps.shape}")
    bad = np.flatnonzero(~np.isfinite(ps) | (ps < 0) | (ps > 1))
    if bad.size:
        raise ValueError(f"holm: p value {float(ps[bad[0]])!r} at position {int(bad[0])} is not a number in [0, 1]")
    return _holm_core(ps)


def classify(st, margin=MARGIN):
    """paired_stats dict with p_holm, p_tost_holm -> class (before the sign-flip override)."""
    if st.get("n", 0) < 2:
        return "unresolved"
    if st["p_tost_holm"] < ALPHA:
        return "equivalent"
    if st["p_holm"] < ALPHA:
        return "material" if abs(st["mean"]) >= margin else "small non-zero"
    return "unresolved"


def signflip_override(st, margin=MARGIN):
    """Sets st['class']: classify(), turned unresolved when the sign-flip test disagrees."""
    cls = classify(st, margin)
    st["class_before_signflip"] = cls
    st["signflip_disagrees"] = False
    if cls in ("material", "small non-zero") and st["p_signflip"] >= ALPHA:
        st["signflip_disagrees"] = True
        cls = "unresolved"
    st["class"] = cls
    return st


def classify_unadjusted(st, margin=MARGIN):
    """paired_stats dict -> the same with its class as a family of one (no Holm)."""
    if st.get("n", 0) < 2:
        st["class"] = "unresolved"
        return st
    st["p_holm"], st["p_tost_holm"] = st["p_t"], st["p_tost"]
    st["holm_family_size"] = 1
    return signflip_override(st, margin)


# ------------------------------------------------------------------ families ---
def family(members, margin=MARGIN):
    """{name: per-seed vector} -> {name: paired_stats dict with Holm p values and class}."""
    margin = _check_margin(margin)
    names = list(members)
    sts = {k: paired_stats(members[k], margin, name=f"family member {k!r}") for k in names}
    pt = holm([sts[k]["p_t"] for k in names])
    pq = holm([sts[k]["p_tost"] for k in names])
    for k, a, b in zip(names, pt, pq):
        sts[k]["p_holm"], sts[k]["p_tost_holm"], sts[k]["holm_family_size"] = a, b, len(names)
        signflip_override(sts[k], margin)
    return sts


def family_at(members, margin):
    """Same as family(members, margin)."""
    return family(members, margin)


def family_ni(members, margin=MARGIN):
    """family() plus, per member, non-inferiority at -margin ('ni': Holm one-sided t and
    sign-flip both < ALPHA) and the mirror inferiority test ('inf': below -margin)."""
    margin = _check_margin(margin)
    names = list(members)
    vals = {k: _check_finite(members[k], f"family member {k!r}") for k in names}
    sts = family(vals, margin)
    p_ni = [ni_p(vals[k], margin) for k in names]
    for k, p, q in zip(names, p_ni, holm(p_ni)):
        s = sts[k]
        s["p_ni"], s["p_ni_holm"] = p, q
        s["p_ni_signflip"] = ni_signflip_p(vals[k], margin)
        s["ni_signflip_disagrees"] = bool(q < ALPHA and s["p_ni_signflip"] >= ALPHA)
        s["ni"] = bool(q < ALPHA and not s["ni_signflip_disagrees"])
        s["margin"] = margin
    p_inf = [ni_p(-vals[k], -margin) for k in names]
    for k, p, q in zip(names, p_inf, holm(p_inf)):
        s = sts[k]
        s["p_inf"], s["p_inf_holm"] = p, q
        s["p_inf_signflip"] = ni_signflip_p(-vals[k], -margin)
        s["inf"] = bool(q < ALPHA and s["p_inf_signflip"] < ALPHA)
        if len(vals[k]) > SIGNFLIP_MAX_N:
            s["p_ni_signflip_method"] = s["p_inf_signflip_method"] = signflip_method(len(vals[k]))
    return sts


def tost_power(sd, n=10, alphas=(0.0125, 0.05), reps=20_000, seed=20261029, margin=MARGIN):
    """sd of a paired contrast -> {alpha: TOST power at a true difference of 0} (normal draws;
    optimistic when sd comes from the same pairs)."""
    if not sd or not np.isfinite(sd):
        return {a: None for a in alphas}
    x = np.random.default_rng(seed).normal(0.0, sd, (reps, n))
    m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
    p = np.maximum(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    return {a: float(np.mean(p < a)) for a in alphas}


# ------------------------------------------------------------------ dicts by seed ---
def paired_vector(a, b, seeds=None, strict=True):
    """{seed: value} twice -> (seeds, a - b in sorted seed order). Different seed sets raise
    unless seeds= names the planned ones or strict=False takes the common ones; NaN raises."""
    if seeds is not None:
        seeds = sorted(int(s) for s in seeds)
        miss = {"a": [s for s in seeds if s not in a], "b": [s for s in seeds if s not in b]}
        if miss["a"] or miss["b"]:
            raise ValueError(f"planned seeds missing: {miss}")
    elif set(a) != set(b):
        if strict:
            raise ValueError(f"the two sides hold different seeds (only in a: {sorted(set(a) - set(b))}, "
                             f"only in b: {sorted(set(b) - set(a))}); pass seeds= or strict=False")
        seeds = sorted(set(a) & set(b))
    else:
        seeds = sorted(a)
    if not seeds:
        raise ValueError("no common seeds")
    for side, d in (("a", a), ("b", b)):
        bad = [s for s in seeds if not np.isfinite(float(d[s]))]
        if bad:
            raise ValueError(f"side {side}, seed {bad[0]}: non-finite value {float(d[bad[0]])!r}; a failed run must be "
                             "dropped (seeds=, strict=False) or re-run, not stored as NaN or inf")
    return seeds, np.array([a[s] - b[s] for s in seeds], float)


def dropped_seeds(a, b, seeds=None):
    """Seeds on one side only (or planned seeds missing from either)."""
    if seeds is not None:
        return sorted(s for s in seeds if s not in a or s not in b)
    return sorted(set(a) ^ set(b))


def paired_contrast(a, b, seeds=None, strict=True, margin=MARGIN):
    """{seed: value} twice -> paired_stats of a - b with its unadjusted class, seeds used and
    seeds dropped."""
    used, x = paired_vector(a, b, seeds, strict)
    st = classify_unadjusted(paired_stats(x, margin, name="paired contrast a - b"), margin)
    st["seeds"] = used
    st["dropped_seeds"] = [] if seeds is not None else dropped_seeds(a, b)
    return st
