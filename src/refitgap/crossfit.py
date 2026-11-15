"""Cross-fitted levels of one model, and the decompositions built from them.

Settings: sep (own-period rules, fitted per period), shared (one rule over `shared_periods`)
and fixed (any threshold vector under `rules`, e.g. "inherited"). Partition b of period id j,
seed s, is the b-th permutation of default_rng(7000 + 100*s + j); half 0 is the first n_j // 2
records. Each rule is fitted on one half and scored on the other, both ways, and averaged.
Every setting uses the same halves, so R = C + M and B = A + dR hold per partition."""
import hashlib
import numbers

import numpy as np

from .fitters import fit_exact, fit_grid, fit_shared_exact, macro_f1

SPLIT_BASE = 7000                       # partition seed: 7000 + 100*seed + j
SPLITS = 200                            # the paper's number of partitions
IDENTITY_TOL = 1e-12


MAX_PERIOD_ID = 100                     # period ids must lie in [0, MAX_PERIOD_ID); see check_partition_key


def _is_int(x):
    return isinstance(x, (numbers.Integral, np.integer)) and not isinstance(x, (bool, np.bool_))


def check_partition_key(seed, j=None, base=SPLIT_BASE):
    """(seed, period id) -> ValueError unless seed >= 0 and 0 <= id < 100, the range in
    which base + 100*seed + j keys partitions uniquely."""
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer (it keys the partitions through "
                         f"{base} + 100*seed + j), got {seed!r}")
    if j is not None and (not _is_int(j) or not 0 <= j < MAX_PERIOD_ID):
        raise ValueError(f"period id must be an integer in [0, {MAX_PERIOD_ID}), got {j!r}: the partition seed "
                         f"{base} + 100*seed + j keys (seed, period id) pairs uniquely only for such ids "
                         f"(period id 100 of seed s would repeat period id 0 of seed s + 1)")
    if not _is_int(base) or base < 0:
        raise ValueError(f"split base must be a non-negative integer, got {base!r}")


def partitions(n, seed, j, splits=SPLITS, base=SPLIT_BASE):
    """The partitions of period id j: `splits` successive permutations of range(n)
    from default_rng(base + 100*seed + j). Half 0 is perm[:n // 2].
    seed: a non-negative integer; j: an integer in [0, 100) (check_partition_key)."""
    check_partition_key(seed, j, base)
    rng = np.random.default_rng(base + 100 * seed + j)
    return [rng.permutation(n) for _ in range(splits)]


def fit_validation_rule(Pval, Yval):
    """Validation scores and labels -> grid rule (19 points 0.05..0.95), as theta_1 was
    fitted. Assumes probability-scale scores; for logits etc. pass your own rule under `rules`."""
    return fit_grid(np.asarray(Pval, np.float64), np.asarray(Yval).astype(int)).astype(np.float64)


def _check_inputs(Ps, Ys, period_ids):
    if len(Ps) != len(Ys) or len(Ps) == 0:
        raise ValueError("Ps and Ys must be non-empty lists of the same length (one entry per period)")
    K = np.asarray(Ps[0]).shape[1]
    for j, (P, Y) in enumerate(zip(Ps, Ys)):
        P, Y = np.asarray(P), np.asarray(Y)
        if P.ndim != 2 or P.shape != Y.shape or P.shape[1] != K:
            raise ValueError(f"period {period_ids[j]}: scores and labels must be (n_j, K) with one K for all periods")
        if P.shape[0] < 2:
            raise ValueError(f"period {period_ids[j]}: needs at least 2 records to split in half")
        if not np.all(np.isfinite(P)):
            raise ValueError(f"period {period_ids[j]}: non-finite scores")
        if not np.isin(Y, (0, 1)).all():
            raise ValueError(f"period {period_ids[j]}: labels must be 0/1")
    if len(set(period_ids)) != len(period_ids):
        raise ValueError("period ids must be distinct")


def _check_weights(w, n, what):
    """Finite, non-negative, one per period, not all zero. Returned as a tuple of floats."""
    w = np.asarray(w, float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"{what}: one finite non-negative weight per period, not all zero")
    return tuple(float(x) for x in w)


def array_hash(a, dtype):
    """sha16 of an array's shape and its bytes in `dtype` (C order). Equal hashes mean the same
    values in the same record order, which is what pairs two evaluations' partitions."""
    a = np.ascontiguousarray(a, dtype=dtype)
    h = hashlib.sha256(repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


class Evaluation:
    """Per-partition levels of one model, arrays of shape (splits, 2, periods): sep, shared,
    U_half[name]; U_full[name] per period. Also rules, label hashes and settings. Made by evaluate()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def _cols(self, periods):
        periods = self.period_ids if periods is None else list(periods)
        missing = [p for p in periods if p not in self.period_ids]
        if missing:
            raise KeyError(f"period ids {missing} were not evaluated")
        return periods, [self.period_ids.index(p) for p in periods]

    def per_period(self):
        """Per-period levels: mean over partitions of 0.5 * (direction 0 + direction 1)."""
        hm = lambda X, c: float(np.mean(0.5 * (X[:, 0, c] + X[:, 1, c])))
        out = {}
        for c, j in enumerate(self.period_ids):
            r = dict(V_sep=hm(self.sep, c),
                     V_shared=(hm(self.shared, c) if self.single_model and j in self.shared_periods else None),
                     U_half={k: hm(v, c) for k, v in self.U_half.items()},
                     U_full={k: float(v[c]) for k, v in self.U_full.items()})
            out[j] = r
        return out

    def summary(self, periods=None, weights=None):
        """Periods and weights -> mean levels and R = C + M (period-weighted per partition, then
        averaged). Weights default to uniform, or to shared_weights on the shared periods."""
        periods, cols = self._cols(periods)
        if weights is None and self.shared_weights is not None:
            if periods != list(self.shared_periods):
                raise ValueError("the shared rule was fitted with shared_weights; pass `weights` for periods "
                                 f"{periods} (they differ from the shared periods {self.shared_periods})")
            weights = self.shared_weights
        if weights is None:
            agg = lambda X: np.mean(0.5 * (X[:, 0, cols] + X[:, 1, cols]), axis=1)
            aggf = lambda v: float(np.mean([v[c] for c in cols]))
        else:
            w = np.asarray(_check_weights(weights, len(cols), "weights"))
            w = w / w.sum()
            agg = lambda X: (0.5 * (X[:, 0, cols] + X[:, 1, cols])) @ w
            aggf = lambda v: float(sum(wi * v[c] for wi, c in zip(w, cols)))
        vsep_b = agg(self.sep)
        out = dict(periods=periods, weights=(None if weights is None else w.tolist()), V_sep=float(vsep_b.mean()))
        has_sh = self.single_model and all(p in self.shared_periods for p in periods)
        if has_sh:
            vsh_b = agg(self.shared)
            out["V_shared"] = float(vsh_b.mean())
            out["C"] = float(np.mean(vsep_b - vsh_b))
        else:
            out["V_shared"] = out["C"] = None
        out["rules"] = {}
        for name, Uh in self.U_half.items():
            u_b = agg(Uh)
            r = dict(U_half=float(u_b.mean()), U_full=aggf(self.U_full[name]),
                     R=float(np.mean(vsep_b - u_b)))
            r["R_full"] = out["V_sep"] - r["U_full"]
            if has_sh:
                r["M"] = float(np.mean(vsh_b - u_b))
                r["identity_residual"] = abs(out["C"] + r["M"] - r["R"])
                if r["identity_residual"] > IDENTITY_TOL:
                    raise AssertionError(f"C + M != R for rule {name}: {r['identity_residual']:.3g}")
            out["rules"][name] = r
        return out


def evaluate(Ps, Ys, seed, period_ids=None, rules=None, shared_periods=None, shared_weights=None,
             splits=SPLITS, split_base=SPLIT_BASE, single_model=True):
    """Per-period scores Ps and 0/1 labels Ys, (n_j, K) each -> Evaluation.

    seed and period_ids (ints in [0, 100)) key the partitions: two models compared on the same
    records need the same of both. rules: {name: (K,) thresholds}. shared_periods and
    shared_weights: where the shared rule is fitted and how periods weigh (default all, uniform).
    splits: partitions per period (200 in the paper). single_model=False for a per-period
    reference (P_j_j on period j): no shared rule is fitted for it."""
    N = len(Ps)
    period_ids = list(range(N)) if period_ids is None else list(period_ids)
    check_partition_key(seed, None, split_base)
    for p in period_ids:
        check_partition_key(seed, p, split_base)
    period_ids = [int(p) for p in period_ids]
    _check_inputs(Ps, Ys, period_ids)
    Ps = [np.asarray(P, np.float64) for P in Ps]
    Ys = [np.asarray(Y).astype(int) for Y in Ys]
    rules = {} if rules is None else {k: np.asarray(v, np.float64) for k, v in rules.items()}
    for k, v in rules.items():
        if v.shape != (Ps[0].shape[1],):
            raise ValueError(f"rule {k}: need one threshold per label")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"rule {k}: non-finite threshold")
    shared_periods = list(period_ids) if shared_periods is None else [int(p) for p in shared_periods]
    if [p for p in shared_periods if p not in period_ids] or len(set(shared_periods)) != len(shared_periods):
        raise ValueError("shared_periods: distinct ids of evaluated periods")
    S = [period_ids.index(p) for p in shared_periods]
    if shared_weights is not None:
        shared_weights = _check_weights(shared_weights, len(S), "shared_weights")
    perms = [partitions(len(Ys[c]), seed, period_ids[c], splits, split_base) for c in range(N)]
    nan = np.full((splits, 2, N), np.nan)
    sep, sh = nan.copy(), nan.copy()
    U = {k: nan.copy() for k in rules}
    for b in range(splits):
        for d in (0, 1):
            fit, ev = [], []
            for c in range(N):
                h = len(Ys[c]) // 2
                A, B = perms[c][b][:h], perms[c][b][h:]
                fit.append(A if d == 0 else B); ev.append(B if d == 0 else A)
            for c in range(N):
                Pe, Ye = Ps[c][ev[c]], Ys[c][ev[c]]
                sep[b, d, c] = macro_f1(Pe, Ye, fit_exact(Ps[c][fit[c]], Ys[c][fit[c]]))
                for k, th in rules.items():
                    U[k][b, d, c] = macro_f1(Pe, Ye, th)
            if not single_model:
                continue                    # no shared rule across different models' scores
            th_sh = fit_shared_exact([Ps[c][fit[c]] for c in S], [Ys[c][fit[c]] for c in S], shared_weights)
            for c in S:
                sh[b, d, c] = macro_f1(Ps[c][ev[c]], Ys[c][ev[c]], th_sh)
    U_full = {k: np.array([macro_f1(Ps[c], Ys[c], th) for c in range(N)]) for k, th in rules.items()}
    # pairing checks for decompose: label counts (readable) and a hash of each label matrix in
    # record order (a permuted or different record set gives a different hash); the fixed rules
    # are kept, with their hashes, so decompose can confirm both models were scored at one rule
    sig = [(int(len(Y)), int(Y.sum())) for Y in Ys]
    label_hash = [array_hash(Y, np.int8) for Y in Ys]
    rule_hash = {k: array_hash(v, np.float64) for k, v in rules.items()}
    return Evaluation(seed=seed, splits=splits, split_base=split_base, period_ids=period_ids,
                      shared_periods=shared_periods, shared_weights=shared_weights, sep=sep, shared=sh,
                      U_half=U, U_full=U_full, label_signature=sig, label_hash=label_hash,
                      rules={k: v.copy() for k, v in rules.items()}, rule_hash=rule_hash,
                      single_model=bool(single_model))


def decompose(ref, final, rule, periods=None, weights=None, halves=False, same_rule=True):
    """Two Evaluations on the same records -> dict with B = A + dR at `rule`, per period and mean.

    B = U_ref - U_final (the deficit), A = V_sep,ref - V_sep,final, dR = R_final - R_ref with
    R_x = V_sep,x - U_x; for two single models also dR = dC + dM. B and dR are also returned
    as D and R (the earlier names). U on the full period unless
    halves=True. Raises unless seed, period ids, splits, labels and the rule match;
    same_rule=False allows a model-specific rule under one name."""
    if ref.seed != final.seed or ref.period_ids != final.period_ids or ref.splits != final.splits \
            or ref.split_base != final.split_base:
        raise ValueError("ref and final must share seed, period ids and splits (same partitions)")
    if ref.label_signature != final.label_signature or ref.label_hash != final.label_hash:
        bad = [p for p, a, b in zip(ref.period_ids, ref.label_hash, final.label_hash) if a != b]
        raise ValueError(f"ref and final were evaluated on different labels or record order (periods {bad})")
    if rule not in ref.rules or rule not in final.rules:
        raise KeyError(f"rule {rule!r} was not evaluated for both models")
    matched = ref.rule_hash[rule] == final.rule_hash[rule]
    if same_rule and not matched:
        raise ValueError(f"rule {rule!r} is a different threshold vector in ref and final; "
                         "pass same_rule=False only for a rule that is model-specific by design")
    if weights is None and ref.shared_weights != final.shared_weights:
        raise ValueError("ref and final were evaluated with different shared_weights "
                         f"({ref.shared_weights} against {final.shared_weights}), so their default period "
                         "weightings differ and D, A and R would mix two weightings; pass `weights` explicitly")
    sr, sf = ref.summary(periods, weights), final.summary(periods, weights)
    key = "U_half" if halves else "U_full"
    ur, uf = sr["rules"][rule][key], sf["rules"][rule][key]
    out = dict(periods=sr["periods"], rule=rule, same_rule=bool(matched), U=key, U_ref=ur, U_final=uf,
               V_ref=sr["V_sep"], V_final=sf["V_sep"],
               D=ur - uf, A=sr["V_sep"] - sf["V_sep"], R_ref=sr["V_sep"] - ur, R_final=sf["V_sep"] - uf)
    out["R"] = out["R_final"] - out["R_ref"]
    out["residual_D_A_R"] = abs(out["D"] - (out["A"] + out["R"]))
    out["B"], out["dR"] = out["D"], out["R"]      # the paper's names; D and R kept for compatibility
    if not (ref.single_model and final.single_model):
        out["shared_terms"] = "omitted: the reference holds different models on different periods"
    elif ref.shared_periods != final.shared_periods or ref.shared_weights != final.shared_weights:
        out["shared_terms"] = "omitted: the two shared rules were fitted on different periods or weights"
    elif sr["C"] is None or sf["C"] is None:
        out["shared_terms"] = "omitted: the periods are not all in the shared set"
    else:
        out["shared_terms"] = "reported"
        out["C_ref"], out["C_final"] = sr["C"], sf["C"]
        out["M_ref"] = sr["V_shared"] - ur
        out["M_final"] = sf["V_shared"] - uf
        out["dC"] = out["C_final"] - out["C_ref"]
        out["dM"] = out["M_final"] - out["M_ref"]
        out["A_shared"] = sr["V_shared"] - sf["V_shared"]
        out["residual_R_C_M"] = abs(out["R"] - (out["dC"] + out["dM"]))
        out["residual_D_Ash_M"] = abs(out["D"] - (out["A_shared"] + out["dM"]))
    for k in [k for k in out if k.startswith("residual")]:
        if out[k] > IDENTITY_TOL:
            raise AssertionError(f"{k} = {out[k]:.3g}")
    return out
