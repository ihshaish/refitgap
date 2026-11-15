"""Synthetic two-learner data for the tests: four periods, four labels, six seeds per method.

make_seed(s) builds one seed's labels and scores; main() prints a decomposition on them. The
forgetful learner loses separation on early periods and its mismatch grows; the
other loses less. The built-in example to run is `refitgap example` (src/refitgap/tiny.py)."""
import numpy as np

from . import crossfit, inference

N_PERIODS, K, N_REC = 4, 4, 400
SEEDS = range(6)
SPLITS = 50          # the paper uses 200; 50 keeps the example quick


def synth_period(rng, n, prev):
    return (rng.random((n, K)) < prev).astype(int)


def scores(rng, Y, sep, offset):
    z = sep * Y + rng.normal(0, 1, Y.shape) + offset
    return 1 / (1 + np.exp(-z))


def make_seed(seed):
    """Labels per period, the reference model of each period (s_j on period j), the final
    model of two learners, and a validation split from period 0 that fits theta_1."""
    rng = np.random.default_rng(seed)
    prev = np.array([0.10, 0.20, 0.30, 0.15])
    Ys = [synth_period(rng, N_REC, prev) for _ in range(N_PERIODS)]
    Yval = synth_period(rng, 300, prev)
    ref = [scores(rng, Ys[j], 2.5, -1.5) for j in range(N_PERIODS)]          # each period's own model
    Pval_first = scores(rng, Yval, 2.5, -1.5)
    # final models: separation lost on early periods, period-specific offset
    sequential = [scores(rng, Ys[j], 2.5 - 0.30 * (N_PERIODS - 1 - j), -1.5 + 0.5 * (N_PERIODS - 1 - j))
             for j in range(N_PERIODS)]
    replay = [scores(rng, Ys[j], 2.5 - 0.10 * (N_PERIODS - 1 - j), -1.5 + 0.15 * (N_PERIODS - 1 - j))
              for j in range(N_PERIODS)]
    return Ys, Yval, Pval_first, ref, sequential, replay


def main():
    hist = [0, 1, 2]                      # historical periods; period 3 is the current one
    per_seed = {"sequential": {}, "replay": {}}
    alone = {"sequential": {}, "replay": {}}
    method = {}
    for s in SEEDS:
        Ys, Yval, Pval_first, ref, sequential, replay = make_seed(s)
        theta_1 = crossfit.fit_validation_rule(Pval_first, Yval)              # the inherited thresholds
        rules = {"inherited": theta_1}
        # the reference holds a different model per period: no shared-rule terms against it
        ev_ref = crossfit.evaluate(ref, Ys, s, rules=rules, splits=SPLITS, single_model=False)
        ev = {}
        for name, Ps in (("sequential", sequential), ("replay", replay)):
            ev[name] = crossfit.evaluate(Ps, Ys, s, rules=rules, splits=SPLITS)
            per_seed[name][s] = crossfit.decompose(ev_ref, ev[name], "inherited", periods=hist)
            alone[name][s] = ev[name].summary(periods=hist)
        # method contrast: both are single final models, so R splits into dC + dM
        method[s] = crossfit.decompose(ev["replay"], ev["sequential"], "inherited", periods=hist)
        if s == 0:
            d = per_seed["sequential"][0]
            print("seed 0, sequential learner, historical periods 0-2, inherited thresholds theta_1:")
            print(f"  against the per-period reference models: B = {d['B']:+.4f} = A {d['A']:+.4f} + dR {d['dR']:+.4f}"
                  f"   (residual {d['residual_D_A_R']:.1e})")
            sm = alone["sequential"][0]
            r = sm["rules"]["inherited"]
            print(f"  final model alone: V_sep {sm['V_sep']:.4f}  V_shared {sm['V_shared']:.4f}  U(theta_1) "
                  f"{r['U_half']:.4f}, so R {r['R']:+.4f} = C {sm['C']:+.4f} + M {r['M']:+.4f}")
            m = method[0]
            print(
                f"  against the replay learner's final model: B = {m['B']:+.4f} = A {m['A']:+.4f} + dR {m['dR']:+.4f};"
                f"  dR = dC {m['dC']:+.4f} + dM {m['dM']:+.4f}"
            )
    print("\nper-seed decomposition, sequential learner (historical mean; B, A, dR against the reference models,"
          " C and M of the final model alone):")
    for s in SEEDS:
        d, sm = per_seed["sequential"][s], alone["sequential"][s]
        print(f"  seed {s}: B {d['B']:+.4f}  A {d['A']:+.4f}  dR {d['dR']:+.4f}  C {sm['C']:+.4f}  "
              f"M {sm['rules']['inherited']['M']:+.4f}")
    # paired contrasts, sequential minus replay, one family of three read together
    members = {}
    for term in ("B", "A", "dR"):
        _, members[term] = inference.paired_vector({s: per_seed["sequential"][s][term] for s in SEEDS},
                                                   {s: per_seed["replay"][s][term] for s in SEEDS},
                                                   seeds=SEEDS)
    fam = inference.family(members)
    print("\nsequential minus replay, paired over 6 seeds (Holm over B, A, dR; margin 0.005):")
    for term, st in fam.items():
        print(f"  {term}: mean {st['mean']:+.4f}  95% CI (marginal) [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]"
              f"  Holm p_t {st['p_holm']:.3g}  class {st['class']}")
    c = inference.paired_contrast({s: alone["replay"][s]["V_shared"] for s in SEEDS},
                                  {s: alone["sequential"][s]["V_shared"] for s in SEEDS}, seeds=SEEDS)
    print(f"\nshared-rule level, replay minus sequential: {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
          f"  class {c['class']} (unadjusted)")
    c = inference.paired_contrast({s: method[s]["dC"] for s in SEEDS}, {s: 0.0 for s in SEEDS}, seeds=SEEDS)
    print(f"dC, sequential final against replay final (cost of one rule for periods 0-2): {c['mean']:+.4f} "
          f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  class {c['class']} (unadjusted)")
