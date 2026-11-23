"""Command line: refitgap {evaluate, report, example} and --version.

    refitgap evaluate --npz run_s0.npz run_s1.npz --seeds 0 1 --out OUT
    refitgap report   (same inputs) --out OUT        # adds figures and report.html
    refitgap example  [--out OUT]"""
import argparse
import sys
from pathlib import Path

from . import __version__
from .analysis import analyse, run_from_arrays, run_from_npz, write_tables
from .inference import MARGIN


def _inputs(p):
    g = p.add_argument_group("inputs (method a, and optionally method b paired by seed)")
    g.add_argument("--npz", nargs="+", default=[], help="stored-run files (*_scores.npz) of method a")
    g.add_argument("--npz-b", nargs="+", default=[], help="stored-run files of method b")
    g.add_argument("--arrays", nargs="+", default=[], help="per-period arrays files (npz) of method a")
    g.add_argument("--arrays-b", nargs="+", default=[], help="per-period arrays files of method b")
    g.add_argument("--seeds", nargs="+", type=int, help="run seed of each file, in file order (stored-run files; "
                                                         "arrays files may carry their own `seed`)")
    g.add_argument("--names", nargs="+", default=["a", "b"], metavar="NAME",
                   help="method names in the outputs: one for a single method, two for a paired comparison")
    s = p.add_argument_group("settings")
    s.add_argument("--periods", nargs="+", type=int, help="period ids to evaluate (stored-run files; default all)")
    s.add_argument("--mean-over", nargs="+", type=int, help="period ids averaged in summaries and contrasts "
                                                             "(default all evaluated)")
    s.add_argument("--shared-periods", nargs="+", type=int, help="period ids of the shared rule (default all)")
    s.add_argument("--shared-weights", nargs="+", type=float, help="weights of the shared periods (default uniform)")
    s.add_argument("--splits", type=int, default=200, help="number of partitions (default 200, as in the paper)")
    s.add_argument("--rule", default="inherited", help="fixed rule of the decomposition (default inherited)")
    s.add_argument("--validation", action="store_true", help="stored-run files: add the rule fitted on the final "
                                                             "model's validation scores as rule 'validation'")
    s.add_argument("--u-convention", choices=["full", "half"], default="full",
                   help="fixed-rule level used in the decomposition and contrasts: full = whole period (default; "
                        "B = A + dR of the paper), half = held-out halves")
    s.add_argument("--no-reference", action="store_true", help="ignore per-period reference scores")
    s.add_argument("--margin", type=float, default=MARGIN,
                   help="margin of the class rules (default 0.005): TOST, the class words and the non-inferiority "
                        "and inferiority tests all use it, and the figures draw it")
    s.add_argument("--family", help="name of a family the caller specified in advance; without it every row is "
                                    "descriptive")
    s.add_argument("--workers", type=int, default=1, help="processes for the evaluations (default 1)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--quiet", action="store_true", help="print only the output paths")


def load_runs(args):
    if (args.npz or args.npz_b) and (args.arrays or args.arrays_b):
        raise ValueError("use stored-run files or arrays files, not both")
    ref = not args.no_reference
    if args.npz:
        if not args.seeds or len(args.seeds) != len(args.npz):
            raise ValueError("--seeds needs one seed per stored-run file of method a (the same seeds pair method b)")
        if args.npz_b and len(args.npz_b) != len(args.npz):
            raise ValueError("--npz-b needs one file per seed, in the order of --seeds")
        a = [run_from_npz(f, s, args.periods, args.validation, ref) for f, s in zip(args.npz, args.seeds)]
        b = [run_from_npz(f, s, args.periods, args.validation, ref) for f, s in zip(args.npz_b, args.seeds)]
        return a, b
    if args.arrays:
        if args.seeds and len(args.seeds) != len(args.arrays):
            raise ValueError("--seeds needs one seed per arrays file of method a")
        sa = args.seeds or [None] * len(args.arrays)
        a = [run_from_arrays(f, s, ref) for f, s in zip(args.arrays, sa)]
        if args.seeds and args.arrays_b and len(args.arrays_b) != len(args.seeds):
            raise ValueError("--arrays-b needs one file per seed, in the order of --seeds")
        sb = args.seeds or [None] * len(args.arrays_b)
        b = [run_from_arrays(f, s, ref) for f, s in zip(args.arrays_b, sb)]
        return a, b
    raise ValueError("give --npz or --arrays")


def run_analysis(args):
    a, b = load_runs(args)
    return analyse(a, b, rule=args.rule, mean_over=args.mean_over, splits=args.splits,
                   shared_periods=args.shared_periods, shared_weights=args.shared_weights,
                   u_convention=args.u_convention, margin=args.margin, family=args.family, names=_names(args),
                   workers=args.workers)


def summary_lines(result):
    conv = result["settings"]["U_convention"]
    out = []
    means = [r for r in result["decomposition"] if str(r["period"]).startswith("mean")]
    if means:
        out.append(f"decomposition against the per-period reference, rule '{result['settings']['rule']}', "
                   f"U_{conv}:")
        for r in means:
            out.append(f"  {r['method']} seed {r['seed']} {r['period']}: B {r['B']:+.4f} = A {r['A']:+.4f} "
                       f"+ dR {r['dR']:+.4f}")
    if result["contrasts"]:
        out.append("paired contrasts:")
        for r in result["contrasts"]:
            ci = (f" [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] (marginal 95% t)" if r["ci95_low"] is not None
                  else " (one run, no interval)")
            cls = f"  class {r['class']}" if r["class"] else ""
            out.append(f"  {r['comparison']}, {r['term']}: {r['mean']:+.4f}{ci}  n {r['n']}  status {r['status']}{cls}")
    out.extend(result["interpretation"])
    return out


def _write(result, args, report):
    out = Path(args.out)
    paths = write_tables(result, out)
    if report:
        from .report import write_report
        try:
            from .plots import save_figures, svg_strings
            figs = save_figures(result, out)
            svgs = svg_strings(result)
        except ImportError as e:
            print(f"note: {e}; the report is written without figures", file=sys.stderr)
            figs, svgs = {}, {}
        paths["report"] = write_report(result, out / "report.html", svgs)
        for name, d in figs.items():
            for fmt, p in d.items():
                paths[f"fig_{name}.{fmt}"] = p
    return paths


def _names(args):
    """One or two method names; a single name is padded so the paired path never sees it."""
    n = list(args.names)
    if len(n) > 2:
        raise SystemExit("refitgap: --names takes one or two names")
    if len(n) == 1:
        n.append("b")
    if len(n) == 2 and (args.npz_b or args.arrays_b) and n[0] == n[1]:
        raise SystemExit("refitgap: the two method names must differ")
    return tuple(n)


def main(argv=None):
    p = argparse.ArgumentParser(prog="refitgap",
                                description="Evaluation of thresholded forgetting from stored per-period scores.")
    p.add_argument("--version", action="version", version=f"refitgap {__version__}")
    sub = p.add_subparsers(dest="cmd")
    pe = sub.add_parser("evaluate", help="levels, decomposition and contrasts as CSV and JSON")
    _inputs(pe)
    pr = sub.add_parser("report", help="evaluate, plus three figures (PDF, SVG) and a self-contained HTML report")
    _inputs(pr)
    px = sub.add_parser("example", help="the tiny example, worked by hand in API.txt")
    px.add_argument("--out", help="also write its tables, figures and report here")
    args = p.parse_args(argv)
    if args.cmd is None:
        p.print_help()
        return 2
    if args.cmd == "example":
        from .tiny import lines, tiny_result
        result = tiny_result()
        for line in lines(result):
            print(line)
        if args.out:
            args.quiet = True
            for k, v in _write(result, args, report=True).items():
                print(f"wrote {v}")
        return 0
    try:
        result = run_analysis(args)
    except (ValueError, KeyError) as e:
        print(f"refitgap: error: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        for line in summary_lines(result):
            print(line)
    for k, v in _write(result, args, report=args.cmd == "report").items():
        print(f"wrote {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
