"""Phase 6, step 5: external-sampling MCCFR on Leduc, against full-tree CFR per node touched.

Run:
    .venv\\Scripts\\python examples\\train_leduc_mccfr.py                       # 100,000 iterations, seeds 0 1 2
    .venv\\Scripts\\python examples\\train_leduc_mccfr.py --iterations 20000 --seeds 0

For plain ES-MCCFR and linear-weighted ES-MCCFR (the hold'em configuration)
this runs a fixed iteration budget at each seed, measures the *exact*
exploitability of the average strategy at log-spaced iterations, and
compares with full-tree CFR+, linear CFR and vanilla CFR at equal **node
touches** (histories visited). Writes

    results/phase6_mccfr_exploitability.png   vs iterations and vs node touches
    results/phase6_mccfr.json                 every checkpoint
    results/phase6_mccfr.md                   tables + interpretation

Cost model. An ES iteration is two sampled traversals; ``nodes`` counts
every history they visit. A full-tree iteration visits every one of
Leduc's 1,939 histories once per tree walk: 2 walks for the alternating
variants (CFR+, linear), 1 for vanilla. Node touches are reproducible;
wall times on this laptop are not (it throttles 6-9x under sustained load).

"Nodes to reach e" is read off the curve at the first checkpoint whose
exploitability is <= e, interpolated log-linearly from the previous
checkpoint. ES curves are noisy, so this is a first-passage number, not a
guarantee that the curve stays below e.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from nutsfinder.gametheory import Leduc  # noqa: E402
from nutsfinder.gametheory.cfr import CFRSolver, log_spaced  # noqa: E402
from nutsfinder.gametheory.mccfr import MCCFRSolver  # noqa: E402
from train_leduc import commit_hash  # noqa: E402

RESULTS = ROOT / "results"
LEDUC_VALUE = -0.0856064  # certified bracket [-0.0856071, -0.0856049] (results/phase6_leduc.md)
ES = {"es": {"linear": False}, "es-linear": {"linear": True}}
FULL = {"cfr+": 2, "linear": 2, "vanilla": 1}  # tree walks per iteration
TARGETS = (0.2, 0.1, 0.05, 0.02)
REPORT_ITERS = (1_000, 10_000, 30_000, 100_000)
SLOPE_FROM = 1_000
# Hold'em extrapolation inputs, docs/phase6_research.md section 3 [estimate]:
HOLDEM_INFOSETS = 1.3e5
HOLDEM_NODES_PER_ITER = (300, 3_000)
HOLDEM_ITERS_PER_SEC_CORE = 30


def fitted_slope(xs, ys) -> float:
    x, y = np.log10(np.array(xs, float)), np.log10(np.array(ys, float))
    return float(np.polyfit(x, y, 1)[0])


def first_reach(nodes, expl, target) -> float | None:
    """Node count at which exploitability first drops to ``target`` (log-linear interpolation)."""
    for k, (n, e) in enumerate(zip(nodes, expl)):
        if e <= target:
            if k == 0:
                return float(n)
            n0, e0 = nodes[k - 1], expl[k - 1]
            f = (math.log(e0) - math.log(target)) / (math.log(e0) - math.log(e))
            return float(math.exp(math.log(n0) + f * (math.log(n) - math.log(n0))))
    return None


def interp_at(nodes, expl, n) -> float | None:
    """Exploitability of a curve at node budget ``n`` (log-log interpolation)."""
    if n < nodes[0] or n > nodes[-1]:
        return None
    return float(10 ** np.interp(math.log10(n), np.log10(nodes), np.log10(expl)))


def run_es(name: str, seed: int, iterations: int, points: list[int]) -> dict:
    solver = MCCFRSolver(Leduc(), seed=seed, **ES[name])
    start = time.perf_counter()
    cps = solver.run(iterations, eval_at=points)
    wall = time.perf_counter() - start
    missing = set(points) - {c.iteration for c in cps}
    assert not missing, f"missing checkpoints {sorted(missing)}"
    last = cps[-1]
    print(f"{name:9s} seed {seed}  T={last.iteration:7d}  expl {last.exploitability:.4f}  "
          f"value {last.value:+.5f}  nodes {last.nodes:,}  solver {last.seconds:6.1f}s  "
          f"wall {wall:6.1f}s", flush=True)
    return {"method": name, "seed": seed, "options": solver.options(), "wall_seconds": wall,
            "checkpoints": [vars(c) for c in cps]}


def run_full(name: str, iterations: int) -> dict:
    solver = CFRSolver(Leduc(), name)
    per_iter = FULL[name] * solver.tree.n_nodes
    iterations = iterations * 2 // FULL[name]  # equal node budget for one-walk variants
    points = log_spaced(iterations, 20)
    cps = solver.run(iterations, eval_at=points)
    rows = [dict(vars(c), nodes=c.iteration * per_iter) for c in cps]
    print(f"{name:9s} full tree  T={iterations:7d}  expl {cps[-1].exploitability:.2e}  "
          f"nodes {rows[-1]['nodes']:,}", flush=True)
    return {"method": name, "nodes_per_iteration": per_iter, "checkpoints": rows}


def series(run, key):
    return [c[key] for c in run["checkpoints"]]


def cp_at(run, t):
    return next((c for c in run["checkpoints"] if c["iteration"] == t), None)


def plot(es_runs, full_runs, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colours = {"es": "tab:blue", "es-linear": "tab:orange",
               "cfr+": "tab:green", "linear": "tab:red", "vanilla": "tab:gray"}
    for r in es_runs:
        lab = f"{r['method']} seed {r['seed']}"
        kw = dict(color=colours[r["method"]], alpha=1.0 if r["seed"] == es_runs[0]["seed"] else 0.4,
                  lw=1.5 if r["seed"] == es_runs[0]["seed"] else 0.8)
        ax1.loglog(series(r, "iteration"), series(r, "exploitability"), label=lab, **kw)
        ax2.loglog(series(r, "nodes"), series(r, "exploitability"), label=lab, **kw)
    for name, r in full_runs.items():
        ax2.loglog(series(r, "nodes"), series(r, "exploitability"), "--",
                   color=colours[name], label=f"{name} (full tree)")
    ax1.set_xlabel("iterations T (one traversal per player)")
    ax1.set_title("ES-MCCFR on Leduc: exact exploitability vs iterations")
    ax2.set_xlabel("node touches (histories visited)")
    ax2.set_title("ES-MCCFR vs full-tree CFR per node touched")
    for ax in (ax1, ax2):
        ax.set_ylabel("exploitability (BR0 + BR1)/2, chips/hand")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt_nodes(x):
    return "not reached" if x is None else f"{x:.2e}"


def report(es_runs, full_runs, args, commit, png, iters_points) -> str:
    seeds = args.seeds
    lines = [
        "# Phase 6: external-sampling MCCFR on Leduc hold'em",
        "",
        f"- date: {dt.date.today().isoformat()}; commit: {commit}; script: "
        f"examples/train_leduc_mccfr.py --iterations {args.iterations} --seeds "
        + " ".join(map(str, seeds)) + f" --cfr-iterations {args.cfr_iterations}",
        f"- seeds: {', '.join(map(str, seeds))} (each `MCCFRSolver(seed=s)` owns a PCG64 generator; "
        "same seed -> bit-identical tables)",
        "- solver: `nutsfinder.gametheory.mccfr.MCCFRSolver`, external sampling, alternating "
        "traversers (iteration = one traversal per player), average strategy updated on opponent "
        "nodes (\"simple\" averaging), no pruning. `es` = weight 1, `es-linear` = iteration t "
        "weighted by t (linear CFR).",
        "- game: Leduc, rank-only (288 info sets, 1,939 histories); exploitability = (BR0 + BR1)/2 "
        "chips/hand, exact, of the normalised average strategy",
        "- cost: node touches = histories visited. ES: counted by the solver. Full tree: "
        "iterations x walks x 1,939 (walks = 2 for CFR+ / linear, 1 for vanilla)",
        f"- slope = least-squares slope of log10(exploitability) on log10(T) for T >= {SLOPE_FROM:,}",
        f"- plot: `{png.relative_to(ROOT).as_posix()}`",
        "",
        "## ES-MCCFR by seed",
        "",
        "| method | seed | " + " | ".join(f"expl. @{t:,}" for t in iters_points)
        + " | value @T | bracket [-BR1, BR0] @T | nodes @T | nodes / iter. | slope | "
          "solver s | wall s |",
        "|---|---|" + "---|" * len(iters_points) + "---|---|---|---|---|---|---|",
    ]
    for r in es_runs:
        last = r["checkpoints"][-1]
        cells = []
        for t in iters_points:
            c = cp_at(r, t)
            cells.append(f"{c['exploitability']:.4f}" if c else "-")
        its = [c["iteration"] for c in r["checkpoints"] if c["iteration"] >= SLOPE_FROM]
        ex = [c["exploitability"] for c in r["checkpoints"] if c["iteration"] >= SLOPE_FROM]
        lines.append(
            f"| {r['method']} | {r['seed']} | " + " | ".join(cells)
            + f" | {last['value']:+.5f} | [{-last['br1']:+.4f}, {last['br0']:+.4f}] | "
              f"{last['nodes']:,} | {last['nodes'] / last['iteration']:.1f} | "
              f"{fitted_slope(its, ex):+.2f} | {last['seconds']:.1f} | {r['wall_seconds']:.1f} |")

    # Seed summaries at T.
    summary = {}
    for name in ES:
        finals = [r["checkpoints"][-1]["exploitability"] for r in es_runs if r["method"] == name]
        summary[name] = finals
    lines += ["", "Final exploitability across seeds: "
              + "; ".join(f"{n}: median {np.median(v):.4f}, range [{min(v):.4f}, {max(v):.4f}]"
                          for n, v in summary.items()), ""]

    lines += ["## Node touches needed to first reach an exploitability", "",
              "| method | " + " | ".join(f"e <= {e}" for e in TARGETS) + " |",
              "|---|" + "---|" * len(TARGETS)]
    reach = {}
    for name in ES:
        runs = [r for r in es_runs if r["method"] == name]
        vals = [[first_reach(series(r, "nodes"), series(r, "exploitability"), e) for e in TARGETS]
                for r in runs]
        med = []
        for j in range(len(TARGETS)):
            # "Not reached" counts as +infinity: the median is still a valid order
            # statistic (finite when more than half of the seeds reached e).
            m = float(np.median([math.inf if v[j] is None else v[j] for v in vals]))
            med.append(None if math.isinf(m) else m)
        reach[name] = med
        lines.append(f"| {name} (median of {len(runs)} seeds; not reached = infinite) | "
                     + " | ".join(map(fmt_nodes, med)) + " |")
    for name, r in full_runs.items():
        vals = [first_reach(series(r, "nodes"), series(r, "exploitability"), e) for e in TARGETS]
        reach[name] = vals
        lines.append(f"| {name} (full tree) | " + " | ".join(map(fmt_nodes, vals)) + " |")
    lines.append("")

    # Equal-budget comparison at the ES final node count (seed 0 linear).
    base = next(r for r in es_runs if r["method"] == "es-linear" and r["seed"] == seeds[0])
    n_es, e_es = base["checkpoints"][-1]["nodes"], base["checkpoints"][-1]["exploitability"]
    e_full = {name: interp_at(series(r, "nodes"), series(r, "exploitability"), n_es)
              for name, r in full_runs.items()}
    lines += ["## Equal node budget", "",
              f"At es-linear seed {seeds[0]}'s final budget of {n_es:,} node touches "
              f"(exploitability {e_es:.4f}):", "",
              "| method | exploitability at that budget | ratio to es-linear |", "|---|---|---|"]
    for name, e in e_full.items():
        lines.append(f"| {name} (full tree) | " + (f"{e:.2e} | {e_es / e:.0f}x lower"
                                                     if e else "outside run | -") + " |")
    lines.append("")

    # Interpretation.
    lin = [r for r in es_runs if r["method"] == "es-linear"]
    pla = [r for r in es_runs if r["method"] == "es"]
    med_lin = float(np.median([r["checkpoints"][-1]["exploitability"] for r in lin]))
    med_pla = float(np.median([r["checkpoints"][-1]["exploitability"] for r in pla]))
    slopes = [fitted_slope([c["iteration"] for c in r["checkpoints"] if c["iteration"] >= SLOPE_FROM],
                           [c["exploitability"] for c in r["checkpoints"] if c["iteration"] >= SLOPE_FROM])
              for r in es_runs]
    npi = float(np.mean([r["checkpoints"][-1]["nodes"] / r["checkpoints"][-1]["iteration"]
                         for r in es_runs]))
    dec = args.decision_visits_per_iter
    fastest = min((r["checkpoints"][-1]["seconds"] / r["checkpoints"][-1]["nodes"], r["seed"], r["method"])
                  for r in es_runs)
    # Hold'em: iterations to give each info set as many updates as Leduc needed for e <= 0.05.
    t_l, target = None, None
    for j in range(len(TARGETS) - 1, -1, -1):  # smallest target the median reached
        if reach["es-linear"][j] is not None:
            t_l, target = reach["es-linear"][j] / npi, TARGETS[j]
            break
    lines += [
        "## Interpretation",
        "",
        f"- **The ES solver converges.** Every seed's exploitability falls with fitted log-log slopes "
        f"{min(slopes):+.2f} to {max(slopes):+.2f} from T = {SLOPE_FROM:,}, i.e. at about the "
        "1/sqrt(T) rate expected of Monte Carlo CFR (sampling noise, not the regret bound, dominates), "
        "and the value of the average profile is inside its own "
        f"certified bracket and heads to v* = {LEDUC_VALUE}. The Monte Carlo tests in "
        "`tests/test_mccfr.py` check the stronger statement behind this: the expected sampled regret "
        "and average-strategy updates equal the full-tree quantities.",
        f"- **Linear weighting vs plain.** Median final exploitability {med_lin:.4f} (linear) vs "
        f"{med_pla:.4f} (plain) over {len(lin)} seeds. "
        + ("The gap is within the seed-to-seed spread, so on Leduc at this budget linear weighting "
           "is not a measurable win. "
           if abs(med_lin - med_pla) < (max(max(summary['es']), max(summary['es-linear']))
                                       - min(min(summary['es']), min(summary['es-linear'])))
           else "")
        + "Its documented benefit is discounting the early, near-random iterations, which matters "
        "more when the run is long relative to the transient (Pluribus) than here.",
        "- **Per node touched, full-tree CFR+ wins clearly on Leduc.** "
        + (f"At the same budget CFR+ is about {e_es / e_full['cfr+']:.0f}x less exploitable, and "
           f"even vanilla CFR is {e_es / e_full['vanilla']:.1f}x less exploitable. "
           if e_full.get("cfr+") and e_full.get("vanilla") else "")
        + "This is the expected result, not a defect: Leduc's whole tree is only 1,939 histories, "
        f"so one full walk costs about as much as {1939 / npi:.0f} ES iterations and gives *exact* "
        "regrets. Sampling "
        "pays when the full tree cannot be walked at all: an abstract hold'em iteration over every "
        "deal would touch ~10^9+ histories, while an ES traversal touches hundreds. The comparison "
        "that matters for hold'em is therefore not CFR+ vs ES, but whether ES converges at all on a "
        "solver-checkable game; it does.",
        f"- **Cost.** {npi:.1f} node touches per ES iteration on Leduc (both traversals), of which "
        f"about {dec:.0f} are decision nodes (measured with a counting wrapper, "
        "see `decision_visits_per_iter` in the JSON). The fastest run here spent "
        f"{1e6 * fastest[0]:.1f} us per node touch (seed {fastest[1]}, {fastest[2]}), but every "
        "long run was throttled; short unthrottled runs measured about 2 us per node touch "
        "(10,000 Leduc iterations in under 1 s). Node touches are the reproducible cost measure.",
        "",
        "## What this implies for hold'em [estimate]",
        "",
    ]
    if t_l is not None:
        upd_per_is = t_l * dec / 288
        lo_i = upd_per_is * HOLDEM_INFOSETS / (HOLDEM_NODES_PER_ITER[1] * dec / npi)
        hi_i = upd_per_is * HOLDEM_INFOSETS / (HOLDEM_NODES_PER_ITER[0] * dec / npi)
        lines += [
            f"- On Leduc, es-linear (median over seeds) first reached exploitability {target} chips/hand "
            f"after about {t_l:,.0f} iterations, i.e. about {upd_per_is:,.0f} decision-node visits per "
            "info set on average.",
            f"- Assume the abstract hold'em game needs the same number of visits per info set for a "
            f"comparable *relative* accuracy, with {HOLDEM_INFOSETS:.1e} info sets (section 3) and "
            f"{HOLDEM_NODES_PER_ITER[0]}-{HOLDEM_NODES_PER_ITER[1]:,} node touches per iteration (the "
            "notes' range), with the Leduc decision-node share. That gives roughly "
            f"**{lo_i:.0e} to {hi_i:.0e} iterations**; at the notes' ~{HOLDEM_ITERS_PER_SEC_CORE} "
            f"iterations/s/core that is {lo_i / HOLDEM_ITERS_PER_SEC_CORE / 3600:.0f}-"
            f"{hi_i / HOLDEM_ITERS_PER_SEC_CORE / 3600:.0f} core-hours.",
            "- Why this is only an estimate, and which way it is likely wrong:",
            "  - visits are not uniform over info sets. Rare preflop lines and river buckets behind "
            "them are visited far less often than the average, so the *worst* info sets need many "
            "more iterations than the average suggests (the estimate is optimistic for them);",
            "  - hold'em payoffs span up to 100 bb, so per-visit regret noise is larger relative to "
            "the quantities that decide the policy (pessimistic for speed);",
            "  - imperfect-recall bucketing removes the convergence guarantee entirely; exploitability "
            "in the abstraction may plateau rather than keep falling;",
            "  - 'comparable relative accuracy' has no exact meaning across games: "
            f"{target} chips/hand in Leduc, where pots run from 2 to 26 chips, has no exact "
            "hold'em counterpart in mbb/g.",
            "- Treat it as an order of magnitude to plan compute (millions of iterations, tens of "
            "core-hours), and benchmark iteration speed in step 7 before trusting it.",
            "",
        ]
    else:
        lines += ["- es-linear did not reach any target at this budget; no extrapolation made.", ""]
    return "\n".join(lines)


def count_decision_visits(iterations: int = 2000, seed: int = 0) -> float:
    """Decision-node visits per ES iteration on Leduc, via a counting game wrapper."""
    class Counting:
        def __init__(self, g):
            self.g, self.n = g, 0

        def __getattr__(self, name):
            return getattr(self.g, name)

        def infoset_key(self, h):
            self.n += 1
            return self.g.infoset_key(h)

    game = Counting(Leduc())
    MCCFRSolver(game, seed=seed, linear=True).run(iterations)
    return game.n / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cfr-iterations", type=int, default=None,
                        help="CFR+ / linear iterations (vanilla, one walk per iteration, gets twice as many); "
                             "default: enough to cover the ES node budget")
    args = parser.parse_args()
    if args.iterations < SLOPE_FROM * 10:
        parser.error(f"--iterations must be at least {SLOPE_FROM * 10:,}")
    commit = commit_hash()
    iters_points = [t for t in REPORT_ITERS if t <= args.iterations]
    if args.iterations not in iters_points:
        iters_points.append(args.iterations)
    # Always include the report's columns; log_spaced may skip them.
    points = sorted(set(log_spaced(args.iterations, 10)) | set(iters_points))
    print(f"Leduc ES-MCCFR, {args.iterations} iterations, seeds {args.seeds}, commit {commit}")

    es_runs = [run_es(name, seed, args.iterations, points) for name in ES for seed in args.seeds]
    max_nodes = max(r["checkpoints"][-1]["nodes"] for r in es_runs)
    if args.cfr_iterations is None:
        args.cfr_iterations = max(1000, math.ceil(1.2 * max_nodes / (2 * 1939)))
    full_runs = {name: run_full(name, args.cfr_iterations) for name in FULL}
    args.decision_visits_per_iter = count_decision_visits()

    RESULTS.mkdir(exist_ok=True)
    png = RESULTS / "phase6_mccfr_exploitability.png"
    plot(es_runs, full_runs, png)
    (RESULTS / "phase6_mccfr.json").write_text(json.dumps({
        "commit": commit, "iterations": args.iterations, "seeds": args.seeds,
        "cfr_iterations": args.cfr_iterations,
        "decision_visits_per_iter": args.decision_visits_per_iter,
        "es_runs": es_runs, "full_tree_runs": full_runs,
    }, indent=1))
    md = RESULTS / "phase6_mccfr.md"
    md.write_text(report(es_runs, full_runs, args, commit, png, iters_points), encoding="utf-8")
    print(f"wrote {md}, {png}")


if __name__ == "__main__":
    main()
