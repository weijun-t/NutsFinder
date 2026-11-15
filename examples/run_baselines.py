"""Play every pairing of the three baseline bots and report the results.

Run:
    .venv\\Scripts\\python examples\\run_baselines.py

The design is fixed *before* anything is played, and nothing here depends
on intermediate results (no optional stopping):

- pairings: every pair from {RandomBot, CallStationBot, AlwaysRaiseBot},
  including each bot against itself (6 pairings);
- each pairing: a duplicate match of N_PAIRS = 10,000 pairs (20,000 hands),
  plus a plain match of the same 20,000 hands' worth for comparing SEs;
- one fixed MASTER_SEED for everything. All pairings therefore see the same
  deals (hand i uses the same deck in every pairing), which makes the
  pairings directly comparable ("common random numbers").

Outputs:
- results/baselines.md: date, seed, n, commit, the results table, the
  plain-vs-duplicate SE table and per-pairing breakdowns;
- results/<date>_<A>_vs_<B>.json (duplicate) and ..._plain.json: all the
  numbers including per-unit results;
- results/<date>_<A>_vs_<B>.png: cumulative bb with the zero-edge 95% band,
  and the histogram of per-pair results.
"""

from __future__ import annotations

import datetime as dt
import math
import subprocess
import time
from itertools import combinations_with_replacement
from pathlib import Path

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, RandomBot
from nutsfinder.eval.breakdown import by_end_street, by_position, leaks
from nutsfinder.eval.match import MatchResult, duplicate_match, plain_match
from nutsfinder.eval.report import save_json, save_plot

MASTER_SEED = 20261115
N_PAIRS = 10_000  # fixed in advance: 20,000 hands per pairing
BOTS = [RandomBot, CallStationBot, AlwaysRaiseBot]
SHORT = {"RandomBot": "Random", "CallStationBot": "CallStation", "AlwaysRaiseBot": "AlwaysRaise"}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def commit_hash() -> str:
    """Current commit, with "+uncommitted changes" if the tree is dirty."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + (" + uncommitted changes" if dirty else "")


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a (no variation)"
    return "< 1e-300" if p == 0 else f"{p:.3g}"


def main() -> None:
    today = dt.date.today().isoformat()
    commit = commit_hash()
    print(f"master seed {MASTER_SEED}, {N_PAIRS:,} duplicate pairs per pairing, commit {commit}")

    rows: list[tuple[MatchResult, MatchResult]] = []
    start = time.perf_counter()
    for make_a, make_b in combinations_with_replacement(BOTS, 2):
        dup = duplicate_match(make_a, make_b, N_PAIRS, MASTER_SEED)
        plain = plain_match(make_a, make_b, 2 * N_PAIRS, MASTER_SEED)
        rows.append((dup, plain))
        print()
        print(dup.summary())
        print(plain.summary())
        stem = f"{today}_{SHORT[dup.bot_a]}_vs_{SHORT[dup.bot_b]}"
        meta = {"date": today, "commit": commit}
        save_json(dup, RESULTS / f"{stem}.json", meta)
        save_json(plain, RESULTS / f"{stem}_plain.json", meta)
        save_plot(dup, RESULTS / f"{stem}.png")
    elapsed = time.perf_counter() - start

    lines = [
        "# Baseline results",
        "",
        f"- date: {today}",
        f"- master seed: {MASTER_SEED} (same deals for every pairing)",
        f"- n: {N_PAIRS:,} duplicate pairs = {2 * N_PAIRS:,} hands per pairing, fixed in advance",
        "- duplicate: each deal played twice, bots swapped between seats (cards and button swap with them)",
        f"- commit: {commit}",
        f"- run time: {elapsed:.0f} s",
        "",
        "Win rate is for bot A (first named), in bb/100 hands. 95% CI = mean +/- 1.96 SE",
        "(normal approximation, per-pair results treated as i.i.d.). p = two-sided z-test vs 0.",
        "",
        "## Duplicate matches",
        "",
        "| pairing (A vs B) | bb/100 | 95% CI (+/-) | SE | SD per pair (bb) | p-value | significant |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dup, _ in rows:
        lo, hi = dup.ci_bb100
        lines.append(
            f"| {SHORT[dup.bot_a]} vs {SHORT[dup.bot_b]} | {dup.mean_bb100:+.1f} | "
            f"{(hi - lo) / 2:.1f} | {dup.se_bb100:.1f} | {dup.sd_per_unit:.1f} | "
            f"{fmt_p(dup.p_value)} | {'exact 0 (no variation)' if dup.stats.no_variation else 'yes' if dup.stats.significant else 'no'} |"
        )
    lines += [
        "",
        "## Variance reduction: plain vs duplicate (same number of hands)",
        "",
        "Plain: 20,000 independent hands, button alternating. "
        "Hands-equivalent = (SE plain / SE duplicate)^2: how many times more plain hands",
        "would be needed for the same precision.",
        "",
        "| pairing | plain bb/100 | plain SE | duplicate SE | SE ratio | hands-equivalent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dup, plain in rows:
        if dup.se_bb100 > 0:
            ratio = plain.se_bb100 / dup.se_bb100
            r_txt, h_txt = f"{ratio:.2f}", f"{ratio**2:.1f}x"
        else:
            r_txt, h_txt = "inf (duplicate SE = 0)", "inf"
        lines.append(
            f"| {SHORT[dup.bot_a]} vs {SHORT[dup.bot_b]} | {plain.mean_bb100:+.1f} | "
            f"{plain.se_bb100:.1f} | {dup.se_bb100:.1f} | {r_txt} | {h_txt} |"
        )
    lines += ["", "## Verdicts and breakdowns (duplicate matches)", ""]
    for dup, _ in rows:
        lines.append(f"### {SHORT[dup.bot_a]} vs {SHORT[dup.bot_b]}")
        lines.append("")
        lines.append(f"- verdict: {dup.verdict()}")
        pos = by_position(dup)
        for label, row in pos.items():
            lines.append(
                f"- A on {label}: {row['bb100']:+.1f} bb/100 "
                f"(SE {row['se_bb100']:.1f}, per hand, no duplicate cancellation)"
            )
        streets = by_end_street(dup)
        parts = [
            f"{s} {v['share']:.0%} ({v['bb100_contribution']:+.1f})" for s, v in streets.items() if v["hands"]
        ]
        lines.append("- hands ended on (share, A's bb/100 from them): " + ", ".join(parts))
        for note in leaks(dup):
            lines.append(f"- leak: {note}")
        lines.append("")
    (RESULTS / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'baselines.md'} ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
