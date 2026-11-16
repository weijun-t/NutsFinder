"""Phase 4 evaluation: PotOddsBot and EquityBot against the baselines and each other.

Run:
    .venv\\Scripts\\python examples\\run_phase4.py

The design is fixed *before* anything is played, and nothing here depends
on intermediate results (no optional stopping, no re-runs with other seeds):

- pairings (bot A first):
    PotOddsBot vs RandomBot, CallStationBot, AlwaysRaiseBot
    EquityBot  vs RandomBot, CallStationBot, AlwaysRaiseBot
    EquityBot  vs PotOddsBot            (the tier test, PLAN rule 2)
- each pairing: a duplicate match of N_PAIRS = 10,000 pairs (20,000 hands),
  plus a plain match of 20,000 hands with the same master seed, only to
  measure how much duplicate reduces the SE (the verdicts use duplicate);
- one fresh MASTER_SEED for everything, chosen before the run. EquityBot's
  defaults (max_fold, range widths) were tuned on seed 777 and checked on
  4242, so neither of those may be used for the official numbers;
- bot parameters: the committed defaults, nothing passed in.

Parallelism: the jobs are independent and each is fully determined by
(pairing, kind, MASTER_SEED): bots are reseeded from the master seed before
every hand. So running them in separate processes cannot change any number,
only the wall-clock time.

Outputs:
- results/phase4.md: the report (tables, verdicts, breakdowns);
- results/<date>_<A>_vs_<B>.json / _plain.json: every number incl. per-unit results;
- results/<date>_<A>_vs_<B>.png / _plain.png: cumulative bb with the zero-edge
  95% band, and the histogram of per-unit results.
"""

from __future__ import annotations

import datetime as dt
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import Street
from nutsfinder.engine.cards import Deck
from nutsfinder.engine.equity import RANDOM_RANGE, equity_vs_range
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.eval.breakdown import _decisions, breakdown
from nutsfinder.eval.match import MatchResult, duplicate_match, plain_match
from nutsfinder.eval.report import save_json, save_plot

MASTER_SEED = 20261116  # fixed before the run; NOT the tuning seed (777) or the holdout (4242)
# Known overlap, kept so the official results stay valid: 20261116 is also the
# SEED of examples/make_preflop_table.py, i.e. the same random.Random seed that
# produced class 0 of both preflop tables (see compute_range_table in
# nutsfinder/engine/preflop.py). No measurable bias: the tables are constants
# shared by every bot and every match (so their tiny errors are fixed offsets,
# not effects that can line up with the deals), and the two uses consume the
# stream differently from the first draw on (rng.sample for the tables' equity
# draws, getrandbits(64) for the unit seeds here). Future match seeds must differ from all
# table seeds, and future generators should derive seeds by hashing
# (purpose, i, j) rather than adding offsets.
N_PAIRS = 10_000  # fixed in advance: 20,000 hands per pairing
PAIRINGS = [
    ("PotOddsBot", "RandomBot"),
    ("PotOddsBot", "CallStationBot"),
    ("PotOddsBot", "AlwaysRaiseBot"),
    ("EquityBot", "RandomBot"),
    ("EquityBot", "CallStationBot"),
    ("EquityBot", "AlwaysRaiseBot"),
    ("EquityBot", "PotOddsBot"),
]
BOTS = {
    "RandomBot": RandomBot,
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": PotOddsBot,
    "EquityBot": EquityBot,
}
SHORT = {
    "RandomBot": "Random",
    "CallStationBot": "CallStation",
    "AlwaysRaiseBot": "AlwaysRaise",
    "PotOddsBot": "PotOdds",
    "EquityBot": "Equity",
}
PROCESSES = 14
STREETS = [s.name.lower() for s in Street]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def commit_hash() -> str:
    """Current commit, with "+ uncommitted changes" if the tree is dirty."""
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


def fmt_p(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "n/a (no variation)"
    return "< 1e-300" if p == 0 else f"{p:.3g}"


# --- Extra diagnostics (descriptive only, no CIs: these are selections) -----


def action_profile(result: MatchResult) -> dict:
    """Per bot and street: action counts, all-ins, and a river "bluff" proxy.

    - ``river_aggr_weak``: river bets/raises made with < 50% exact equity
      against a uniformly random hand (a proxy for bluffs; a value bet
      against a random hand needs well over 50%).
    - ``folds_to_bet``: folds made when facing a bet (something to call).
    - ``folds_to_bet_best_hand``: of those, the folds where the folder's two
      cards would have won (or tied) on the full five-card board that the
      seed had in store. Read with care: "would have won at showdown"
      ignores the betting that would have followed, so it measures how much
      equity a bot gives up, not how much EV.
    - ``free_folds``: folds when checking was free (RandomBot does this);
      counted separately so they do not inflate the two numbers above.

    The bluff proxy uses a fixed ``seed=0`` so the diagnostics are
    reproducible (2,000 samples, or exact when the river allows it).
    """
    out: dict = {}
    for who in ("a", "b"):
        prof = {s: {"fold": 0, "check": 0, "call": 0, "bet": 0, "raise": 0, "all_in": 0} for s in STREETS}
        river_aggr = river_weak = folds = folds_best = free_folds = 0
        for g in result.games:
            seat = g.a_seat if who == "a" else 1 - g.a_seat
            log = g.log
            for r in log.actions:
                if r.player != seat:
                    continue
                st = r.street.name.lower()
                kind = {Fold: "fold", Check: "check", Call: "call", Bet: "bet", Raise: "raise"}[type(r.action)]
                prof[st][kind] += 1
                prof[st]["all_in"] += bool(r.all_in)
                if st == "river" and kind in ("bet", "raise"):
                    river_aggr += 1
                    e = equity_vs_range(
                        log.hole_cards[seat], RANDOM_RANGE, log.board, samples=2000, seed=0
                    ).equity
                    river_weak += e < 0.5
            if log.folded == seat:
                # The fold is the hand's last action: was there a bet to call?
                *_, (_, _, facing, _) = _decisions(log)
                if not facing:
                    free_folds += 1
                    continue
                folds += 1
                deck = Deck(log.seed)
                deck.deal(4)
                board = deck.deal(5)
                assert tuple(board[: len(log.board)]) == tuple(log.board), "deck reconstruction"
                mine = evaluate(list(log.hole_cards[seat]) + board)
                theirs = evaluate(list(log.hole_cards[1 - seat]) + board)
                folds_best += mine >= theirs
        out[who] = {
            "actions": prof,
            "river_aggressive": river_aggr,
            "river_aggressive_weak": river_weak,
            "folds_to_bet": folds,
            "folds_to_bet_best_hand": folds_best,
            "free_folds": free_folds,
        }
    return out


def run_job(job: tuple[str, str, str, str, str]) -> dict:
    """Play one match, save its JSON and plot, return a compact summary."""
    kind, a_name, b_name, today, commit = job
    start = time.perf_counter()
    if kind == "duplicate":
        res = duplicate_match(BOTS[a_name], BOTS[b_name], N_PAIRS, MASTER_SEED)
    else:
        res = plain_match(BOTS[a_name], BOTS[b_name], 2 * N_PAIRS, MASTER_SEED)
    play_s = time.perf_counter() - start
    profile = action_profile(res) if kind == "duplicate" else None
    stem = f"{today}_{SHORT[a_name]}_vs_{SHORT[b_name]}" + ("" if kind == "duplicate" else "_plain")
    meta = {"date": today, "commit": commit, "run_time_s": play_s, "bot_params": "committed defaults"}
    if profile:
        meta["action_profile"] = profile
    save_json(res, RESULTS / f"{stem}.json", meta)
    save_plot(res, RESULTS / f"{stem}.png")
    lo, hi = res.ci_bb100
    return {
        "kind": kind,
        "a": a_name,
        "b": b_name,
        "hands": res.hands_played,
        "units": res.n_units,
        "mean": res.mean_bb100,
        "ci": (lo, hi),
        "se": res.se_bb100,
        "sd": res.sd_per_unit,
        "p": res.p_value,
        "sig": res.stats.significant,
        "no_var": res.stats.no_variation,
        "verdict": res.verdict(),
        "summary": res.summary(),
        "breakdown": breakdown(res) if kind == "duplicate" else None,
        "profile": profile,
        "time_s": time.perf_counter() - start,
        "stem": stem,
    }


# --- Report -----------------------------------------------------------------


def pairing_section(d: dict) -> list[str]:
    a, b = SHORT[d["a"]], SHORT[d["b"]]
    bd, prof = d["breakdown"], d["profile"]
    lines = [f"### {a} vs {b}", "", f"- verdict: {d['verdict']}"]
    for label, row in bd["by_position"].items():
        lines.append(
            f"- A on {label}: {row['bb100']:+.1f} bb/100 "
            f"(SE {row['se_bb100']:.1f}, per hand, no duplicate cancellation)"
        )
    parts = [
        f"{s} {v['share']:.0%} ({v['bb100_contribution']:+.1f})"
        for s, v in bd["by_end_street"].items()
        if v["hands"]
    ]
    lines.append("- hands ended on (share, A's bb/100 from them; descriptive, no CI): " + ", ".join(parts))
    lines += [
        "",
        "| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for who, name in (("a", a), ("b", b)):
        folds = bd[f"folds_{who}"]
        for s in STREETS:
            p = prof[who]["actions"][s]
            n = sum(p[k] for k in ("fold", "check", "call", "bet", "raise"))
            if not n:
                continue
            f = folds[s]
            rate = f["fold_to_bet_rate"]
            rate_txt = "-" if rate is None or (isinstance(rate, float) and math.isnan(rate)) else f"{rate:.0%}"
            lines.append(
                f"| {name} | {s} | {n:,} | {p['fold']:,} | {p['check']:,} | {p['call']:,} | "
                f"{p['bet']:,} | {p['raise']:,} | {p['all_in']:,} | {f['faced_bet']:,} | {rate_txt} | "
                f"{f['free_folds']:,} |"
            )
    lines.append("")
    for who, name in (("a", a), ("b", b)):
        p = prof[who]
        weak = p["river_aggressive_weak"]
        tot = p["river_aggressive"]
        share = f" ({weak / tot:.0%})" if tot else ""
        best = f" ({p['folds_to_bet_best_hand'] / p['folds_to_bet']:.0%})" if p["folds_to_bet"] else ""
        lines.append(
            f"- {name}: river bets/raises {tot:,}, of which with < 50% equity vs a random hand "
            f"(bluff proxy) {weak:,}{share}; folds to a bet {p['folds_to_bet']:,}, of which the folded "
            f"hand would have won or tied on the full board {p['folds_to_bet_best_hand']:,}{best}; "
            f"free folds {p['free_folds']:,}"
        )
    for note in bd["leaks"]:
        lines.append(f"- leak: {note}")
    lines.append("")
    return lines


def main() -> None:
    today = dt.date.today().isoformat()
    commit = commit_hash()
    print(f"master seed {MASTER_SEED}, {N_PAIRS:,} duplicate pairs per pairing, commit {commit}")
    jobs = [("duplicate", a, b, today, commit) for a, b in PAIRINGS]
    jobs += [("plain", a, b, today, commit) for a, b in PAIRINGS]

    start = time.perf_counter()
    with mp.get_context("spawn").Pool(PROCESSES) as pool:
        out = pool.map(run_job, jobs, chunksize=1)
    elapsed = time.perf_counter() - start
    dup = {(d["a"], d["b"]): d for d in out if d["kind"] == "duplicate"}
    plain = {(d["a"], d["b"]): d for d in out if d["kind"] == "plain"}
    for d in out:
        print()
        print(d["summary"])
    cpu_total = sum(d["time_s"] for d in out)

    lines = [
        "# Phase 4 results: PotOddsBot and EquityBot",
        "",
        f"- date: {today}",
        f"- master seed: {MASTER_SEED} (fixed before the run; same deals for every pairing)",
        f"- n: {N_PAIRS:,} duplicate pairs = {2 * N_PAIRS:,} hands per pairing, fixed in advance, no early stopping",
        "- duplicate: each deal played twice, bots swapped between seats (cards and button swap with them)",
        "- bot parameters: the committed defaults (PotOddsBot: bet > 60%, raise > 70%, 300 samples; "
        "EquityBot: ranges 100/50/30/15%, sizes 1/2 pot, pot, all-in, max_fold 0.1, 400 samples)",
        f"- commit: {commit}",
        f"- run time: {elapsed:.0f} s wall clock ({PROCESSES} processes; "
        f"{cpu_total:.0f} s summed over the {len(jobs)} jobs). Every job is determined by its "
        "pairing and the master seed alone, so parallelism cannot change any number.",
        "",
        "## Tuning history (why this seed)",
        "",
        "EquityBot's `max_fold` (0.1, chosen from 0, 0.1, 0.2, 0.3, 0.5) and its range widths were tuned on",
        "master seed 777 (1,000 pairs per setting) and checked on seed 4242 as a holdout (1,000 pairs).",
        f"Choosing parameters on the data you then report is overfitting, so the official evaluation uses a",
        f"fresh seed, {MASTER_SEED}, fixed before this run, with no parameter changes and no re-runs.",
        "",
        "Win rate is for bot A (first named), in bb/100 hands. 95% CI = mean +/- 1.96 SE",
        "(normal approximation, per-pair results treated as i.i.d.). p = two-sided z-test vs 0.",
        "",
        "## Duplicate matches",
        "",
        "| pairing (A vs B) | hands | bb/100 | 95% CI | +/- | SE | SD per pair (bb) | p-value | significant |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for a, b in PAIRINGS:
        d = dup[(a, b)]
        lo, hi = d["ci"]
        sig = "exact 0 (no variation)" if d["no_var"] else "yes" if d["sig"] else "no"
        lines.append(
            f"| {SHORT[a]} vs {SHORT[b]} | {d['hands']:,} | {d['mean']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
            f"{(hi - lo) / 2:.1f} | {d['se']:.1f} | {d['sd']:.1f} | {fmt_p(d['p'])} | {sig} |"
        )
    lines += [
        "",
        "All pairings use the same master seed, so they share deals (common random numbers). Their",
        "estimates are therefore correlated: do not compare two pairings using independent-sample SEs.",
        "",
        "## Phase 4 checks",
        "",
    ]
    for bot in ("PotOddsBot", "EquityBot"):
        beats = all(dup[(bot, b)]["sig"] and dup[(bot, b)]["mean"] > 0
                    for b in ("RandomBot", "CallStationBot", "AlwaysRaiseBot"))
        lines.append(f"- {bot} beats every baseline significantly: {'yes' if beats else 'NO'}")
    t = dup[("EquityBot", "PotOddsBot")]
    lo, hi = t["ci"]
    tier = t["sig"] and t["mean"] > 0
    lines.append(
        f"- tier test (rule 2), EquityBot vs PotOddsBot: {t['mean']:+.1f} bb/100, 95% CI [{lo:+.1f}, {hi:+.1f}]: "
        f"{'EquityBot counts as a new tier' if tier else 'EquityBot does NOT count as a new tier'}"
    )
    lines += [
        "",
        "## Variance reduction: plain vs duplicate (same number of hands, same seed)",
        "",
        "Plain: 20,000 independent hands, button alternating. "
        "Hands-equivalent = (SE plain / SE duplicate)^2: how many times more plain hands",
        "would be needed for the same precision.",
        "",
        "| pairing | plain bb/100 | plain 95% CI | plain SE | duplicate SE | SE ratio | hands-equivalent |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for a, b in PAIRINGS:
        dd, pp = dup[(a, b)], plain[(a, b)]
        ratio = pp["se"] / dd["se"] if dd["se"] > 0 else math.inf
        plo, phi = pp["ci"]
        lines.append(
            f"| {SHORT[a]} vs {SHORT[b]} | {pp['mean']:+.1f} | [{plo:+.1f}, {phi:+.1f}] | {pp['se']:.1f} | "
            f"{dd['se']:.1f} | {ratio:.2f} | {ratio**2:.1f}x |"
        )
    lines += ["", "## Verdicts and breakdowns (duplicate matches)", ""]
    for a, b in PAIRINGS:
        lines += pairing_section(dup[(a, b)])
    lines += ["## Files", ""]
    for a, b in PAIRINGS:
        s = dup[(a, b)]["stem"]
        lines.append(f"- results/{s}.json, results/{s}.png, results/{s}_plain.json, results/{s}_plain.png")
    lines.append("")
    (RESULTS / "phase4.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'phase4.md'} ({elapsed:.0f} s wall clock)")


if __name__ == "__main__":
    main()
