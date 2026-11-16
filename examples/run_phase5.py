"""Phase 5 evaluation (skeleton): BayesBot, a learning bot, against the baselines.

Run (only once BayesBot is finished and its defaults are frozen):
    .venv\\Scripts\\python examples\\run_phase5.py            # the official run
    .venv\\Scripts\\python examples\\run_phase5.py --plan     # print the design and budget only

Why sessions. BayesBot keeps per-instance opponent memory across hands, so
the hands of one long match are neither independent nor identically
distributed and a per-pair SE is invalid (see nutsfinder/eval/sessions.py).
Every pairing is therefore played as independent sessions of
PAIRS_PER_SESSION duplicate pairs, each with fresh bot instances in each of
its two rooms. The unit of analysis is the session. EquityBot (no memory)
is evaluated with exactly the same design so the two are comparable; for it
the session estimate equals a long duplicate match in expectation.

Every number reported is "bb/100 over the first 500 hands against a fresh
opponent model". A different session length is a different estimand.

Common random numbers: BayesBot vs X and EquityBot vs X use the same master
seed and the same number of sessions, so session s has the same deals in
both. "BayesBot beats X by more than EquityBot does" is then tested with a
PAIRED difference per session (CI across sessions), which is tighter than
combining two independent SEs.

The design is fixed in advance (no optional stopping, no re-runs with other
seeds). It comes from a pilot on seed 999 (NOT the official seed).

DESIGN NOTES
------------
Pilot: seed 999, EquityBot vs each opponent, 250 pairs (500 hands) per
session, 22 processes. SD = per-session SD of bb/100; ms = wall-clock ms per
hand with the whole pool busy (this machine scales only ~6x on 22 processes).

    opponent        sessions  bb/100 [95% CI]      SD/session  SD/pair (bb)  ms/hand
    RandomBot             88  +772  [+746, +798]        123.6        40.0       0.91
    PotOddsBot            88  +227  [+207, +247]         94.9        29.8       6.36
    AlwaysRaiseBot        66  +1189 [+1145, +1233]      182.9        59.1       2.81
    CallStationBot        66  +2040 [+2002, +2078]      157.6        48.4       4.90
    EquityBot (self)      44  -6.5  [-20.9, +7.9]        48.8        16.1       8.02

Checks: for i.i.d. pairs SD_session = SD_pair / (2 sqrt(250)) * 100, which
gives 126.6 / 94.3 / 186.8 / 153.0 from the per-pair SDs, matching the
measured values; the design effect (session SE / naive per-pair SE)^2 was
0.92-1.06 everywhere (no within-session correlation, as expected without
memory). Self-play covers 0.

Power: n_sessions = ((z_0.975 + z_0.8) * SD / 50)^2 = 7.85 (SD / 50)^2 for a
50 bb/100 edge at 80% power, alpha 0.05. BayesBot's SD is unknown (exploiting
an opponent usually means bigger pots, so more variance) and the pilot SDs
are themselves estimates (+/- ~8%), so the design SD is the pilot SD x 1.2
(= 1.44x the sessions). For BayesBot vs EquityBot there is no non-self-play
measurement, so the RandomBot value is used. Rounded up to a multiple of 10,
with a floor of 50 sessions (CLT and a stable SD estimate):

    opponent        design SD  required  N_SESSIONS  hands per pairing
    RandomBot           148.3        70          70        35,000
    CallStationBot      189.1       113         120        60,000
    AlwaysRaiseBot      219.5       152         160        80,000
    PotOddsBot          113.9        41          50        25,000
    EquityBot           148.3        70          70        35,000

Budget: hands x pilot ms/hand, BayesBot assumed as expensive as EquityBot
(BayesBot vs EquityBot costed as EquityBot self-play): ~28 min wall clock
for all 9 pairings (see --plan). One uniform N for every pairing would need
160 sessions everywhere, ~45 min. The total number of hands needed does not
depend on the session length (SD_session ~ 1 / sqrt(length) for a fixed
edge), so shorter sessions would not save time; longer ones would cut the
number of units below the floor for the cheapest-to-detect pairings.

Why 250 pairs per session: the bot needs hands to learn (a Beta posterior on
a frequency after ~100 observations has an SD of about 0.05), and the
learning-gain test compares the first 100 hands with the last 100; 500
hands leaves 300 in between.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.eval.report import save_session_json, save_session_plot
from nutsfinder.eval.sessions import SessionMatchResult, combine_sessions, learning_gain, play_session
from nutsfinder.eval.stats import required_units, summarize

try:  # being built in parallel; fail clearly (at run time, not import time) if missing
    from nutsfinder.bots.bayes_bot import BayesBot
except ImportError as exc:  # pragma: no cover - depends on the state of the repo
    BayesBot = None
    _BAYES_IMPORT_ERROR: ImportError | None = exc
else:
    _BAYES_IMPORT_ERROR = None

# --- Fixed design (chosen before the official run; see DESIGN NOTES) -------------
MASTER_SEED = 20261117  # reserved for the official Phase 5 run; the pilot used 999
PAIRS_PER_SESSION = 250  # 500 hands per session
BUCKET_PAIRS = 25  # learning-curve resolution (50 hands per bucket)
TARGET_EDGE_BB100 = 50.0  # smallest edge each pairing must detect...
TARGET_POWER = 0.8  # ...with this power, at alpha = 0.05 (two-sided)
SD_MARGIN = 1.2  # BayesBot's SD is unknown: inflate the pilot SDs by 20%
MIN_SESSIONS = 50
# Pilot per-session SD (bb/100, 250-pair sessions, EquityBot vs opponent, seed 999).
PILOT_SD = {
    "RandomBot": 123.6,
    "CallStationBot": 157.6,
    "AlwaysRaiseBot": 182.9,
    "PotOddsBot": 94.9,
    "EquityBot": 123.6,  # proxy: RandomBot's value (self-play's 48.8 is not representative)
}
# Sessions per opponent; the same for BayesBot and EquityBot (paired comparison).
N_SESSIONS = {
    "RandomBot": 70,
    "CallStationBot": 120,
    "AlwaysRaiseBot": 160,
    "PotOddsBot": 50,
    # The tier test (rule 2). Raised from the power minimum of 70 BEFORE the
    # official run, because the seed-555/556 tuning runs put BayesBot's edge
    # over EquityBot at only about +50 to +90 bb/100: 70 sessions give ~80%
    # power at 50, 150 sessions give ~98%. Fixed in advance, not after
    # looking at official data (that would be optional stopping).
    "EquityBot": 150,
}
# Pilot wall-clock ms per hand with 22 busy processes; BayesBot assumed equal to EquityBot.
MS_PER_HAND_WALL = {
    "RandomBot": 0.91,
    "CallStationBot": 4.90,
    "AlwaysRaiseBot": 2.81,
    "PotOddsBot": 6.36,
    "EquityBot": 8.02,
}
PROCESSES = 22  # of 24 logical cores
EARLY, LATE = (0, 50), (200, 250)  # pair ranges for the learning-gain test (first / last 100 hands)

BASELINES = ["RandomBot", "CallStationBot", "AlwaysRaiseBot", "PotOddsBot"]
PAIRINGS = [("BayesBot", b) for b in BASELINES + ["EquityBot"]] + [("EquityBot", b) for b in BASELINES]
BOTS = {
    "RandomBot": RandomBot,
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": PotOddsBot,
    "EquityBot": EquityBot,
    "BayesBot": BayesBot,
}
SHORT = {k: k.removesuffix("Bot") for k in BOTS}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def sessions_required(opponent: str) -> int:
    """Sessions needed to detect TARGET_EDGE_BB100 with TARGET_POWER against ``opponent``."""
    n = required_units(SD_MARGIN * PILOT_SD[opponent], TARGET_EDGE_BB100, power=TARGET_POWER)
    return max(MIN_SESSIONS, n)


def commit_hash() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + (" + uncommitted changes" if dirty else "")


def plan() -> None:
    print(f"master seed {MASTER_SEED}; {PAIRS_PER_SESSION} pairs ({2 * PAIRS_PER_SESSION} hands) per session")
    total_s = 0.0
    for a, b in PAIRINGS:
        n = N_SESSIONS[b]
        hands = n * 2 * PAIRS_PER_SESSION
        sec = hands * MS_PER_HAND_WALL[b] / 1000
        total_s += sec
        sd = SD_MARGIN * PILOT_SD[b]
        mde = (1.959964 + 0.841621) * sd / math.sqrt(n)
        print(f"  {a} vs {b}: {n} sessions = {hands:,} hands (required {sessions_required(b)}); "
              f"design SD {sd:.0f} -> SE {sd / math.sqrt(n):.1f}, 80%-power MDE {mde:.1f} bb/100; "
              f"~{sec / 60:.1f} min")
    print(f"estimated wall clock: {total_s / 60:.1f} min on {PROCESSES} processes")


def paired_difference(x: SessionMatchResult, y: SessionMatchResult):
    """x - y per session (same master seed, same session s = same deals); CI across sessions."""
    assert x.n_sessions == y.n_sessions and x.pairs_per_session == y.pairs_per_session
    assert [s.seed for s in x.sessions] == [s.seed for s in y.sessions]
    diffs = [sx.total_bb - sy.total_bb for sx, sy in zip(x.sessions, y.sessions)]
    return summarize(diffs, hands_per_unit=x.hands_per_session)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the design and budget, play nothing")
    args = ap.parse_args()
    short = [b for b in N_SESSIONS if N_SESSIONS[b] < sessions_required(b)]
    if short:
        raise SystemExit(f"design error: N_SESSIONS below the power requirement for {short}")
    if args.plan:
        plan()
        return
    if BayesBot is None:
        raise SystemExit(
            "BayesBot is not available (src/nutsfinder/bots/bayes_bot.py could not be imported: "
            f"{_BAYES_IMPORT_ERROR}). Finish and freeze BayesBot before the official run."
        )

    today = dt.date.today().isoformat()
    commit = commit_hash()
    print(f"master seed {MASTER_SEED}, {PAIRS_PER_SESSION} pairs per session, commit {commit}")
    start = time.perf_counter()
    results: dict[tuple[str, str], SessionMatchResult] = {}
    lines = [
        "# Phase 5 results: BayesBot (session duplicate matches)",
        "",
        f"- date: {today}; commit: {commit}; master seed {MASTER_SEED} (fixed in advance)",
        f"- design: independent sessions of {PAIRS_PER_SESSION} duplicate pairs ({2 * PAIRS_PER_SESSION} "
        "hands), fresh bots per session and room; sessions per opponent: "
        + ", ".join(f"{SHORT[k]} {v}" for k, v in N_SESSIONS.items()),
        "- unit = session; 95% CI = mean +/- 1.96 SE over sessions; numbers are bb/100 over the "
        f"first {2 * PAIRS_PER_SESSION} hands against a fresh opponent model",
        "",
        "| pairing | hands | bb/100 | 95% CI | SE | SD/session | p | design effect | "
        f"gain, pairs {LATE} minus {EARLY} [95% CI] |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    with mp.get_context("spawn").Pool(PROCESSES) as pool:
        # Submit every session of every pairing at once so the pool never idles
        # between pairings. Each session is fully determined by its arguments.
        pending = {
            (a, b): pool.starmap_async(
                play_session,
                [(BOTS[a], BOTS[b], PAIRS_PER_SESSION, MASTER_SEED, s, True) for s in range(N_SESSIONS[b])],
                chunksize=1,
            )
            for a, b in PAIRINGS
        }
        for (a, b), job in pending.items():
            res = combine_sessions(job.get(), MASTER_SEED, PAIRS_PER_SESSION)
            stem = f"{today}_{SHORT[a]}_vs_{SHORT[b]}_sessions"
            gain = learning_gain(res, EARLY, LATE)
            meta = {"date": today, "commit": commit, "bot_params": "committed defaults",
                    "learning_gain": {"early_pairs": EARLY, "late_pairs": LATE, "mean_bb100": gain.mean_bb100,
                                      "ci95_bb100": list(gain.ci_bb100), "p_value": gain.p_value}}
            save_session_json(res, RESULTS / f"{stem}.json", BUCKET_PAIRS, meta)
            save_session_plot(res, RESULTS / f"{stem}.png", BUCKET_PAIRS)
            lo, hi = res.ci_bb100
            glo, ghi = gain.ci_bb100
            lines.append(
                f"| {SHORT[a]} vs {SHORT[b]} | {res.hands_played:,} | {res.mean_bb100:+.1f} | "
                f"[{lo:+.1f}, {hi:+.1f}] | {res.se_bb100:.1f} | {res.sd_session_bb100:.1f} | "
                f"{res.p_value:.3g} | {res.design_effect():.2f} | {gain.mean_bb100:+.1f} [{glo:+.1f}, {ghi:+.1f}] |"
            )
            print(res.summary(), flush=True)
            # Keep only per-hand nets (drop hand logs) for the paired comparisons below.
            results[(a, b)] = combine_sessions(
                [s.__class__(s.index, s.seed, s.bot_a, s.bot_b, s.hand_results, s.pair_results, ())
                 for s in res.sessions],
                MASTER_SEED, PAIRS_PER_SESSION,
            )
    lines += [
        "",
        "## BayesBot minus EquityBot against the same opponent (paired by session: same deals)",
        "",
        "| opponent | difference bb/100 | 95% CI | SE | p |",
        "|---|---:|---|---:|---:|",
    ]
    for b in BASELINES:
        d = paired_difference(results[("BayesBot", b)], results[("EquityBot", b)])
        lo, hi = d.ci_bb100
        lines.append(f"| {SHORT[b]} | {d.mean_bb100:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {d.se_bb100:.1f} | {d.p_value:.3g} |")
    elapsed = time.perf_counter() - start
    lines += ["", f"run time: {elapsed:.0f} s wall clock on {PROCESSES} processes", ""]
    (RESULTS / "phase5.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'phase5.md'} ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
