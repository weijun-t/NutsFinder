r"""Phase 7, 6-max evaluation: the multiway tier ladder against a fixed mixed field (rotation duplicate, paired arms).

Run:
    .venv\Scripts\python examples\run_6max.py --plan             # design, power, budget; plays nothing
    .venv\Scripts\python examples\run_6max.py --smoke            # 2 short sessions per arm, pilot seed (plumbing)
    .venv\Scripts\python examples\run_6max.py --pilot            # the pilot (seed 75025, NOT official)
    .venv\Scripts\python examples\run_6max.py                    # THE official run (once, clean tree)
    .venv\Scripts\python examples\run_6max.py --resume-partial   # finish a crashed official run
    .venv\Scripts\python examples\run_6max.py --rebuild-outputs results\phase7_6max_official_partial.jsonl
                                                                 # outputs only, from a complete log

The pre-registration is the top of ``results/phase7_6max.md`` (committed before the official
run). The official run writes its numbers below ``RESULTS_MARKER`` there and refuses to run if
anything is already below it, if official result files exist, or if a partial log exists without
``--resume-partial``. It refuses ``--allow-dirty`` and a dirty tree (untracked files included).

Design (docs/phase7_multiway_research.md section 3; reasons in the pre-registration)
------------------------------------------------------------------------------------
- Table: 6 seats, 10,000 chips (100bb) reset every hand, blinds 50/100.
- Field ``mixed`` = [MultiEquityBot, MultiPotOddsBot, CallStation, AlwaysRaise, Random] in a fixed
  cyclic order; session s uses that order rotated by ``s % 5`` (balanced over sessions when N is a
  multiple of 5), so no single seating arrangement around the test seat is privileged.
- Block = one deal (deal seed + one reseed per seat) played in 6 rooms. In room r the test bot sits
  in seat r and the field fills seats r+1..r+5 in the session's order; the cards per seat, the
  button (``block % 6``) and the per-seat reseeds are the same in all rooms. So per block every bot
  sits in every seat exactly once on the same deal (tested).
- Every bot is reseeded before every hand from ``(session, block, seat)``: field bots draw the same
  random numbers in every arm unless the test bot's actions change what they are asked.
- Session = ``BLOCKS_PER_SESSION`` = 250 blocks = 6 rooms x 250 hands = 1,500 test-seat hands.
  Every room is a fresh table (fresh bot objects): a learning test bot (MultiBayesBot) learns for
  250 hands at its table, as each heads-up BayesBot instance did in Phases 5/6 (250 hands each).
- Arms = the test bot in {Bayes, Equity} (CFR and PotOdds arms not run: see ``ARMS``); all arms use the SAME seeds (CRN):
  same deals, same field, same reseeds. Unit = session; CIs across sessions.
- PRIMARY (one test, alpha 0.05 two-sided): D = (MultiBayesBot's session bb/100) - (MultiEquityBot's),
  same seat, same deals, same field. "Beats" iff the 95% CI lies above 0. Everything else is
  secondary / descriptive.

Crash-safe: every finished room (one task = one arm, session, room) is appended (and fsynced) to a
JSONL partial log after a header identifying the design; ``--resume-partial`` re-reads it (header
must match, a torn last line is dropped) and plays only the missing rooms. Rooms are deterministic,
so the numbers equal those of an uninterrupted run (tested). Helpers are reused from
examples/run_exploit_back.py and examples/run_phase6.py.

Memory: OPENBLAS/OMP/MKL threads are set to 1 before numpy is imported. Workers return a compact
per-room record (per-hand nets, end streets, fold counts, mode counts), never hand logs.
"""

from __future__ import annotations

import os

# Before ANY import that loads numpy (this module, and spawned workers that re-import it).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from nutsfinder.bots.multiway import MultiBayesBot, MultiCFRBot, MultiEquityBot, MultiPotOddsBot  # noqa: E402
from nutsfinder.engine.actions import Bet, Call, Fold, Raise  # noqa: E402
from nutsfinder.engine.betting import Street  # noqa: E402
from nutsfinder.engine.multiway import (  # noqa: E402
    MultiAlwaysRaiseBot,
    MultiCallStationBot,
    MultiRandomBot,
    play_hand_multi,
    position_names,
)
from nutsfinder.eval.sessions import session_seed  # noqa: E402
from nutsfinder.eval.stats import required_units, z_critical  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MD = RESULTS / "phase7_6max.md"
RESULTS_MARKER = "<!-- OFFICIAL RESULTS BELOW: written by examples/run_6max.py, do not edit by hand -->"
OFFICIAL_PARTIAL = "phase7_6max_official_partial.jsonl"

# Reuse the exploit-back machinery (partial log, git checks, stats) and, through it, Phase 6's
# (pinned model check, splice, memory probe, power helpers).
_spec = importlib.util.spec_from_file_location("run_exploit_back_for_6max", ROOT / "examples" / "run_exploit_back.py")
reb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reb)
rp6 = reb.rp6

# --- Fixed design ------------------------------------------------------------------------------
# Seeds used before (never reuse): 999, 555, 556, 777, 4242, 20261115, 20261116, 20261117,
# 20261116+k, 202611186, 202611188, 202611231, 7, 61, 27182818, 16180339, 31415926, 17320508,
# 14142135, 46368, 20261124 (examples/run_multiway_sanity.py), 20251123, 987654321.
# Both seeds below were grepped against the repo (no occurrence before this file).
MASTER_SEED = 202611241  # official
PILOT_SEED = 75025  # pilot and smoke (NOT official)
N_SEATS = 6
BB = 100
BLOCKS_PER_SESSION = 250  # hands per room = learning window of each bot instance; 1,500 test-seat hands
SMOKE_BLOCKS = 12
WINDOW_BLOCKS = 25  # learning-curve windows (25 blocks = 150 test-seat hands)
PROCESSES = 22
ALPHA = 0.05
TARGET_POWER = 0.8
SD_MARGIN = 1.2
MIN_SESSIONS = 50
SESSION_STEP = 10  # N rounded up to a multiple of 10 (keeps the 5 field orders balanced)
EDGE_BB100 = 60.0  # minimum detectable D of the primary (80% power). Pre-pilot value was 50; changed to 60 AFTER
# the pilot by the project owner, for run time (pilot D -8.0 [-96.5, +80.5]); disclosed in results/phase7_6max.md.
EDGE_BB100_PRE_PILOT = 50.0
BUDGET_HOURS = 3.0  # official wall-clock target at the pilot rate x THROTTLE_MARGIN
THROTTLE_MARGIN = 1.6
# Scope-reduction rule, fixed BEFORE the pilot: if the official estimate exceeds BUDGET_HOURS, drop
# the PotOdds arm (secondary); never the primary arms, never change EDGE_BB100; if the primary alone
# does not fit, say so and let the owner decide (no silent change of the edge).
# What happened: the primary alone did not fit (3.9 h at the pilot rate, 6.3 h x THROTTLE_MARGIN), so that last
# branch was triggered; the owner then OVERRODE the "never change EDGE_BB100" clause (50 -> 60). The final design
# is still over budget by this definition (2.9 h; 4.6 h x THROTTLE_MARGIN > 3 h), and the owner accepted that.

FIELD_KINDS = ("Equity", "PotOdds", "CallStation", "AlwaysRaise", "Random")
FIELDS: dict[str, tuple[str, ...]] = {"mixed": FIELD_KINDS}
# No CFR arm: MultiCFRBot uses the HU blueprint only when a hand is heads-up preflop between some X and
# the BB after everyone else folded. CallStation and AlwaysRaise never fold, and both sit in every hand
# of this field, so such a spot cannot occur: MultiCFRBot here IS MultiEquityBot with another seed
# (the smoke run confirmed 0 blueprint decisions of 269). The arm would cost ~1/4 of the run for a
# comparison that is null by construction. ``make_bot("CFR")`` is kept for other fields.
# The pilot ran ("Bayes", "Equity", "PotOdds"). The official estimate with PotOdds at N = 150 is 5.0 h at the
# pilot rate (8.0 h x THROTTLE_MARGIN; 3.7 h at the final N = 110), over BUDGET_HOURS, so the pre-pilot scope rule dropped the PotOdds arm
# (secondary). Re-adding it is this one line (+1.1 h expected); every pair involving it is then reported again.
ARMS: dict[str, tuple[str, ...]] = {"mixed": ("Bayes", "Equity")}
PRIMARY = ("mixed", "Bayes", "Equity")
SECONDARY_PAIRS = (("mixed", "Equity", "PotOdds"), ("mixed", "Bayes", "PotOdds"))
ROTATE_FIELD_ORDER = True

# Pilot (seed 75025, NOT official). Filled from `--pilot` output; see results/phase7_6max.md.
PILOT_SESSIONS = 16
# Pilot 2026-11-24, seed 75025, 16 sessions x 1,500 test-seat hands per arm, arms Bayes/Equity/PotOdds,
# 22 processes, 1,953 s, commit 7cb9678 + uncommitted run_6max.py. NOT OFFICIAL (results/phase7_6max.md).
PILOT_SD: dict[str, float] = {  # SD per session (bb/100 over 1,500 test-seat hands)
    "mixed|Bayes-Equity": 180.7, "mixed|Equity-PotOdds": 179.6, "mixed|Bayes-PotOdds": 171.9,
    "mixed|Bayes": 220.9, "mixed|Equity": 197.1, "mixed|PotOdds": 192.5,
}
MS_PER_HAND_WALL: dict[str, float] = {  # pool-wall ms per test-seat hand on 22 processes (pilot)
    "mixed|Bayes": 31.24, "mixed|Equity": 31.71, "mixed|PotOdds": 17.20,
}
# Fixed after the pilot: design SD 1.2 x 180.7 = 216.8, edge 60, 80% power, alpha 0.05 -> 103 -> 110
# (at the pre-pilot edge 50 it was 148 -> 150; power at 50 with N = 110 is 68%).
N_SESSIONS: dict[str, int] = {"mixed": 110}

STREETS = ("preflop", "flop", "turn", "river")
END_STREETS = ("preflop", "flop", "turn", "river", "showdown")
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
MODEL = rp6.MODEL


# --- Bots ---------------------------------------------------------------------------------------


def make_bot(kind: str):
    """A fresh bot. Every bot is reseeded before every hand, so the construction seed is irrelevant."""
    if kind == "Bayes":
        return MultiBayesBot(seed=0)
    if kind == "Equity":
        return MultiEquityBot(seed=0)
    if kind == "PotOdds":
        return MultiPotOddsBot(seed=0)
    if kind == "CFR":
        return MultiCFRBot(str(MODEL), seed=0)
    if kind == "CallStation":
        return MultiCallStationBot()
    if kind == "AlwaysRaise":
        return MultiAlwaysRaiseBot()
    if kind == "Random":
        return MultiRandomBot(seed=0)
    raise ValueError(kind)


def field_order(field: str, s: int) -> tuple[str, ...]:
    """The field's cyclic order in session s (rotated by s % 5 when ROTATE_FIELD_ORDER)."""
    base = FIELDS[field]
    k = s % len(base) if ROTATE_FIELD_ORDER else 0
    return base[k:] + base[:k]


def seat_layout(field: str, arm: str, s: int, room: int) -> list[str]:
    """Bot kind per seat in room ``room``: the test bot in seat ``room``, the field clockwise after it."""
    layout = [""] * N_SEATS
    layout[room] = arm
    for i, kind in enumerate(field_order(field, s)):
        layout[(room + 1 + i) % N_SEATS] = kind
    return layout


# --- Seeds ----------------------------------------------------------------------------------------


def field_seed(master_seed: int, field: str) -> int:
    """Master seed of one field: shared by all its arms, different across fields."""
    key = f"nutsfinder/6max|{master_seed}|{field}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def block_seeds(sess_seed: int, blocks: int) -> list[tuple[int, list[int]]]:
    """Per block: (deal seed, one reseed per seat). The same in every room and every arm."""
    rng = random.Random(sess_seed)
    return [(rng.getrandbits(64), [rng.getrandbits(64) for _ in range(N_SEATS)]) for _ in range(blocks)]


def seeds_digest(seeds: list[tuple[int, list[int]]]) -> str:
    return hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:16]


# --- Per-hand bookkeeping -------------------------------------------------------------------------


def exit_street(log, seat: int) -> str:
    """Where the hand ended FOR ``seat``: the street it folded on, else ``"showdown"`` (or won uncontested).

    Not the street the hand ended on: in this field CallStation and AlwaysRaise never fold, so every
    hand reaches a showdown. ``"showdown"`` here means "did not fold" (it includes winning when all
    others folded, which cannot happen with those two seated).
    """
    for r in log.actions:
        if r.player == seat and isinstance(r.action, Fold):
            return r.street.name.lower()
    return "showdown"


def decisions(log):
    """Yield (seat, street, facing a bet, folded, action) for every action of a multiway hand.

    Facing a bet = some seat has more chips in on this street than the actor (blinds count preflop).
    """
    n = log.n_players
    street, bets = None, [0] * n
    for r in log.actions:
        if r.street != street:
            street = r.street
            bets = list(log.blinds) if street == Street.PREFLOP else [0] * n
        me = r.player
        yield me, street.name.lower(), max(bets) > bets[me], isinstance(r.action, Fold), r.action
        bets[me] += r.chips


def play_room(field: str, arm: str, master_seed: int, s: int, room: int, blocks: int) -> dict:
    """One room of one session of one arm: ``blocks`` hands with the test bot in seat ``room``."""
    layout = seat_layout(field, arm, s, room)
    bots = [make_bot(k) for k in layout]
    seeds = block_seeds(session_seed(field_seed(master_seed, field), s), blocks)
    nets, streets = [], []
    cards = hashlib.sha256()
    folds = {st: [0, 0, 0] for st in STREETS}  # faced a bet, folded to it, folded when checking was free
    vpip = pfr = 0
    field_net: Counter[str] = Counter()
    for b, (deal, reseeds) in enumerate(seeds):
        for bot, sd in zip(bots, reseeds):
            bot.reseed(sd)
        log = play_hand_multi(bots, deal, button=b % N_SEATS)
        net = log.result.net
        cards.update(repr((log.seed, log.button, [[str(c) for c in h] for h in log.hole_cards],
                           [str(c) for c in log.board])).encode())
        nets.append(int(net[room]))
        streets.append(exit_street(log, room)[0].upper())
        for j in range(N_SEATS):
            if j != room:
                field_net[layout[j]] += int(net[j])
        v = p = False
        for actor, st, facing, folded, action in decisions(log):
            if actor != room:
                continue
            c = folds[st]
            if facing:
                c[0] += 1
                c[1] += folded
            elif folded:
                c[2] += 1
            if st == "preflop" and isinstance(action, (Call, Bet, Raise)):
                v = True
                p = p or isinstance(action, (Bet, Raise))
        vpip += v
        pfr += p
    test = bots[room]
    modes = dict(test.mode_counts) if isinstance(test, MultiCFRBot) else {}
    stats = dict(getattr(test, "stats", {}) or {})
    if isinstance(test, MultiCFRBot):
        stats = dict(test.fallback.stats)
    return {"field": field, "arm": arm, "index": s, "room": room, "layout": layout,
            "seeds_digest": seeds_digest(seeds), "cards_digest": cards.hexdigest()[:16], "nets": nets, "streets": "".join(streets), "folds": folds,
            "vpip": vpip, "pfr": pfr, "field_net": dict(field_net), "modes": modes, "bot_stats": stats}


# --- Workers and tasks -----------------------------------------------------------------------------


def _init_worker(turn_tables: str | None, need_cfr: bool) -> None:
    from nutsfinder.bots.cfr_bot import preload_turn_tables, require_trained

    if turn_tables:
        preload_turn_tables(turn_tables)
    if need_cfr:
        require_trained(MultiCFRBot(str(MODEL), seed=0).cfr, min_iterations=rp6.MODEL_ITERATION)
    from nutsfinder.bots.multiway.equity import warm_up

    warm_up()


def _task(args: tuple) -> tuple:
    field, arm, master_seed, s, room, blocks = args
    t0 = time.perf_counter()
    rec = play_room(field, arm, master_seed, s, room, blocks)
    return (field, arm, s, room), rec, time.perf_counter() - t0, os.getpid(), rp6.process_memory_mb()


def build_tasks(n_sessions: dict[str, int], master_seed: int, blocks: int,
                arms: dict[str, tuple[str, ...]] | None = None) -> list[tuple]:
    """Every (field, arm, seed, s, room, blocks); session-major so a crash leaves whole sessions."""
    arms = arms or ARMS
    return [(field, arm, master_seed, s, room, blocks)
            for field, n in n_sessions.items() for s in range(n) for arm in arms[field] for room in range(N_SEATS)]


class _Interrupted(Exception):
    """Raised by the ``stop_after`` test hook to simulate a crash."""


def play_all(tasks: list[tuple], processes: int, initargs: tuple, partial: Path, done: dict,
             stop_after: int | None = None) -> tuple[dict, list]:
    todo = [t for t in tasks if (t[0], t[1], t[3], t[4]) not in done]
    mem: list = []
    if done:
        print(f"resuming: {len(done)} rooms already done, {len(todo)} to play", flush=True)
    if not todo:
        return done, mem
    if processes <= 1:
        results = map(_task, todo)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(min(processes, len(todo)), initializer=_init_worker, initargs=initargs)
        results = pool.imap_unordered(_task, todo, chunksize=1)
    start = time.perf_counter()
    try:
        for k, (key, rec, sec, pid, m) in enumerate(results, 1):
            reb._append_line(partial, {"key": list(key), "seconds": sec, "pid": pid, "memory_mb": m, "session": rec})
            done[key] = (rec, sec)
            mem.append((pid, m))
            mem.append((os.getpid(), rp6.process_memory_mb()))
            if k % 50 == 0 or k == len(todo):
                print(f"[{(time.perf_counter() - start) / 60:.1f} min] {k}/{len(todo)} rooms", flush=True)
            if stop_after is not None and k >= stop_after:
                raise _Interrupted
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return done, mem


def official_guard(results: Path, partial: Path, resume: bool) -> None:
    """Once only: refuse a second official run; allow resuming the same partial one."""
    finished = sorted(p.name for p in results.glob("*_6max_*")
                      if p.suffix in (".json", ".png") and "_pilot_" not in p.name and "_smoke_" not in p.name)
    if finished and not resume:
        sys.exit(f"official 6-max results already exist ({', '.join(finished[:3])}...): the run happens once")
    if partial.exists() and not resume:
        sys.exit(f"{partial.name} exists: an official run was started; pass --resume-partial to finish it")
    if resume and not partial.exists():
        sys.exit(f"--resume-partial: {partial.name} not found")


# --- Statistics ---------------------------------------------------------------------------------------


stats_dict = reb.stats_dict


def tier_verdict(d: dict) -> str:
    """The pre-registered rule on the two-sided 95% CI of D: "beats" / "loses" / "no detectable edge".

    No variation (SD 0) is never significant.
    """
    if d["no_variation"] or d["p_value"] is None or math.isnan(d["p_value"]):
        return "no detectable edge"
    lo, hi = d["ci95_bb100"]
    if lo > 0 and d["mean_bb100"] > 0:
        return "beats"
    if hi < 0 and d["mean_bb100"] < 0:
        return "loses"
    return "no detectable edge"


def sessions_needed(sd: float, edge: float) -> int | None:
    """Sessions for 80% power at alpha 0.05 to detect ``edge`` (bb/100) with per-session SD ``sd``."""
    if not edge or not sd or math.isnan(edge) or math.isnan(sd):
        return None
    return required_units(sd, edge, alpha=ALPHA, power=TARGET_POWER)


def group_sessions(done: dict, field: str, arm: str, n: int) -> list[list[dict]]:
    """Per session 0..n-1 the 6 room records (ordered by room); ValueError if any is missing."""
    by: dict[int, dict[int, dict]] = {}
    for (f, a, s, room), (rec, _sec) in done.items():
        if f == field and a == arm:
            by.setdefault(s, {})[room] = rec
    if sorted(by) != list(range(n)):
        raise ValueError(f"{field}/{arm}: sessions must be 0..{n - 1}, got {sorted(by)[:5]}...")
    out = []
    for s in range(n):
        if sorted(by[s]) != list(range(N_SEATS)):
            raise ValueError(f"{field}/{arm} session {s}: rooms {sorted(by[s])} (need all {N_SEATS})")
        out.append([by[s][r] for r in range(N_SEATS)])
    return out


def session_total_bb(rooms: list[dict]) -> float:
    return math.fsum(x for r in rooms for x in r["nets"]) / BB


def block_totals_bb(rooms: list[dict]) -> list[float]:
    """Per block: the test seat's net summed over the 6 rooms (bb)."""
    return [math.fsum(r["nets"][b] for r in rooms) / BB for b in range(len(rooms[0]["nets"]))]


def position_values(rooms: list[dict]) -> dict[str, float]:
    """Test seat's bb per position over the session (each position has exactly ``blocks`` hands)."""
    acc = dict.fromkeys(POSITIONS, 0.0)
    for r in rooms:
        for b, x in enumerate(r["nets"]):
            acc[position_names(N_SEATS, b % N_SEATS)[r["room"]]] += x / BB
    return acc


def street_values(rooms: list[dict]) -> dict[str, tuple[int, float]]:
    acc = {k: [0, 0.0] for k in END_STREETS}
    code = {k[0].upper(): k for k in END_STREETS}
    for r in rooms:
        for ch, x in zip(r["streets"], r["nets"]):
            acc[code[ch]][0] += 1
            acc[code[ch]][1] += x / BB
    return {k: (v[0], v[1]) for k, v in acc.items()}


def integrity(done: dict, n_sessions: dict[str, int], master_seed: int, blocks: int) -> dict:
    """Rotation and pairing checks: layouts, every record filed under its own key, the seed digest
    recomputed from (master seed, field, s), and the dealt cards (seed, button, hole cards, board)
    identical in every room and arm of a session."""
    problems: list[str] = []
    for key, (rec, _sec) in done.items():
        if (rec["field"], rec["arm"], rec["index"], rec["room"]) != tuple(key):
            problems.append(f"record filed under {key} is {rec['field']}/{rec['arm']}/s{rec['index']}/room{rec['room']}")
    for field, n in n_sessions.items():
        fseed = field_seed(master_seed, field)
        for s in range(n):
            digests = set()
            cards = set()
            want = seeds_digest(block_seeds(session_seed(fseed, s), blocks))
            field_seats = None
            for arm in ARMS[field]:
                rooms = [done[(field, arm, s, r)][0] for r in range(N_SEATS)]
                cards |= {rm.get("cards_digest") for rm in rooms}
                if any(rm["seeds_digest"] != want for rm in rooms):
                    problems.append(f"{field}/{arm}/s{s}: seed digest is not the one of (master seed, s)")
                expected = Counter((arm,) + FIELDS[field])
                for seat in range(N_SEATS):
                    if Counter(rm["layout"][seat] for rm in rooms) != expected:
                        problems.append(f"{field}/{arm}/s{s}: seat {seat} does not see every bot exactly once")
                for rm in rooms:
                    if rm["layout"][rm["room"]] != arm:
                        problems.append(f"{field}/{arm}/s{s}/room{rm['room']}: test bot not in its seat")
                    digests.add(rm["seeds_digest"])
                fs = [[k for j, k in enumerate(rm["layout"]) if j != rm["room"]] for rm in rooms]
                if field_seats is None:
                    field_seats = fs
                elif fs != field_seats:
                    problems.append(f"{field}/s{s}: field seating differs between arms")
            if len(digests) != 1:
                problems.append(f"{field}/s{s}: deals/reseeds differ between rooms or arms")
            if len(cards) != 1 or None in cards:
                problems.append(f"{field}/s{s}: dealt cards differ between rooms or arms")
    return {"ok": not problems, "problems": problems[:20], "n_problems": len(problems)}


def _fold_pool(sessions: list[list[dict]]) -> dict[str, dict]:
    tot = {st: [0, 0, 0] for st in STREETS}
    for rooms in sessions:
        for r in rooms:
            for st, c in r["folds"].items():
                for i in range(3):
                    tot[st][i] += c[i]
    return {st: {"faced_bet": c[0], "fold_to_bet_rate": c[1] / c[0] if c[0] else None, "free_folds": c[2]}
            for st, c in tot.items()}


def leaks(folds: dict[str, dict], name: str) -> list[str]:
    """The project's leak rules (nutsfinder.eval.breakdown.leaks) on pooled counts."""
    notes = []
    free = sum(v["free_folds"] for v in folds.values())
    if free:
        notes.append(f"{name} folded {free:,} times when checking was free")
    for st, v in folds.items():
        n, rate = v["faced_bet"], v["fold_to_bet_rate"]
        if n >= 100 and rate is not None and rate > 0.5:
            notes.append(f"{name} folds {rate:.0%} to {st} bets ({n:,} times faced): very exploitable by betting")
        if n >= 100 and rate == 0:
            notes.append(f"{name} never folds to {st} bets ({n:,} times faced): bluffing it is pointless")
    return notes


def analyse_arm(sessions: list[list[dict]], blocks: int) -> dict:
    hps = N_SEATS * blocks
    totals = [session_total_bb(rm) for rm in sessions]
    hands = [x / BB for rm in sessions for r in rm for x in r["nets"]]
    m = math.fsum(hands) / len(hands)
    sd_hand = math.sqrt(math.fsum((x - m) ** 2 for x in hands) / (len(hands) - 1)) if len(hands) > 1 else 0.0
    d = stats_dict(totals, hps)
    d["sd_bb_per_hand"] = sd_hand
    d["sd_eff_bb_per_hand"] = d["sd_bb100_per_session"] / 100 * math.sqrt(hps)  # per-hand-equivalent SD of the unit
    pos = [position_values(rm) for rm in sessions]
    st = [street_values(rm) for rm in sessions]
    folds = _fold_pool(sessions)
    n_hands = len(hands)
    modes: Counter[str] = Counter()
    field_net: dict[str, list[float]] = {}
    for rm in sessions:
        acc: Counter[str] = Counter()
        for r in rm:
            modes.update(r["modes"])
            acc.update(r["field_net"])
        for k, v in acc.items():
            field_net.setdefault(k, []).append(v / BB)
    total_modes = sum(modes.values())
    blueprint = sum(v for k, v in modes.items() if k.startswith("hu-blueprint"))
    return {
        "overall": d,
        "position": {p: stats_dict([x[p] for x in pos], blocks) for p in POSITIONS},
        "street": {k: {**stats_dict([x[k][1] for x in st], hps),
                       "share": sum(x[k][0] for x in st) / n_hands} for k in END_STREETS},
        "folds": folds,
        "vpip": sum(r["vpip"] for rm in sessions for r in rm) / n_hands,
        "pfr": sum(r["pfr"] for rm in sessions for r in rm) / n_hands,
        "field_net_bb100_per_bot_hand": {
            k: stats_dict(v, hps * field_count(k, sessions)) for k, v in sorted(field_net.items())},
        "modes": dict(modes),
        "blueprint_decision_share": blueprint / total_modes if total_modes else None,
    }


def field_count(kind: str, sessions: list[list[dict]]) -> int:
    """How many field seats hold ``kind`` (each plays every test-seat hand)."""
    r = sessions[0][0]
    return sum(1 for j, k in enumerate(r["layout"]) if j != r["room"] and k == kind)


def analyse_pair(x: list[list[dict]], y: list[list[dict]], blocks: int) -> dict:
    hps = N_SEATS * blocks
    d = stats_dict([session_total_bb(a) - session_total_bb(b) for a, b in zip(x, y)], hps)
    d["verdict"] = tier_verdict(d)
    d["sessions_needed_observed"] = sessions_needed(d["sd_bb100_per_session"], d["mean_bb100"])
    px, py = [position_values(rm) for rm in x], [position_values(rm) for rm in y]
    sx, sy = [street_values(rm) for rm in x], [street_values(rm) for rm in y]
    per_hand = [(a - b) / BB for ra, rb in zip(x, y) for ka, kb in zip(ra, rb) for a, b in zip(ka["nets"], kb["nets"])]
    identical = sum(1 for v in per_hand if v == 0) / len(per_hand)
    return {"difference": d,
            "position": {p: stats_dict([a[p] - b[p] for a, b in zip(px, py)], blocks) for p in POSITIONS},
            "street": {k: stats_dict([a[k][1] - b[k][1] for a, b in zip(sx, sy)], hps) for k in END_STREETS},
            "share_hands_equal_net": identical}


def analyse(done: dict, n_sessions: dict[str, int], blocks: int, master_seed: int) -> dict:
    arms: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[list[dict]]] = {}
    for field, n in n_sessions.items():
        for arm in ARMS[field]:
            groups[(field, arm)] = group_sessions(done, field, arm, n)
            arms[f"{field}|{arm}"] = analyse_arm(groups[(field, arm)], blocks)
    pairs = {}
    for f, a, b in (PRIMARY,) + SECONDARY_PAIRS:
        if (f, a) in groups and (f, b) in groups:
            pairs[f"{f}|{a}-{b}"] = analyse_pair(groups[(f, a)], groups[(f, b)], blocks)
    ig = integrity(done, n_sessions, master_seed, blocks)
    pk = f"{PRIMARY[0]}|{PRIMARY[1]}-{PRIMARY[2]}"
    verdict = overall_verdict(pairs[pk]["difference"], ig["ok"]) if pk in pairs else "primary not played"
    n_secondary = len(pairs) - (pk in pairs)
    return {"arms": arms, "pairs": pairs, "integrity": ig, "verdict": verdict,
            "multiple_testing": f"one primary test (alpha {ALPHA}, two-sided); {n_secondary} secondary paired "
                                f"differences and {len(arms)} absolute win rates are descriptive (no correction; "
                                f"expect about 1 in 20 to exclude 0 by chance)"}


def overall_verdict(d: dict, integrity_ok: bool) -> str:
    if not integrity_ok:
        return "INVALID: an integrity check failed (see integrity); no conclusion"
    v = d["verdict"]
    who = f"MultiBayesBot vs MultiEquityBot in the {PRIMARY[0]} field"
    if v == "beats":
        return f"{who}: beats (tier 3 counts in this field)"
    if v == "loses":
        return f"{who}: loses (tier 3 does not count in this field)"
    need = d.get("sessions_needed_observed")
    tail = f"; about {need:,} sessions would be needed to detect the observed difference" if need else ""
    return f"{who}: no detectable edge (tier 3 does not count in this field){tail}"


def timing(done: dict, processes: int) -> dict[str, float]:
    """Pool-wall ms per test-seat hand per (field, arm): sum of room seconds / hands / processes."""
    acc: dict[str, list[float]] = {}
    for (f, arm, _s, _r), (rec, sec) in done.items():
        a = acc.setdefault(f"{f}|{arm}", [0.0, 0])
        a[0] += sec
        a[1] += len(rec["nets"])
    return {k: 1000 * v[0] / v[1] / processes for k, v in acc.items()}


# --- Output -----------------------------------------------------------------------------------------


def plot_unit(title: str, series: list[list[float]], per_hand: list[float], path: Path, ylabel: str) -> None:
    """Cumulative bb (mean over sessions, 95% CI band across sessions), window rates, histograms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = z_critical(0.95)
    n, blocks = len(series), len(series[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (ax1, ax2), (ax3, ax4) = axes
    cums = []
    for row in series:
        tot, c = 0.0, []
        for v in row:
            tot += v
            c.append(tot)
        cums.append(c)
    mu, lo, hi = [], [], []
    for i in range(blocks):
        col = [c[i] for c in cums]
        m = math.fsum(col) / n
        sd = math.sqrt(math.fsum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        mu.append(m)
        lo.append(m - z * sd / math.sqrt(n))
        hi.append(m + z * sd / math.sqrt(n))
    hx = [N_SEATS * (i + 1) for i in range(blocks)]
    ax1.plot(hx, mu, lw=1, label="mean over sessions")
    ax1.fill_between(hx, lo, hi, alpha=0.2, label="95% CI (across sessions)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("test-seat hands in session (6 rooms per block)")
    ax1.set_ylabel(f"cumulative bb, {ylabel}")
    ax1.set_title(f"{title}: {n} sessions x {N_SEATS * blocks} hands")
    ax1.legend(fontsize=8)

    wins = [(lo_, min(lo_ + WINDOW_BLOCKS, blocks)) for lo_ in range(0, blocks, WINDOW_BLOCKS)]
    xm, wm, wl, wh = [], [], [], []
    for a, b in wins:
        vals = [math.fsum(row[a:b]) * 100 / (N_SEATS * (b - a)) for row in series]
        m = math.fsum(vals) / n
        sd = math.sqrt(math.fsum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        xm.append(N_SEATS * (a + b) / 2)
        wm.append(m)
        wl.append(m - z * sd / math.sqrt(n))
        wh.append(m + z * sd / math.sqrt(n))
    ax2.plot(xm, wm, "o-", ms=3)
    ax2.fill_between(xm, wl, wh, alpha=0.2, label="95% CI per window (pointwise)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel(f"test-seat hand index (windows of {WINDOW_BLOCKS} blocks)")
    ax2.set_ylabel(f"bb/100, {ylabel}")
    ax2.set_title("win rate by window (learning curve)")
    ax2.legend(fontsize=8)

    per_session = [math.fsum(row) * 100 / (N_SEATS * blocks) for row in series]
    m = math.fsum(per_session) / n
    sd = math.sqrt(math.fsum((v - m) ** 2 for v in per_session) / (n - 1)) if n > 1 else 0.0
    ax3.hist(per_session, bins=min(50, max(10, n // 3)))
    ax3.axvline(0, color="k", lw=0.5)
    ax3.axvline(m, color="C3", lw=1)
    ax3.set_xlabel(f"bb/100 per session, {ylabel}")
    ax3.set_ylabel("sessions")
    ax3.set_title(f"per session: mean {m:+.1f}, SD {sd:.1f}, 95% CI [{m - z * sd / math.sqrt(n):+.1f}, "
                  f"{m + z * sd / math.sqrt(n):+.1f}]")

    ax4.hist(per_hand, bins=80)
    ax4.set_yscale("log")
    ax4.set_xlabel(f"bb per hand, {ylabel}")
    ax4.set_ylabel("count (log scale)")
    ax4.set_title("per-hand results (descriptive)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_outputs(done: dict, analysis: dict, n_sessions: dict[str, int], stem_prefix: str, meta: dict,
                 write: bool = True) -> list[str]:
    """Per arm and per paired difference: JSON (per-session totals, per-hand nets) + PNG."""
    files = []
    groups = {}
    for field, n in n_sessions.items():
        for arm in ARMS[field]:
            groups[(field, arm)] = group_sessions(done, field, arm, n)
    for (field, arm), sessions in groups.items():
        stem = f"{stem_prefix}{field}_{arm}_vs_field_sessions"
        if write:
            series = [block_totals_bb(rm) for rm in sessions]
            body = {**meta, "field": field, "field_order_base": list(FIELDS[field]), "test_bot": arm,
                    "unit": "session", "seats_rotated": True, "duplicate": "6-room seat rotation per deal",
                    "fresh_bots_per_session_and_room": True,
                    "session_seeds": [session_seed(field_seed(meta["master_seed"], field), s) for s in range(len(sessions))],
                    "analysis": analysis["arms"][f"{field}|{arm}"],
                    "session_bb100": [math.fsum(r) * 100 / (N_SEATS * len(r)) for r in series],
                    "block_totals_bb": series,
                    "hand_results_bb_by_room": [[[x / BB for x in r["nets"]] for r in rm] for rm in sessions]}
            (RESULTS / f"{stem}.json").write_text(json.dumps(_clean(body), default=float), encoding="utf-8")
            plot_unit(f"{arm} vs {field} field", series, [x / BB for rm in sessions for r in rm for x in r["nets"]],
                      RESULTS / f"{stem}.png", f"{arm}")
        files += [f"{stem}.json", f"{stem}.png"]
    for key in analysis["pairs"]:
        field, rest = key.split("|")
        a, b = rest.split("-")
        stem = f"{stem_prefix}{field}_{a}_minus_{b}"
        if write:
            x, y = groups[(field, a)], groups[(field, b)]
            series = [[p - q for p, q in zip(block_totals_bb(ra), block_totals_bb(rb))] for ra, rb in zip(x, y)]
            per_hand = [(p - q) / BB for ra, rb in zip(x, y) for ka, kb in zip(ra, rb)
                        for p, q in zip(ka["nets"], kb["nets"])]
            body = {**meta, "field": field, "x": a, "y": b, "paired": "same deals, seats, field and reseeds",
                    "analysis": analysis["pairs"][key], "block_differences_bb": series}
            (RESULTS / f"{stem}.json").write_text(json.dumps(_clean(body), default=float), encoding="utf-8")
            plot_unit(f"{a} - {b} (paired), {field} field", series, per_hand, RESULTS / f"{stem}.png", f"{a} - {b}")
        files += [f"{stem}.json", f"{stem}.png"]
    return files


def _clean(x):
    from nutsfinder.eval.report import _clean as c

    return c(x)


def _ci(d: dict) -> str:
    lo, hi = d["ci95_bb100"]
    return f"[{lo:+.1f}, {hi:+.1f}]"


def results_markdown(analysis: dict, meta: dict) -> str:
    lines = [f"- date {meta['date']}; commit {meta['commit']}; master seed {meta['master_seed']}; "
             f"{meta['processes']} processes; run time {meta['wall_seconds']:.0f} s"
             + (f" ({meta['resumed_rooms']} rooms resumed from the partial log)" if meta.get("resumed_rooms") else ""),
             f"- sessions per arm {meta['n_sessions']}; {meta['blocks']} blocks x {N_SEATS} rooms = "
             f"{meta['blocks'] * N_SEATS:,} test-seat hands per session; seats rotated (6-room duplicate per deal)",
             f"- model (CFR arm): {meta['model']['description'] if meta.get('model') else 'not used'}"
             + (f"; sha256 {meta['model']['sha256']}" if meta.get("model") else ""), "",
             f"**Verdict: {analysis['verdict']}.** {analysis['multiple_testing']}. "
             "Field-dependent: these numbers hold for this field (and the 5 cyclic seatings of it) only.", "",
             "| comparison (same seat, same deals) | sessions | hands per arm | D bb/100 | 95% CI | SE | SD/session | "
             "p | verdict | sessions for observed D |", "|---|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for key, p in analysis["pairs"].items():
        d = p["difference"]
        tag = " **(primary)**" if key == f"{PRIMARY[0]}|{PRIMARY[1]}-{PRIMARY[2]}" else " (secondary)"
        need = f"{d['sessions_needed_observed']:,}" if d["sessions_needed_observed"] else "n/a"
        lines.append(f"| {key}{tag} | {d['n']} | {d['hands']:,} | {d['mean_bb100']:+.1f} | {_ci(d)} | "
                     f"{d['se_bb100']:.1f} | {d['sd_bb100_per_session']:.1f} | {d['p_value']:.3g} | {d['verdict']} | {need} |")
    lines += ["", "Absolute win rates against the field (descriptive):", "",
              "| test bot | bb/100 | 95% CI | SE | SD/session | SD/hand (bb) | VPIP | PFR | blueprint share |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key, a in analysis["arms"].items():
        o = a["overall"]
        bp = f"{a['blueprint_decision_share']:.1%}" if a["blueprint_decision_share"] is not None else "-"
        lines.append(f"| {key} | {o['mean_bb100']:+.1f} | {_ci(o)} | {o['se_bb100']:.1f} | "
                     f"{o['sd_bb100_per_session']:.1f} | {o['sd_bb_per_hand']:.1f} | {a['vpip']:.0%} | {a['pfr']:.0%} | {bp} |")
    for key, a in analysis["arms"].items():
        lines += ["", f"### {key}", ""]
        lines.append("- by position (bb/100 in that position, 95% CI): " + ", ".join(
            f"{p} {v['mean_bb100']:+.0f} {_ci(v)}" for p, v in a["position"].items()))
        lines.append("- by the test bot's exit street (fold street or showdown; contribution to bb/100; share of hands): " + ", ".join(
            f"{k} {v['mean_bb100']:+.1f} {_ci(v)} ({v['share']:.0%})" for k, v in a["street"].items()))
        lines.append("- fold to a bet: " + ", ".join(
            f"{st} {v['fold_to_bet_rate']:.0%} of {v['faced_bet']:,}" for st, v in a["folds"].items() if v["faced_bet"]))
        lines.append("- field bots' bb/100 per hand they played: " + ", ".join(
            f"{k} {v['mean_bb100']:+.0f}" for k, v in a["field_net_bb100_per_bot_hand"].items()))
        if a["modes"]:
            lines.append(f"- CFR decision modes: {a['modes']}")
        for leak in leaks(a["folds"], key.split("|")[1]):
            lines.append(f"  - leak: {leak}")
    for key, p in analysis["pairs"].items():
        lines += ["", f"- {key} by position: " + ", ".join(
            f"{q} {v['mean_bb100']:+.0f} {_ci(v)}" for q, v in p["position"].items()),
                  f"  - by exit street: " + ", ".join(
                      f"{k} {v['mean_bb100']:+.1f} {_ci(v)}" for k, v in p["street"].items())
                  + f"; hands with equal net in both arms {p['share_hands_equal_net']:.0%}"]
    ig = analysis["integrity"]
    lines += ["", f"- integrity (rotation: every bot in every seat once per deal; pairing: same deals/reseeds/field "
                  f"in every arm): {'ok' if ig['ok'] else 'FAILED ' + str(ig['problems'])}"]
    mem = meta.get("memory_mb") or {}
    lines += [f"- peak memory: worker working set max {rp6._fmt_mb(mem.get('max_worker_peak_working_set'))}; parent "
              f"{rp6._fmt_mb((mem.get('parent_peak') or {}).get('peak_working_set'))}", ""]
    return "\n".join(lines)


# --- Plan ---------------------------------------------------------------------------------------------


def required_sessions(sd: float) -> int:
    return rp6.sessions_for(sd, EDGE_BB100, margin=SD_MARGIN, floor=MIN_SESSIONS, power=TARGET_POWER,
                            alpha=ALPHA, step=SESSION_STEP)


def estimate_minutes(n_sessions: dict[str, int], ms: dict[str, float] | None = None,
                     blocks: int = BLOCKS_PER_SESSION) -> float | None:
    ms = MS_PER_HAND_WALL if ms is None else ms
    total = 0.0
    for field, n in n_sessions.items():
        for arm in ARMS[field]:
            v = ms.get(f"{field}|{arm}")
            if v is None:
                return None
            total += n * N_SEATS * blocks * v / 1000 / 60
    return total


def plan() -> None:
    print(f"master seed {MASTER_SEED} (official); pilot seed {PILOT_SEED}; {BLOCKS_PER_SESSION} blocks x {N_SEATS} rooms "
          f"= {BLOCKS_PER_SESSION * N_SEATS:,} test-seat hands per session; {PROCESSES} processes")
    for f, kinds in FIELDS.items():
        print(f"  field {f}: {', '.join(kinds)} (cyclic order rotated by s % {len(kinds)}); arms {', '.join(ARMS[f])}; "
              f"field seed (official) {field_seed(MASTER_SEED, f)}")
    print(f"primary: D = {PRIMARY[1]} - {PRIMARY[2]} in the {PRIMARY[0]} field, alpha {ALPHA} two-sided; "
          f"edge {EDGE_BB100} bb/100 at {TARGET_POWER:.0%} power, design SD = pilot SD x {SD_MARGIN}, floor {MIN_SESSIONS}")
    pk = f"{PRIMARY[0]}|{PRIMARY[1]}-{PRIMARY[2]}"
    if pk not in PILOT_SD or not N_SESSIONS:
        print("PILOT_SD / N_SESSIONS not fixed yet: run --pilot first")
        return
    sd = PILOT_SD[pk]
    n = N_SESSIONS[PRIMARY[0]]
    print(f"  pilot SD of D {sd:.1f} -> required {required_sessions(sd)}; N = {n}; "
          f"MDE {rp6.mde(SD_MARGIN * sd, n):.1f} bb/100 (design SD), {rp6.mde(sd, n):.1f} (pilot SD); "
          f"power at {EDGE_BB100:.0f} = {rp6.power_at(EDGE_BB100, SD_MARGIN * sd, n):.0%}, "
          f"at the pre-pilot {EDGE_BB100_PRE_PILOT:.0f} = {rp6.power_at(EDGE_BB100_PRE_PILOT, SD_MARGIN * sd, n):.0%}")
    for k, v in sorted(PILOT_SD.items()):
        if k != pk:
            played = all(x in ARMS[k.split("|")[0]] for x in k.split("|")[1].split("-"))
            print(f"  secondary {k}: pilot SD {v:.1f}, MDE at N {rp6.mde(SD_MARGIN * v, n):.1f} bb/100"
                  + ("" if played else " (not in the official run)"))
    minutes = estimate_minutes(N_SESSIONS)
    hands = sum(n * N_SEATS * BLOCKS_PER_SESSION * len(ARMS[f]) for f, n in N_SESSIONS.items())
    if minutes is not None:
        print(f"total {hands:,} test-seat hands; estimate {minutes:.0f} min at the pilot rate, x{THROTTLE_MARGIN} = "
              f"{minutes * THROTTLE_MARGIN:.0f} min (budget {BUDGET_HOURS * 60:.0f} min)")


# --- Main ---------------------------------------------------------------------------------------------


def design_check() -> None:
    """Official only: the design must be complete and powered as pre-registered."""
    pk = f"{PRIMARY[0]}|{PRIMARY[1]}-{PRIMARY[2]}"
    if set(N_SESSIONS) != set(FIELDS) or pk not in PILOT_SD:
        sys.exit("design not fixed: N_SESSIONS must cover every field and PILOT_SD the primary")
    need = required_sessions(PILOT_SD[pk])
    for f, n in N_SESSIONS.items():
        if n < need:
            sys.exit(f"design error: N_SESSIONS[{f}] = {n} < required {need}")
        if ROTATE_FIELD_ORDER and n % len(FIELDS[f]):
            sys.exit(f"design error: N_SESSIONS[{f}] = {n} is not a multiple of {len(FIELDS[f])} (field orders)")


def commit_hash(allow_dirty: bool) -> str:
    """run_exploit_back's check: a dirty tree (untracked files included) is refused unless ``allow_dirty``."""
    return reb.commit_hash(allow_dirty)


def check_model() -> dict:
    return rp6.check_model()


def needs_cfr() -> bool:
    return any("CFR" in arms for arms in ARMS.values())


def rebuild_outputs(partial: Path) -> dict:
    """Outputs-only recovery: re-analyse a COMPLETE partial log, rewrite its summary JSON and plots.

    Plays nothing and never touches the pre-registration (for a crash after the verdict was spliced
    below the marker but before the outputs were written). The log's own header defines the run.
    """
    header = json.loads(partial.read_text(encoding="utf-8").splitlines()[0])["header"]
    done = reb.read_partial(partial, {k: v for k, v in header.items() if k != "started"})
    if header["arms"] != {k: list(v) for k, v in ARMS.items()} or header["fields"] != {k: list(v) for k, v in FIELDS.items()}:
        sys.exit("the log's fields/arms differ from this script's: cannot rebuild")
    n_sessions, blocks, seed, started = header["n_sessions"], header["blocks"], header["master_seed"], header.get("started")
    missing = [t for t in build_tasks(n_sessions, seed, blocks) if (t[0], t[1], t[3], t[4]) not in done]
    if missing:
        sys.exit(f"{partial.name} is incomplete ({len(missing)} rooms missing): use --resume-partial instead")
    analysis = analyse(done, n_sessions, blocks, seed)
    meta = {"date": started, "commit": header["commit"], "official": header["official"], "kind": header["kind"],
            "master_seed": seed, "n_sessions": n_sessions, "blocks": blocks, "rebuilt_from": partial.name,
            "rebuilt_on": dt.date.today().isoformat()}
    stem_prefix = f"{started}_6max_" if header["official"] else f"{started}_6max_{header['kind']}_seed{seed}_"
    files = save_outputs(done, analysis, n_sessions, stem_prefix, meta, write=False)
    out = RESULTS / f"{stem_prefix}summary.json"
    out.write_text(json.dumps(_clean({"meta": meta, "analysis": analysis, "files": files}), indent=1, default=float),
                   encoding="utf-8")
    save_outputs(done, analysis, n_sessions, stem_prefix, meta)
    print(f"rebuilt {out.name} and {len(files)} files from {partial.name}; {MD.name} untouched")
    return {"meta": meta, "analysis": analysis}


def main(argv: list[str] | None = None, stop_after: int | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the design and budget, play nothing")
    ap.add_argument("--pilot", action="store_true", help=f"pilot on seed {PILOT_SEED} (NOT official)")
    ap.add_argument("--smoke", action="store_true", help="2 short sessions per arm on the pilot seed (plumbing)")
    ap.add_argument("--allow-dirty", action="store_true", help="pilot/smoke only; refused for the official run")
    ap.add_argument("--processes", type=int, default=PROCESSES)
    ap.add_argument("--resume-partial", action="store_true", help="finish the run recorded in the partial log")
    ap.add_argument("--rebuild-outputs", metavar="PARTIAL_LOG",
                    help="re-analyse a complete partial log; rewrite summary JSON and plots only (plays nothing)")
    args = ap.parse_args(argv)
    if args.plan:
        plan()
        return None
    if args.rebuild_outputs:
        return rebuild_outputs(Path(args.rebuild_outputs))
    official = not (args.pilot or args.smoke)
    kind = "official" if official else ("smoke" if args.smoke else "pilot")
    if official and args.allow_dirty:
        sys.exit("--allow-dirty is refused for the official run: commit first (the pre-registration must be committed)")
    if args.processes > 22:
        sys.exit("at most 22 processes on this machine")
    seed = MASTER_SEED if official else PILOT_SEED
    partial = RESULTS / (OFFICIAL_PARTIAL if official else f"phase7_6max_{kind}_seed{seed}_partial.jsonl")
    if official:
        design_check()
        official_guard(RESULTS, partial, args.resume_partial)
        rp6.splice_results(MD.read_text(encoding="utf-8"), "check", RESULTS_MARKER)  # marker present, nothing below
    elif args.resume_partial and not partial.exists():
        sys.exit(f"--resume-partial: {partial.name} not found")
    commit = commit_hash(not official)
    model = check_model() if needs_cfr() else None
    if official:
        n_sessions, blocks = dict(N_SESSIONS), BLOCKS_PER_SESSION
    elif args.smoke:
        n_sessions, blocks = {f: 2 for f in FIELDS}, SMOKE_BLOCKS
    else:
        n_sessions, blocks = {f: PILOT_SESSIONS for f in FIELDS}, BLOCKS_PER_SESSION
    header = {"official": official, "kind": kind, "master_seed": seed, "n_sessions": n_sessions, "blocks": blocks,
              "fields": {k: list(v) for k, v in FIELDS.items()}, "arms": {k: list(v) for k, v in ARMS.items()},
              "rotate_field_order": ROTATE_FIELD_ORDER, "commit": commit,
              "model_sha256": model["sha256"] if model else None}
    if args.resume_partial:
        done = reb.read_partial(partial, header)
        started = json.loads(partial.read_text(encoding="utf-8").splitlines()[0])["header"].get("started")
    else:
        started = dt.date.today().isoformat()
        reb.start_partial(partial, {**header, "started": started})
        done = {}
    tasks = build_tasks(n_sessions, seed, blocks)
    label = "OFFICIAL" if official else f"{kind.upper()} (NOT OFFICIAL)"
    print(f"{label}: master seed {seed}, {len(tasks)} rooms ({blocks} hands each), commit {commit}", flush=True)
    resumed = len(done)
    t0 = time.perf_counter()
    turn = str(rp6.TURN_TABLES) if needs_cfr() and rp6.TURN_TABLES.exists() else None
    done, mem = play_all(tasks, args.processes, (turn, needs_cfr()), partial, done, stop_after)
    wall = time.perf_counter() - t0
    analysis = analyse(done, n_sessions, blocks, seed)
    peaks = rp6.merge_peaks(mem)
    parent = peaks.pop(os.getpid(), None)
    memory = {"parent_peak": parent, "workers_peak": {str(k): v for k, v in peaks.items()},
              "max_worker_peak_working_set": max((v["peak_working_set"] for v in peaks.values()), default=None),
              "max_worker_peak_private": max((v["peak_private"] for v in peaks.values()), default=None)}
    meta = {"date": started, "finished": dt.date.today().isoformat(), "commit": commit, "official": official,
            "kind": kind, "master_seed": seed, "processes": args.processes, "wall_seconds": wall,
            "resumed_rooms": resumed, "model": model, "n_sessions": n_sessions, "blocks": blocks,
            "design": {"seats": N_SEATS, "stack_bb": 100, "blinds": [50, 100], "fields": {k: list(v) for k, v in FIELDS.items()},
                       "arms": {k: list(v) for k, v in ARMS.items()}, "rotate_field_order": ROTATE_FIELD_ORDER,
                       "primary": list(PRIMARY), "alpha": ALPHA, "edge_bb100": EDGE_BB100, "power": TARGET_POWER,
                       "sd_margin": SD_MARGIN, "unit": "session", "reseed": "every bot before every hand, per seat"},
            "env": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "memory_mb": memory, "ms_per_hand_wall": timing(done, args.processes), "partial_log": partial.name}
    stem_prefix = f"{started}_6max_" if official else f"{started}_6max_{kind}_seed{seed}_"
    md = results_markdown(analysis, meta)
    print(md)
    if official:
        MD.write_text(rp6.splice_results(MD.read_text(encoding="utf-8"), md, RESULTS_MARKER), encoding="utf-8")
        print(f"wrote {MD}")
    files = save_outputs(done, analysis, n_sessions, stem_prefix, meta, write=False)
    out = RESULTS / f"{stem_prefix}summary.json"
    out.write_text(json.dumps(_clean({"meta": meta, "analysis": analysis, "files": files}), indent=1, default=float),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)
    save_outputs(done, analysis, n_sessions, stem_prefix, meta)
    print(f"wrote {len(files)} files (rooms also in {partial.name})")
    return {"meta": meta, "analysis": analysis}


if __name__ == "__main__":
    main()
