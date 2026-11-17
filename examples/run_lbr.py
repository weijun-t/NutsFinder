r"""Phase 6 step 8 (exploitability half): Local Best Response against CFRBot, BayesBot and EquityBot.

Run:
    .venv\Scripts\python examples\run_lbr.py --plan                     # design, sample size and budget only
    .venv\Scripts\python examples\run_lbr.py --pilot --seed 999 --sessions 8 --pairs 25   # NOT official
    .venv\Scripts\python examples\run_lbr.py                            # the official run (once)
    .venv\Scripts\python examples\run_lbr.py --resume-partial           # finish a crashed official run

Once only, crash-safe. Every finished session (any rule, any opponent) is appended to
``results/phase6_lbr_official_partial.jsonl`` (pilots: ``..._pilot_seed<s>_<rules>_partial.jsonl``)
and flushed to disk, after a header line identifying the design, commit, blueprint and preflop
table. ``--resume-partial`` re-reads it (the header must match exactly, a torn last line is
ignored) and plays only the missing sessions; sessions are deterministic, so the final numbers
are those of an uninterrupted run (timings aside). An OFFICIAL run refuses to start if any
non-pilot ``results/phase6_lbr_*.json`` / ``.md`` exists (the run happens once), refuses a
partial log unless ``--resume-partial`` is given, refuses ``--allow-dirty``, and refuses to
build or use a preflop equity table whose SHA-256 is not PREFLOP_TABLE_SHA256.

Pre-registered verdict (primary rule "ev"): for X in {BayesBot, EquityBot}, the paired
session difference LBR(X) - LBR(CFRBot) has estimate > 0 and a 95% CI excluding 0;
CFRBot is "hard to exploit" (relative to them) iff both hold. The same is computed for the
"call" rule and labelled sensitivity; if the two rules disagree the output says so
explicitly ("holds only under the chosen rule").

What is measured. LBR (``nutsfinder.eval.lbr``) plays each opponent on duplicate
deals (every deal twice, LBR in each seat). Its win rate is a LOWER BOUND on the
opponent's exploitability, in mbb/g (milli-big-blinds per game) and bb/100. The
gating claim of Phase 6 (docs/phase6_research.md section 6, "secondary"):

    LBR(CFRBot) < LBR(BayesBot)   and   LBR(CFRBot) < LBR(EquityBot),

each tested as a PAIRED difference: every opponent sees exactly the same deals
(session s has seed ``lbr_session_seed(master, s)`` for every opponent, and the
same per-hand opponent reseeds), so a per-session difference cancels the card
luck the two runs share. Two-sided 95% CI of the session-level difference.

Design.
- Sessions of PAIRS_PER_SESSION duplicate pairs with FRESH instances (BayesBot
  learns from LBR during the session, as in Phase 5; its number is "LBR against
  a BayesBot that has had up to 2 * PAIRS_PER_SESSION hands to adapt").
  The unit of analysis is the session (valid for the learning bot); per-pair
  statistics are also reported (valid for the memoryless CFRBot / EquityBot).
- LBR: fold / check-call / 1/2 pot / pot / all-in by one-step EV with the
  check/call-down assumption, postflop and (PRIMARY rule, preflop="ev") preflop
  too, preflop equities from the precomputed table
  (nutsfinder.eval.exact_equity.preflop_vs_each, models/preflop_vs_combo.npz,
  built on first use, SHA-256 recorded). SENSITIVITY rule: preflop="call"
  (Lisy & Bowling: always check/call preflop). Both rules are played on the
  SAME deals (same session seeds), so their difference per opponent is paired
  too. The rule was chosen before the official run (user decision 2026-11-17);
  reporting the larger of the two would be a post-hoc selection.
- Policy queries: CFRBot exact (blueprint + enumerated translation branches);
  EquityBot / BayesBot: the bot's own decision code on a start-of-hand snapshot
  with exact turn/river equities and FLOP_RUNOUTS evenly spaced flop run-outs
  (instead of their 400-sample Monte Carlo), Bayes likelihood floor 0.02.
- The official blueprint is pinned by SHA-256 (as examples/run_phase6.py).

Pilot (seed 999, NOT official) and the sample size: see DESIGN NOTES below and
the pilot JSON/MD in results/ (``phase6_lbr_pilot_seed999_<date>``).

DESIGN NOTES
------------
Pilot 2, 2026-11-17, seed 999 (NOT official), 8 sessions x 25 pairs = 200 pairs = 400 hands
per opponent and rule (2,400 hands in all), 20 processes, machine otherwise idle, 59 s wall.
results/phase6_lbr_pilot_seed999_ev-call_2026-11-17.{json,md}. mbb/g [95% CI over sessions]:

    preflop  CFRBot              BayesBot            EquityBot           Bayes-CFR (SD/pair)  Equity-CFR (SD/pair)
    ev       +379 [-1413,+2171]  +3172 [+1381,+4964] +2255 [+78,+4432]   +2794 (40.1 bb)      +1876 (41.0 bb)
    call     +1726 [-394,+3847]  +2135 [-135,+4405]  +2362 [+1055,+3670] +409 (40.5 bb)       +636 (37.6 bb)

(The "call" row reproduces pilot 1 exactly: same seed, deterministic code.) Pilot 1 (same
seed and size, 400 hands per opponent and rule, busy shared machine) ran "call" and
"fold_call" (results/phase6_lbr_pilot_seed999_2026-11-17.* and ..._fold_call_2026-11-17.*);
fold_call (superseded): CFR +1911 [+80, +3742], Bayes +4298 [+2361, +6234],
Equity +2608 [+925, +4290]; Bayes-CFR +2386 (SD/pair 38.6 bb), Equity-CFR +696 (35.2 bb).

Sample size for the primary paired differences (preflop="ev"): SD per pair 40.1 / 41.0 bb
-> design SD = 1.2 x 41.0 = 49.2 bb per pair; smallest difference of interest 700 mbb/g
(1.4 bb per pair), 80% power, alpha 0.05 two-sided: n = 7.85 (49.2 / 1.4)^2 = 9,695 pairs
-> 100 sessions x 100 pairs = 10,000 pairs (20,000 hands) per opponent and rule; detectable
difference with that plan 689 mbb/g. Per-session SD of the difference ~ 41 sqrt(100) = 410 bb
= 2,050 mbb/g, SE of the mean over 100 sessions ~205 mbb/g.

Cost (pilot 2, worker-seconds per hand with 20 busy workers, burst): ev 0.16 (CFRBot),
0.58 (BayesBot), 0.55 (EquityBot); call 0.24 / 0.65 / 0.55. Both rules, 20,000 hands each:
~15.5 worker-hours -> ~0.8 h wall on 22 processes before thermal throttling; with the
laptop's sustained 7-9x throttling up to ~6.5 h (--plan uses SUSTAINED_THROTTLE = 8).
Pilot 1 (busy shared machine, before the equity caches) ran 10-20x slower per hand.

Why 100 pairs per session: BayesBot learns from LBR for 200 hands per session (the estimand
is stated with the number); 100 sessions keep the CLT comfortable for the session-level CIs.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # see examples/run_phase6.py
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from nutsfinder.bots import BayesBot, CFRBot, EquityBot  # noqa: E402
from nutsfinder.bots.cfr_bot import preload_turn_tables, require_trained  # noqa: E402
from nutsfinder.eval.exact_equity import ensure_preflop_table, load_preflop_table  # noqa: E402
from nutsfinder.eval.exact_equity import PREFLOP_TABLE_FILE  # noqa: E402
from nutsfinder.eval.lbr import PREFLOP_POLICIES, LBRHand, LBRSession, lbr_session  # noqa: E402
from nutsfinder.eval.match import describe_bot  # noqa: E402
from nutsfinder.eval.stats import required_units, summarize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase6 import merge_peaks, process_memory_mb  # noqa: E402  (the same memory probe)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PREFLOP_TABLE_SHA256 = "20068f56b1d58c4e5b38cab7fb981da6552d199d44dfb5d44792263448b0f2f2"
"""The preflop equity table the pilots used (100,000 boards per class); pinned for the official run."""
OFFICIAL_PARTIAL = "phase6_lbr_official_partial.jsonl"
ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
MODEL = ROOT / "models" / "holdem_official" / "latest.npz"
MODEL_ITERATION = 13_200_000
MODEL_SHA256 = "4539e9c7adbc62ace90b989ea257c1ce220639f18fd8a179f8d03b712ac024f3"
TURN_TABLES = ROOT / "models" / "turn_ehs.npz"

# --- Fixed design (official) ----------------------------------------------------------------
OFFICIAL_SEED = 202611188     # reserved for the official LBR run; the pilot used 999
PAIRS_PER_SESSION = 100       # 200 hands per session
N_SESSIONS = 100              # per opponent and rule (same sessions = same deals for all)
TARGET_DIFF_MBB = 700.0       # smallest paired difference of interest (DESIGN NOTES)
DESIGN_SD_PAIR_BB = 1.2 * 41.0  # pilot 2 (preflop="ev") SD per pair of the paired difference, +20%
PROCESSES = 22
FLOP_RUNOUTS = 200            # equity-family oracle: flop run-outs (of 1,176)
LBR_OPTIONS = {"bet_fractions": (0.5, 1.0), "all_in": True}
PREFLOP_RULES = ("ev", "call")  # primary first; "call" (Lisy & Bowling) is the sensitivity analysis
ORACLE_OPTIONS = {"EquityBot": {"flop_runouts": FLOP_RUNOUTS}, "BayesBot": {"flop_runouts": FLOP_RUNOUTS},
                  "CFRBot": {}}
OPPONENTS = ("CFRBot", "BayesBot", "EquityBot")
COMPARISONS = (("BayesBot", "CFRBot"), ("EquityBot", "CFRBot"))  # difference = first - second

# Pilot 2 measurements (seed 999), used by --plan: worker-seconds per hand, 20 busy workers, 1-minute burst.
PILOT_SEC_PER_HAND = {
    "ev": {"CFRBot": 0.158, "BayesBot": 0.583, "EquityBot": 0.554},
    "call": {"CFRBot": 0.241, "BayesBot": 0.648, "EquityBot": 0.548},
}
THROTTLE = 1.1                # wall-time factor for the pool tail
SUSTAINED_THROTTLE = 8.0      # the laptop slows 7-9x under sustained load (the pilot was a burst)

FACTORIES = {
    "CFRBot": functools.partial(CFRBot, model=str(MODEL)),
    "BayesBot": BayesBot,
    "EquityBot": EquityBot,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> tuple[str, str | None]:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return head, dirty


def commit_hash(allow_dirty: bool) -> str:
    """The commit to record; refuses a dirty or unknown tree for an official run unless allowed."""
    head, dirty = git_state()
    if dirty is None:
        if not allow_dirty:
            sys.exit("cannot read the git state (a pilot may pass --allow-dirty)")
        return "unknown"
    if dirty:
        if not allow_dirty:
            sys.exit("the work tree has uncommitted changes; commit first (a pilot may pass --allow-dirty):\n" + dirty)
        return head + " + uncommitted changes"
    return head


def check_model() -> dict:
    if not MODEL.exists():
        sys.exit(f"{MODEL} not found (models/ is gitignored: copy the official blueprint there)")
    sha = file_sha256(MODEL)
    if sha != MODEL_SHA256:
        sys.exit(f"{MODEL} has SHA-256 {sha}, expected {MODEL_SHA256}")
    bot = FACTORIES["CFRBot"]()
    require_trained(bot, min_iterations=MODEL_ITERATION)
    info = bot.blueprint.info()
    return {"path": str(MODEL.relative_to(ROOT)), "sha256": sha, "description": describe_bot(bot),
            "iteration": info["iteration"], "trained_at_commit": info["commit"],
            "abstraction_fingerprint": bot.blueprint.meta.get("abstraction")}


# --- Workers ------------------------------------------------------------------------------------


def _init_worker(turn_tables: str | None, preflop_table: str) -> None:
    load_preflop_table(Path(preflop_table))
    if turn_tables:
        preload_turn_tables(turn_tables)
    require_trained(FACTORIES["CFRBot"](), min_iterations=MODEL_ITERATION)


def _task(args: tuple) -> tuple:
    """One session; also how long it took and this worker's memory afterwards."""
    name, preflop, pairs, seed, s, lbr_options = args
    t0 = time.perf_counter()
    sess = lbr_session(FACTORIES[name], pairs, seed, s, lbr_options=lbr_options,
                       oracle_options=ORACLE_OPTIONS[name])
    return name, preflop, sess, time.perf_counter() - t0, os.getpid(), process_memory_mb()


# --- Crash-safe partial log ---------------------------------------------------------------------


def session_to_json(sess: LBRSession) -> dict:
    return dataclasses.asdict(sess)


def session_from_json(d: dict) -> LBRSession:
    hands = tuple(LBRHand(**{**h, "truth_mass": tuple(h["truth_mass"])}) for h in d["hands"])
    return LBRSession(**{**d, "hands": hands})


def _append_line(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def start_partial(path: Path, header: dict) -> None:
    """A new partial log: just the header line (replaces any old file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"header": header}, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_partial(path: Path, header: dict) -> dict[tuple[str, str, int], tuple[LBRSession, float]]:
    """The finished sessions of a partial log whose header equals ``header`` (exits otherwise).

    A torn last line (crash while writing) is ignored; a bad line anywhere else is an error.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        sys.exit(f"{path} is empty")
    try:
        old = json.loads(lines[0])["header"]
    except (ValueError, KeyError):
        sys.exit(f"{path}: no header line")
    old = {k: v for k, v in old.items() if k != "started"}  # the start date is not part of the design
    want = json.loads(json.dumps(header, default=float))
    if old != want:
        diff = {k: (old.get(k), want.get(k)) for k in set(old) | set(want) if old.get(k) != want.get(k)}
        sys.exit(f"{path} belongs to a different run (header differs: {diff}); refusing to resume")
    done: dict[tuple[str, str, int], tuple[LBRSession, float]] = {}
    for i, line in enumerate(lines[1:], 2):
        try:
            rec = json.loads(line)
        except ValueError:
            if i == len(lines):
                print(f"{path}: ignoring a torn last line", flush=True)
                break
            sys.exit(f"{path}: line {i} is corrupt")
        sess = session_from_json(rec["session"])
        done[(rec["rule"], rec["opponent"], sess.index)] = (sess, rec["seconds"])
    return done


class _Interrupted(Exception):
    """Raised by the ``stop_after`` test hook to simulate a crash."""


def play_all(tasks: list[tuple], processes: int, initargs: tuple | None, partial: Path,
             done: dict, stop_after: int | None = None) -> tuple[dict, list]:
    """Play the tasks not in ``done``, appending each to ``partial`` as it finishes.

    Returns ``(done, memory samples)``; ``done`` maps (rule, opponent, session) to
    (session, seconds) for every task. ``processes <= 1`` plays in this process (tests).
    """
    todo = [t for t in tasks if (t[1], t[0], t[4]) not in done]
    mem: list = []
    if done:
        print(f"resuming: {len(done)} sessions already done, {len(todo)} to play", flush=True)
    if not todo:
        return done, mem
    if processes <= 1:
        results = map(_task, todo)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(min(processes, len(todo)), initializer=_init_worker, initargs=initargs)
        results = pool.imap_unordered(_task, todo, chunksize=1)
    try:
        for k, (name, rule, sess, sec, pid, m) in enumerate(results, 1):
            _append_line(partial, {"rule": rule, "opponent": name, "seconds": sec, "pid": pid, "memory_mb": m,
                                   "session": session_to_json(sess)})
            done[(rule, name, sess.index)] = (sess, sec)
            mem.append((pid, m))
            mem.append((os.getpid(), process_memory_mb()))
            print(f"[{k}/{len(todo)}] {rule} {name} session {sess.index}: {sess.mbb_per_game:+.0f} mbb/g, "
                  f"{sec:.0f} s ({sec / len(sess.hands):.2f} s/hand)", flush=True)
            if stop_after is not None and k >= stop_after:
                raise _Interrupted
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return done, mem


def official_guard(results: Path, partial: Path, resume: bool) -> None:
    """Once only: refuse a second complete official run; allow resuming the same partial one."""
    finished = sorted(p.name for p in results.glob("phase6_lbr_*")
                      if p.suffix in (".json", ".md") and "_pilot_" not in p.name)
    if finished:
        sys.exit(f"official LBR results already exist ({', '.join(finished)}): the official run happens once")
    if partial.exists() and not resume:
        sys.exit(f"{partial.name} exists: an official run was started; pass --resume-partial to finish it")
    if resume and not partial.exists():
        sys.exit(f"--resume-partial: {partial.name} not found")


def check_preflop_table(official: bool, processes: int) -> dict:
    """Official: the pinned table must exist with the pinned hash (never built here). Pilot: may build."""
    if official:
        if not PREFLOP_TABLE_FILE.exists():
            sys.exit(f"{PREFLOP_TABLE_FILE} not found: the official run never builds the preflop table "
                     "(build it with a pilot run first; its SHA-256 must be PREFLOP_TABLE_SHA256)")
        sha = file_sha256(PREFLOP_TABLE_FILE)
        if sha != PREFLOP_TABLE_SHA256:
            sys.exit(f"{PREFLOP_TABLE_FILE} has SHA-256 {sha}, expected {PREFLOP_TABLE_SHA256}")
        return load_preflop_table(PREFLOP_TABLE_FILE)
    info = ensure_preflop_table(processes=processes)
    if info["sha256"] != PREFLOP_TABLE_SHA256:
        print(f"WARNING: preflop table SHA-256 {info['sha256']} is not the pinned {PREFLOP_TABLE_SHA256}", flush=True)
    return info


# --- Analysis -------------------------------------------------------------------------------------


def _stats(values: list[float], hands_per_unit: int) -> dict:
    s = summarize(values, hands_per_unit=hands_per_unit)
    lo, hi = s.ci_bb100
    return {"n": s.n, "mean_bb100": s.mean_bb100, "mean_mbb_g": 10 * s.mean_bb100, "se_bb100": s.se_bb100,
            "ci_bb100": [lo, hi], "ci_mbb_g": [10 * lo, 10 * hi], "sd_per_unit_bb": s.sd, "p_value": s.p_value}


def analyse(sessions: dict[str, list[LBRSession]], pairs: int) -> dict:
    out: dict = {"opponents": {}, "differences": {}}
    hands_per_session = 2 * pairs
    for name, ss in sessions.items():
        totals = [s.total_bb for s in ss]
        pair_results = [p for s in ss for p in s.pair_results]
        hands = [h for s in ss for h in s.hands]
        tm = [m for h in hands for m in h.truth_mass]
        by_seat = {}
        for label, pred in (("lbr_on_button", lambda h: h.lbr_seat == h.button),
                            ("lbr_big_blind", lambda h: h.lbr_seat != h.button)):
            vals = [h.net_bb for h in hands if pred(h)]
            by_seat[label] = 1000 * sum(vals) / len(vals) if vals else None
        out["opponents"][name] = {
            "sessions": _stats(totals, hands_per_session),
            "pairs_naive": _stats(pair_results, 2),
            "hands": len(hands),
            "sd_per_hand_bb": float(summarize([h.net_bb for h in hands]).sd),
            "mbb_g_by_position": by_seat,
            "worker_seconds_per_hand": sum(h.seconds for h in hands) / len(hands),
            "queries": sum(s.queries for s in ss), "cache_hits": sum(s.cache_hits for s in ss),
            "query_seconds": sum(s.query_seconds for s in ss),
            "belief_mass_on_truth_median": (sorted(tm)[len(tm) // 2] if tm else None),
            "belief_truth_below_1e-6": (sum(m < 1e-6 for m in tm) / len(tm) if tm else None),
            "opponent_description": ss[0].opponent_desc, "lbr_description": ss[0].lbr_desc,
        }
    for a, b in COMPARISONS:
        if a in sessions and b in sessions:
            sa = {s.index: s for s in sessions[a]}
            sb = {s.index: s for s in sessions[b]}
            common = sorted(set(sa) & set(sb))
            diffs = [sa[i].total_bb - sb[i].total_bb for i in common]
            pair_diffs = [x - y for i in common for x, y in zip(sa[i].pair_results, sb[i].pair_results)]
            st = _stats(diffs, hands_per_session) if len(diffs) >= 2 else None
            pst = _stats(pair_diffs, 2) if len(pair_diffs) >= 2 else None
            entry = {"sessions": st, "pairs_naive": pst}
            if st is not None and st["mean_bb100"] != 0:
                sd_session_bb100 = st["sd_per_unit_bb"] / hands_per_session * 100
                entry["sd_session_bb100"] = sd_session_bb100
                entry["sessions_for_observed_diff_80pct"] = required_units(sd_session_bb100, st["mean_bb100"])
                entry["sessions_for_half_observed_diff_80pct"] = required_units(sd_session_bb100,
                                                                                st["mean_bb100"] / 2)
            out["differences"][f"{a} - {b}"] = entry
    return out


def policy_differences(by_rule: dict[str, dict[str, list[LBRSession]]], pairs: int) -> dict:
    """Per opponent, the paired session difference between preflop rules (same deals): rule - primary."""
    out = {}
    rules = list(by_rule)
    for other in rules[1:]:
        for name in by_rule[rules[0]]:
            if name not in by_rule[other]:
                continue
            sa = {s.index: s for s in by_rule[other][name]}
            sb = {s.index: s for s in by_rule[rules[0]][name]}
            diffs = [sa[i].total_bb - sb[i].total_bb for i in sorted(set(sa) & set(sb))]
            if len(diffs) >= 2:
                out[f"{name}: {other} - {rules[0]}"] = _stats(diffs, 2 * pairs)
    return out


def _table(res: dict) -> list[str]:
    lines = ["| opponent | mbb/g | 95% CI (mbb/g) | bb/100 | SD/session (bb) | SD/pair (bb) | SD/hand (bb) | "
             "s/hand | truth mass median |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["opponents"].items():
        s = r["sessions"]
        lo, hi = s["ci_mbb_g"]
        tm = r["belief_mass_on_truth_median"]
        lines.append(f"| {name} | {s['mean_mbb_g']:+.0f} | [{lo:+.0f}, {hi:+.0f}] | {s['mean_bb100']:+.1f} | "
                     f"{s['sd_per_unit_bb']:.1f} | {r['pairs_naive']['sd_per_unit_bb']:.1f} | "
                     f"{r['sd_per_hand_bb']:.2f} | {r['worker_seconds_per_hand']:.2f} | "
                     f"{'-' if tm is None else f'{tm:.3f}'} |")
    lines += ["", "Paired differences (same deals), first minus second:", "",
              "| difference | mbb/g | 95% CI (mbb/g) | SD/session (bb/100) | SD/pair (bb) | "
              "sessions for 80% power at observed / half |",
              "|---|---:|---|---:|---:|---|"]
    for name, d in res["differences"].items():
        s, p = d["sessions"], d["pairs_naive"]
        if s is None:
            continue
        lo, hi = s["ci_mbb_g"]
        lines.append(f"| {name} | {s['mean_mbb_g']:+.0f} | [{lo:+.0f}, {hi:+.0f}] | {d.get('sd_session_bb100', 0):.0f} | "
                     f"{p['sd_per_unit_bb']:.1f} | {d.get('sessions_for_observed_diff_80pct')} / "
                     f"{d.get('sessions_for_half_observed_diff_80pct')} |")
    return lines


def rule_verdict(res: dict) -> dict:
    """Pre-registered rule for one preflop rule's results: LBR(X) - LBR(CFRBot) > 0 with the
    session-level 95% CI excluding 0, for X in BayesBot and EquityBot; "hard to exploit" iff both."""
    out: dict = {"comparisons": {}}
    holds = []
    for a, b in COMPARISONS:
        d = res["differences"].get(f"{a} - {b}", {}).get("sessions")
        if d is None:
            out["comparisons"][f"{a} - {b}"] = None
            holds.append(None)
            continue
        ok = d["mean_mbb_g"] > 0 and d["ci_mbb_g"][0] > 0
        out["comparisons"][f"{a} - {b}"] = {"mean_mbb_g": d["mean_mbb_g"], "ci_mbb_g": d["ci_mbb_g"],
                                            "p_value": d["p_value"], "significant_positive": ok}
        holds.append(ok)
    out["hard_to_exploit"] = None if None in holds else all(holds)
    return out


def verdict(by_preflop: dict[str, dict], primary: str) -> dict:
    """The pre-registered verdict (primary rule) plus the sensitivity rules, flagged if they disagree."""
    per = {r: rule_verdict(res) for r, res in by_preflop.items()}
    main_v = per[primary]["hard_to_exploit"]
    others = {r: v for r, v in per.items() if r != primary}
    differs = any(v["hard_to_exploit"] != main_v for v in others.values())

    def word(v):
        return "incomplete" if v is None else ("HOLDS" if v else "does NOT hold")

    sentence = (f"Pre-registered verdict (preflop={primary!r}): 'CFRBot is harder to exploit than BayesBot and "
                f"EquityBot' {word(main_v)}.")
    for r, v in others.items():
        sentence += f" Sensitivity (preflop={r!r}): {word(v['hard_to_exploit'])}."
    if differs:
        if main_v:
            sentence += " The verdict holds only under the chosen rule."
        else:
            sentence += " The verdicts differ between rules; the pre-registered rule decides."
    return {"primary_rule": primary, "hard_to_exploit": main_v, "by_rule": per,
            "labels": {r: ("primary" if r == primary else "sensitivity") for r in per},
            "differs_between_rules": differs, "sentence": sentence}


def _fmt_mb(x) -> str:
    """Memory figures may be missing (probe unavailable): never crash the report on them."""
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.0f} MB"


def markdown(meta: dict, results: dict) -> str:
    rules = meta["preflop_rules"]
    lines = [f"# LBR (Local Best Response) {'PILOT (not official)' if meta['pilot'] else 'official run'}",
             "", f"- date {meta['date']}, commit {meta['commit']}, master seed {meta['master_seed']}",
             f"- {meta['sessions']} sessions x {meta['pairs_per_session']} duplicate pairs per opponent and preflop "
             f"rule (same deals for every opponent and rule), {meta['processes']} processes, "
             f"{meta['wall_seconds']:.0f} s wall",
             f"- model: {meta['model']['description']} (sha256 {meta['model']['sha256'][:12]})",
             f"- preflop equity table: {meta['preflop_table']['boards']} boards per class "
             f"(sha256 {meta['preflop_table']['sha256'][:12]})",
             f"- preflop rules: {', '.join(rules)} (primary: {rules[0]})",
             f"- environment: {meta.get('env')}; peak memory per worker: working set max "
             f"{_fmt_mb(meta.get('memory_mb', {}).get('max_worker_peak_working_set'))}, private max "
             f"{_fmt_mb(meta.get('memory_mb', {}).get('max_worker_peak_private'))}", "",
             "LBR's win rate is a lower bound on the opponent's exploitability. CI: 95%, normal, across sessions.", ""]
    v = results.get("verdict")
    if v:
        lines += [f"**{v['sentence']}**", ""]
        for r, rv in v["by_rule"].items():
            for name, c in rv["comparisons"].items():
                if c is not None:
                    lines.append(f"- {v['labels'][r]} ({r}): {name} = {c['mean_mbb_g']:+.0f} mbb/g "
                                 f"[{c['ci_mbb_g'][0]:+.0f}, {c['ci_mbb_g'][1]:+.0f}], "
                                 f"{'significant > 0' if c['significant_positive'] else 'not significant > 0'}")
        lines.append("")
    for rule, res in results["by_preflop"].items():
        tag = "primary" if rule == rules[0] else "sensitivity"
        lines += [f"## preflop = {rule} ({tag})", "", f"LBR: {meta['lbr'][rule]}", ""] + _table(res) + [""]
    if results["policy_differences"]:
        lines += ["## Preflop rule effect (same deals, per opponent)", "",
                  "| difference | mbb/g | 95% CI (mbb/g) |", "|---|---:|---|"]
        for name, st in results["policy_differences"].items():
            lo, hi = st["ci_mbb_g"]
            lines.append(f"| {name} | {st['mean_mbb_g']:+.0f} | [{lo:+.0f}, {hi:+.0f}] |")
    return "\n".join(lines) + "\n"


def detectable_mbb(pairs: int) -> float:
    """Smallest paired difference (mbb/g) detectable at 80% power, alpha 0.05 two-sided, design SD."""
    z = 1.959964 + 0.841621
    return z * DESIGN_SD_PAIR_BB / pairs ** 0.5 / 2 * 1000  # per pair = 2 hands


def plan() -> None:
    hands = 2 * PAIRS_PER_SESSION * N_SESSIONS
    need = required_units(DESIGN_SD_PAIR_BB, 2 * TARGET_DIFF_MBB / 1000)
    print(f"official seed {OFFICIAL_SEED}; LBR {LBR_OPTIONS}; preflop rules {PREFLOP_RULES} "
          f"(primary {PREFLOP_RULES[0]}); flop run-outs {FLOP_RUNOUTS}")
    print(f"{N_SESSIONS} sessions x {PAIRS_PER_SESSION} pairs ({2 * PAIRS_PER_SESSION} hands) per opponent and rule "
          f"= {hands:,} hands each; same deals for {', '.join(OPPONENTS)} and every rule")
    print(f"power (primary rule): paired difference {TARGET_DIFF_MBB:.0f} mbb/g, design SD {DESIGN_SD_PAIR_BB:.1f} "
          f"bb/pair -> {need:,} pairs needed at 80% (planned {N_SESSIONS * PAIRS_PER_SESSION:,}); "
          f"detectable at 80% with the plan: {detectable_mbb(N_SESSIONS * PAIRS_PER_SESSION):.0f} mbb/g")
    grand = 0.0
    for rule in PREFLOP_RULES:
        total = 0.0
        for name in OPPONENTS:
            rate = PILOT_SEC_PER_HAND[rule][name]
            sec = hands * rate
            total += sec
            print(f"  {rule:5s} {name:10s} {rate:.2f} s/hand/worker -> {sec / 3600:.1f} worker-hours")
        grand += total
        print(f"  {rule:5s} total {total / 3600:.1f} worker-hours, ~{total / PROCESSES * THROTTLE / 3600:.1f} h wall")
    wall = grand / PROCESSES * THROTTLE / 3600
    print(f"all rules: {grand / 3600:.1f} worker-hours; on {PROCESSES} processes x tail factor {THROTTLE}: "
          f"~{wall:.1f} h wall at pilot speed, up to ~{wall * SUSTAINED_THROTTLE:.1f} h with sustained "
          f"throttling x{SUSTAINED_THROTTLE:.0f}")


def main(argv: list[str] | None = None, stop_after: int | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the design and budget, play nothing")
    ap.add_argument("--pilot", action="store_true", help="unofficial run (requires --seed != the official seed)")
    ap.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    ap.add_argument("--sessions", type=int, default=N_SESSIONS)
    ap.add_argument("--pairs", type=int, default=PAIRS_PER_SESSION)
    ap.add_argument("--processes", type=int, default=PROCESSES)
    ap.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=list(OPPONENTS))
    ap.add_argument("--preflop", nargs="+", choices=list(PREFLOP_POLICIES), default=list(PREFLOP_RULES),
                    help="LBR preflop rules, primary first (pilot only; the official run uses PREFLOP_RULES)")
    ap.add_argument("--resume-partial", action="store_true",
                    help="continue the run recorded in the partial JSONL log (same design, same commit)")
    ap.add_argument("--allow-dirty", action="store_true", help="pilot only: run on a dirty tree (recorded)")
    args = ap.parse_args(argv)
    if args.plan:
        plan()
        return None
    official = not args.pilot
    if official and args.allow_dirty:
        sys.exit("--allow-dirty is refused for the official run: commit first")
    if official and (args.seed != OFFICIAL_SEED or args.sessions != N_SESSIONS or args.pairs != PAIRS_PER_SESSION
                     or tuple(args.opponents) != OPPONENTS or tuple(args.preflop) != PREFLOP_RULES):
        sys.exit("the official run uses the fixed design; use --pilot for anything else")
    if args.pilot and args.seed == OFFICIAL_SEED:
        sys.exit("a pilot must not use the official seed")
    rules = list(dict.fromkeys(args.preflop))
    partial = RESULTS / (OFFICIAL_PARTIAL if official else
                         f"phase6_lbr_pilot_seed{args.seed}_{'-'.join(rules)}_partial.jsonl")
    if official:
        official_guard(RESULTS, partial, args.resume_partial)
    elif args.resume_partial and not partial.exists():
        sys.exit(f"--resume-partial: {partial.name} not found")
    commit = commit_hash(args.allow_dirty or args.pilot)
    model = check_model()
    preflop_table = check_preflop_table(official, args.processes)
    turns = str(TURN_TABLES) if TURN_TABLES.exists() else None
    header = {"official": official, "master_seed": args.seed, "sessions": args.sessions, "pairs": args.pairs,
              "opponents": list(args.opponents), "preflop_rules": rules,
              "lbr_options": {k: list(v) if isinstance(v, tuple) else v for k, v in LBR_OPTIONS.items()},
              "oracle_options": ORACLE_OPTIONS, "commit": commit, "model_sha256": model["sha256"],
              "preflop_table_sha256": preflop_table["sha256"]}
    if args.resume_partial:
        done = read_partial(partial, header)
        started = json.loads(partial.read_text(encoding="utf-8").splitlines()[0])["header"]
    else:
        start_partial(partial, {**header, "started": dt.date.today().isoformat()})
        done = {}
    if args.resume_partial:
        header = {**header, "started": started.get("started")}
    # Heavy tasks first so the pool's tail is short.
    cost = {(r, n): PILOT_SEC_PER_HAND.get(r, PILOT_SEC_PER_HAND["ev"])[n] for r in rules for n in args.opponents}
    tasks = sorted(((name, rule, args.pairs, args.seed, s, {**LBR_OPTIONS, "preflop": rule})
                    for rule in rules for name in args.opponents for s in range(args.sessions)),
                   key=lambda t: -cost[(t[1], t[0])])
    resumed = len(done)
    start = time.perf_counter()
    done, mem = play_all(tasks, args.processes, (turns, preflop_table["path"]), partial, done, stop_after)
    wall = time.perf_counter() - start
    by_rule: dict[str, dict[str, list[LBRSession]]] = {r: {n: [] for n in args.opponents} for r in rules}
    for (rule, name, _), (sess, _sec) in done.items():
        by_rule[rule][name].append(sess)
    for sessions in by_rule.values():
        for ss in sessions.values():
            ss.sort(key=lambda s: s.index)
    by_preflop = {r: analyse(by_rule[r], args.pairs) for r in rules}
    results = {"primary": rules[0], "by_preflop": by_preflop,
               "policy_differences": policy_differences(by_rule, args.pairs),
               "verdict": verdict(by_preflop, rules[0])}
    peaks = merge_peaks(mem)
    parent = peaks.pop(os.getpid(), None)
    memory = {"parent_peak": parent, "workers_peak": {str(k): v for k, v in peaks.items()},
              "max_worker_peak_working_set": max((v["peak_working_set"] for v in peaks.values()), default=None),
              "max_worker_peak_private": max((v["peak_private"] for v in peaks.values()), default=None),
              "note": "this process's sessions only" + (f" ({resumed} resumed sessions not measured)" if resumed else "")}
    date = header.get("started") or dt.date.today().isoformat()
    meta = {"date": date, "finished": dt.date.today().isoformat(), "commit": commit, "pilot": args.pilot,
            "master_seed": args.seed, "sessions": args.sessions, "pairs_per_session": args.pairs,
            "processes": args.processes, "wall_seconds": wall, "resumed_sessions": resumed,
            "model": model, "preflop_rules": rules, "preflop_table": preflop_table,
            "preflop_table_pinned_sha256": PREFLOP_TABLE_SHA256,
            "lbr": {r: by_rule[r][args.opponents[0]][0].lbr_desc for r in rules},
            "lbr_options": header["lbr_options"], "oracle_options": ORACLE_OPTIONS,
            "env": {v: os.environ.get(v) for v in ENV_VARS}, "memory_mb": memory,
            "partial_log": partial.name, "python": sys.version.split()[0]}
    stem = f"phase6_lbr_pilot_seed{args.seed}_{'-'.join(rules)}_{date}" if args.pilot else f"phase6_lbr_{date}"
    per_session = {r: {n: [{"index": s.index, "seed": s.seed, "total_bb": s.total_bb,
                            "pair_results": s.pair_results, "seconds": s.seconds} for s in ss]
                       for n, ss in sessions.items()} for r, sessions in by_rule.items()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "results": results, "sessions": per_session},
                                                     indent=1, default=float))
    (RESULTS / f"{stem}.md").write_text(markdown(meta, results))
    print(markdown(meta, results))
    print(f"wrote {RESULTS / stem}.json / .md (sessions also in {partial.name})")
    return {"meta": meta, "results": results}


if __name__ == "__main__":
    main()
