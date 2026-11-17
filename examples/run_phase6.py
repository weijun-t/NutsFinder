r"""Phase 6, step 8: the head-to-head tier test. CFRBot (tier 4) against BayesBot (tier 3) and the rest.

Run:
    .venv\Scripts\python examples\run_phase6.py --plan       # design, power and wall-clock budget; plays nothing
    .venv\Scripts\python examples\run_phase6.py --pilot      # the pilot (seed 999, NOT official; dirty tree allowed)
    .venv\Scripts\python examples\run_phase6.py              # THE official run (once, clean tree, no re-runs)

The pre-registration is the top of ``results/phase6.md`` (committed before the
official run). The official run writes its numbers below the line
``RESULTS_MARKER`` of that file and refuses to run if anything is already
there: one official run, no re-runs.

Design (mirrors Phase 5, see ``examples/run_phase5.py`` and ``nutsfinder/eval/sessions.py``)
--------------------------------------------------------------------------------------------
- Unit = an independent *session* of ``PAIRS_PER_SESSION`` = 250 duplicate
  pairs (500 hands): every deal is played twice with the seats swapped, by
  FRESH bot instances (one copy of each bot per room, new objects every
  session). BayesBot therefore gets exactly its Phase 5 learning window
  (500 hands against a fresh opponent model). The estimand is "bb/100 over
  the first 500 hands against a fresh opponent model".
- Session s of every pairing has seed ``session_seed(MASTER_SEED, s)``
  (SHA-256 of (purpose, master seed, s)), so it has the same deals in every
  pairing (common random numbers), and within a session every bot is
  reseeded before every hand from the unit seeds (``duplicate_match``):
  CFRBot's constructor seed (default 0) is overwritten before its first
  decision, so it plays the same as with any other constructor seed
  (``tests/test_run_phase6.py`` checks it). The opponent X is bot B in both
  "CFRBot vs X" and "BayesBot vs X", so X gets the same seed stream too.
- PRIMARY (the only gating test of this script, rule 2 of PLAN.md):
  CFRBot vs BayesBot, two-sided 95% CI over sessions. "CFRBot beats the
  previous tier" iff the CI excludes 0 AND the mean is > 0
  (:func:`tier_verdict`).
- Secondary / descriptive: CFRBot vs EquityBot, PotOddsBot, CallStationBot,
  AlwaysRaiseBot, RandomBot; BayesBot vs the same five on the same session
  seeds (recomputed on the new master seed: Phase 5's BayesBot rows used
  seed 20261117 and other session counts, so they cannot be paired with
  these); and the PAIRED difference per session, (CFRBot vs X) minus
  (BayesBot vs X), with a CI across sessions.
- The exploitability criterion (LBR) is a separate secondary gating test
  run by ``examples/run_lbr.py`` (not this script).

Memory (an earlier 22-worker job was killed for low memory)
-----------------------------------------------------------
Measured on this machine (Windows, numpy with scipy-openblas, MAX_THREADS=24):
importing numpy *commits* ~785 MB of private memory per process (OpenBLAS
reserves a buffer per potential thread) although only ~30 MB is touched. 22
workers + the parent = ~18 GB of commit charge before a single hand, which is
what exhausts the commit limit. With ``OPENBLAS_NUM_THREADS=1`` the import
commits ~18 MB. This script sets it (and the OMP/MKL equivalents) before
anything imports numpy, so the spawned workers inherit it; no bot uses
multithreaded BLAS, and the numbers are identical either way (checked on a
pilot session). Further:

- sessions are fed with ``imap`` and consumed in order, one pairing at a
  time: the parent holds the hand logs of the current pairing only (needed
  for the position / street / fold breakdowns), then keeps per-hand nets;
- the blueprint (1.1 MB of float64 strategy) is loaded once per worker
  (``Blueprint.load`` is cached per process), not once per bot;
- every worker preloads the exact turn E[HS] tables (``models/turn_ehs.npz``,
  43 MB in memory): CardAbstraction needs a turn table for every turn
  decision, and computing them lazily costs ~8 ms per new turn board (4x
  slower CFRBot vs cheap opponents in a cold worker). Same tables either way
  (the trainer wrote them with the same function), so the same numbers;
- every session reports its worker's working set and private bytes; the run
  records the peak per worker and the parent's.

DESIGN NOTES
------------
Pilot 2026-11-17 on seed 999 (NOT official; table in results/phase6.md): 44 sessions of the primary
pairing (SD 73.1 bb/100 per session), 22 of every other pairing, 1,279 s on 22 processes. Design SD =
pilot SD x 1.2. Primary: edge +20 bb/100 at 80% power -> 151 -> N = 160 (82% power at 20, MDE 19.4).
Secondary: per opponent X, the largest requirement at 50 bb/100 among CFRBot vs X, BayesBot vs X and
their paired difference, floor 50: Equity 50, PotOdds 50, CallStation 120, AlwaysRaise 200, Random 130.
1,260 sessions = 630,000 hands; ~81 min at the pilot rate, budget ~130 min (x1.6). ``--plan`` prints it.
"""

from __future__ import annotations

import os

# Before ANY import that loads numpy (this module, and spawned workers that re-import it).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, CFRBot, EquityBot, PotOddsBot, RandomBot  # noqa: E402
from nutsfinder.bots.cfr_bot import preload_turn_tables, require_trained  # noqa: E402
from nutsfinder.eval.breakdown import end_street  # noqa: E402
from nutsfinder.eval.report import save_session_json, save_session_plot  # noqa: E402
from nutsfinder.eval.sessions import (  # noqa: E402
    Session, SessionMatchResult, combine_sessions, learning_gain, play_session, session_breakdown,
)
from nutsfinder.eval.stats import Summary, required_units, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PHASE6_MD = RESULTS / "phase6.md"
RESULTS_MARKER = "<!-- OFFICIAL RESULTS BELOW: written by examples/run_phase6.py, do not edit by hand -->"

# --- The model under test (pinned: the run refuses any other file) ------------------
MODEL = ROOT / "models" / "holdem_official" / "latest.npz"
MODEL_ITERATION = 13_200_000
MODEL_SHA256 = "4539e9c7adbc62ace90b989ea257c1ce220639f18fd8a179f8d03b712ac024f3"
MODEL_ABSTRACTION = "1cad42d6a4bed370ef4b525265f24940f624da8a5736f9dd12981562f2a0008f"
MODEL_COMMIT = "902376d"
TURN_TABLES = ROOT / "models" / "turn_ehs.npz"

# --- Fixed design (chosen before the official run from the pilot; see results/phase6.md) ----
# Never used before: results/ JSON master seeds are 999, 20261115-20261117; the preflop tables use
# 20261116 + class index and 20261116 + 100 * class + j (so 20261117..~20278000 is taken); the
# training seed was 7, the bucket-edge seed 61. Session seeds are hashed anyway.
MASTER_SEED = 202611186
PILOT_SEED = 999  # the pilot (NOT official; also Phase 4/5's pilot seed, never an official seed)
PAIRS_PER_SESSION = 250  # 500 hands per session, as Phase 5
BUCKET_PAIRS = 25
EARLY, LATE = (0, 50), (200, 250)  # learning gain: last 100 hands minus first 100 hands
PROCESSES = 22  # of 24 logical cores
ALPHA = 0.05
TARGET_POWER = 0.8
SD_MARGIN = 1.2  # inflate the pilot SDs by 20% (they are estimates, +/- ~10-15%)
MIN_SESSIONS = 50  # CLT floor and a stable SD estimate
PRIMARY_EDGE_BB100 = 20.0  # minimum detectable edge of the primary test (80% power)
SECONDARY_EDGE_BB100 = 50.0  # ... of the secondary pairings and paired differences

PRIMARY = ("CFRBot", "BayesBot")
OPPONENTS = ["EquityBot", "PotOddsBot", "CallStationBot", "AlwaysRaiseBot", "RandomBot"]
PAIRINGS = [PRIMARY] + [("CFRBot", x) for x in OPPONENTS] + [("BayesBot", x) for x in OPPONENTS]

# Pilot (seed 999, NOT official): per-session SD (bb/100, 250-pair sessions) of each pairing and of
# the paired difference (CFRBot vs X) - (BayesBot vs X); session wall seconds in a busy 22-process pool.
# Filled from `--pilot` output (results/<date>_phase6_pilot_seed999.json); see results/phase6.md.
PILOT_SESSIONS = {PRIMARY: 44, **{p: 22 for p in PAIRINGS if p != PRIMARY}}
# Pilot, 2026-11-17, seed 999, 22 processes, commit 76c05bf + uncommitted run_phase6.py (NOT OFFICIAL).
PILOT_SD: dict[tuple[str, str], float] = {
    ("CFRBot", "BayesBot"): 73.1,  # 44 sessions; the research note's proxy was 78 (Bayes vs Equity)
    ("CFRBot", "EquityBot"): 87.7, ("CFRBot", "PotOddsBot"): 67.9, ("CFRBot", "CallStationBot"): 71.9,
    ("CFRBot", "AlwaysRaiseBot"): 119.0, ("CFRBot", "RandomBot"): 83.1,
    ("BayesBot", "EquityBot"): 68.4, ("BayesBot", "PotOddsBot"): 87.9, ("BayesBot", "CallStationBot"): 158.5,
    ("BayesBot", "AlwaysRaiseBot"): 209.0, ("BayesBot", "RandomBot"): 169.0,
}
PILOT_DIFF_SD: dict[str, float] = {  # SD per session of (CFRBot vs X) - (BayesBot vs X), 22 sessions
    "EquityBot": 80.5, "PotOddsBot": 94.2, "CallStationBot": 148.7, "AlwaysRaiseBot": 202.0, "RandomBot": 148.4,
}
MS_PER_HAND_WALL: dict[tuple[str, str], float] = {  # sum of session seconds / hands / 22 (pilot total 1,279 s)
    ("CFRBot", "BayesBot"): 10.84, ("CFRBot", "EquityBot"): 11.47, ("CFRBot", "PotOddsBot"): 7.77,
    ("CFRBot", "CallStationBot"): 0.45, ("CFRBot", "AlwaysRaiseBot"): 0.30, ("CFRBot", "RandomBot"): 0.10,
    ("BayesBot", "EquityBot"): 23.67, ("BayesBot", "PotOddsBot"): 20.13, ("BayesBot", "CallStationBot"): 11.72,
    ("BayesBot", "AlwaysRaiseBot"): 14.90, ("BayesBot", "RandomBot"): 2.16,
}
THROTTLE_MARGIN = 1.6  # Phase 5: the official run took 1.58x its pilot-based estimate
# N per opponent X = the largest of the requirements (design SD = 1.2 x pilot SD, 80% power, alpha 0.05)
# for CFRBot vs X, BayesBot vs X (edge 50) and their paired difference (edge 50); the same N for both
# rows so the difference is paired. Primary: edge 20 -> required 151 -> 160 sessions (82% power at 20).
N_SESSIONS: dict[tuple[str, str], int] = {
    ("CFRBot", "BayesBot"): 160,
    **{(a, x): n for x, n in {"EquityBot": 50, "PotOddsBot": 50, "CallStationBot": 120,
                              "AlwaysRaiseBot": 200, "RandomBot": 130}.items() for a in ("CFRBot", "BayesBot")},
}

CFR_FACTORY = functools.partial(CFRBot, model=str(MODEL))
BOTS = {
    "CFRBot": CFR_FACTORY,
    "BayesBot": BayesBot,
    "EquityBot": EquityBot,
    "PotOddsBot": PotOddsBot,
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "RandomBot": RandomBot,
}
SHORT = {k: k.removesuffix("Bot") for k in BOTS}


# --- Pure helpers (tested in tests/test_run_phase6.py) ----------------------------


def sessions_for(sd: float, edge: float, margin: float = SD_MARGIN, floor: int = MIN_SESSIONS,
                 power: float = TARGET_POWER, alpha: float = ALPHA, step: int = 10) -> int:
    """Sessions to detect ``edge`` (bb/100) with ``power``, design SD = ``margin * sd``.

    ``required_units`` rounded UP to a multiple of ``step``, at least ``floor``.
    """
    n = required_units(margin * sd, edge, alpha=alpha, power=power)
    return max(floor, step * math.ceil(n / step))


def mde(sd: float, n: int, power: float = TARGET_POWER, alpha: float = ALPHA) -> float:
    """Minimum detectable edge (bb/100) with n sessions of SD ``sd``: (z_a/2 + z_b) sd / sqrt(n)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * sd / math.sqrt(n)


def power_at(edge: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    """Power of the two-sided z-test (upper tail) to detect a true edge with n sessions of SD sd."""
    from statistics import NormalDist
    nd = NormalDist()
    return 1 - nd.cdf(nd.inv_cdf(1 - alpha / 2) - edge * math.sqrt(n) / sd)


def paired_difference(x: SessionMatchResult, y: SessionMatchResult) -> Summary:
    """x - y per session (same master seed, same session s = same deals); CI across sessions."""
    if x.n_sessions != y.n_sessions or x.pairs_per_session != y.pairs_per_session:
        raise ValueError("paired difference needs the same sessions and session length")
    if [s.seed for s in x.sessions] != [s.seed for s in y.sessions]:
        raise ValueError("paired difference needs the same session seeds")
    if x.bot_b != y.bot_b:
        raise ValueError("paired difference needs the same opponent")
    diffs = [sx.total_bb - sy.total_bb for sx, sy in zip(x.sessions, y.sessions)]
    return summarize(diffs, hands_per_unit=x.hands_per_session)


def tier_verdict(stats: Summary) -> str:
    """The pre-registered decision rule of the primary test (two-sided 95% CI, alpha 0.05).

    ``"beats"`` iff the CI excludes 0 and the mean is > 0; ``"loses"`` iff
    the CI excludes 0 and the mean is < 0; else ``"no detectable edge"``.
    No variation (SD 0) is never significant.
    """
    lo, hi = stats.ci_bb100
    if stats.no_variation or math.isnan(stats.p_value):
        return "no detectable edge"
    if lo > 0 and stats.mean > 0:
        return "beats"
    if hi < 0 and stats.mean < 0:
        return "loses"
    return "no detectable edge"


def street_contributions(result: SessionMatchResult) -> dict[str, dict[str, float]]:
    """A's bb/100 contributed by hands ending on each street, with a CI ACROSS sessions.

    Per session: sum of A's nets over the hands that ended on street k, / hands
    in the session x 100. The five contributions add up to the session's
    bb/100. Each session gives one value per street and sessions are
    independent, so the CIs are valid (pointwise) for learning bots too.
    They describe where the chips moved; which hands reach a street is
    decided by the bots (a selection), so they are not causal.
    """
    if not all(s.games for s in result.sessions):
        raise ValueError("needs the hand logs (keep_games=True)")
    streets = ("preflop", "flop", "turn", "river", "showdown")
    per = {k: [] for k in streets}
    share = {k: 0 for k in streets}
    for s in result.sessions:
        acc = dict.fromkeys(streets, 0.0)
        for g in s.games:
            k = end_street(g.log)
            acc[k] += g.a_net_bb
            share[k] += 1
        for k in streets:
            per[k].append(acc[k])
    out = {}
    for k in streets:
        st = summarize(per[k], hands_per_unit=result.hands_per_session)
        out[k] = {"share": share[k] / result.hands_played, "bb100": st.mean_bb100,
                  "se_bb100": st.se_bb100, "ci_low": st.ci_bb100[0], "ci_high": st.ci_bb100[1]}
    return out


def splice_results(md: str, results: str, marker: str = RESULTS_MARKER) -> str:
    """Put ``results`` below ``marker`` in the pre-registration file, refusing a second run.

    Raises ``ValueError`` if the marker is missing or anything but whitespace
    already follows it (the official run happens once; no re-runs).
    """
    head, sep, tail = md.partition(marker)
    if not sep:
        raise ValueError("results marker not found: the pre-registration file was changed")
    if tail.strip():
        raise ValueError("results are already present below the marker: the official run happens once")
    return head + marker + "\n\n" + results.rstrip() + "\n"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def process_memory_mb() -> dict[str, float] | None:
    """This process's working set, peak working set and private (committed) bytes, in MB.

    Windows only (``GetProcessMemoryInfo``); on other systems the peak RSS
    from ``resource`` (no private-bytes figure). ``None`` if unavailable.
    """
    if sys.platform == "win32":
        from ctypes import wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return {"working_set": c.WorkingSetSize / 1e6, "peak_working_set": c.PeakWorkingSetSize / 1e6,
                "private": c.PrivateUsage / 1e6, "peak_private": c.PeakPagefileUsage / 1e6}
    try:
        import resource
    except ImportError:  # pragma: no cover
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # KB on Linux
    return {"working_set": math.nan, "peak_working_set": peak, "private": math.nan, "peak_private": math.nan}


def merge_peaks(samples: Sequence[tuple[int, dict[str, float] | None]]) -> dict[int, dict[str, float]]:
    """Per process id, the maximum of each memory figure over its samples."""
    out: dict[int, dict[str, float]] = {}
    for pid, m in samples:
        if m is None:
            continue
        cur = out.setdefault(pid, dict(m))
        for k, v in m.items():
            cur[k] = max(cur[k], v)
    return out


# --- Repo state and the model ---------------------------------------------------------


def git_state() -> tuple[str, str | None]:
    """``(short HEAD, porcelain status)``; ``("unknown", None)`` if git is unavailable."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return head, dirty


def commit_hash(allow_dirty: bool) -> str:
    """The commit to record; exits on a dirty or unknown tree unless ``allow_dirty`` (as train_holdem.py)."""
    head, dirty = git_state()
    if dirty is None:
        if not allow_dirty:
            sys.exit("cannot read the git state: refusing an unidentifiable official run (a pilot may pass --allow-dirty)")
        return "unknown"
    if dirty:
        if not allow_dirty:
            sys.exit("the work tree has uncommitted changes (git status --porcelain):\n" + dirty
                     + "\ncommit them first (the pre-registration must be committed), or, for a pilot only, pass --allow-dirty")
        return head + " + uncommitted changes"
    return head


def check_model() -> dict:
    """Assert the pinned blueprint is the one that will play; return its identification for the metadata."""
    if not MODEL.exists():
        sys.exit(f"{MODEL} not found (models/ is gitignored: copy the official blueprint there)")
    sha = file_sha256(MODEL)
    if sha != MODEL_SHA256:
        sys.exit(f"{MODEL} has SHA-256 {sha}, expected {MODEL_SHA256}: not the pinned official blueprint")
    bot = CFR_FACTORY()
    require_trained(bot, min_iterations=MODEL_ITERATION)
    info = bot.blueprint.info()
    if info["iteration"] != MODEL_ITERATION or bot.blueprint.meta.get("abstraction") != MODEL_ABSTRACTION:
        sys.exit(f"unexpected blueprint: {info}")
    return {"path": str(MODEL.relative_to(ROOT)), "sha256": sha, "description": repr(bot),
            "iteration": info["iteration"], "trained_at_commit": info["commit"],
            "abstraction_fingerprint": bot.blueprint.meta["abstraction"],
            "training_seed": bot.blueprint.meta.get("seed"), "n_infosets": bot.blueprint.meta.get("n_infosets")}


# --- Workers --------------------------------------------------------------------------


def _init_worker(turn_tables: str | None) -> None:
    if turn_tables:
        preload_turn_tables(turn_tables)
    require_trained(CFR_FACTORY(), min_iterations=MODEL_ITERATION)  # also warms the per-process blueprint cache


def _task(args: tuple) -> tuple:
    """One session, plus how long it took in this worker and the worker's memory afterwards."""
    pairing, pairs, master_seed, s, keep_games = args
    a, b = pairing
    t0 = time.perf_counter()
    sess = play_session(BOTS[a], BOTS[b], pairs, master_seed, s, keep_games)
    return pairing, sess, time.perf_counter() - t0, os.getpid(), process_memory_mb()


# --- Design table, plan -----------------------------------------------------------------


def design_rows() -> list[dict]:
    """Per pairing: N, design SD, required N, MDE, power at the target edge, estimated minutes."""
    rows = []
    for p in PAIRINGS:
        n = N_SESSIONS.get(p)
        sd = PILOT_SD.get(p)
        edge = PRIMARY_EDGE_BB100 if p == PRIMARY else SECONDARY_EDGE_BB100
        row = {"pairing": p, "n": n, "pilot_sd": sd, "edge": edge}
        if sd is not None:
            row["required"] = sessions_for(sd, edge)
            if n:
                row["mde"] = mde(SD_MARGIN * sd, n)
                row["power"] = power_at(edge, SD_MARGIN * sd, n)
        if p[0] == "CFRBot" and p != PRIMARY and p[1] in PILOT_DIFF_SD:
            dsd = PILOT_DIFF_SD[p[1]]
            row["diff_sd"] = dsd
            row["diff_required"] = sessions_for(dsd, SECONDARY_EDGE_BB100)
            if n:
                row["diff_mde"] = mde(SD_MARGIN * dsd, n)
        ms = MS_PER_HAND_WALL.get(p)
        if ms is not None and n:
            row["minutes"] = n * 2 * PAIRS_PER_SESSION * ms / 1000 / 60
        rows.append(row)
    return rows


def plan() -> None:
    print(f"master seed {MASTER_SEED} (official); pilot seed {PILOT_SEED}; {PAIRS_PER_SESSION} pairs "
          f"({2 * PAIRS_PER_SESSION} hands) per session; {PROCESSES} processes")
    print(f"model {MODEL.relative_to(ROOT)} @ {MODEL_ITERATION:,} it, sha256 {MODEL_SHA256[:16]}..., "
          f"abstraction {MODEL_ABSTRACTION[:12]}")
    print(f"primary {PRIMARY[0]} vs {PRIMARY[1]}: target edge {PRIMARY_EDGE_BB100} bb/100 at "
          f"{TARGET_POWER:.0%} power, alpha {ALPHA} two-sided; secondary target {SECONDARY_EDGE_BB100}; "
          f"design SD = pilot SD x {SD_MARGIN}")
    if not N_SESSIONS:
        print("N_SESSIONS not fixed yet: run --pilot first and fill PILOT_SD / N_SESSIONS")
        return
    total_min = 0.0
    for r in design_rows():
        a, b = r["pairing"]
        line = f"  {a} vs {b}: N {r['n']} sessions = {r['n'] * 2 * PAIRS_PER_SESSION:,} hands"
        if "required" in r:
            line += (f"; pilot SD {r['pilot_sd']:.1f} -> required {r['required']} for {r['edge']:.0f}; "
                     f"MDE {r['mde']:.1f}, power at {r['edge']:.0f} = {r['power']:.0%}")
        if "diff_sd" in r:
            line += (f"; paired diff vs Bayes SD {r['diff_sd']:.1f} -> required {r['diff_required']}, "
                     f"MDE {r['diff_mde']:.1f}")
        if "minutes" in r:
            line += f"; ~{r['minutes']:.1f} min"
            total_min += r["minutes"]
        print(line)
    print(f"estimated wall clock: {total_min:.0f} min at the pilot rate; budget x{THROTTLE_MARGIN} for "
          f"sustained throttling = {total_min * THROTTLE_MARGIN:.0f} min")


# --- Running ----------------------------------------------------------------------------


def run(n_sessions: dict[tuple[str, str], int], master_seed: int, stem_prefix: str, meta: dict,
        processes: int = PROCESSES) -> dict:
    """Play every pairing on one pool (imap, in order), save per-pairing JSON/PNG, return a summary."""
    today = meta["date"]
    tasks = [(p, PAIRS_PER_SESSION, master_seed, s, True) for p in PAIRINGS for s in range(n_sessions[p])]
    expected_desc = meta["model"]["description"]
    results: dict[tuple[str, str], SessionMatchResult] = {}
    rows: dict[str, dict] = {}
    mem: list[tuple[int, dict | None]] = []
    seconds: dict[tuple[str, str], list[float]] = {p: [] for p in PAIRINGS}
    current: list[Session] = []
    start = time.perf_counter()
    turn = str(TURN_TABLES) if TURN_TABLES.exists() else None
    with mp.get_context("spawn").Pool(processes, initializer=_init_worker, initargs=(turn,)) as pool:
        for pairing, sess, sec, pid, m in pool.imap(_task, tasks, chunksize=1):
            mem.append((pid, m))
            mem.append((os.getpid(), process_memory_mb()))
            seconds[pairing].append(sec)
            if "CFRBot" in pairing:
                desc = sess.bot_a_desc if pairing[0] == "CFRBot" else sess.bot_b_desc
                if desc != expected_desc:
                    raise RuntimeError(f"session {sess.index} of {pairing} played {desc!r}, expected {expected_desc!r}")
            current.append(sess)
            if len(current) < n_sessions[pairing]:
                continue
            res = combine_sessions(current, master_seed, PAIRS_PER_SESSION)
            current = []
            rows[f"{pairing[0]} vs {pairing[1]}"] = _finish_pairing(pairing, res, seconds[pairing], stem_prefix, today, meta)
            print(f"[{(time.perf_counter() - start) / 60:.1f} min] {res.summary()}", flush=True)
            results[pairing] = combine_sessions([dataclasses.replace(s, games=()) for s in res.sessions],
                                                master_seed, PAIRS_PER_SESSION)  # drop hand logs
            del res
    elapsed = time.perf_counter() - start
    diffs = {}
    for x in OPPONENTS:
        d = paired_difference(results[("CFRBot", x)], results[("BayesBot", x)])
        diffs[x] = {"mean_bb100": d.mean_bb100, "ci95_bb100": list(d.ci_bb100), "se_bb100": d.se_bb100,
                    "sd_bb100_per_session": d.sd * 100 / (2 * PAIRS_PER_SESSION), "p_value": d.p_value,
                    "n_sessions": d.n}
    peaks = merge_peaks(mem)
    parent = peaks.pop(os.getpid(), None)
    return {"rows": rows, "paired_differences": diffs, "elapsed_s": elapsed, "processes": processes,
            "memory_mb": {"parent_peak": parent, "workers_peak": {str(k): v for k, v in peaks.items()},
                          "max_worker_peak_working_set": max((v["peak_working_set"] for v in peaks.values()), default=None),
                          "max_worker_peak_private": max((v["peak_private"] for v in peaks.values()), default=None)},
            "primary": rows[f"{PRIMARY[0]} vs {PRIMARY[1]}"]}


def pairing_row(pairing: tuple[str, str], res: SessionMatchResult, secs: Sequence[float], processes: int,
                stem: str = "") -> dict:
    """Everything reported for one pairing (needs the hand logs for the breakdowns). No I/O."""
    a, b = pairing
    gain = learning_gain(res, EARLY, LATE)
    bd = session_breakdown(res)
    lo, hi = res.ci_bb100
    return {"pairing": f"{a} vs {b}", "n_sessions": res.n_sessions, "hands": res.hands_played,
            "mean_bb100": res.mean_bb100, "ci95_bb100": [lo, hi], "se_bb100": res.se_bb100,
            "sd_bb100_per_session": res.sd_session_bb100, "p_value": res.p_value,
            "design_effect": res.design_effect(), "verdict": res.verdict(),
            "tier_verdict": tier_verdict(res.stats) if pairing == PRIMARY else None,
            "sessions_needed_80pct_power_observed": res.sessions_needed(),
            "learning_gain": [gain.mean_bb100, *gain.ci_bb100],
            "by_position": bd["by_position"], "street_contributions": street_contributions(res),
            "folds_a": bd["folds_a"], "folds_b": bd["folds_b"], "leaks": bd["leaks"],
            "session_seconds_mean": math.fsum(secs) / len(secs),
            "ms_per_hand_wall": 1000 * math.fsum(secs) / res.hands_played / processes,
            "json": f"{stem}.json", "png": f"{stem}.png"}


def _finish_pairing(pairing, res: SessionMatchResult, secs: list[float], stem_prefix: str, today: str,
                    meta: dict) -> dict:
    a, b = pairing
    stem = f"{today}_{stem_prefix}{SHORT[a]}_vs_{SHORT[b]}_sessions"
    row = pairing_row(pairing, res, secs, meta["processes"], stem)
    extra = {**meta, "pairing": row["pairing"], "primary": pairing == PRIMARY,
             "verdict_tier_rule": row["tier_verdict"],
             "learning_gain": {"early_pairs": EARLY, "late_pairs": LATE, "mean_bb100": row["learning_gain"][0],
                               "ci95_bb100": row["learning_gain"][1:]},
             "street_contributions_session_ci": row["street_contributions"],
             "timing": {"session_seconds_mean": row["session_seconds_mean"],
                        "ms_per_hand_wall": row["ms_per_hand_wall"]}}
    save_session_json(res, RESULTS / f"{stem}.json", BUCKET_PAIRS, extra)
    save_session_plot(res, RESULTS / f"{stem}.png", BUCKET_PAIRS)
    return row


def _fmt_mb(x) -> str:
    """A memory figure for the report; the probe may return None (or NaN off Windows): never crash on it."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f} MB"


def results_markdown(summary: dict, meta: dict) -> str:
    """The official results section (below the marker of results/phase6.md)."""
    p = summary["primary"]
    lo, hi = p["ci95_bb100"]
    lines = [
        f"- date {meta['date']}; commit {meta['commit']}; master seed {meta['master_seed']}; "
        f"{meta['processes']} processes; run time {summary['elapsed_s']:.0f} s",
        f"- model: {meta['model']['description']}; sha256 {meta['model']['sha256']}",
        "",
        f"**Primary test: CFRBot vs BayesBot = {p['mean_bb100']:+.1f} bb/100, 95% CI [{lo:+.1f}, {hi:+.1f}], "
        f"SE {p['se_bb100']:.1f}, SD/session {p['sd_bb100_per_session']:.1f}, p = {p['p_value']:.3g}, "
        f"{p['n_sessions']} sessions = {p['hands']:,} hands. Pre-registered verdict: {p['tier_verdict']}.**",
        "",
        "| pairing | sessions | hands | bb/100 | 95% CI | SE | SD/session | p | design effect | "
        f"learning gain {LATE} - {EARLY} [95% CI] |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in summary["rows"].values():
        g = r["learning_gain"]
        lines.append(f"| {r['pairing']} | {r['n_sessions']} | {r['hands']:,} | {r['mean_bb100']:+.1f} | "
                     f"[{r['ci95_bb100'][0]:+.1f}, {r['ci95_bb100'][1]:+.1f}] | {r['se_bb100']:.1f} | "
                     f"{r['sd_bb100_per_session']:.1f} | {r['p_value']:.3g} | {r['design_effect']:.2f} | "
                     f"{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}] |")
    lines += ["", "### CFRBot minus BayesBot against the same opponent (paired by session: same deals)", "",
              "| opponent | difference bb/100 | 95% CI | SE | SD/session | p |", "|---|---:|---|---:|---:|---:|"]
    for x, d in summary["paired_differences"].items():
        lines.append(f"| {SHORT[x]} | {d['mean_bb100']:+.1f} | [{d['ci95_bb100'][0]:+.1f}, {d['ci95_bb100'][1]:+.1f}] | "
                     f"{d['se_bb100']:.1f} | {d['sd_bb100_per_session']:.1f} | {d['p_value']:.3g} |")
    lines += ["", "### Breakdown of every pairing (position: CI across sessions; street: contribution to A's "
              "bb/100 by the street the hand ended on, CI across sessions, descriptive)", ""]
    for r in summary["rows"].values():
        pos = r["by_position"]
        st = r["street_contributions"]
        lines.append(f"- **{r['pairing']}**: button {pos['button']['bb100']:+.1f} "
                     f"[{pos['button']['ci_low']:+.1f}, {pos['button']['ci_high']:+.1f}], big blind "
                     f"{pos['big_blind']['bb100']:+.1f} [{pos['big_blind']['ci_low']:+.1f}, {pos['big_blind']['ci_high']:+.1f}]; "
                     + ", ".join(f"{k} {v['share']:.0%} of hands {v['bb100']:+.1f} [{v['ci_low']:+.1f}, {v['ci_high']:+.1f}]"
                                 for k, v in st.items()))
        fa = r["folds_a"]
        lines.append("  - A fold-to-bet: " + ", ".join(
            f"{k} {v['fold_to_bet_rate']:.0%} of {v['faced_bet']:,}" for k, v in fa.items() if v["faced_bet"]))
        for leak in r["leaks"]:
            lines.append(f"  - leak: {leak}")
    mm = summary["memory_mb"]
    lines += ["", f"peak memory: worker working set max {_fmt_mb(mm.get('max_worker_peak_working_set'))}, "
              f"worker private max {_fmt_mb(mm.get('max_worker_peak_private'))}; parent "
              f"{_fmt_mb((mm.get('parent_peak') or {}).get('peak_working_set'))} working set", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the design and budget, play nothing")
    ap.add_argument("--pilot", action="store_true", help=f"pilot on seed {PILOT_SEED} (NOT official)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="pilot only (refused for the official run): run on a dirty tree (recorded); --pilot implies it")
    ap.add_argument("--processes", type=int, default=PROCESSES)
    ap.add_argument("--smoke", action="store_true", help="2 sessions per pairing on the pilot seed (a plumbing check)")
    args = ap.parse_args(argv)
    if args.plan:
        plan()
        return
    today = dt.date.today().isoformat()
    args.pilot = args.pilot or args.smoke
    official = not args.pilot
    if official and args.allow_dirty:
        sys.exit("--allow-dirty is refused for the official run: commit first (the pre-registration must be committed)")
    commit = commit_hash(args.allow_dirty or args.pilot)
    model = check_model()
    if official:
        if not N_SESSIONS or set(N_SESSIONS) != set(PAIRINGS):
            sys.exit("design not fixed: N_SESSIONS must cover every pairing before the official run")
        low = [p for p in PAIRINGS if p in PILOT_SD and N_SESSIONS[p] < sessions_for(
            PILOT_SD[p], PRIMARY_EDGE_BB100 if p == PRIMARY else SECONDARY_EDGE_BB100)]
        low += [("CFRBot - BayesBot", x) for x in OPPONENTS
                if N_SESSIONS[("CFRBot", x)] < sessions_for(PILOT_DIFF_SD[x], SECONDARY_EDGE_BB100)]
        if low:
            sys.exit(f"design error: N_SESSIONS below the power requirement for {low}")
        for x in OPPONENTS:
            if N_SESSIONS[("CFRBot", x)] != N_SESSIONS[("BayesBot", x)]:
                sys.exit(f"design error: CFRBot and BayesBot vs {x} need the same sessions (paired)")
        md = PHASE6_MD.read_text(encoding="utf-8")
        splice_results(md, "check")  # refuse before playing if the marker is missing or results exist
    n_sessions = N_SESSIONS if official else ({p: 2 for p in PAIRINGS} if args.smoke else PILOT_SESSIONS)
    seed = MASTER_SEED if official else PILOT_SEED
    meta = {"date": today, "commit": commit, "official": official, "master_seed": seed,
            "design": {"pairs_per_session": PAIRS_PER_SESSION, "hands_per_session": 2 * PAIRS_PER_SESSION,
                       "fresh_bots_per_session_and_room": True, "duplicate_seats_swapped": True,
                       "unit": "session", "n_sessions": {f"{a} vs {b}": n for (a, b), n in n_sessions.items()},
                       "primary": f"{PRIMARY[0]} vs {PRIMARY[1]}", "alpha": ALPHA,
                       "primary_edge_bb100": PRIMARY_EDGE_BB100, "secondary_edge_bb100": SECONDARY_EDGE_BB100,
                       "power": TARGET_POWER, "sd_margin": SD_MARGIN, "env_openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
            "processes": args.processes, "model": model,
            "bot_params": "committed defaults (CFRBot: the pinned blueprint, below='pseudo_harmonic')"}
    label = "OFFICIAL" if official else "PILOT (NOT OFFICIAL)"
    print(f"{label}: master seed {seed}, {sum(n_sessions.values())} sessions, commit {commit}, model {model['description']}")
    kind = "smoke" if args.smoke else "pilot"
    # Per-pairing files: {date}_phase6_<A>_vs_<B>_sessions.* (official) so they cannot be confused with Phase 5's.
    summary = run(n_sessions, seed, "phase6_" if official else f"phase6_{kind}_seed{seed}_", meta, args.processes)
    out = RESULTS / (f"{today}_phase6_summary.json" if official else f"{today}_phase6_{kind}_seed{seed}.json")
    out.write_text(json.dumps({"meta": meta, **{k: v for k, v in summary.items()}}, indent=1, default=str),
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"memory_mb": summary["memory_mb"], "elapsed_s": summary["elapsed_s"]}, indent=1, default=str))
    if official:
        PHASE6_MD.write_text(splice_results(PHASE6_MD.read_text(encoding="utf-8"), results_markdown(summary, meta)),
                             encoding="utf-8")
        print(f"wrote {PHASE6_MD}")


if __name__ == "__main__":
    main()
