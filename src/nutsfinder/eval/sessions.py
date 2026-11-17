"""Session-based duplicate matches: valid statistics for bots that learn.

The problem with per-hand (or per-pair) SEs for learning bots
-------------------------------------------------------------
:func:`~nutsfinder.eval.match.duplicate_match` builds each bot once per room
and keeps it for the whole match. For the bots of Phases 1-4 that is
harmless: they have no memory, so pair i's result depends only on pair i's
seeds, and the pair results are i.i.d. That is the assumption behind
SE = s / sqrt(n) (see :mod:`nutsfinder.eval.stats`).

A learning bot (Phase 5: opponent modelling) breaks it. Its decision in hand
i depends on everything it saw in hands 1..i-1, so

- *not identically distributed*: early pairs are played by a bot that knows
  nothing, late pairs by one that has learned. The "true edge per pair" is a
  different number for every i; there is no single mu that the pair mean
  estimates.
- *not independent*: one lucky early showdown changes the model, which
  changes every later decision. Pair results are correlated through the
  bot's memory (typically positively: a good or bad read persists), and a
  positive correlation makes s / sqrt(n) *too small*: the true variance of a
  sum of n correlated units is n sigma^2 + 2 sum_{i<j} Cov(X_i, X_j), not
  n sigma^2.
- *length-dependent*: the overall mean averages over the learning phase,
  so a 1,000-hand match and a 100,000-hand match estimate different things,
  and "the bb/100" of a learning bot is not defined without saying after how
  many hands.

The fix: independent sessions
-----------------------------
A *session* is a complete duplicate match of ``pairs_per_session`` pairs
played by FRESH bot instances (new objects from the factories, one copy of
A and B per room, exactly as in :func:`duplicate_match`). Nothing carries
over from one session to the next: different deal seeds, different bot
seeds, no shared memory. So whatever happens inside a session (learning,
correlations between its hands), the session totals are i.i.d. across
sessions, and the unit of analysis is the session:

    session result = A's total net over the session's 2 * pairs_per_session
                     hands, in bb;   bb/100 = total / hands_in_session * 100.

Mean, SD, SE, CI and p-value are then taken *over sessions* with the usual
:func:`~nutsfinder.eval.stats.summarize`, and they are valid for any bot,
learning or not. The estimand is explicit: "A's expected bb/100 over the
first ``2 * pairs_per_session`` hands against a fresh opponent model".

Trade-off: session length vs number of sessions
-----------------------------------------------
- Long sessions give the bot time to learn, so the estimate reflects the
  learned behaviour rather than the cold start.
- Many sessions give many units. The SE of the mean is SD_session /
  sqrt(n_sessions), and the CLT needs a few dozen units at the very least
  (session totals of hundreds of hands are much less heavy-tailed than
  single hands, so ~50+ is comfortable; below ~30 the normal CI is suspect).
- For a fixed budget of hands H = n_sessions * hands_per_session, and a bot
  *without* memory, the SE does not depend on how H is split: SD_session in
  bb/100 scales like 1 / sqrt(hands_per_session), so SE = SD_session /
  sqrt(n_sessions) ~ 1 / sqrt(H). The split only matters through learning
  (which wants long sessions) and through the CLT and the SD estimate itself
  (which want many sessions).
- Results are specific to the chosen session length. A bot that learns
  slowly looks worse in short sessions. Always report the session length
  with the number, and compare bots only at the same session length.

Agreement with duplicate_match for bots without memory
------------------------------------------------------
Session s is *literally* ``duplicate_match(make_a, make_b, pairs_per_session,
session_seed(master_seed, s))``: same per-pair deal / button / reseed design,
fresh instances. For bots without memory the pairs are i.i.d. whichever way
they are grouped, so the session mean (bb/100) estimates the same quantity as
one long duplicate match, in expectation; only the seeds differ, so the two
numbers agree within their CIs, not to the last digit (tested).

The learning curve
------------------
:func:`learning_curve` splits every session into buckets of consecutive
pairs (pairs 0-24, 25-49, ...) and computes A's bb/100 in each bucket per
session. Each session contributes exactly one value per bucket, and sessions
are independent, so each bucket's CI computed *across sessions* is valid. The
CIs are pointwise (each covers its own bucket's mean 95% of the time; with
many buckets, some will miss by chance). To test "did it improve?", use
:func:`learning_gain`: per session, late bucket minus early bucket, then a CI
across sessions (a paired comparison, which cancels between-session noise).

The x-axis is the *pair index*: at pair i every bot instance in the session
has played exactly i earlier hands (each room's copy plays one hand per
pair), so pair index = "hands of experience" of the bots.

Seeds and parallelism
---------------------
``session_seed(master_seed, s)`` hashes (purpose, master_seed, s) with
SHA-256, so session s's seed does not depend on how many sessions are played
or in which order or process they run, and sessions never reuse a stream.
Because each session is fully determined by (factories, pairs_per_session,
master_seed, s), running sessions in a process pool cannot change any number
(tested); ``pool.map`` also keeps them in order.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool

from nutsfinder.eval.match import BotFactory, Game, duplicate_match
from nutsfinder.eval.stats import Summary, required_units, summarize

__all__ = [
    "Session",
    "SessionMatchResult",
    "Bucket",
    "session_seed",
    "play_session",
    "combine_sessions",
    "session_duplicate_match",
    "learning_curve",
    "learning_gain",
    "session_by_position",
    "session_breakdown",
]


def session_seed(master_seed: int, session_index: int) -> int:
    """64-bit seed of session ``session_index``, hashed from the master seed.

    Hashing (rather than ``master_seed + s`` or one shared ``random.Random``)
    means: different master seeds never produce overlapping session streams,
    and session s's seed is the same whatever ``n_sessions`` is.
    """
    if session_index < 0:
        raise ValueError("session index must be >= 0")
    key = f"nutsfinder/session_duplicate|{master_seed}|{session_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class Session:
    """One session: a duplicate match with fresh bots.

    Attributes:
        index: session number s.
        seed: ``session_seed(master_seed, s)``, the master seed of this
            session's duplicate match.
        bot_a, bot_b: bot names.
        hand_results: A's net in bb for every hand, in play order (game 1 and
            game 2 of pair 0, then pair 1, ...).
        pair_results: A's net in bb per pair (sum of the pair's two hands).
        games: the hand logs (empty if ``keep_games=False``).
        bot_a_desc, bot_b_desc: ``repr`` of the bots (see
            :func:`~nutsfinder.eval.match.describe_bot`); ``""`` if unknown.
    """

    index: int
    seed: int
    bot_a: str
    bot_b: str
    hand_results: tuple[float, ...]
    pair_results: tuple[float, ...]
    games: tuple[Game, ...]
    bot_a_desc: str = ""
    bot_b_desc: str = ""

    @property
    def n_pairs(self) -> int:
        return len(self.pair_results)

    @property
    def hands(self) -> int:
        return 2 * self.n_pairs

    @property
    def total_bb(self) -> float:
        """A's net over the whole session (the unit of analysis), in bb."""
        return math.fsum(self.hand_results)

    @property
    def bb100(self) -> float:
        return self.total_bb / self.hands * 100


def play_session(
    make_a: BotFactory,
    make_b: BotFactory,
    pairs_per_session: int,
    master_seed: int,
    session_index: int,
    keep_games: bool = True,
) -> Session:
    """Play session ``session_index``: a fresh duplicate match.

    Module-level (picklable) so it can run in a worker process; the factories
    must be picklable too (classes, or ``functools.partial`` of classes).
    """
    if pairs_per_session < 2:
        raise ValueError("a session needs at least 2 pairs")
    seed = session_seed(master_seed, session_index)
    m = duplicate_match(make_a, make_b, pairs_per_session, seed)  # builds 4 fresh instances
    return Session(
        index=session_index,
        seed=seed,
        bot_a=m.bot_a,
        bot_b=m.bot_b,
        hand_results=tuple(g.a_net_bb for g in m.games),
        pair_results=m.unit_results,
        games=m.games if keep_games else (),
        bot_a_desc=m.bot_a_desc,
        bot_b_desc=m.bot_b_desc,
    )


def _play_session_args(args: tuple) -> Session:
    return play_session(*args)


@dataclass(frozen=True, slots=True)
class SessionMatchResult:
    """A session match, from A's point of view. The unit is the session.

    ``stats`` is :func:`summarize` of the session totals (bb per session)
    with ``hands_per_unit = hands per session``, so ``stats.mean_bb100`` is
    the mean session bb/100 and the SE/CI/p are across independent sessions.
    """

    bot_a: str
    bot_b: str
    master_seed: int
    pairs_per_session: int
    sessions: tuple[Session, ...]
    stats: Summary
    bot_a_desc: str = ""
    bot_b_desc: str = ""

    kind = "session_duplicate"
    unit = "session"

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_units(self) -> int:
        return self.n_sessions

    @property
    def hands_per_session(self) -> int:
        return 2 * self.pairs_per_session

    @property
    def hands_played(self) -> int:
        return self.n_sessions * self.hands_per_session

    @property
    def unit_results(self) -> tuple[float, ...]:
        """A's total bb per session."""
        return tuple(s.total_bb for s in self.sessions)

    @property
    def session_bb100(self) -> tuple[float, ...]:
        return tuple(s.bb100 for s in self.sessions)

    @property
    def games(self) -> tuple[Game, ...]:
        """All hand logs, session after session (for the descriptive breakdowns)."""
        return tuple(g for s in self.sessions for g in s.games)

    @property
    def mean_bb100(self) -> float:
        return self.stats.mean_bb100

    @property
    def sd_session_bb100(self) -> float:
        """SD of one session's bb/100 (the number the power calculation needs)."""
        return self.stats.sd * 100 / self.hands_per_session

    @property
    def se_bb100(self) -> float:
        return self.stats.se_bb100

    @property
    def ci_bb100(self) -> tuple[float, float]:
        return self.stats.ci_bb100

    @property
    def p_value(self) -> float:
        return self.stats.p_value

    def naive_pair_stats(self) -> Summary:
        """Per-pair statistics pooling all sessions, IGNORING the session structure.

        Valid only for bots without memory. Shown for comparison: the ratio
        ``(se_bb100 / naive SE)^2`` is the "design effect". About 1 means the
        pairs inside a session behave as independent; well above 1 means
        within-session correlation (e.g. learning) that a per-pair SE would
        have hidden.
        """
        return summarize([x for s in self.sessions for x in s.pair_results], hands_per_unit=2)

    def design_effect(self) -> float:
        naive = self.naive_pair_stats().se_bb100
        return (self.se_bb100 / naive) ** 2 if naive > 0 else math.nan

    def sessions_needed(self, edge_bb100: float | None = None, power: float = 0.8) -> int | None:
        """Sessions (of this length) needed to detect ``edge_bb100`` with ``power``.

        Defaults to the observed mean. ``None`` if the edge is 0.
        """
        edge = self.mean_bb100 if edge_bb100 is None else edge_bb100
        if edge == 0:
            return None
        return required_units(self.sd_session_bb100, edge, power=power)

    def verdict(self) -> str:
        lo, hi = self.ci_bb100
        if self.stats.no_variation:
            return (
                f"no variation: every session gave exactly {self.mean_bb100:+g} bb/100, so there "
                "is no CI or p-value (only meaningful if the setup forces it)"
            )
        if self.stats.significant:
            who = self.bot_a if self.stats.mean > 0 else self.bot_b
            p = "< 1e-300" if self.p_value == 0 else f"= {self.p_value:.2g}"
            return f"significant edge for {who} (p {p}) over {self.hands_per_session}-hand sessions"
        bound = max(abs(lo), abs(hi))
        text = f"no detectable edge: the 95% CI rules out edges beyond +/-{bound:.1f} bb/100"
        needed = self.sessions_needed()
        if needed is None:
            return text + " (observed mean is exactly 0)"
        return (
            text + f"; underpowered for the observed {self.mean_bb100:+.2f} bb/100: detecting it with "
            f"80% power would need about {needed:,} sessions = {needed * self.hands_per_session:,} hands"
        )

    def summary(self) -> str:
        lo, hi = self.ci_bb100
        return "\n".join(
            [
                f"{self.bot_a} vs {self.bot_b}: session duplicate match, master seed {self.master_seed}",
                f"  {self.n_sessions:,} sessions x {self.pairs_per_session:,} pairs "
                f"({self.hands_per_session:,} hands each, fresh bots per session), "
                f"{self.hands_played:,} hands",
                f"  win rate for {self.bot_a}: {self.mean_bb100:+.2f} bb/100 "
                f"(95% CI {lo:+.2f} to {hi:+.2f}, normal approximation over sessions)",
                f"  SD {self.sd_session_bb100:.1f} bb/100 per session, SE {self.se_bb100:.2f} bb/100, "
                f"p = {self.p_value:.3g}; design effect vs per-pair SE {self.design_effect():.2f}",
                f"  verdict: {self.verdict()}",
            ]
        )


def combine_sessions(
    sessions: Sequence[Session], master_seed: int, pairs_per_session: int
) -> SessionMatchResult:
    """Build the result from already-played sessions (in session order)."""
    if len(sessions) < 2:
        raise ValueError("need at least 2 sessions to estimate a standard deviation")
    if [s.index for s in sessions] != list(range(len(sessions))):
        raise ValueError("sessions must be 0..n-1 in order")
    if any(s.n_pairs != pairs_per_session for s in sessions):
        raise ValueError("every session must have pairs_per_session pairs")
    hands = 2 * pairs_per_session
    return SessionMatchResult(
        bot_a=sessions[0].bot_a,
        bot_b=sessions[0].bot_b,
        master_seed=master_seed,
        pairs_per_session=pairs_per_session,
        sessions=tuple(sessions),
        stats=summarize([s.total_bb for s in sessions], hands_per_unit=hands),
        bot_a_desc=sessions[0].bot_a_desc,
        bot_b_desc=sessions[0].bot_b_desc,
    )


def session_duplicate_match(
    make_a: BotFactory,
    make_b: BotFactory,
    n_sessions: int,
    pairs_per_session: int,
    master_seed: int,
    processes: int = 1,
    pool: Pool | None = None,
    keep_games: bool = True,
) -> SessionMatchResult:
    """``n_sessions`` independent sessions of ``pairs_per_session`` duplicate pairs.

    Every session builds fresh bots (two copies of A and two of B, one per
    room) and plays ``duplicate_match(make_a, make_b, pairs_per_session,
    session_seed(master_seed, s))``. The unit is the session; see the module
    docstring for why. The length (sessions and pairs) is fixed up front;
    there is no early stop.

    Args:
        processes: run sessions in a ``spawn`` pool of this size (1 = in this
            process). Results are identical for any value.
        pool: an existing pool to use instead (``processes`` is then ignored).
        keep_games: keep the hand logs (needed for the street / fold
            breakdowns; per-hand nets are always kept).
    """
    if n_sessions < 2:
        raise ValueError("need at least 2 sessions to estimate a standard deviation")
    if pairs_per_session < 2:
        raise ValueError("a session needs at least 2 pairs")
    args = [(make_a, make_b, pairs_per_session, master_seed, s, keep_games) for s in range(n_sessions)]
    if pool is not None:
        sessions = pool.map(_play_session_args, args, chunksize=1)
    elif processes > 1:
        with mp.get_context("spawn").Pool(min(processes, n_sessions)) as p:
            sessions = p.map(_play_session_args, args, chunksize=1)
    else:
        sessions = [_play_session_args(a) for a in args]
    return combine_sessions(sessions, master_seed, pairs_per_session)


# --- Learning curve ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bucket:
    """Pairs ``[start, stop)`` of every session; stats across sessions.

    ``stats`` summarizes one value per session (A's bb over the bucket's
    pairs) with ``hands_per_unit = 2 * (stop - start)``, so ``mean_bb100``,
    ``se_bb100`` and ``ci_bb100`` are in bb/100.
    """

    start: int
    stop: int
    stats: Summary

    @property
    def mid(self) -> float:
        return (self.start + self.stop - 1) / 2


def _bucket_bounds(n_pairs: int, bucket_pairs: int) -> list[tuple[int, int]]:
    if bucket_pairs < 1:
        raise ValueError("bucket_pairs must be >= 1")
    return [(lo, min(lo + bucket_pairs, n_pairs)) for lo in range(0, n_pairs, bucket_pairs)]


def learning_curve(result: SessionMatchResult, bucket_pairs: int = 25) -> list[Bucket]:
    """A's bb/100 by pair index within session, in buckets, CI across sessions.

    Bucket k covers pairs ``[k * bucket_pairs, (k + 1) * bucket_pairs)`` of
    every session (the last bucket may be shorter). Each session contributes
    one value per bucket, so the CIs are valid (pointwise) whether or not
    the bots learn.
    """
    out = []
    for lo, hi in _bucket_bounds(result.pairs_per_session, bucket_pairs):
        vals = [math.fsum(s.pair_results[lo:hi]) for s in result.sessions]
        out.append(Bucket(lo, hi, summarize(vals, hands_per_unit=2 * (hi - lo))))
    return out


def learning_gain(
    result: SessionMatchResult, early: tuple[int, int], late: tuple[int, int]
) -> Summary:
    """Late-minus-early bb/100, paired within session, CI across sessions.

    ``early`` and ``late`` are pair ranges ``(start, stop)`` of equal length.
    The returned Summary's ``*_bb100`` fields give the change in bb/100.
    A CI excluding 0 is evidence that A's results changed during the session.
    """
    (e0, e1), (l0, l1) = early, late
    if e1 - e0 != l1 - l0 or e1 <= e0:
        raise ValueError("early and late ranges must be non-empty and of equal length")
    diffs = [math.fsum(s.pair_results[l0:l1]) - math.fsum(s.pair_results[e0:e1]) for s in result.sessions]
    return summarize(diffs, hands_per_unit=2 * (e1 - e0))


# --- Breakdowns ---------------------------------------------------------------


def session_by_position(result: SessionMatchResult) -> dict[str, dict[str, float]]:
    """A's bb/100 on the button and in the big blind, CI across sessions.

    Per session: A's bb/100 over its button hands (one per pair) and over
    its big-blind hands. Unlike :func:`nutsfinder.eval.breakdown.by_position`
    (a per-hand SE), this is valid for learning bots. Each half carries the
    full card luck (the duplicate cancellation needs both halves), so these
    CIs are much wider than the headline CI.
    """
    if not all(s.games for s in result.sessions):
        raise ValueError("needs the hand logs (keep_games=True)")
    out: dict[str, dict[str, float]] = {}
    for label, on_button in (("button", True), ("big_blind", False)):
        vals = [math.fsum(g.a_net_bb for g in s.games if g.a_on_button == on_button) for s in result.sessions]
        st = summarize(vals, hands_per_unit=result.pairs_per_session)  # one hand per pair per position
        out[label] = {
            "hands": result.pairs_per_session * result.n_sessions,
            "bb100": st.mean_bb100,
            "se_bb100": st.se_bb100,
            "ci_low": st.ci_bb100[0],
            "ci_high": st.ci_bb100[1],
        }
    return out


def session_breakdown(result: SessionMatchResult) -> dict:
    """Like :func:`nutsfinder.eval.breakdown.breakdown`, with session-level position CIs.

    Street and fold diagnostics are descriptive (no CIs) and are pooled over
    all hands of all sessions.
    """
    from nutsfinder.eval.breakdown import _fold_dict, by_end_street, fold_stats, leaks

    return {
        "by_position": session_by_position(result),
        "by_end_street": by_end_street(result),  # type: ignore[arg-type]  (duck-typed)
        "folds_a": _fold_dict(fold_stats(result.games, "a")),
        "folds_b": _fold_dict(fold_stats(result.games, "b")),
        "leaks": leaks(result),  # type: ignore[arg-type]
    }
