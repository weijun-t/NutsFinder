"""Monte Carlo equity: estimate how often one hand beats another.

The idea: deal the unknown board cards at random many times, see who wins
each run-out, and average. Each run-out ("sample") gives the hero a payoff

    X = 1 (win), 1/2 (split), 0 (loss),

and the equity is the true mean ``p = E[X]``. The estimate is the sample
mean of ``n`` independent draws of ``X``.

Why the error shrinks like ``1/sqrt(n)``: the samples are independent with
the same variance ``sigma^2 = Var(X)``, so the variance of their *sum* is
``n * sigma^2`` and the variance of their *mean* is ``n * sigma^2 / n^2 =
sigma^2 / n``. The standard error (the standard deviation of the estimate) is
the square root of that, ``sigma / sqrt(n)``. So 4x the samples halves the
error, and 100x the samples gives one more decimal digit. By the central
limit theorem the estimate is roughly normal, so ``equity +/- 2 * std_error``
is about a 95% confidence interval.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from nutsfinder.engine.cards import FULL_DECK, Card
from nutsfinder.engine.evaluator import evaluate

__all__ = [
    "EquityResult",
    "Combo",
    "RANDOM_RANGE",
    "monte_carlo_equity",
    "equity_vs_range",
    "equity_vs_nested_ranges",
]

Combo = tuple[Card, Card]
"""Two hole cards. The order of the two cards does not matter."""

RANDOM_RANGE: tuple[Combo, ...] = tuple(combinations(FULL_DECK, 2))
"""All C(52, 2) = 1,326 starting hands: "a uniformly random opponent hand"."""


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Outcome of a Monte Carlo equity run, from the hero's point of view.

    Attributes:
        equity: estimated share of the pot the hero wins, in ``[0, 1]``, with a
            tie counted as half a win: ``(wins + ties / 2) / samples``.
        std_error: estimated standard error of ``equity``.
        samples: number of simulated run-outs.
        wins: run-outs the hero won outright.
        ties: run-outs that were split.
        losses: run-outs the villain won outright.

    ``wins + ties + losses == samples``.
    """

    equity: float
    std_error: float
    samples: int
    wins: int
    ties: int
    losses: int


def monte_carlo_equity(
    hero: Sequence[Card],
    villain: Sequence[Card],
    board: Sequence[Card] = (),
    samples: int = 10_000,
    seed: int | None = None,
) -> EquityResult:
    """Estimate the hero's equity against one known villain hand.

    Contract (checked by ``tests/test_equity.py``):

    - ``hero`` and ``villain`` are two hole cards each; ``board`` holds 0, 3, 4
      or 5 community cards. All cards must be distinct: an overlapping or
      repeated card, a wrong number of cards, or ``samples < 1`` raises
      ``ValueError``.
    - Known cards (both hands and the board) are removed from the deck; each
      sample deals the rest of the board from the remaining cards and compares
      the two best hands with :func:`nutsfinder.engine.evaluator.evaluate`.
    - All randomness comes from a private ``random.Random(seed)``; never the
      global :mod:`random` or numpy state. The same arguments with the same
      non-``None`` seed give an identical result. Caveat: ``random.Random``
      uses ``abs(seed)``, so ``seed=5`` and ``seed=-5`` give the same run.
    - ``equity`` counts a tie as half a win, and ``std_error`` is the estimated
      standard error of that estimate, so that ``equity +/- 2 * std_error`` is
      roughly a 95% confidence interval for the true equity.
    - That interval is a normal approximation and it breaks down for small
      ``samples`` or equity near 0 or 1. With ``samples == 1`` there is no
      spread to measure and ``std_error`` is ``nan``. If every sample happens
      to give the same outcome (say 20 losses for a 2% hand), ``std_error``
      is 0 even though the estimate is far from certain.
    """
    _validate(hero, villain, board, samples)
    hero, villain, board = list(hero), list(villain), list(board)

    # The stub: every card not already known, in a fixed (id) order, so that
    # the same seed always draws the same run-outs.
    known = set(hero) | set(villain) | set(board)
    stub = [card for card in FULL_DECK if card not in known]
    rng = random.Random(seed)  # private RNG: never the global random state
    cards_to_come = 5 - len(board)

    wins = ties = 0
    for _ in range(samples):
        # rng.sample draws without replacement, so a run-out never repeats a
        # card; every set of cards_to_come cards is equally likely.
        full_board = board + rng.sample(stub, cards_to_come)
        hero_rank = evaluate(hero + full_board)
        villain_rank = evaluate(villain + full_board)
        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
    losses = samples - wins - ties

    return EquityResult(
        equity=(wins + ties / 2) / samples,
        std_error=_std_error(wins, ties, samples),
        samples=samples,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def _validate(
    hero: Sequence[Card], villain: Sequence[Card], board: Sequence[Card], samples: int
) -> None:
    """Raise ``ValueError`` unless the arguments describe a legal deal."""
    if len(hero) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hero)}")
    if len(villain) != 2:
        raise ValueError(f"villain needs exactly 2 hole cards, got {len(villain)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must hold 0, 3, 4 or 5 cards, got {len(board)}")
    known = [*hero, *villain, *board]
    if len(set(known)) != len(known):
        raise ValueError(f"cards must all be distinct: {[str(c) for c in known]}")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")


def _std_error(wins: int, ties: int, n: int) -> float:
    """Standard error of the equity estimate: sample std of X over sqrt(n).

    X is 1 for ``wins`` samples, 1/2 for ``ties`` and 0 otherwise, so

        sum of X   = wins + ties / 2       (so the mean is the equity p)
        sum of X^2 = wins + ties / 4       (since (1/2)^2 = 1/4)

    The sample variance (dividing by n - 1, which makes it unbiased) is

        s^2 = (sum of X^2 - n * p^2) / (n - 1).

    Without ties X is 0 or 1, X^2 = X, and this is the familiar binomial
    ``p(1 - p)`` (times n / (n - 1)). A tie contributes 1/4 to the sum of
    squares instead of the 1/2 it adds to the sum, so ties pull ``E[X^2]``
    below ``E[X] = p``. Therefore ``Var(X) = E[X^2] - p^2`` falls below
    ``p - p^2 = p(1 - p)``: a split is a less extreme outcome than a coin flip
    between winning and losing.
    """
    if n < 2:
        return math.nan  # one sample says nothing about the spread
    mean = (wins + ties / 2) / n
    sum_sq = wins + ties / 4
    # max() guards against a tiny negative value from floating-point rounding.
    variance = max(0.0, (sum_sq - n * mean * mean) / (n - 1))
    return math.sqrt(variance / n)


# --- Equity against a range ------------------------------------------------


def equity_vs_range(
    hero: Sequence[Card],
    villain_range: Sequence[Combo],
    board: Sequence[Card] = (),
    samples: int = 10_000,
    seed: int | None = None,
    rng: random.Random | None = None,
    deadline: float | None = None,
) -> EquityResult:
    """Estimate the hero's equity against a villain hand drawn from a *range*.

    A range is a list of combos (two-card hands) the villain might hold, each
    equally likely *a priori*. ``RANDOM_RANGE`` (all 1,326) means "any hand".

    Card removal. The villain cannot hold a card we can see (our hole cards
    or the board). So the combos that use a known card are dropped first, and
    the villain is drawn uniformly from the *live* combos that are left. This
    is just conditioning: P(combo | it avoids the known cards) is uniform over
    the live combos. Example: holding As Ah, the villain has only one combo of
    AA left (Ac Ad) instead of six, so AA is much less likely for them.

    Each sample: draw a live combo uniformly, deal the rest of the board from
    the cards nobody holds, compare hands (win 1, split 1/2, loss 0). Equity
    and ``std_error`` are the mean and its standard error, exactly as in
    :func:`monte_carlo_equity`. Against a single-combo range this is the same
    quantity that :func:`monte_carlo_equity` estimates.

    On the river (5 board cards) nothing is left to deal, and a sample is
    just a villain combo. If the range has no more live combos than
    ``samples``, we loop over every live combo once instead: the answer is
    *exact*, ``std_error`` is 0 and ``samples`` is the number of live combos
    (at most C(45, 2) = 990 against a random hand, about 4 ms). Otherwise we
    sample villain combos as usual (with the hero's hand evaluated once).

    Randomness comes from ``rng`` if given (a bot passes its own seeded
    generator), else from a private ``random.Random(seed)``. ``deadline`` is
    an optional ``time.perf_counter()`` value: sampling stops early once it
    has passed. A deadline makes the result depend on machine speed, so it is
    not reproducible; leave it ``None`` for experiments.

    Raises ``ValueError`` for a bad hand/board, ``samples < 1``, or a range
    with no live combos.
    """
    return equity_vs_nested_ranges(
        hero, [villain_range], board, samples, seed=seed, rng=rng, deadline=deadline
    )[0]


def equity_vs_nested_ranges(
    hero: Sequence[Card],
    ranges: Sequence[Sequence[Combo]],
    board: Sequence[Card] = (),
    samples: int = 10_000,
    min_samples: int = 30,
    seed: int | None = None,
    rng: random.Random | None = None,
    deadline: float | None = None,
) -> list[EquityResult]:
    """Equity against several ranges at once, where ``ranges[1:]`` are subsets of ``ranges[0]``.

    Typical use: ``ranges[0]`` is everything the villain might hold, and the
    later ranges are the parts of it that would *continue* against bets of
    different sizes. We need the equity against each of them.

    The trick (rejection sampling): draw ``samples`` villain combos
    uniformly from the widest range. The draws that happen to land inside a
    sub-range S are a uniform sample from S, because every combo of S had the
    same chance of being drawn. So one simulation answers every sub-range:
    the equity against S is the average payoff over the samples that fell in
    S. The price is precision: if S holds a fraction q of the wide range, it
    gets only about ``q * samples`` samples, and its standard error is about
    ``1 / sqrt(q)`` times larger. If a sub-range ends up with fewer than
    ``min_samples`` samples, it is topped up with direct draws from S.

    Returns one :class:`EquityResult` per range, in order. On the river, if
    the widest range has at most ``samples`` live combos, every result is
    exact (see :func:`equity_vs_range`). Raises ``ValueError`` if a
    later range is not a subset of the first, or any range has no live combos.
    """
    hero, board = list(hero), list(board)
    _validate_range_args(hero, board, samples)
    known = set(hero) | set(board)
    live = [_live_combos(r, known) for r in ranges]
    if not live or not live[0]:
        raise ValueError("the villain range has no combos left after removing known cards")
    widest = set(live[0])
    if any(not widest.issuperset(r) for r in live[1:]):
        raise ValueError("every later range must be a subset of the first one")
    if any(not r for r in live):
        raise ValueError("a range has no combos left after removing known cards")

    if len(board) == 5 and len(live[0]) <= samples:
        return [_exact_river(hero, board, r) for r in live]

    rng = rng if rng is not None else random.Random(seed)
    members = [set(r) for r in live]
    tallies = [[0, 0, 0] for _ in live]  # per range: [samples, wins, ties]
    for combo, outcome in _sample_outcomes(hero, board, live[0], samples, rng, deadline):
        for member, tally in zip(members, tallies):
            if combo in member:
                _add_outcome(tally, outcome)
    for combos, tally in zip(live, tallies):
        if tally[0] < min_samples:
            extra = min_samples - tally[0]
            for _, outcome in _sample_outcomes(hero, board, combos, extra, rng, deadline):
                _add_outcome(tally, outcome)
    return [_result(*tally) for tally in tallies]


_LOSS, _TIE, _WIN = 0, 1, 2


def _validate_range_args(hero: list[Card], board: list[Card], samples: int) -> None:
    """Raise ``ValueError`` unless hero, board and sample count make sense."""
    if len(hero) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hero)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must hold 0, 3, 4 or 5 cards, got {len(board)}")
    if len(set(hero + board)) != len(hero) + len(board):
        raise ValueError(f"cards must all be distinct: {[str(c) for c in hero + board]}")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")


def _live_combos(combos: Sequence[Combo], known: set[Card]) -> list[Combo]:
    """The combos that avoid every known card, each written low card first, no repeats."""
    live = [c for c in combos if c[0] not in known and c[1] not in known]
    if any(a == b for a, b in live):
        raise ValueError("a combo cannot hold the same card twice")
    # Writing each combo low card first makes (Ks, As) and (As, Ks) the same
    # dict key, so dict.fromkeys drops repeats (it keeps the first-seen order).
    return list(dict.fromkeys(c if c[0] < c[1] else (c[1], c[0]) for c in live))


def _compare(hero_rank: tuple[int, ...], villain_rank: tuple[int, ...]) -> int:
    if hero_rank > villain_rank:
        return _WIN
    return _TIE if hero_rank == villain_rank else _LOSS


def _sample_outcomes(hero, board, combos, n, rng, deadline):
    """Yield ``(villain combo, outcome)`` for up to ``n`` random deals.

    Dealing the run-out: the stub holds every card except the hero's and the
    board's, so it still contains the villain's two cards. We draw
    ``need + 2`` cards from it, throw away the villain's cards if they came
    up, and keep the first ``need`` of the rest. That is a uniform draw from
    the cards nobody holds: ``rng.sample`` gives the start of a random
    shuffle of the stub, deleting two cards from a random shuffle leaves a
    random shuffle of the rest, and at most two cards are deleted, so
    ``need + 2`` drawn cards always leave at least ``need``.
    """
    known = set(hero) | set(board)
    stub = [c for c in FULL_DECK if c not in known]
    need = 5 - len(board)
    river_rank = evaluate(hero + board) if need == 0 else None
    for i in range(n):
        if deadline is not None and i > 0 and i % 32 == 0 and time.perf_counter() > deadline:
            return
        combo = combos[rng.randrange(len(combos))]
        if river_rank is not None:  # complete board: nothing to deal
            yield combo, _compare(river_rank, evaluate([*combo, *board]))
            continue
        drawn = rng.sample(stub, need + 2)
        full_board = board + [c for c in drawn if c not in combo][:need]
        yield combo, _compare(evaluate(hero + full_board), evaluate([*combo, *full_board]))


def _exact_river(hero: list[Card], board: list[Card], combos: list[Combo]) -> EquityResult:
    """Exact equity on a complete board: every live combo once, equally weighted."""
    hero_rank = evaluate(hero + board)
    tally = [0, 0, 0]
    for combo in combos:
        _add_outcome(tally, _compare(hero_rank, evaluate([*combo, *board])))
    n, wins, ties = tally
    return EquityResult(
        equity=(wins + ties / 2) / n,
        std_error=0.0,  # nothing was sampled: this is the exact average
        samples=n,
        wins=wins,
        ties=ties,
        losses=n - wins - ties,
    )


def _add_outcome(tally: list[int], outcome: int) -> None:
    tally[0] += 1
    if outcome == _WIN:
        tally[1] += 1
    elif outcome == _TIE:
        tally[2] += 1


def _result(n: int, wins: int, ties: int) -> EquityResult:
    return EquityResult(
        equity=(wins + ties / 2) / n,
        std_error=_std_error(wins, ties, n),
        samples=n,
        wins=wins,
        ties=ties,
        losses=n - wins - ties,
    )
