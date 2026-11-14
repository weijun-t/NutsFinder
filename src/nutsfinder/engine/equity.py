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
from collections.abc import Sequence
from dataclasses import dataclass

from nutsfinder.engine.cards import FULL_DECK, Card
from nutsfinder.engine.evaluator import evaluate

__all__ = ["EquityResult", "monte_carlo_equity"]


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
