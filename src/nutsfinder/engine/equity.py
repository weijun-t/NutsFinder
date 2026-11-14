"""Monte Carlo equity: estimate how often one hand beats another.

Phase 1, PLAN.md rule 4: the estimator is the project owner's to write. This
module only fixes the interface and the contract that ``tests/test_equity.py``
checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nutsfinder.engine.cards import Card

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
    - All randomness comes from one RNG seeded with ``seed`` (for example
      ``Deck(seed)``); never the global :mod:`random` or numpy state. The same
      arguments with the same non-``None`` seed give an identical result.
    - ``equity`` counts a tie as half a win, and ``std_error`` is the estimated
      standard error of that estimate, so that ``equity +/- 2 * std_error`` is
      roughly a 95% confidence interval for the true equity.
    """
    raise NotImplementedError("Phase 1: yours to write (PLAN.md rule 4)")
