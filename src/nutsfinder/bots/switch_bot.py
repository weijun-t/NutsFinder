"""SwitchBot: plays one style for the first K hands it sees, then another style.

Purpose: the deferred Phase 5/7 test "can the exploiter be exploited back".
BayesBot (tier 3) learns a Beta-Binomial model of its opponent and never
forgets (every hand weighs the same, no decay; see
:mod:`nutsfinder.bots.bayes_bot`, "Known weaknesses"). An opponent that shows
one style long enough for the model to settle, then switches to a style that
punishes the exploit the model has adopted, should cost BayesBot chips until
the new evidence outweighs the old. SwitchBot is that opponent.

Rules
=====
- ``SwitchBot(first, second, switch_after)`` builds one instance of each style
  from the two factories (classes, or ``functools.partial`` of classes, so
  the bot stays picklable for worker processes).
- It counts the hands it has *finished* (calls to :meth:`end_hand`). While
  that count is below ``switch_after`` every decision is delegated to the
  ``first`` style; from then on to the ``second``. So its hands 0 .. K-1 are
  style A and hands K, K+1, ... are style B, where K = ``switch_after``.
- Duplicate matches. Each room of
  :func:`~nutsfinder.eval.match.duplicate_match` has its own SwitchBot
  instance, and each instance plays exactly one hand per pair. Its hand count
  is therefore the pair index, and both rooms switch at the *same pair*:
  pair K is the first style-B pair in both rooms. A session of P pairs
  (2P hands) is "A for pairs 0 .. K-1, B for pairs K .. P-1" in both rooms.
- :meth:`reseed` is forwarded to *both* styles, before every hand, with the
  same seed. A randomised style therefore draws exactly the stream it would
  draw if it were playing alone (``RandomBot`` reseeded with the same seed
  makes the same choices), so "SwitchBot in its A phase" is hand-for-hand
  identical to "style A alone", and "SwitchBot in its B phase" to "style B
  alone", on the same deals (tested). That identity is what lets a run pair
  the switched session with control sessions on the same deals.
- :meth:`end_hand` is forwarded only to the style that played the hand. The
  dormant style sees nothing. For memoryless styles (every baseline bot) this
  changes nothing; a learning style would start its phase with an empty
  memory, which is what "style B from hand K" means here.

What SwitchBot does not do
--------------------------
It never looks at its opponent: the switch time is fixed in advance (not
"when the opponent has adapted"), so it is a fixed, reproducible strategy,
not an adaptive exploiter. That keeps the test honest: it measures what a
*known* style change costs, not how clever the switcher is.
"""

from __future__ import annotations

from collections.abc import Callable

from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card

__all__ = ["SwitchBot"]


class SwitchBot(Bot):
    """Style ``first`` for the first ``switch_after`` hands, then style ``second``.

    Args:
        first: factory of style A (e.g. ``CallStationBot``).
        second: factory of style B (e.g. ``AlwaysRaiseBot``).
        switch_after: K, the number of hands played as style A (>= 0). K = 0
            is style B from the start.
        seed: optional; if given, both styles are reseeded with it now (match
            code reseeds before every hand anyway).
    """

    def __init__(self, first: Callable[[], Bot], second: Callable[[], Bot], switch_after: int,
                 seed: int | None = None) -> None:
        if switch_after < 0:
            raise ValueError(f"switch_after must be >= 0, got {switch_after}")
        self.first = first()
        self.second = second()
        self.switch_after = switch_after
        self.hands_played = 0
        if seed is not None:
            self.reseed(seed)

    @property
    def switched(self) -> bool:
        """True once ``switch_after`` hands have been finished (style B is active)."""
        return self.hands_played >= self.switch_after

    @property
    def active(self) -> Bot:
        """The style that plays the current hand."""
        return self.second if self.switched else self.first

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        return self.active.act(state, hole_cards)

    def end_hand(self, summary: HandSummary) -> None:
        self.active.end_hand(summary)
        self.hands_played += 1

    def reseed(self, seed: int) -> None:
        """Forward ``seed`` to both styles (so each draws the stream it would draw alone)."""
        self.first.reseed(seed)
        self.second.reseed(seed)

    def __repr__(self) -> str:
        return f"SwitchBot({self.first!r} -> {self.second!r} after {self.switch_after} hands)"
