"""Kuhn poker (Kuhn 1950) on the :class:`~nutsfinder.gametheory.game.Game` protocol.

Rules (identical to ``examples/kuhn_cfr.py``)
---------------------------------------------
- Deck J < Q < K, one card each, the third card unused.
- Both players ante 1. Player 0 acts first.
- Actions: ``"p"`` (pass: check, or fold when facing a bet) and ``"b"``
  (bet 1, or call a bet).
- Terminal lines and payoff to player 0 (``+`` means P0 holds the higher card):

  ====== ============================ =========
  line   meaning                      payoff
  ====== ============================ =========
  pp     check, check, showdown       +-1
  bb     bet, call, showdown          +-2
  pbb    check, bet, call, showdown   +-2
  bp     bet, fold                    +1
  pbp    check, bet, fold             -1
  ====== ============================ =========

Histories are tuples of actions, chance first: ``("K", "Q", "p", "b", "b")``
means P0 got K, P1 got Q, P0 checked, P1 bet, P0 called (payoff +2).

Info set keys are strings: own card + betting so far, e.g. ``"Kpb"`` (P0
holding K after check-bet). Player is determined by the betting length
(even = P0), so keys of the two players never collide. There are 12 info
sets: 6 per player.

Equilibrium (Kuhn 1950): a one-parameter family for P0 with alpha in
[0, 1/3] and a unique P1 strategy; P0's value is -1/18. See
``tests/test_gametheory_games.py`` for the full family.
"""

from __future__ import annotations

from nutsfinder.gametheory.game import CHANCE

CARDS: tuple[str, ...] = ("J", "Q", "K")
_RANK = {c: i for i, c in enumerate(CARDS)}
_TERMINAL = frozenset({"pp", "bb", "pbb", "bp", "pbp"})

KuhnHistory = tuple[str, ...]


class Kuhn:
    """Kuhn poker. Stateless: all state lives in the (tuple) history."""

    def initial(self) -> KuhnHistory:
        return ()

    def is_terminal(self, h: KuhnHistory) -> bool:
        return "".join(h[2:]) in _TERMINAL

    def utility(self, h: KuhnHistory) -> float:
        line = "".join(h[2:])
        sign = 1.0 if _RANK[h[0]] > _RANK[h[1]] else -1.0
        if line == "pp":
            return sign
        if line in ("bb", "pbb"):
            return 2.0 * sign
        if line == "bp":
            return 1.0
        if line == "pbp":
            return -1.0
        raise ValueError(f"not terminal: {h}")

    def player(self, h: KuhnHistory) -> int:
        if len(h) < 2:
            return CHANCE
        return (len(h) - 2) % 2

    def chance_outcomes(self, h: KuhnHistory) -> list[tuple[str, float]]:
        left = [c for c in CARDS if c not in h]
        return [(c, 1.0 / len(left)) for c in left]

    def infoset_key(self, h: KuhnHistory) -> str:
        return h[self.player(h)] + "".join(h[2:])

    def actions(self, h: KuhnHistory) -> tuple[str, ...]:
        return ("p", "b")

    def next(self, h: KuhnHistory, a: str) -> KuhnHistory:
        return h + (a,)
