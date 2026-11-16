"""Game theory for NutsFinder: a generic ``Game`` protocol, small poker games
(Kuhn, Leduc) and exact best response / exploitability.

Solvers (CFR and variants) are written against :class:`game.Game` and
validated here by exact exploitability before they touch hold'em.
"""

from nutsfinder.gametheory.best_response import (
    BestResponse,
    best_response,
    best_response_value,
    expected_value,
    exploitability,
    nash_conv,
)
from nutsfinder.gametheory.game import CHANCE, Game, Profile, action_probs, infosets, iter_histories
from nutsfinder.gametheory.kuhn import Kuhn
from nutsfinder.gametheory.leduc import Leduc

__all__ = [
    "CHANCE", "Game", "Profile", "action_probs", "infosets", "iter_histories",
    "Kuhn", "Leduc",
    "BestResponse", "best_response", "best_response_value", "expected_value",
    "exploitability", "nash_conv",
]
