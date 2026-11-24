"""Multiway (2-8 player) bots for the N-player engine (:mod:`nutsfinder.engine.multiway`).

Kept apart from the heads-up bots in :mod:`nutsfinder.bots`, whose behaviour
official results depend on: nothing here changes them.

- :mod:`.equity`: vectorised multiway Monte Carlo equity, exact enumeration oracle,
  preflop table vs k random hands.
- :mod:`.common`: side-pot-aware one-street EVs, sizing, the decision record.
- :class:`MultiPotOddsBot` (tier 1), :class:`MultiEquityBot` (tier 2),
  :class:`MultiBayesBot` (tier 3), :class:`MultiCFRBot` (tier 4, heuristic).

The baselines (``MultiRandomBot``, ``MultiCallStationBot``, ``MultiAlwaysRaiseBot``)
live with the engine in :mod:`nutsfinder.engine.multiway.bots`.
"""

from nutsfinder.bots.multiway.bayes_bot import MultiBayesBot, PooledModels
from nutsfinder.bots.multiway.cfr_bot import MultiCFRBot
from nutsfinder.bots.multiway.common import MultiDecision
from nutsfinder.bots.multiway.equity import exact_multiway_equity, multiway_equity, sample_deals
from nutsfinder.bots.multiway.equity_bot import MultiEquityBot
from nutsfinder.bots.multiway.pot_odds import MultiPotOddsBot

__all__ = [
    "MultiPotOddsBot",
    "MultiEquityBot",
    "MultiBayesBot",
    "MultiCFRBot",
    "MultiDecision",
    "PooledModels",
    "multiway_equity",
    "exact_multiway_equity",
    "sample_deals",
]
