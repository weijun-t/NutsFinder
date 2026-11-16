r"""Watch PotOddsBot and EquityBot explain their decisions.

Run:
    .venv\Scripts\python examples\explain_decisions.py [n_hands] [seed]

Plays a few hands of EquityBot (seat 0) against PotOddsBot (seat 1) with
``debug=True`` and prints every decision: equity (with its standard error)
against the assumed range, pot odds, the EV of every action considered, and
the action taken. The bots log each decision at DEBUG level, so all this
script has to do is switch logging on.
"""

from __future__ import annotations

import logging
import sys

from nutsfinder.bots import EquityBot, PotOddsBot
from nutsfinder.engine.runner import play_hand


def main() -> None:
    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")
    bots = [EquityBot(seed, debug=True), PotOddsBot(seed + 1, debug=True)]
    for i in range(n_hands):
        print(f"=== hand {i + 1}: seat 0 = EquityBot, seat 1 = PotOddsBot, button = seat {i % 2} ===")
        log = play_hand(bots, seed=seed * 1_000 + i, button=i % 2)
        print(f"net chips: {log.result.net}\n")


if __name__ == "__main__":
    main()
