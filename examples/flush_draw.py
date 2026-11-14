"""Phase 1 exercise: P(flopping a flush draw with two suited hole cards).

Exact answer: C(11,2) * C(39,1) / C(50,3) = 2145 / 19600 ~ 10.94%.
This script checks it by simulation and reports a standard error.

Run:  .venv\\Scripts\\python examples\\flush_draw.py          (new random seed)
      .venv\\Scripts\\python examples\\flush_draw.py 12345    (replay a seed)
"""

import math
import random
import sys
from math import comb

from nutsfinder.engine.cards import Deck, Suit, parse_cards

TRIALS = 100_000

hole = parse_cards("As Ks")
# Use the seed given on the command line, or pick a fresh random one.
# Printing it means any run can be replayed exactly.
seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(2**32)
rng = random.Random(seed)

hits = 0
for _ in range(TRIALS):
    deck = Deck(seed=rng.getrandbits(64))  # fresh shuffled 52-card deck
    deck.remove(hole)                      # we can see our own 2 cards
    flop = deck.deal(3)
    spades = sum(card.suit == Suit.SPADES for card in flop)
    if spades == 2:                        # exactly 2 more = flush draw
        hits += 1

p_hat = hits / TRIALS
std_error = math.sqrt(p_hat * (1 - p_hat) / TRIALS)
exact = comb(11, 2) * comb(39, 1) / comb(50, 3)

print(f"seed:      {seed}")
print(f"simulated: {p_hat:.4%}  +/- {2 * std_error:.4%} (95% CI)")
print(f"exact:     {exact:.4%}")
print(f"difference in standard errors: {(p_hat - exact) / std_error:+.2f}")
