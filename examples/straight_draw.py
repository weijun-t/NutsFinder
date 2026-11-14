"""Phase 1 exercise: straights and straight draws on the flop with 9-8.

Holding 9-8 (connectors), the flop can give us:

- a made straight: the 3 flop ranks plus 9 and 8 make 5 in a row;
- an 8-out draw: no straight yet, but 2 different turn ranks would make one
  (open-ended like 7-6 on the flop, or a "double gutshot");
- a gutshot: exactly 1 turn rank makes a straight (4 outs);
- nothing.

Exact answer for a made straight, by hand: the straights that use both 9 and
8 need these 3 other ranks: 5-6-7, 6-7-T, 7-T-J or T-J-Q. That's 4 rank
patterns, each with 4 x 4 x 4 = 64 suit choices, so 4 * 64 = 256 flops out of
C(50,3) = 19,600, or about 1.31%.

The draws are fiddly to count by hand, so the exact numbers below come from
checking every one of the 19,600 possible flops. Then the simulation checks
them again by sampling.

Run:  .venv\\Scripts\\python examples\\straight_draw.py          (new random seed)
      .venv\\Scripts\\python examples\\straight_draw.py 12345    (replay a seed)
"""

import itertools
import math
import random
import sys

from nutsfinder.engine.cards import FULL_DECK, Deck, parse_cards

TRIALS = 100_000
HOLE = parse_cards("9h 8c")  # suits don't matter here: we only look at ranks

CATEGORIES = ["made straight", "8-out draw", "gutshot (4 outs)", "nothing"]


def has_straight(ranks: set[int]) -> bool:
    """True if the ranks contain 5 in a row. The ace (14) also counts as 1."""
    if 14 in ranks:
        ranks = ranks | {1}
    return any(all(r in ranks for r in range(low, low + 5)) for low in range(1, 11))


def classify(flop) -> str:
    """Put a flop into one of CATEGORIES (for our hole cards)."""
    ranks = {card.rank for card in HOLE} | {card.rank for card in flop}
    if has_straight(ranks):
        return "made straight"
    # Which turn ranks would complete a straight?
    completing = [r for r in range(2, 15) if has_straight(ranks | {r})]
    if len(completing) >= 2:
        return "8-out draw"
    if len(completing) == 1:
        return "gutshot (4 outs)"
    return "nothing"


# --- Exact: check every possible flop -----------------------------------
stub = [card for card in FULL_DECK if card not in HOLE]  # the 50 unseen cards
exact_counts = dict.fromkeys(CATEGORIES, 0)
for flop in itertools.combinations(stub, 3):
    exact_counts[classify(flop)] += 1
total_flops = math.comb(50, 3)
assert sum(exact_counts.values()) == total_flops

# --- Simulation ----------------------------------------------------------
# Use the seed given on the command line, or pick a fresh random one.
# Printing it means any run can be replayed exactly.
seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(2**32)
rng = random.Random(seed)

sim_counts = dict.fromkeys(CATEGORIES, 0)
for _ in range(TRIALS):
    deck = Deck(seed=rng.getrandbits(64))  # fresh shuffled 52-card deck
    deck.remove(HOLE)                      # we can see our own 2 cards
    sim_counts[classify(deck.deal(3))] += 1

# --- Report --------------------------------------------------------------
print(f"hole cards: {' '.join(str(c) for c in HOLE)}   seed: {seed}   trials: {TRIALS:,}\n")
print(f"{'':18} {'exact':>18} {'simulated':>20} {'off by (SE)':>12}")
for name in CATEGORIES:
    exact = exact_counts[name] / total_flops
    p_hat = sim_counts[name] / TRIALS
    std_error = math.sqrt(p_hat * (1 - p_hat) / TRIALS)
    off_by = (p_hat - exact) / std_error if std_error else float("nan")
    print(
        f"{name:18} {exact_counts[name]:>6,}/{total_flops:,} = {exact:6.2%}"
        f"   {p_hat:6.2%} +/- {2 * std_error:.2%}   {off_by:+8.2f}"
    )
