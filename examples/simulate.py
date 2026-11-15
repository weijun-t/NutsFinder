"""Watch two RandomBots play, and measure the result.

Run:
    .venv\\Scripts\\python examples\\simulate.py                  1,000 hands, random seed
    .venv\\Scripts\\python examples\\simulate.py --hands 50000    a longer match
    .venv\\Scripts\\python examples\\simulate.py --seed 123       replay a match exactly
    .venv\\Scripts\\python examples\\simulate.py --show 3         also print the first 3 hands

What to expect: both seats play the same random strategy, so the true win
rate is 0 bb/100. The measured one won't be exactly 0: that's luck (variance),
and the confidence interval tells you how much luck to expect. If the interval
excludes 0 much more often than about 1 run in 20, something is wrong.
"""

import argparse
import math
import random
import time

from nutsfinder.bots.random_bot import RandomBot
from nutsfinder.engine.betting import BIG_BLIND
from nutsfinder.engine.hand_log import format_log
from nutsfinder.engine.runner import play_match

parser = argparse.ArgumentParser(description="RandomBot vs RandomBot")
parser.add_argument("--hands", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=None, help="master seed (default: random)")
parser.add_argument("--show", type=int, default=0, help="print the first N hand logs")
args = parser.parse_args()

# One master seed drives everything (the cards and both bots), and is printed,
# so any run can be replayed exactly with --seed.
seed = args.seed if args.seed is not None else random.randrange(2**32)
bot_seeds = random.Random(seed)
bots = [RandomBot(seed=bot_seeds.getrandbits(64)), RandomBot(seed=bot_seeds.getrandbits(64))]

start = time.perf_counter()
logs = play_match(bots, args.hands, master_seed=seed)
elapsed = time.perf_counter() - start

for log in logs[: args.show]:
    print(format_log(log))
    print()

# Seat 0's result per hand, in big blinds.
results = [log.result.net[0] / BIG_BLIND for log in logs]
n = len(results)
mean = sum(results) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in results) / (n - 1))
se = sd / math.sqrt(n)

showdowns = sum(log.folded is None for log in logs)
chips_ok = all(sum(log.result.final_stacks) == sum(log.starting_stacks) for log in logs)

print(f"seed: {seed}   hands: {n:,}   time: {elapsed:.1f}s ({n / elapsed:,.0f} hands/s)")
print(f"hands reaching showdown: {showdowns / n:.1%}")
print(f"chips conserved in every hand: {chips_ok}")
print(f"seat 0 net: {sum(results):+,.1f} bb")
print(f"seat 0 win rate: {100 * mean:+.1f} bb/100  +/- {200 * se:.1f} (95% CI)")
print(f"std dev per hand: {sd:.1f} bb  (this is why poker results are so noisy)")
