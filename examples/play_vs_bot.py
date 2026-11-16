"""Play heads-up against any NutsFinder bot in the terminal.

Run:  .venv\\Scripts\\python examples\\play_vs_bot.py --bot bayes
      (bots: random, call, raise, potodds, equity, bayes; default bayes)

Type f (fold), x (check), c (call), or a number to bet/raise TO that many
chips (e.g. 300). Type r to see what BayesBot has learned about you so far
(BayesBot only). Type q to quit. Blinds are 50/100 and stacks reset to
10,000 every hand. You are seat 0; the button alternates.
"""

import argparse
import random

from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import BIG_BLIND
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.hand_log import format_log
from nutsfinder.engine.runner import play_hand


class Quit(Exception):
    pass


class HumanBot(Bot):
    """A "bot" that asks you at the keyboard. It sees exactly what bots see."""

    def __init__(self, opponent=None):
        self.opponent = opponent

    def act(self, state, hole_cards):
        legal = state.legal
        me, opp = state.seat, 1 - state.seat
        board = " ".join(str(c) for c in state.board) or "-"
        print(f"\n[{state.street.name}] board: {board}   pot: {state.pot}")
        print(f"you: {' '.join(str(c) for c in hole_cards)}   stack {state.stacks[me]}"
              f"   |   bot stack {state.stacks[opp]}")
        if state.history:
            last = state.history[-1]
            if last.player == opp:
                print(f"bot did: {last.action}")
        options = ["f fold"]
        options.append("x check" if legal.can_check else f"c call {legal.call_amount}")
        if legal.can_bet or legal.can_raise:
            word = "bet" if legal.can_bet else "raise to"
            options.append(f"<number> {word} {legal.min_to}..{legal.max_to}")
        while True:
            choice = input("  " + "  |  ".join(options) + "  > ").strip().lower()
            if choice == "q":
                raise Quit
            if choice == "r":
                show_read(self.opponent)
                continue
            if choice == "f":
                action = Fold()
            elif choice == "x":
                action = Check()
            elif choice == "c":
                action = Call()
            elif choice.isdigit():
                action = Bet(int(choice)) if legal.can_bet else Raise(int(choice))
            else:
                print("  ?")
                continue
            if legal.is_legal(action):
                return action
            print("  not legal right now")


def show_read(bot):
    if isinstance(bot, BayesBot):
        print("\n".join(bot.model.lines()))
    else:
        print("  (only BayesBot keeps a model of you)")


BOTS = {
    "random": lambda seed: RandomBot(seed=seed),
    "call": lambda seed: CallStationBot(),
    "raise": lambda seed: AlwaysRaiseBot(),
    "potodds": lambda seed: PotOddsBot(seed=seed),
    "equity": lambda seed: EquityBot(seed=seed),
    "bayes": lambda seed: BayesBot(seed=seed),
}
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bot", choices=BOTS, default="bayes")
ap.add_argument("--seed", type=int, default=None, help="fix the deals (default: random)")
args = ap.parse_args()

seed = random.randrange(2**32) if args.seed is None else args.seed
rng = random.Random(seed)
opponent = BOTS[args.bot](rng.getrandbits(64))
bots = [HumanBot(opponent), opponent]
print(f"you vs {type(opponent).__name__}, seed {seed}. f/x/c/<number>, r = bot's read on you, q = quit")
total = 0
hand = 0
try:
    while True:
        log = play_hand(bots, seed=rng.getrandbits(64), button=hand % 2)
        hand += 1
        print("\n" + "=" * 50)
        print(format_log(log))
        total += log.result.net[0]
        print(f"you after {hand} hands: {total:+} chips ({total / BIG_BLIND:+.1f} bb)")
        print("=" * 50)
except (Quit, EOFError, KeyboardInterrupt):
    print(f"\nbye. {hand} hands, {total / BIG_BLIND:+.1f} bb")
    if isinstance(opponent, BayesBot):
        show_read(opponent)
