"""Vanilla CFR on Kuhn poker (cards J=0, Q=1, K=2; actions p=pass/check/fold, b=bet/call).
Nothing here mentions bluffing. Bluffing appears in the learned strategy anyway."""
import itertools

ACTIONS = "pb"
regret_sum, strategy_sum = {}, {}


def strategy(info, reach):
    r = regret_sum.setdefault(info, [0.0, 0.0])
    pos = [max(x, 0.0) for x in r]                      # regret matching
    total = sum(pos)
    s = [x / total for x in pos] if total > 0 else [0.5, 0.5]
    ss = strategy_sum.setdefault(info, [0.0, 0.0])
    for i in range(2):
        ss[i] += reach * s[i]                           # accumulate average strategy
    return s


def cfr(cards, h, p0, p1):
    """Returns the value of history h for the player about to act."""
    me = len(h) % 2
    if len(h) > 1:                                      # terminal states
        higher = cards[me] > cards[1 - me]
        if h[-1] == "p":
            if h == "pp":
                return 1 if higher else -1              # check-check showdown
            return 1                                    # opponent folded
        if h[-2:] == "bb":
            return 2 if higher else -2                  # bet-call showdown
    info = f"{cards[me]}{h}"
    s = strategy(info, p0 if me == 0 else p1)
    util = [0.0, 0.0]
    for i, a in enumerate(ACTIONS):
        if me == 0:
            util[i] = -cfr(cards, h + a, p0 * s[i], p1)
        else:
            util[i] = -cfr(cards, h + a, p0, p1 * s[i])
    node = sum(s[i] * util[i] for i in range(2))
    opp_reach = p1 if me == 0 else p0
    for i in range(2):                                  # regret = how much better action i would have been
        regret_sum[info][i] += opp_reach * (util[i] - node)
    return node


def average(info):
    ss = strategy_sum[info]
    return ss[1] / sum(ss)                              # probability of b


if __name__ == "__main__":
    deals = list(itertools.permutations(range(3), 2))
    iters = 50_000
    value = 0.0
    for _ in range(iters):
        for d in deals:
            value += cfr(d, "", 1, 1) / len(deals)
    names = "JQK"
    print(f"game value for player 1: {value / iters:+.4f}   (theory: -1/18 = {-1/18:+.4f})\n")
    print("Player 1, first action, P(bet):")
    for c in range(3):
        print(f"  {names[c]}: {average(str(c)):.3f}")
    print("Player 2 after a check, P(bet):")
    for c in range(3):
        print(f"  {names[c]}: {average(f'{c}p'):.3f}")
    print("Player 2 facing a bet, P(call):")
    for c in range(3):
        print(f"  {names[c]}: {average(f'{c}b'):.3f}")
