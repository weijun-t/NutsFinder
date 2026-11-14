# NutsFinder

A poker bot for heads-up No-Limit Texas Hold'em, built from scratch to learn probability, statistics and game theory.

In poker, "the nuts" is the best possible hand. This bot will not always have it. The project is about how a program should decide what to do when it doesn't know what the opponent holds.

## What it will do

The bot is built in levels. Each level has to beat the one before it, and that has to be proven with statistics, not with a lucky session.

| Level | Bot | How it decides |
|---|---|---|
| 0 | Random / always-call / always-raise | No thinking at all. These are the punching bags. |
| 1 | Pot-odds bot | "Do I win often enough to justify paying to call?" |
| 2 | Equity bot | Simulates thousands of random deals to estimate its chance of winning. |
| 3 | Bayesian bot | Learns each opponent's habits as it plays (e.g. "this player folds a lot"). |
| 4 | CFR bot | Learns by playing itself millions of times, and ends up bluffing at mathematically correct rates. |

## The key ideas, in one line each

- **Equity:** your chance of winning the hand if all the cards were dealt out now.
- **Pot odds:** the share of the final pot you have to pay to call. If your equity is higher, calling makes money in the long run.
- **Expected value (EV):** the average result of a decision if you made it many times.
- **Monte Carlo:** estimating a number by random simulation instead of calculating it exactly.
- **Confidence interval:** the range the bot's true win rate probably lies in. Poker is so noisy that a bot needs tens of thousands of hands before you can trust its results.
- **CFR (Counterfactual Regret Minimization):** after every hand, the bot asks "how much better would each other choice have done?" and shifts towards the choices it regrets not making.

## Try it now

The project is at an early stage. One example already works: a CFR bot for Kuhn poker, a tiny 3-card version of poker. Nobody programs it to bluff, but it learns to bluff anyway.

```
python examples/kuhn_cfr.py
```

It runs in a few seconds with plain Python, nothing to install.

## Project layout

```
NutsFinder/
├── README.md          this file
├── PLAN.md            the full roadmap, phase by phase
├── examples/          small working demos (Kuhn poker CFR)
├── .claude/agents/    AI helper definitions used while building the project
└── src/nutsfinder/    the bot itself (coming in Phase 0)
    ├── engine/        cards, hand ranking, game rules
    ├── bots/          all the bot levels
    └── eval/          matches between bots, and the statistics
```

## Roadmap

See [PLAN.md](PLAN.md) for the full plan. In short:

1. Cards, hand ranking and win-chance calculator
2. Game rules engine
3. Baseline bots and a measuring tool with proper statistics
4. Pot-odds and equity bots
5. Opponent modelling
6. Game theory (CFR)
7. Stretch: a UI to replay hands and play against the bot
