# NutsFinder plan

**Goal:** build a heads-up no-limit Texas Hold'em bot from scratch, in tiers, and use it to learn the probability, statistics and game theory used in quant roles.

**What counts as success:** not "the bot wins", but that every claim of the form "bot X beats bot Y" comes with the maths behind it and a confidence interval.

## Rules for myself

1. No claim without a confidence interval.
2. A new bot tier only counts if it beats the previous tier with statistical significance.

## Overview

| Phase | Build | Learn | Done when |
|---|---|---|---|
| 0 Setup | Repo, virtual environment, pytest, GitHub Actions CI | - | Tests run on every push |
| 1 Cards and equity | Card, Deck, 7-card hand evaluator, Monte Carlo equity calculator | Combinatorics, conditional probability, outs, standard error | A♠K♠ vs 9♥8♥ preflop = 62.1%, and 60.6% on a K♥7♥2♠ flop |
| 2 Game engine | Heads-up rules, Bot interface, seeded randomness, hand logs | - | Chips are conserved over 100k random hands |
| 3 Measurement | Baseline bots, match runner, statistics report | Variance, confidence intervals, sample size, optional stopping | Every result is reported as an interval |
| 4 EV bot | `PotOddsBot`, `EquityBot` | EV, pot odds, implied odds, equity denial | Beats every baseline significantly |
| 5 Opponent modelling | `BayesBot` | Bayesian updating (Beta-Binomial), priors, credible intervals | Exploits the baselines harder than `EquityBot` |
| 6 Game theory | CFR on Kuhn poker ✅, then Leduc poker, then heads-up hold'em with MCCFR and abstraction | Nash equilibrium, mixed strategies, indifference, bluff ratios | The CFR bot is hard to exploit (measured) |
| 7 Stretch | Hand replayer and play-against-the-bot UI, 8-player tables, public write-up | Kelly criterion, variance reduction | A write-up of results is published |

## Phase details

### Phase 0: Setup
- Python 3.12, a virtual environment, `numpy`, `pytest`, `matplotlib`, `hypothesis`.
- Folder layout: `src/nutsfinder/{engine,bots,eval}`, `tests/`, `docs/`, `results/`.
- GitHub Actions running `pytest` on every push.

### Phase 1: Cards, evaluator, equity
- Build `Card`, `Deck` (seeded shuffle) and a 7-card hand evaluator. Make it correct first, then fast.
- Test the evaluator against known counts: 133,784,560 seven-card hands in total, of which 41,584 are straight flushes.
- Build a Monte Carlo equity calculator that also reports its standard error.
- Maths: 1,326 starting-hand combos and 169 hand types; outs; the "rule of 2 and 4" and why it is only an approximation; why standard error shrinks like 1/√n (1,000 samples gives a standard error of about 1.6 percentage points).
- Exercise: work out P(flopping a flush draw with two suited cards) by hand, then confirm it by simulation.

### Phase 2: Game engine
- Heads-up no-limit hold'em: blinds (the button posts the small blind and acts first preflop), betting rounds, minimum raises, all-ins, showdown, split pots.
- A `Bot` interface: `act(state, hole_cards) -> action`. Bots never see the opponent's cards or the deck.
- Everything is reproducible from a seed. Every hand is logged.
- Fuzz test: total chips are unchanged after 100k random hands.
- Decisions: chips are integers with SB = 50, BB = 100, and both stacks reset to 10,000 (100bb) every hand.
  Raise amounts mean "raise to X". Any whole-number size is legal. Minimum raise to = current bet + last raise increment;
  an all-in for less than a full raise does not reopen the betting.

### Phase 3: Baselines and measurement
- `RandomBot`, `CallStationBot`, `AlwaysRaiseBot`.
- Match runner reporting win rate in bb/100, standard error and a 95% confidence interval.
- Duplicate matches: play every deal twice with seats swapped, so card luck cancels out.
- Maths: why poker results are so noisy; how many hands it takes to detect a small edge; why stopping a run early "because it looks good" is cheating.
- Sanity check: a bot playing a copy of itself must come out at about 0.

### Phase 4: EV bot
- `PotOddsBot`: call when equity > pot odds = call / (pot + call); bet for value with strong hands.
- `EquityBot`: Monte Carlo equity against an assumed range of opponent hands, plus bet sizing.
- Every decision is explainable: with debug on, log equity, pot odds and the EV of each action.
- Maths: EV, implied odds, minimum defence frequency, equity denial (the smallest bet that makes a draw's call wrong is b/P > e/(1−2e)).
- Why a pure equity bot is exploitable: it never bluffs and it folds to pressure.

### Phase 5: Opponent modelling
- Track each opponent's VPIP (how often they voluntarily put money in), aggression and fold-to-bet rate.
- Maths: Beta-Binomial updating, priors, credible intervals, and why small samples mislead.
- Exploit the baselines harder, then check whether the exploiting bot can be exploited back.

### Phase 6: Game theory
- Kuhn poker with CFR: done, see `examples/kuhn_cfr.py`. It reproduces the theoretical game value of −1/18, and bluffing emerges without being programmed.
- Leduc poker with CFR.
- Heads-up hold'em: abstraction (group hands into buckets by equity, allow only a few bet sizes) plus Monte Carlo CFR (MCCFR).
- Maths: Nash equilibrium, mixed strategies, indifference, and the river bluff ratio B/(P + 2B) (1/3 of bets are bluffs for a pot-sized bet).
- Measure how exploitable the bot is with an approximate best response.

### Phase 7: Stretch
- Hand replayer that shows the bot's reasoning, and a mode to play against the bot.
- 8-player tables (no clean equilibrium theory, so this is where it gets hard).
- Variance reduction, Kelly sizing and bankroll simulation.
- A public write-up of the results. This is the part recruiters read.

## How to use the subagents

The agent definitions live in `.claude/agents/`, and Claude Code picks them up automatically.

| Agent | Use it for |
|---|---|
| `prob-tutor` | Whenever I'm unsure about a formula. It says "Wrong" first, then derives the right answer and sets an exercise. |
| `sim-analyst` | Before I believe any bot is better. Runs matches and reports confidence intervals and significance. |
| `code-reviewer` | After every phase. Hunts for information leaks, chip-conservation bugs and statistics errors. |
| `strategy-researcher` | Before Phase 6. Short, cited notes on Leduc poker, MCCFR and abstraction. |
| `engine-dev`, `bot-builder` | Writing the code: the engine and evaluator (`engine-dev`) and the bots (`bot-builder`), maths included. |

## Current status

- [x] Plan and subagents
- [x] Kuhn poker CFR example
- [x] Phase 0: setup
- [x] Phase 1: cards, evaluator, equity
- [x] Phase 2: game engine (rules, Bot interface, seeded runner, hand logs; chips conserved over 100k random hands)
- [x] Phase 3: baselines and measurement (RandomBot, CallStationBot, AlwaysRaiseBot; bb/100 with SE, 95% CI and p-value; duplicate matches; CI coverage checked at 94.5%; results in `results/baselines.md`)
- [x] Phase 4: EV bots (PotOddsBot and EquityBot both beat every baseline significantly; EquityBot beats PotOddsBot by +267 bb/100, 95% CI [+238, +296]; tuned on seed 777, official run on a fresh seed; results in `results/phase4.md`)
- [x] Phase 5: opponent modelling (BayesBot: Beta-Binomial model of VPIP, aggression, fold-to-bet by size and showdown bluff rate; beats EquityBot by +61.0 bb/100, 95% CI [+48.4, +73.5], 150 sessions of 500 hands, pre-registered; exploits CallStation/AlwaysRaise/PotOdds harder than EquityBot by +190/+333/+155 bb/100, no gain vs RandomBot (nothing to exploit); seed 20261117, commit 5df9721; results in `results/phase5.md`)
- [x] Phase 6: game theory (Kuhn and Leduc solved exactly with CFR/CFR+/DCFR, Leduc value -0.08561; external-sampling MCCFR validated on Leduc; hold'em abstraction: 169 preflop classes, 50 exact-E[HS] buckets per street, fold/call/3bb open/pot/all-in, 57,304 info sets; blueprint trained 13.2M iterations, seed 7, commit 902376d). **Tier test (pre-registered): CFRBot beats BayesBot by +68.2 bb/100, 95% CI [+58.0, +78.3]**, 160 sessions of 500 hands, seed 202611186, commit d8f494f. **Exploitability (local best response, pre-registered, seed 202611188):** under the primary rule (preflop = ev) CFRBot is less exploitable than BayesBot (difference +1,986 mbb/g [+1,563, +2,408]) and EquityBot (+1,740 [+1,333, +2,146]); under the paper's always-call rule it is less exploitable than EquityBot (+2,609 [+2,190, +3,027]) but not detectably less than BayesBot (+479 [-129, +1,088]), so "less exploitable than BayesBot" holds only under the chosen rule. In absolute terms it is still very exploitable: LBR wins at least 1.1 bb/hand (always-fold loses 0.75). Results in `results/phase6.md` and `results/phase6_lbr_2026-11-17.md`.
- [ ] Deferred until every bot is finished: "can the exploiter be exploited back" test (an opponent that switches style mid-session, against BayesBot and the later bots)
