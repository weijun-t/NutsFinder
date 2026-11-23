# Phase 7 research notes: 3 to 8 player tables

These notes are for NutsFinder's next step: NLHE with 3 to 8 seats, SB 50 / BB 100, 100bb stacks, pure Python.
Confidence tags work as in `phase6_research.md`:
- **[rule]**: a published rulebook (Poker TDA).
- **[paper]**: a peer-reviewed paper.
- **[preprint]**: arXiv only.
- **[blog]**: a vendor or press page, not a paper.
- **[memory]**: something I recall but did not re-check.
- **[estimate]**: my own back-of-envelope number.

---

## 0. TL;DR

1. **Write a separate N-player engine** (`engine/multiway/`) and leave the heads-up engine byte-for-byte unchanged.
   Prove the two agree with a differential test at N=2. Then cross-check against **PokerKit** in the test suite only.
2. **Model:** fixed seats, nobody busts, every stack resets to 10,000 each hand, and the button moves one seat per
   hand. That removes dead and missed blinds and seat changes. It also means that **with equal starting stacks,
   side pots can never occur** (every all-in is the same 10,000). Still implement general side pots, because the UI,
   the tests and future variants with unequal stacks need them.
3. **Bots:** the rule bots carry over directly. PotOdds, Equity and Bayes need equity against several opponents
   (Monte Carlo) and per-opponent models. **CFRBot's heads-up blueprint does not transfer.** Use it only as a
   labelled heuristic in pots that start heads-up. The feasible game-theory work is a **6-max preflop-only or
   push/fold MCCFR**, not a Pluribus clone.
4. **Evaluation:** use **seat-rotation duplicate**, with a paired comparison "bot X vs bot Y in the same seat, on the
   same deals, against the same fixed field". Report every result per field. A multiway win rate is a property of
   (bot, field), not of the bot alone.

---

## 1. Engine

### 1.1 Rules (cash game, no-limit, N = 2..8)

**Positions.** Seats go clockwise and the button moves one seat clockwise every hand.
- **N ≥ 3:** SB is the first seat left of the button and BB the second. Preflop the first player to act is the seat
  left of the BB (UTG). Postflop the first player is the first *active* seat left of the button, so the BTN acts last.
- **Names, counted back from the button:**
  - 8-handed: UTG, UTG+1, LJ (MP), HJ, CO, BTN, SB, BB.
  - 6-max: UTG (LJ), HJ, CO, BTN, SB, BB.
  - 3-handed: BTN, SB, BB, where the BTN acts first preflop.
  - Naming conventions vary, so compute the names from `(seat - button) mod N`.
- **Heads-up exception:** the button posts the SB, acts first preflop and acts last postflop
  ([TDA 2015, Rule 33](https://homepokertourney.org/poker-tda-rules.htm) [rule], via a secondary HTML copy because
  pokertda.com was unreachable). This is exactly what the current engine does.

**Blinds.**
- SB posts min(50, stack) and BB posts min(100, stack). A short blind is all-in.
- The preflop "current bet" is the BB amount, but the opening raise increment is the full BB, even if the BB posted
  less. The HU engine already does this.
- Dead and missed blinds, the dead button and "no BB twice in a row" exist only because players leave and bust (TDA
  Rules 31 and 33). Under the fixed-seat reset model they never arise. Say so in the docstring and do not implement them.

**Action order and street close.** A street closes when every player who is neither folded nor all-in meets both
conditions:
- they have acted since the last bet or raise that changed what they owe;
- their street bet equals the current bet.

Two more rules:
- The BB keeps its option after limps. `needs_to_act` stays True after a forced post, as in the HU engine.
- If at most one player who still has chips remains, and nobody owes chips, betting is over for the hand and the
  board runs out.

**Min-raise and reopening.**
- min raise-to = current bet + the last *full* increment. The increment starts at 1 BB on every street, and a raise
  must add at least the largest previous full bet or raise of the round (TDA Rule 43).
- An all-in may be for less.
- A short all-in does not change the increment, and does not reopen betting for a player who has already acted.
- **Multiway subtlety (TDA Rule 44):** several short all-ins *add up*. A player who has already acted may raise again
  if, when the action comes back, they face at least one full raise above the amount they last acted on.
  - Example: A bets 100, B shoves to 150 and C shoves to 210. A now faces 110 more, which is at least 100, so A may
    re-raise. The min raise-to is 210 + 100 = 310.
  - The HU engine sets `raise_allowed` per action (`increment >= last_raise`). That is equivalent heads-up but
    **wrong multiway**.
- Implementation: store `acted_at[i]`, the current bet when seat i last acted. When it is i's turn, i may raise iff:
  - i has not acted this street (or only posted a blind), or `current_bet - acted_at[i] >= last_full_increment`;
  - and i has chips beyond the call;
  - and **at least one other non-folded player still has chips**. This replaces `stacks[opponent] > 0`.
- Rooms differ on this rule. Pick TDA and write the choice in the docstring.

**Uncalled bet.** When a street closes, or everyone else folds, find the highest total contribution
`c_max = max_i total_bets[i]` over **all** players, folded ones included. Return `c_max - c_second` to its owner.
Folded players count because a folded caller did match part of the bet. Ties mean nothing is returned.
After this step the top level is always shared by at least two players, or the hand ended by folds.

### 1.2 Side pots: the algorithm

The inputs are each seat's final `total_bets` after the uncalled return, and the set of live (non-folded) players.

```
levels = sorted({c[i] for i live})                  # distinct contribution levels of live players
prev = 0
for L in levels:
    amount   = sum(min(c[j], L) - min(c[j], prev) for every seat j, folded or not)
    eligible = [i live with c[i] >= L]
    pots.append((amount, eligible));  prev = L
```

- Chips from folded players fall into whichever layers they reach. Because of the uncalled-bet step, no chips are
  left above the top level.
- Each pot is awarded separately. The best hand among `eligible` wins, and a tie splits the pot.
- **Odd chip:** split each pot separately (TDA Rules 19 and 20). The odd chip goes to the first winner clockwise
  from the button. With more than one odd chip, one each goes to winners in that order.
- Heads-up, the first seat left of the button is the BB, so the current `split_pot` rule is the N=2 case.

**Worked example.** 6-max with the button on seat 0, so SB = 1, BB = 2, UTG = 3, HJ = 4 and CO = 5.
- Stacks: UTG 1,005, HJ 3,000, CO 6,000, BTN 10,000.
- Preflop: UTG shoves 1,005 (a full raise), HJ shoves 3,000, CO shoves 6,000 and BTN shoves 10,000. SB folds (50 in)
  and BB folds (100 in).
- Uncalled: c_max = 10,000 and c_second = 6,000, so 4,000 goes back to BTN.
- Contributions: SB 50, BB 100, UTG 1,005, HJ 3,000, CO 6,000, BTN 6,000. The total is 16,155.

| pot | layer | amount | eligible |
|---|---|---|---|
| main | 0 to 1,005 | 4 × 1,005 + 50 + 100 = 4,170 | UTG, HJ, CO, BTN |
| side 1 | 1,005 to 3,000 | 3 × 1,995 = 5,985 | HJ, CO, BTN |
| side 2 | 3,000 to 6,000 | 2 × 3,000 = 6,000 | CO, BTN |

The three pots sum to 16,155. At showdown, UTG has the best hand, CO and BTN tie, and HJ is worst.
- Main: 4,170 to UTG.
- Side 1: 5,985 split between CO and BTN, 2,992 each plus 1 odd chip. Going clockwise from the button (1, 2, 3, 4, 5,
  0), CO (seat 5) comes before BTN (seat 0), so CO gets 2,993.
- Side 2: 3,000 each.
- Nets: UTG +3,165, HJ −3,000, CO −7, BTN −8, SB −50, BB −100. They sum to 0.

This is a good first hand-written test.

**Showdown order (TDA Rule 16).** The last bettor or raiser on the final round shows first. If nobody bet, the first
active seat left of the button shows first. Once betting is over with someone all-in, all hands are tabled at once
(Rule 15).

For bots, recommend **revealing every live hand at showdown** and never revealing after a fold. That is equivalent to
"nobody mucks". It keeps `HandSummary.shown_cards` simple, and the Bayes model already assumes it.

### 1.3 What must change relative to the 2-player code

| place | heads-up assumption | N-player replacement |
|---|---|---|
| `betting.HandState` | 2-tuples; `folded: int \| None`; `big_blind_seat = 1 - button` | `n` plus N-tuples; `folded: tuple[bool, ...]`; `sb_seat` and `bb_seat` from n; `acted_at`; `last_aggressor` |
| `legal_actions` | `opponent = 1 - p`; `stacks[opponent] > 0` | "any other live player with chips"; the reopening test from 1.1 |
| `_pass_turn` | tries `(first, 1 - first)` | walks clockwise from `first`, skipping folded and all-in seats |
| `_fold` | a fold ends the hand | a fold ends the hand only when one live player is left |
| `uncalled_bet` / `contested_pot` | `2 * min` | 1.1 and 1.2: c_max − c_second, then layered pots |
| `showdown.settle` | one pot, `_best` over 2 seats | per-pot winners and odd chips; `Settlement.winners` becomes per-pot |
| `runner.play_hand` | `len(bots) != 2`; deals seat 0 then 1 | n bots; deal by seat index from the seed (keep the "same seed means same cards per seat" property) |
| `bot.PublicState` / `HandSummary` | 2-tuples; `shown_cards` is a pair | N-tuples, `n_players`, `positions`, `folded` tuple, public `pots` |
| bots/common.py, bayes, cfr, eval, table | many `1 - seat` (about 94 matches of `1 - `, `== 2` and `len(...) != 2` in `src/`) | per-opponent loops |

**Generalise or fork?** I recommend a fork: a new `engine/multiway/` package that reuses `cards`, `evaluator` and
`actions` unchanged.

Reasons against generalising:
- **Reproducibility.** Every official HU result depends on the exact HU code path, including RNG consumption in
  bots that read `PublicState` fields. Changing the types (`folded` from int to tuple, N-tuples) would change the bot
  inputs and risk silent drift.
- The HU engine is small (about 800 lines) and well documented. Duplicating it costs little.
- Generalising in place would mean re-running and re-validating every phase's results.

Costs of forking:
- Two engines to maintain, and bug fixes do not propagate.

Mitigations:
1. A **differential test**: for N=2 and thousands of seeds, the multiway engine with the HU dealing order must
   produce the same action records, refunds and nets as the HU engine, using deterministic bots. Map the bots
   through a small adapter.
2. Freeze the HU path with golden hand-log hashes for a few official seeds, if those tests do not already exist.
3. Only later, and optionally, make the HU engine a thin wrapper, and only if the golden hashes stay identical.

### 1.4 Test strategy

- **Invariants after every transition:**
  - `sum(stacks) + pot == sum(starting_stacks)`;
  - nets sum to 0;
  - `to_act` is never folded or all-in;
  - the street always matches the board (as in the HU engine).
- **Property tests**, with random legal action sequences for N in 2..8, random unequal stacks and random hole cards:
  - side pots sum to the pot;
  - every eligible set is nested and shrinks with the level;
  - no player wins more than `Σ_j min(c_j, c_i)`;
  - the uncalled amount is ≥ 0, and is 0 at showdown unless one player was strictly the top contributor;
  - the minimum `min_to` satisfies the TDA rules.
  - [Hypothesis](https://hypothesis.readthedocs.io/) suits this, but it is not installed in `.venv`. A seeded
    random generator also works.
- **Reference implementation: PokerKit.** [Kim, arXiv 2308.07327](https://arxiv.org/abs/2308.07327) [preprint; IEEE
  ToG submission]. It is an MIT-licensed, multi-variant, N-player Python engine from the U. Toronto CPRG that
  exposes side pots (`pots`) and the showdown order
  ([docs](https://pokerkit.readthedocs.io/en/stable/simulation.html)). Use it as a **test-only dependency**:
  - replay the same action sequence in both engines;
  - compare legal ranges, stacks, pots and payouts.
  - I did not verify PokerKit's seat-index convention (I believe index 0 is the first blind poster [memory]), its
    heads-up blind order, or whether it implements cumulative short all-in reopening. Map carefully, and investigate
    any disagreement on reopening before assuming either side is wrong.
- **Hand-history examples:**
  - The PHH format ([Kim, arXiv 2312.11753](https://arxiv.org/abs/2312.11753) [preprint, IEEE CoG 2024];
    [spec](https://github.com/uoftcprg/phh-std)) ships a dataset that **includes the Pluribus 6-max hands**. Replay
    their action lines through the engine and check that the final stacks match. This gives free real-world 6-max
    test vectors.
  - Add a handful of hand-written cases: the worked example above, a BB option after limps, the cumulative
    short-all-in reopen, a 3-way split with an odd chip, and everyone folding to the BB.

---

## 2. Bots

| bot | how it generalises |
|---|---|
| Random, CallStation, AlwaysRaise | Unchanged logic: they only read `legal`. A field of AlwaysRaise copies plays a multiway all-in almost every hand, with huge variance (see 3). |
| PotOdds | Call iff e > c/(P+c), where e is the equity against **k random hands**: P(best of k+1), with ties counting 1/(number tied). Precompute a 169 × 7 preflop table (k = 1..7) once by Monte Carlo. Postflop, sample k hands without replacement each time. Equity falls fast with k: AA is about 85% against 1 random hand and roughly 50% against 5 [memory, approximate]. |
| Equity | One range per opponent, from that opponent's own bets and raises. Multiway Monte Carlo: sample one hand from each range with card-removal rejection, then the board, and score win, share or lose. The cost is about (k+1) evaluations per sample, so keep `samples` × k bounded. **The fold model needs everyone to fold:** f_all = Π f_i, assuming independent opponents (optimistic, because a squeeze can come behind). EV(bet) uses equity against the expected caller set. With equal reset stacks no side pots appear, so effective-stack logic can use the smallest covering stack. Known gap: players still to act behind are ignored. |
| Bayes | One Beta-Binomial set **per opponent seat**. Seats are fixed, so seat identity is stable within a room. Condition the stats on the number of opponents in the pot (headsup or multiway): folding to a bet 4-way is a different event. Data per opponent per spot arrives about (N−1) times slower, so use **partial pooling**: a shared population prior whose strength is fitted across all opponents. Information comes mostly from showdowns, and those are rarer multiway. |
| CFR | See below. |

**Why the HU blueprint does not transfer.** It is an equilibrium of a *different game*:
- the tree has 2 seats;
- the preflop ranges come from the HU tree (the SB opens most hands);
- the postflop strategy is a best response to those ranges.

A 6-max pot that reaches the flop heads-up has much tighter, position-dependent ranges and different pot and stack
ratios. Mapping it onto the HU tree applies the right algorithm to the wrong beliefs. On top of that:
- **No guarantee is left with 3 or more players.** CFR does not necessarily converge to a Nash equilibrium.
- Even an exact Nash strategy can lose, because the others' choices can transfer value between players.
- Sources: Pluribus authors ([Meta AI blog](https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/)
  [blog]) and [Abou Risk & Szafron, AAMAS 2010](https://www.cs.ualberta.ca/~games/poker/publications/AAMAS10.pdf)
  [paper]. Despite the lost guarantee, their CFR agents won the 2009 ACPC 3-player limit event.
- Ganzfried's 3-player Kuhn study notes that results "depend heavily" on the other agents' choices
  ([arXiv 1804.04789](https://arxiv.org/abs/1804.04789) [preprint]).

**What Pluribus did** ([Brown & Sandholm, Science 2019](https://www.science.org/doi/10.1126/science.aay2400) [paper];
compute figures from the Meta blog and the CMU press release):
- **Game:** 6-max, 100bb, with stacks reset each hand.
- **Blueprint:** external-sampling MCCFR self-play with linear weighting early and regret-based pruning, plus card
  and action abstraction.
  - It took 8 days on a 64-core server: about 12,400 core-hours, under 512 GB of RAM, under $150 of cloud compute.
  - Preflop it mostly plays the blueprint [memory].
  - Later streets use **depth-limited real-time search**. At the leaves, each player picks one of 4 continuation
    strategies (blueprint, or the blueprint biased towards fold, call or raise).
  - Live play used 2 CPUs and under 128 GB.
- **Results:** in 10,000 hands of 5 humans + 1 AI it won about 5 bb/100 (p = 0.021, with AIVAT). In 1 human + 5 AI
  play, the humans lost about 2.3 bb/100 in aggregate.
- **Feasibility for NutsFinder:** pure Python is perhaps 50–100× slower per iteration than optimised C++ [estimate],
  on about 8 laptop cores instead of 64. A comparable blueprint is roughly **3–4 orders of magnitude** out of reach.
  So is real-time search for 6 players.

**What is feasible** [estimate], in order of value per effort:
1. **CFRBot-M fallback (1 day).** If the hand is heads-up *from the start* (everyone else folded preflop before
   acting, e.g. BTN vs BB), use the HU blueprint with seats mapped by position, tagged `mode=hu-blueprint`. In every
   other spot fall back to EquityBot-M. Treat it as a heuristic and measure it, not trust it.
2. **N-player push/fold at short stacks (10–15bb) (3–5 days).**
   - Each position has a 169-class jam range, and each later player has a call range given the jams before them.
   - Ganzfried & Sandholm solved the 3-player tournament version with fictitious-play variants
     ([AAMAS 2008](https://aamas.csc.liv.ac.uk/Proceedings/aamas08/proceedings/pdf/paper/AAMAS08_0313.pdf) [paper];
     ε verified ex post at IJCAI 2009). The chip-EV cash version is simpler.
   - The cost is in the multiway all-in equity terminals. Sample them (outcome or chance sampling) instead of
     tabulating them.
   - It is small, verifiable and teaches the "no guarantee" lesson: measure ε by best response per player.
3. **6-max preflop-only MCCFR (5–8 days, optional).**
   - Actions: fold, call, a 2.5bb open, a 3x re-raise and all-in, with a cap of about 4 raises.
   - Postflop terminals are valued as check-down equity × pot (optionally with a realisation factor), or sampled.
   - The check-down assumption biases towards loose calls, so state that clearly.
   - Use it preflop only. EquityBot-M plays postflop.
4. A full 3-handed postflop abstraction is possible but coarse and slow. I would skip it.

**Recommended plan:**
- Ship EquityBot-M and BayesBot-M first, since they carry the project's edge.
- Add CFRBot-M as option 1, then option 2 if there is time.
- Option 3 is a stretch.

---

## 3. Evaluation with N players

**Why multiway bb/100 depends on the field:**
- A seat's profit comes from *all* other seats, so "X beats the field" is a property of the (X, field) pair.
- Results are **non-transitive and field-dependent**. For example, a tight bot profits when two maniacs build pots
  and bust each other, and loses in a field of nits.
- With identical field copies there is implicit "team" play: correlated strategies that act like soft collusion,
  such as squeezes by a block of AlwaysRaise copies. Instances share no information (they are separate objects in
  separate rooms), which is the "collusion-free" assumption, but strategy correlation cannot be removed.
- So: report per field, never pool across fields, and make no transitive claims.

**Seat-rotation duplicate.**
- Practice: the ACPC used duplicate for its 3-player events. Ganzfried's 3-player Kuhn evaluation played all
  3! = 6 seat permutations per deal set ([arXiv 1804.04789](https://arxiv.org/abs/1804.04789)). I could not fetch
  the ACPC rules page to confirm its exact scheme [memory].
- For N = 6, 720 permutations is too many. With **one test bot and N−1 identical field copies**, the field seats are
  interchangeable, so the **N cyclic rotations** of X's seat cover every distinct case.
- **Block** = one deal seed played in N rooms. In room r, X sits in seat r and the field fills the rest. The cards
  per seat and the button stay fixed, and fresh bot instances are used per room (as in `eval/match.py`).
- **Unit = session** = B blocks (for example, 100 blocks × 6 rooms = 600 hands) with the button advancing inside each
  room. Then bootstrap over sessions.
- **Sanity check:** X = field copy gives an exact mean of 0 per block for deterministic bots.

**Paired comparison (the tier test).**
- Run X and Y through the same seeds, rotations, field and field RNG seeds.
- Statistic: D_s = (X's session total) − (Y's session total), in bb/100.
- Var(D) = Var(S_X) + Var(S_Y) − 2Cov, and the shared cards and field make Cov large.
- This answers "is X better than Y *in this seat role against this field*", which is the only clean multiway claim.

**Variance and sample size.**
- With σ_eff = the per-hand-equivalent SD of the unit (σ_session / √hands_per_session) and a 95% CI half-width of
  h bb/100: **n_hands ≈ (1.96 × 100 × σ_eff / h)²**.
- For a paired difference, use σ_D,eff in its place.
- HU reference from this project (`results/phase7_bankroll.md`): σ_hand is about 16–35 bb, and the duplicate SD is
  about 15 bb for CFR vs Bayes.
- Multiway guesses [estimate]:
  - σ_hand for one seat is about 15–40 bb. Fields of AlwaysRaise copies sit at the top end, because 6-way all-in pots
    reach 600bb.
  - Rotation duplicate cancels less than the HU mirror, so expect a variance ratio of about 0.5–0.9.
- Examples:
  - σ_eff = 25 and h = 10 → about 240,000 hands.
  - h = 20 → about 60,000 hands.
  - A paired difference with σ_D,eff = 12 and h = 10 → about 55,000 hands.
- In pure Python, 6 Monte Carlo bots per hand will make **compute the bottleneck**. Run a 10–20 session pilot on a
  separate seed to measure σ, then fix n before the official run.
- **Chance-node control variate** (`eval/variance_reduction.py`): it generalises to N players. For each pot, use
  V_i = equity_i among that pot's eligible players × pot amount, minus contributions, with exact enumeration.
  - Use only the flop→turn and turn→river nodes: at most about 45 × 44 boards, which is cheap.
  - Skip the preflop→flop node (C(40,5) = 658k boards when 6-way all-in is too slow). This keeps it unbiased, because
    the inclusion rule uses only pre-deal information.
  - Full AIVAT would also need the bots' action probabilities
    ([Burch et al., AAAI 2018, arXiv 1612.06915](https://arxiv.org/abs/1612.06915) [paper]; Pluribus used AIVAT for
    6-max). Skip that.

**Pre-registered tier test template:**
- **Table:** 6-max, 100bb reset, 50/100.
- **Fields:** F1 = 5 × CallStation, F2 = 5 × EquityBot-M, F3 = a mixed field [Random, CallStation, AlwaysRaise,
  PotOdds, EquityBot-M] in a fixed cyclic order. F3 breaks the "identical copies" shortcut, so all 5! orders, or a
  pre-chosen sample of them, would be needed. An easier route is to keep F3's order fixed and say so.
- **Primary endpoint:** paired D = BayesBot-M − EquityBot-M in the test seat, against F2, with the n fixed from the
  pilot, a fresh master seed, a two-sided 95% bootstrap CI over sessions, and no peeking or early stopping.
- **Secondary endpoints:** the same comparison in F1 and F3; each bot's absolute bb/100 per field; and the effect of
  the variance-reduction estimator.
- **Pass rule:** CI lower bound > 0 in the primary field. Report any field where the sign flips, rather than hiding it.

---

## 4. Build order

| step | content | effort [estimate] | risk |
|---|---|---|---|
| 1 | `engine/multiway` betting: positions, blinds, turn order, min-raise, cumulative reopening, uncalled return | 2–3 d | reopening edge cases |
| 2 | Pots and settlement: layers, per-pot odd chip, showdown order, `MultiSettlement` | 1–2 d | low |
| 3 | Tests: invariants, property tests, N=2 differential vs HU, hand-written cases, PokerKit and PHH (Pluribus) replay | 2–4 d | convention mapping with PokerKit |
| 4 | Runner, hand log, `MultiPublicState`, `MultiBot` base plus an HU-bot adapter; golden hashes for HU | 1–2 d | info leak (extend `test_info_leak`) |
| 5 | Table UI: N seats, position labels, side pots display, rotation replays | 1–2 d | low (layout exists) |
| 6 | Bots: rule bots (0.5 d), multiway MC equity and PotOdds-M (1–2 d), EquityBot-M (2 d), BayesBot-M with pooling (2–3 d), CFRBot-M fallback (1 d) | 6–9 d | speed of MC × k opponents × 6 seats |
| 7 | Eval: rotation duplicate, paired sessions, N-player chance control variate, pilot | 2–3 d | variance, compute time |
| 8 | Optional: N-player push/fold solver; 6-max preflop MCCFR | 3–8 d | scope creep, unverifiable quality |
| 9 | Official pre-registered run and write-up | compute-bound (days) | field-dependence makes claims narrow |

**Main risks:**
1. Rule bugs that only show up multiway, such as reopening and folded chips in side pots. The mitigation is PokerKit,
   PHH and property tests.
2. Python speed. Profile early, cap the number of samples, and consider precomputed preflop multiway tables.
3. Sample sizes of 10^5 hands or more for 10 bb/100 precision.
4. The temptation to claim "beats the field" in general.

## Sources
- [Poker TDA Rules 2015 v1.0, HTML copy (Rules 15, 16, 19, 20, 31, 33, 43, 44)](https://homepokertourney.org/poker-tda-rules.htm). This is a secondary copy; the official site is [pokertda.com](https://www.pokertda.com/view-poker-tda-rules/), which was unreachable when I checked. The 2024 wording may differ.
- [Kim, PokerKit (arXiv 2308.07327)](https://arxiv.org/abs/2308.07327); [PokerKit docs](https://pokerkit.readthedocs.io/en/stable/simulation.html)
- [Kim, Recording and Describing Poker Hands, PHH (arXiv 2312.11753)](https://arxiv.org/abs/2312.11753); [phh-std](https://github.com/uoftcprg/phh-std)
- [Brown & Sandholm, Superhuman AI for multiplayer poker, Science 2019](https://www.science.org/doi/10.1126/science.aay2400); [Meta AI blog (blog)](https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/); [CMU press release (blog)](https://www.cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html)
- [Abou Risk & Szafron, CFR for multiplayer poker agents, AAMAS 2010](https://www.cs.ualberta.ca/~games/poker/publications/AAMAS10.pdf)
- [Ganzfried & Sandholm, 3-player jam/fold equilibrium, AAMAS 2008](https://aamas.csc.liv.ac.uk/Proceedings/aamas08/proceedings/pdf/paper/AAMAS08_0313.pdf); [IJCAI 2009 follow-up](https://ijcai.org/proceedings/2009/34)
- [Ganzfried et al., Successful Nash equilibrium agent for a 3-player imperfect-information game (arXiv 1804.04789)](https://arxiv.org/abs/1804.04789)
- [Burch, Schmid, Moravčík, Bowling, AIVAT (arXiv 1612.06915)](https://arxiv.org/abs/1612.06915)
