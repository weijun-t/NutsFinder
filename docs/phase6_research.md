# Phase 6 research notes: CFR, Leduc, abstraction, exploitability

Written for NutsFinder (HUNL, 100bb, SB 50 / BB 100, pure Python). Read time is about 20 minutes.
Units: **mbb/g** means milli-big-blinds per game (hand). **1 bb/100 = 10 mbb/g**, so BayesBot's
+61 bb/100 over EquityBot is +610 mbb/g.
Confidence tags: **[paper]** means a peer-reviewed source, **[preprint]** means arXiv only,
**[memory]** means I recall it but did not re-check it, and **[estimate]** means my own back-of-envelope number.

---

## 0. TL;DR recommendations

1. Write one **`Game` interface**. Kuhn, Leduc and the abstract hold'em game all implement it, and every
   solver (CFR, CFR+, DCFR, external-sampling MCCFR) and the exact best response run on any of them.
   The hold'em MCCFR code is then *proven correct on Leduc* by the exact-exploitability tests before it
   ever sees hold'em.
2. **Leduc:** vanilla CFR first, then CFR+ and DCFR. All are full-traversal, exact, and fast enough in Python.
3. **Hold'em:** **external-sampling MCCFR with linear weighting** (as in Pluribus). Use an abstraction of
   169 preflop classes, 50 equity buckets per postflop street (imperfect recall), and the actions
   {fold, check/call, pot, all-in} plus a 3bb preflop open. That is about 10^5 info sets and tens of MB.
4. **Off-tree bets:** use the pseudo-harmonic mapping.
5. **Measuring:** use exact exploitability for Kuhn and Leduc. For hold'em, use local best response (LBR) and an
   in-abstraction best response.
6. **Tier test:** CFRBot vs BayesBot head-to-head (same session-duplicate design as Phase 5) is what
   counts. Win rates against the weak baselines are reported but **not** used to decide the tier.

---

## 1. Leduc hold'em

**Rules** (Southey et al. 2005) [paper]:
- The deck has 6 cards: J, J, Q, Q, K, K (two suits, and suits never matter).
- Each player antes 1 chip and gets one private card.
- **Round 1:** fixed bet/raise size 2. **Round 2:** one public card is dealt, then the bet/raise size is 4.
- Each round allows **at most 2 bets** (a bet plus one raise). Player 1 acts first in both rounds.
- **Showdown:** a private card that pairs the board wins. Otherwise the higher card wins. Equal ranks split the pot.

**Size.** Each round has 6 decision points: `"" , r, rr, c, cr, crr`. Five round-1 lines continue to
round 2: `cc, rc, rrc, crc, crrc`.
- With all 6 cards distinct: 6×6 + 5×6×6×5 = **936 info sets**. This is the commonly quoted figure
  ([Waugh et al. 2015, AAMAS](https://aamas.csc.liv.ac.uk/Proceedings/aamas2015/aamas/p37.pdf)).
- With ranks only (suit isomorphism): 6×3 + 5×6×3×3 = **288**. Both numbers appear in papers. Use
  ranks only, which is lossless here.

**Game value.** At equilibrium Player 1 gets about **−0.0856 chips/hand**:
- [AlphaExploitem 2026](https://arxiv.org/pdf/2605.09150) [preprint] gives −0.0856.
- [Waugh & Bagnell 2014](https://arxiv.org/abs/1411.7974) [preprint] says "about 0.08".

I did not verify the fourth decimal against a solver. Your own CFR+ run is the real check (§6).

**How to verify a CFR implementation on Leduc**
- Compute **exact exploitability** after every k iterations. Exploitability is defined as
  `(BR_0(σ_1) + BR_1(σ_0)) / 2`. That is half of what OpenSpiel calls "NashConv". State which one you use.
  Exact BR is cheap here (§4).
- **Expected curve:** on log-log axes, exploitability against iterations is roughly a straight line.
  - Vanilla CFR has a slope of about −0.5 to −1. Theory guarantees only −0.5, and practice is usually better.
  - CFR+ and DCFR have a slope of about −1 or steeper.
- **Rough magnitudes** [memory/estimate, calibrate yourself]:
  - Vanilla: about 10^-2 chips after 10^3 iterations.
  - CFR+: about 10^-3 or below after 10^3 iterations.
- Other checks:
  - The value of the average profile tends to −0.0856.
  - Every info-set strategy sums to 1.
  - Exploitability of the *uniform random* strategy is large (on the order of 1 chip). This is a sanity
    anchor for the BR code.
- **Test thresholds:** run once, record the curve in `results/`, then set the test threshold at about 2× the
  measured value. Don't guess thresholds in advance.

---

## 2. CFR variants and which to use

**Plain words.** At every info set, keep a running total of "how much better each action would have
been than what I actually did" (regret). Next time, play each action in proportion to its positive
regret. The *time-averaged* strategy converges to Nash in two-player zero-sum games. The current strategy
does not need to converge.

**Counterfactual regret** (Zinkevich et al. 2007) [paper].
- v_i(σ, I) is the value of info set I to player i, weighted by the opponent's and chance's reach
  π_{−i}(I) only. It counts as if i had played to reach I.
- The instantaneous regret of action a is r^t(I,a) = v_i(σ^t_{I→a}, I) − v_i(σ^t, I).
- R^T(I,a) = Σ_t r^t(I,a), and σ^{T+1}(I,a) ∝ max(R^T(I,a), 0). If there is no positive regret, play uniform.
- The average strategy weights by the player's *own* reach: σ̄(I,a) ∝ Σ_t π_i^t(I) σ^t(I,a).
  Your Kuhn code does this as `ss[i] += reach * s[i]`, which is correct.

| Variant | Change | Notes |
|---|---|---|
| Vanilla CFR | as above, full tree each iteration | baseline, O(1/√T) guarantee |
| CFR+ (Tammelin 2014; Bowling et al. 2015) | regrets floored at 0 each iteration (R ← max(R + r, 0)); players update **alternately**; average weighted by t (or with a delay) | usually about an order of magnitude faster in practice; used to solve HU *limit* hold'em |
| Linear CFR (Brown & Sandholm 2019) | weight iteration t's regret and average contributions by t | equals DCFR(1,1,1); works well **with sampling** |
| DCFR(α,β,γ) (Brown & Sandholm 2019) | after iteration t: positive regrets × t^α/(t^α+1), negative regrets × t^β/(t^β+1), average × (t/(t+1))^γ | recommended α=1.5, β=0, γ=2; reported to beat CFR+ on many games |
| MCCFR (Lanctot et al. 2009) | sample part of the tree, use importance-weighted regret estimates | needed for hold'em |

**MCCFR flavours** (Lanctot et al. 2009) [paper]:
- **Chance sampling.** Sample the cards once, then walk the *whole* betting tree for both players.
  The variance is low. Cost per iteration equals the size of the betting tree.
- **External sampling (ES).** Sample chance *and the opponent's* actions, and expand all of the traverser's
  actions. The variance is moderate. This is the standard workhorse: Pluribus used linear-weighted ES with
  pruning ([Brown & Sandholm 2019, Science](https://www.science.org/doi/10.1126/science.aay2400)).
- **Outcome sampling (OS).** Sample one single path. It is cheapest per iteration, but the importance weights
  1/q(z) create **high variance**. Use it only when the tree cannot be expanded at all, or for online play.

**Recommendation.**
- **Leduc:** use full-traversal CFR, CFR+ and DCFR. Leduc is tiny, so you get exact numbers to learn from.
- **Hold'em (pure Python):** use **ES-MCCFR with linear weighting**.
- Two pitfalls:
  - CFR+'s regret floor tends to behave worse under sampling noise. That is why the big sampled systems
    use linear or discounted weighting [memory].
  - The *ES code itself* should first run on Leduc and reach a low exact exploitability.

**Pseudocode: ES-MCCFR, linear weighting.** Alternate the traverser i, and sample all cards up front.

```
for t in 1..T:
    for i in (0, 1):
        deal = sample_cards(rng)                 # hole cards + full board
        traverse(root, deal, i, t)
    if t % D == 0:                               # Linear CFR, Pluribus-style discount
        scale R and S by (t/D) / (t/D + 1)

traverse(h, deal, i, t):
    if terminal(h): return utility_i(h, deal)
    I = infoset_key(h, deal)                     # (street, bucket, betting history)
    sigma = regret_matching(R[I])
    if player(h) == i:
        v = [traverse(h+a, deal, i, t) for a in actions(h)]
        node = dot(sigma, v)
        for a: R[I][a] += v[a] - node            # sampling already weights by opponent and chance reach
        return node
    else:
        S[I] += sigma                            # average strategy, updated on opponent nodes
        a = sample(sigma, rng)
        return traverse(h+a, deal, i, t)
```

The bot plays the normalised S. A common variant also zeros negative-regret actions in the policy that
is finally played [memory].

---

## 3. Abstraction for HUNL

**Card abstraction (put similar hands into a bucket)**
- **Hand strength, E[HS].** E[HS] is the equity against a uniformly random hand, which you can already compute:
  `equity_vs_range(hole, RANDOM_RANGE, board)`. It is cheap, but it treats a made hand and a draw with
  the same mean as the same hand.
- **E[HS²].** This is the mean over run-outs of the river hand strength squared. A higher variance gives a
  higher score, so draws separate from medium made hands. It is a standard baseline
  (Johanson et al. 2013, *Evaluating State-Space Abstractions in Extensive-Form Games*, AAMAS) [paper].
  Exact E[HS²] needs nested enumeration, which is expensive in Python.
- **Potential-aware, earth mover's distance (EMD).** Describe each hand by a *histogram* over next-street
  buckets, then run k-means with earth mover's distance. This was the best of the methods tested at the time
  ([Ganzfried & Sandholm 2014, AAAI](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf)
  [paper; link from memory]). Treat it as a stretch goal.
- **Imperfect recall.** On each street the bot forgets its earlier buckets and knows only the current bucket
  plus the betting history. That allows many more buckets per street for the same memory
  (Waugh et al. 2009, *A Practical Use of Imperfect Recall*, SARA) [paper]. CFR's guarantees formally no
  longer hold (Lanctot et al. 2012, ICML), but it works well in practice.
- **Bucket edges.** Use **equal-frequency percentiles**. Sample about 10^5 random (hand, board) pairs per
  street once, sort the scores and cut them into B equal groups. Store the edges in a data file, as you did
  with `preflop_data.py`.

**Action abstraction.** Allow only a few sizes, as fractions of the pot after calling. Use
`raise_to_for_fraction` from `bots/common.py`, then clamp to `[legal.min_to, legal.max_to]`.
**Build the abstract betting tree once, using `betting.py` itself** (`new_hand` / `apply_action`). That
guarantees every abstract action is legal. Then store it as integer node ids with child arrays, so the
MCCFR inner loop never creates `HandState` objects.

**Action translation (opponent bets off-tree).** Suppose the abstraction has sizes A < B (pot fractions)
and the opponent bets x with A ≤ x ≤ B. Treat the bet as A with probability

    f(x) = (B − x)(1 + A) / ((B − A)(1 + x)),   otherwise treat it as B

This is the **pseudo-harmonic mapping**
([Ganzfried & Sandholm 2013, IJCAI](https://ijcai.org/proceedings/2013/29)) [paper]. It satisfies
f(A)=1 and f(B)=0, is monotone, and was much less exploitable than earlier mappings.
- Example: A = ½ pot, B = pot, x = ¾ pot gives f = (0.25·1.5)/(0.5·1.75) ≈ **0.43** to "half pot".
- Bets below the smallest size map to check/call or the smallest size. Above the largest size, map to all-in.
- **Pitfall:** after translation, the abstract pot differs from the real pot. Keep sizing your own bets
  from the **real** pot.

**Recommended minimal abstraction** [estimate; benchmark before committing]

| Street | Card buckets | Actions | Raise cap |
|---|---|---|---|
| Preflop | 169 classes (lossless, `hand_class`) | fold, call, open to 3bb, pot raise, all-in | 3 non-all-in raises |
| Flop/turn/river | 50 equal-frequency E[HS] buckets each, imperfect recall | fold, check/call, pot, all-in | 2 (bet + raise) |

**Counting info sets.**
- One postflop street has 12 decision nodes and 5 lines that carry on to the next street (4 calls plus check-check).
- About 7 preflop lines reach the flop.
- So there are about 84 / 420 / 2,100 flop / turn / river betting nodes.
- Info sets ≈ (84+420+2,100)×50 + about 15×169 ≈ **1.3×10^5**. That is an upper bound: at 100bb, pot-size
  raises reach all-in quickly, so many lines collapse.
- **Memory:** about 10–15 MB as numpy arrays `R[node_id, bucket, action]` and `S[...]` (float64), or about
  50–100 MB as Python dicts.
- Adding a ½-pot size multiplies the betting nodes by roughly 10–30× (up to about 10^6–10^7 info sets).
  That needs numpy arrays, so keep it for later.

**Compute** [estimate]:
- One ES iteration visits a few hundred to a few thousand nodes. At about 10–20 µs per node in Python,
  that is about 10–30 ms.
- Bucket lookups per iteration cost 2 players × 3 streets of Monte Carlo E[HS]. At about 100 samples,
  2 evaluations each, ~11 µs per evaluation, that is ~13 ms. **Cache** the results by suit-canonical
  (hole, board).
- Flop E[HS] could even be **precomputed** for all 1,286,792 suit-isomorphic (hand, flop) pairs. That is
  about 1–2 h of CPU at 200 samples, or minutes on 24 cores. The count comes from Waugh 2013,
  *A Fast and Optimal Hand Isomorphism Algorithm* [memory].
- Total: about 30 iterations/s/core, about 2–3 M iterations/hour on 24 cores.
- **Parallelism under the GIL:** use `multiprocessing` workers.
  - Each worker plays a batch of iterations against a frozen snapshot of R and returns regret and strategy
    *deltas*. The master sums them and broadcasts.
  - Give each worker its own seed, derived by hashing (see the note in `preflop.py`), so runs are reproducible.
  - Updates from slightly stale snapshots are an approximation. I know of no formal guarantee, but it is
    standard practice.

**Libratus and Pluribus, conceptually** [paper]. Both precompute a **blueprint** with MCCFR on an abstraction,
then improve it *during play*.
- [Libratus](https://www.science.org/doi/10.1126/science.aao1733) uses nested safe subgame solving and
  beat pros by 147 mbb/g.
- [Pluribus](https://www.science.org/doi/10.1126/science.aay2400) uses depth-limited search and plays 6-max.
  Its blueprint took about 12,400 core-hours.
- [DeepStack](https://arxiv.org/abs/1701.01724) instead re-solves every decision with a learned value network.
- Real-time solving is out of scope for pure Python. A blueprint-only bot is the realistic target.

**Heads-up vs 8-player.** In two-player zero-sum games, equilibrium strategies are interchangeable, guarantee
the game value, and CFR provably converges. With 3 or more players none of this holds: CFR has no convergence
guarantee to Nash, and even an exact Nash does not guarantee winning. Pluribus's success was empirical.
For Phase 7, expect to rely on measurement, not theory.

---

## 4. Measuring exploitability

**Exact best response (Kuhn, Leduc).**
- Fix σ_{−i}. At every info set of player i, choose the action that maximises Σ_{h∈I} π_c(h)π_{−i}(h)·u_i(h·a).
  Go bottom-up, memoising the chosen action per info set.
- Exploitability is the average of the two BR values (see §1). It is 0 exactly at Nash.
- For large games, see Johanson et al. 2011, *Accelerating Best Response Calculation in Large Extensive
  Games*, IJCAI [paper].

```
best = {}
u(h): terminal -> payoff_i; chance -> Σ p·u(child); i to act -> u(h + best_action(I(h)));
      opponent -> Σ σ(I,a)·u(h+a)
best_action(I): if I not in best: best[I] = argmax_a Σ_{h in I} w(h)·u(h+a)   # w = chance × opponent reach
```

**Local best response (LBR)** ([Lisý & Bowling 2017](https://arxiv.org/abs/1612.07547)) [paper, AAAI workshop].
- How it works:
  - LBR tracks the bot's range over all 1,326 hands with Bayes' rule, using the bot's own action probabilities.
  - At each of its decisions it assumes both players will check or call to the end.
  - It picks the action with the best one-step EV: fold, call, or bets with the bot's fold probability
    computed from the range.
- Its average winnings are a **lower bound** on exploitability.
- Results:
  - LBR exploited the abstraction bots from the 2013–2016 Annual Computer Poker Competition (ACPC) by
    **thousands of mbb/g**. With fold/call/pot/all-in, active on the turn and river, it took about
    2,600–4,000 mbb/g.
  - All of those bots were exploitable for more than 3,180 mbb/g.
  - For scale, *always folding* loses 750 mbb/g.
  - DeepStack was reported as not exploitable by LBR (Moravčík et al. 2017).
- **Fit to this repo:** LBR is structurally EquityBot with the right range.
  - Reuse `equity_vs_range` and the EV formulas.
  - It needs `strategy(hand, state)` for all 1,326 hands, which CFRBot can provide.
  - Running LBR against BayesBot means 1,326 `act` calls per action. That is slow: restrict it to turn and
    river, and make sure `act` neither changes state nor consumes the shared RNG.

**Practical proxies** (report all of them, each with a CI):
1. **In-abstraction BR.** Run ES-MCCFR updating only the challenger while CFRBot is frozen. The challenger
   converges to a best response inside the abstraction. This is a lower bound, blind to off-tree bets.
2. **LBR**, as above, in the real game. This also catches translation leaks.
3. **Fixed exploiters:** AlwaysRaise, CallStation, EquityBot and BayesBot head-to-head. These are weak
   lower bounds.
4. **Variance reduction:**
   - Duplicate matches are already in place.
   - [AIVAT (Burch et al. 2018)](https://arxiv.org/abs/1612.06915) [paper] uses the known strategy of
     one's own bot to subtract luck. It cut the standard deviation in DeepStack's match several-fold.
     It is a good Phase 7 item, because CFRBot's strategy is known exactly.

**Reference values.**
- Cepheus (limit hold'em) is about 1 mbb/g exploitable ([Bowling et al. 2015](https://www.science.org/doi/10.1126/science.1259433)).
- Exact numbers for no-limit bots are not known, only LBR lower bounds. **Expect a 50-bucket bot to be very
  exploitable by LBR** (likely above 1,000 mbb/g [estimate]). The meaningful claim is *relative*: CFRBot's LBR
  value is significantly lower than BayesBot's and EquityBot's.

---

## 5. The maths

**Nash equilibrium (two-player zero-sum).**
- A profile is a Nash equilibrium if neither player gains by deviating alone.
- Von Neumann's minimax theorem: max_{σ1} min_{σ2} u1 = min_{σ2} max_{σ1} u1 = v, the game value. For Kuhn,
  v = −1/18 for Player 1.
- Every equilibrium strategy guarantees v against *any* opponent. That is why it is "unexploitable", and also
  why it does not punish mistakes as much as possible.

**Regret matching** (Hart & Mas-Colell 2000, *Econometrica*) [paper].
- Play σ^{t+1}(a) ∝ max(R^t(a), 0).
- After T rounds the external regret is at most Δ·√(|A|·T), where Δ is the payoff range. So the average regret
  R^T/T is at most Δ√|A|/√T, which tends to 0.
- CFR applies this at every info set. Zinkevich et al. show that total regret is at most the sum of the
  info-set counterfactual regrets, which gives R_i^T/T ≤ Δ·|I_i|·√|A|/√T.

**Why the *average* converges.** Let v̄ be the average payoff (1/T)Σ_t u1(σ^t), and let both players have average
regret at most ε.
- Player 1's regret condition: max_{σ1} u1(σ1, σ̄2) ≤ v̄ + ε. This uses linearity of u1 in σ2. In sequence
  form, linearity needs the average strategy to be weighted by the player's own reach, hence the weighting in
  the update.
- Player 2's regret condition: min_{σ2} u1(σ̄1, σ2) ≥ v̄ − ε.
- So σ̄ is a 2ε-equilibrium. In other words, exploitability ≤ 2ε, which is O(1/√T).
- Nothing forces the current strategy σ^t to converge, and it often cycles.

**Indifference principle.** In an equilibrium, every action a player mixes between has the same EV.
Otherwise they would drop the worse one. So you pick *your* mix to make the *opponent* indifferent.

**River bluff ratio** (polarised bettor; bet B into pot P).
- The caller risks B to win P + B. Let α be the share of bets that are bluffs. The caller is indifferent when
  α(P+B) − (1−α)B = 0, which gives **α = B/(P+2B)**.
- **Minimum defence frequency (MDF):** a pure bluff risks B to win P. It breaks even if the opponent folds
  B/(P+B) of the time, so the defender must continue with **MDF = P/(P+B)**. This is already used in
  `equity_bot.py`.

**Worked example.**
- P = 100, B = 50: α = 50/200 = **25%** bluffs (1 bluff : 3 value bets), and MDF = 100/150 = **66.7%**.
- Pot-sized bet (B = P): α = 1/3 and MDF = 1/2.
- **Kuhn check:** pot = 2 antes, bet = 1, so α = 1/(2+2) = 1/4. Player 1's equilibrium bets the K with
  probability 3x and bluffs the J with x, which is exactly 1 bluff : 3 value. Your `kuhn_cfr.py` output should
  show P(bet | K) ≈ 3·P(bet | J).

**Exploitation vs equilibrium.**
- BayesBot is the *exploiter*: it maximises EV against a model and can be exploited back.
- CFRBot is the *safe* strategy.
- Methods that sit between the two:
  - Restricted Nash response (Johanson et al. 2007, *Computing Robust Counter-Strategies*, NeurIPS) [paper].
  - Safe opponent exploitation (Ganzfried & Sandholm 2015, *ACM TEAC*) [paper].
- A good Phase 7 bridge: start from the CFR strategy and shift toward exploitation only as BayesBot's credible
  intervals tighten.

---

## 6. Build order for this repo

| Step | Add | Test (in `tests/`) | Done when |
|---|---|---|---|
| 1 | `src/nutsfinder/gametheory/game.py`: `Game` protocol (`initial`, `is_terminal`, `utility`, `player`, `chance_outcomes`, `infoset_key`, `actions`, `next`) | — | Kuhn ported to it |
| 2 | `gametheory/kuhn.py`, `gametheory/best_response.py` (exact BR + exploitability) | BR of the known Kuhn equilibrium (α=⅓ family) ≈ 0; BR value vs "always bet" matches a hand calculation; value −1/18 ± 1e-3 | exploitability < 1e-3 after 10k vanilla iterations (calibrate) |
| 3 | `gametheory/regret.py` (regret matching), `gametheory/cfr.py` (vanilla / CFR+ / linear / DCFR via options) | regret matching unit tests (all-negative → uniform); log-log slope of the exploitability curve on Kuhn | all variants converge on Kuhn |
| 4 | `gametheory/leduc.py` + `examples/train_leduc.py` (plot of exploitability vs iterations to `results/`) | 288 rank info sets; value → −0.0856 ± 0.005; exploitability thresholds at 2× the recorded run; CFR+ beats vanilla at equal iterations | curve committed in `results/phase6_leduc.md` |
| 5 | `gametheory/mccfr.py` (ES, linear weighting, seeded) | **ES on Leduc** reaches low exact exploitability (e.g. < 0.05 chips at a fixed seed; calibrate); same seed gives identical tables | the solver is validated |
| 6 | `abstraction/cards.py` (E[HS] buckets, percentile edges in a data file, suit-canonical cache), `abstraction/actions.py` (abstract sizes, pseudo-harmonic translation), `abstraction/tree.py` (betting tree built with `betting.py`) | bucket is monotone in equity; equal-frequency within ±1% on fresh samples; f(A)=1, f(B)=0, monotone; every tree action legal under `apply_action` | info-set count printed and matching the estimate |
| 7 | `examples/train_holdem.py` (multiprocess, checkpoints as `.npz`), `bots/cfr_bot.py` (maps `PublicState` to the abstract node, samples from σ̄ with its own seeded RNG, implements `reseed`) | info-leak spy test; reproducibility test; illegal-action fuzz over 10k hands | training run logged with seed and commit |
| 8 | `eval/lbr.py` (+ in-abstraction BR script) | LBR vs AlwaysFold ≈ +750 mbb/g; LBR vs CallStation is large and positive | Phase 6 results table |

**The tier test (pre-register it in `results/phase6.md` before running).** CFRBot will probably win *less*
than BayesBot against CallStation and the other weak baselines. Equilibrium play does not punish mistakes as
hard as possible, so that comparison is the wrong yardstick.

- **Primary (gating): CFRBot vs BayesBot head-to-head.** Use the same session-duplicate design as Phase 5
  (independent sessions of 250 pairs, fresh bots per session so BayesBot gets its normal learning window),
  a new master seed fixed in advance, and a two-sided 95% CI that excludes 0.
- **Sample size:** Bayes vs Equity had an SD of about 78 bb/100 per session. To detect +20 bb/100 at 80% power
  you need n ≈ ((1.96+0.84)·78/20)² ≈ **120 sessions**. Fix n in advance, and never extend the run because
  it "looks close".
- **Secondary (gating for "hard to exploit"):** LBR lower bound of CFRBot < LBR of BayesBot and of EquityBot,
  paired on the same deals, with CIs.
- **Reported, not gating:** win rates against Random, CallStation, AlwaysRaise and PotOdds, plus the
  in-abstraction BR value.
- If the primary test fails, report it as is. That is still a valid result about exploitation versus safety,
  and the fix (more buckets, a ½-pot size) is a new pre-registered run.

**Risks to flag:**
- Bucket noise from Monte Carlo E[HS].
- The 100bb stack depth makes pot-size raises jump to all-in, so the preflop tree is coarse.
- Translation artefacts on unusual bet sizes.
- Python speed. **Benchmark the iteration speed in step 7 before trusting any of the compute estimates above.**

---

## Sources

- Kuhn, H. W. (1950). *A Simplified Two-Person Poker*. Contributions to the Theory of Games I.
- Southey et al. (2005). [Bayes' Bluff: Opponent Modelling in Poker](https://arxiv.org/abs/1207.1411). UAI. (Leduc rules)
- Zinkevich, Johanson, Bowling, Piccione (2007). [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html). NeurIPS.
- Hart & Mas-Colell (2000). *A Simple Adaptive Procedure Leading to Correlated Equilibrium*. Econometrica.
- Lanctot, Waugh, Zinkevich, Bowling (2009). [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html). NeurIPS.
- Neller & Lanctot (2013). [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf). Tutorial notes, not peer-reviewed; the best first read.
- Tammelin (2014). [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/abs/1407.5042). arXiv.
- Bowling, Burch, Johanson, Tammelin (2015). [Heads-up Limit Hold'em Poker Is Solved](https://www.science.org/doi/10.1126/science.1259433). Science.
- Brown & Sandholm (2019). [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040). AAAI.
- Waugh et al. (2009). *A Practical Use of Imperfect Recall*. SARA. Lanctot et al. (2012). *No-Regret Learning in Extensive-Form Games with Imperfect Recall*. ICML.
- Johanson, Burch, Valenzano, Bowling (2013). *Evaluating State-Space Abstractions in Extensive-Form Games*. AAMAS.
- Ganzfried & Sandholm (2014). [Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf). AAAI. (link from memory)
- Ganzfried & Sandholm (2013). [Action Translation in Extensive-Form Games with Large Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping](https://ijcai.org/proceedings/2013/29). IJCAI.
- Waugh (2013). *A Fast and Optimal Hand Isomorphism Algorithm*. AAAI workshop.
- Johanson, Waugh, Bowling, Zinkevich (2011). *Accelerating Best Response Calculation in Large Extensive Games*. IJCAI.
- Lisý & Bowling (2017). [Equilibrium Approximation Quality of Current No-Limit Poker Bots](https://arxiv.org/abs/1612.07547). AAAI workshop.
- Burch, Schmid, Moravčík, Morrill, Bowling (2018). [AIVAT: A New Variance Reduction Technique for Agent Evaluation](https://arxiv.org/abs/1612.06915). AAAI.
- Moravčík et al. (2017). [DeepStack](https://arxiv.org/abs/1701.01724). Science.
- Brown & Sandholm (2018). [Superhuman AI for heads-up no-limit poker: Libratus](https://www.science.org/doi/10.1126/science.aao1733). Science.
- Brown & Sandholm (2019). [Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400). Science.
- Johanson, Zinkevich, Bowling (2007). *Computing Robust Counter-Strategies*. NeurIPS. Ganzfried & Sandholm (2015). *Safe Opponent Exploitation*. ACM TEAC.
- Waugh et al. (2015). [Solving Games with Functional Regret Estimation](https://arxiv.org/abs/1411.7974) (arXiv) / [AAMAS 2015 paper](https://aamas.csc.liv.ac.uk/Proceedings/aamas2015/aamas/p37.pdf). Source for the Leduc value of about 0.08 and the 936 count.
- AlphaExploitem (2026). [arXiv 2605.09150](https://arxiv.org/pdf/2605.09150). Preprint; source for −0.0856.
