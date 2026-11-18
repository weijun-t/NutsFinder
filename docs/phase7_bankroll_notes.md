# Phase 7 notes: risk of ruin, Kelly sizing and parameter uncertainty

These notes derive every formula used in `src/nutsfinder/bankroll/` and `results/phase7_bankroll.md`.
They assume you know expectations, variances and the normal distribution, and nothing else.

Notation used throughout:

- **X**: the result of one hand, in big blinds (bb) of the stake being played.
- **mu = E[X]**: the win rate per hand. 68 bb/100 means mu = 0.68 bb per hand.
- **sigma^2 = Var(X)**: the variance per hand. Section 6 explains which variance to use when hands are not independent.
- **B**: the bankroll, in bb of the stake played.

## 1. The diffusion approximation

The cumulative result after n hands is S_n = X_1 + ... + X_n. If the X_i are i.i.d., then E[S_n] = n mu and
Var(S_n) = n sigma^2, and by the central limit theorem S_n is approximately normal. The *diffusion approximation*
goes one step further. It replaces the whole path (S_1, S_2, ...) by a Brownian motion with drift,

    S_t ~ mu t + sigma W_t,      W = standard Brownian motion,

which has the same mean and variance at every t, and continuous paths. Two things are lost in this step:

- **Jumps.** A real path can lose 100 bb in a single hand. A diffusion cannot jump, so it hits a level exactly,
  never overshoots it.
- **Shape.** Skew and fat tails are gone: every increment is normal.

Both errors shrink when the bankroll is large compared with a single hand's swing. Section 7 measures them on real data.

## 2. Risk of ruin at a fixed stake

**Claim.** With drift mu > 0, the probability that mu t + sigma W_t ever reaches -B is

    P(ruin) = exp(-2 mu B / sigma^2).

**Proof (exponential martingale).** Look for a theta with E[exp(-theta X)] = 1. Then M_n = exp(-theta S_n) is a
martingale: each new factor exp(-theta X) has mean 1. For a normal X, E[exp(-theta X)] = exp(-theta mu + theta^2 sigma^2 / 2).
This equals 1 when theta = 2 mu / sigma^2.

Stop at the first time tau at which the path hits -B or +A. Optional stopping gives E[M_tau] = M_0 = 1. A
continuous path hits the levels exactly, so with p = P(hit -B first):

    p e^{theta B} + (1 - p) e^{-theta A} = 1.

Let A go to infinity. The second term vanishes, so p = e^{-theta B} = exp(-2 mu B / sigma^2). For mu <= 0 the path
reaches every level eventually, so the ruin probability is 1.

**Bankroll for a target ruin r.** Solve exp(-2 mu B / sigma^2) = r for B:

    B(r) = sigma^2 ln(1/r) / (2 mu).

The bankroll scales with **sigma^2 / mu**. Halving the edge doubles the bankroll you need. Doubling the SD
quadruples it.

**Without the diffusion (Cramer-Lundberg).** For any i.i.d. X with mu > 0 and some losing outcomes, let R > 0 solve
E[exp(-R X)] = 1. This is the *adjustment coefficient*, and it exists because log E[exp(-R X)] is convex, is 0 at
R = 0 and has slope -mu there. The same martingale argument applies, with one change: at ruin the path is at -B
*or below* (it overshoots), so M_tau >= e^{R B}. That gives the inequality

    P(ruin) <= exp(-R B).

Expanding E[exp(-R X)] = 1 - R mu + R^2 E[X^2] / 2 - R^3 E[X^3] / 6 + ... = 1 shows that R ~ 2 mu / sigma^2 for a
small mu. Skew moves it. Big losses (negative skew) make E[exp(-R X)] larger, so R shrinks and the ruin risk
rises. Big wins make R larger. `kelly.lundberg_exponent` solves the equation on the empirical distribution of X,
so no normal assumption is involved.

## 3. Kelly: choosing the stake

Now you may change the stake. You have wealth W (in money) and play a stake s (money per bb). One hand changes
the wealth to W + s X = W (1 + f X), where f = s / W. What matters in the long run is the growth of log-wealth,
because log W_n = log W_0 + sum log(1 + f X_i), and that sum grows at rate E[log(1 + f X)] per hand (law of large
numbers). For a small f X, the Taylor expansion log(1 + y) = y - y^2/2 + ... gives

    g(f) = E[log(1 + f X)] ~ f mu - f^2 E[X^2] / 2 ~ f mu - f^2 sigma^2 / 2,

where the last step uses E[X^2] = sigma^2 + mu^2 and mu^2 << sigma^2 (for example 0.46 against 286 for CFR vs Bayes).
In terms of the stake, g(s) = mu s / W - sigma^2 s^2 / (2 W^2).

Here sigma is the **per-hand** SD. The drift of log W is an expectation over single hands, so it stays per-hand even
when hands are dependent. The code uses E[X^2] itself, not sigma^2. The long-run variance enters only through the
*fluctuation* of log W (section 4) and through fixed-stake ruin (section 2).

- **Kelly fraction.** Setting g'(f) = mu - f sigma^2 = 0 gives f* = mu / sigma^2. In poker terms: hold a bankroll
  of **W / s = sigma^2 / mu big blinds** of the stake you play.
- **Full-Kelly growth.** g* = g(f*) = mu^2 / (2 sigma^2) per hand. The doubling time is ln 2 / g*.
- **Fractional Kelly.** Play f = c f*. Then g(c f*) = c mu^2/sigma^2 - c^2 mu^2/(2 sigma^2) = **(2c - c^2) g***.
  Half Kelly keeps 75% of the growth. At c = 2 the growth is 0, and above that it is negative: you go broke in
  log terms with a *positive* edge.
- **Exact version.** `kelly.exact_kelly_fraction` maximises the sample mean of log(1 + f x) on real outcomes. It
  agrees with mu / sigma^2 when single outcomes are small compared with the bankroll. It does *not* agree when
  the bet is a whole session: if you only resize between 250-hand sessions, a session can lose 500 bb, and
  log(1 + f S) needs 1 + f S > 0, which forces a smaller f.

## 4. Fractional Kelly: probability of ever losing half

**Claim.** With continuous resizing at c times Kelly, P(wealth ever falls to x times its current value) = x^(2/c - 1).
So P(ever halving) = 2^-(2/c - 1).

**Proof.** Write the wealth as a diffusion. Staking f W per bb gives dW = f W (mu dt + sigma dB). By Ito's lemma
the log-wealth L = log W follows

    dL = (f mu - f^2 sigma^2 / 2) dt + f sigma dB = m dt + v dB,

a Brownian motion with drift m = g(f) and volatility v = f sigma. Falling to x W_0 means L dropping by
a = ln(1/x). Section 2 says this happens with probability exp(-2 m a / v^2). Put in f = c mu / sigma^2:

    2 m / v^2 = 2 (mu^2/sigma^2)(c - c^2/2) / (c^2 mu^2 / sigma^2) = (2c - c^2) / c^2 = 2/c - 1,

so P = exp(-(2/c - 1) ln(1/x)) = x^(2/c - 1). The formula in the brief, 2^-(2/c - 1), is therefore correct.

| c (fraction of Kelly) | growth / max | P(ever halve) | P(ever lose 90%) |
|---|---|---|---|
| 0.25 | 0.44 | 2^-7 = 0.8% | 10^-7 |
| 0.5 | 0.75 | 2^-3 = 12.5% | 0.1% |
| 1 | 1 | 1/2 | 10% |
| 1.5 | 0.75 | 2^-1/3 = 79% | 46% |
| 2 | 0 | 1 | 1 |

**Dependent hands: two different variances.** The derivation above treats hands as i.i.d. When they are not
(section 6), the two ingredients of 2m/v^2 come from different places:

- **The drift is a per-hand expectation.** m = E[log(1 + f X)] ~ f mu - f^2 E[X^2]/2. Dependence between hands does
  not change an expectation of a single hand, so the Kelly fraction with per-hand resizing is
  **f* = mu / E[X^2]** (E[X^2] = sigma_hand^2 + mu^2). The Kelly bankroll is E[X^2]/mu, and the growth is
  mu^2 / (2 E[X^2]).
- **The fluctuation is a long-run variance.** log W is a sum of log(1 + f X_i), and its variance over many hands
  is f^2 sigma_room^2, so v^2 = f^2 sigma_room^2 (section 6).

With f = c mu / E[X^2]:

    2 m / v^2 = (2/c - 1) * E[X^2] / sigma_room^2,   so   P(fall to x) = x^((2/c - 1) E[X^2] / sigma_room^2).

Positive dependence (sigma_room > sigma_hand) makes drawdowns *more* likely at the same Kelly fraction. For
BayesBot vs RandomBot, E[X^2]/sigma_room^2 = 0.63, so full Kelly halves the bankroll with probability
2^-0.63 = 65% instead of 50%. In the code: `kelly.prob_ever_fall_to(c, x, scale)` and
`estimate.point_estimates(...)["drawdown_exponent_scale"]`. sigma_room^2 alone governs fixed-stake ruin
(section 2) and any stake that is only changed between sessions (then the "bet" is a whole session, with mean
n mu and variance n sigma_room^2, which gives the session-level Kelly bankroll sigma_room^2 / mu).

Full Kelly halves your bankroll at some point with probability 1/2, and that is a perfectly normal outcome of
the optimal strategy. Half Kelly cuts this to 1/8 and costs only 25% of the growth, which is the classic argument
for fractional Kelly. Above 1 you take more risk *and* get less growth, so it is never worth it.

**Ruin versus drawdown.** With continuous resizing, ruin (W = 0) is impossible in the diffusion: L is finite
forever. Every loss shrinks the next stake. Compare a fixed stake with the Kelly bankroll B = sigma^2 / mu
(i.i.d. hands):
exp(-2 mu B / sigma^2) = **e^-2 = 13.5% ruin**. The "Kelly bankroll" played at a fixed stake is a bankroll with a 13.5%
chance of going bust. Kelly survives only *because* it moves down. Real poker sits in between:

- stakes come in discrete levels (1/2, 2/5, 5/10: roughly factors of 2);
- you resize between sessions, not between hands, so one bad session can take a large bite;
- below the smallest stake you cannot move down, and there ruin is again a fixed-stake problem.

`simulate.simulate_stake_levels` simulates exactly this situation.

## 5. Parameter uncertainty: the key quant point

mu is never known, only estimated: mu_hat comes with a standard error se. Kelly punishes overestimates much more
than underestimates.

**Misspecified edge.** Suppose you bet c times the Kelly fraction for mu_hat, but the truth is mu = lambda mu_hat.
In units of the true maximum growth,

    growth ratio = (2 c lambda - c^2) / lambda^2.

| true mu / estimate (lambda) | full Kelly (c=1) | half Kelly (c=1/2) |
|---|---|---|
| 0.5 (you overestimated 2x) | **0** | 1.00 |
| 0.75 | 0.89 | 0.89 |
| 1 | 1 | 0.75 |
| 1.5 | 0.89 | 0.56 |
| 2 | 0.75 | 0.44 |

Betting full Kelly on twice the true edge earns nothing. Half Kelly never does worse than 44% across this range,
while full Kelly falls to 0. The losses are asymmetric: g is a parabola, and overbetting moves you towards the
zero at 2x.

**Plugging in a noisy estimate.** Let mu_hat ~ N(mu, se^2), the sampling distribution of the estimate, and bet
f = k mu_hat / sigma^2. Then

    E[g] = k mu E[mu_hat]/sigma^2 - k^2 E[mu_hat^2]/(2 sigma^2) = (k mu^2 - k^2 (mu^2 + se^2)/2) / sigma^2,
    E[g] / g* = 2k - k^2 (1 + 1/t^2),        t = mu / se.

- Plug-in Kelly (k = 1) keeps 1 - 1/t^2 of the growth. At t = 1, i.e. an edge only one SE from zero, it has
  zero expected growth.
- The best k is **t^2 / (1 + t^2)**: a shrinkage factor. At t = 2 it is 0.8, and at t = 13 (CFRBot vs BayesBot) it is 0.994.

**The Bayesian view, and why it alone does not shrink.** With a flat prior the posterior is mu ~ N(mu_hat, se^2).
The quadratic growth is *linear* in mu, so E_post[g(f)] = f mu_hat - f^2 sigma^2 / 2. That is maximised at the
plug-in f = mu_hat / sigma^2. So "maximise posterior expected growth" with a flat prior and the quadratic g gives
no shrinkage at all. Shrinkage enters through three routes:

- **An informative prior.** Most measured edges are smaller than they look: a winner's curse, and the selection
  of which bot to deploy. A prior mu ~ N(0, tau^2) gives the posterior mean mu_hat tau^2 / (tau^2 + se^2), the
  same functional form as t^2/(1+t^2).
- **The frequentist risk of the plug-in rule.** This is the calculation above: the noise in mu_hat adds se^2 to
  the "variance of the bet".
- **Risk aversion or robustness.** Examples are Kelly at the lower confidence bound, c = mu_lo / mu_hat, or the
  fraction that maximises a low quantile of growth across the bootstrap. Both are computed in the results.

sigma^2 is estimated too. 1/sigma_hat^2 overestimates 1/sigma^2 on average (Jensen's inequality), another small
push towards overbetting.

**Model uncertainty dominates.** All of the above assumes the edge stays the same. A win rate measured against
one opponent, at one session length, without rake, is not the edge against the next opponent. That uncertainty
has no SE attached, and it is the real reason practitioners use 1/4 to 1/2 Kelly.

## 6. Which sigma? Hands are not i.i.d. in our data

The official matches are duplicate sessions. Each 500-hand session plays 250 deals twice with the seats swapped,
in two "rooms" (separate bot instances), with fresh bots every session. Three variances can be computed from this:

| name | definition | what it is |
|---|---|---|
| var_hand | variance of single hands, pooled | the i.i.d. plug-in |
| var_room | Var(250-hand room total) / 250 | what one player in one seat experiences, including within-session dependence |
| var_dup | Var(500-hand duplicate total) / 500 | the luck left after the mirror hand cancels the cards |

For dependent hands the diffusion of the *cumulative result* needs the **long-run variance**,
lim Var(S_n)/n = sigma^2 + 2 sum_k Cov(X_i, X_{i+k}). The functional CLT then gives the same Brownian limit, with
this variance in place of sigma^2. var_room estimates it at the session scale. var_room / var_hand measures the
dependence: it is 1 for independent hands.

Which variance answers which question:

| question | variance |
|---|---|
| fixed-stake risk of ruin, bankroll for ruin r | var_room (the path of S_t) |
| Kelly fraction and growth with per-hand resizing | E[X^2] = var_hand + mu^2 (the drift of log W is a per-hand expectation) |
| drawdown probabilities under per-hand resizing | both: exponent (2/c - 1) E[X^2] / var_room (section 4) |
| stakes changed only between sessions | var_room per hand, i.e. the session total's variance |
| the SE of a measured edge (duplicate design) | var_dup |

**var_dup is the right variance for measuring an edge, and the wrong one for a bankroll.** Duplicate removes card
luck by playing the mirror hand. A real player never gets the mirror hand.

- In our data var_dup / var_hand is 0.24 to 0.88 for ten pairings. For BayesBot vs RandomBot it is 1.34: there
  the duplicate sessions vary *more* than single hands, because whole sessions run hot or cold together.
- Plugging var_dup into the ruin formula scales the bankroll by var_dup / var_room. That is 0.22 to 0.93 here, an
  understatement in every pairing.

## 7. The Monte Carlo check

`simulate.py` resamples whole room sessions (250 consecutive real hands) with replacement. This is a *block
bootstrap*: it keeps the fat tails and the within-session dependence that the diffusion throws away. Two tricks:

- **The running minimum gives the whole curve.** Ruin from B at a fixed stake means min_t S_t <= -B. Simulate
  M = min_t S_t once per path, and RoR(B) = P(M <= -B) for every B at once. Each block stores its own lowest
  prefix sum, so ruin is checked after every hand while each path is touched only once per block.
- **Wilson intervals.** Simulated probabilities get the Wilson score interval, which stays inside [0, 1] and works
  near 0, where p +/- 1.96 SE does not.

These intervals measure *simulation* error only. They treat the 160 sessions of data as the truth. The
uncertainty of the data itself (mu and sigma) comes from the session bootstrap of the estimates, and it is
reported separately.

**Finite horizon.** A simulation can only see "ruin within T hands". With drift, late ruin is very unlikely. Each
ruin curve is also computed at T/2 on the same paths: if the two agree, the horizon is long enough to stand in
for "ever".

## 8. Rake

A rake of r bb/100 lowers mu to mu - r/100 and leaves sigma unchanged. The required bankroll scales with
1/(mu - r), so a small edge is hit hardest, and at mu <= r every bankroll is eventually ruined. Real rake is a
percentage of the pot with a cap, so it costs more in big pots. Modelling it as a flat per-hand charge is the
first-order version.
