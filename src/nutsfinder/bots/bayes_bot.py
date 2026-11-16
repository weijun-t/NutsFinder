"""BayesBot (tier 3): EquityBot plus a Bayesian (Beta-Binomial) model of the opponent.

EquityBot (tier 2) reads the opponent the same way against everyone: one
bet means "top 50%", and every bet of ours gets at most 10% folds. Phase 4
showed what that costs: it folds about half the time to RandomBot's and
AlwaysRaiseBot's bets (and about 44% of those folds would have won), and it
keeps making river bluffs against a calling station that never folds.
BayesBot is EquityBot with those fixed numbers replaced by *frequencies it
measures*, with a prior so that small samples do not fool it.

Everything else (equity estimation, the EV formulas, bet sizes, effective
stacks, tie-breaking: see :mod:`nutsfinder.bots.equity_bot` and
:mod:`nutsfinder.bots.common`) is inherited unchanged.

1. The Beta-Binomial model
==========================
Each statistic is the probability ``p`` of a yes/no event, e.g. "facing a bet
from us, the opponent folds". We do not know ``p``; we describe our belief
about it by a Beta distribution::

    Beta(a, b):  density(p) ∝ p^(a-1) (1-p)^(b-1),   mean = a / (a + b)

Why Beta: it is *conjugate* to the binomial. If we observe ``k`` yeses in
``n`` chances, the likelihood is ``C(n,k) p^k (1-p)^(n-k)``, and Bayes' rule
(posterior ∝ prior × likelihood) gives::

    p^(a-1) (1-p)^(b-1) × p^k (1-p)^(n-k) = p^(a+k-1) (1-p)^(b+n-k-1)

which is again a Beta: the posterior is ``Beta(a + k, b + n - k)``, with mean::

    (a + k) / (a + b + n)

So the prior acts like ``a`` imaginary yeses and ``b`` imaginary nos seen
before the match ("pseudo-counts"); ``a + b`` is the prior's *strength*.
The posterior mean is a weighted average of the prior mean and the raw rate::

    (a + k)/(a + b + n) = w * a/(a + b) + (1 - w) * k/n,   w = (a + b)/(a + b + n)

Worked example: fold-to-bet prior Beta(1, 9) (mean 10%, strength 10). The
opponent folds 4 of the first 5 times we bet. Raw rate 80%; posterior
Beta(5, 10), mean 5/15 = 33%. After 40 of 50: Beta(41, 19), mean 68%.

Why small samples mislead, and how the prior protects
-----------------------------------------------------
The raw rate ``k/n`` has standard error ``sqrt(p(1-p)/n)``: after 5
chances, a true 30% folder shows anything from 0/5 (17% of the time) to
4/5. Acting on "80% folds" after 4/5 would make us bluff off our stack on
noise. The prior pulls small samples towards a sensible default (weight
``w = 10/15`` on the prior after 5 chances) and gets out of the way as data
arrives (``w = 10/60`` after 50, ``10/510`` after 500). It also makes the
estimate well defined with no data at all (``n = 0`` gives the prior mean,
where ``k/n`` would be 0/0).

Credible intervals
------------------
:meth:`BetaStat.credible_interval` gives the central 95% interval of the
posterior: the 2.5% and 97.5% quantiles of ``Beta(a + k, b + n - k)``,
computed *exactly* by bisection on the regularized incomplete beta function
``I_x(a, b) = P(p <= x)`` (evaluated by its continued fraction, as in
Numerical Recipes, section 6.4). Unlike a confidence interval, it reads
directly as "given the prior and the data, p is in here with probability
95%". The normal approximation ``mean ± 1.96 sqrt(mean (1 - mean) /
(a + b + n + 1))`` (:meth:`BetaStat.normal_interval`, shown for comparison
only) is fine for large ``n`` but can leave [0, 1] near the edges: Beta(1, 9)
gives [-0.07, 0.27] with it, the exact interval is [0.003, 0.34]. The width
shrinks like ``1 / sqrt(n)``.

2. What is tracked (public information only)
============================================
At the end of every hand (:meth:`BayesBot.end_hand`) the hand is replayed
from its public record (blinds, stacks, action history, board) with the
engine's own rules, so for each opponent decision we know exactly what was
legal. Opponent hole cards are used only if they were shown at a showdown.

==================  ==============================================  ===========================================
statistic           a chance is ...                                 a "yes" is ...
==================  ==============================================  ===========================================
``vpip``            a hand where the opponent faced a preflop       they called or raised preflop at least once
                    bet, i.e. staying in cost chips (the small      (posting a blind is not voluntary)
                    blind facing the big blind, or anyone facing
                    a raise)
``agg_pre``,        an opponent decision (preflop / flop-river)     they bet or raised
``agg_post``        where betting or raising was legal
``fold_pre``,       an opponent decision facing a bet or raise      they folded
``fold_post``       *from us* of at most 1.25 pot (see below);
                    facing the big blind alone does not count
``fold_pre_big``,   the same, facing an overbet (more than          they folded
``fold_post_big``   1.25 times the pot after calling: in practice
                    our all-ins)
``bluff``           at a showdown, a street on which the            their shown hand had below 50% equity
                    opponent bet or raised                          against a random hand on that street's board
==================  ==============================================  ===========================================

On ``vpip``: tracking software counts every hand as a chance, so a big
blind who checks its option after a limp scores a "no". We leave those
hands out (checking was free, and the player still sees the flop), because
the number is used as "how wide is the range that pays to play": with the
usual definition a calling station, which never folds, would score about
0.6 and look tight.

Bet size of ours = ``(to - current bet) / (P + c)``, the same "fraction of the
pot after calling" as :func:`nutsfinder.bots.common.raise_to_for_fraction`
(half pot = 0.5, pot = 1.0). Rounding can push a pot-sized bet slightly
above 1, hence the 1.25 cut-off. Two size buckets, because folding
depends strongly on size (PotOddsBot and EquityBot fold far more to an
all-in than to half the pot); more buckets would split the data thinner.

``bluff`` uses the "below 50% equity vs a random hand" proxy of the Phase 4
breakdowns (preflop: the precomputed table; flop/turn/river: a
``bluff_samples``-draw Monte Carlo with the bot's RNG). Known bias: we only
see the hands that reached a showdown, and bluffs that made us fold never
do, so honest bluffers look a little more honest than they are. That is
fine for its job, which is to spot opponents whose bets say *nothing* about
their cards (RandomBot, AlwaysRaiseBot: about 50%).

Priors (``DEFAULT_PRIORS``)
---------------------------
Strength ``a + b = 10`` for every statistic: worth about 10 observations,
so a few hands barely move it and a few dozen dominate it. The means are
chosen so that, with no data, every formula below gives *exactly*
EquityBot's numbers (so BayesBot starts as EquityBot, and any difference in
results comes from what it learns):

- ``vpip`` Beta(7, 3), mean 0.70: a typical heads-up player plays most hands.
- ``agg_pre``, ``agg_post`` Beta(5, 5), mean 0.50: matches EquityBot's
  "one bet = top 50%".
- ``fold_*`` Beta(1, 9), mean 0.10: EquityBot's tuned ``max_fold``. That is
  pessimistic compared with a balanced player (who folds 1/3 to a half-pot
  bet), on purpose: Phase 4's tuning showed that assuming too many folds
  is expensive, so the bot should need evidence before bluffing more.
- ``bluff`` Beta(1.5, 8.5), mean 0.15: about the bluff proxy measured for
  EquityBot itself in Phase 4 (14-21%), i.e. what our crude preflop-ranked
  ranges already tolerate.

3. How the model changes decisions
==================================
Notation: ``â`` posterior mean aggression of the bet's street group,
``a0 = 0.5`` its prior mean; ``v̂``, ``v0 = 0.7`` for VPIP; ``p̂``, ``p0 = 0.1``
fold-to-bet for the bucket of the size considered; ``b̂``, ``b0 = 0.15`` bluff.

(a) How much the opponent's actions tell us: ``m``
--------------------------------------------------
A random hand has below 50% equity against a random hand half the time, so
an opponent whose bets carry no information shows ``bluff ≈ 0.5``; one as
honest as the prior shows ``b0``. We interpolate linearly::

    m = clamp((b̂ - b0) / (0.5 - b0), 0, 1)

``m`` is read as "the share of their actions that are random". Every
range-narrowing step in (b) is mixed with "no narrowing" in that
proportion: a step that would keep a fraction ``k`` of their range keeps
``k + m (1 - k)`` instead (``k`` of the informative part plus all of the
random part). (c) and (d) use ``m`` too. ``m = 0`` at the prior; ``m -> 1``
against AlwaysRaiseBot within a couple of hundred hands (0.88 after 100),
which fixes EquityBot's "folds to every bet" leak against it. Against
RandomBot it is slow and noisy (between 0.16 and 0.68 after 100 hands in
our sessions): RandomBot folds so often that few of its bets reach a
showdown.

(b) The opponent's range ``w`` after their bets and raises
---------------------------------------------------------
EquityBot assumes widths ``W = (1, 0.5, 0.3, 0.15)`` after 0, 1, 2, 3+
bets/raises. Write each as a power of the prior bet frequency ``a0 = 0.5``:
``W_k = a0^(t_k)`` with ``t = (0, 1, 1.74, 2.74)``. That is EquityBot's model:
"each bet keeps the top ``a0`` of the range, and repeated bets act like
``t_k`` such filters". BayesBot keeps the ``t_k`` and substitutes the
measured frequency: an opponent who bets ``â`` of the time keeps the top
``â`` per filter::

    k_i = (W_i / W_(i-1)) ^ (ln â / ln a0)        for the i-th bet/raise this hand
    w   = base × Π_i [k_i + m (1 - k_i)]

So the first bet gives exactly ``top â`` (``W_1 / W_0 = a0`` and
``a0 ^ (ln â / ln a0) = â``): bets 80% of the time -> top 80%, 10% -> top 10%.
``â = a0`` reproduces EquityBot exactly; ``â -> 1`` (a maniac) gives
``w -> 1`` however often they bet. Each bet uses the ``â`` of its own street
group (preflop / postflop). ``w`` is then snapped down to the range grid as
in EquityBot (after a ``1 + 1e-9`` nudge so that float error in the
product, e.g. 0.14999999999999997, cannot drop it a grid step).

``base``: EquityBot treats an opponent who only called as holding any hand.
BayesBot keeps that for one who plays as many hands as the prior's typical
player, and narrows in proportion for a tighter one: if they put money in
voluntarily preflop, ``base = r + m (1 - r)`` with ``r = min(1, v̂ / v0)``
(VPIP 35% = half as loose as typical -> top 50%), else ``base = 1``.

Worked example: opponent with ``â_post = 0.8`` (bets 80%), ``m = 0``, no
preflop narrowing, bets the flop and raises the turn::

    k_1 = 0.5 ^ (ln 0.8 / ln 0.5) = 0.8,  k_2 = 0.6 ^ 0.322 = 0.848
    w = 0.8 × 0.848 = 0.678 -> grid 0.60      (EquityBot: 0.30)

(c) Fold probability ``f`` for a bet of ours
-------------------------------------------
EquityBot: ``f = min(0.1, s)`` with ``s = x / (P + x) = 1 - MDF`` (the fold
rate that makes a pure bluff break even). BayesBot first decides how much
of the observed rate ``p̂`` to believe::

    p* = p̂                        if p̂ <= p0   (folds less than assumed: always believed)
    p* = p0 + m (p̂ - p0)          if p̂ >  p0   (folds more: believed only as far as m)

and then, with ``ρ = p* / p0`` ("how much more they fold than the prior
assumed")::

    f = min(p*, ρ s) = ρ × min(p0, s)

So normal-sized bets get ``f = p*``, and tiny bets (where ``s < p0``, e.g. a
50-chip all-in into 2,000) get proportionally fewer folds, exactly as
EquityBot scales them. At the prior ``p* = p0``, ``ρ = 1`` and this is
EquityBot's formula. Below the prior the data is used in full: against a
calling station ``f -> 0`` and it stops bluffing. Above the prior there is no
cap at ``s``: against a random opponent who folds 40% to a half-pot bet
(``s = 1/3``), a pure bluff is worth ``0.4 P - 0.6 × P/2 = +0.1 P``, and it bets.

Why the asymmetry (``m`` gating). This was *not* in the first version, which
used ``p*`` = ``p̂`` everywhere. The ablation on master seed 555 (section 5)
showed that this lost chips against PotOddsBot and EquityBot, which fold
often (19% and 39% to normal bets) but *honestly*: what they keep are the
hands that are strong on this board. Our calling range is "the top x% by
preflop strength", which cannot represent that, so the more folds we expect,
the more we bet medium hands into callers that are much stronger than we
model. (At 10% folds EquityBot hid that error.) Against an opponent whose
actions are random (``m`` near 1) the callers really are a random part of
the range, so there the observed folds can be used. A stronger version
would build calling ranges from board strength; that changes EquityBot's
equity machinery, so it is left for later.

(d) Who calls: the calling range
--------------------------------
EquityBot assumes the callers are the strongest ``MDF = P / (P + x)`` share
of the range (pessimistic). Against someone who folds less than the prior
(``ρ < 1``) more of the range calls, ending at the whole range for a
calling station (``ρ = 0``), but only as far as they *call* rather than
raise (``â`` = their aggression on this street group, which includes
raising our bets)::

    u = 1 - (1 - max(0, 1 - ρ)) (1 - m)       (widen: they fold less, or act randomly)
    q = MDF + (1 - MDF) u (1 - â)
    calling range = top w q, snapped to the grid

At ``ρ >= 1`` it stays at MDF (if they fold more, the callers are at least
as strong as EquityBot assumed). This is what lets BayesBot value-bet
thinner against CallStationBot: its equity "when called" becomes its equity
against the full range. Why ``(1 - â)``: the one-street EV formula says a
bet that is not folded to is *called* and the hand checks down. AlwaysRaiseBot
never folds either, but it re-raises, so "everyone calls" is the wrong
picture there. The same goes for the second reason to widen, ``m`` ("its
calls are as random as its bets"): AlwaysRaiseBot's are, but it does not
call. The ablation on seed 555 measured the widening at +254 bb/100
against CallStationBot but about -240 against AlwaysRaiseBot before the
``(1 - â)`` factor was applied to both reasons (section 5). At the prior
``u = 0``, so this is EquityBot's MDF exactly.

4. Memory, reseeding and sessions
=================================
The model (:attr:`BayesBot.model`) is per-instance memory that lasts for
the whole match: each hand's observations are added in :meth:`end_hand`,
and decisions during a hand use the model as it stood at the start of that
hand (plus the hand's own action history). :mod:`nutsfinder.eval.match`
calls :meth:`reseed` before *every* hand; that restarts **only the RNG** and
never touches the model, otherwise the bot could never learn anything.
:meth:`reset_model` forgets everything (back to the priors), and
:meth:`fresh_copy` builds a new, independent bot with the same settings,
for evaluations that play independent sessions. With ``learn=False`` the
model stays at the prior forever, and the bot plays exactly like
``EquityBot`` (tested), which makes it a clean control.

The model must never see more than EquityBot could: it reads the public
history and, after a showdown only, the shown cards (``tests/test_bayes_bot.py``
runs the spy check of ``tests/test_info_leak.py`` on it).

5. Tuning history
=================
Everything was tuned on master seed 555 only, checked once on seed 556 as
a holdout, and nothing was chosen on seeds 20261117 or later (reserved for
the official run). Measure: 20 independent sessions, each a
``duplicate_match`` of 200 pairs with fresh bots (so each instance learns
over 200 hands), session seeds drawn from ``random.Random(555)``; the
number is BayesBot's bb/100 minus EquityBot's in the *same* sessions
(paired), ± 1.96 SE over sessions.

Fixed before any match (not tuned): the prior means (the values that
reproduce EquityBot), prior strength 10, the 1.25 overbet cut-off, the
"bluff" proxy and the range formula in (b).

0. A 200-hand smoke run on seed 555 showed CallStationBot with ``vpip``
   0.62, because the big blind's free check counted as a "no". Changed the
   ``vpip`` chance to "staying in cost chips" (section 2). A definition
   fix, not a parameter search.
1. First version (p* = p̂, calling range ``q = MDF + (1 - MDF) max(0, 1 - ρ)``
   then mixed with ``m``), with ablations:

   ===========  =============  ===============  ===========
   vs           full model     fold model only  ranges only
   ===========  =============  ===============  ===========
   Random       +148 ± 93      +217 ± 102       +5 ± 71
   PotOdds      +73 ± 48       -110 ± 32        +135 ± 45
   Equity       -23 ± 44       -129 ± 44        +81 ± 40
   AlwaysRaise  +88 ± 80       +63 ± 72         +304 ± 83
   ===========  =============  ===============  ===========

   The learned fold rate hurt against the honest folders: see (c).
2. Fold rates above the prior believed only as far as ``m`` (c). Also
   tried, and rejected: a weaker ``bluff`` prior, Beta(0.75, 4.25) (no
   gain anywhere, -3 ± 83 vs Random). Result: Random -26 ± 66,
   CallStation +254 ± 81, AlwaysRaise +88 ± 80, PotOdds +128 ± 44, Equity
   +74 ± 45. A variant without any calling-range widening scored
   CallStation -12 ± 18 but AlwaysRaise +326 ± 83: the widening is right
   against callers and wrong against raisers.
3. ``(1 - â)`` applied to the ``ρ`` widening only: AlwaysRaise still +92 ± 80.
4. ``(1 - â)`` applied to both reasons to widen (the final version, (d)):
   Random +10 ± 70, CallStation +248 ± 90, AlwaysRaise +334 ± 89, PotOdds
   +135 ± 45, Equity +60 ± 40. Tuning stopped here.

Holdout, seed 556, final version, run once: Random -65 ± 68, CallStation
+217 ± 88, AlwaysRaise +319 ± 89, PotOdds +153 ± 44, Equity +52 ± 44.
These are short sessions and unofficial numbers; the tier claim needs the
official evaluation.

Known weaknesses
================
- The model is only as good as EquityBot's range machinery: ranges are
  still "top x% by preflop strength", even on the river.
- It assumes the opponent is stationary. A player who changes style mid-
  match is tracked slowly (all hands weigh the same; no forgetting).
- Exploitation cuts both ways: a thinking opponent could fold a lot early,
  let ``fold_*`` rise, then call down the bluffs that follow.
- ``vpip`` and ``agg_*`` assume the opponent's frequent actions come from
  its strong hands; ``bluff`` corrects that only after enough showdowns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from nutsfinder.bots.equity_bot import EquityBot, calling_share
from nutsfinder.engine.actions import Bet, Call, Fold, Raise
from nutsfinder.engine.betting import (
    ActionRecord,
    HandState,
    PublicState,
    Street,
    apply_action,
    deal_board,
    legal_actions,
    new_hand,
)
from nutsfinder.engine.bot import HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.equity import RANDOM_RANGE, equity_vs_range
from nutsfinder.engine.preflop import PREFLOP_EQUITY, hand_class, snap_to_grid

__all__ = [
    "BayesBot",
    "BetaStat",
    "OpponentModel",
    "DEFAULT_PRIORS",
    "OVERBET",
    "beta_cdf",
    "beta_quantile",
    "bet_fraction",
    "fold_stat_name",
    "opponent_events",
    "public_replay",
]

DEFAULT_PRIORS: dict[str, tuple[float, float]] = {
    "vpip": (7.0, 3.0),
    "agg_pre": (5.0, 5.0),
    "agg_post": (5.0, 5.0),
    "fold_pre": (1.0, 9.0),
    "fold_pre_big": (1.0, 9.0),
    "fold_post": (1.0, 9.0),
    "fold_post_big": (1.0, 9.0),
    "bluff": (1.5, 8.5),
}
"""``{statistic: (a, b)}`` of each Beta prior (see the module docstring)."""

OVERBET = 1.25
"""Bets of more than this fraction of the pot (after calling) use the ``*_big`` fold stats."""

RANDOM_BLUFF_RATE = 0.5
"""``bluff`` of an opponent whose bets carry no information (a random hand is below median half the time)."""


# --- Beta maths ------------------------------------------------------------------


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """The continued fraction for ``I_x(a, b)`` (modified Lentz, Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 1e-15
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),  # even step
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),  # odd step
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """``P(p <= x)`` for ``p ~ Beta(a, b)``: the regularized incomplete beta ``I_x(a, b)``.

    ``I_x(a, b) = x^a (1-x)^b / (a B(a, b)) × CF(x)``, with ``B`` the beta
    function and ``CF`` a continued fraction that converges fast for
    ``x < (a + 1) / (a + b + 2)``; above that we use the symmetry
    ``I_x(a, b) = 1 - I_(1-x)(b, a)``. Checks: ``Beta(1, 1)`` is uniform, so
    ``I_x = x``; ``Beta(1, b)`` has ``I_x = 1 - (1 - x)^b``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    """The ``x`` with ``P(p <= x) = q`` for ``p ~ Beta(a, b)``, by bisection.

    The CDF increases from 0 to 1 on [0, 1], so halving the bracket 60
    times pins ``x`` down to 2^-60 (far below anything that matters).
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class BetaStat:
    """One yes/no frequency: a ``Beta(prior_a, prior_b)`` prior plus counts.

    Attributes:
        prior_a, prior_b: the prior's pseudo-counts of yes / no.
        yes: observed yeses ``k``.
        n: observed chances ``n``.
    """

    prior_a: float
    prior_b: float
    yes: int = 0
    n: int = 0

    def record(self, yes: bool) -> None:
        """Add one observation."""
        self.yes += bool(yes)
        self.n += 1

    @property
    def alpha(self) -> float:
        """Posterior ``a + k``."""
        return self.prior_a + self.yes

    @property
    def beta(self) -> float:
        """Posterior ``b + n - k``."""
        return self.prior_b + self.n - self.yes

    @property
    def prior_mean(self) -> float:
        return self.prior_a / (self.prior_a + self.prior_b)

    @property
    def mean(self) -> float:
        """Posterior mean ``(a + k) / (a + b + n)``."""
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Exact central credible interval of the posterior (Beta quantiles)."""
        tail = (1.0 - level) / 2
        return beta_quantile(tail, self.alpha, self.beta), beta_quantile(1.0 - tail, self.alpha, self.beta)

    def normal_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Normal approximation ``mean ± z sd`` (for comparison; can leave [0, 1]).

        The Beta(A, B) variance is ``mean (1 - mean) / (A + B + 1)``.
        """
        a, b = self.alpha, self.beta
        sd = math.sqrt(self.mean * (1 - self.mean) / (a + b + 1))
        return self.mean - z * sd, self.mean + z * sd

    def reset(self) -> None:
        self.yes = self.n = 0

    def __str__(self) -> str:
        lo, hi = self.credible_interval()
        return f"{self.mean:.3f} [{lo:.3f}, {hi:.3f}] ({self.yes}/{self.n})"


@dataclass
class OpponentModel:
    """The opponent's statistics, each a :class:`BetaStat` (see the module docstring)."""

    priors: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRIORS))
    stats: dict[str, BetaStat] = field(init=False)
    hands: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if set(self.priors) != set(DEFAULT_PRIORS):
            raise ValueError(f"priors must have exactly the keys {sorted(DEFAULT_PRIORS)}")
        self.stats = {name: BetaStat(a, b) for name, (a, b) in self.priors.items()}

    def __getitem__(self, name: str) -> BetaStat:
        return self.stats[name]

    def observe(self, events: list[tuple[str, bool]]) -> None:
        """Add one hand's ``(statistic, yes)`` events."""
        for name, yes in events:
            self.stats[name].record(yes)
        self.hands += 1

    def reset(self) -> None:
        """Forget every observation (back to the priors)."""
        for stat in self.stats.values():
            stat.reset()
        self.hands = 0

    def randomness(self) -> float:
        """``m = clamp((b̂ - b0) / (0.5 - b0), 0, 1)``: how little their actions say."""
        s = self.stats["bluff"]
        m = (s.mean - s.prior_mean) / (RANDOM_BLUFF_RATE - s.prior_mean)
        return min(1.0, max(0.0, m))

    def lines(self) -> list[str]:
        """One line per statistic, for debug logs."""
        return [f"model ({self.hands} hands) m={self.randomness():.2f}"] + [
            f"  {name}: {stat}" for name, stat in self.stats.items()
        ]


# --- Reading a hand's public record ------------------------------------------------


def _group(street: Street) -> str:
    return "pre" if street == Street.PREFLOP else "post"


def bet_fraction(pot: int, call: int, current_bet: int, to: int) -> float:
    """Size of a bet/raise to ``to`` as a fraction of the pot after calling: ``(to - current) / (P + c)``."""
    return (to - current_bet) / (pot + call)


def fold_stat_name(street: Street, fraction: float) -> str:
    """``fold_pre`` / ``fold_post``, with ``_big`` for overbets (fraction above :data:`OVERBET`)."""
    return f"fold_{_group(street)}" + ("_big" if fraction > OVERBET else "")


def public_replay(view: PublicState) -> list[tuple[HandState, ActionRecord]]:
    """Replay a hand from public information: ``(state before, action)`` for every action.

    Uses only what every player saw (button, blinds, starting stacks, the
    history and the board); the states hold no hole cards. Replaying with
    the engine's own rules gives the exact legal actions at every decision.
    """
    state = new_hand(view.button, view.starting_stacks, view.small_blind, view.big_blind)
    out = []
    for rec in view.history:
        while state.cards_needed:
            n = len(state.board)
            state = deal_board(state, view.board[n : n + state.cards_needed])
        out.append((state, rec))
        state = apply_action(state, rec.player, rec.action)
    return out


def opponent_events(view: PublicState) -> list[tuple[str, bool]]:
    """The betting events of one finished hand, for the opponent of ``view.seat``.

    Returns ``(statistic, yes)`` pairs for ``vpip``, ``agg_*`` and ``fold_*``
    (see the table in the module docstring); ``bluff`` needs the shown cards
    and is handled by :meth:`BayesBot.end_hand`.
    """
    hero, opp = view.seat, 1 - view.seat
    events: list[tuple[str, bool]] = []
    faced_preflop_bet = voluntary = False
    last_aggressor: dict[Street, int] = {}  # street -> seat of the last bet/raise
    our_size = 0.0  # fraction of the pot of our last bet/raise
    for state, rec in public_replay(view):
        legal = legal_actions(state)
        if rec.player == hero:
            if isinstance(rec.action, (Bet, Raise)):
                to = rec.action.amount if isinstance(rec.action, Bet) else rec.action.to
                our_size = bet_fraction(state.pot, legal.call_amount, state.current_bet, to)
        else:
            g = _group(rec.street)
            if rec.street == Street.PREFLOP:
                faced_preflop_bet |= legal.can_call
                voluntary |= isinstance(rec.action, (Call, Raise, Bet))
            if legal.can_bet or legal.can_raise:
                events.append((f"agg_{g}", isinstance(rec.action, (Bet, Raise))))
            if legal.can_call and last_aggressor.get(rec.street) == hero:
                events.append((fold_stat_name(rec.street, our_size), isinstance(rec.action, Fold)))
        if isinstance(rec.action, (Bet, Raise)):
            last_aggressor[rec.street] = rec.player
    if faced_preflop_bet:
        events.insert(0, ("vpip", voluntary))
    return events


# --- The bot -----------------------------------------------------------------------


class BayesBot(EquityBot):
    """EquityBot whose ranges and fold model come from a Beta-Binomial opponent model.

    Args:
        seed: seeds the RNG (flop/turn Monte Carlo, showdown bluff checks).
        priors: ``{statistic: (a, b)}`` Beta priors; default :data:`DEFAULT_PRIORS`.
        learn: update the model after each hand. ``False`` keeps it at the
            prior forever, which plays exactly like ``EquityBot``.
        bluff_samples: Monte Carlo samples for each postflop showdown check
            of the ``bluff`` statistic.
        range_widths, bet_fractions, consider_all_in, samples, time_budget,
        debug: as in :class:`EquityBot` (``range_widths`` is the schedule
            ``W`` the aggression model rescales). ``max_fold`` does not
            exist here: the ``fold_*`` priors play its role.
    """

    def __init__(
        self,
        seed: int | None = None,
        priors: dict[str, tuple[float, float]] | None = None,
        learn: bool = True,
        bluff_samples: int = 200,
        range_widths: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15),
        bet_fractions: tuple[float, ...] = (0.5, 1.0),
        consider_all_in: bool = True,
        samples: int = 400,
        time_budget: float | None = None,
        debug: bool = False,
    ) -> None:
        self._init_kwargs: dict[str, Any] = dict(
            seed=seed, priors=None if priors is None else dict(priors), learn=learn,
            bluff_samples=bluff_samples, range_widths=range_widths, bet_fractions=bet_fractions,
            consider_all_in=consider_all_in, samples=samples, time_budget=time_budget, debug=debug,
        )  # fmt: skip
        model = OpponentModel(dict(DEFAULT_PRIORS) if priors is None else dict(priors))
        super().__init__(
            seed=seed,
            range_widths=range_widths,
            bet_fractions=bet_fractions,
            consider_all_in=consider_all_in,
            max_fold=model["fold_post"].prior_mean,  # unused: see fold_and_calling
            samples=samples,
            time_budget=time_budget,
            debug=debug,
        )
        self.model = model
        self.learn = learn
        self.bluff_samples = bluff_samples

    # --- Memory ----------------------------------------------------------------

    def reseed(self, seed: int) -> None:
        """Restart the RNG from ``seed``. Does **not** touch the opponent model.

        The match code reseeds before every hand; the model must survive
        that, or the bot would forget its opponent every hand.
        """
        super().reseed(seed)

    def reset_model(self) -> None:
        """Forget everything learned about the opponent (back to the priors)."""
        self.model.reset()

    def fresh_copy(self) -> BayesBot:
        """A new, independent BayesBot with the same settings and an empty model."""
        return type(self)(**self._init_kwargs)

    def end_hand(self, summary: HandSummary) -> None:
        """Add this hand's observations to the model (public record + shown cards only)."""
        if not self.learn:
            return
        view = summary.final_view
        opp = 1 - summary.seat
        events = opponent_events(view)
        shown = summary.shown_cards[opp]
        if shown is not None:
            events += [("bluff", e < 0.5) for e in self._bet_equities(view, opp, shown)]
        self.model.observe(events)

    def _bet_equities(self, view: PublicState, opp: int, cards: tuple[Card, ...]) -> list[float]:
        """The shown hand's equity vs a random hand, on each street the opponent bet or raised."""
        streets = sorted(
            {r.street for r in view.history if r.player == opp and isinstance(r.action, (Bet, Raise))}
        )
        out = []
        for street in streets:
            if street == Street.PREFLOP:
                out.append(PREFLOP_EQUITY[hand_class(cards)])
            else:
                board = view.board[: street.board_size]
                r = equity_vs_range(cards, RANDOM_RANGE, board, self.bluff_samples, rng=self._rng)
                out.append(r.equity)
        return out

    # --- The model's effect on decisions ------------------------------------------

    def _keep(self, k: float) -> float:
        """A narrowing step that keeps ``k`` of the range, mixed with "random": ``k + m (1 - k)``."""
        return k + self.model.randomness() * (1.0 - k)

    def villain_range_width(self, state: PublicState) -> float:
        """``w = base × Π_i [k_i + m (1 - k_i)]``, snapped to the grid (module docstring, 3b)."""
        opp = 1 - state.seat
        w = 1.0
        mine = [r for r in state.history if r.player == opp]
        if any(r.street == Street.PREFLOP and isinstance(r.action, (Call, Raise, Bet)) for r in mine):
            vpip = self.model["vpip"]
            w *= self._keep(min(1.0, vpip.mean / vpip.prior_mean))
        W = self.range_widths
        k = 0
        for rec in mine:
            if not isinstance(rec.action, (Bet, Raise)):
                continue
            k += 1
            step = W[min(k, len(W) - 1)] / W[min(k - 1, len(W) - 1)]
            agg = self.model[f"agg_{_group(rec.street)}"]
            w *= self._keep(step ** (math.log(agg.mean) / math.log(agg.prior_mean)))
        return snap_to_grid(w * (1 + 1e-9))

    def trusted_fold_rate(self, stat: BetaStat) -> float:
        """``p̂`` if at most the prior mean ``p0``, else ``p0 + m (p̂ - p0)`` (module docstring, 3c)."""
        if stat.mean <= stat.prior_mean:
            return stat.mean
        return stat.prior_mean + self.model.randomness() * (stat.mean - stat.prior_mean)

    def fold_and_calling(self, state: PublicState, width: float, to: int, added: int) -> tuple[float, float]:
        """``(f, calling range width)`` for a bet/raise to ``to`` (module docstring, 3c and 3d)."""
        legal = state.legal
        assert legal is not None
        pot = state.pot
        stat = self.model[fold_stat_name(state.street, bet_fraction(pot, legal.call_amount, state.current_bet, to))]
        mdf = calling_share(pot, added)
        p = self.trusted_fold_rate(stat)
        rho = p / stat.prior_mean
        f = min(p, (1 - mdf) * rho)
        # Two reasons to widen the callers: they fold less than assumed, or their
        # actions are random. u combines them: 1 - u = (1 - max(0, 1 - rho)) (1 - m).
        u = 1.0 - (1.0 - max(0.0, 1.0 - rho)) * (1.0 - self.model.randomness())
        raises = self.model[f"agg_{_group(state.street)}"].mean
        q = mdf + (1 - mdf) * u * (1.0 - raises)
        return f, snap_to_grid(width * q)

    def model_notes(self, state: PublicState) -> list[str]:
        """Appended to each debug :class:`Decision`: where the range and fold numbers came from."""
        opp = 1 - state.seat
        bets = sum(1 for r in state.history if r.player == opp and isinstance(r.action, (Bet, Raise)))
        lines = [
            f"range: top {self.villain_range_width(state):.1%} after {bets} opponent bet(s)/raise(s), "
            f"m = {self.model.randomness():.2f}"
        ]
        legal = state.legal
        for to in self.candidate_sizes(state):
            frac = bet_fraction(state.pot, legal.call_amount, state.current_bet, to)
            name = fold_stat_name(state.street, frac)
            stat = self.model[name]
            p = self.trusted_fold_rate(stat)
            lines.append(
                f"to {to} ({frac:.2f} pot): {name} mean {stat.mean:.3f}, trusted {p:.3f}, "
                f"rho = {p / stat.prior_mean:.2f}"
            )
        return lines + self.model.lines()

    def __repr__(self) -> str:
        return f"BayesBot(learn={self.learn}, hands_seen={self.model.hands}, samples={self.samples})"
