"""Closed-form bankroll maths: risk of ruin, Kelly stakes, fractional Kelly, parameter uncertainty.

Notation (derivations in ``docs/phase7_bankroll_notes.md``)
----------------------------------------------------------
X = winnings of one hand at stake 1 (big blinds of that stake). E[X] = mu.
Which "sigma" to pass depends on the question (hands may be dependent):

- fixed-stake ruin, and anything resized only between sessions: the long-run
  per-hand variance sigma_room^2 = Var(sum of n hands) / n (for long n);
- the Kelly fraction and growth with per-hand resizing: the per-hand second
  moment E[X^2] (= sigma_hand^2 + mu^2), because the drift of log W per hand is
  E[log(1 + f X)] ~ f mu - f^2 E[X^2] / 2 whatever the dependence;
- the fluctuation of log W (hence drawdown probabilities): sigma_room again,
  see :func:`prob_ever_fall_to`'s ``scale``.

Fixed stake (bankroll B in bb, never resized), Brownian approximation of the
cumulative winnings, drift mu and variance sigma^2 per hand:

    P(ruin, ever)            = exp(-2 mu B / sigma^2)            (mu > 0)
    bankroll for ruin r      = sigma^2 ln(1 / r) / (2 mu)

Choosing the stake s (money per bb) for bankroll W (money), with resizing:

    log-growth per hand      g(s) ~= mu s / W - sigma^2 s^2 / (2 W^2)
    Kelly stake              s* / W = mu / sigma^2   i.e. a bankroll of sigma^2 / mu big blinds
    full-Kelly growth        g* = mu^2 / (2 sigma^2) per hand
    fraction c of Kelly      g(c) = (2c - c^2) g*,   zero at c = 2
    P(wealth ever falls to x of its start) = x^(2/c - 1)   (continuous resizing, diffusion)

The last line gives P(ever halving) = 2^-(2/c - 1): 1/2 at full Kelly, 1/8 at
half Kelly. With continuous resizing ruin (W = 0) is impossible in the
diffusion; the fixed-stake ruin formula is the contrast.

Exact (non-diffusion) versions on data: :func:`exact_kelly_fraction`
maximises the sample mean of log(1 + f x), and :func:`lundberg_exponent`
solves mean(exp(-R x)) = 1, which gives the Cramer-Lundberg bound
P(ruin) <= exp(-R B) for an i.i.d. random walk observed at the x's (no
diffusion, keeps skew and fat tails; R ~= 2 mu / sigma^2 for small mu).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "risk_of_ruin", "bankroll_for_ruin", "kelly_fraction", "kelly_bankroll", "growth_rate",
    "kelly_growth", "fractional_growth_ratio", "prob_ever_fall_to", "prob_ever_halve",
    "exact_kelly_fraction", "lundberg_exponent", "noisy_plugin_growth_ratio",
    "optimal_shrinkage", "growth_ratio_misspecified",
]


def _check(mu: float, sigma: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")


def risk_of_ruin(mu: float, sigma: float, bankroll: float | np.ndarray) -> float | np.ndarray:
    """exp(-2 mu B / sigma^2) for mu > 0; 1 if mu <= 0 (a driftless or losing walk hits any level)."""
    _check(mu, sigma)
    if mu <= 0:
        return np.ones_like(np.asarray(bankroll, dtype=float)) if np.ndim(bankroll) else 1.0
    out = np.exp(-2.0 * mu * np.asarray(bankroll, dtype=float) / sigma**2)
    return out if np.ndim(bankroll) else float(out)


def bankroll_for_ruin(mu: float, sigma: float, ruin: float) -> float:
    """Bankroll (bb) with fixed-stake risk of ruin ``ruin``: sigma^2 ln(1/r) / (2 mu). inf if mu <= 0."""
    _check(mu, sigma)
    if not 0 < ruin < 1:
        raise ValueError("ruin must be in (0, 1)")
    if mu <= 0:
        return math.inf
    return sigma**2 * math.log(1 / ruin) / (2 * mu)


def kelly_fraction(mu: float, sigma: float) -> float:
    """Kelly stake as a fraction of the bankroll, in "bb per unit of bankroll": mu / sigma^2."""
    _check(mu, sigma)
    return mu / sigma**2


def kelly_bankroll(mu: float, sigma: float) -> float:
    """Bankroll in big blinds of the stake played at full Kelly: sigma^2 / mu (inf if mu <= 0)."""
    _check(mu, sigma)
    return math.inf if mu <= 0 else sigma**2 / mu


def growth_rate(f: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """Quadratic log-growth per hand when staking a fraction ``f`` (bb per bankroll unit): f mu - f^2 sigma^2 / 2."""
    return f * mu - np.square(f) * sigma**2 / 2


def kelly_growth(mu: float, sigma: float) -> float:
    """Full-Kelly log-growth per hand: mu^2 / (2 sigma^2) (0 if mu <= 0: then do not play)."""
    _check(mu, sigma)
    return mu**2 / (2 * sigma**2) if mu > 0 else 0.0


def fractional_growth_ratio(c: float | np.ndarray) -> float | np.ndarray:
    """Growth at c x Kelly divided by full-Kelly growth: 2c - c^2."""
    return 2 * np.asarray(c) - np.square(c) if np.ndim(c) else 2 * c - c * c


def prob_ever_fall_to(c: float, x: float, scale: float = 1.0) -> float:
    """P(wealth ever falls to the fraction ``x`` of today's) at c x Kelly, continuous resizing: x^((2/c - 1) scale).

    For i.i.d. hands ``scale = 1``. With dependent hands the DRIFT of log W per
    hand is f mu - f^2 E[X^2] / 2 (a per-hand expectation, unaffected by
    dependence) while its long-run VARIANCE is f^2 sigma_room^2; with
    f = c mu / E[X^2] the exponent 2 m / v^2 becomes (2/c - 1) E[X^2] / sigma_room^2,
    so pass ``scale = E[X^2] / sigma_room^2``.

    Valid for 0 < c < 2 (positive growth). For c >= 2 the log-wealth has no
    upward drift and every level is eventually hit: returns 1.
    """
    if not 0 < x < 1:
        raise ValueError("x must be in (0, 1)")
    if c <= 0:
        raise ValueError("c must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if c >= 2:
        return 1.0
    return x ** ((2 / c - 1) * scale)


def prob_ever_halve(c: float, scale: float = 1.0) -> float:
    """P(ever losing half the bankroll) at c x Kelly: 2^-((2/c - 1) scale) (see :func:`prob_ever_fall_to`)."""
    return prob_ever_fall_to(c, 0.5, scale)


def exact_kelly_fraction(x: Sequence[float] | np.ndarray, tol: float = 1e-12) -> float:
    """argmax_f mean(log(1 + f x)) over the sample ``x`` (no quadratic approximation).

    Returns 0 if mean(x) <= 0 (do not play) and ``inf`` if no x is negative
    (the sample never loses, so it gives no finite optimum). The objective is concave in f on [0, -1/min(x)); its derivative
    mean(x / (1 + f x)) is decreasing, so bisection finds the root.
    """
    x = np.asarray(x, dtype=float)
    if x.mean() <= 0:
        return 0.0
    lo_x = x.min()
    if lo_x >= 0:
        return math.inf
    hi = -1.0 / lo_x * (1 - 1e-12)
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.mean(x / (1 + mid * x)) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return (lo + hi) / 2


def lundberg_exponent(x: Sequence[float] | np.ndarray, tol: float = 1e-12) -> float:
    """R > 0 with mean(exp(-R x)) = 1 (adjustment coefficient). Ruin bound: P(ruin from B) <= exp(-R B).

    psi(R) = log mean(exp(-R x)) is convex with psi(0) = 0 and psi'(0) = -mean < 0,
    so if some x < 0 there is exactly one positive root. For small mu it is
    ~2 mu / sigma^2, the diffusion's exponent; negative skew lowers it.
    """
    x = np.asarray(x, dtype=float)
    if x.mean() <= 0:
        return 0.0
    if x.min() >= 0:
        return math.inf

    def psi(r: float) -> float:
        z = -r * x
        m = z.max()
        return m + math.log(np.mean(np.exp(z - m)))

    hi = 2 * x.mean() / x.var() if x.var() > 0 else 1.0
    while psi(hi) < 0:
        hi *= 2
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if psi(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return (lo + hi) / 2


# --- Parameter uncertainty -------------------------------------------------------------


def growth_ratio_misspecified(c: float | np.ndarray, lam: float | np.ndarray) -> float | np.ndarray:
    """Growth when betting c x the Kelly of an ESTIMATE while the truth is mu_true = lam * mu_hat.

    In units of the true maximum: (2 c lam - c^2) / lam^2. Betting full
    Kelly on a mu that is twice the truth (lam = 1/2) gives 0; half Kelly
    keeps (lam - 1/4) / lam^2 of the true maximum (1 at lam = 1/2, 0.75 at lam = 1).
    """
    c = np.asarray(c, dtype=float)
    lam = np.asarray(lam, dtype=float)
    out = (2 * c * lam - c**2) / lam**2
    return out if out.ndim else float(out)


def noisy_plugin_growth_ratio(k: float, t: float) -> float:
    """Expected growth / max growth when staking k x mu_hat / sigma^2 with mu_hat ~ N(mu, se^2), t = mu / se.

    E[g] = (k mu^2 - k^2 (mu^2 + se^2) / 2) / sigma^2, so the ratio is
    2k - k^2 (1 + 1/t^2). Full plug-in Kelly (k = 1) loses 1/t^2 of the growth.
    """
    return 2 * k - k**2 * (1 + 1 / t**2)


def optimal_shrinkage(t: float) -> float:
    """k maximising :func:`noisy_plugin_growth_ratio`: t^2 / (1 + t^2) (t = mu / se of the estimate)."""
    return t**2 / (1 + t**2)
