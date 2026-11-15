"""The statistics behind every "bot X beats bot Y" claim.

The setting
-----------
A match produces one number per *unit*: a unit is one hand in a plain match,
or one pair of hands in a duplicate match (see :mod:`nutsfinder.eval.match`).
We treat the units as independent draws from some unknown distribution with
mean mu (the bot's true edge per unit) and standard deviation sigma. Stacks
reset every hand and every unit gets fresh seeds, so independence is a fair
assumption *between* units. (It is *not* fair between the two hands of a
duplicate pair, which is exactly why the pair, not the hand, is the unit.)

From n units x_1..x_n we estimate:

- the mean            m  = sum(x) / n
- the sample SD       s  = sqrt( sum((x - m)^2) / (n - 1) )
  (n - 1, "Bessel's correction": m was fitted to the same data, so the
  deviations from m are a little smaller than deviations from the true mu
  would be; dividing by n - 1 instead of n makes s^2 unbiased for sigma^2.)
- the standard error  SE = s / sqrt(n)
  (the SD of the *mean*, not of a single unit. Var(sum) = n sigma^2 for
  independent units, so Var(mean) = sigma^2 / n. This is why halving the
  error costs four times as many hands.)

Confidence interval (normal approximation)
------------------------------------------
By the central limit theorem the mean of many independent units is
approximately normal, m ~ N(mu, SE^2), whatever the shape of one unit's
distribution. So m +/- 1.96 SE covers mu in about 95% of runs.

Assumptions, stated plainly:

1. Units are independent and identically distributed.
2. n is large enough for the CLT. Poker results are heavy-tailed (most hands
   move a blind or two, a few move 100bb stacks), so "large enough" means
   hundreds to thousands of units, not 30. With 10,000 units it is fine; with
   50 units the real coverage is noticeably below 95% (the coverage test in
   ``tests/test_eval_coverage.py`` measures this).
3. We plug s in for the unknown sigma. Strictly that calls for Student's t
   distribution instead of the normal, but for n above a few hundred the two
   critical values agree to two decimals (t_{0.975, 1000} = 1.962 vs
   z = 1.960), so we use the normal throughout and call it a z-test.

The 95% CI is a statement about the *procedure*: run it many times and about
95% of the intervals contain the true value. It is not "a 95% probability
that the true edge is in this particular interval".

p-value
-------
Null hypothesis H0: mu = 0 (no edge). z = m / SE. The two-sided p-value is
P(|Z| >= |z|) for a standard normal Z: the chance that pure luck would give
a result at least this far from 0 *if there were no edge*. p < 0.05 is the
same event as "the 95% CI excludes 0".

Sample size
-----------
See :func:`required_units`.

Only the standard library is used: :class:`statistics.NormalDist` gives the
normal CDF and its inverse, so no SciPy is needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "Summary",
    "mean",
    "sample_sd",
    "standard_error",
    "z_critical",
    "normal_ci",
    "z_test_p_value",
    "summarize",
    "required_units",
    "rule_of_thumb_units",
]

_STD_NORMAL = NormalDist(0.0, 1.0)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Raises ``ValueError`` on an empty sequence."""
    if not values:
        raise ValueError("mean of an empty sequence")
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation with Bessel's correction (divide by n - 1).

    Needs at least 2 values: with one value there is no spread to measure.
    """
    n = len(values)
    if n < 2:
        raise ValueError("sample SD needs at least 2 values")
    m = mean(values)
    return math.sqrt(math.fsum((x - m) ** 2 for x in values) / (n - 1))


def standard_error(values: Sequence[float]) -> float:
    """SE of the mean: s / sqrt(n). How far the *mean* typically is from mu."""
    return sample_sd(values) / math.sqrt(len(values))


def z_critical(confidence: float = 0.95) -> float:
    """The two-sided critical value: 1.96 for 95%, 2.576 for 99%."""
    return _STD_NORMAL.inv_cdf(0.5 + confidence / 2)


def normal_ci(m: float, se: float, confidence: float = 0.95) -> tuple[float, float]:
    """``m +/- z * se``: the normal-approximation confidence interval."""
    half = z_critical(confidence) * se
    return (m - half, m + half)


def z_test_p_value(m: float, se: float) -> float:
    """Two-sided p-value for H0: mu = 0, using z = m / se.

    If se is 0 (every unit gave exactly the same number) the test is
    undefined and we return ``nan``. Zero spread is *not* proof of an edge:
    it can be exact by construction (deterministic bots whose duplicate
    pairs cancel) or a fluke of a tiny sample (``[3, 3, 3]``), and the
    numbers alone cannot tell which. ``nan < 0.05`` is False, so such a
    result is never reported as significant.

    The tail is computed as ``2 * cdf(-z)`` rather than ``2 * (1 - cdf(z))``:
    for large z, ``cdf(z)`` rounds to exactly 1.0 (around z = 8.3) and the
    subtraction gives 0, while ``cdf(-z)`` stays accurate down to ~1e-300.
    """
    if se == 0:
        return math.nan
    z = abs(m) / se
    return 2 * _STD_NORMAL.cdf(-z)


@dataclass(frozen=True, slots=True)
class Summary:
    """Statistics of a list of per-unit results, plus the same in bb/100.

    Per-unit fields are in the units of the input (big blinds per unit). The
    ``*_bb100`` fields rescale to big blinds per 100 *hands*, the usual poker
    unit: a unit of ``hands_per_unit`` hands is divided by that, then x100.
    Scaling by a constant c scales the mean, SD, SE and CI by c and leaves
    the p-value unchanged.
    """

    n: int
    hands_per_unit: int
    mean: float
    sd: float
    se: float
    ci: tuple[float, float]
    p_value: float

    @property
    def _scale(self) -> float:
        return 100 / self.hands_per_unit

    @property
    def mean_bb100(self) -> float:
        return self.mean * self._scale

    @property
    def se_bb100(self) -> float:
        return self.se * self._scale

    @property
    def ci_bb100(self) -> tuple[float, float]:
        return (self.ci[0] * self._scale, self.ci[1] * self._scale)

    @property
    def no_variation(self) -> bool:
        """True when every unit gave the same result (SD = 0).

        The CI then has zero width and there is no p-value: this is not a
        statistical estimate. Report it as "exact" only if you know why
        (e.g. the setup forces it), never as evidence of an edge.
        """
        return self.sd == 0

    @property
    def significant(self) -> bool:
        """True when p < 0.05, i.e. the 95% CI excludes 0. Never true when SD = 0."""
        return self.p_value < 0.05


def summarize(values: Sequence[float], hands_per_unit: int = 1) -> Summary:
    """Mean, SD, SE, 95% CI and p-value of per-unit results."""
    m = mean(values)
    sd = sample_sd(values)
    se = sd / math.sqrt(len(values))
    return Summary(
        n=len(values),
        hands_per_unit=hands_per_unit,
        mean=m,
        sd=sd,
        se=se,
        ci=normal_ci(m, se),
        p_value=z_test_p_value(m, se),
    )


def required_units(sigma: float, edge: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units needed to detect a true edge ``edge`` with the given power.

    ``sigma`` and ``edge`` must be in the same units (e.g. bb per unit).

    Derivation. We declare "significant" when |m| > z_{a/2} * sigma/sqrt(n).
    If the true mean is e > 0, m ~ N(e, sigma^2/n). We want that to happen
    with probability ``power`` (call it 1 - beta), i.e. the threshold must
    sit z_b standard errors *below* e:

        z_{a/2} * sigma/sqrt(n) = e - z_b * sigma/sqrt(n)
        sqrt(n) = (z_{a/2} + z_b) * sigma / e
        n       = ((z_{a/2} + z_b) * sigma / e)^2

    With alpha = 0.05 and power = 0.8: z_{a/2} = 1.96, z_b = 0.84, so
    n = (2.8 sigma / e)^2 = 7.85 (sigma/e)^2. (The tiny chance of being
    significant in the *wrong* direction is ignored.)

    Example: RandomBot-like noise of sigma = 40 bb/hand and an edge of
    5 bb/100 = 0.05 bb/hand gives n = (2.8 * 40 / 0.05)^2 = 5 million hands.
    """
    if edge == 0:
        raise ValueError("an edge of 0 can never be detected")
    z_a = _STD_NORMAL.inv_cdf(1 - alpha / 2)
    z_b = _STD_NORMAL.inv_cdf(power)
    return math.ceil(((z_a + z_b) * sigma / abs(edge)) ** 2)


def rule_of_thumb_units(sigma: float, edge: float) -> int:
    """The quick version: n = (2 sigma / e)^2.

    This is the n at which the 95% CI half-width (about 2 SE = 2 sigma/sqrt(n))
    shrinks to the size of the edge. At that n a true edge e is only detected
    about *half* the time (power 50%), because the estimate is equally likely
    to land just above or just below the threshold. It is a lower bound for
    "how long must I play", not a target; :func:`required_units` with
    power 0.8 asks for about twice as many units.
    """
    if edge == 0:
        raise ValueError("an edge of 0 can never be detected")
    return math.ceil((2 * sigma / abs(edge)) ** 2)
