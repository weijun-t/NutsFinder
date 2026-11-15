"""Does the 95% CI really cover the truth about 95% of the time?

RandomBot vs RandomBot is a known-zero matchup: both seats play the same
strategy, so the true edge is exactly 0. We play many short, independent
duplicate matches (different master seeds) and count how often the CI
contains 0.

Each run either covers 0 or not, so the count is Binomial(R, p). If the CI
is right, p is about 0.95 and the observed fraction has SD
sqrt(0.95 * 0.05 / R): 0.69 percentage points for R = 1,000. We accept
0.95 +/- 3 SD. A CI that is badly too narrow (e.g. forgetting the sqrt(n))
would cover far less often and fail this test.

What this test can NOT catch (measured in code review):
- Treating the two hands of a pair as independent units. In RandomBot
  self-play the two halves of a pair are almost uncorrelated (r ~ +0.05),
  so the wrong SE is nearly the same and covers ~0.94 too. For card-aware
  bots the halves are negatively correlated and the wrong SE is *wider*.
- A wrong bb/100 factor: it scales the mean and the CI together, so
  coverage is unchanged.
Those are tested directly in test_eval_match.py
(test_se_is_computed_over_pairs, test_bb100_conversion_for_pairs).

The runs are short (30-50 pairs), which is the hard case for the normal
approximation: per-pair results are heavy-tailed (stack-sized pots are
rare but huge), so slight under-coverage would not be surprising. The
seeds are fixed, so the test is deterministic.
"""

from __future__ import annotations

import math

import pytest

from nutsfinder.bots import RandomBot
from nutsfinder.eval.match import duplicate_match


def coverage(n_runs: int, n_pairs: int, seed0: int) -> float:
    """Fraction of independent runs whose 95% CI contains the true edge 0."""
    covered = 0
    for run in range(n_runs):
        lo, hi = duplicate_match(RandomBot, RandomBot, n_pairs, master_seed=seed0 + run).ci_bb100
        covered += lo <= 0 <= hi
    return covered / n_runs


def test_ci_coverage_quick() -> None:
    # 200 runs: SD of the fraction is 1.5 points, so the bounds are loose.
    assert 0.88 <= coverage(200, 30, seed0=50_000) <= 0.995


@pytest.mark.slow
def test_ci_coverage_many_runs() -> None:
    n_runs = 1000
    tol = 3 * math.sqrt(0.95 * 0.05 / n_runs)
    c = coverage(n_runs, 50, seed0=1_000_000)
    assert 0.95 - tol <= c <= 0.95 + tol, c
