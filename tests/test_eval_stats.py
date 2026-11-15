"""Tests for nutsfinder.eval.stats against values worked out by hand.

Data [1, 2, 3, 4]:
    mean = 10 / 4 = 2.5
    squared deviations = 2.25 + 0.25 + 0.25 + 2.25 = 5
    sample SD = sqrt(5 / 3) = 1.2909944
    SE = SD / sqrt(4) = 0.6454972
    95% CI = 2.5 +/- 1.959964 * 0.6454972 = 2.5 +/- 1.2651512
    z = 2.5 / 0.6454972 = 3.8729833, two-sided p = 1.0757e-4
"""

from __future__ import annotations

import math

import pytest

from nutsfinder.eval.stats import (
    mean,
    normal_ci,
    required_units,
    rule_of_thumb_units,
    sample_sd,
    standard_error,
    summarize,
    z_critical,
    z_test_p_value,
)

DATA = [1.0, 2.0, 3.0, 4.0]


def test_mean_sd_se_by_hand() -> None:
    assert mean(DATA) == 2.5
    assert sample_sd(DATA) == pytest.approx(math.sqrt(5 / 3))
    assert sample_sd(DATA) == pytest.approx(1.2909944, abs=1e-7)
    assert standard_error(DATA) == pytest.approx(0.6454972, abs=1e-7)


def test_sd_uses_n_minus_1() -> None:
    # [0, 2]: deviations from the mean 1 are -1, +1. Dividing by n - 1 = 1
    # gives variance 2 (dividing by n would give 1).
    assert sample_sd([0.0, 2.0]) == pytest.approx(math.sqrt(2))


def test_critical_values() -> None:
    assert z_critical(0.95) == pytest.approx(1.959964, abs=1e-6)
    assert z_critical(0.99) == pytest.approx(2.575829, abs=1e-6)


def test_ci_by_hand() -> None:
    lo, hi = normal_ci(2.5, 0.6454972)
    assert lo == pytest.approx(2.5 - 1.2651512, abs=1e-6)
    assert hi == pytest.approx(2.5 + 1.2651512, abs=1e-6)


def test_p_values_by_hand() -> None:
    # z = 1: P(|Z| >= 1) = 0.3173105 (the "68% within 1 SD" rule).
    assert z_test_p_value(1.0, 1.0) == pytest.approx(0.3173105, abs=1e-6)
    # z = 1.959964 is the 95% boundary, so p = 0.05.
    assert z_test_p_value(1.959964, 1.0) == pytest.approx(0.05, abs=1e-6)
    # Sign does not matter for a two-sided test.
    assert z_test_p_value(-2.0, 1.0) == z_test_p_value(2.0, 1.0)
    assert z_test_p_value(2.5, 0.6454972) == pytest.approx(1.0757e-4, rel=1e-3)


def test_p_value_with_zero_se_is_undefined() -> None:
    assert math.isnan(z_test_p_value(0.0, 0.0))
    assert math.isnan(z_test_p_value(0.3, 0.0))


def test_zero_spread_is_never_significant() -> None:
    # Three identical results are not proof of an edge, however large.
    s = summarize([3.0, 3.0, 3.0], hands_per_unit=2)
    assert s.no_variation
    assert not s.significant
    assert math.isnan(s.p_value)


def test_p_value_accurate_in_far_tail() -> None:
    # 1 - cdf(z) rounds to 0 near z = 8.3; cdf(-z) must not.
    assert z_test_p_value(9.0, 1.0) == pytest.approx(2.2571768e-19, rel=1e-6)
    assert z_test_p_value(30.0, 1.0) > 0


def test_summarize_and_bb100_scaling() -> None:
    s = summarize(DATA, hands_per_unit=2)
    assert (s.n, s.mean) == (4, 2.5)
    # 2.5 bb per 2-hand pair = 1.25 bb/hand = 125 bb/100.
    assert s.mean_bb100 == pytest.approx(125.0)
    assert s.se_bb100 == pytest.approx(0.6454972 * 50, abs=1e-5)
    assert s.ci_bb100[0] == pytest.approx((2.5 - 1.2651512) * 50, abs=1e-4)
    assert s.significant  # p = 1e-4
    # p-value is scale-free: same as the unscaled test.
    assert s.p_value == pytest.approx(z_test_p_value(2.5, 0.6454972))


def test_p_below_005_iff_ci_excludes_zero() -> None:
    for m in (0.5, 1.0, 1.9, 1.97, 2.5, -1.97, -1.9):
        lo, hi = normal_ci(m, 1.0)
        assert (z_test_p_value(m, 1.0) < 0.05) == (not lo <= 0 <= hi)


def test_required_units_by_hand() -> None:
    # sigma/e = 10: n = ((1.959964 + 0.841621) * 10)^2 = 784.89 -> 785.
    assert required_units(1.0, 0.1) == 785
    # Only the ratio sigma/e matters, and the sign of the edge does not.
    assert required_units(40.0, -4.0) == 785
    # Rule of thumb: (2 * 10)^2 = 400, about half (it is the 50%-power n).
    assert rule_of_thumb_units(1.0, 0.1) == 400
    # Docstring example: sigma 40 bb/hand, edge 5 bb/100 -> about 5 million.
    assert required_units(40.0, 0.05) == pytest.approx(5.02e6, rel=0.01)


def test_errors() -> None:
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        sample_sd([1.0])
    with pytest.raises(ValueError):
        required_units(1.0, 0.0)
