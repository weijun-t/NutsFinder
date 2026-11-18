"""Phase 7 bankroll maths: closed forms, exact sample versions, estimators and the Monte Carlo simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nutsfinder.bankroll import kelly as K
from nutsfinder.bankroll.estimate import SessionData, bootstrap_estimates, point_estimates
from nutsfinder.bankroll.simulate import (
    block_summaries, ror_curve, running_min_fixed_stake, simulate_continuous_kelly, simulate_stake_levels, wilson_ci,
)

# --- Closed forms -----------------------------------------------------------------------


def test_risk_of_ruin_and_bankroll_are_inverse():
    mu, sigma = 0.68, 16.9
    for r in (0.1, 0.01, 0.001):
        b = K.bankroll_for_ruin(mu, sigma, r)
        assert K.risk_of_ruin(mu, sigma, b) == pytest.approx(r, rel=1e-12)
    assert K.risk_of_ruin(mu, sigma, 0.0) == 1.0
    assert K.risk_of_ruin(-0.1, sigma, 1000.0) == 1.0
    assert K.bankroll_for_ruin(0.0, sigma, 0.01) == math.inf
    # Kelly bankroll = the fixed-stake bankroll at ruin e^-2.
    assert K.risk_of_ruin(mu, sigma, K.kelly_bankroll(mu, sigma)) == pytest.approx(math.exp(-2))


def test_kelly_maximises_quadratic_growth():
    mu, sigma = 0.5, 20.0
    f = np.linspace(0, 3 * mu / sigma**2, 30001)
    g = K.growth_rate(f, mu, sigma)
    assert f[np.argmax(g)] == pytest.approx(K.kelly_fraction(mu, sigma), rel=1e-3)
    assert g.max() == pytest.approx(K.kelly_growth(mu, sigma), rel=1e-6)
    # Fractional Kelly: (2c - c^2) of the maximum; zero at 2x Kelly.
    for c in (0.25, 0.5, 1.0, 1.5, 2.0):
        assert K.growth_rate(c * mu / sigma**2, mu, sigma) == pytest.approx(
            K.fractional_growth_ratio(c) * K.kelly_growth(mu, sigma), abs=1e-15)
    assert K.fractional_growth_ratio(2.0) == 0.0


def test_halving_formula_values():
    assert K.prob_ever_halve(1.0, scale=2.0) == pytest.approx(0.25)
    assert K.prob_ever_halve(1.0) == pytest.approx(0.5)
    assert K.prob_ever_halve(0.5) == pytest.approx(1 / 8)
    assert K.prob_ever_halve(2.0) == 1.0
    assert K.prob_ever_fall_to(1.0, 0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        K.prob_ever_fall_to(1.0, 1.5)


def test_exact_kelly_two_point_closed_form():
    # Win 1 w.p. p, lose 1 w.p. q: Kelly f = p - q; Lundberg R = ln(p / q).
    x = np.array([1.0] * 60 + [-1.0] * 40)
    assert K.exact_kelly_fraction(x) == pytest.approx(0.2, rel=1e-9)
    assert K.lundberg_exponent(x) == pytest.approx(math.log(1.5), rel=1e-9)
    # Win b w.p. p: f = p - q / b.
    x = np.array([2.0] * 50 + [-1.0] * 50)
    assert K.exact_kelly_fraction(x) == pytest.approx(0.5 - 0.5 / 2, rel=1e-9)
    assert K.exact_kelly_fraction(np.array([-1.0, 0.5])) == 0.0  # losing game: do not play
    assert K.exact_kelly_fraction(np.array([1.0, 2.0])) == math.inf


def test_exact_versions_match_diffusion_for_small_gaussian_edges():
    rng = np.random.default_rng(16_180_339)
    mu, sigma = 0.05, 1.0
    x = rng.normal(mu, sigma, 400_000)
    m, v = x.mean(), x.var()
    assert K.exact_kelly_fraction(x) == pytest.approx(m / v, rel=0.01)
    assert K.lundberg_exponent(x) == pytest.approx(2 * m / v, rel=0.01)  # Gaussian: exactly 2 mu / sigma^2


def test_parameter_uncertainty_formulas():
    for t in (1.0, 2.0, 13.0):
        ks = np.linspace(0, 1.5, 15001)
        r = K.noisy_plugin_growth_ratio(ks, t)
        assert ks[np.argmax(r)] == pytest.approx(K.optimal_shrinkage(t), abs=2e-4)
    assert K.noisy_plugin_growth_ratio(1.0, 1.0) == pytest.approx(0.0)  # t = 1: plug-in Kelly has zero growth
    assert K.growth_ratio_misspecified(1.0, 0.5) == pytest.approx(0.0)  # full Kelly on a 2x overestimate
    assert K.growth_ratio_misspecified(0.5, 0.5) == pytest.approx(1.0)  # half Kelly is then exactly Kelly
    assert K.growth_ratio_misspecified(0.5, 1.0) == pytest.approx(0.75)


def test_noisy_plugin_formula_by_monte_carlo():
    rng = np.random.default_rng(16_180_339)
    mu, sigma, se = 1.0, 2.0, 0.5
    mu_hat = rng.normal(mu, se, 400_000)
    for k in (0.5, 1.0):
        f = k * mu_hat / sigma**2
        g = (f * mu - f**2 * sigma**2 / 2).mean() / K.kelly_growth(mu, sigma)
        assert g == pytest.approx(K.noisy_plugin_growth_ratio(k, mu / se), abs=0.01)


# --- Simulator ----------------------------------------------------------------------------


def test_running_min_matches_brute_force():
    rng = np.random.default_rng(1)
    blocks = rng.normal(0.1, 3.0, size=(7, 5)).round(2)
    paths, horizon = 50, 9
    m, fin = running_min_fixed_stake(blocks, paths, horizon, np.random.default_rng(42), chunk=horizon)
    idx = np.random.default_rng(42).integers(0, 7, size=(paths, horizon))  # same draws (one chunk)
    seq = blocks[idx].reshape(paths, -1)
    cs = np.cumsum(seq, axis=1)
    np.testing.assert_allclose(m, np.minimum(cs.min(axis=1), 0), atol=1e-9)
    np.testing.assert_allclose(fin, cs[:, -1], atol=1e-9)
    tot, mp = block_summaries(blocks)
    np.testing.assert_allclose(tot, blocks.sum(1))
    np.testing.assert_allclose(mp, np.cumsum(blocks, 1).min(1))


def test_simulation_is_deterministic():
    blocks = np.random.default_rng(3).normal(0.1, 2, size=(20, 10))
    a = running_min_fixed_stake(blocks, 100, 30, np.random.default_rng(5))
    b = running_min_fixed_stake(blocks, 100, 30, np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    c = simulate_continuous_kelly(blocks, 0.01, 50, 10, np.random.default_rng(5))
    d = simulate_continuous_kelly(blocks, 0.01, 50, 10, np.random.default_rng(5))
    np.testing.assert_array_equal(c["logw"], d["logw"])


def test_fixed_stake_ruin_matches_formula_for_small_gaussian_steps():
    # Small steps: the diffusion is accurate (up to the discrete-monitoring overshoot,
    # which lowers the MC value by about exp(-2 mu * 0.58 sigma / sigma^2) ~ 6%).
    rng = np.random.default_rng(16_180_339)
    mu, sigma, B = 0.05, 1.0, 20.0
    blocks = rng.normal(mu, sigma, size=(4000, 100))
    m, _ = running_min_fixed_stake(blocks, 20_000, 60, np.random.default_rng(7))
    r = ror_curve(m, np.array([B]))
    formula = K.risk_of_ruin(mu, sigma, B)
    corrected = K.risk_of_ruin(mu, sigma, B + 0.5826 * sigma)  # Siegmund's overshoot correction
    assert r["lo"][0] - 0.01 <= corrected <= formula <= r["hi"][0] + 0.02


def test_continuous_kelly_halving_matches_formula():
    rng = np.random.default_rng(16_180_339)
    mu, sigma = 0.05, 1.0  # per-step log move ~0.05 << ln 2: the diffusion is accurate
    blocks = rng.normal(mu, sigma, size=(4000, 100))
    f = mu / sigma**2
    for c, want in ((1.0, 0.5), (0.5, 0.125)):
        r = simulate_continuous_kelly(blocks, c * f, 3000, 80, np.random.default_rng(11))
        p = (r["minlog"] <= math.log(0.5)).mean()
        assert p == pytest.approx(want, abs=0.04)
        # Growth: compare with the pool's own mean/variance (the finite pool's mean is ~5% off mu).
        g = r["logw"].mean() / r["hands"]
        m, s = blocks.mean(), blocks.std()
        want_g = c * f * m - (c * f) ** 2 * s**2 / 2
        assert g == pytest.approx(want_g, rel=0.05)


def test_stake_levels_rule_and_ruin():
    # Never losing: never ruined, stakes only go up.
    blocks = np.ones((3, 4))
    r = simulate_stake_levels(blocks, 100.0, 50.0, 10, 20, np.random.default_rng(0))
    assert not r["ruined"].any() and not r["halved"].any()
    # W0 = 100, N = 50 -> stake 2: +8 per block for 20 blocks at first, then more.
    assert (r["W"] > 100 + 8 * 20).all()
    # Always losing 10 per hand, bankroll 30, N huge -> smallest stake 2^-6, ruined eventually.
    r = simulate_stake_levels(-10 * np.ones((2, 4)), 30.0, 1e6, 5, 200, np.random.default_rng(0))
    assert r["ruined"].all()


def test_wilson_ci_basic():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


# --- Estimates ------------------------------------------------------------------------------


def test_estimates_on_synthetic_duplicate_data():
    rng = np.random.default_rng(2)
    n, pairs = 200, 50
    g1 = rng.normal(0.5, 10.0, size=(n, pairs))
    hands = np.empty((n, 2 * pairs))
    hands[:, 0::2] = g1
    hands[:, 1::2] = -g1 + 1.0 + rng.normal(0, 1.0, size=(n, pairs))  # mirror hand: luck cancels
    d = SessionData("synthetic", hands)
    p = point_estimates(d)
    assert p["mu"] == pytest.approx(0.5, abs=0.05)
    assert p["ratio_room"] == pytest.approx(1.0, abs=0.15)  # rooms: independent hands
    assert p["ratio_dup"] < 0.02                             # duplicate: luck cancelled
    assert p["kelly_bankroll"] == pytest.approx((p["var_hand"] + p["mu"] ** 2) / p["mu"])  # per-hand resizing
    assert p["kelly_bankroll_session_gauss"] == pytest.approx(p["var_room"] / p["mu"])
    assert p["bankroll_1pct"] == pytest.approx(p["var_room"] * math.log(100) / (2 * p["mu"]))
    ci = bootstrap_estimates(d, np.random.default_rng(0), 500, ("mu", "kelly_bankroll"))
    assert ci["mu"][0] < p["mu"] < ci["mu"][1]
    assert d.room_blocks.shape == (2 * n, pairs)
    np.testing.assert_array_equal(d.room_blocks[0], hands[0, 0::2])
    np.testing.assert_array_equal(d.room_blocks[1], hands[0, 1::2])


def test_continuous_kelly_block_trick_matches_hand_by_hand_loop():
    rng = np.random.default_rng(0)
    blocks = rng.normal(0.3, 10, (30, 25))
    f = 0.01
    r = simulate_continuous_kelly(blocks, f, 40, 12, np.random.default_rng(5), record_paths=3, record_every=5)
    g = np.random.default_rng(5)  # the same draws, applied hand by hand
    lw, ml, paths = np.zeros(40), np.zeros(40), []
    for _ in range(12):
        idx = g.integers(0, 30, size=40)
        cs = lw[:, None] + np.cumsum(np.log1p(f * blocks[idx]), 1)
        ml = np.minimum(ml, cs.min(1))
        paths.append(cs[:3, 4::5])
        lw = cs[:, -1]
    np.testing.assert_allclose(r["logw"], lw, atol=1e-12)
    np.testing.assert_allclose(r["minlog"], ml, atol=1e-12)
    np.testing.assert_allclose(r["paths"][:, 1:], np.concatenate(paths, 1), atol=1e-12)
    # A fraction that can lose more than everything in one hand wipes the path out.
    r = simulate_continuous_kelly(np.array([[-200.0, 300.0]]), 0.01, 5, 3, np.random.default_rng(0))
    assert r["dead"].all() and np.isneginf(r["logw"]).all()


def test_checkpoint_is_the_same_paths_cut_early():
    blocks = np.random.default_rng(1).normal(0.1, 3.0, size=(7, 5))
    cp = {4: None}
    m, _ = running_min_fixed_stake(blocks, 30, 9, np.random.default_rng(42), chunk=3, checkpoints=cp)
    g = np.random.default_rng(42)
    idx = np.concatenate([g.integers(0, 7, size=(30, k)) for k in (3, 1, 3, 2)], axis=1)  # the chunking used
    cs = np.cumsum(blocks[idx].reshape(30, -1), axis=1)
    np.testing.assert_allclose(cp[4], np.minimum(cs[:, :20].min(axis=1), 0), atol=1e-9)
    np.testing.assert_allclose(m, np.minimum(cs.min(axis=1), 0), atol=1e-9)
    assert (cp[4] >= m).all()


def test_kelly_with_dependent_hands_uses_second_moment_for_drift_and_room_variance_for_drawdowns():
    # Hands within a block share a random offset: var_room ~ 3 x var_hand.
    # Small edge and short blocks: one block moves log W by ~0.25 << ln 2, so the diffusion applies.
    rng = np.random.default_rng(16_180_339)
    mu, n_blocks, L = 0.02, 4000, 50
    offset = rng.normal(0, 0.2, size=(n_blocks, 1))
    blocks = rng.normal(mu, 1.0, size=(n_blocks, L)) + offset
    m = blocks.mean()
    ex2 = (blocks**2).mean()
    var_room = blocks.sum(1).var() / L
    assert var_room / blocks.var() > 2.5
    f_hand, f_room = m / ex2, m / var_room
    g_hand = simulate_continuous_kelly(blocks, f_hand, 3000, 400, np.random.default_rng(3))["logw"].mean()
    g_room = simulate_continuous_kelly(blocks, f_room, 3000, 400, np.random.default_rng(3))["logw"].mean()
    assert g_hand > 1.5 * g_room  # the per-hand Kelly grows faster: sigma_room^2 is the wrong plug-in
    r = simulate_continuous_kelly(blocks, f_hand, 3000, 3000, np.random.default_rng(4))
    p = (r["minlog"] <= math.log(0.5)).mean()
    want = K.prob_ever_halve(1.0, scale=ex2 / var_room)  # ~0.79, not 0.5
    assert p == pytest.approx(want, abs=0.05)
    assert abs(p - 0.5) > 0.15
