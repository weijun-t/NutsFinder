"""Tests for external-sampling MCCFR (``nutsfinder.gametheory.mccfr``).

Thresholds were calibrated from recorded runs (commit 24d73e6 + this file's
module, *after* the fix for lost regret updates on array growth) at about 2x
the recorded value, as docs/phase6_research.md asks, checked over several
seeds. Exploitability (BR0 + BR1) / 2 and |value - v*|:

    game   iters    method   seed 0              other seeds
    Kuhn   10,000   plain    1.28e-2, 5.1e-4     seeds 1-4: 4.9e-3 6.3e-3 3.8e-3 5.0e-3; |diff| <= 2.7e-4
    Kuhn   10,000   linear   1.04e-2, 1.5e-4     seeds 1-4: 7.3e-3 8.1e-3 4.5e-3 7.5e-3; |diff| <= 4.7e-4
    Leduc  5,000    linear   0.210, 0.0436       seeds 1-7: 0.170-0.249; |diff| 0.003-0.035
    Leduc  30,000   linear   0.0964, 0.0060      seeds 1-3: 0.0936 0.0894 0.0698; |diff| <= 0.0048
    Leduc  100,000  linear   0.0477, 0.0070      seeds 1-2: 0.0437 0.0515; |diff| 0.0029 0.0012
    Leduc  100,000  plain    0.0528, 0.0010      seeds 1-2: 0.0439 0.0460; |diff| 0.0002 0.0069

Seed 0 is the worst of five Kuhn seeds and the worst |value - v*| of eight
Leduc 5k seeds, so the thresholds are not tuned to a lucky seed; the
multi-seed tests check other seeds against the same thresholds. The value
tolerances are about 2x the worst seed; the certified-bracket checks
(-BR1 <= v* <= BR0, |value - v*| <= BR0 + BR1) hold for any profile and
are kept as exact checks. Unlike the full-tree solver, these numbers depend on the RNG
stream: a change in them means either the arithmetic or the sampling order
changed.

The unbiasedness tests are Monte Carlo tests of the identities derived in
the module docstring: from a fixed (random) regret table, the mean of many
single-traversal updates must match the exact full-tree quantity within a
few standard errors. With ~900 (Leduc) or 24 (Kuhn) slots, 5 SE per slot is
a family-wise false-alarm rate below 1e-3.
"""

from __future__ import annotations

import numpy as np
import pytest

from nutsfinder.gametheory import Kuhn, Leduc, exploitability, infosets
from nutsfinder.gametheory.cfr import Checkpoint, CFRSolver
from nutsfinder.gametheory.mccfr import MCCFRCheckpoint, MCCFRSolver
from nutsfinder.gametheory.regret import regret_matching_flat

KUHN, LEDUC = Kuhn(), Leduc()
KUHN_VALUE = -1 / 18
LEDUC_VALUE = -0.0856064  # certified bracket [-0.0856071, -0.0856049], see test_cfr.py

KUHN_ITERS = 10_000
KUHN_THRESHOLD = {False: 2.6e-2, True: 2.1e-2}  # recorded 1.28e-2 / 1.04e-2 (seed 0)
LEDUC_FAST_ITERS = 5_000
LEDUC_FAST_THRESHOLD = 0.40  # linear, recorded 0.210 (seed 0); worst of 8 seeds 0.249
LEDUC_FAST_VALUE_TOL = 0.08  # recorded |diff| 0.0436 (seed 0, worst of 8 seeds)
LEDUC_SEED_ITERS = 30_000
LEDUC_SEED_THRESHOLD = 0.19  # linear, recorded 0.0964 at seed 0 (seeds 1-3 <= 0.0936)
LEDUC_LONG_ITERS = 100_000
LEDUC_LONG_THRESHOLD = 0.10  # linear, recorded 0.0477 (seed 0); 6 runs at 100k: 0.0437-0.0528
LEDUC_LONG_VALUE_TOL = 0.015  # recorded |diff| 0.0070; worst of the 6 runs 0.0070


def known(game):
    """Info sets in the full-tree solver's order (P0 first), as ``key -> (player, actions)``."""
    tree = CFRSolver(game).tree
    return {k: (int(p), a) for k, p, a in zip(tree.keys, tree.owner, tree.actions)}


def check_profile(game, prof):
    sets = infosets(game)
    for key, probs in prof.items():
        assert tuple(probs) == sets[key][1]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
        assert min(probs.values()) >= 0.0


# ------------------------------------------------------------------- basics


def test_checkpoint_is_a_cfr_checkpoint_and_run_contract():
    s = MCCFRSolver(KUHN, seed=3)
    seen = []
    cps = s.run(10, eval_every=4, eval_at=[1, 10], callback=seen.append)
    assert [c.iteration for c in cps] == [1, 4, 8, 10] and seen == cps
    assert all(isinstance(c, MCCFRCheckpoint) and isinstance(c, Checkpoint) for c in cps)
    assert [c.traversals for c in cps] == [2, 8, 16, 20]
    assert all(a.nodes < b.nodes for a, b in zip(cps, cps[1:]))
    more = s.run(5, eval_every=4)
    assert [c.iteration for c in more] == [12] and s.iteration == 15 and s.traversals == 30
    lo, hi = more[0].value_bounds
    assert lo - 1e-12 <= KUHN_VALUE <= hi + 1e-12


def test_max_seconds_stops_early():
    s = MCCFRSolver(KUHN, seed=0)
    s.run(10**9, max_seconds=0.05)
    assert 0 < s.iteration < 10**9 and s.seconds >= 0.05


def test_weights():
    assert [MCCFRSolver(KUHN).weight(t) for t in (1, 5)] == [1.0, 1.0]
    assert [MCCFRSolver(KUHN, linear=True).weight(t) for t in (1, 5)] == [1.0, 5.0]
    assert [MCCFRSolver(KUHN, linear=True, linear_until=3).weight(t) for t in (2, 3, 9)] == [2.0, 3.0, 3.0]
    with pytest.raises(ValueError):
        MCCFRSolver(KUHN, linear_until=0)
    with pytest.raises(ValueError):
        MCCFRSolver(KUHN).traverse(2)


def test_zero_iterations_is_uniform_and_matches_cfr():
    s = MCCFRSolver(LEDUC, known_infosets=known(LEDUC))
    assert s.evaluate().exploitability == pytest.approx(2.3736111111, abs=1e-9)
    assert MCCFRSolver(LEDUC).average_strategy() == {}  # nothing discovered -> all uniform


def test_lazy_discovery_finds_every_infoset_with_correct_owner_and_actions():
    s = MCCFRSolver(LEDUC, seed=0)
    s.run(3000)
    sets = infosets(LEDUC)
    assert len(s.keys) == len(sets) == 288
    for key, p, acts, off in zip(s.keys, s.owner, s.actions, s.offsets):
        assert sets[key] == (p, acts)
    assert list(s.offsets) == list(np.cumsum([0] + [len(a) for a in s.actions])[:-1])
    check_profile(LEDUC, s.average_strategy())
    check_profile(LEDUC, s.current_strategy())


def test_known_infosets_fix_the_order():
    k = known(KUHN)
    s = MCCFRSolver(KUHN, seed=0, known_infosets=k)
    s.run(50)
    assert s.keys == list(k)


def test_lazy_discovery_equals_preregistered_run():
    """Regression: growing the tables mid-traversal used to drop the regret
    updates of every node above the discovery (they were written into the
    discarded array). Index order does not affect the RNG stream, so a run
    that discovers info sets lazily (many reallocations) must equal one that
    registered them all up front (no reallocation), key by key, bit for bit."""
    lazy = MCCFRSolver(LEDUC, seed=0)
    pre = MCCFRSolver(LEDUC, seed=0, known_infosets=infosets(LEDUC))
    lazy.run(2000)
    pre.run(2000)
    assert lazy.nodes == pre.nodes
    assert lazy.tables() == pre.tables()


def test_known_infosets_are_checked_against_the_game():
    k = known(KUHN)
    bad_actions = dict(k, Kb=(1, ("b", "p")))
    with pytest.raises(ValueError, match="Kb"):
        MCCFRSolver(KUHN, seed=0, known_infosets=bad_actions).run(200)
    bad_player = dict(k, Kb=(0, ("p", "b")))
    with pytest.raises(ValueError, match="Kb"):
        MCCFRSolver(KUHN, seed=0, known_infosets=bad_player).run(200)


def test_tables_grow_past_initial_capacity():
    """Leduc's 288 info sets need ~700 slots, far over the 64 allocated at start."""
    s = MCCFRSolver(LEDUC, seed=0, known_infosets=known(LEDUC))
    assert len(s.regrets) == sum(len(a) for a in s.actions) > 64
    assert not s.regrets.any() and not s.strategy_sum.any()


# --------------------------------------------------------------- determinism


def test_same_seed_bit_identical_different_seed_differs():
    def solve(seed, **kw):
        s = MCCFRSolver(LEDUC, seed=seed, **kw)
        s.run(500)
        return s

    for kw in ({}, {"linear": True}, {"prune_threshold": -5.0, "prune_after": 100}):
        a, b, c = solve(7, **kw), solve(7, **kw), solve(8, **kw)
        assert a.keys == b.keys
        assert np.array_equal(a.regrets, b.regrets) and np.array_equal(a.strategy_sum, b.strategy_sum)
        assert a.average_strategy() == b.average_strategy() and a.nodes == b.nodes
        ra, sa = a.tables()
        rc, sc = c.tables()
        assert ra != rc and sa != sc


def test_no_global_rng_use():
    np.random.seed(1)
    import random
    random.seed(1)
    before = (np.random.random(), random.random())
    np.random.seed(1)
    random.seed(1)
    MCCFRSolver(KUHN, seed=0).run(200)
    assert (np.random.random(), random.random()) == before


# ------------------------------------------------------------- unbiasedness


def mc_check(game, n, seed=0):
    """Mean single-traversal updates from a fixed random regret table vs exact values.

    Returns z-scores (mean - exact) / SE for regret increments, average-strategy
    increments and root values. Slots with SE 0 must match exactly.
    """
    cfr = CFRSolver(game, "vanilla")  # simultaneous, unweighted: delta = exact instant regret
    tree = cfr.tree
    base = np.random.default_rng(seed).normal(size=tree.n_slots)
    cfr.regrets[:] = base
    cfr.step()
    exact_r = cfr.regrets - base
    sigma = regret_matching_flat(base, tree.slot_infoset, tree.slot_size)
    ep = tree.edge_probs(sigma)
    reach = tree.reaches(ep)
    exact_s = np.zeros(tree.n_slots)
    for p in (0, 1):  # visited with prob pi_c(h) pi_p(h) during the other player's traversal
        child, par, slot = tree.edges[p]
        exact_s += np.bincount(slot, weights=reach[2, par] * reach[p, par] * ep[child],
                               minlength=tree.n_slots)
    exact_v = tree.values(ep)[0]

    m = MCCFRSolver(game, seed=seed + 1, known_infosets=known(game))
    assert m.keys == tree.keys
    sums = {k: np.zeros(tree.n_slots) for k in ("r", "r2", "s", "s2")}
    vals = {0: [], 1: []}
    for _ in range(n):
        for p in (0, 1):
            m.regrets[:] = base
            m.strategy_sum[:] = 0.0
            vals[p].append(m.traverse(p))
            dr, ds = m.regrets - base, m.strategy_sum.copy()
            sums["r"] += dr
            sums["r2"] += dr * dr
            sums["s"] += ds
            sums["s2"] += ds * ds

    def z(total, total2, exact):
        mean = total / n
        se = np.sqrt(np.maximum(total2 / n - mean ** 2, 0.0) / n)
        dev = np.abs(mean - exact)
        assert np.all(dev[se == 0.0] < 1e-12)
        return np.where(se > 0.0, dev / np.where(se > 0.0, se, 1.0), 0.0)

    zv = [abs(np.mean(vals[p]) - (exact_v if p == 0 else -exact_v)) / (np.std(vals[p]) / np.sqrt(n))
          for p in (0, 1)]
    return z(sums["r"], sums["r2"], exact_r), z(sums["s"], sums["s2"], exact_s), zv


def test_kuhn_sampled_updates_are_unbiased():
    """E[regret update] = full-tree CFR regret; E[average update] = pi_c * own reach * sigma;
    E[sampled root value] = exact value. 20,000 traversals per player."""
    zr, zs, zv = mc_check(KUHN, 20_000)
    assert zr.max() < 5.0 and zs.max() < 5.0 and max(zv) < 5.0


@pytest.mark.slow
def test_leduc_sampled_updates_are_unbiased():
    zr, zs, zv = mc_check(LEDUC, 20_000, seed=4)
    assert zr.max() < 5.0 and zs.max() < 5.0 and max(zv) < 5.0


def test_kuhn_worked_example_expected_update():
    """Module-doc example: from zero regrets, E[update of "Kb"] = [-1/4, +1/4];
    a traversal deals one card pair, so it visits "Kb" at most once (prob
    1/3 * 1/2 = 1/6) and then adds exactly [-1.5, +1.5]."""
    m = MCCFRSolver(KUHN, seed=0, known_infosets=known(KUHN))
    off = m.offsets[m.keys.index("Kb")]
    n, total = 4000, np.zeros(2)
    for _ in range(n):
        m.regrets[:] = 0.0
        m.traverse(1)
        d = m.regrets[off:off + 2]
        assert list(d) in ([0.0, 0.0], [-1.5, 1.5])
        total += d
    # Per traversal the update is 1.5 * Bernoulli(1/6): SE of the mean 1.5 * sqrt(p (1 - p) / n).
    assert total / n == pytest.approx([-0.25, 0.25], abs=5 * 1.5 * np.sqrt((1 / 6) * (5 / 6) / n))


# --------------------------------------------------------------- convergence


@pytest.fixture(scope="module")
def kuhn_solved():
    out = {}
    for linear in (False, True):
        s = MCCFRSolver(KUHN, seed=0, linear=linear)
        s.run(KUHN_ITERS)
        out[linear] = s
    return out


@pytest.mark.parametrize("linear", [False, True], ids=["plain", "linear"])
def test_kuhn_converges(kuhn_solved, linear):
    s = kuhn_solved[linear]
    prof = s.average_strategy()
    assert len(prof) == 12
    check_profile(KUHN, prof)
    cp = s.evaluate()
    assert cp.exploitability == pytest.approx(exploitability(KUHN, prof), abs=1e-15)
    assert cp.exploitability < KUHN_THRESHOLD[linear]
    assert abs(cp.value - KUHN_VALUE) < 2e-3  # recorded |diff| <= 6e-4 over 10 runs
    lo, hi = cp.value_bounds
    assert lo - 1e-12 <= KUHN_VALUE <= hi + 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kuhn_other_seeds_converge(seed):
    s = MCCFRSolver(KUHN, seed=seed)
    s.run(KUHN_ITERS)
    assert s.evaluate().exploitability < KUHN_THRESHOLD[False]


def test_kuhn_pruning_saves_nodes_and_still_converges():
    """Recorded (seed 0, 10k iterations, C = -20 after 1,000): 126,203 vs 147,270
    nodes, exploitability 1.81e-2 vs 1.28e-2. Pruning trades accuracy per
    iteration for cost; on a game this small it is not worth it (default off)."""
    plain = MCCFRSolver(KUHN, seed=0)
    pruned = MCCFRSolver(KUHN, seed=0, prune_threshold=-20.0, prune_after=1000)
    plain.run(KUHN_ITERS)
    pruned.run(KUHN_ITERS)
    assert pruned.nodes < 0.95 * plain.nodes
    assert pruned.evaluate().exploitability < 3.6e-2


def test_prune_threshold_must_be_negative():
    for c in (0.0, 5.0):
        with pytest.raises(ValueError):
            MCCFRSolver(KUHN, prune_threshold=c)


def test_pruning_never_skips_actions_still_played():
    """Regression: with every regret -1 regret matching is uniform, so no
    action may be pruned even though all are below C = -0.5. The bug pruned
    them anyway (their child values stayed 0), and the mean sampled root value
    came out 0.0 instead of the exact value of the uniform profile, +1/8 to P0."""
    from nutsfinder.gametheory import expected_value

    s = MCCFRSolver(KUHN, seed=0, prune_threshold=-0.5, prune_prob=1.0,
                    known_infosets=known(KUHN))
    s.iteration = 1  # past the warm-up (prune_after=0)
    exact = expected_value(KUHN, {})  # uniform everywhere
    vals = []
    for _ in range(20_000):
        s.regrets[:] = -1.0
        vals.append(s.traverse(0))
    se = np.std(vals) / np.sqrt(len(vals))
    assert abs(np.mean(vals) - exact) < 5 * se


def assert_leduc_value(cp, tol):
    lo, hi = cp.value_bounds
    assert lo <= LEDUC_VALUE <= hi  # each player's average guarantees its end of the bracket
    assert abs(cp.value - LEDUC_VALUE) <= hi - lo  # so |value - v*| <= BR0 + BR1
    assert abs(cp.value - LEDUC_VALUE) < tol


def test_leduc_fast():
    s = MCCFRSolver(LEDUC, seed=0, linear=True)
    s.run(LEDUC_FAST_ITERS)
    cp = s.evaluate()
    check_profile(LEDUC, s.average_strategy())
    assert cp.exploitability < LEDUC_FAST_THRESHOLD
    assert_leduc_value(cp, LEDUC_FAST_VALUE_TOL)


@pytest.mark.slow
def test_leduc_long_run_reaches_low_exploitability():
    """Research-note target: ES on Leduc below ~0.05 chips. Recorded 0.0477 at
    100,000 linear iterations (value -0.09259, |diff| 0.0070)."""
    s = MCCFRSolver(LEDUC, seed=0, linear=True)
    s.run(LEDUC_LONG_ITERS)
    cp = s.evaluate()
    assert cp.exploitability < LEDUC_LONG_THRESHOLD
    assert_leduc_value(cp, LEDUC_LONG_VALUE_TOL)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_leduc_other_seeds_converge(seed):
    s = MCCFRSolver(LEDUC, seed=seed, linear=True)
    s.run(LEDUC_SEED_ITERS)
    cp = s.evaluate()
    assert cp.exploitability < LEDUC_SEED_THRESHOLD
    assert_leduc_value(cp, 0.02)


def test_reseed_equals_a_fresh_solver():
    a = MCCFRSolver(LEDUC, seed=5, linear=True, known_infosets=known(LEDUC))
    a.run(50)
    a.reseed(9)
    r0, s0 = a.regrets.copy(), a.strategy_sum.copy()
    a.run(30)
    assert a.seed == 9
    # A fresh solver seeded 9, given a's tables and iteration count, must make the same 30 iterations.
    c = MCCFRSolver(LEDUC, seed=9, linear=True, known_infosets=known(LEDUC))
    c.regrets[:] = r0
    c.strategy_sum[:] = s0
    c.iteration = 50
    c.run(30)
    assert np.array_equal(a.regrets, c.regrets) and np.array_equal(a.strategy_sum, c.strategy_sum)
