"""Tests for regret matching and the full-tree CFR solver (vanilla, CFR+, linear, DCFR).

Thresholds were calibrated from recorded runs (this machine, commit a5a8c62
+ the Phase 6 game files) and set at about 2x the recorded value, as
docs/phase6_research.md asks. Recorded exploitability, (BR0 + BR1) / 2:

    game   iterations  vanilla    cfr+       linear     dcfr
    Kuhn   2000        5.29e-3    4.37e-5    4.53e-5    9.19e-5
    Leduc  1000        3.98e-2    2.71e-4    5.62e-3    1.67e-4

Kuhn thresholds: vanilla 1.1e-2, cfr+ 1e-4, linear 1e-4, dcfr 2e-4. Leduc thresholds (1000
iterations): vanilla 8e-2, cfr+ 6e-4, linear 1.2e-2, dcfr 4e-4. The solver
is deterministic, so a change in these numbers means the arithmetic changed.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from nutsfinder.gametheory import CHANCE, Kuhn, Leduc, action_probs, expected_value, exploitability
from nutsfinder.gametheory.cfr import VARIANTS, CFRSolver, Variant, log_spaced
from nutsfinder.gametheory.regret import (
    regret_matching,
    regret_matching_flat,
    regret_matching_plus_update,
)

KUHN, LEDUC = Kuhn(), Leduc()
KUHN_VALUE = -1 / 18
# Leduc value to P0, measured: CFR+ and DCFR at 60,000 iterations both give
# -0.08560642. Certified brackets [-BR1, BR0]: CFR+ [-0.0856075, -0.0856049],
# DCFR [-0.0856071, -0.0856044]; each bounds v*, so v* is in [-0.0856071, -0.0856049].
LEDUC_VALUE = -0.0856064
ALL = list(VARIANTS)

# ------------------------------------------------------------ regret matching


def test_regret_matching_examples():
    assert regret_matching([2.0, -1.0, 1.0]) == pytest.approx([2 / 3, 0.0, 1 / 3])
    assert regret_matching([0.0, 5.0]) == [0.0, 1.0]


@pytest.mark.parametrize("regrets", [[-3.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -0.5], [-1e-300]])
def test_regret_matching_non_positive_is_uniform(regrets):
    assert regret_matching(regrets) == [1.0 / len(regrets)] * len(regrets)


def test_regret_matching_empty_raises():
    with pytest.raises(ValueError):
        regret_matching([])


def test_regret_matching_is_scale_invariant_and_a_distribution():
    rng = random.Random(7)
    for _ in range(200):
        r = [rng.uniform(-5, 5) for _ in range(rng.randint(1, 4))]
        s = regret_matching(r)
        assert sum(s) == pytest.approx(1.0) and min(s) >= 0.0
        assert regret_matching([3.7 * x for x in r]) == pytest.approx(s)
        # Zero probability exactly on the non-positive-regret actions (unless uniform).
        if any(x > 0 for x in r):
            assert all((p == 0.0) == (x <= 0.0) for p, x in zip(s, r))


def test_regret_matching_plus_worked_example():
    """Module-doc example: from R = [3, 0], r = [-5, 2] then [3, -1]."""
    rm, rmp = [3.0, 0.0], [3.0, 0.0]
    rm = [a + b for a, b in zip(rm, [-5.0, 2.0])]
    rmp = regret_matching_plus_update(rmp, [-5.0, 2.0])
    assert rm == [-2.0, 2.0] and rmp == [0.0, 2.0]
    assert regret_matching(rm) == regret_matching(rmp) == [0.0, 1.0]
    rm = [a + b for a, b in zip(rm, [3.0, -1.0])]
    rmp = regret_matching_plus_update(rmp, [3.0, -1.0])
    assert regret_matching(rm) == [0.5, 0.5]
    assert regret_matching(rmp) == [0.75, 0.25]
    with pytest.raises(ValueError):
        regret_matching_plus_update([1.0], [1.0, 2.0])


def test_regret_matching_flat_matches_scalar():
    rng = random.Random(3)
    sizes = [rng.randint(1, 4) for _ in range(300)]
    regrets, segment, size = [], [], []
    for i, n in enumerate(sizes):
        vals = [rng.choice([rng.uniform(-3, 3), 0.0, -1.0]) for _ in range(n)]
        regrets += vals
        segment += [i] * n
        size += [float(n)] * n
    flat = regret_matching_flat(np.array(regrets), np.array(segment), np.array(size))
    pos = 0
    for n in sizes:
        assert list(flat[pos:pos + n]) == pytest.approx(regret_matching(regrets[pos:pos + n]),
                                                        abs=1e-15)
        pos += n


# --------------------------------------------------- recursive reference CFR


def reference_cfr(game, variant: Variant, iterations: int):
    """Textbook recursive CFR straight over the Game protocol, dict tables.

    Independent of the flat solver: same maths, written the slow and
    obvious way. Regrets are buffered during a walk and applied after it, so
    every history of an info set sees the same sigma^t.
    """
    R: dict = {}
    S: dict = {}

    def walk(h, reach, players, inst, avg):
        if game.is_terminal(h):
            return game.utility(h)
        p = game.player(h)
        if p == CHANCE:
            return sum(q * walk(game.next(h, a), (reach[0], reach[1], reach[2] * q), players, inst, avg)
                       for a, q in game.chance_outcomes(h))
        key, acts = game.infoset_key(h), game.actions(h)
        sigma = regret_matching(R.setdefault(key, [0.0] * len(acts)))
        S.setdefault(key, [0.0] * len(acts))
        vals = []
        for a, s in zip(acts, sigma):
            nxt = list(reach)
            nxt[p] *= s
            vals.append(walk(game.next(h, a), tuple(nxt), players, inst, avg))
        node = sum(s * v for s, v in zip(sigma, vals))
        if p in players:
            sign = 1.0 if p == 0 else -1.0
            cf = reach[2] * reach[1 - p]
            ri = inst.setdefault(key, [0.0] * len(acts))
            ai = avg.setdefault(key, [0.0] * len(acts))
            for j in range(len(acts)):
                ri[j] += cf * sign * (vals[j] - node)
                ai[j] += reach[p] * sigma[j]
        return node

    for t in range(1, iterations + 1):
        for players in ([(0,), (1,)] if variant.alternating else [(0, 1)]):
            inst, avg = {}, {}
            walk(game.initial(), (1.0, 1.0, 1.0), players, inst, avg)
            for key in inst:
                r = [x + y * t ** variant.regret_power for x, y in zip(R[key], inst[key])]
                if variant.regret_floor:
                    r = [max(x, 0.0) for x in r]
                a = [x + y * t ** variant.average_power for x, y in zip(S[key], avg[key])]
                if variant.discount is not None:
                    al, be, ga = variant.discount
                    r = [x * (t ** al / (t ** al + 1) if x > 0 else t ** be / (t ** be + 1)) for x in r]
                    a = [x * (t / (t + 1)) ** ga for x in a]
                R[key], S[key] = r, a
    return R, S


def tables(solver: CFRSolver):
    tree = solver.tree
    out_r, out_s = {}, {}
    for key, acts, off in zip(tree.keys, tree.actions, tree.offset):
        out_r[key] = list(solver.regrets[off:off + len(acts)])
        out_s[key] = list(solver.strategy_sum[off:off + len(acts)])
    return out_r, out_s


@pytest.mark.parametrize("variant", ALL)
@pytest.mark.parametrize("game,iters", [(KUHN, 25), (LEDUC, 4)], ids=["kuhn", "leduc"])
def test_flat_solver_matches_recursive_reference(variant, game, iters):
    solver = CFRSolver(game, variant)
    solver.run(iters)
    ref_r, ref_s = reference_cfr(game, solver.variant, iters)
    got_r, got_s = tables(solver)
    assert set(got_r) == set(ref_r)
    for key in ref_r:
        scale = 1.0 + max(abs(x) for x in ref_r[key] + ref_s[key])
        assert got_r[key] == pytest.approx(ref_r[key], abs=1e-12 * scale), key
        assert got_s[key] == pytest.approx(ref_s[key], abs=1e-12 * scale), key


def test_first_iteration_regret_worked_example():
    """Module-doc example: after one vanilla iteration R("Kb") = [-1/4, +1/4]."""
    s = CFRSolver(KUHN, "vanilla")
    s.step()
    off = s.tree.offset[s.tree.keys.index("Kb")]
    assert list(s.regrets[off:off + 2]) == pytest.approx([-0.25, 0.25], abs=1e-15)
    assert s.current_strategy()["Kb"] == {"p": 0.0, "b": 1.0}


@pytest.mark.parametrize("game", [KUHN, LEDUC], ids=["kuhn", "leduc"])
def test_flat_tree_values_match_expected_value(game):
    """The vectorised forward/backward passes compute the same root value as
    best_response.expected_value for an arbitrary profile."""
    solver = CFRSolver(game)
    tree = solver.tree
    rng = random.Random(11)
    probs = np.empty(tree.n_slots)
    profile = {}
    for key, acts, off in zip(tree.keys, tree.actions, tree.offset):
        w = [rng.random() + 1e-3 for _ in acts]
        probs[off:off + len(acts)] = [x / sum(w) for x in w]
        profile[key] = list(probs[off:off + len(acts)])
    root = tree.values(tree.edge_probs(probs))[0]
    assert root == pytest.approx(expected_value(game, profile), abs=1e-12)


def test_linear_cfr_equals_dcfr_1_1_1():
    """Weighting iteration t by t equals discounting by t/(t+1): same strategies."""
    lin = CFRSolver(LEDUC, "linear")
    dis = CFRSolver(LEDUC, Variant("dcfr111", alternating=True, discount=(1.0, 1.0, 1.0)))
    lin.run(30)
    dis.run(30)
    for a, b in ((lin.average_strategy(), dis.average_strategy()),
                 (lin.current_strategy(), dis.current_strategy())):
        for key in a:
            assert list(a[key].values()) == pytest.approx(list(b[key].values()), abs=1e-9), key


def test_overrides_and_unknown_variant():
    s = CFRSolver(KUHN, "cfr+", alternating=False)
    assert s.variant.alternating is False and s.variant.regret_floor is True
    assert VARIANTS["cfr+"].alternating is True  # the registry is not mutated
    with pytest.raises(KeyError):
        CFRSolver(KUHN, "mccfr")


# -------------------------------------------------------------- Kuhn results

KUHN_ITERS = 2000
KUHN_THRESHOLD = {"vanilla": 1.1e-2, "cfr+": 1e-4, "linear": 1e-4, "dcfr": 2e-4}


@pytest.fixture(scope="module")
def kuhn_solved():
    out = {}
    for v in ALL:
        s = CFRSolver(KUHN, v)
        s.run(KUHN_ITERS)
        out[v] = s
    return out


def assert_normalised(profile, game_sets):
    for key, probs in profile.items():
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
        assert min(probs.values()) >= 0.0
        assert tuple(probs) == game_sets[key]


@pytest.mark.parametrize("variant", ALL)
def test_kuhn_every_variant_converges(kuhn_solved, variant):
    s = kuhn_solved[variant]
    prof = s.average_strategy()
    assert_normalised(prof, dict(zip(s.tree.keys, s.tree.actions)))
    assert len(prof) == 12
    cp = s.evaluate()
    assert cp.exploitability == pytest.approx(exploitability(KUHN, prof), abs=1e-15)
    assert cp.exploitability < KUHN_THRESHOLD[variant]
    assert abs(cp.value - KUHN_VALUE) < 1e-3
    lo, hi = cp.value_bounds
    assert lo - 1e-12 <= KUHN_VALUE <= hi + 1e-12


@pytest.mark.parametrize("variant", ALL)
def test_kuhn_strategy_in_equilibrium_family(kuhn_solved, variant):
    """Kuhn (1950): P0 bets J w.p. alpha in [0, 1/3], K w.p. 3 alpha, never Q;
    calls Q after check-bet w.p. alpha + 1/3. P1 is unique: calls Q w.p. 1/3,
    bluffs J after a check w.p. 1/3. Tolerance 0.02 (vanilla 0.05) covers the
    remaining exploitability."""
    p = kuhn_solved[variant].average_strategy()
    tol = 0.05 if variant == "vanilla" else 0.02
    alpha = p["J"]["b"]
    assert -tol <= alpha <= 1 / 3 + tol
    assert p["K"]["b"] == pytest.approx(3 * alpha, abs=3 * tol)
    assert p["Q"]["b"] == pytest.approx(0.0, abs=tol)
    assert p["Qpb"]["b"] == pytest.approx(alpha + 1 / 3, abs=tol)
    assert p["Jpb"]["b"] == pytest.approx(0.0, abs=tol)
    assert p["Kpb"]["b"] == pytest.approx(1.0, abs=tol)
    assert p["Qb"]["b"] == pytest.approx(1 / 3, abs=tol)
    assert p["Jp"]["b"] == pytest.approx(1 / 3, abs=tol)
    assert p["Kb"]["b"] == pytest.approx(1.0, abs=tol)
    assert p["Jb"]["b"] == pytest.approx(0.0, abs=tol)


def fitted_slope(checkpoints, start: int) -> float:
    pts = [(math.log10(c.iteration), math.log10(c.exploitability))
           for c in checkpoints if c.iteration >= start]
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def test_kuhn_vanilla_log_log_slope():
    """Vanilla CFR's exploitability falls roughly like 1/sqrt(T), the
    theoretical rate. Recorded least-squares slope of log10(exploitability)
    on log10(T) over T = 10..2000: -0.546. Threshold -0.45 (a rate claim, so
    a margin on the slope rather than 2x a value)."""
    s = CFRSolver(KUHN, "vanilla")
    cps = s.run(KUHN_ITERS, eval_at=log_spaced(KUHN_ITERS, 8))
    assert [c.iteration for c in cps] == log_spaced(KUHN_ITERS, 8)
    assert fitted_slope(cps, 10) < -0.45


def test_determinism_bit_identical():
    for v in ALL:
        a, b = CFRSolver(LEDUC, v), CFRSolver(LEDUC, v)
        a.run(20)
        b.run(20)
        assert np.array_equal(a.regrets, b.regrets) and np.array_equal(a.strategy_sum, b.strategy_sum)
        assert a.average_strategy() == b.average_strategy()


def test_zero_iterations_is_uniform():
    """Nothing accumulated -> every info set uniform (the never-reached rule)."""
    s = CFRSolver(LEDUC, "cfr+")
    prof = s.average_strategy()
    assert all(action_probs(prof, k, tuple(v)) == pytest.approx([1 / len(v)] * len(v))
               for k, v in prof.items())
    assert s.evaluate().exploitability == pytest.approx(2.3736111111, abs=1e-9)


def test_run_checkpoints():
    s = CFRSolver(KUHN, "dcfr")
    seen = []
    cps = s.run(10, eval_every=4, eval_at=[1, 10], callback=seen.append)
    assert [c.iteration for c in cps] == [1, 4, 8, 10] and seen == cps
    more = s.run(5, eval_every=4)
    assert [c.iteration for c in more] == [12] and s.iteration == 15
    assert all(c.seconds >= 0 for c in cps)


# ------------------------------------------------------------- Leduc results

LEDUC_ITERS = 1000
LEDUC_THRESHOLD = {"vanilla": 8e-2, "cfr+": 6e-4, "linear": 1.2e-2, "dcfr": 4e-4}


@pytest.fixture(scope="module")
def leduc_solved():
    out = {}
    for v in ALL:
        s = CFRSolver(LEDUC, v)
        s.run(LEDUC_ITERS)
        out[v] = s.evaluate()
    return out


@pytest.mark.parametrize("variant", ALL)
def test_leduc_thresholds(leduc_solved, variant):
    assert leduc_solved[variant].exploitability < LEDUC_THRESHOLD[variant]


def test_leduc_cfr_plus_beats_vanilla_at_equal_iterations(leduc_solved):
    e = {v: cp.exploitability for v, cp in leduc_solved.items()}
    assert e["cfr+"] < e["vanilla"] / 10
    assert e["dcfr"] < e["vanilla"] / 10
    assert e["linear"] < e["vanilla"]
    for n in (50, 200):  # and already early on
        runs = {}
        for v in ("vanilla", "cfr+"):
            s = CFRSolver(LEDUC, v)
            s.run(n)
            runs[v] = s.evaluate().exploitability
        assert runs["cfr+"] < runs["vanilla"]


def test_leduc_value_near_equilibrium(leduc_solved):
    """At 1000 CFR+ / DCFR iterations the value is already within 1e-3 of the
    value found by the long run (LEDUC_VALUE below)."""
    for v in ("cfr+", "dcfr"):
        assert abs(leduc_solved[v].value - LEDUC_VALUE) < 1e-3
        assert abs(leduc_solved[v].value - (-0.0856)) < 5e-3  # research-note target


@pytest.mark.slow
def test_leduc_value_certified_by_long_cfr_plus_run():
    """CFR+ 10,000 iterations: the BR values bracket the game value,
    -BR1(avg_0) <= v* <= BR0(avg_1). Recorded width 1.07e-5; threshold 2.2e-5.
    The bracket must contain LEDUC_VALUE (from the 60,000-iteration runs)."""
    s = CFRSolver(LEDUC, "cfr+")
    s.run(10000)
    cp = s.evaluate()
    lo, hi = cp.value_bounds
    assert hi - lo < 2.2e-5
    assert lo <= LEDUC_VALUE <= hi
