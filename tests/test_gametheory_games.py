"""Tests for the Game protocol, Kuhn, Leduc and exact best response.

Kuhn equilibrium family (Kuhn 1950; restated e.g. in Wikipedia "Kuhn poker"
and Neller & Lanctot 2013, sec. 3). With P0 the first player and alpha in
[0, 1/3]:

- P0 opens: bets J with prob alpha, always checks Q, bets K with 3 alpha.
- P0 after check-bet: folds J, calls Q with alpha + 1/3, calls K.
- P1 facing a bet: folds J, calls Q with 1/3, calls K.
- P1 after a check: bets J with 1/3, checks Q, bets K.

Every member gives P0 the value -1/18, and P1's strategy is unique.
"""

from __future__ import annotations

import random
import time

import pytest

from nutsfinder.gametheory import (
    CHANCE,
    Kuhn,
    Leduc,
    action_probs,
    best_response,
    best_response_value,
    expected_value,
    exploitability,
    infosets,
    iter_histories,
    nash_conv,
)
from nutsfinder.gametheory.leduc import _parse

KUHN, LEDUC = Kuhn(), Leduc()


def kuhn_equilibrium(alpha: float) -> dict[str, dict[str, float]]:
    """The Kuhn (1950) family as a profile: info set -> {action: prob}."""
    bet = {"J": alpha, "Q": 0.0, "K": 3 * alpha}
    call_after_check = {"J": 0.0, "Q": alpha + 1 / 3, "K": 1.0}
    p1_call = {"J": 0.0, "Q": 1 / 3, "K": 1.0}
    p1_bet = {"J": 1 / 3, "Q": 0.0, "K": 1.0}
    profile = {}
    for c in "JQK":
        for key, b in ((c, bet[c]), (c + "pb", call_after_check[c]),
                       (c + "b", p1_call[c]), (c + "p", p1_bet[c])):
            profile[key] = {"p": 1.0 - b, "b": b}
    return profile


def always_bet() -> dict[str, dict[str, float]]:
    return {key: {"b": 1.0} for key in infosets(KUHN)}


def cheating_br_value(game, profile, br: int) -> float:
    """Per-history max (sees the opponent's card). Only used to show BR is NOT this."""
    sign = 1.0 if br == 0 else -1.0

    def v(h):
        if game.is_terminal(h):
            return sign * game.utility(h)
        if game.player(h) == CHANCE:
            return sum(p * v(game.next(h, a)) for a, p in game.chance_outcomes(h))
        acts = game.actions(h)
        if game.player(h) == br:
            return max(v(game.next(h, a)) for a in acts)
        probs = action_probs(profile, game.infoset_key(h), acts)
        return sum(p * v(game.next(h, a)) for a, p in zip(acts, probs))

    return v(game.initial())


def random_profile(game, seed: int) -> dict:
    rng = random.Random(seed)
    out = {}
    for key, (_, acts) in infosets(game).items():
        w = [rng.random() + 1e-3 for _ in acts]
        out[key] = [x / sum(w) for x in w]
    return out


# --------------------------------------------------------------------- protocol


def test_action_probs_missing_is_uniform_and_validates():
    assert action_probs({}, "x", ("a", "b", "c")) == [1 / 3] * 3
    assert action_probs({"x": {"b": 1.0}}, "x", ("a", "b")) == [0.0, 1.0]
    assert action_probs({"x": [0.25, 0.75]}, "x", ("a", "b")) == [0.25, 0.75]
    for bad in ({"x": [0.5, 0.6]}, {"x": [1.0]}, {"x": {"z": 1.0}}, {"x": [1.5, -0.5]}):
        with pytest.raises(ValueError):
            action_probs(bad, "x", ("a", "b"))


@pytest.mark.parametrize("game", [KUHN, LEDUC], ids=["kuhn", "leduc"])
def test_chance_probabilities_sum_to_one(game):
    for h in iter_histories(game):
        if not game.is_terminal(h) and game.player(h) == CHANCE:
            outs = game.chance_outcomes(h)
            assert all(p > 0 for _, p in outs)
            assert sum(p for _, p in outs) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("game", [KUHN, LEDUC], ids=["kuhn", "leduc"])
def test_infosets_consistent_and_players_disjoint(game):
    sets = infosets(game)  # raises if one key maps to two players/action tuples
    assert {p for p, _ in sets.values()} == {0, 1}


# ------------------------------------------------------------------------- Kuhn


def test_kuhn_terminal_payoffs():
    u = KUHN.utility
    assert u(("K", "Q", "p", "p")) == 1
    assert u(("J", "Q", "b", "b")) == -2
    assert u(("Q", "J", "p", "b", "b")) == 2
    assert u(("J", "K", "b", "p")) == 1
    assert u(("K", "J", "p", "b", "p")) == -1
    assert len(infosets(KUHN)) == 12


@pytest.mark.parametrize("alpha", [0.0, 1 / 6, 1 / 3])
def test_kuhn_equilibrium_family_is_unexploitable(alpha):
    profile = kuhn_equilibrium(alpha)
    assert expected_value(KUHN, profile) == pytest.approx(-1 / 18, abs=1e-12)
    assert exploitability(KUHN, profile) == pytest.approx(0.0, abs=1e-9)
    # At equilibrium each BR value equals the game value (to that player).
    assert best_response_value(KUHN, profile, 0) == pytest.approx(-1 / 18, abs=1e-9)
    assert best_response_value(KUHN, profile, 1) == pytest.approx(1 / 18, abs=1e-9)


def test_kuhn_perturbed_equilibrium_is_exploitable():
    profile = kuhn_equilibrium(1 / 6)
    profile["Qb"] = {"p": 0.5, "b": 0.5}  # P1 calls Q too often
    assert exploitability(KUHN, profile) > 1e-3


def test_kuhn_br_vs_always_bet_matches_hand_calculation():
    """Opponent always bets/calls. P0's BR, by card (each dealt w.p. 1/3):

    J: check then fold = -1 (betting gets called and loses 2).      -> -1
    Q: vs J (+2) or K (-2) whether bet or check-call: average 0;
       check-fold is -1, so play on.                                  ->  0
    K: bet, gets called, +2.                                          -> +2
    BR_0 = (-1 + 0 + 2) / 3 = 1/3. P1's BR facing a bet is the same
    calculation (fold J -1, call Q 0, call K +2): BR_1 = 1/3.
    Both always-bet: every hand is bet-call showdown, EV 0.
    """
    profile = always_bet()
    br0 = best_response(KUHN, profile, 0)
    assert br0.value == pytest.approx(1 / 3)
    assert br0.policy["Jpb"] == "p" and br0.policy["Qpb"] == "b" and br0.policy["Kpb"] == "b"
    # K opening is a tie (bet: +2; check, P1 bets, call: +2) -> first action "p".
    assert br0.policy["K"] == "p"
    assert best_response_value(KUHN, profile, 1) == pytest.approx(1 / 3)
    assert expected_value(KUHN, profile) == pytest.approx(0.0)
    assert exploitability(KUHN, profile) == pytest.approx(1 / 3)
    assert nash_conv(KUHN, profile) == pytest.approx(2 / 3)


def test_br_is_not_a_per_history_max():
    """Same opponent: a card-peeking P0 folds Q vs K (-1) and calls vs J (+2),
    so Q is worth +0.5 instead of 0 and the total is (-1 + 0.5 + 2)/3 = 1/2."""
    profile = always_bet()
    assert cheating_br_value(KUHN, profile, 0) == pytest.approx(1 / 2)
    assert best_response_value(KUHN, profile, 0) == pytest.approx(1 / 3)
    # Against uniform play: peeking gains nothing for Kuhn P0 (its decisions do
    # not hinge on P1's card) but helps everywhere else.
    assert cheating_br_value(KUHN, {}, 0) == pytest.approx(best_response_value(KUHN, {}, 0))
    assert cheating_br_value(KUHN, {}, 1) == pytest.approx(1 / 2)
    assert best_response_value(KUHN, {}, 1) == pytest.approx(5 / 12)
    assert cheating_br_value(LEDUC, {}, 0) == pytest.approx(2.375)
    assert cheating_br_value(LEDUC, {}, 1) == pytest.approx(2.925)


def test_kuhn_uniform_exploitability_recorded():
    """Uniform random: BR_0 = 1/2, BR_1 = 5/12, exploitability 11/24 = 0.4583
    (OpenSpiel's NashConv for uniform Kuhn is 0.9167, i.e. twice this)."""
    assert best_response_value(KUHN, {}, 0) == pytest.approx(1 / 2)
    assert best_response_value(KUHN, {}, 1) == pytest.approx(5 / 12)
    assert exploitability(KUHN, {}) == pytest.approx(11 / 24)


@pytest.mark.parametrize("game", [KUHN, LEDUC], ids=["kuhn", "leduc"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_br_policy_achieves_br_value_and_bounds_ev(game, seed):
    """Playing the returned pure BR policy against the profile earns exactly the
    BR value, and that is at least what the profile itself earns."""
    profile = random_profile(game, seed)
    ev0 = expected_value(game, profile)
    for br in (0, 1):
        res = best_response(game, profile, br)
        mixed = dict(profile)
        for key, a in res.policy.items():
            mixed[key] = {a: 1.0}
        own_ev = expected_value(game, mixed) * (1 if br == 0 else -1)
        assert own_ev == pytest.approx(res.value, abs=1e-12)
        assert res.value >= (ev0 if br == 0 else -ev0) - 1e-12
        assert set(res.policy) == {k for k, (p, _) in infosets(game).items() if p == br}
    assert exploitability(game, profile) > 0


# ------------------------------------------------------------------------ Leduc


def test_leduc_terminal_payoffs():
    u = LEDUC.utility
    assert u(("J", "K", "r", "f")) == 1                       # P1 folds to a bet
    assert u(("J", "K", "c", "r", "f")) == -1                 # P0 folds
    assert u(("K", "Q", "r", "c", "K", "r", "f")) == 3        # 1 ante + 2
    assert u(("J", "K", "c", "c", "J", "r", "c")) == 5        # pair of J beats K
    assert u(("Q", "K", "r", "r", "c", "J", "c", "c")) == -5  # K high beats Q
    assert u(("Q", "Q", "r", "c", "K", "c", "c")) == 0        # split
    assert u(("K", "J", "r", "r", "c", "Q", "r", "r", "c")) == 13  # max pot
    assert u(("K", "K", "c", "c", "J", "c", "c")) == 0        # split, both K


def test_leduc_rules_and_cap():
    assert LEDUC.actions(("K", "J")) == ("c", "r")
    assert LEDUC.actions(("K", "J", "r")) == ("f", "c", "r")
    assert LEDUC.actions(("K", "J", "r", "r")) == ("f", "c")      # cap: 2 bets
    assert LEDUC.actions(("K", "J", "c", "r", "r")) == ("f", "c")
    assert LEDUC.player(("K", "J", "c", "c")) == CHANCE           # board deal
    assert LEDUC.player(("K", "J", "c", "c", "Q")) == 0           # P0 first again
    assert LEDUC.player(("K", "J", "c", "c", "Q", "c")) == 1
    with pytest.raises(ValueError):
        _parse(("K", "J", "r", "r", "r"))
    with pytest.raises(ValueError):
        _parse(("K", "J", "f"))


def test_leduc_chance_probabilities():
    assert dict(LEDUC.chance_outcomes(())) == pytest.approx({"J": 1 / 3, "Q": 1 / 3, "K": 1 / 3})
    assert dict(LEDUC.chance_outcomes(("J",))) == pytest.approx({"J": 1 / 5, "Q": 2 / 5, "K": 2 / 5})
    assert dict(LEDUC.chance_outcomes(("J", "J", "c", "c"))) == pytest.approx({"Q": 1 / 2, "K": 1 / 2})
    assert dict(LEDUC.chance_outcomes(("J", "Q", "c", "c"))) == pytest.approx(
        {"J": 1 / 4, "Q": 1 / 4, "K": 1 / 2})


def test_leduc_utilities_zero_sum_and_balanced():
    """Payoff to P0 is minus payoff to P1 by construction; check the chip
    accounting behind it: a fold loses exactly the folder's chips, a showdown
    happens with equal contributions and moves exactly one stack."""
    for h in iter_histories(LEDUC):
        if not LEDUC.is_terminal(h):
            continue
        s, u = _parse(h), LEDUC.utility(h)
        if s.folder is None:
            assert s.put_in[0] == s.put_in[1]
            assert u in (0.0, s.put_in[0], -s.put_in[0])
        else:
            assert u == (-s.put_in[0] if s.folder == 0 else s.put_in[1])
        assert 1 <= abs(u) <= 13 or u == 0


def test_leduc_infoset_count():
    """Rank-only: 6 x 3 + 5 x 6 x 3 x 3 = 288. Lifting back to suited cards:
    each rank-only round-1 set is 2 suited sets (own suit); a round-2 set is
    2 x 1 if the board pairs the own card (one copy left) else 2 x 2, giving
    18 x 2 + 30 x (3 x 2 + 6 x 4) = 936, the figure in Waugh et al. 2015."""
    sets = infosets(LEDUC)
    assert len(sets) == 288
    assert sum(1 for p, _ in sets.values() if p == 0) == 144
    round1 = [k for k in sets if "/" not in k]
    assert len(round1) == 18
    suited = len(round1) * 2
    for k in sets:
        if "/" in k:
            suited += 2 if k[0] == k[1] else 4
    assert suited == 936


def test_leduc_infoset_key_hides_opponent_card():
    """Spy test: swapping the opponent's private card (to any rank still
    available) never changes the acting player's key or actions."""
    checked = 0
    for h in iter_histories(LEDUC):
        if LEDUC.is_terminal(h) or LEDUC.player(h) == CHANCE:
            continue
        me = LEDUC.player(h)
        opp = 1 - me
        for r in "JQK":
            alt = h[:opp] + (r,) + h[opp + 1:]  # opponent's card is h[opp]
            if any(alt.count(c) > 2 for c in "JQK"):
                continue
            assert LEDUC.infoset_key(alt) == LEDUC.infoset_key(h)
            assert LEDUC.actions(alt) == LEDUC.actions(h)
            checked += 1
        key = LEDUC.infoset_key(h)
        assert key[0] == h[me]  # own card, then board (if any), then betting
    assert checked > 1000


def test_leduc_uniform_exploitability_recorded_and_fast():
    """Uniform random: BR_0 = 2.0875, BR_1 = 2.65972, exploitability 2.37361
    chips (OpenSpiel's NashConv for uniform Leduc is 4.7472, twice this)."""
    t = time.perf_counter()
    assert best_response_value(LEDUC, {}, 0) == pytest.approx(2.0875, abs=1e-9)
    assert best_response_value(LEDUC, {}, 1) == pytest.approx(2.6597222222, abs=1e-9)
    assert exploitability(LEDUC, {}) == pytest.approx(2.3736111111, abs=1e-9)
    assert expected_value(LEDUC, {}) == pytest.approx(-0.078125, abs=1e-12)
    # Loose bound (~0.05 s normally): this laptop throttles 6-9x under sustained load.
    assert time.perf_counter() - t < 10.0


def test_leduc_always_check_is_zero_by_symmetry():
    check = {k: {"c": 1.0} for k in infosets(LEDUC)}
    assert expected_value(LEDUC, check) == pytest.approx(0.0, abs=1e-12)
