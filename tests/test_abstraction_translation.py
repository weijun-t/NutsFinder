"""Action translation: the pseudo-harmonic mapping and real <-> abstract bet sizes.

Part 1: the mapping itself (f(A)=1, f(B)=0, monotone, the 0.43 example from
docs/phase6_research.md, its "price" identity). Part 2: translating real
bets, with the edge cases from the module docstring and a Monte Carlo check
that the randomised choice follows f. Part 3: fuzz over random real states
built with ``new_hand`` / ``apply_action``: every helper output is legal and
the history replay rebuilds exactly the view the bettor had.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from nutsfinder.abstraction.translation import (
    ALL_IN,
    bet_fraction,
    fraction_for_bb,
    pseudo_harmonic,
    real_action_for_fraction,
    size_points,
    to_for_bb,
    to_for_fraction,
    translate_bet,
    translate_last_action,
    view_before_action,
)
from nutsfinder.bots.common import effective_max_to
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, PublicState, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.cards import Deck, parse_cards

FLOP = parse_cards("Kh7d2c")
POSTFLOP = [1.0, ALL_IN]
PREFLOP_SIZES_BB = 3.0  # open to 3bb; plus pot and all-in


def view(state: HandState) -> PublicState:
    assert state.to_act is not None
    return public_view(state, state.to_act)


def flop_spot(pot_each: int = 500, stacks: int = 10_000) -> HandState:
    """Both players put ``pot_each`` in preflop (button raises, BB calls); BB to act on the flop."""
    s = new_hand(button=0, stacks=(stacks, stacks))
    s = apply_action(s, 0, Raise(pot_each))
    s = apply_action(s, 1, Call())
    return deal_board(s, FLOP)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- Part 1: the mapping ------------------------------------------------------


def test_worked_example_043():
    assert pseudo_harmonic(0.75, 0.5, 1.0) == pytest.approx(0.375 / 0.875)
    assert round(pseudo_harmonic(0.75, 0.5, 1.0), 2) == 0.43


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (0.5, 1.0), (1.0, 49.5), (0.25, 0.3)])
def test_endpoints_and_monotone(a, b):
    assert pseudo_harmonic(a, a, b) == 1.0
    assert pseudo_harmonic(b, a, b) == 0.0
    xs = np.linspace(a, b, 201)
    fs = [pseudo_harmonic(float(x), a, b) for x in xs]
    assert all(f1 > f2 for f1, f2 in zip(fs, fs[1:]))
    assert all(0.0 <= f <= 1.0 for f in fs)


@pytest.mark.parametrize(("a", "b", "x"), [(0.5, 1.0, 0.75), (0.0, 1.0, 0.3), (1.0, 20.0, 4.0)])
def test_matches_calling_frequency(a, b, x):
    """The derivation: mixing the A- and B-responses calls with the right frequency 1/(1+x)."""
    f = pseudo_harmonic(x, a, b)
    assert f / (1 + a) + (1 - f) / (1 + b) == pytest.approx(1 / (1 + x))


def test_symmetry_in_price_space():
    """f is linear in 1/(1+x): at the midpoint of the prices it is exactly 1/2."""
    a, b = 0.5, 2.0
    u = (1 / (1 + a) + 1 / (1 + b)) / 2
    x_mid = 1 / u - 1
    assert pseudo_harmonic(x_mid, a, b) == pytest.approx(0.5)
    # ... and that point is below the arithmetic midpoint (bets are read as "bigger").
    assert x_mid < (a + b) / 2
    assert pseudo_harmonic((a + b) / 2, a, b) < 0.5


def test_limits():
    # B -> infinity with A = 0: f -> 1/(1+x), the plain calling frequency.
    assert pseudo_harmonic(1.0, 0.0, 1e9) == pytest.approx(0.5, rel=1e-6)
    # Very close sizes: f is (almost) linear between them.
    assert pseudo_harmonic(1.0005, 1.0, 1.001) == pytest.approx(0.5, abs=1e-3)
    # Passive (A = 0) form.
    assert pseudo_harmonic(0.5, 0.0, 1.0) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("x", "a", "b"), [(0.4, 0.5, 1.0), (1.1, 0.5, 1.0), (0.5, 1.0, 0.5), (0.5, -1, 1), (1, 0, math.inf)]
)
def test_bad_arguments(x, a, b):
    with pytest.raises(ValueError):
        pseudo_harmonic(x, a, b)


# --- Part 2: translating real bets --------------------------------------------


def test_fraction_and_preflop_bb_conversion():
    s = view(new_hand(button=0))  # blinds 50/100, button to act: P = 150, c = 50
    assert bet_fraction(s, 300) == pytest.approx(1.0)
    assert fraction_for_bb(s, 3) == pytest.approx(1.0)  # open to 3bb == pot here
    assert bet_fraction(s, 200) == pytest.approx(0.5)  # min-raise
    assert to_for_bb(s, 3) == 300 == to_for_fraction(s, 1.0)
    limp = view(apply_action(new_hand(button=0), 0, Call()))  # BB after a limp: P = 200, c = 0
    assert fraction_for_bb(limp, 3) == pytest.approx(1.0)
    flop = view(flop_spot())
    assert bet_fraction(flop, 750) == pytest.approx(0.75)


def test_open_3bb_and_pot_are_merged():
    s = view(new_hand(button=0))
    sizes = [fraction_for_bb(s, PREFLOP_SIZES_BB), 1.0, ALL_IN]
    pts = size_points(s, sizes)
    assert [p.to for p in pts] == [300, 10_000]
    assert pts[0].indices == (0, 1) and pts[1].indices == (2,)
    t = translate_bet(s, 300, sizes, rng())
    assert t.index == 1 and t.equivalent == (0, 1) and t.p_lower == 1.0


def test_exact_size_and_all_in_are_deterministic():
    s = view(flop_spot())  # pot 1,000, 9,500 behind
    g = rng()
    state_before = g.bit_generator.state
    assert translate_bet(s, 1000, POSTFLOP, g).index == 0
    assert translate_bet(s, 9500, POSTFLOP, g).index == 1
    assert g.bit_generator.state == state_before  # no random draw when not mixing


def test_between_sizes_and_above_largest():
    s = view(flop_spot())
    t = translate_bet(s, 3000, POSTFLOP, rng())  # x = 3, between pot (1) and all-in (9.5)
    assert t.lower == pytest.approx(1.0) and t.upper == pytest.approx(9.5)
    assert t.p_lower == pytest.approx(pseudo_harmonic(3.0, 1.0, 9.5))
    # Without an all-in size, anything above pot is pot.
    t = translate_bet(s, 3000, [1.0], rng())
    assert t.index == 0 and t.p_lower == 1.0


def test_below_smallest_rules():
    s = view(flop_spot())
    t = translate_bet(s, 500, POSTFLOP, rng(), below="smallest")
    assert t.index == 0
    hits = [translate_bet(s, 500, POSTFLOP, rng(i)).passive for i in range(4000)]
    p = pseudo_harmonic(0.5, 0.0, 1.0)  # 1/3
    se = math.sqrt(p * (1 - p) / len(hits))
    assert abs(np.mean(hits) - p) < 4 * se
    with pytest.raises(ValueError):
        translate_bet(s, 500, POSTFLOP, rng(), below="nearest")  # type: ignore[arg-type]


def test_min_raise_maps_to_smallest_when_it_clamps_to_min():
    # Postflop sizes [0.05 pot, pot]: 0.05 * 1,000 = 50 clamps up to the min bet 100.
    s = view(flop_spot())
    sizes = [0.05, 1.0, ALL_IN]
    t = translate_bet(s, s.legal.min_to, sizes, rng())
    assert t.index == 0 and t.p_lower == 1.0


def test_pot_clamps_to_all_in_and_dedupes():
    s = view(flop_spot(pot_each=500, stacks=1_200))  # pot 1,000, only 700 behind
    pts = size_points(s, POSTFLOP)
    assert len(pts) == 1 and pts[0].to == 700 and pts[0].indices == (0, 1)
    t = translate_bet(s, 700, POSTFLOP, rng())
    assert t.index == 1 and t.equivalent == (0, 1)
    # A bet below that single point uses the passive rule.
    assert translate_bet(s, 300, POSTFLOP, rng()).lower == 0.0


def test_all_in_for_less_reads_as_all_in():
    # Button has 500: raise to 200, BB 3-bets to 400 (next min raise 600), button can only shove 500.
    s = new_hand(button=0, stacks=(500, 10_000))
    s = apply_action(s, 0, Raise(200))
    s = apply_action(s, 1, Raise(400))
    v = view(s)
    assert v.legal.min_to == v.legal.max_to == 500
    t = translate_bet(v, 500, [1.0, ALL_IN], rng())
    assert t.index == 1 and t.equivalent == (0, 1)


def test_overbet_beyond_effective_stack_is_all_in():
    # Deep button (10,000) vs short BB (1,000): limp, check; flop BB checks, button bets 5,000.
    s = new_hand(button=0, stacks=(10_000, 1_000))
    s = apply_action(apply_action(s, 0, Call()), 1, Check())
    s = apply_action(deal_board(s, FLOP), 1, Check())
    v = view(s)
    assert effective_max_to(v) == 900
    assert bet_fraction(v, 5000) == bet_fraction(v, 900) == pytest.approx(900 / 200)
    assert translate_bet(v, 5000, POSTFLOP, rng()).index == 1


def test_illegal_inputs():
    s = view(flop_spot())
    with pytest.raises(ValueError):
        translate_bet(s, 50, POSTFLOP, rng())  # below the min bet
    with pytest.raises(ValueError):
        translate_bet(s, 1000, [], rng())
    st = new_hand(button=0, stacks=(10_000, 300))
    st = apply_action(st, 0, Raise(10_000))  # BB faces an all-in: no raise possible
    with pytest.raises(ValueError):
        bet_fraction(view(st), 500)
    assert real_action_for_fraction(view(st), 1.0) == Call()


@pytest.mark.parametrize("x", [0.3, 0.75, 2.0, 6.0])
def test_monte_carlo_frequency_matches_f(x):
    s = view(flop_spot())  # pot 1,000
    sizes = [0.5, 1.0, ALL_IN]
    to = round(1000 * x)
    g = rng(123)
    results = [translate_bet(s, to, sizes, g) for _ in range(5000)]
    first = results[0]
    p = first.p_lower
    if first.lower == 0.0:
        p_expected = pseudo_harmonic(to / 1000, 0.0, 0.5)
    else:
        p_expected = pseudo_harmonic(to / 1000, first.lower, first.upper)
    assert p == pytest.approx(p_expected)
    # Index of the lower neighbour (None = passive), then its observed frequency.
    lower_index = None if first.lower == 0.0 else next(
        pt.indices[-1] for pt in size_points(s, sizes) if pt.fraction == first.lower
    )
    freq = np.mean([r.index == lower_index for r in results])
    se = math.sqrt(p * (1 - p) / len(results))
    assert abs(freq - p) < 4 * se


def test_seeded_generator_is_deterministic():
    s = view(flop_spot())
    a = [translate_bet(s, 2500, POSTFLOP, rng(7)).index for _ in range(3)]
    g1, g2 = rng(99), rng(99)
    seq1 = [translate_bet(s, 750, [0.5, 1.0, ALL_IN], g1).index for _ in range(200)]
    seq2 = [translate_bet(s, 750, [0.5, 1.0, ALL_IN], g2).index for _ in range(200)]
    assert seq1 == seq2
    assert len(set(a)) == 1
    assert len(set(seq1)) == 2  # it really mixes


def test_own_bets_use_real_pot():
    """Our "pot" is the real pot, whatever the abstract pot was after translation."""
    s = view(flop_spot())  # real pot 1,000
    assert to_for_fraction(s, 1.0) == 1000
    assert real_action_for_fraction(s, 1.0) == Bet(1000)
    assert real_action_for_fraction(s, ALL_IN) == Bet(9500)
    facing = view(apply_action(flop_spot(), 1, Bet(400)))  # P = 1,400, c = 400
    assert real_action_for_fraction(facing, 1.0) == Raise(400 + 1800)


def test_translate_last_action():
    s = apply_action(flop_spot(), 1, Bet(750))
    v = public_view(s, 0)
    t = translate_last_action(v, [0.5, 1.0, ALL_IN], rng(1))
    assert t.x == pytest.approx(0.75)
    assert t.p_lower == pytest.approx(pseudo_harmonic(0.75, 0.5, 1.0))
    with pytest.raises(ValueError):
        translate_last_action(public_view(apply_action(s, 0, Call()), 0), POSTFLOP, rng())


# --- Part 3: fuzz over random real states -------------------------------------


def _random_action(r: random.Random, legal) -> Action:
    options: list[Action] = [Check() if legal.can_check else Call()]
    if r.random() < 0.1:
        options.append(Fold())
    if legal.can_bet or legal.can_raise:
        lo, hi = legal.min_to, legal.max_to
        to = r.choice([lo, hi, r.randint(lo, hi), min(hi, lo + r.randint(0, 3 * lo))])
        options += [Bet(to) if legal.can_bet else Raise(to)] * 2
    return r.choice(options)


SIZE_SETS = [
    [1.0, ALL_IN],
    [0.5, 1.0, ALL_IN],
    [1.0],
    [0.33, 0.75, 1.5, ALL_IN],
]


def test_fuzz_random_states():
    r = random.Random(20261116)
    g = np.random.default_rng(5)
    checked_bets = checked_sizes = 0
    for hand in range(400):
        stacks = (
            r.choice([150, 400, 1_000, 10_000, r.randint(60, 20_000)]),
            r.choice([300, 10_000, r.randint(60, 20_000)]),
        )
        state = new_hand(button=hand % 2, stacks=stacks)
        deck = Deck(seed=hand)
        while not state.is_hand_over:
            if state.cards_needed:
                state = deal_board(state, deck.deal(state.cards_needed))
                continue
            v = view(state)
            legal = v.legal
            if legal.can_bet or legal.can_raise:
                # Our own bets: every abstract size is a legal real action.
                for f in (0.0, 0.33, 0.5, 1.0, 2.5, ALL_IN, r.uniform(0, 5)):
                    act = real_action_for_fraction(v, f)
                    assert legal.is_legal(act), (f, act, legal)
                    apply_action(state, v.to_act, act)  # the engine agrees
                    to = to_for_fraction(v, f)
                    assert legal.min_to <= to <= min(legal.max_to, effective_max_to(v))
                    checked_sizes += 1
                assert legal.is_legal(Bet(to_for_bb(v, 3)) if legal.can_bet else Raise(to_for_bb(v, 3)))
            else:
                assert real_action_for_fraction(v, 1.0) in (Call(), Check())
            action = _random_action(r, legal)
            state = apply_action(state, v.to_act, action)
            if isinstance(action, (Bet, Raise)):
                # Replaying the history rebuilds exactly the bettor's view.
                after = public_view(state, 1 - v.to_act)
                assert view_before_action(after) == v
                sizes = r.choice(SIZE_SETS)
                t = translate_last_action(after, sizes, g)
                pts = size_points(v, sizes)
                assert t.index is None or t.index in t.equivalent
                assert t.index is None or any(t.equivalent == p.indices for p in pts)
                assert 0.0 <= t.p_lower <= 1.0
                assert t.lower <= t.upper
                x = bet_fraction(v, action.amount if isinstance(action, Bet) else action.to)
                assert x > 0 and t.x == x
                checked_bets += 1
    assert checked_bets > 500 and checked_sizes > 3000
