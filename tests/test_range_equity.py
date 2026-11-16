"""Tests for equity against a range and for the preflop tables.

Monte Carlo checks compare two estimates with a tolerance of 4 combined
standard errors, sqrt(se1^2 + se2^2): a correct implementation fails such
a check with probability about 6e-5, and the seeds are fixed anyway.
"""

from __future__ import annotations

import math
import random

import pytest

from nutsfinder.engine.cards import FULL_DECK, Card, parse_cards
from nutsfinder.engine.equity import (
    RANDOM_RANGE,
    equity_vs_nested_ranges,
    equity_vs_range,
    monte_carlo_equity,
)
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.engine.preflop import (
    ALL_CLASSES,
    PREFLOP_EQUITY,
    PREFLOP_SE,
    RANGE_GRID,
    RANKING,
    class_combos,
    equity_vs_top,
    hand_class,
    representative,
    snap_to_grid,
    top_classes,
    top_range,
)


def cards(s: str) -> list[Card]:
    return parse_cards(s)


def combo(s: str) -> tuple[Card, Card]:
    a, b = parse_cards(s)
    return (a, b)


def close(a, b, k: float = 4.0) -> bool:
    return abs(a.equity - b.equity) <= k * math.hypot(a.std_error, b.std_error)


# --- equity_vs_range ---------------------------------------------------------


@pytest.mark.parametrize("board", ["", "Kh7h2s", "Kh7h2s3d"])
def test_single_hand_range_agrees_with_monte_carlo_equity(board):
    hero, villain = cards("AsKs"), cards("9h8h")
    a = equity_vs_range(hero, [combo("9h8h")], cards(board), samples=20_000, seed=1)
    b = monte_carlo_equity(hero, villain, cards(board), samples=20_000, seed=2)
    assert close(a, b)


def test_single_hand_range_is_exact_on_the_river():
    hero, board = cards("AsKs"), cards("Kh7h2s3d9c")
    a = equity_vs_range(hero, [combo("9h8h")], board)
    b = monte_carlo_equity(hero, cards("9h8h"), board, samples=10)
    assert a.equity == b.equity == 1.0  # a pair of kings beats a pair of nines
    assert a.std_error == 0.0 and a.samples == 1


def test_card_removal_drops_combos_that_use_known_cards():
    # Holding the As, the villain cannot have As Kd, so this range is just QcQd.
    hero = cards("AsAh")
    with_dead = equity_vs_range(hero, [combo("AsKd"), combo("QcQd")], samples=2_000, seed=3)
    without = equity_vs_range(hero, [combo("QcQd")], samples=2_000, seed=3)
    assert with_dead == without


def test_card_removal_on_the_river_counts_only_live_combos():
    # Against "any hand" on the river, the villain has C(45, 2) = 990 possible hands.
    hero, board = cards("AsKs"), cards("Kh7h2s3d9c")
    r = equity_vs_range(hero, RANDOM_RANGE, board)
    assert r.samples == math.comb(45, 2) == 990
    # Brute force: count by hand.
    known = set(hero) | set(board)
    rest = [c for c in FULL_DECK if c not in known]
    mine = evaluate(hero + board)
    score = 0.0
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            theirs = evaluate([rest[i], rest[j], *board])
            score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    assert r.equity == pytest.approx(score / 990)


def test_random_range_matches_the_preflop_table():
    hero = cards("AsKs")
    r = equity_vs_range(hero, RANDOM_RANGE, samples=20_000, seed=4)
    assert abs(r.equity - PREFLOP_EQUITY["AKs"]) <= 4 * math.hypot(r.std_error, PREFLOP_SE["AKs"])


def test_same_seed_same_result_and_rng_argument():
    hero, board = cards("QdQc"), cards("Ts9s2h")
    a = equity_vs_range(hero, RANDOM_RANGE, board, samples=500, seed=9)
    b = equity_vs_range(hero, RANDOM_RANGE, board, samples=500, seed=9)
    c = equity_vs_range(hero, RANDOM_RANGE, board, samples=500, rng=random.Random(9))
    assert a == b == c


def test_a_past_deadline_stops_sampling_early():
    r = equity_vs_range(cards("AsKs"), RANDOM_RANGE, samples=10_000, seed=1, deadline=0.0)
    assert 1 <= r.samples < 10_000


@pytest.mark.parametrize(
    ("hero", "villain_range", "board"),
    [
        ("AsKs", [], ""),  # empty range
        ("AsKs", [("As", "Kd")], ""),  # every combo dead
        ("AsKs", [("Qc", "Qc")], ""),  # a combo with the same card twice
        ("As", [("Qc", "Qd")], ""),  # one hole card
        ("AsKs", [("Qc", "Qd")], "AsTd9c"),  # hero card on the board
    ],
)
def test_bad_arguments_raise(hero, villain_range, board):
    rng = [(Card.from_str(a), Card.from_str(b)) for a, b in villain_range]
    with pytest.raises(ValueError):
        equity_vs_range(cards(hero), rng, cards(board), samples=10)


# --- nested ranges -------------------------------------------------------------


def test_nested_ranges_agree_with_separate_runs():
    hero, board = cards("JhTh"), cards("9h8c2d")
    wide, narrow = top_range(1.0), top_range(0.3)
    both = equity_vs_nested_ranges(hero, [wide, narrow], board, samples=20_000, seed=5)
    alone = equity_vs_range(hero, narrow, board, samples=20_000, seed=6)
    assert both[0].samples == 20_000
    assert 0.2 * 20_000 < both[1].samples < 0.4 * 20_000  # about 30% of the draws
    assert close(both[1], alone)


def test_nested_ranges_top_up_small_subranges():
    hero = cards("7c6c")
    results = equity_vs_nested_ranges(hero, [top_range(1.0), top_range(0.01)], samples=200, min_samples=150, seed=7)
    assert results[1].samples >= 150


def exact_turn_equity(hero: list[Card], board: list[Card], combos) -> float:
    """Exact equity on the turn: every live villain combo equally likely, every river card too."""
    known = set(hero) | set(board)
    live = {tuple(sorted(c)) for c in combos if c[0] not in known and c[1] not in known}
    total = 0.0
    for combo in live:
        rivers = [c for c in FULL_DECK if c not in known and c not in combo]
        for river in rivers:
            mine, theirs = evaluate(hero + board + [river]), evaluate([*combo, *board, river])
            total += (1.0 if mine > theirs else 0.5 if mine == theirs else 0.0) / len(rivers)
    return total / len(live)


def test_nested_ranges_top_up_is_unbiased():
    # The top 5% is about 5% of the draws from "any hand", so out of 40 draws it gets
    # about 2 and is topped up to 30 with direct draws. Both kinds of draw are uniform
    # over the sub-range, so the pooled estimate must be unbiased. Check the mean over
    # many seeds against exact enumeration, within 4 standard errors of that mean.
    # (Topping up from the wide range would give about 0.30 instead of 0.20: 20+ SEs off.)
    hero, board = cards("7c6c"), cards("8c9dKh2s")
    narrow = top_range(0.05)
    exact = exact_turn_equity(hero, board, narrow)
    estimates = [
        equity_vs_nested_ranges(hero, [RANDOM_RANGE, narrow], board, samples=40, min_samples=30, seed=s)[1]
        for s in range(300)
    ]
    assert all(r.samples >= 30 for r in estimates)
    values = [r.equity for r in estimates]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    assert abs(mean - exact) < 4 * sd / math.sqrt(len(values))


def test_equity_vs_range_uses_exactly_the_samples_asked_for():
    # equity_vs_range has a single range, so nothing may be topped up behind our back.
    r = equity_vs_range(cards("AsKs"), RANDOM_RANGE, cards("Kh7h2s"), samples=5, seed=1)
    assert r.samples == 5


def test_nested_ranges_require_subsets():
    with pytest.raises(ValueError):
        equity_vs_nested_ranges(cards("AsKs"), [top_range(0.1), top_range(0.5)], samples=10)


def test_nested_ranges_are_exact_on_the_river():
    hero, board = cards("AsKs"), cards("Kh7h2s3d9c")
    results = equity_vs_nested_ranges(hero, [top_range(1.0), top_range(0.2)], board)
    assert all(r.std_error == 0.0 for r in results)
    assert results[1] == equity_vs_range(hero, top_range(0.2), board)


# --- preflop classes and tables --------------------------------------------------


def test_169_classes_and_1326_combos():
    assert len(ALL_CLASSES) == len(set(ALL_CLASSES)) == 169
    assert sum(len(class_combos(c)) for c in ALL_CLASSES) == 1326
    assert (len(class_combos("AA")), len(class_combos("AKs")), len(class_combos("AKo"))) == (6, 4, 12)
    for cls in ("AA", "T9s", "72o"):
        assert all(hand_class(c) == cls for c in class_combos(cls))
        assert hand_class(representative(cls)) == cls


@pytest.mark.parametrize(
    ("hand", "expected"), [("AsAh", "AA"), ("KdAd", "AKs"), ("2c7h", "72o"), ("TsJs", "JTs")]
)
def test_hand_class(hand, expected):
    assert hand_class(cards(hand)) == expected


@pytest.mark.parametrize(
    ("cls", "known"),
    # Published exact all-in equities against a random hand.
    [("AA", 0.8520), ("KK", 0.8240), ("AKs", 0.6704), ("AKo", 0.6540), ("72o", 0.3458)],
)
def test_preflop_table_matches_known_values(cls, known):
    assert PREFLOP_SE[cls] < 0.002  # under 0.2 percentage points
    assert abs(PREFLOP_EQUITY[cls] - known) < 3 * PREFLOP_SE[cls] + 0.0005


def test_ranking_and_top_ranges():
    assert RANKING[0] == "AA" and RANKING[-1] == "32o"
    assert top_classes(0.01) == ("AA", "KK", "QQ")
    assert len(top_classes(1.0)) == 169
    assert set(top_range(0.15)) < set(top_range(0.3)) < set(top_range(1.0))
    assert len(top_range(0.5)) >= 0.5 * 1326


def test_snap_to_grid_rounds_down():
    assert snap_to_grid(0.66) == 0.60
    assert snap_to_grid(0.5) == 0.5
    assert snap_to_grid(0.001) == 0.01
    assert snap_to_grid(2.0) == 1.0


def test_range_table_lookup():
    assert equity_vs_top("AA", 1.0) == (PREFLOP_EQUITY["AA"], PREFLOP_SE["AA"])
    # Narrower ranges are stronger, so AA's and 72o's equity drop.
    assert equity_vs_top("AA", 0.05)[0] < equity_vs_top("AA", 0.5)[0] < PREFLOP_EQUITY["AA"]
    assert equity_vs_top("72o", 0.3)[0] < PREFLOP_EQUITY["72o"]
    with pytest.raises(ValueError):
        equity_vs_top("AA", 0.33)  # not on the grid


@pytest.mark.parametrize(("cls", "width"), [("AKo", 0.15), ("55", 0.5), ("T9s", 0.03)])
def test_range_table_agrees_with_a_fresh_simulation(cls, width):
    eq, se = equity_vs_top(cls, width)
    hero = list(class_combos(cls)[-1])  # a different combo than the one the table used
    r = equity_vs_range(hero, top_range(width), samples=5_000, seed=11)
    assert abs(r.equity - eq) <= 4 * math.hypot(r.std_error, se)
    assert width in RANGE_GRID


def test_river_samples_villain_hands_when_the_range_is_bigger_than_samples():
    hero, board = cards("QsJs"), cards("Kh7h2s3d9c")
    exact = equity_vs_range(hero, RANDOM_RANGE, board)  # 10,000 >= 990: exact
    sampled = equity_vs_range(hero, RANDOM_RANGE, board, samples=400, seed=12)
    assert exact.std_error == 0.0 and sampled.samples == 400 and sampled.std_error > 0
    assert close(exact, sampled)
