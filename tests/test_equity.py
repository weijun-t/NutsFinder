"""Tests for the Monte Carlo equity calculator (Phase 1).

The whole module is skipped while ``monte_carlo_equity`` is still a stub
(see conftest.py).
"""

import math

import pytest

from nutsfinder.engine.cards import parse_cards
from nutsfinder.engine.equity import EquityResult, monte_carlo_equity

pytestmark = pytest.mark.usefixtures("equity_implemented")

HERO = parse_cards("As Ks")
VILLAIN = parse_cards("9h 8h")
FLOP = parse_cards("Kh 7h 2s")

# Exact values, found by enumerating every possible run-out (tie = half a win).
# Preflop: 1,059,724 wins + 7,790 ties out of C(48, 5) = 1,712,304 boards.
PREFLOP_EXPECTED = (1_059_724 + 7_790 / 2) / 1_712_304  # 0.62116...
# Flop: 600 wins, 0 ties out of C(45, 2) = 990 turn+river run-outs.
FLOP_EXPECTED = 600 / 990  # 0.60606...

SAMPLES = 10_000
SEED = 2024


def assert_consistent(r: EquityResult, samples: int) -> None:
    assert r.samples == samples
    assert r.wins + r.ties + r.losses == samples
    assert min(r.wins, r.ties, r.losses) >= 0
    assert r.equity == pytest.approx((r.wins + r.ties / 2) / samples)
    assert 0.0 <= r.equity <= 1.0
    assert r.std_error >= 0.0


def assert_std_error_sane(r: EquityResult) -> None:
    """std_error should be close to the binomial sqrt(p(1-p)/n), within 20%.

    Ties count as half a win, which makes the true per-sample variance a bit
    smaller than p(1-p); for these hands ties are rare, so 20% is ample.
    """
    binomial = math.sqrt(r.equity * (1 - r.equity) / r.samples)
    assert r.std_error == pytest.approx(binomial, rel=0.20)


@pytest.mark.parametrize(
    ("board", "expected"),
    [((), PREFLOP_EXPECTED), (FLOP, FLOP_EXPECTED)],
    ids=["preflop", "flop Kh7h2s"],
)
def test_aks_vs_98h_matches_known_equity(board, expected):
    r = monte_carlo_equity(HERO, VILLAIN, board, samples=SAMPLES, seed=SEED)
    assert_consistent(r, SAMPLES)
    assert_std_error_sane(r)
    assert abs(r.equity - expected) < 4 * r.std_error, (
        f"equity {r.equity:.4f} +/- {r.std_error:.4f} is more than 4 standard "
        f"errors from {expected}"
    )


def test_std_error_shrinks_like_one_over_sqrt_n():
    small = monte_carlo_equity(HERO, VILLAIN, samples=1_000, seed=1)
    large = monte_carlo_equity(HERO, VILLAIN, samples=16_000, seed=1)
    # 16x the samples => about 1/4 of the standard error.
    assert large.std_error / small.std_error == pytest.approx(0.25, rel=0.25)


def test_same_seed_same_result():
    a = monte_carlo_equity(HERO, VILLAIN, FLOP, samples=2_000, seed=7)
    b = monte_carlo_equity(HERO, VILLAIN, FLOP, samples=2_000, seed=7)
    assert a == b


def test_different_seeds_usually_differ():
    results = {monte_carlo_equity(HERO, VILLAIN, samples=500, seed=s).wins for s in range(5)}
    assert len(results) > 1


@pytest.mark.parametrize("board", [(), FLOP], ids=["preflop", "flop"])
def test_swapping_hands_sums_to_one(board):
    a = monte_carlo_equity(HERO, VILLAIN, board, samples=SAMPLES, seed=11)
    b = monte_carlo_equity(VILLAIN, HERO, board, samples=SAMPLES, seed=12)
    tolerance = 4 * math.hypot(a.std_error, b.std_error)
    assert abs(a.equity + b.equity - 1.0) < tolerance


def test_complete_board_is_exact():
    # River Kh 7h 2s 3c 4d: AsKs has a pair of kings, 9h8h only king-high.
    board = parse_cards("Kh 7h 2s 3c 4d")
    r = monte_carlo_equity(HERO, VILLAIN, board, samples=200, seed=0)
    assert_consistent(r, 200)
    assert (r.wins, r.ties, r.losses) == (200, 0, 0)
    assert r.equity == 1.0
    assert r.std_error == 0.0


def test_board_plays_is_a_split():
    # Royal flush on the board: every run-out is a split.
    board = parse_cards("Ac Kc Qc Jc Tc")
    r = monte_carlo_equity(parse_cards("2d 3d"), parse_cards("4h 5h"), board, samples=100, seed=0)
    assert (r.wins, r.ties, r.losses) == (0, 100, 0)
    assert r.equity == 0.5


@pytest.mark.parametrize(
    ("hero", "villain", "board"),
    [
        ("As Ks", "As 8h", ""),  # hero and villain share a card
        ("As Ks", "9h 8h", "As 7h 2s"),  # board repeats a hero card
        ("As Ks", "9h 8h", "Kh 8h 2s"),  # board repeats a villain card
        ("As As", "9h 8h", ""),  # repeated card within a hand
        ("As Ks", "9h 8h", "Kh Kh 2s"),  # repeated card on the board
    ],
)
def test_duplicate_cards_raise(hero, villain, board):
    with pytest.raises(ValueError):
        monte_carlo_equity(parse_cards(hero), parse_cards(villain), parse_cards(board), samples=10)


@pytest.mark.parametrize(
    ("hero", "villain", "board"),
    [
        ("As", "9h 8h", ""),  # hero needs two cards
        ("As Ks Qs", "9h 8h", ""),
        ("As Ks", "9h", ""),  # villain needs two cards
        ("As Ks", "9h 8h", "Kh"),  # board must be 0, 3, 4 or 5 cards
        ("As Ks", "9h 8h", "Kh 7h"),
        ("As Ks", "9h 8h", "Kh 7h 2s 3c 4d 5d"),
    ],
)
def test_wrong_card_counts_raise(hero, villain, board):
    with pytest.raises(ValueError):
        monte_carlo_equity(parse_cards(hero), parse_cards(villain), parse_cards(board), samples=10)


@pytest.mark.parametrize("samples", [0, -1])
def test_non_positive_samples_raise(samples):
    with pytest.raises(ValueError):
        monte_carlo_equity(HERO, VILLAIN, samples=samples, seed=0)
