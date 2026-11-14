"""Tests for the hand evaluator (Phase 1)."""

from collections import Counter
from itertools import combinations

import pytest

from nutsfinder.engine.cards import FULL_DECK, parse_cards
from nutsfinder.engine.evaluator import HandCategory, category, evaluate

C = HandCategory


def ev(text: str):
    """Evaluate a hand written as a string, e.g. ``ev("As Ks Qs Js Ts")``."""
    return evaluate(parse_cards(text))


def showdown(board: str, hero: str, villain: str) -> int:
    """+1 if hero wins, 0 for a split, -1 if villain wins."""
    h, v = ev(f"{hero} {board}"), ev(f"{villain} {board}")
    return (h > v) - (h < v)


# --- Plumbing ---------------------------------------------------------------


def test_result_is_tuple_starting_with_category():
    r = ev("As Ks Qs Js Ts")
    assert isinstance(r, tuple)
    assert r[0] == C.STRAIGHT_FLUSH
    assert category(r) is C.STRAIGHT_FLUSH


@pytest.mark.parametrize("text", ["", "As", "As Ks Qs Js", "As Ks Qs Js Ts 9s 8s 7s"])
def test_wrong_number_of_cards_raises(text):
    with pytest.raises(ValueError):
        ev(text)


def test_accepts_tuples_and_lists():
    cards = parse_cards("As Ks Qs Js Ts 2c 3d")
    assert evaluate(cards) == evaluate(tuple(cards))


def test_input_order_does_not_matter():
    cards = parse_cards("9c 9d 4h 4s Kd 2c 7h")
    assert evaluate(cards) == evaluate(cards[::-1]) == evaluate(sorted(cards))


# --- One example per category; each category beats the one below ------------

# (worst, best) 5-card hand in each category.
EXTREMES = {
    C.HIGH_CARD: ("7s 5d 4h 3c 2s", "As Kd Qh Jc 9s"),
    C.PAIR: ("2s 2d 3h 4c 5d", "As Ad Kh Qc Js"),
    C.TWO_PAIR: ("3s 3d 2h 2c 4d", "As Ad Kh Kc Qs"),
    C.THREE_OF_A_KIND: ("2s 2d 2h 3c 4d", "As Ad Ah Kc Qs"),
    C.STRAIGHT: ("As 2d 3h 4c 5s", "As Kd Qh Jc Ts"),
    C.FLUSH: ("7h 5h 4h 3h 2h", "Ah Kh Qh Jh 9h"),
    C.FULL_HOUSE: ("2s 2d 2h 3c 3d", "As Ad Ah Kc Ks"),
    C.FOUR_OF_A_KIND: ("2s 2d 2h 2c 3d", "As Ad Ah Ac Ks"),
    C.STRAIGHT_FLUSH: ("Ah 2h 3h 4h 5h", "As Ks Qs Js Ts"),
}


@pytest.mark.parametrize("cat", list(HandCategory))
def test_category_extremes(cat):
    worst, best = (ev(t) for t in EXTREMES[cat])
    assert category(worst) is cat
    assert category(best) is cat
    assert worst < best


@pytest.mark.parametrize("cat", list(HandCategory)[:-1])
def test_best_of_category_loses_to_worst_of_next(cat):
    best_lower = ev(EXTREMES[cat][1])
    worst_upper = ev(EXTREMES[HandCategory(cat + 1)][0])
    assert best_lower < worst_upper


# --- Straights and the wheel ---------------------------------------------------


def test_wheel_is_lowest_straight():
    assert category(ev("As 2d 3h 4c 5s")) is C.STRAIGHT
    assert ev("As 2d 3h 4c 5s") < ev("2s 3d 4h 5c 6s")


def test_wheel_straight_flush_is_lowest_straight_flush():
    assert category(ev("Ah 2h 3h 4h 5h")) is C.STRAIGHT_FLUSH
    assert ev("Ah 2h 3h 4h 5h") < ev("2d 3d 4d 5d 6d")


def test_broadway_beats_king_high_straight():
    assert ev("As Kd Qh Jc Ts") > ev("Ks Qd Jh Tc 9s")


def test_royal_flush_beats_king_high_straight_flush():
    assert ev("As Ks Qs Js Ts") > ev("Kh Qh Jh Th 9h")


def test_straights_do_not_wrap_around():
    assert category(ev("Qs Kd Ah 2c 3s")) is C.HIGH_CARD
    assert category(ev("Js Qd Kh Ac 2s")) is C.HIGH_CARD


def test_straight_ranks_by_top_card():
    straights = [
        "As 2d 3h 4c 5s",
        "2s 3d 4h 5c 6s",
        "3s 4d 5h 6c 7s",
        "6s 7d 8h 9c Ts",
        "9s Td Jh Qc Ks",
        "Ts Jd Qh Kc As",
    ]
    ranks = [ev(s) for s in straights]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


# --- Kickers and tiebreaks --------------------------------------------------------


def test_high_card_compares_all_five():
    assert ev("As Jd 9h 7c 4s") > ev("Ah Jc 9d 7h 3d")
    assert ev("As Jd 9h 7c 4s") > ev("Ah Tc 9d 7h 4d")


def test_pair_kicker_ak_beats_aq():
    assert ev("As Ad Kh 7c 4s") > ev("Ah Ac Qd 7h 4d")


def test_pair_last_kicker_matters():
    assert ev("Ks Kd 9h 7c 4s") > ev("Kh Kc 9d 7h 3d")


def test_higher_pair_beats_better_kickers():
    assert ev("3s 3d 4h 5c 7s") > ev("2s 2d As Kc Qd")


def test_pair_kicker_seven_cards():
    # Pair of aces, AK vs AQ.
    assert showdown("Ad 9c 7h 4s 2d", "As Kh", "Ac Qd") == 1


def test_two_pair_top_pair_first():
    assert ev("As Ad 2h 2c 3s") > ev("Ks Kd Qh Qc Js")


def test_two_pair_second_pair_next():
    assert ev("Ks Kd 8h 8c 2s") > ev("Kh Kc 7d 7s As")


def test_two_pair_kicker_matters():
    assert ev("Ks Kd 7h 7c As") > ev("Kh Kc 7d 7s Qd")


def test_trips_kickers():
    assert ev("7s 7d 7h Ac 2s") > ev("7c 7d 7h Kc Qs")
    assert ev("7s 7d 7h Ac 3s") > ev("7c 7d 7h Ac 2d")


def test_full_house_compares_trips_first():
    assert ev("3s 3d 3h 2c 2d") > ev("2s 2h 2c As Ad")


def test_full_house_then_pair():
    assert ev("Qs Qd Qh 9c 9d") > ev("Qs Qd Qh 8c 8d")


def test_full_house_seven_cards_two_trips():
    # Two sets of trips: the higher trips plus a pair from the lower trips.
    assert ev("3s 3d 3h 2c 2d 2s Kh") == ev("3s 3d 3h 2c 2d")
    assert category(ev("3s 3d 3h 2c 2d 2s Kh")) is C.FULL_HOUSE
    # KKK22 beats 33322 on the same board.
    assert showdown("3h 2c 2d 2s Kh", "3s 3d", "Kd Kc") == -1


def test_full_house_seven_cards_trips_and_two_pairs():
    assert ev("As Ad Ah Kc Kd Qs Qh") == ev("As Ad Ah Kc Kd")


def test_quads_kicker():
    assert ev("9s 9d 9h 9c Ks") > ev("9s 9d 9h 9c Qs")
    assert ev("9s 9d 9h 9c Ks Kd Kh") == ev("9s 9d 9h 9c Ks")


def test_flush_compares_all_five():
    assert ev("Ah Kh Qh Jh 9h") > ev("As Ks Qs Js 8s")
    assert ev("Ah Kh 7h 5h 3h") > ev("Ad Kd 7d 5d 2d")
    assert ev("Ah 9h 7h 5h 3h") > ev("Kd Qd Jd 9d 7d")


# --- Suits never break ties -------------------------------------------------------


def test_suits_never_break_ties():
    assert ev("As Kd 9h 7c 3s") == ev("Ad Ks 9c 7h 3d")
    assert ev("Ah Kh Qh Jh 9h") == ev("As Ks Qs Js 9s")
    assert ev("Kc Kd 7h 7s 2c") == ev("Kh Ks 7c 7d 2d")
    royals = {ev(" ".join(r + s for r in "AKQJT")) for s in "cdhs"}
    assert len(royals) == 1


# --- The board plays (splits) ------------------------------------------------------


def test_board_plays_broadway_split():
    assert showdown("As Ks Qd Jh Tc", "2c 3d", "4c 5d") == 0


def test_quads_on_board_kicker_decides():
    assert showdown("Kc Kd Kh Ks 9c", "2c 3d", "8c 7d") == 0
    assert showdown("Kc Kd Kh Ks 9c", "2c 3d", "Ac 2d") == -1


def test_counterfeited_two_pair_split():
    # Hero's 55 is counterfeited by the paired board: both play KK 99 Q.
    assert showdown("Kh Kd 9c 9s Qh", "5c 5d", "2c 3d") == 0


def test_higher_pocket_pair_beats_board_two_pair():
    # Hero plays KK QQ 9; villain plays KK 99 A.
    assert showdown("Kh Kd 9c 9s 5h", "Qc Qd", "As 2c") == 1


def test_flush_on_board_higher_hole_card_wins():
    assert showdown("2h 5h 8h Jh Kh", "Ah 3c", "Qh Qc") == 1
    assert showdown("2h 5h 8h Jh Kh", "3c 4c", "6c 7c") == 0


# --- 6 and 7 cards pick the best 5 ------------------------------------------------


def test_six_cards_best_five():
    assert ev("2s 3d 4h 5c 6s Kd") == ev("2s 3d 4h 5c 6s")
    assert ev("2s 3d 4h 5c 6s 7d") == ev("3d 4h 5c 6s 7d")
    assert ev("As Ad Kh Qc Js 2d") == ev("As Ad Kh Qc Js")


def test_seven_cards_flush_beats_straight():
    r = ev("9h Th Jh Qc Kd 2h 4h")
    assert category(r) is C.FLUSH
    assert r == ev("9h Th Jh 2h 4h")


def test_seven_cards_six_card_flush_uses_top_five():
    assert ev("Ah Kh 9h 7h 4h 2h Qs") == ev("Ah Kh 9h 7h 4h")


def test_seven_cards_straight_flush_beats_higher_straight():
    r = ev("5h 6h 7h 8h 9h Tc Jd")
    assert category(r) is C.STRAIGHT_FLUSH
    assert r == ev("5h 6h 7h 8h 9h")


def test_seven_cards_six_in_a_row_uses_top_straight():
    assert ev("4c 5d 6h 7s 8c 9d Kh") == ev("5d 6h 7s 8c 9d")


def test_seven_cards_wheel_with_six_is_six_high():
    assert ev("Ac 2d 3h 4s 5c 6d Kh") == ev("2d 3h 4s 5c 6d")


def test_seven_cards_three_pairs_uses_top_two_and_best_kicker():
    assert ev("As Ad Kc Kd Qs Qh 2c") == ev("As Ad Kc Kd Qs")
    assert ev("As Ad Kc Kd 2s 2h Qc") == ev("As Ad Kc Kd Qc")


def test_seven_cards_quads_beat_full_house():
    assert category(ev("9s 9d 9h 9c Ks Kd 2h")) is C.FOUR_OF_A_KIND


def test_seven_card_result_equals_best_five_card_subset():
    cards = parse_cards("Ks Kd 7h 7c 2s 9d Ah")
    best = max(evaluate(list(five)) for five in combinations(cards, 5))
    assert evaluate(cards) == best


# --- Exhaustive category counts -----------------------------------------------------

FIVE_CARD_COUNTS = {
    C.STRAIGHT_FLUSH: 40,
    C.FOUR_OF_A_KIND: 624,
    C.FULL_HOUSE: 3_744,
    C.FLUSH: 5_108,
    C.STRAIGHT: 10_200,
    C.THREE_OF_A_KIND: 54_912,
    C.TWO_PAIR: 123_552,
    C.PAIR: 1_098_240,
    C.HIGH_CARD: 1_302_540,
}

SEVEN_CARD_COUNTS = {
    C.STRAIGHT_FLUSH: 41_584,  # includes 4,324 royal flushes
    C.FOUR_OF_A_KIND: 224_848,
    C.FULL_HOUSE: 3_473_184,
    C.FLUSH: 4_047_644,
    C.STRAIGHT: 6_180_020,
    C.THREE_OF_A_KIND: 6_461_620,
    C.TWO_PAIR: 31_433_400,
    C.PAIR: 58_627_800,
    C.HIGH_CARD: 23_294_460,
}
SEVEN_CARD_ROYAL_FLUSHES = 4_324


def test_expected_count_tables_are_consistent():
    assert sum(FIVE_CARD_COUNTS.values()) == 2_598_960  # C(52, 5)
    assert sum(SEVEN_CARD_COUNTS.values()) == 133_784_560  # C(52, 7)


def test_exhaustive_five_card_category_counts():
    """All C(52, 5) = 2,598,960 five-card hands. The main correctness guard.

    Also checks there are exactly 7,462 distinct 5-card hand strengths, which
    catches tiebreak bugs that the category counts alone cannot see.
    """
    counts: Counter[HandCategory] = Counter()
    distinct = set()
    for hand in combinations(FULL_DECK, 5):
        r = evaluate(hand)
        counts[category(r)] += 1
        distinct.add(r)
    assert sum(counts.values()) == 2_598_960
    assert dict(counts) == FIVE_CARD_COUNTS
    assert len(distinct) == 7_462


@pytest.mark.slow
def test_exhaustive_seven_card_category_counts():
    """All C(52, 7) = 133,784,560 seven-card hands. Opt in with ``pytest -m slow``."""
    royal = ev("As Ks Qs Js Ts")
    counts: Counter[HandCategory] = Counter()
    royals = 0
    for hand in combinations(FULL_DECK, 7):
        r = evaluate(hand)
        counts[category(r)] += 1
        royals += r == royal
    assert sum(counts.values()) == 133_784_560
    assert dict(counts) == SEVEN_CARD_COUNTS
    assert royals == SEVEN_CARD_ROYAL_FLUSHES
