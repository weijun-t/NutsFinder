"""Tests for settling hands: fold wins, showdowns, split pots, the odd chip."""

from dataclasses import replace

import pytest

from nutsfinder.engine.actions import Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, apply_action, deal_board, new_hand
from nutsfinder.engine.cards import parse_cards
from nutsfinder.engine.evaluator import HandCategory, category
from nutsfinder.engine.showdown import settle, split_pot


def hand(hole0: str, hole1: str, board: str, *actions, button: int = 0, stacks=(10_000, 10_000)):
    """Play ``actions`` (by whoever is to act), dealing ``board`` as needed."""
    s = new_hand(button, stacks, hole_cards=(parse_cards(hole0), parse_cards(hole1)))
    cards = parse_cards(board)
    queue = list(actions)
    while not s.is_hand_over:
        if s.cards_needed:
            s = deal_board(s, cards[len(s.board) : len(s.board) + s.cards_needed])
        else:
            s = apply_action(s, s.to_act, queue.pop(0))
    assert not queue, "unused actions"
    return s


def check_down(hole0: str, hole1: str, board: str) -> HandState:
    """Limp, then check every street: pot 200, both put in 100."""
    return hand(hole0, hole1, board, Call(), *[Check()] * 7)


def assert_chips_accounted(s: HandState) -> None:
    r = settle(s)
    assert sum(r.final_stacks) == sum(s.starting_stacks)
    assert sum(r.net) == 0
    assert sum(r.winnings) == s.pot
    for seat in (0, 1):
        assert r.net[seat] == r.winnings[seat] - s.total_bets[seat]
        assert r.final_stacks[seat] >= 0


def test_clear_win_at_showdown():
    s = check_down("As Ah", "Kd Kc", "2c 7d 9h Jc 3s")
    r = settle(s)
    assert r.winners == (0,)
    assert r.winnings == (200, 0)
    assert r.final_stacks == (10_100, 9_900)
    assert r.net == (100, -100)
    assert category(r.ranks[0]) == HandCategory.PAIR
    assert_chips_accounted(s)


def test_board_plays_split_pot():
    s = check_down("2c 3d", "4h 5h", "As Ks Qs Js Ts")  # royal flush on board
    r = settle(s)
    assert r.winners == (0, 1)
    assert r.ranks[0] == r.ranks[1]
    assert r.winnings == (100, 100)
    assert r.net == (0, 0)
    assert_chips_accounted(s)


def test_same_hand_different_suits_split():
    s = check_down("Ac Kd", "Ah Ks", "Qc Jd 8h 4s 2c")
    assert settle(s).winners == (0, 1)


def test_kicker_decides():
    s = check_down("Ac Kd", "Ah Qs", "Ad 9c 7h 4s 2c")  # both pair of aces
    r = settle(s)
    assert r.winners == (0,)
    assert r.net == (100, -100)


def test_fold_preflop_winner_takes_pot():
    s = hand("As Ah", "Kd Kc", "", Fold())  # button (seat 0) folds its blind
    r = settle(s)
    assert r.winners == (1,)
    assert r.ranks == (None, None)
    assert r.winnings == (0, 100)
    assert r.net == (-50, 50)
    assert_chips_accounted(s)


def test_fold_needs_no_hole_cards():
    s = apply_action(new_hand(), 0, Fold())
    assert settle(s).net == (-50, 50)


@pytest.mark.parametrize(
    ("hole0", "hole1", "final", "net"),
    [
        ("As Ah", "Kd Kc", (13_000, 0), (3_000, -3_000)),  # big stack wins
        ("Kd Kc", "As Ah", (7_000, 6_000), (-3_000, 3_000)),  # short stack doubles
    ],
)
def test_all_in_with_uneven_stacks(hole0, hole1, final, net):
    s = hand(hole0, hole1, "2c 7d 9h Jc 3s", Raise(10_000), Call(), stacks=(10_000, 3_000))
    r = settle(s)
    assert r.final_stacks == final
    assert r.net == net
    assert s.refunds[0].chips == 7_000  # the part the short stack couldn't call
    assert_chips_accounted(s)


def test_settle_rejects_unfinished_hand():
    with pytest.raises(ValueError):
        settle(new_hand())


def test_showdown_needs_hole_cards():
    s = new_hand()
    s = apply_action(s, 0, Call())
    s = apply_action(s, 1, Check())
    for cards in ("2c 7d 9h", "Jc", "3s"):
        s = deal_board(s, parse_cards(cards))
        s = apply_action(s, 1, Check())
        s = apply_action(s, 0, Check())
    with pytest.raises(ValueError):
        settle(s)


@pytest.mark.parametrize("button", [0, 1])
def test_split_pot_odd_chip_goes_to_big_blind(button):
    for pot in range(0, 1_001):
        a, b = split_pot(pot, button)
        assert a + b == pot
        assert abs(a - b) <= 1
        if a != b:
            bigger = 0 if a > b else 1
            assert bigger == 1 - button


@pytest.mark.parametrize("button", [0, 1])
def test_odd_pot_at_showdown_loses_no_chips(button):
    # Heads-up the engine never produces an odd showdown pot, so forge one:
    # take a tied showdown and move one extra chip from the BB's stack into
    # the pot. The split must still pay out every chip, the odd one to the BB.
    s = check_down("2c 3d", "4h 5h", "As Ks Qs Js Ts")
    s = replace(s, button=button)
    bb = 1 - button
    tb, st = list(s.total_bets), list(s.stacks)
    tb[bb] += 1
    st[bb] -= 1
    s = replace(s, total_bets=(tb[0], tb[1]), stacks=(st[0], st[1]))
    r = settle(s)
    assert s.pot == 201
    assert r.winnings[bb] == 101 and r.winnings[button] == 100
    assert_chips_accounted(s)
