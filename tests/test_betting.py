"""Tests for the heads-up betting rules (Phase 2, steps 1-3).

Seat 0 is on the button in most tests, so seat 0 = small blind, seat 1 = big
blind. Stacks are 10,000 and blinds 50/100 unless a test says otherwise.
"""

import random

import pytest

from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    BIG_BLIND,
    SMALL_BLIND,
    STARTING_STACK,
    HandState,
    PublicState,
    Refund,
    Street,
    apply_action,
    contested_pot,
    deal_board,
    legal_actions,
    new_hand,
    public_view,
    uncalled_bet,
)
from nutsfinder.engine.cards import Deck, parse_cards

SB, BB = 0, 1  # with button=0
FLOP = parse_cards("Kh 7h 2s")
TURN = parse_cards("9c")
RIVER = parse_cards("3d")


def deal_next(state: HandState) -> HandState:
    """Deal the next street from a fixed board: flop, then turn, then river."""
    cards = {0: FLOP, 3: TURN, 4: RIVER}[len(state.board)]
    assert len(cards) == state.cards_needed
    return deal_board(state, cards)


def assert_conserved(state: HandState) -> None:
    assert sum(state.stacks) + state.pot == sum(state.starting_stacks)
    assert min(state.stacks) >= 0


def play(state: HandState, *actions) -> HandState:
    """Apply actions in order, each by whoever is to act, dealing when needed."""
    for action in actions:
        while state.cards_needed:
            state = deal_next(state)
        state = apply_action(state, state.to_act, action)
        assert_conserved(state)
    return state


# --- Starting a hand -------------------------------------------------------


def test_defaults():
    assert (SMALL_BLIND, BIG_BLIND, STARTING_STACK) == (50, 100, 10_000)


def test_blinds_posted_and_button_acts_first():
    s = new_hand(button=0)
    assert s.stacks == (9_950, 9_900)
    assert s.street_bets == (50, 100)
    assert s.pot == 150
    assert s.current_bet == 100
    assert s.street == Street.PREFLOP
    assert s.to_act == SB
    assert_conserved(s)


def test_button_one_posts_small_blind():
    s = new_hand(button=1)
    assert s.street_bets == (100, 50)
    assert s.to_act == 1


def test_preflop_legal_actions_for_button():
    legal = legal_actions(new_hand())
    assert legal.player == SB
    assert legal.can_fold and legal.can_call and not legal.can_check
    assert legal.call_amount == 50
    assert legal.can_raise and not legal.can_bet
    assert legal.min_to == 200  # the BB counts as an opening bet of 100
    assert legal.max_to == 10_000  # all-in: 50 already in + 9,950 behind


@pytest.mark.parametrize(
    "kwargs",
    [
        {"button": 2},
        {"stacks": (0, 100)},
        {"stacks": (100, -5)},
        {"stacks": (100.0, 100)},
        {"small_blind": 0},
        {"small_blind": 200, "big_blind": 100},
        {"hole_cards": [parse_cards("As Ks")]},
        {"hole_cards": [parse_cards("As Ks"), parse_cards("As Qs")]},
    ],
)
def test_new_hand_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        new_hand(**kwargs)


# --- Limp and the big blind's option ----------------------------------------


def test_limp_gives_big_blind_the_option():
    s = play(new_hand(), Call())
    assert s.street == Street.PREFLOP
    assert s.to_act == BB
    legal = legal_actions(s)
    assert legal.can_check and not legal.can_call and legal.can_fold
    assert legal.can_raise and legal.min_to == 200


def test_limp_then_bb_checks_goes_to_flop_with_bb_first():
    s = play(new_hand(), Call(), Check())
    # Preflop is closed, but it stays "preflop" until the flop is dealt.
    assert s.street == Street.PREFLOP and s.board == ()
    assert s.pot == 200
    assert s.cards_needed == 3
    assert s.to_act is None  # nobody acts until the flop is dealt
    s = deal_next(s)
    assert s.street == Street.FLOP
    assert s.street_bets == (0, 0)
    assert s.to_act == BB


def test_limp_then_bb_raises_reopens_for_button():
    s = play(new_hand(), Call(), Raise(300))
    assert s.to_act == SB
    legal = legal_actions(s)
    assert legal.can_call and legal.call_amount == 200
    assert legal.can_raise and legal.min_to == 500


# --- Minimum raises --------------------------------------------------------


def test_three_bet_example():
    s = play(new_hand(), Raise(300))
    assert legal_actions(s).min_to == 500  # 300 + increment 200
    s = play(s, Raise(1_000))
    assert s.last_raise == 700
    assert legal_actions(s).min_to == 1_700  # 1,000 + increment 700


def test_raise_below_minimum_rejected():
    with pytest.raises(ValueError):
        apply_action(new_hand(), SB, Raise(199))
    s = play(new_hand(), Raise(300))
    with pytest.raises(ValueError):
        apply_action(s, BB, Raise(499))


def test_any_whole_number_between_min_and_max_is_legal():
    s = play(new_hand(), Raise(237))
    assert s.street_bets == (237, 100)
    assert legal_actions(s).min_to == 237 + 137


def test_postflop_first_bet_minimum_is_one_big_blind():
    s = deal_next(play(new_hand(), Call(), Check()))
    legal = legal_actions(s)
    assert legal.can_bet and not legal.can_raise and legal.can_check
    assert legal.min_to == 100
    with pytest.raises(ValueError):
        apply_action(s, BB, Bet(99))


def test_postflop_min_raise_is_twice_the_bet():
    s = deal_next(play(new_hand(), Call(), Check()))
    s = play(s, Bet(100))
    assert legal_actions(s).min_to == 200
    s2 = play(deal_next(play(new_hand(), Call(), Check())), Bet(250))
    assert legal_actions(s2).min_to == 500


def test_bet_and_raise_not_interchangeable():
    with pytest.raises(ValueError):
        apply_action(new_hand(), SB, Bet(300))  # preflop there is already a bet
    flop = deal_next(play(new_hand(), Call(), Check()))
    with pytest.raises(ValueError):
        apply_action(flop, BB, Raise(300))  # nothing to raise yet


def test_increment_resets_each_street():
    s = play(new_hand(), Raise(300), Raise(1_000), Call())
    assert s.cards_needed == 3
    s = deal_next(s)
    assert s.street == Street.FLOP
    assert s.last_raise == BIG_BLIND
    assert legal_actions(s).min_to == 100


# --- All-ins and reopening --------------------------------------------------


def short_all_in_setup(sb_stack: int) -> HandState:
    """Limp/check to the flop; BB bets 1,000. The SB then has sb_stack - 100."""
    s = new_hand(stacks=(sb_stack, 10_000))
    return play(s, Call(), Check(), Bet(1_000))


def test_all_in_allowed_below_min_raise():
    s = short_all_in_setup(1_600)  # SB has 1,500 behind, a full raise needs 2,000
    legal = legal_actions(s)
    assert legal.can_raise
    assert legal.min_to == legal.max_to == 1_500  # only the all-in
    with pytest.raises(ValueError):
        apply_action(s, SB, Raise(1_200))  # not all-in and below the minimum


def test_short_all_in_does_not_reopen_or_change_increment():
    s = short_all_in_setup(1_600)
    s = play(s, Raise(1_500))  # increment 500 < 1,000: a short all-in
    assert s.stacks[SB] == 0
    assert s.last_raise == 1_000  # unchanged
    assert s.raise_allowed[BB] is False  # BB already acted: not reopened
    legal = legal_actions(s)
    assert legal.player == BB
    assert legal.can_call and legal.can_fold
    assert not legal.can_raise and not legal.can_bet
    assert legal.call_amount == 500


def test_full_all_in_raise_does_reopen():
    s = short_all_in_setup(2_200)  # 2,100 behind: increment 1,100 >= 1,000
    s = play(s, Raise(2_100))
    assert s.last_raise == 1_100  # a full raise sets the new increment
    assert s.raise_allowed[BB] is True  # and reopens the betting
    # Heads-up the BB still cannot raise: the only opponent is all-in.
    legal = legal_actions(s)
    assert legal.can_call and not legal.can_raise


def test_full_raise_reopens_for_player_who_already_acted():
    s = deal_next(play(new_hand(), Call(), Check()))
    s = play(s, Bet(200))  # BB bets
    assert s.raise_allowed[SB] is True
    s = play(s, Raise(600))  # SB raises: BB, who already acted, may re-raise
    assert s.raise_allowed[BB] is True
    legal = legal_actions(s)
    assert legal.can_raise and legal.min_to == 1_000


def test_call_for_less_when_short():
    s = new_hand(stacks=(10_000, 3_000))
    s = play(s, Raise(5_000))
    legal = legal_actions(s)
    assert legal.call_amount == 2_900  # BB only has 2,900 behind
    assert not legal.can_raise


def test_uneven_all_in_refunds_uncalled_excess():
    s = new_hand(stacks=(10_000, 3_000))
    s = play(s, Raise(10_000))  # SB shoves
    s = play(s, Call())  # BB calls all-in for 3,000
    assert s.stacks == (7_000, 0)  # 7,000 uncalled returned to the SB
    assert s.pot == 6_000
    assert s.total_bets == (3_000, 3_000)
    assert_conserved(s)
    # Betting is over: deal flop, turn, river one street at a time.
    seen = []
    while s.cards_needed:
        assert s.to_act is None
        seen.append(s.cards_needed)
        s = deal_next(s)
    assert seen == [3, 1, 1]
    assert s.is_hand_over and s.is_showdown
    assert s.street == Street.RIVER and len(s.board) == 5
    with pytest.raises(ValueError):
        legal_actions(s)


def test_bigger_stack_calls_short_shove():
    s = new_hand(stacks=(10_000, 3_000))
    s = play(s, Call(), Raise(3_000))  # limp, BB shoves its 3,000
    assert legal_actions(s).call_amount == 2_900
    s = play(s, Call())
    assert s.stacks == (7_000, 0) and s.pot == 6_000
    assert s.cards_needed == 3 and s.to_act is None


def test_all_in_on_flop_runs_out_turn_and_river():
    s = deal_next(play(new_hand(), Call(), Check()))
    s = play(s, Bet(9_900), Call())
    assert s.stacks == (0, 0) and s.pot == 20_000
    assert s.street == Street.FLOP and s.cards_needed == 1 and s.to_act is None
    s = deal_next(s)
    assert s.street == Street.TURN and s.cards_needed == 1 and s.to_act is None
    s = deal_next(s)
    assert s.street == Street.RIVER and s.is_showdown


def test_uncalled_bet_and_contested_pot_helpers():
    assert uncalled_bet((10_000, 3_000)) == (0, 7_000)
    assert uncalled_bet((150, 400)) == (1, 250)
    assert uncalled_bet((500, 500)) == (None, 0)
    assert contested_pot((10_000, 3_000)) == 6_000
    assert contested_pot((500, 500)) == 1_000
    for bets in [(10_000, 3_000), (150, 400), (0, 0)]:
        _, refund = uncalled_bet(bets)
        assert contested_pot(bets) + refund == sum(bets)


# --- Short-stacked blinds --------------------------------------------------


def test_big_blind_short_posts_all_in():
    s = new_hand(stacks=(10_000, 60))
    assert s.stacks == (9_950, 0) and s.street_bets == (50, 60)
    legal = legal_actions(s)
    assert legal.call_amount == 10
    assert not legal.can_raise  # the BB is all-in
    s = play(s, Call())
    assert s.pot == 120 and s.cards_needed == 3 and s.to_act is None


def test_small_blind_short_closes_preflop_immediately():
    s = new_hand(stacks=(30, 10_000))
    # SB all-in for 30; the BB's extra 70 is uncalled and comes straight back.
    assert s.stacks == (0, 9_970)
    assert s.pot == 60
    assert s.street == Street.PREFLOP and s.cards_needed == 3
    assert s.to_act is None
    assert_conserved(s)


def test_both_blinds_short():
    s = new_hand(stacks=(50, 30))
    assert s.stacks == (20, 0)  # 20 of the SB's 50 was uncalled
    assert s.pot == 60
    assert s.to_act is None and s.cards_needed == 3


def test_big_blind_shorter_than_small_blind():
    s = new_hand(stacks=(10_000, 40))
    # The BB is all-in for 40, less than the SB's 50: SB's extra 10 returned.
    assert s.stacks == (9_960, 0) and s.pot == 80
    assert s.to_act is None


# --- Folding ---------------------------------------------------------------


def test_fold_preflop_ends_hand():
    s = play(new_hand(), Fold())
    assert s.folded == SB
    assert s.is_hand_over and not s.is_showdown
    assert s.to_act is None and s.cards_needed == 0
    # The SB's 50 was matched; the other 50 of the BB was uncalled.
    assert s.pot == 100 and s.stacks == (9_950, 9_950)
    assert_conserved(s)


def test_fold_to_postflop_bet_returns_the_bet():
    s = deal_next(play(new_hand(), Call(), Check()))
    s = play(s, Bet(500), Fold())
    assert s.folded == SB
    assert s.pot == 200
    assert s.stacks == (9_900, 9_900)


# --- Illegal actions -------------------------------------------------------


def test_check_when_facing_a_bet_rejected():
    with pytest.raises(ValueError):
        apply_action(new_hand(), SB, Check())


def test_call_when_nothing_to_call_rejected():
    s = play(new_hand(), Call())
    with pytest.raises(ValueError):
        apply_action(s, BB, Call())


def test_fold_when_check_is_free_is_allowed():
    # Never a good move, but legal: the BB folds after a limp and loses the blind.
    s = play(new_hand(), Call())
    s = apply_action(s, BB, Fold())
    assert s.folded == BB
    assert s.is_hand_over and not s.is_showdown
    assert sum(s.stacks) + s.pot == 2 * STARTING_STACK


def test_bet_or_raise_more_than_stack_rejected():
    with pytest.raises(ValueError):
        apply_action(new_hand(), SB, Raise(10_001))
    flop = deal_next(play(new_hand(), Call(), Check()))
    with pytest.raises(ValueError):
        apply_action(flop, BB, Bet(9_901))
    play(flop, Bet(9_900))  # exactly all-in is fine


def test_out_of_turn_rejected():
    with pytest.raises(ValueError):
        apply_action(new_hand(), BB, Call())
    with pytest.raises(ValueError):
        apply_action(new_hand(), 5, Call())


def test_acting_after_hand_over_rejected():
    s = play(new_hand(), Fold())
    for seat in (0, 1):
        with pytest.raises(ValueError):
            apply_action(s, seat, Check())


def test_acting_before_board_dealt_rejected():
    s = play(new_hand(), Call(), Check())
    with pytest.raises(ValueError):
        apply_action(s, BB, Check())
    with pytest.raises(ValueError):
        legal_actions(s)


def test_acting_during_all_in_run_out_rejected():
    s = play(new_hand(), Raise(10_000), Call())
    s = deal_next(s)
    assert s.to_act is None
    with pytest.raises(ValueError):
        apply_action(s, BB, Check())


def test_failed_action_leaves_state_unchanged():
    s = new_hand()
    before = s
    with pytest.raises(ValueError):
        apply_action(s, SB, Raise(150))
    assert s == before


@pytest.mark.parametrize("bad", [0, -100, 1.5, True, "100", None])
def test_action_amounts_must_be_positive_ints(bad):
    with pytest.raises(ValueError):
        Bet(bad)
    with pytest.raises(ValueError):
        Raise(bad)


# --- Turn order on every street --------------------------------------------


@pytest.mark.parametrize("button", [0, 1])
def test_turn_order_every_street(button):
    bb = 1 - button
    s = new_hand(button=button)
    assert s.to_act == button  # preflop: button first
    s = play(s, Raise(300))
    assert s.to_act == bb
    s = play(s, Call())
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        s = deal_next(s)
        assert s.street == street
        assert s.to_act == bb  # postflop: big blind first
        s = play(s, Check())
        assert s.to_act == button  # button last
        s = play(s, Check())
    assert s.is_showdown
    assert s.pot == 600


@pytest.mark.parametrize("button", [0, 1])
def test_bet_call_on_each_street_moves_on(button):
    bb = 1 - button
    s = play(new_hand(button=button), Call(), Check())
    for _ in range(3):
        s = deal_next(s)
        assert s.to_act == bb
        s = play(s, Bet(100))
        assert s.to_act == button
        s = play(s, Call())
    assert s.is_showdown and s.pot == 800


def test_river_closing_ends_hand():
    s = play(new_hand(), Call(), Check(), Check(), Check(), Check(), Check())
    assert s.street == Street.TURN and len(s.board) == 4 and s.cards_needed == 1
    s = deal_next(s)
    s = play(s, Bet(400), Raise(1_200), Call())
    assert s.is_showdown and s.pot == 200 + 2_400
    with pytest.raises(ValueError):
        apply_action(s, BB, Check())


# --- History, conservation, public view, dealing ----------------------------


def test_history_records_every_action_with_chips_moved():
    s = play(new_hand(), Raise(300), Call())
    assert s.blinds == (50, 100)
    assert [(r.street, r.player, r.action, r.chips, r.all_in) for r in s.history] == [
        (Street.PREFLOP, SB, Raise(300), 250, False),
        (Street.PREFLOP, BB, Call(), 200, False),
    ]
    assert s.refunds == ()


def test_fold_to_raise_records_the_refund():
    # The reviewer's example: SB raises to 300 and the BB folds. The SB put in
    # 50 + 250 = 300 but only 100 of it was matched; 200 comes back.
    s = play(new_hand(), Raise(300), Fold())
    assert s.blinds == (50, 100)
    assert [r.chips for r in s.history] == [250, 0]
    assert s.refunds == (Refund(Street.PREFLOP, SB, 200),)
    assert s.pot == 200
    assert rebuild_from_record(s) == (s.stacks, s.total_bets)


def test_short_blind_refund_is_recorded():
    s = new_hand(stacks=(30, 10_000))
    assert s.blinds == (30, 100)
    assert s.refunds == (Refund(Street.PREFLOP, BB, 70),)
    assert rebuild_from_record(s) == (s.stacks, s.total_bets)


def test_all_in_flag():
    s = new_hand(stacks=(10_000, 3_000))
    s = play(s, Raise(10_000), Call())
    assert [(r.action, r.chips, r.all_in) for r in s.history] == [
        (Raise(10_000), 9_950, True),
        (Call(), 2_900, True),
    ]
    assert "(all-in)" in str(s.history[0])
    assert s.refunds == (Refund(Street.PREFLOP, SB, 7_000),)


def rebuild_from_record(s: HandState) -> tuple[tuple[int, int], tuple[int, int]]:
    """Recompute (stacks, total_bets) from starting stacks + blinds + history + refunds."""
    stacks = [s.starting_stacks[i] - s.blinds[i] for i in (0, 1)]
    for r in s.history:
        stacks[r.player] -= r.chips
    for r in s.refunds:
        stacks[r.player] += r.chips
    totals = [s.starting_stacks[i] - stacks[i] for i in (0, 1)]
    return (stacks[0], stacks[1]), (totals[0], totals[1])


@pytest.mark.parametrize("seed", range(300))
def test_record_rebuilds_stacks_and_pot_for_random_hands(seed):
    rng = random.Random(seed)
    stacks = (rng.randint(1, 15_000), rng.randint(1, 15_000))
    deck = Deck(seed)
    s = new_hand(rng.randint(0, 1), stacks)
    while True:
        assert rebuild_from_record(s) == (s.stacks, s.total_bets)
        assert sum(rebuild_from_record(s)[1]) == s.pot
        if s.is_hand_over:
            break
        if s.cards_needed:
            s = deal_board(s, deck.deal(s.cards_needed))
            continue
        legal = legal_actions(s)
        options: list = [Fold(), Check() if legal.can_check else Call()]
        if legal.can_bet or legal.can_raise:
            size = rng.choice([legal.min_to, legal.max_to, rng.randint(legal.min_to, legal.max_to)])
            options.append(Bet(size) if legal.can_bet else Raise(size))
        s = apply_action(s, legal.player, rng.choice(options))


def test_street_matches_board_during_all_in_run_out():
    # The reviewer's example: before the fix this said TURN with 3 board cards.
    s = new_hand(stacks=(10_000, 3_000))
    s = play(s, Raise(10_000), Call())
    assert s.street == Street.PREFLOP and s.board == ()
    s = deal_next(s)
    assert s.street == Street.FLOP and len(s.board) == 3 and s.cards_needed == 1
    s = deal_next(s)
    assert s.street == Street.TURN and len(s.board) == 4 and s.cards_needed == 1
    s = deal_next(s)
    assert s.street == Street.RIVER and len(s.board) == 5 and s.is_showdown


def test_street_for_board():
    assert [Street.for_board(n) for n in (0, 3, 4, 5)] == list(Street)
    with pytest.raises(ValueError):
        Street.for_board(2)


@pytest.mark.parametrize("bad", [True, False, 1.0, 0.0, None, "0", 2, -1])
def test_seats_must_be_int_0_or_1(bad):
    with pytest.raises(ValueError):
        new_hand(button=bad)
    with pytest.raises(ValueError):
        apply_action(new_hand(), bad, Call())


def test_chips_conserved_through_long_hand():
    s = new_hand(stacks=(7_300, 12_400))
    s = play(
        s,
        Raise(250), Raise(800), Call(),          # preflop
        Bet(600), Raise(1_800), Call(),          # flop
        Check(), Bet(2_000), Call(),             # turn
        Check(), Bet(2_700),                     # river: SB shoves the rest
    )
    assert s.stacks[SB] == 0
    s = play(s, Call())
    assert s.is_showdown
    assert s.pot == 2 * 7_300
    assert s.stacks == (0, 12_400 - 7_300)


# Every field a bot may see. Adding a field to PublicState must be a
# deliberate decision, so this list has to be edited by hand when it changes.
ALLOWED_PUBLIC_FIELDS = {
    "seat", "button", "small_blind", "big_blind", "starting_stacks", "blinds",
    "stacks", "street_bets", "total_bets", "pot", "current_bet", "last_raise",
    "street", "board", "to_act", "folded", "history", "refunds", "legal",
}


def test_public_view_has_only_allowed_fields():
    assert set(PublicState.__dataclass_fields__) == ALLOWED_PUBLIC_FIELDS


def test_public_view_hides_hole_cards():
    hole = (parse_cards("As Ks"), parse_cards("9h 8h"))
    s = new_hand(hole_cards=hole)
    for seat in (0, 1):
        view = public_view(s, seat)
        assert isinstance(view, PublicState)
        assert view.seat == seat
        assert view.legal == legal_actions(s)
        assert view.pot == 150 and view.to_act == SB
        # No hole card appears anywhere in the view, even as text.
        for card in (c for h in hole for c in h):
            assert str(card) not in str(view) and repr(card) not in repr(view)
    assert public_view(play(s, Fold()), BB).legal is None


@pytest.mark.parametrize("bad", [2, -1, True, False, 1.0, None, "0"])
def test_public_view_rejects_bad_seat(bad):
    with pytest.raises(ValueError):
        public_view(new_hand(), bad)


def test_states_are_immutable():
    # Try to overwrite every real dataclass field (``pot`` is a property, so
    # assigning to it would fail even on a mutable class and proves nothing).
    s = play(new_hand(hole_cards=(parse_cards("As Ks"), parse_cards("9h 8h"))), Raise(300))
    objects = [s, public_view(s, 0), legal_actions(s), s.history[0]]
    objects.append(play(s, Fold()).refunds[0])
    for obj in objects:
        for name in obj.__dataclass_fields__:
            with pytest.raises(AttributeError):
                setattr(obj, name, getattr(obj, name))
    # And the containers inside are tuples, which cannot be edited in place.
    for value in (s.stacks, s.blinds, s.board, s.hole_cards, s.hole_cards[0], s.history):
        assert isinstance(value, tuple)


def test_new_hand_copies_mutable_inputs():
    stacks = [10_000, 10_000]
    hole = [list(parse_cards("As Ks")), list(parse_cards("9h 8h"))]
    s = new_hand(stacks=stacks, hole_cards=hole)  # type: ignore[arg-type]
    stacks[0] = 5
    hole[0][0] = parse_cards("2d")[0]
    assert s.starting_stacks == (10_000, 10_000)
    assert s.hole_cards[0] == tuple(parse_cards("As Ks"))


def test_deal_board_validation():
    s = play(new_hand(hole_cards=(parse_cards("As Ks"), parse_cards("9h 8h"))), Call(), Check())
    with pytest.raises(ValueError):
        deal_board(s, parse_cards("2c 3c"))  # flop needs 3
    with pytest.raises(ValueError):
        deal_board(s, parse_cards("2c 2c 3c"))  # duplicate
    with pytest.raises(ValueError):
        deal_board(s, parse_cards("As 2c 3c"))  # a hole card
    s = deal_board(s, parse_cards("2c 3c 4c"))
    with pytest.raises(ValueError):
        deal_board(s, parse_cards("5c"))  # betting must happen first
    with pytest.raises(ValueError):
        deal_board(new_hand(), FLOP)  # preflop betting not done
