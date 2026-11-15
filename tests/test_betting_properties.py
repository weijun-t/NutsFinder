"""Property tests: random legal play never breaks the betting invariants.

Hypothesis generates random stacks, blinds, button and a random sequence of
legal actions (with random bet sizes), and plays the hand to the end. After
every step we check the invariants that must hold in *any* reachable state.
Along the way it also throws random actions at the engine and checks that
anything not listed as legal is rejected without changing the state.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    HandState,
    LegalActions,
    apply_action,
    deal_board,
    legal_actions,
    new_hand,
)
from nutsfinder.engine.cards import Deck

MAX_STEPS = 200  # far more actions than any hand can take with these stacks


def check_invariants(s: HandState) -> None:
    """Things that are true in every reachable state."""
    # Chips are conserved and never negative.
    assert sum(s.stacks) + s.pot == sum(s.starting_stacks)
    assert min(s.stacks) >= 0
    assert all(0 <= sb <= tb for sb, tb in zip(s.street_bets, s.total_bets))
    # The street always matches the board that has been dealt.
    assert s.cards_needed in (0, 1, 3)
    assert len(s.board) == s.street.board_size
    assert s.cards_needed == 0 or s.to_act is None
    assert s.last_raise >= s.big_blind
    # Between streets, previous streets' bets are always matched, so any gap
    # between the totals is on the current street.
    assert abs(s.total_bets[0] - s.total_bets[1]) == abs(s.street_bets[0] - s.street_bets[1])
    if s.to_act is not None:
        p = s.to_act
        assert s.stacks[p] > 0  # an all-in player is never asked to act
        legal = legal_actions(s)
        assert legal.can_check != legal.can_call
        assert legal.can_fold
        assert not (legal.can_bet and legal.can_raise)
        if legal.can_bet or legal.can_raise:
            assert s.current_bet < legal.min_to <= legal.max_to
            assert legal.max_to == s.street_bets[p] + s.stacks[p]
        if legal.can_call:
            assert 0 < legal.call_amount <= s.stacks[p]
    if s.is_hand_over:
        assert s.to_act is None and s.cards_needed == 0
        # After the hand, no chips remain uncalled.
        assert s.total_bets[0] == s.total_bets[1] or s.folded is not None
        if s.folded is None:
            assert len(s.board) == 5


def legal_choices(legal: LegalActions) -> list[str]:
    kinds = [k for k, ok in [
        ("fold", legal.can_fold),
        ("check", legal.can_check),
        ("call", legal.can_call),
        ("bet", legal.can_bet),
        ("raise", legal.can_raise),
    ] if ok]
    assert kinds
    return kinds


def draw_legal_action(data: st.DataObject, legal: LegalActions) -> Action:
    kind = data.draw(st.sampled_from(legal_choices(legal)))
    if kind in ("bet", "raise"):
        # Favour the edges (min raise, all-in), where the bugs usually are.
        size = data.draw(
            st.one_of(
                st.sampled_from([legal.min_to, legal.max_to]),
                st.integers(legal.min_to, legal.max_to),
            )
        )
        return Bet(size) if kind == "bet" else Raise(size)
    return {"fold": Fold(), "check": Check(), "call": Call()}[kind]


any_action = st.one_of(
    st.just(Fold()),
    st.just(Check()),
    st.just(Call()),
    st.builds(Bet, st.integers(1, 25_000)),
    st.builds(Raise, st.integers(1, 25_000)),
)


@settings(max_examples=400, deadline=None)
@given(
    data=st.data(),
    button=st.integers(0, 1),
    stacks=st.tuples(st.integers(1, 20_000), st.integers(1, 20_000)),
    blinds=st.sampled_from([(50, 100), (1, 2), (5, 10), (100, 100)]),
    seed=st.integers(0, 2**32),
)
def test_random_legal_play_keeps_invariants(data, button, stacks, blinds, seed):
    deck = Deck(seed)
    hole = (tuple(deck.deal(2)), tuple(deck.deal(2)))
    s = new_hand(button, stacks, blinds[0], blinds[1], hole_cards=hole)
    check_invariants(s)
    for _ in range(MAX_STEPS):
        if s.is_hand_over:
            break
        if s.cards_needed:
            s = deal_board(s, deck.deal(s.cards_needed))
        else:
            legal = legal_actions(s)
            # Fire a random action at the engine: it must be accepted exactly
            # when LegalActions says so, and a rejection changes nothing.
            probe = data.draw(any_action)
            if not legal.is_legal(probe):
                before = s
                try:
                    apply_action(s, legal.player, probe)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"illegal {probe} was accepted")
                assert s == before
            s = apply_action(s, legal.player, draw_legal_action(data, legal))
        check_invariants(s)
    assert s.is_hand_over, "hand did not finish"


@settings(max_examples=200, deadline=None)
@given(data=st.data(), stacks=st.tuples(st.integers(1, 5_000), st.integers(1, 5_000)))
def test_wrong_seat_always_rejected(data, stacks):
    s = new_hand(0, stacks)
    deck = Deck(0)
    while not s.is_hand_over:
        if s.cards_needed:
            s = deal_board(s, deck.deal(s.cards_needed))
            continue
        legal = legal_actions(s)
        action = draw_legal_action(data, legal)
        try:
            apply_action(s, 1 - legal.player, action)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-turn action was accepted")
        s = apply_action(s, legal.player, action)
