"""Tests for the N-player engine (``nutsfinder.engine.multiway``).

Covers: the worked side-pot example from ``docs/phase7_multiway_research.md``,
TDA cumulative reopening, BB option, action order for 2/3/6/8 players, the
heads-up exception, odd chips, uncalled returns, run-outs, information
hiding, a brute random fuzz against an independent side-pot reference, an
N=2 differential test against the heads-up engine, and golden hashes that
freeze the heads-up engine's hand logs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections.abc import Iterator

import pytest

from nutsfinder.bots.always_raise import AlwaysRaiseBot
from nutsfinder.bots.call_station import CallStationBot
from nutsfinder.bots.random_bot import RandomBot
from nutsfinder.engine import hand_log as hu_hand_log
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState, Street
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import FULL_DECK, Card, Deck, parse_cards
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.engine.multiway import (
    HUBotAdapter,
    MultiAlwaysRaiseBot,
    MultiBot,
    MultiCallStationBot,
    MultiHandState,
    MultiHandSummary,
    MultiPublicState,
    MultiRandomBot,
    apply_action,
    deal_board,
    format_log,
    legal_actions,
    make_log,
    new_hand,
    play_hand_multi,
    play_match_multi,
    position_names,
    public_view,
    replay,
    settle,
    side_pots,
    split_amount,
    uncalled_bet,
)
from nutsfinder.engine.runner import hand_seeds, play_hand

from multiway_reference import RefLegal, RefTable, illegal_probes, ref_deal, ref_pick, ref_stacks

# --- Helpers -----------------------------------------------------------------


def cards(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


def run(state: MultiHandState, *actions: Action) -> MultiHandState:
    """Apply actions, each by whoever is to act."""
    for action in actions:
        assert state.to_act is not None, f"nobody to act for {action}"
        state = apply_action(state, state.to_act, action)
    return state


def deal_out(state: MultiHandState, board: tuple[Card, ...]) -> MultiHandState:
    """Deal the remaining board while betting is over (run-out)."""
    while state.cards_needed:
        k = state.cards_needed
        done = len(state.board)
        state = deal_board(state, board[done : done + k])
    return state


def check_down(state: MultiHandState, board: tuple[Card, ...]) -> MultiHandState:
    """Deal and check/call every street to the end."""
    while not state.is_hand_over:
        if state.cards_needed:
            done, k = len(state.board), state.cards_needed
            state = deal_board(state, board[done : done + k])
        else:
            legal = legal_actions(state)
            state = run(state, Check() if legal.can_check else Call())
    return state


# --- Worked side-pot example (research doc 1.2) -------------------------------


def test_worked_side_pot_example() -> None:
    # Button 0, SB 1, BB 2, UTG 3, HJ 4, CO 5.
    stacks = (10_000, 10_000, 10_000, 1_005, 3_000, 6_000)
    hole = [
        cards("Kh Qc"),  # BTN: ties CO
        cards("4h 4c"),  # SB (folds)
        cards("5c 5d"),  # BB (folds)
        cards("As Ac"),  # UTG: best
        cards("8c 6d"),  # HJ: worst
        cards("Kd Qd"),  # CO: ties BTN
    ]
    board = cards("2c 7d 9h Js 3s")
    state = new_hand(stacks, button=0, hole_cards=hole)
    assert state.to_act == 3
    state = run(state, Raise(1005), Raise(3000), Raise(6000), Raise(10_000), Fold(), Fold())
    assert state.to_act is None and state.street == Street.PREFLOP
    assert [(r.player, r.chips) for r in state.refunds] == [(0, 4000)]
    assert state.total_bets == (6000, 50, 100, 1005, 3000, 6000)
    assert state.pot == 16_155
    pots = [(p.amount, p.eligible) for p in state.pots()]
    assert pots == [(4170, (0, 3, 4, 5)), (5985, (0, 4, 5)), (6000, (0, 5))]

    state = deal_out(state, board)
    assert state.is_showdown
    result = settle(state)
    assert [(p.amount, p.winners, p.shares) for p in result.pots] == [
        (4170, (3,), (4170,)),
        (5985, (0, 5), (2992, 2993)),  # CO (seat 5) comes before BTN clockwise
        (6000, (0, 5), (3000, 3000)),
    ]
    assert result.net == (-8, -50, -100, 3165, -3000, -7)
    assert sum(result.net) == 0
    text = format_log(make_log(state, None))
    assert "Side pot 1 5985 (seats 0 4 5): seat 0 gets 2992, seat 5 gets 2993" in text


# --- Pot helpers --------------------------------------------------------------


def test_uncalled_bet_counts_folded_players() -> None:
    assert uncalled_bet((300, 50, 100)) == (0, 200)
    assert uncalled_bet((300, 300, 100)) == (None, 0)
    # A folded caller matched part of the bet: only the excess over them returns.
    assert uncalled_bet((1000, 800, 100)) == (0, 200)


def test_split_amount_odd_chips_clockwise() -> None:
    assert split_amount(170, (0, 2, 3), n=4, button=0) == {2: 57, 3: 57, 0: 56}
    assert split_amount(170, (0, 1, 2), n=4, button=2) == {0: 57, 1: 57, 2: 56}
    assert split_amount(7, (4,), n=6, button=1) == {4: 7}


# --- Reopening (TDA cumulative short all-ins) ----------------------------------


def _four_way_flop(stacks: tuple[int, ...]) -> MultiHandState:
    """Button 0, SB 1, BB 2, UTG 3; everyone limps; flop dealt; seat 1 to act."""
    state = new_hand(stacks, button=0)
    state = run(state, Call(), Call(), Call(), Check())
    state = deal_board(state, cards("2c 7d 9h"))
    assert state.to_act == 1
    return state


def test_cumulative_short_all_ins_reopen() -> None:
    # A (seat 1) bets 100, B (seat 2) shoves to 150, C (seat 3) shoves to 210.
    state = _four_way_flop((10_000, 10_000, 250, 310))
    state = run(state, Bet(100), Raise(150), Raise(210))
    assert state.last_raise == 100  # neither short all-in changed the increment
    d = legal_actions(state)  # seat 0 has not acted yet: may raise
    assert d.player == 0 and d.can_raise and d.min_to == 310
    state = run(state, Call())
    a = legal_actions(state)
    assert a.player == 1
    assert a.can_raise and a.min_to == 310 and a.call_amount == 110


def test_single_short_all_in_does_not_reopen() -> None:
    state = _four_way_flop((10_000, 10_000, 250, 10_000))
    state = run(state, Bet(100), Raise(150))
    c = legal_actions(state)  # seat 3 has not acted: may raise, min 150 + 100
    assert c.player == 3 and c.can_raise and c.min_to == 250
    state = run(state, Call(), Call())
    a = legal_actions(state)
    assert a.player == 1 and a.can_call and a.call_amount == 50
    assert not a.can_raise and a.min_to == a.max_to == 0
    with pytest.raises(ValueError):
        apply_action(state, 1, Raise(400))


def test_two_short_all_ins_below_a_full_raise_do_not_reopen() -> None:
    state = _four_way_flop((10_000, 10_000, 230, 280))  # B to 130, C to 180
    state = run(state, Bet(100), Raise(130), Raise(180), Call())
    a = legal_actions(state)
    assert a.player == 1 and a.call_amount == 80 and not a.can_raise


def test_full_raise_then_short_all_in_reopens_only_earlier_actors() -> None:
    # A bets 100, D... here: seat 1 bets 100, seat 2 raises to 400 (full),
    # seat 3 shoves 500 (short). Seat 1 faces 400 >= 300: open. Seat 2 faces 100: closed.
    state = _four_way_flop((10_000, 10_000, 10_000, 600))
    state = run(state, Bet(100), Raise(400), Raise(500), Call())
    a = legal_actions(state)
    assert a.player == 1 and a.can_raise and a.min_to == 800
    state = run(state, Call())
    b = legal_actions(state)
    assert b.player == 2 and b.call_amount == 100 and not b.can_raise


def test_no_raise_when_no_other_live_player_has_chips() -> None:
    state = new_hand((10_000, 10_000, 500), button=0)  # BB (seat 2) is short
    state = run(state, Call(), Fold())  # BTN limps, SB folds
    state = run(state, Raise(500))  # BB shoves; BTN is the only one with chips
    legal = legal_actions(state)
    assert legal.player == 0 and legal.can_call and not legal.can_raise


def test_short_all_in_opening_bet_needs_raise_to_two_big_blinds() -> None:
    state = new_hand((10_000, 10_000, 150), button=0)
    state = run(state, Call(), Call(), Check())
    state = deal_board(state, cards("2c 7d 9h"))
    state = run(state, Check(), Bet(50))  # seat 2 opens all-in for less than 1 BB
    legal = legal_actions(state)
    assert legal.player == 0 and legal.call_amount == 50 and legal.min_to == 200
    state = run(state, Raise(200))
    legal = legal_actions(state)  # seat 1 checked; faces 200 >= 100: open
    assert legal.player == 1 and legal.can_raise and legal.min_to == 300


# --- Blinds, BB option, action order -----------------------------------------


def test_bb_option_after_limps() -> None:
    state = new_hand((10_000,) * 3, button=0)
    assert state.to_act == 0  # 3-handed the button is UTG
    state = run(state, Call(), Call())
    legal = legal_actions(state)
    assert legal.player == 2 and legal.can_check and legal.can_raise and legal.min_to == 200
    state = run(state, Check())
    assert state.to_act is None and state.cards_needed == 3


def test_heads_up_exception() -> None:
    for button in (0, 1):
        state = new_hand((10_000, 10_000), button=button)
        assert state.blinds[button] == 50 and state.blinds[1 - button] == 100
        assert state.to_act == button  # button (SB) acts first preflop
        state = run(state, Call(), Check())
        state = deal_board(state, cards("2c 7d 9h"))
        assert state.to_act == 1 - button  # BB acts first postflop


def _action_orders(n: int, button: int) -> tuple[list[int], list[int]]:
    state = new_hand((10_000,) * n, button=button)
    pre: list[int] = []
    while state.to_act is not None:
        pre.append(state.to_act)
        state = run(state, Check() if legal_actions(state).can_check else Call())
    state = deal_board(state, cards("2c 7d 9h"))
    flop: list[int] = []
    while state.to_act is not None:
        flop.append(state.to_act)
        state = run(state, Check())
    return pre, flop


@pytest.mark.parametrize("n", [2, 3, 6, 8])
@pytest.mark.parametrize("button", [0, 1])
def test_action_order(n: int, button: int) -> None:
    pre, flop = _action_orders(n, button)
    if n == 2:
        assert pre == [button, 1 - button]
        assert flop == [1 - button, button]
        return
    sb, bb = (button + 1) % n, (button + 2) % n
    assert pre == [(bb + 1 + k) % n for k in range(n)]
    assert pre[-1] == bb and pre[-2] == sb
    assert flop == [(button + 1 + k) % n for k in range(n)]
    assert flop[-1] == button


def test_folded_and_all_in_seats_are_skipped() -> None:
    state = new_hand((10_000, 10_000, 10_000, 10_000, 300), button=0)
    # UTG (3) folds, seat 4 shoves 300, BTN calls, SB calls, BB calls.
    state = run(state, Fold(), Raise(300), Call(), Call(), Call())
    state = deal_board(state, cards("2c 7d 9h"))
    order = []
    while state.to_act is not None:
        order.append(state.to_act)
        state = run(state, Check())
    assert order == [1, 2, 0]


def test_fold_does_not_end_multiway_hand() -> None:
    state = new_hand((10_000,) * 4, button=0)
    state = run(state, Fold())
    assert not state.is_hand_over and state.to_act == 0
    state = run(state, Fold(), Fold())
    assert state.is_hand_over and not state.is_showdown
    result = settle(state)
    assert result.net == (0, -50, 50, 0) and result.winners == (2,)


def test_position_names() -> None:
    assert position_names(8, 0) == ("BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO")
    assert position_names(6, 0) == ("BTN", "SB", "BB", "UTG", "HJ", "CO")
    assert position_names(6, 2) == ("HJ", "CO", "BTN", "SB", "BB", "UTG")
    assert position_names(3, 1) == ("BB", "BTN", "SB")
    assert position_names(2, 1) == ("BB", "BTN")


def test_short_blinds_are_all_in() -> None:
    state = new_hand((10_000, 30, 60), button=0)
    assert state.blinds == (0, 30, 60) and state.stacks == (10_000, 0, 0)
    # Only the button can act: it faces 60.
    legal = legal_actions(state)
    assert legal.player == 0 and legal.call_amount == 60 and not legal.can_raise
    state = run(state, Call())
    assert state.to_act is None and state.cards_needed == 3


@pytest.mark.parametrize(
    "stacks, button",
    [((10_000,), 0), ((10_000,) * 9, 0), ((10_000, 10_000, 0), 0), ((10_000,) * 3, 3)],
)
def test_new_hand_rejects_bad_tables(stacks: tuple[int, ...], button: int) -> None:
    with pytest.raises(ValueError):
        new_hand(stacks, button=button)


# --- Odd chip, uncalled return, run-out ---------------------------------------

_ROYAL = cards("Ah Kh Qh Jh Th")
_NO_HEARTS = [cards("2c 3c"), cards("2d 3d"), cards("4c 5c"), cards("4d 5d")]


@pytest.mark.parametrize("button, expected", [(0, (56, 0, 57, 57)), (2, (57, 57, 56, 0))])
def test_odd_chip_three_way_split(button: int, expected: tuple[int, ...]) -> None:
    state = new_hand((10_000,) * 4, button=button, small_blind=20, big_blind=50,
                     hole_cards=_NO_HEARTS)
    sb = (button + 1) % 4
    # UTG calls, BTN calls, SB folds, BB checks; everyone plays the royal board.
    state = run(state, Call(), Call(), Fold(), Check())
    assert state.folded[sb]
    state = check_down(state, _ROYAL)
    result = settle(state)
    assert result.winnings == expected
    assert sum(result.winnings) == 170


def test_odd_chip_goes_to_small_blind_first() -> None:
    # 3-handed, blinds 25/51: BTN limps 51, SB completes, BB checks (pot 153).
    # Flop: SB bets 51, BB calls, BTN folds. SB and BB tie for 51 + 2*102 = 255.
    state = new_hand((10_000,) * 3, button=0, small_blind=25, big_blind=51,
                     hole_cards=_NO_HEARTS[:3])
    state = run(state, Call(), Call(), Check())
    state = deal_board(state, _ROYAL[:3])
    state = run(state, Bet(51), Call(), Fold())
    state = check_down(state, _ROYAL)
    result = settle(state)
    assert result.pots[0].amount == 255
    assert result.winnings == (0, 128, 127)  # SB is first clockwise from the button
    assert split_amount(5, (1, 2), n=3, button=0) == {1: 3, 2: 2}


def test_uncalled_raise_returned_when_all_fold() -> None:
    state = new_hand((10_000,) * 3, button=0)
    state = run(state, Raise(300), Fold(), Fold())
    assert [(r.player, r.chips) for r in state.refunds] == [(0, 200)]
    assert settle(state).net == (150, -50, -100)


def test_uncalled_shove_returned_over_short_caller() -> None:
    state = new_hand((10_000, 10_000, 3_000), button=0, hole_cards=_NO_HEARTS[:3])
    state = run(state, Raise(10_000), Fold(), Call())
    assert [(r.player, r.chips) for r in state.refunds] == [(0, 7_000)]
    assert state.pot == 6_050 and state.cards_needed == 3  # run-out
    state = deal_out(state, _ROYAL)
    assert state.is_showdown
    assert sum(settle(state).net) == 0


def test_run_out_when_one_player_has_chips() -> None:
    state = new_hand((10_000, 10_000, 2_000), button=0, hole_cards=_NO_HEARTS[:3])
    state = run(state, Raise(2000), Call(), Call())  # BB calls all-in
    assert state.cards_needed == 3
    state = deal_board(state, _ROYAL[:3])
    assert state.to_act == 1  # two players still have chips: they keep betting
    state = run(state, Check(), Check())
    state = deal_board(state, _ROYAL[3:4])
    state = run(state, Bet(8000), Call())  # both all-in now
    assert state.to_act is None and state.cards_needed == 1
    state = deal_board(state, _ROYAL[4:])
    assert state.is_showdown


# --- Bots and runner ----------------------------------------------------------


def test_runner_is_deterministic_and_replays() -> None:
    for n in range(2, 9):
        logs = [
            play_hand_multi([MultiRandomBot(s) for s in range(n)], seed=99, button=n - 1)
            for _ in range(2)
        ]
        assert logs[0] == logs[1]
        assert make_log(replay(logs[0]), logs[0].seed) == logs[0]


def test_play_match_multi_rotates_button() -> None:
    logs = play_match_multi([MultiCallStationBot() for _ in range(5)], 7, master_seed=3)
    assert [log.button for log in logs] == [0, 1, 2, 3, 4, 0, 1]
    assert all(log.show_order for log in logs)  # call stations always show down


def test_always_raise_field_conserves_chips() -> None:
    bots = [MultiAlwaysRaiseBot() for _ in range(6)]
    for i, seed in enumerate(hand_seeds(5, 50)):
        log = play_hand_multi(bots, seed, button=i % 6)
        assert sum(log.result.net) == 0


class _BadBot(MultiBot):
    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        return Bet(1)


def test_illegal_bot_action_raises() -> None:
    with pytest.raises(ValueError, match="seat 0"):
        play_hand_multi([_BadBot(), MultiCallStationBot(), MultiCallStationBot()], seed=1)


# --- Information hiding ---------------------------------------------------------


class _SpyBot(MultiRandomBot):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.acts: list[tuple[MultiPublicState, tuple[Card, ...]]] = []
        self.ends: list[MultiHandSummary] = []

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        self.acts.append((view, hole_cards))
        return super().act(view, hole_cards)

    def end_hand(self, summary: MultiHandSummary) -> None:
        self.ends.append(summary)


def _reachable(obj: object) -> Iterator[object]:
    stack, seen = [obj], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        yield x
        if isinstance(x, Card | str | int | float | bool | type(None)):
            continue
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            stack.extend(getattr(x, f.name) for f in dataclasses.fields(x))
        elif isinstance(x, tuple | list | set | frozenset):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())


def test_bots_never_see_hidden_cards() -> None:
    for n in (3, 6, 8):
        for seed in range(40):
            spies = [_SpyBot(seed * 10 + s) for s in range(n)]
            log = play_hand_multi(spies, seed, button=seed % n)
            for seat, spy in enumerate(spies):
                own = set(log.hole_cards[seat])
                for view, hole in spy.acts:
                    assert set(hole) == own
                    for x in _reachable(view):
                        assert not isinstance(x, MultiHandState | Deck)
                        if isinstance(x, Card):
                            assert x in view.board
                (summary,) = spy.ends
                live = [not f for f in log.folded]
                showdown = bool(log.show_order)
                for s in range(n):
                    expected = log.hole_cards[s] if showdown and live[s] else None
                    assert summary.shown_cards[s] == expected
                shown = {c for h in summary.shown_cards if h for c in h}
                for x in _reachable(summary.final_view):
                    assert not isinstance(x, MultiHandState | Deck)
                    if isinstance(x, Card):
                        assert x in log.board
                assert shown <= set(FULL_DECK)


# --- Fuzz: random hands, invariants, independent side-pot reference -----------


def _random_action(rng: random.Random, state: MultiHandState) -> Action:
    legal = legal_actions(state)
    r = rng.random()
    if r < 0.12:
        return Fold()
    if (legal.can_bet or legal.can_raise) and r < 0.45:
        pick = rng.random()
        if pick < 0.3:
            size = legal.min_to
        elif pick < 0.5:
            size = legal.max_to
        else:
            size = rng.randint(legal.min_to, legal.max_to)
        return Bet(size) if legal.can_bet else Raise(size)
    return Check() if legal.can_check else Call()


def _random_stacks(rng: random.Random, n: int) -> tuple[int, ...]:
    mode = rng.random()
    if mode < 0.2:
        return (10_000,) * n
    if mode < 0.5:
        return tuple(rng.randint(1, 600) for _ in range(n))
    return tuple(rng.choice((rng.randint(1, 300), rng.randint(100, 20_000))) for _ in range(n))


def _check_invariants(state: MultiHandState, total: int) -> None:
    assert sum(state.stacks) + state.pot == total
    assert all(s >= 0 for s in state.stacks)
    assert state.street == Street.for_board(len(state.board))
    p = state.to_act
    if p is None:
        return
    assert not state.folded[p] and state.stacks[p] > 0
    legal = legal_actions(state)
    assert legal.can_check != legal.can_call
    assert not (legal.can_bet and legal.can_raise)
    if legal.can_bet or legal.can_raise:
        all_in_to = state.street_bets[p] + state.stacks[p]
        assert legal.max_to == all_in_to and legal.min_to <= legal.max_to
        current = state.current_bet
        base = max(current, state.big_blind) if current else 0
        assert legal.min_to == min(base + state.last_raise, all_in_to)
        assert legal.min_to > current


def _reference_pots(c: tuple[int, ...], live: tuple[bool, ...]) -> list[tuple[int, tuple[int, ...]]]:
    """Peel layers off the smallest remaining live contribution (independent)."""
    rem, pots = list(c), []
    while any(rem[i] > 0 for i in range(len(c)) if live[i]):
        m = min(rem[i] for i in range(len(c)) if live[i] and rem[i] > 0)
        eligible = tuple(i for i in range(len(c)) if live[i] and rem[i] > 0)
        pots.append((sum(min(r, m) for r in rem), eligible))
        rem = [r - min(r, m) for r in rem]
    assert sum(rem) == 0, "folded chips above the top live level"
    return pots


def _reference_winnings(state: MultiHandState) -> list[int]:
    n = state.n
    ranks = {s: evaluate([*state.hole_cards[s], *state.board]) for s in range(n) if not state.folded[s]}
    winnings = [0] * n
    for amount, eligible in _reference_pots(state.total_bets, state.live):
        best = max(ranks[s] for s in eligible)
        winners = sorted((s for s in eligible if ranks[s] == best), key=lambda s: (s - state.button - 1) % n)
        q, r = divmod(amount, len(winners))
        for k, s in enumerate(winners):
            winnings[s] += q + (1 if k < r else 0)
    return winnings


def _fuzz_hand(seed: int) -> MultiHandState:
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    stacks = _random_stacks(rng, n)
    button = rng.randrange(n)
    sb, bb = rng.choice(((50, 100), (50, 100), (25, 50), (1, 2), (100, 100)))
    deck = Deck(seed)
    hole = [deck.deal(2) for _ in range(n)]
    state = new_hand(stacks, button, sb, bb, hole_cards=hole)
    total = sum(stacks)
    _check_invariants(state, total)
    while not state.is_hand_over:
        if state.cards_needed:
            state = deal_board(state, deck.deal(state.cards_needed))
        else:
            state = apply_action(state, state.to_act, _random_action(rng, state))  # type: ignore[arg-type]
        _check_invariants(state, total)
    return state


def _check_settlement(state: MultiHandState) -> None:
    result = settle(state)
    n, c = state.n, state.total_bets
    assert sum(result.net) == 0
    assert sum(result.final_stacks) == sum(state.starting_stacks)
    pots = side_pots(c, state.live)
    assert sum(p.amount for p in pots) == state.pot
    for lower, upper in zip(pots, pots[1:]):
        assert set(upper.eligible) < set(lower.eligible)
    if not state.is_showdown:
        (winner,) = [s for s in range(n) if not state.folded[s]]
        assert result.winnings[winner] == state.pot
        return
    assert uncalled_bet(c) == (None, 0)  # the top level is always matched
    assert [(p.amount, p.eligible) for p in pots] == _reference_pots(c, state.live)
    assert list(result.winnings) == _reference_winnings(state)
    for i in range(n):
        assert result.winnings[i] <= sum(min(c[j], c[i]) for j in range(n))
        if state.folded[i]:
            assert result.winnings[i] == 0


def _fuzz(n_hands: int, base: int) -> dict[str, int]:
    stats = {"showdowns": 0, "side_pots": 0, "split_pots": 0}
    for k in range(n_hands):
        state = _fuzz_hand(base + k)
        _check_settlement(state)
        if state.is_showdown:
            stats["showdowns"] += 1
            result = settle(state)
            stats["side_pots"] += len(result.pots) > 1
            stats["split_pots"] += any(len(p.winners) > 1 for p in result.pots)
        if k % 50 == 0:  # replay a sample through the log
            log = make_log(state, None)
            assert make_log(replay(log), None) == log
    return stats


def test_fuzz_quick() -> None:
    stats = _fuzz(5_000, base=0)
    # The fuzz must actually exercise the interesting cases.
    assert stats["showdowns"] > 500 and stats["side_pots"] > 200 and stats["split_pots"] > 20


@pytest.mark.slow
def test_fuzz_100k() -> None:
    stats = _fuzz(100_000, base=1_000_000)
    assert stats["side_pots"] > 5_000


# --- N=2 differential test against the heads-up engine --------------------------


class _ActiveRandomBot(Bot):
    """A heads-up bot that rarely folds and mixes min, all-in and random sizes."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = state.legal
        assert legal is not None
        r = self._rng.random()
        if r < 0.06:
            return Fold()
        if (legal.can_bet or legal.can_raise) and r < 0.4:
            pick = self._rng.random()
            size = (
                legal.min_to if pick < 0.3
                else legal.max_to if pick < 0.45
                else self._rng.randint(legal.min_to, legal.max_to)
            )
            return Bet(size) if legal.can_bet else Raise(size)
        return Check() if legal.can_check else Call()


class _RecordingBot(Bot):
    """Wraps a heads-up bot and records every view and summary it receives."""

    def __init__(self, inner: Bot) -> None:
        self.inner = inner
        self.views: list[tuple[PublicState, tuple[Card, ...]]] = []
        self.summaries: list[HandSummary] = []

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        self.views.append((state, hole_cards))
        return self.inner.act(state, hole_cards)

    def end_hand(self, summary: HandSummary) -> None:
        self.summaries.append(summary)


_HU_MAKERS = {
    "random": lambda s: RandomBot(s),
    "active": lambda s: _ActiveRandomBot(s),
    "station": lambda s: CallStationBot(),
    "raise": lambda s: AlwaysRaiseBot(),
}
_MULTI_MAKERS = {
    "random": lambda s: MultiRandomBot(s),
    "station": lambda s: MultiCallStationBot(),
    "raise": lambda s: MultiAlwaysRaiseBot(),
}


def _diff_setup(k: int) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    rng = random.Random(10_000 + k)
    seed = rng.getrandbits(64)
    button = rng.randrange(2)
    mode = rng.random()
    if mode < 0.4:
        stacks = (10_000, 10_000)
    elif mode < 0.7:
        stacks = (rng.randint(1, 400), rng.randint(1, 20_000))
    else:
        stacks = (rng.randint(100, 20_000), rng.randint(100, 20_000))
    blinds = rng.choice(((50, 100), (50, 100), (25, 50), (1, 2), (100, 100)))
    return seed, button, stacks, blinds


def _assert_same_hand(hu, multi) -> None:  # type: ignore[no-untyped-def]
    assert multi.hole_cards == hu.hole_cards
    assert multi.blinds == hu.blinds
    assert multi.actions == hu.actions
    assert multi.refunds == hu.refunds
    assert multi.board == hu.board
    assert multi.result.net == hu.result.net
    assert multi.result.winnings == hu.result.winnings
    assert multi.result.final_stacks == hu.result.final_stacks
    assert multi.result.winners == hu.result.winners
    assert multi.result.ranks == hu.result.ranks
    expected_folded = tuple(hu.folded == s for s in (0, 1))
    assert multi.folded == expected_folded


@pytest.mark.parametrize("pair", [("random", "random"), ("active", "active"),
                                  ("active", "raise"), ("station", "active"),
                                  ("raise", "random")])
def test_differential_hu_bots_through_adapter(pair: tuple[str, str]) -> None:
    """Same HU bots, same seeds: every view, action, summary and net agree."""
    for k in range(600):
        seed, button, stacks, (sb, bb) = _diff_setup(k)
        hu_bots = [_RecordingBot(_HU_MAKERS[pair[i]](k * 2 + i)) for i in (0, 1)]
        mw_inner = [_RecordingBot(_HU_MAKERS[pair[i]](k * 2 + i)) for i in (0, 1)]
        hu = play_hand(hu_bots, seed, button, stacks, sb, bb)
        multi = play_hand_multi([HUBotAdapter(b) for b in mw_inner], seed, button, stacks, sb, bb)
        _assert_same_hand(hu, multi)
        for a, b in zip(hu_bots, mw_inner, strict=True):
            assert a.views == b.views
            assert a.summaries == b.summaries


@pytest.mark.parametrize("pair", [("random", "random"), ("random", "raise"),
                                  ("station", "raise"), ("raise", "raise")])
def test_differential_native_multiway_bots(pair: tuple[str, str]) -> None:
    """Native multiway baselines at N=2 reproduce the heads-up baselines."""
    for k in range(800):
        seed, button, stacks, (sb, bb) = _diff_setup(k)
        hu_bots = [_HU_MAKERS[pair[i]](k * 2 + i) for i in (0, 1)]
        mw_bots = [_MULTI_MAKERS[pair[i]](k * 2 + i) for i in (0, 1)]
        hu = play_hand(hu_bots, seed, button, stacks, sb, bb)
        multi = play_hand_multi(mw_bots, seed, button, stacks, sb, bb)
        _assert_same_hand(hu, multi)


# --- Golden hashes of heads-up hand logs (catch any HU engine drift) ------------


def _canonical_hu_log(log: hu_hand_log.HandLog) -> str:
    acts = ";".join(
        f"{int(r.street)},{r.player},{r.action},{r.chips},{int(r.all_in)}" for r in log.actions
    )
    refs = ";".join(f"{int(r.street)},{r.player},{r.chips}" for r in log.refunds)
    res = log.result
    return "|".join([
        str(log.seed), str(log.button), str(log.starting_stacks),
        " ".join(str(c) for h in log.hole_cards for c in h), str(log.blinds), acts, refs,
        " ".join(map(str, log.board)), str(log.folded), str(res.winners), str(res.ranks),
        str(res.winnings), str(res.final_stacks), str(res.net), hu_hand_log.format_log(log),
    ])


_GOLDEN = {
    "random_vs_random": (
        lambda: (RandomBot(1), RandomBot(2)),
        "41775a86acb671146330b2f687f127f5cee86f1bb81988f246f15d88b0448ea8",
    ),
    "random_vs_raise": (
        lambda: (RandomBot(3), AlwaysRaiseBot()),
        "29192a73d1c3e304df82eca163c5fdacd95169cb5f96dc3d5fa75629f13530b1",
    ),
    "station_vs_raise": (
        lambda: (CallStationBot(), AlwaysRaiseBot()),
        "eec724b7e1a48c3d00a4d22c6ef8c2e43c9cb241f540bc4eef0818e80efe4454",
    ),
    "raise_vs_random": (
        lambda: (AlwaysRaiseBot(), RandomBot(4)),
        "a1da710d68728231f50b33de694be3de974127be97f7df810c7e75810dba48a3",
    ),
}


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_heads_up_golden_hashes(name: str) -> None:
    make, expected = _GOLDEN[name]
    bots = make()
    digest = hashlib.sha256()
    for i, seed in enumerate(hand_seeds(20251123, 200)):
        log = play_hand(bots, seed, button=i % 2)
        digest.update(_canonical_hu_log(log).encode())
        digest.update(b"\n")
    assert digest.hexdigest() == expected


def test_public_view_rejects_bad_seat() -> None:
    state = new_hand((10_000,) * 3)
    with pytest.raises(ValueError):
        public_view(state, 3)


@pytest.mark.slow
def test_differential_many_seeds() -> None:
    """20k extra seeds per pairing through the adapter (slow suite)."""
    for pair in (("active", "active"), ("random", "raise"), ("active", "station")):
        for k in range(1_000, 21_000):
            seed, button, stacks, (sb, bb) = _diff_setup(k)
            hu_bots = [_HU_MAKERS[pair[i]](k * 2 + i) for i in (0, 1)]
            mw_bots = [HUBotAdapter(_HU_MAKERS[pair[i]](k * 2 + i)) for i in (0, 1)]
            hu = play_hand(hu_bots, seed, button, stacks, sb, bb)
            multi = play_hand_multi(mw_bots, seed, button, stacks, sb, bb)
            _assert_same_hand(hu, multi)


# --- Review follow-ups ----------------------------------------------------------


def test_short_big_blind_limp_calls_only_what_was_posted() -> None:
    """Deliberate deviation from TDA: callers match the 60 posted, not 100."""
    state = new_hand((10_000, 10_000, 60, 10_000), button=0)
    assert state.blinds == (0, 50, 60, 0) and state.stacks[2] == 0
    legal = legal_actions(state)
    assert legal.player == 3 and legal.call_amount == 60
    assert legal.can_raise and legal.min_to == 200  # raise base is still a full BB
    state = run(state, Call())
    assert state.street_bets == (0, 50, 60, 60)
    legal = legal_actions(state)
    assert legal.player == 0 and legal.call_amount == 60


@pytest.mark.parametrize("n", range(2, 9))
def test_positions_and_orders_for_every_button(n: int) -> None:
    for button in range(n):
        names = position_names(n, button)
        assert len(set(names)) == n and names[button] == "BTN"
        pre, flop = _action_orders(n, button)
        assert sorted(pre) == sorted(flop) == list(range(n))
        if n == 2:
            assert names[1 - button] == "BB"
            assert pre == [button, 1 - button] and flop == [1 - button, button]
            continue
        sb, bb = (button + 1) % n, (button + 2) % n
        assert names[sb] == "SB" and names[bb] == "BB"
        assert names[(bb + 1) % n] == ("BTN" if n == 3 else "UTG")  # first preflop
        right_of_button = names[(button - 1) % n]
        assert right_of_button == {3: "BB", 4: "UTG"}.get(n, "CO")
        assert pre == [(bb + 1 + k) % n for k in range(n)]
        assert flop == [(button + 1 + k) % n for k in range(n)]


def _reference_hand(seed: int) -> MultiHandState:
    """Play one random hand in the engine and the reference model side by side."""
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    stacks = ref_stacks(rng, n)
    button = rng.randrange(n)
    sb, bb = rng.choice(((50, 100), (25, 50), (1, 2), (100, 100), (40, 100), (50, 101)))
    deck, hole = ref_deal(seed, n)
    state = new_hand(stacks, button, sb, bb, hole_cards=hole)
    ref = RefTable(stacks, button, sb, bb)
    while True:
        assert state.to_act == ref.actor
        assert list(state.stacks) == ref.stack
        assert list(state.street_bets) == ref.put and list(state.total_bets) == ref.tot
        if state.is_hand_over:
            break
        if state.cards_needed:
            state = deal_board(state, deck.deal(state.cards_needed))
            ref.next_street()
            continue
        expected = ref.legal()
        got = legal_actions(state)
        assert got.can_fold
        assert RefLegal(got.player, got.can_check, got.can_call, got.call_amount, got.can_bet,
                        got.can_raise, got.min_to, got.max_to) == expected, state.history
        for bad in illegal_probes(expected):
            with pytest.raises(ValueError):
                apply_action(state, expected.player, bad)
        action = ref_pick(rng, expected)
        state = apply_action(state, expected.player, action)
        ref.act(action)
    result = settle(state)
    winnings = ref.payout(state.hole_cards, state.board)
    assert list(result.winnings) == winnings
    assert list(result.final_stacks) == [s + w for s, w in zip(ref.stack, winnings, strict=True)]
    assert sum(result.final_stacks) == sum(stacks)
    return state


def test_against_reference_model_quick() -> None:
    showdowns = sum(_reference_hand(seed).is_showdown for seed in range(3_000))
    assert showdowns > 500


@pytest.mark.slow
def test_against_reference_model_100k() -> None:
    for seed in range(500_000, 600_000):
        _reference_hand(seed)


def _golden_short_stack_logs() -> Iterator[hu_hand_log.HandLog]:
    """HU hands with random short stacks and odd blinds (frozen at HEAD)."""
    makers = (
        lambda: (RandomBot(11), RandomBot(12)),
        lambda: (RandomBot(13), AlwaysRaiseBot()),
        lambda: (CallStationBot(), RandomBot(14)),
    )
    for make in makers:
        bots = make()
        for i, seed in enumerate(hand_seeds(4242, 150)):
            rng = random.Random(seed)
            stacks = (rng.choice((rng.randint(1, 300), rng.randint(1, 20_000))), rng.randint(1, 2_000))
            sb, bb = rng.choice(((1, 2), (7, 13), (25, 50), (40, 100), (50, 101), (100, 100)))
            yield play_hand(bots, seed, button=i % 2, stacks=stacks, small_blind=sb, big_blind=bb)


_GOLDEN_SHORT = "598e22c5a047c6bfa0b6b81f8ccf47390e5dd8555ecc9581aad8feeb3134795a"


def test_heads_up_golden_hash_short_stacks_odd_blinds() -> None:
    digest = hashlib.sha256()
    for log in _golden_short_stack_logs():
        digest.update(_canonical_hu_log(log).encode())
        digest.update(b"\n")
    assert digest.hexdigest() == _GOLDEN_SHORT
