"""Tests for the hand runner, hand logs, replay, and the RandomBot fuzz test.

The fuzz test plays RandomBot against RandomBot and, after every hand,
checks the chip accounting from the log alone, and that the log replays to
exactly the same hand. ``pytest -m slow`` runs it for 100,000 hands.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from nutsfinder.bots.random_bot import RandomBot
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    STARTING_STACK,
    PublicState,
    Street,
    apply_action,
    new_hand,
    public_view,
)
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.hand_log import HandLog, format_log, make_log, replay
from nutsfinder.engine.runner import hand_seeds, play_hand, play_match


class ScriptedBot(Bot):
    """Plays a fixed list of actions, in order."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        return self.actions.pop(0)


def random_bots(seed: int) -> list[Bot]:
    return [RandomBot(seed), RandomBot(seed + 1)]


# --- Running a hand -----------------------------------------------------------


def test_hole_cards_come_from_the_seeded_deck():
    log = play_hand(random_bots(0), seed=123)
    deck = Deck(123)
    assert log.hole_cards == (tuple(deck.deal(2)), tuple(deck.deal(2)))
    # Board cards come straight off the top after the hole cards.
    assert log.board == tuple(deck.deal(len(log.board)))


def test_heads_up_blind_order_through_runner():
    for button in (0, 1):
        bots: list[Bot] = [ScriptedBot([]), ScriptedBot([])]
        bots[button] = ScriptedBot([Call()] + [Check()] * 3)
        bots[1 - button] = ScriptedBot([Check()] * 4)
        log = play_hand(bots, seed=5, button=button)
        assert log.blinds[button] == 50 and log.blinds[1 - button] == 100
        assert log.actions[0].player == button  # button acts first preflop
        flop_actions = [r for r in log.actions if r.street == Street.FLOP]
        assert flop_actions[0].player == 1 - button  # big blind first postflop


def test_scripted_hand_log_and_text():
    # Seat 0 (button) raises, seat 1 calls; flop: check, bet, fold.
    bots = [ScriptedBot([Raise(300), Bet(400)]), ScriptedBot([Call(), Check(), Fold()])]
    log = play_hand(bots, seed=42)
    assert log.folded == 1
    assert log.result.winners == (0,)
    assert log.result.net == (300, -300)
    assert log.refunds[0].chips == 400 and log.refunds[0].street == Street.FLOP
    text = format_log(log)
    expected_lines = [
        "Hand (seed 42) | blinds 50/100 | seat 0 is the button",
        "Seat 0 posts small blind 50",
        "Seat 1 posts big blind 100",
        "*** PREFLOP ***",
        "Seat 0 raises to 300",
        "Seat 1 calls 200",
        f"*** FLOP *** [{' '.join(str(c) for c in log.board)}]",
        "Seat 1 checks",
        "Seat 0 bets 400",
        "Seat 1 folds",
        "Uncalled bet of 400 returned to seat 0",
        "Seat 0 wins 600",
        "Net: seat 0 +300, seat 1 -300",
    ]
    lines = text.splitlines()
    positions = [lines.index(line) for line in expected_lines]
    assert positions == sorted(positions)  # all present, in this order


def test_all_in_text_and_showdown():
    bots = [ScriptedBot([Raise(10_000)]), ScriptedBot([Call()])]
    log = play_hand(bots, seed=3)
    text = format_log(log)
    assert "Seat 0 raises to 10000 and is all-in" in text
    assert "Seat 1 calls 9900 and is all-in" in text
    assert "*** RIVER ***" in text and "*** SHOWDOWN ***" in text
    assert text.count(" shows ") == 2


def test_illegal_bot_action_raises():
    with pytest.raises(ValueError, match="seat 0"):
        play_hand([ScriptedBot([Check()]), ScriptedBot([])], seed=1)  # can't check preflop
    with pytest.raises(ValueError):
        play_hand([ScriptedBot(["call"]), ScriptedBot([])], seed=1)  # type: ignore[list-item]


def test_needs_two_bots():
    with pytest.raises(ValueError):
        play_hand([RandomBot(0)], seed=1)


# --- Reproducibility and replay -----------------------------------------------


def test_same_seed_same_bots_same_log():
    for seed in range(50):
        a = play_hand(random_bots(seed), seed=seed, button=seed % 2)
        b = play_hand(random_bots(seed), seed=seed, button=seed % 2)
        assert a == b
        assert format_log(a) == format_log(b)


def test_match_is_reproducible_and_alternates_button():
    a = play_match(random_bots(7), n_hands=40, master_seed=99)
    b = play_match(random_bots(7), n_hands=40, master_seed=99)
    assert a == b
    assert [log.button for log in a] == [i % 2 for i in range(40)]
    assert [log.seed for log in a] == hand_seeds(99, 40)
    assert all(log.starting_stacks == (STARTING_STACK, STARTING_STACK) for log in a)
    assert play_match(random_bots(7), n_hands=40, master_seed=100) != a


def test_runner_does_not_touch_global_random():
    random.seed(12345)
    before = random.getstate()
    play_match(random_bots(1), n_hands=30, master_seed=1)
    assert random.getstate() == before


def test_replay_gives_same_log():
    for log in play_match(random_bots(3), n_hands=200, master_seed=3):
        assert make_log(replay(log), log.seed) == log


def test_tampered_log_fails_to_replay():
    bots = [ScriptedBot([Raise(300), Bet(400)]), ScriptedBot([Call(), Check(), Fold()])]
    log = play_hand(bots, seed=42)
    with pytest.raises(ValueError):
        replay(replace(log, actions=log.actions[:-1]))  # truncated
    with pytest.raises(ValueError):
        replay(replace(log, actions=(*log.actions, log.actions[-1])))  # extra action
    bad = replace(log.actions[0], action=Raise(150))  # below the minimum raise
    with pytest.raises(ValueError):
        replay(replace(log, actions=(bad, *log.actions[1:])))
    with pytest.raises(ValueError):
        replay(replace(log, board=log.board[:2]))  # missing board cards


# --- Fuzz: RandomBot vs RandomBot ---------------------------------------------


def check_hand_log(log: HandLog) -> None:
    """Chip accounting from the log alone, plus replay; for any finished hand."""
    r = log.result
    assert sum(r.final_stacks) == sum(log.starting_stacks) == 2 * STARTING_STACK
    assert sum(r.net) == 0
    # Walk the log keeping running stacks: nobody ever goes below zero.
    stacks = [log.starting_stacks[s] - log.blinds[s] for s in (0, 1)]
    assert min(stacks) >= 0
    for record in log.actions:
        stacks[record.player] -= record.chips
        assert stacks[record.player] >= 0
        assert record.all_in == (stacks[record.player] == 0 and record.chips > 0)
    for refund in log.refunds:
        stacks[refund.player] += refund.chips
    put_in = [log.starting_stacks[s] - stacks[s] for s in (0, 1)]
    assert sum(r.winnings) == sum(put_in)  # the whole pot is paid out
    for seat in (0, 1):
        assert r.net[seat] == r.winnings[seat] - put_in[seat]
        assert r.net[seat] == r.final_stacks[seat] - log.starting_stacks[seat]
        assert r.final_stacks[seat] == stacks[seat] + r.winnings[seat] >= 0
    if log.folded is None:
        assert len(log.board) == 5 and put_in[0] == put_in[1]
    assert make_log(replay(log), log.seed) == log


def run_fuzz(n_hands: int, master_seed: int) -> dict[str, int]:
    """Play a RandomBot match, check every hand; return a few outcome counts."""
    counts = {"fold": 0, "showdown": 0, "split": 0, "all_in": 0}
    for log in play_match(random_bots(master_seed), n_hands, master_seed):
        check_hand_log(log)
        counts["fold" if log.folded is not None else "showdown"] += 1
        counts["split"] += len(log.result.winners) == 2
        counts["all_in"] += any(a.all_in for a in log.actions)
    return counts


def test_fuzz_random_bots():
    counts = run_fuzz(2_000, master_seed=2024)
    # Make sure the fuzz actually reached the interesting cases.
    assert all(n > 0 for n in counts.values()), counts


@pytest.mark.slow
def test_fuzz_random_bots_100k():
    counts = run_fuzz(100_000, master_seed=7)
    assert all(n > 0 for n in counts.values()), counts


# --- RandomBot itself -----------------------------------------------------------


def test_random_bot_uses_every_kind_and_size_range():
    # After a limp the BB may fold, check or raise: all three should appear,
    # including the (bad but legal) fold when checking is free.
    s = apply_action(new_hand(), 0, Call())
    view = public_view(s, 1)
    bot = RandomBot(0)
    picks = [bot.act(view, ()) for _ in range(3_000)]
    kinds = {type(a) for a in picks}
    assert kinds == {Fold, Check, Raise}
    sizes = [a.to for a in picks if isinstance(a, Raise)]
    assert min(sizes) >= 200 and max(sizes) <= 10_000
    assert min(sizes) < 1_000 and max(sizes) > 9_000  # spread over the range
    for kind in (Fold, Check, Raise):
        share = sum(isinstance(a, kind) for a in picks) / len(picks)
        assert abs(share - 1 / 3) < 0.05  # uniform over kinds


def test_random_bot_same_seed_same_choices():
    view = public_view(new_hand(), 0)
    bot1, bot2, bot3 = RandomBot(5), RandomBot(5), RandomBot(6)
    picks1 = [bot1.act(view, ()) for _ in range(100)]
    assert picks1 == [bot2.act(view, ()) for _ in range(100)]
    assert picks1 != [bot3.act(view, ()) for _ in range(100)]


def test_random_bot_refuses_to_act_out_of_turn():
    with pytest.raises(ValueError):
        RandomBot(0).act(public_view(new_hand(), 1), ())
