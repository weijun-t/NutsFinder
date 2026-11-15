"""Differential test: the engine against an independent model of the rules.

``RefModel`` below is a second, deliberately different implementation of the
heads-up betting rules: mutable lists instead of frozen dataclasses, no
shared helpers with ``betting.py``. Two implementations written separately
rarely share a bug, so if they agree on thousands of random hands (who acts,
what is legal, the min/max sizes, every stack after every action) that is
strong evidence both are right. When they disagree, the assertion message
shows the hand so the bug can be found.

The random hands cover the awkward cases on purpose: tiny stacks (so blinds
are all-in), very uneven stacks, odd blind sizes like 3/7, and bet sizes
that favour the exact minimum and the all-in. It also fires a one-chip-too-
small bet/raise at the engine every time and checks that it is rejected.

Adapted from the Phase 2 code reviewer's fuzzer. The normal suite plays a
few thousand hands; ``pytest -m slow`` plays 100,000.
"""

from __future__ import annotations

import random

import pytest

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    HandState,
    Street,
    apply_action,
    deal_board,
    legal_actions,
    new_hand,
)
from nutsfinder.engine.cards import Deck


class RefModel:
    """Independent, mutable model of one heads-up hand's betting.

    ``street`` here counts *betting rounds*: it moves on as soon as a round
    closes (before the cards come). The engine's street only moves when the
    cards are dealt, so the two are compared only while someone is to act.
    """

    def __init__(self, button: int, stacks: tuple[int, int], sb: int, bb: int) -> None:
        self.bb = bb
        self.bb_seat = 1 - button
        self.stack = list(stacks)
        self.street_in = [0, 0]
        self.total_in = [0, 0]
        for seat, blind in ((button, sb), (1 - button, bb)):
            posted = min(blind, self.stack[seat])
            self.stack[seat] -= posted
            self.street_in[seat] += posted
            self.total_in[seat] += posted
        self.street = 0
        self.increment = bb
        self.acted = [False, False]
        self.reopened = [True, True]
        self.folded: int | None = None
        self.over = False
        self.runout = False
        self.actor: int | None = button
        self._next_actor(first=button)

    def _can_act(self, p: int) -> bool:
        if self.stack[p] == 0:
            return False
        if self.street_in[p] < max(self.street_in):
            return True
        return not self.acted[p] and self.stack[1 - p] > 0

    def _next_actor(self, first: int) -> None:
        for p in (first, 1 - first):
            if self._can_act(p):
                self.actor = p
                return
        self._close_round()

    def _refund(self) -> None:
        diff = self.total_in[0] - self.total_in[1]
        seat = 0 if diff > 0 else 1
        for lst in (self.total_in, self.street_in):
            lst[seat] -= abs(diff)
        self.stack[seat] += abs(diff)

    def _close_round(self) -> None:
        self._refund()
        self.actor = None
        if self.street == 3:
            self.over = True
            return
        self.street += 1
        self.street_in = [0, 0]
        self.increment = self.bb
        self.acted = [False, False]
        self.reopened = [True, True]
        if self.stack[0] > 0 and self.stack[1] > 0:
            self.actor = self.bb_seat
        else:
            self.runout = True

    def legal(self) -> dict[str, object]:
        p = self.actor
        assert p is not None
        current = max(self.street_in)
        to_call = current - self.street_in[p]
        all_in = self.street_in[p] + self.stack[p]
        aggress = self.reopened[p] and all_in > current and self.stack[1 - p] > 0
        return {
            "check": to_call == 0,
            "call": to_call > 0,
            "call_amount": min(to_call, self.stack[p]) if to_call > 0 else 0,
            "bet": aggress and current == 0,
            "raise": aggress and current > 0,
            "min": min(current + self.increment, all_in) if aggress else 0,
            "max": all_in if aggress else 0,
        }

    def act(self, a: Action) -> None:
        p = self.actor
        assert p is not None
        current = max(self.street_in)
        if isinstance(a, Fold):
            self.folded, self.over, self.actor = p, True, None
            self._refund()
            return
        if isinstance(a, Check):
            target = self.street_in[p]
        elif isinstance(a, Call):
            target = min(current, self.street_in[p] + self.stack[p])
        elif isinstance(a, Bet):
            target = a.amount
        else:
            assert isinstance(a, Raise)
            target = a.to
        chips = target - self.street_in[p]
        self.stack[p] -= chips
        self.street_in[p] += chips
        self.total_in[p] += chips
        self.acted[p] = True
        self.reopened[p] = False
        if target > current:
            self.acted[1 - p] = False
            if target - current >= self.increment:
                self.increment = target - current
                self.reopened[1 - p] = True
        self._next_actor(first=1 - p)


def random_setup(rng: random.Random) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """A random (button, stacks, blinds), biased towards awkward stack sizes."""
    mode = rng.random()
    if mode < 0.4:
        stacks = (10_000, 10_000)
    elif mode < 0.7:
        stacks = (rng.randint(1, 400), rng.randint(1, 20_000))  # short stacks
    else:
        stacks = (rng.randint(1, 20_000), rng.randint(1, 20_000))
    if rng.random() < 0.5:
        stacks = (stacks[1], stacks[0])
    blinds = rng.choice([(50, 100), (1, 2), (5, 10), (100, 100), (3, 7)])
    return rng.randint(0, 1), stacks, blinds


def random_action(rng: random.Random, state: HandState) -> Action:
    """Weighted random legal action; sizes favour the min and the all-in."""
    legal = legal_actions(state)
    kinds = ["fold"] + ["check"] * 4 * legal.can_check + ["call"] * 4 * legal.can_call
    kinds += ["bet"] * 3 * legal.can_bet + ["raise"] * 3 * legal.can_raise
    kind = rng.choice(kinds)
    if kind in ("bet", "raise"):
        size = rng.choice([legal.min_to, legal.max_to, rng.randint(legal.min_to, legal.max_to)])
        return Bet(size) if kind == "bet" else Raise(size)
    return {"fold": Fold(), "check": Check(), "call": Call()}[kind]


def assert_too_small_rejected(state: HandState) -> None:
    """A bet/raise one chip under the minimum (when not all-in) must fail."""
    legal = legal_actions(state)
    if not (legal.can_bet or legal.can_raise) or not 1 < legal.min_to < legal.max_to:
        return
    bad = Bet(legal.min_to - 1) if legal.can_bet else Raise(legal.min_to - 1)
    with pytest.raises(ValueError):
        apply_action(state, legal.player, bad)


def play_one(rng: random.Random, hand_no: int) -> None:
    """Play one random hand in engine and model side by side; compare each step."""
    button, stacks, (sb, bb) = random_setup(rng)
    deck = Deck(rng.randrange(2**32))
    hole = (tuple(deck.deal(2)), tuple(deck.deal(2)))
    s = new_hand(button, stacks, sb, bb, hole_cards=hole)
    ref = RefModel(button, stacks, sb, bb)
    for _ in range(500):
        ctx = (hand_no, s.history, ref.__dict__)
        assert list(s.stacks) == ref.stack, ctx
        assert list(s.total_bets) == ref.total_in, ctx
        assert sum(s.stacks) + s.pot == sum(stacks)
        assert s.street == Street.for_board(len(s.board)), ctx
        if s.is_hand_over:
            break
        if s.cards_needed:
            assert s.to_act is None
            s = deal_board(s, deck.deal(s.cards_needed))
            continue
        assert s.to_act == ref.actor, ctx
        assert s.street == ref.street, ctx
        legal = legal_actions(s)
        got = {
            "check": legal.can_check, "call": legal.can_call, "call_amount": legal.call_amount,
            "bet": legal.can_bet, "raise": legal.can_raise, "min": legal.min_to, "max": legal.max_to,
        }
        assert got == ref.legal(), ctx
        assert_too_small_rejected(s)
        action = random_action(rng, s)
        s = apply_action(s, legal.player, action)
        ref.act(action)
    else:
        raise AssertionError(f"hand {hand_no} did not finish")
    assert (ref.over or ref.runout) and ref.folded == s.folded
    if s.folded is None:
        assert s.total_bets[0] == s.total_bets[1] and len(s.board) == 5


def run_model_check(n_hands: int, seed: int) -> None:
    rng = random.Random(seed)
    for hand_no in range(n_hands):
        play_one(rng, hand_no)


def test_engine_matches_reference_model():
    run_model_check(3_000, seed=1)


@pytest.mark.slow
def test_engine_matches_reference_model_100k():
    run_model_check(100_000, seed=2)
