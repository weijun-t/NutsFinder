"""An independent reference model of N-player NLHE betting and payouts.

Written from the rules, not from ``engine/multiway/state.py`` (adapted from
the code reviewer's oracle). It keeps its own mutable state and represents
the reopening rule differently: per player it stores ``acted_level`` (the
bet level when it last acted), ``acted_inc`` (the full increment then) and
``full_since`` (a full raise happened since it acted). Uncalled chips are
returned from street bets, and the payout is computed on its own.

Used by ``tests/test_multiway_engine.py`` to compare every legal-action
field at every decision, probe illegal actions, and compare payouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.evaluator import evaluate

__all__ = ["RefLegal", "RefTable", "ref_stacks", "ref_pick", "illegal_probes", "ref_deal"]


@dataclass(frozen=True)
class RefLegal:
    player: int
    can_check: bool
    can_call: bool
    call_amount: int
    can_bet: bool
    can_raise: bool
    min_to: int
    max_to: int


class RefTable:
    """Mutable reference state for one hand."""

    def __init__(self, stacks: list[int], button: int, small_blind: int, big_blind: int) -> None:
        n = self.n = len(stacks)
        self.bb, self.button = big_blind, button
        self.stack = list(stacks)
        self.put = [0] * n
        self.tot = [0] * n
        self.folded = [False] * n
        sb_seat, bb_seat = (button, 1 - button) if n == 2 else ((button + 1) % n, (button + 2) % n)
        for seat, amount in ((sb_seat, small_blind), (bb_seat, big_blind)):
            a = min(amount, self.stack[seat])
            self.stack[seat] -= a
            self.put[seat] += a
            self.tot[seat] += a
        self._start_round(first=(bb_seat + 1) % n)

    # --- rounds and turn order

    def _start_round(self, first: int) -> None:
        n = self.n
        self.level = max(self.put)
        self.inc = self.bb
        self.acted = [False] * n
        self.acted_level: list[int | None] = [None] * n
        self.acted_inc: list[int | None] = [None] * n
        self.full_since = [False] * n
        self.actor = self._next(first)
        if self.actor is None:
            self._refund()

    def _chip_holders(self, excl: int) -> list[int]:
        return [j for j in range(self.n) if not self.folded[j] and self.stack[j] > 0 and j != excl]

    def _wants(self, i: int) -> bool:
        if self.folded[i] or self.stack[i] == 0:
            return False
        if self.put[i] < self.level:
            return True
        return not self.acted[i] and len(self._chip_holders(i)) > 0

    def _next(self, first: int) -> int | None:
        for k in range(self.n):
            i = (first + k) % self.n
            if self._wants(i):
                return i
        return None

    def live(self) -> list[int]:
        return [i for i in range(self.n) if not self.folded[i]]

    # --- decisions

    def legal(self) -> RefLegal:
        i = self.actor
        assert i is not None
        owe = self.level - self.put[i]
        all_in = self.put[i] + self.stack[i]
        acted_level, acted_inc = self.acted_level[i], self.acted_inc[i]
        if acted_level is None or acted_inc is None or self.full_since[i]:
            is_open = True
        else:
            is_open = self.level - acted_level >= acted_inc
        agg = is_open and all_in > self.level and len(self._chip_holders(i)) > 0
        if agg:
            mn = self.inc if self.level == 0 else max(self.level, self.bb) + self.inc
            mn, mx = min(mn, all_in), all_in
        else:
            mn = mx = 0
        return RefLegal(
            player=i,
            can_check=owe <= 0,
            can_call=owe > 0,
            call_amount=min(owe, self.stack[i]) if owe > 0 else 0,
            can_bet=agg and self.level == 0,
            can_raise=agg and self.level > 0,
            min_to=mn,
            max_to=mx,
        )

    def act(self, action: Action) -> None:
        i = self.actor
        assert i is not None
        if isinstance(action, Fold):
            self.folded[i] = True
            if len(self.live()) == 1:
                self._refund()
                self.actor = None
                return
        else:
            self._put_in(i, self._target(i, action))
        self.actor = self._next((i + 1) % self.n)
        if self.actor is None:
            self._refund()

    def _target(self, i: int, action: Action) -> int:
        if isinstance(action, Check):
            return self.put[i]
        if isinstance(action, Call):
            return min(self.level, self.put[i] + self.stack[i])
        if isinstance(action, Bet):
            return action.amount
        assert isinstance(action, Raise)
        return action.to

    def _put_in(self, i: int, target: int) -> None:
        d = target - self.put[i]
        self.stack[i] -= d
        self.put[i] += d
        self.tot[i] += d
        if target > self.level:
            base = max(self.level, self.bb) if self.level > 0 else 0
            if target - base >= self.inc:
                self.inc = target - base
                for j in range(self.n):
                    if j != i:
                        self.full_since[j] = True
            for j in range(self.n):
                if j != i:
                    self.acted[j] = False
            self.level = target
        self.acted[i] = True
        self.acted_level[i] = self.level
        self.acted_inc[i] = self.inc
        self.full_since[i] = False

    def _refund(self) -> None:
        order = sorted(range(self.n), key=lambda j: -self.put[j])
        d = self.put[order[0]] - self.put[order[1]]
        if d > 0:
            j = order[0]
            self.put[j] -= d
            self.tot[j] -= d
            self.stack[j] += d

    def next_street(self) -> None:
        self.put = [0] * self.n
        self._start_round(first=(self.button + 1) % self.n)

    # --- payout

    def payout(self, hole: tuple[tuple[Card, ...], ...], board: tuple[Card, ...]) -> list[int]:
        n, live = self.n, self.live()
        win = [0] * n
        if len(live) == 1:
            win[live[0]] = sum(self.tot)
            return win
        rank = {i: evaluate([*hole[i], *board]) for i in live}
        c, prev = list(self.tot), 0
        for cap in sorted({c[i] for i in live}):
            amount = sum(max(0, min(x, cap) - prev) for x in c)
            eligible = [i for i in live if c[i] >= cap]
            best = max(rank[i] for i in eligible)
            winners = sorted(
                (i for i in eligible if rank[i] == best), key=lambda i: (i - self.button - 1) % n
            )
            q, r = divmod(amount, len(winners))
            for k, i in enumerate(winners):
                win[i] += q + (1 if k < r else 0)
            prev = cap
        assert sum(win) == sum(c), (win, c)
        return win


# --- random hand generation -------------------------------------------------


def ref_stacks(rng: random.Random, n: int) -> list[int]:
    m = rng.random()
    if m < 0.15:
        return [10_000] * n
    if m < 0.45:
        return [rng.randint(1, 700) for _ in range(n)]
    if m < 0.7:
        return [
            rng.choice([rng.randint(1, 250), rng.randint(150, 1200), rng.randint(1000, 20_000)])
            for _ in range(n)
        ]
    return [rng.randint(80, 600) for _ in range(n)]


def ref_deal(seed: int, n: int) -> tuple[Deck, tuple[tuple[Card, ...], ...]]:
    deck = Deck(seed ^ 0x5EED)
    return deck, tuple(tuple(deck.deal(2)) for _ in range(n))


def ref_pick(rng: random.Random, legal: RefLegal, fold_p: float = 0.1) -> Action:
    options: list[Action] = [Fold()] if rng.random() < fold_p else []
    options.append(Check() if legal.can_check else Call())
    if legal.can_bet or legal.can_raise:
        make = Bet if legal.can_bet else Raise
        sizes = {
            legal.min_to,
            legal.max_to,
            rng.randint(legal.min_to, legal.max_to),
            min(legal.max_to, legal.min_to + rng.randint(0, 30)),
        }
        for size in sorted(sizes):
            options += [make(size), make(size)]
    return rng.choice(options)


def illegal_probes(legal: RefLegal) -> list[Action]:
    """Actions that must be rejected in this spot (3 to 6 of them)."""
    probes: list[Action] = [Raise(legal.max_to + 1)]
    if legal.min_to > 1:
        probes.append(Bet(legal.min_to - 1))
    if not (legal.can_bet or legal.can_raise):
        probes += [Bet(10**6), Raise(10**6), Bet(1), Raise(legal.call_amount + 1)]
    probes.append(Call() if legal.can_check else Check())
    if legal.can_raise and legal.min_to < legal.max_to:
        probes.append(Raise(legal.min_to - 1))
    if legal.can_bet:
        probes.append(Raise(legal.min_to))  # a raise when nothing is bet
    return probes
