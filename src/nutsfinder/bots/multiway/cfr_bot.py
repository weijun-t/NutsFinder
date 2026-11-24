"""MultiCFRBot: the heads-up blueprint where a hand really is heads-up, MultiEquityBot elsewhere. A heuristic.

The HU blueprint (:class:`~nutsfinder.bots.cfr_bot.CFRBot`) is an
approximate equilibrium of the **two-player** game. Its preflop ranges are
heads-up ranges (the small blind opens most hands) and its postflop play is
a response to those ranges. With 3+ players there is neither a matching
tree nor any equilibrium guarantee (docs/phase7_multiway_research.md,
section 2). So it is used only where the spot looks like the game it was
solved for, and every decision records which logic decided
(:attr:`MultiCFRBot.last_mode`, :attr:`MultiCFRBot.mode_counts`, and the
``mode`` of the debug record).

When the blueprint plays (``mode = "hu-blueprint"``)
----------------------------------------------------
1. **Two seats at the table.** The multiway engine reproduces the HU engine
   exactly at N = 2, so the view is converted
   (:func:`~nutsfinder.engine.multiway.bots.to_hu_view`) and CFRBot plays
   unchanged (the same actions as ``HUBotAdapter(CFRBot(...))``, tested).
2. **3+ seats, a hand that is heads-up preflop between some player X and
   the big blind, X not the small blind:** every other player's only action
   was a preflop fold, and exactly X and the BB are left. Then the hand is
   re-expressed as a HU hand: X is the HU button (tree seat 0), the BB is
   the BB, the stacks are theirs, and the history is X's and the BB's own
   actions, replayed in the HU engine (so the walk, translations and sizes
   use a real HU state). This is the doc's "BTN vs BB after the small blind
   folds" case, generalised to any X that the others folded to.

Why X vs the BB and not other pairs: the tree's seat 0 acts *first*
preflop and *last* postflop. X (any seat from UTG to the button) does both
against the BB. The small blind acts first on every street, so SB vs BB
has the wrong postflop order for the tree; two players neither of whom is
the BB do not match the tree's preflop structure (nobody holds the BB's
option). Those hands use the fallback.

What is knowingly wrong even then (it is a heuristic, measure it, do not trust it):

- X's real range is a multiway range (it opened into players who then
  folded), much tighter than the HU small blind's; the blueprint's
  postflop strategy answers HU ranges.
- The re-expressed HU pot is missing the folded players' dead chips (the
  small blind's 50 when X is not the SB). They cannot simply be added: the
  HU state is rebuilt by the HU engine, whose pot is the two players'
  contributions, and CFRBot replays that history (``view_before_action``)
  to translate bets, so a patched ``pot`` field would disagree with its own
  replay. The effect, kept and documented: in a single-raised pot (X opens
  to 250, BB calls) the real pot is 550 and the HU pot 500, so the
  blueprint's pot-fraction bets are about 9% smaller than intended in real
  pot terms, its translation reads our opponents' real bets as about 10%
  larger, and its equity rule prices calls against a 10% smaller pot
  (slightly too tight). The bucket and the abstract node do not depend on
  the pot. The gap shrinks as the pot grows.
- X's *first* decision is never heads-up (the players behind have not
  folded yet), so X opens with the fallback; the blueprint takes over once
  the others have folded, reading X's real open as a translated bet.

Everything else (``mode = "equity-fallback"``): :class:`MultiEquityBot`.

Mode accounting: :attr:`MultiCFRBot.mode_counts` counts ``hu-blueprint``
only when the blueprint's own sampled action was played; CFRBot's equity
rule overrides and off-tree fallbacks inside heads-up spots get their own
labels (:func:`hu_mode_of`).

Actions coming back from the HU view are mapped onto the real legal
actions: fold where a check is free becomes a check, a bet/raise is
clamped to the real ``[min_to, max_to]`` (street totals agree between the
two views), and an aggressive action where the real rules forbid one
becomes a call or check.

Randomness: the CFRBot generator is seeded with ``seed``, the fallback's
with a value derived from it; both restart on :meth:`reseed`.
Requires a trained blueprint (``models/holdem_official/latest.npz``).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path

from nutsfinder.abstraction.blueprint import Blueprint
from nutsfinder.bots.cfr_bot import CFRBot
from nutsfinder.bots.common import aggressive_action
from nutsfinder.bots.multiway.common import check_turn
from nutsfinder.bots.multiway.equity_bot import MultiEquityBot
from nutsfinder.engine import betting as hu
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState, Street
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.bots import to_hu_view
from nutsfinder.engine.multiway.state import MultiPublicState

__all__ = ["MultiCFRBot", "heads_up_subgame", "hu_mode_of", "HU_MODE", "FALLBACK_MODE"]

log = logging.getLogger(__name__)

HU_MODE = "hu-blueprint"
FALLBACK_MODE = "equity-fallback"


def hu_mode_of(d: object) -> str:
    """The mode label of a CFRBot decision record in a heads-up spot.

    ``hu-blueprint`` only when the blueprint's own sampled action was played;
    ``hu-blueprint:equity-rule`` when CFRBot overrode it with its equity rule
    (soft node, or an abstract check facing a real bet); ``hu-blueprint:passive-ahead`` /
    ``hu-blueprint:terminal`` when CFRBot was off the tree and used its fallback.
    """
    mode = getattr(d, "mode", "")
    if mode != "blueprint":
        return f"{HU_MODE}:{mode}"
    if any("equity rule" in n for n in getattr(d, "notes", ())):
        return f"{HU_MODE}:equity-rule"
    return HU_MODE


def _derive(seed: int | None, salt: int) -> int | None:
    return None if seed is None else (seed * 1_000_003 + salt) % (2**63)


def heads_up_subgame(view: MultiPublicState) -> tuple[PublicState | None, str]:
    """``(HU view for view.seat, reason)``; the view is ``None`` when the hand is not HU-blueprint material.

    See the module docstring for the rule. Uses public information only.
    """
    if view.n_players == 2:
        return to_hu_view(view), "two seats"
    live = [s for s in range(view.n_players) if view.in_hand[s]]
    if len(live) != 2:
        return None, f"{len(live)} players live"
    if view.bb_seat not in live:
        return None, "the big blind folded: no HU tree for this pair"
    x = live[0] if live[1] == view.bb_seat else live[1]
    if x == view.sb_seat:
        return None, "SB vs BB: the SB acts first postflop, unlike the HU tree's button"
    for rec in view.history:
        if rec.player not in live and not (isinstance(rec.action, Fold) and rec.street == Street.PREFLOP):
            return None, f"seat {rec.player} acted voluntarily before folding"
    seat_of = {x: 0, view.bb_seat: 1}
    try:
        state = hu.new_hand(0, (view.starting_stacks[x], view.starting_stacks[view.bb_seat]),
                            view.small_blind, view.big_blind)
        for rec in view.history:
            if rec.player not in seat_of:
                continue
            state = _deal_to(state, view.board, rec.street)
            state = hu.apply_action(state, seat_of[rec.player], rec.action)
        state = _deal_to(state, view.board, view.street)
    except ValueError as err:
        return None, f"HU replay failed: {err}"
    if state.board != view.board or state.to_act != seat_of.get(view.seat):
        return None, "HU replay out of step with the real hand"
    return hu.public_view(state, seat_of[view.seat]), f"seat {x} vs BB {view.bb_seat}, others folded preflop"


def _deal_to(state: hu.HandState, board: tuple[Card, ...], street: Street) -> hu.HandState:
    while state.cards_needed and len(state.board) < street.board_size:
        n = len(state.board)
        state = hu.deal_board(state, board[n : n + state.cards_needed])
    return state


def _to_real(action: Action, view: MultiPublicState) -> Action:
    """Map an action chosen on the HU view onto the real legal actions."""
    legal = check_turn(view)
    match action:
        case Fold():
            return Check() if legal.can_check else Fold()
        case Check() | Call():
            return Check() if legal.can_check else Call()
        case Bet(amount=to) | Raise(to=to):
            if legal.can_bet or legal.can_raise:
                return aggressive_action(legal, min(max(to, legal.min_to), legal.max_to))
            return Check() if legal.can_check else Call()
    raise AssertionError(f"unknown action {action!r}")  # pragma: no cover


class MultiCFRBot(MultiBot):
    """HU blueprint in heads-up-preflop spots, MultiEquityBot elsewhere (see the module doc).

    Args:
        model: checkpoint path or :class:`Blueprint` (share one Blueprint across seats to save memory).
        seed: seeds the CFRBot generator; the fallback gets a derived seed.
        debug: record decisions (``CFRDecision`` with mode ``hu-blueprint:<cfr mode>`` or
            ``MultiDecision`` with mode ``equity-fallback``) in :attr:`last_decision`.
        time_budget: per decision, passed to both parts.
        fallback_samples: Monte Carlo deals of the fallback.
    """

    def __init__(self, model: str | Path | Blueprint, seed: int | None = 0, *, debug: bool = False,
                 time_budget: float | None = None, fallback_samples: int = 600) -> None:
        # The CFR record is always built (it draws no random numbers), so the mode of every
        # decision is known; it is only exposed in last_decision when debug=True.
        self.cfr = CFRBot(model, seed=seed, debug=True, time_budget=time_budget)
        self.fallback = MultiEquityBot(seed=_derive(seed, 1), samples=fallback_samples,
                                       time_budget=time_budget, debug=debug)
        self.debug = debug
        self.last_mode: str | None = None
        self.last_reason: str = ""
        self.mode_counts: Counter[str] = Counter()
        self.last_decision: object | None = None

    def reseed(self, seed: int) -> None:
        self.cfr.reseed(seed)
        self.fallback.reseed(_derive(seed, 1))  # type: ignore[arg-type]

    def end_hand(self, summary: MultiHandSummary) -> None:
        # CFRBot.end_hand only forgets the hand's walks and never reads its argument.
        self.cfr.end_hand(summary)  # type: ignore[arg-type]
        self.fallback.end_hand(summary)

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        check_turn(view)
        hu_view, reason = heads_up_subgame(view)
        self.last_reason = reason
        if hu_view is not None:
            action = _to_real(self.cfr.act(hu_view, hole_cards), view)
            d = self.cfr.last_decision
            assert d is not None
            self.last_mode = hu_mode_of(d)
            if self.debug:
                self.last_decision = dataclasses.replace(
                    d, mode=self.last_mode, action=action, notes=d.notes + (f"heads-up spot: {reason}",))
        else:
            self.last_mode = FALLBACK_MODE
            action = self.fallback.act(view, hole_cards)
            if self.debug and self.fallback.last_decision is not None:
                d = self.fallback.last_decision
                self.last_decision = dataclasses.replace(
                    d, mode=FALLBACK_MODE, notes=d.notes + (f"not a heads-up blueprint spot: {reason}",))
        self.mode_counts[self.last_mode] += 1
        return action

    def __repr__(self) -> str:
        return f"MultiCFRBot({self.cfr.blueprint.describe()}, modes={dict(self.mode_counts)})"
