"""Hand logs: a complete, replayable record of one hand, and its text form.

A :class:`HandLog` holds everything needed to understand a hand afterwards:
the seed, seats, stacks, hole cards, blinds, every action (street, chips,
all-in), every refund, the board, and the payout. It is built from the final
:class:`HandState` by :func:`make_log`.

Replaying
---------
:func:`replay` rebuilds the hand from the log alone (no seed, no bots): it
starts a new hand with the logged stacks and hole cards, then applies the
logged actions in order, dealing the logged board cards whenever the engine
asks for them. Because the engine is deterministic, the result must be the
same hand; ``make_log(replay(log), log.seed) == log`` is checked in the
tests. A log that is edited or truncated fails to replay (``ValueError``).

Text form
---------
:func:`format_log` prints the hand like an online poker site's hand
history::

    Hand (seed 42) | blinds 50/100 | seat 0 is the button
    Seat 0 (button): 10000 chips [As Ks]
    Seat 1: 10000 chips [9h 8h]
    Seat 0 posts small blind 50
    Seat 1 posts big blind 100
    *** PREFLOP ***
    Seat 0 raises to 300
    Seat 1 calls 200
    *** FLOP *** [Kh 7h 2s]
    ...
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    ActionRecord,
    HandState,
    Refund,
    Street,
    apply_action,
    deal_board,
    new_hand,
)
from nutsfinder.engine.cards import Card
from nutsfinder.engine.evaluator import HandRank, category
from nutsfinder.engine.showdown import Settlement, settle

__all__ = ["HandLog", "make_log", "replay", "format_log"]


@dataclass(frozen=True, slots=True)
class HandLog:
    """Everything that happened in one hand. Per-seat values indexed by seat.

    Attributes:
        seed: the seed the deck was shuffled with (``None`` if not seeded).
        button: the seat on the button.
        small_blind, big_blind: blind sizes.
        starting_stacks: stacks before the blinds.
        hole_cards: both seats' hole cards (the log is for humans and tests,
            never for bots).
        blinds: chips each seat actually posted as a blind.
        actions: every player action, in order.
        refunds: uncalled chips returned, in order.
        board: the board cards that were dealt (0 to 5).
        folded: the seat that folded, or ``None`` for a showdown.
        result: the payout, including showdown ranks and net per seat.
    """

    seed: int | None
    button: int
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, int]
    hole_cards: tuple[tuple[Card, ...], ...]
    blinds: tuple[int, int]
    actions: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]
    board: tuple[Card, ...]
    folded: int | None
    result: Settlement


def make_log(state: HandState, seed: int | None) -> HandLog:
    """Build the log of a finished hand (settling it on the way)."""
    return HandLog(
        seed=seed,
        button=state.button,
        small_blind=state.small_blind,
        big_blind=state.big_blind,
        starting_stacks=state.starting_stacks,
        hole_cards=state.hole_cards,
        blinds=state.blinds,
        actions=state.history,
        refunds=state.refunds,
        board=state.board,
        folded=state.folded,
        result=settle(state),
    )


def replay(log: HandLog) -> HandState:
    """Re-run the hand from the log alone and return the final state.

    Raises ``ValueError`` if the logged actions are illegal, are by the wrong
    seat, run out before the hand ends, or are left over after it ends (and
    likewise for the board cards).
    """
    state = new_hand(
        log.button, log.starting_stacks, log.small_blind, log.big_blind, log.hole_cards
    )
    board = list(log.board)
    actions = list(log.actions)
    while not state.is_hand_over:
        if state.cards_needed:
            cards, board = board[: state.cards_needed], board[state.cards_needed :]
            state = deal_board(state, cards)  # raises if the log ran out of cards
        elif actions:
            record = actions.pop(0)
            state = apply_action(state, record.player, record.action)
        else:
            raise ValueError("the log ends before the hand does")
    if actions or board:
        raise ValueError("the log has actions or board cards left after the hand ended")
    return state


# --- Text form --------------------------------------------------------------


def format_log(log: HandLog) -> str:
    """The hand as readable text, like a poker site's hand history."""
    lines = _header_lines(log)
    last_street = Street.for_board(len(log.board))
    for street in Street:
        if street > last_street:
            break
        lines.append(_street_header(street, log.board))
        lines += [_action_line(r) for r in log.actions if r.street == street]
        lines += [
            f"Uncalled bet of {r.chips} returned to seat {r.player}"
            for r in log.refunds
            if r.street == street
        ]
    lines += _result_lines(log)
    return "\n".join(lines)


def _cards(cards: tuple[Card, ...]) -> str:
    return "[" + " ".join(str(c) for c in cards) + "]"


def _header_lines(log: HandLog) -> list[str]:
    lines = [
        f"Hand (seed {log.seed}) | blinds {log.small_blind}/{log.big_blind}"
        f" | seat {log.button} is the button"
    ]
    for seat in (0, 1):
        role = " (button)" if seat == log.button else ""
        hole = f" {_cards(log.hole_cards[seat])}" if log.hole_cards else ""
        lines.append(f"Seat {seat}{role}: {log.starting_stacks[seat]} chips{hole}")
    for seat, name in ((log.button, "small"), (1 - log.button, "big")):
        all_in = " (all-in)" if log.blinds[seat] == log.starting_stacks[seat] else ""
        lines.append(f"Seat {seat} posts {name} blind {log.blinds[seat]}{all_in}")
    return lines


def _street_header(street: Street, board: tuple[Card, ...]) -> str:
    if street == Street.PREFLOP:
        return "*** PREFLOP ***"
    return f"*** {street.name} *** {_cards(board[: street.board_size])}"


def _action_line(record: ActionRecord) -> str:
    match record.action:
        case Fold():
            text = "folds"
        case Check():
            text = "checks"
        case Call():
            text = f"calls {record.chips}"
        case Bet(amount=amount):
            text = f"bets {amount}"
        case Raise(to=to):
            text = f"raises to {to}"
        case _:  # pragma: no cover
            text = str(record.action)
    tail = " and is all-in" if record.all_in else ""
    return f"Seat {record.player} {text}{tail}"


def _hand_name(rank: HandRank) -> str:
    """e.g. ``"full house"`` for a full-house rank."""
    return category(rank).name.replace("_", " ").lower()


def _result_lines(log: HandLog) -> list[str]:
    result = log.result
    lines: list[str] = []
    if log.folded is None:
        lines.append("*** SHOWDOWN ***")
        for seat in (0, 1):
            rank = result.ranks[seat]
            assert rank is not None
            lines.append(f"Seat {seat} shows {_cards(log.hole_cards[seat])} ({_hand_name(rank)})")
    else:
        lines.append("*** RESULT ***")
    if len(result.winners) == 2:
        lines.append(
            f"Split pot: seat 0 gets {result.winnings[0]}, seat 1 gets {result.winnings[1]}"
        )
    else:
        winner = result.winners[0]
        lines.append(f"Seat {winner} wins {result.winnings[winner]}")
    lines.append(f"Net: seat 0 {result.net[0]:+d}, seat 1 {result.net[1]:+d}")
    return lines
