"""N-player hand logs: a replayable record of one hand, and its text form.

:func:`make_log` builds a :class:`MultiHandLog` from a finished state;
:func:`replay` rebuilds the hand from the log alone and must reproduce it
(``make_log(replay(log), log.seed) == log``); :func:`format_log` prints it
like a poker site's hand history, including every pot and its winners.
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import ActionRecord, Refund, Street
from nutsfinder.engine.cards import Card
from nutsfinder.engine.evaluator import HandRank, category
from nutsfinder.engine.multiway.positions import blind_seats, position_names
from nutsfinder.engine.multiway.showdown import MultiSettlement, settle, showdown_order
from nutsfinder.engine.multiway.state import (
    MultiHandState,
    apply_action,
    deal_board,
    new_hand,
)

__all__ = ["MultiHandLog", "make_log", "replay", "format_log"]


@dataclass(frozen=True, slots=True)
class MultiHandLog:
    """Everything that happened in one hand (for humans and tests, never bots).

    ``folded`` is per seat; ``show_order`` lists the live seats in showdown
    order (empty if the hand ended by folds).
    """

    seed: int | None
    n_players: int
    button: int
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, ...]
    hole_cards: tuple[tuple[Card, ...], ...]
    blinds: tuple[int, ...]
    actions: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]
    board: tuple[Card, ...]
    folded: tuple[bool, ...]
    show_order: tuple[int, ...]
    result: MultiSettlement


def make_log(state: MultiHandState, seed: int | None) -> MultiHandLog:
    """The log of a finished hand (settling it on the way)."""
    return MultiHandLog(
        seed=seed,
        n_players=state.n,
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
        show_order=showdown_order(state) if state.is_showdown else (),
        result=settle(state),
    )


def replay(log: MultiHandLog) -> MultiHandState:
    """Re-run the hand from the log alone (``ValueError`` if it does not fit)."""
    state = new_hand(
        log.starting_stacks, log.button, log.small_blind, log.big_blind, log.hole_cards
    )
    board, actions = list(log.board), list(log.actions)
    while not state.is_hand_over:
        if state.cards_needed:
            cards, board = board[: state.cards_needed], board[state.cards_needed :]
            state = deal_board(state, cards)
        elif actions:
            record = actions.pop(0)
            state = apply_action(state, record.player, record.action)
        else:
            raise ValueError("the log ends before the hand does")
    if actions or board:
        raise ValueError("the log has actions or board cards left after the hand ended")
    return state


# --- Text form --------------------------------------------------------------


def format_log(log: MultiHandLog) -> str:
    """The hand as readable text."""
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


def _header_lines(log: MultiHandLog) -> list[str]:
    names = position_names(log.n_players, log.button)
    lines = [
        f"Hand (seed {log.seed}) | {log.n_players} players | blinds "
        f"{log.small_blind}/{log.big_blind} | seat {log.button} is the button"
    ]
    for seat in range(log.n_players):
        hole = f" {_cards(log.hole_cards[seat])}" if log.hole_cards else ""
        lines.append(f"Seat {seat} ({names[seat]}): {log.starting_stacks[seat]} chips{hole}")
    sb, bb = blind_seats(log.n_players, log.button)
    for seat, name in ((sb, "small"), (bb, "big")):
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
    return category(rank).name.replace("_", " ").lower()


def _pot_name(index: int) -> str:
    return "Main pot" if index == 0 else f"Side pot {index}"


def _result_lines(log: MultiHandLog) -> list[str]:
    result = log.result
    lines: list[str] = []
    if log.show_order:
        lines.append("*** SHOWDOWN ***")
        for seat in log.show_order:
            rank = result.ranks[seat]
            assert rank is not None
            lines.append(f"Seat {seat} shows {_cards(log.hole_cards[seat])} ({_hand_name(rank)})")
    else:
        lines.append("*** RESULT ***")
    for i, pot in enumerate(result.pots):
        shares = ", ".join(
            f"seat {s} gets {c}" for s, c in zip(pot.winners, pot.shares, strict=True)
        )
        eligible = " ".join(str(s) for s in pot.eligible)
        lines.append(f"{_pot_name(i)} {pot.amount} (seats {eligible}): {shares}")
    nets = ", ".join(f"seat {s} {v:+d}" for s, v in enumerate(result.net))
    lines.append(f"Net: {nets}")
    return lines
