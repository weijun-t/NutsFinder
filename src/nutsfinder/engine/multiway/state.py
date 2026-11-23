"""N-player (2..8) no-limit hold'em betting: the state of one hand and its rules.

This is a separate engine from :mod:`nutsfinder.engine.betting` (the
heads-up engine), which stays untouched because official results depend on
it bit for bit. It reuses the heads-up engine's *value types* unchanged
(:class:`Street`, :class:`ActionRecord`, :class:`Refund`,
:class:`LegalActions`) and the action types from ``actions.py``, so records
from both engines compare equal and heads-up bots can be adapted easily.
At two players this engine reproduces the heads-up engine exactly
(``tests/test_multiway_engine.py`` checks it action by action).

Model
-----
Fixed seats, cash game, nobody busts. Stacks are given per seat for every
hand (the runner's default resets everyone to 10,000). Because nobody joins,
leaves or busts, dead blinds, missed blinds, the dead button and "no BB twice"
never arise and are **not implemented**. Stacks may differ, so side pots
really occur.

Rules
-----
- **Blinds.** n >= 3: SB = first seat left of the button, BB = second.
  Heads-up: the button posts the SB (see :mod:`.positions`). A seat posts
  ``min(blind, stack)``; a short blind is all-in.
- **Action order.** Preflop UTG (left of the BB) acts first, and the BB keeps
  its option after limps (posting is not a choice). Postflop the first seat
  left of the button that can act goes first. Folded and all-in seats are
  skipped.
- **Street close.** A street closes when every live (not folded) player with
  chips has acted since the last bet/raise that changed what they owe and
  matches the current bet. If at most one live player has chips and nobody
  owes anything, no-one has a decision, so each remaining street opens and
  closes at once: the board runs out.
- **Current bet / calling (deliberate deviation from TDA).** The amount to
  match is the highest street bet actually in (``max(street_bets)``), as in
  the heads-up engine. So a big blind that is all-in for less than a full
  blind lowers what a limper must call: 4-handed with the BB all-in for 60,
  UTG's call costs 60, not 100. TDA's "players must call the full big
  blind" is **not** implemented, on purpose, to keep N=2 identical to the
  heads-up engine. It can only arise when some stack is smaller than the
  big blind, which never happens with the default 10,000 reset stacks. The
  minimum *raise* still uses a full big blind as its base (next rule), and
  ``test_short_big_blind_limp_calls_only_what_was_posted`` pins this.
- **Minimum bet / raise.** ``min_to = base + last_raise`` capped at the
  all-in total, where ``last_raise`` is the last *full* increment (one big
  blind at the start of each street) and ``base`` is the current bet, but at
  least one big blind once anyone has bet (so a short all-in opening bet of
  50 with a 100 big blind still needs a raise to 200, and the increment of a
  raise is measured from that base). Heads-up this never differs from the
  heads-up engine, because there a raise is impossible whenever the current
  bet is below one big blind (the other player is all-in).
- **Short all-ins and reopening (TDA Rule 44, cumulative).** An all-in for
  less than a full raise does not change ``last_raise``. Each seat remembers
  ``acted_at[i]``, the current bet right after it last acted voluntarily
  this street (``None`` before that; posting a blind is not acting). Seat i
  may bet/raise iff

  * ``acted_at[i] is None`` or ``current_bet - acted_at[i] >= last_raise``
    (several short all-ins add up to a full raise), and
  * it has chips beyond the call, and
  * at least one *other* live player still has chips.

  Example: A bets 100, B shoves 150, C shoves 210. A faces 110 >= 100 more,
  so A may re-raise; the minimum is 210 + 100 = 310. Other rooms use
  different rules; this engine follows TDA.
- **Uncalled bets** go back when a street closes or the hand ends by folds
  (see :mod:`.pots`); every return is a :class:`Refund`.
- **Fold** is always legal for the player to act (as in the heads-up
  engine). The hand ends when only one live player is left.

Streets always match the board, exactly as in the heads-up engine: a closed
street keeps its label until :func:`deal_board` brings the next cards.

Chips are conserved: ``sum(stacks) + pot == sum(starting_stacks)`` always.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    BIG_BLIND,
    SMALL_BLIND,
    ActionRecord,
    LegalActions,
    Refund,
    Street,
)
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.pots import Pot, side_pots, uncalled_bet
from nutsfinder.engine.multiway.positions import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    blind_seats,
    first_postflop,
    first_preflop,
    position_names,
)

__all__ = [
    "MultiHandState",
    "MultiPublicState",
    "new_hand",
    "legal_actions",
    "apply_action",
    "deal_board",
    "public_view",
]


@dataclass(frozen=True, slots=True)
class MultiHandState:
    """The complete state of one N-player hand, including private cards.

    Only the engine sees this; bots get a :class:`MultiPublicState`. Per-seat
    values are n-tuples indexed by seat. Fields mean the same as in the
    heads-up ``HandState`` except:

    Attributes:
        acted_at: per seat, the current bet right after its last voluntary
            action this street, or ``None`` (the TDA reopening test).
        folded: per seat, whether it has folded.
        last_aggressor: the seat that made the last bet/raise this street
            (used for the showdown order), or ``None``.
    """

    button: int
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, ...]
    blinds: tuple[int, ...]
    stacks: tuple[int, ...]
    street_bets: tuple[int, ...]
    total_bets: tuple[int, ...]
    street: Street
    board: tuple[Card, ...]
    hole_cards: tuple[tuple[Card, ...], ...]
    actor: int | None
    last_raise: int
    needs_to_act: tuple[bool, ...]
    acted_at: tuple[int | None, ...]
    folded: tuple[bool, ...]
    last_aggressor: int | None
    history: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]

    @property
    def n(self) -> int:
        """Number of seats."""
        return len(self.starting_stacks)

    @property
    def sb_seat(self) -> int:
        return blind_seats(self.n, self.button)[0]

    @property
    def bb_seat(self) -> int:
        return blind_seats(self.n, self.button)[1]

    @property
    def live(self) -> tuple[bool, ...]:
        """Per seat: still in the hand (not folded)."""
        return tuple(not f for f in self.folded)

    @property
    def n_live(self) -> int:
        return self.folded.count(False)

    @property
    def pot(self) -> int:
        return sum(self.total_bets)

    @property
    def current_bet(self) -> int:
        return max(self.street_bets)

    @property
    def cards_needed(self) -> int:
        """Board cards to deal now (positive only when a street just closed)."""
        if self.n_live <= 1 or self.actor is not None or len(self.board) == 5:
            return 0
        return Street(self.street + 1).board_size - len(self.board)

    @property
    def is_hand_over(self) -> bool:
        if self.n_live <= 1:
            return True
        return self.actor is None and len(self.board) == 5

    @property
    def is_showdown(self) -> bool:
        return self.is_hand_over and self.n_live >= 2

    @property
    def to_act(self) -> int | None:
        return self.actor

    def pots(self) -> tuple[Pot, ...]:
        """The pots as layered now. Mid-street the top pot may hold a bet that
        nobody has called yet (it is returned if the street closes unmatched)."""
        return side_pots(self.total_bets, self.live)


@dataclass(frozen=True, slots=True)
class MultiPublicState:
    """What one seat may see: the table minus every hidden card, plus legal moves.

    Attributes:
        seat: whose view this is.
        n_players: number of seats.
        sb_seat, bb_seat: the blind seats.
        positions: each seat's position name (``"UTG"``, ``"BTN"``, ...).
        pots: the layered pots right now (see :meth:`MultiHandState.pots`).
        folded: per seat, folded or not; ``in_hand`` is its negation.
        all_in: per seat, live with no chips behind.
        legal: legal actions of :attr:`to_act` (``None`` if nobody acts);
            ``legal.call_amount`` is the cost of a call, ``min_to``/``max_to``
            the bet/raise-to range.
    """

    seat: int
    n_players: int
    button: int
    sb_seat: int
    bb_seat: int
    positions: tuple[str, ...]
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, ...]
    blinds: tuple[int, ...]
    stacks: tuple[int, ...]
    street_bets: tuple[int, ...]
    total_bets: tuple[int, ...]
    pot: int
    pots: tuple[Pot, ...]
    current_bet: int
    last_raise: int
    street: Street
    board: tuple[Card, ...]
    to_act: int | None
    folded: tuple[bool, ...]
    in_hand: tuple[bool, ...]
    all_in: tuple[bool, ...]
    history: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]
    legal: LegalActions | None


# --- Starting a hand -------------------------------------------------------


def new_hand(
    stacks: Sequence[int],
    button: int = 0,
    small_blind: int = SMALL_BLIND,
    big_blind: int = BIG_BLIND,
    hole_cards: Sequence[Sequence[Card]] = (),
) -> MultiHandState:
    """Start a hand for ``len(stacks)`` seats: post blinds, give UTG the turn.

    Raises ``ValueError`` unless 2..8 positive int stacks, a valid button,
    ``1 <= small_blind <= big_blind`` and ``hole_cards`` empty or n distinct
    two-card hands. If the blinds alone settle preflop (blind all-ins), the
    street closes at once.
    """
    stacks = tuple(stacks)
    _validate_new_hand(stacks, button, small_blind, big_blind, hole_cards)
    n = len(stacks)
    sb, bb = blind_seats(n, button)
    posted = [0] * n
    posted[sb] = min(small_blind, stacks[sb])
    posted[bb] = min(big_blind, stacks[bb])
    state = MultiHandState(
        button=button,
        small_blind=small_blind,
        big_blind=big_blind,
        starting_stacks=stacks,
        blinds=tuple(posted),
        stacks=tuple(s - p for s, p in zip(stacks, posted, strict=True)),
        street_bets=tuple(posted),
        total_bets=tuple(posted),
        street=Street.PREFLOP,
        board=(),
        hole_cards=tuple(tuple(h) for h in hole_cards),
        actor=None,
        last_raise=big_blind,
        needs_to_act=(True,) * n,
        acted_at=(None,) * n,
        folded=(False,) * n,
        last_aggressor=None,
        history=(),
        refunds=(),
    )
    return _pass_turn(state, first=first_preflop(n, button))


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_seat(name: str, seat: object, n: int) -> None:
    if not (_is_int(seat) and 0 <= seat < n):  # type: ignore[operator]
        raise ValueError(f"{name} must be a seat 0..{n - 1}, got {seat!r}")


def _validate_new_hand(
    stacks: tuple[int, ...],
    button: int,
    small_blind: int,
    big_blind: int,
    hole_cards: Sequence[Sequence[Card]],
) -> None:
    n = len(stacks)
    if not MIN_PLAYERS <= n <= MAX_PLAYERS:
        raise ValueError(f"need {MIN_PLAYERS}..{MAX_PLAYERS} stacks, got {n}")
    if not all(_is_int(s) and s > 0 for s in stacks):
        raise ValueError(f"stacks must be positive ints, got {stacks!r}")
    _check_seat("button", button, n)
    if not (_is_int(small_blind) and _is_int(big_blind)):
        raise ValueError("blinds must be ints")
    if not 1 <= small_blind <= big_blind:
        raise ValueError(f"need 1 <= small blind <= big blind, got {small_blind}/{big_blind}")
    if len(hole_cards) == 0:
        return
    if len(hole_cards) != n or any(len(h) != 2 for h in hole_cards):
        raise ValueError(f"hole_cards must be {n} hands of two cards each")
    cards = [c for h in hole_cards for c in h]
    if not all(isinstance(c, Card) for c in cards) or len(set(cards)) != len(cards):
        raise ValueError(f"hole cards must be distinct cards, got {cards!r}")


# --- Legal actions ---------------------------------------------------------


def _raise_base(current: int, big_blind: int) -> int:
    """The bet level a raise is measured from: the current bet, at least 1 BB."""
    return max(current, big_blind) if current > 0 else 0


def _reopened(state: MultiHandState, seat: int) -> bool:
    """TDA cumulative test: betting is open to ``seat`` (see module doc)."""
    acted = state.acted_at[seat]
    return acted is None or state.current_bet - acted >= state.last_raise


def _others_have_chips(state: MultiHandState, seat: int) -> bool:
    """Some other live player still has chips behind."""
    return any(
        j != seat and not state.folded[j] and state.stacks[j] > 0 for j in range(state.n)
    )


def legal_actions(state: MultiHandState) -> LegalActions:
    """What the player to act may do (raises ``ValueError`` if nobody acts)."""
    p = state.to_act
    if p is None:
        raise ValueError("the hand is over" if state.is_hand_over else "nobody is to act now")
    current = state.current_bet
    to_call = current - state.street_bets[p]
    all_in_to = state.street_bets[p] + state.stacks[p]
    facing_bet = to_call > 0
    may_aggress = _reopened(state, p) and all_in_to > current and _others_have_chips(state, p)
    if may_aggress:
        min_to = min(_raise_base(current, state.big_blind) + state.last_raise, all_in_to)
        max_to = all_in_to
    else:
        min_to = max_to = 0
    return LegalActions(
        player=p,
        can_fold=True,
        can_check=not facing_bet,
        can_call=facing_bet,
        call_amount=min(to_call, state.stacks[p]) if facing_bet else 0,
        can_bet=may_aggress and current == 0,
        can_raise=may_aggress and current > 0,
        min_to=min_to,
        max_to=max_to,
    )


# --- Applying an action ----------------------------------------------------


def apply_action(state: MultiHandState, player: int, action: Action) -> MultiHandState:
    """Validate ``action`` by ``player`` and return the new state.

    Raises ``ValueError`` (old state untouched) if ``player`` is not a seat,
    nobody may act, it is not ``player``'s turn, or the action is illegal.
    """
    _check_seat("player", player, state.n)
    legal = legal_actions(state)
    if player != legal.player:
        raise ValueError(f"seat {player} acted out of turn; seat {legal.player} is to act")
    if not legal.is_legal(action):
        raise ValueError(f"illegal action {action!r} for seat {player}: {legal}")
    if isinstance(action, Fold):
        return _fold(state, player)
    return _put_in(state, player, action, _street_total_after(state, player, action))


def _street_total_after(state: MultiHandState, player: int, action: Action) -> int:
    match action:
        case Check():
            return state.street_bets[player]
        case Call():
            return min(state.current_bet, state.street_bets[player] + state.stacks[player])
        case Bet(amount=amount):
            return amount
        case Raise(to=to):
            return to
    raise AssertionError(f"unhandled action {action!r}")  # pragma: no cover


def _put_in(state: MultiHandState, player: int, action: Action, target: int) -> MultiHandState:
    """Move chips so ``player``'s street total is ``target``; pass the turn."""
    chips = target - state.street_bets[player]
    current = state.current_bet
    needs = list(state.needs_to_act)
    needs[player] = False
    last_raise, aggressor = state.last_raise, state.last_aggressor
    if target > current:
        # Any bet or raise, full or short, gives everyone else something to answer.
        for j in range(state.n):
            if j != player and not state.folded[j]:
                needs[j] = True
        aggressor = player
        increment = target - _raise_base(current, state.big_blind)
        if increment >= state.last_raise:
            last_raise = increment  # a full bet/raise sets the new minimum
    acted = list(state.acted_at)
    acted[player] = max(current, target)
    stacks = _add(state.stacks, player, -chips)
    record = ActionRecord(state.street, player, action, chips, all_in=stacks[player] == 0)
    state = replace(
        state,
        stacks=stacks,
        street_bets=_add(state.street_bets, player, chips),
        total_bets=_add(state.total_bets, player, chips),
        last_raise=last_raise,
        needs_to_act=tuple(needs),
        acted_at=tuple(acted),
        last_aggressor=aggressor,
        history=(*state.history, record),
    )
    return _pass_turn(state, first=(player + 1) % state.n)


def _fold(state: MultiHandState, player: int) -> MultiHandState:
    """``player`` gives up; the hand ends if only one live player is left."""
    record = ActionRecord(state.street, player, Fold(), 0, all_in=False)
    folded = list(state.folded)
    folded[player] = True
    needs = list(state.needs_to_act)
    needs[player] = False
    state = replace(
        state,
        folded=tuple(folded),
        needs_to_act=tuple(needs),
        actor=None,
        history=(*state.history, record),
    )
    if state.n_live == 1:
        return _return_uncalled(state)
    return _pass_turn(state, first=(player + 1) % state.n)


# --- Turn order and street transitions ------------------------------------


def _has_decision(state: MultiHandState, seat: int) -> bool:
    """Live, has chips, and either faces a bet or still has a meaningful choice."""
    if state.folded[seat] or state.stacks[seat] == 0:
        return False
    if state.street_bets[seat] < state.current_bet:
        return True
    return state.needs_to_act[seat] and _others_have_chips(state, seat)


def _pass_turn(state: MultiHandState, first: int) -> MultiHandState:
    """Give the turn to the first seat clockwise from ``first`` with a decision."""
    n = state.n
    for k in range(n):
        seat = (first + k) % n
        if _has_decision(state, seat):
            return replace(state, actor=seat)
    return replace(_return_uncalled(state), actor=None)


def _open_street(state: MultiHandState) -> MultiHandState:
    """Start the street of the just-dealt cards: reset bets; first seat left of BTN."""
    n = state.n
    state = replace(
        state,
        street=Street.for_board(len(state.board)),
        street_bets=(0,) * n,
        last_raise=state.big_blind,
        needs_to_act=tuple(not f for f in state.folded),
        acted_at=(None,) * n,
        last_aggressor=None,
    )
    return _pass_turn(state, first=first_postflop(n, state.button))


def _return_uncalled(state: MultiHandState) -> MultiHandState:
    seat, amount = uncalled_bet(state.total_bets)
    if seat is None:
        return state
    return replace(
        state,
        stacks=_add(state.stacks, seat, amount),
        street_bets=_add(state.street_bets, seat, -amount),
        total_bets=_add(state.total_bets, seat, -amount),
        refunds=(*state.refunds, Refund(state.street, seat, amount)),
    )


def _add(values: tuple[int, ...], seat: int, delta: int) -> tuple[int, ...]:
    out = list(values)
    out[seat] += delta
    return tuple(out)


# --- Dealing the board -----------------------------------------------------


def deal_board(state: MultiHandState, cards: Sequence[Card]) -> MultiHandState:
    """Add the next street's board cards (3, then 1, then 1); open the street."""
    needed = state.cards_needed
    if needed == 0:
        raise ValueError("no board cards are needed now")
    cards = tuple(cards)
    if len(cards) != needed:
        raise ValueError(f"need exactly {needed} board card(s), got {len(cards)}")
    known = [*state.board, *(c for h in state.hole_cards for c in h), *cards]
    if not all(isinstance(c, Card) for c in cards) or len(set(known)) != len(known):
        raise ValueError(f"board cards must be new, distinct cards: {cards!r}")
    return _open_street(replace(state, board=state.board + cards))


# --- What bots see ---------------------------------------------------------


def public_view(state: MultiHandState, seat: int) -> MultiPublicState:
    """The bot-safe view for ``seat``. Every field is copied explicitly so a
    private field added to the state later can never leak by accident."""
    _check_seat("seat", seat, state.n)
    to_act = state.to_act
    return MultiPublicState(
        seat=seat,
        n_players=state.n,
        button=state.button,
        sb_seat=state.sb_seat,
        bb_seat=state.bb_seat,
        positions=position_names(state.n, state.button),
        small_blind=state.small_blind,
        big_blind=state.big_blind,
        starting_stacks=state.starting_stacks,
        blinds=state.blinds,
        stacks=state.stacks,
        street_bets=state.street_bets,
        total_bets=state.total_bets,
        pot=state.pot,
        pots=state.pots(),
        current_bet=state.current_bet,
        last_raise=state.last_raise,
        street=state.street,
        board=state.board,
        to_act=to_act,
        folded=state.folded,
        in_hand=state.live,
        all_in=tuple(not f and s == 0 for f, s in zip(state.folded, state.stacks, strict=True)),
        history=state.history,
        refunds=state.refunds,
        legal=legal_actions(state) if to_act is not None else None,
    )
