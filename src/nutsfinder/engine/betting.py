"""Heads-up no-limit hold'em betting: the state of one hand and its rules.

This module knows about chips, turns and streets. It does *not* shuffle or
deal: whoever runs the hand (see ``runner.py``) owns the deck, and hands the
board cards to :func:`deal_board` when the state asks for them.

The rules, in one place
-----------------------
Seats are ``0`` and ``1``. ``button`` is the seat on the button.

- **Blinds.** Heads-up, the button posts the small blind (default 50) and the
  other player posts the big blind (default 100). A player whose stack is
  smaller than their blind posts everything they have and is all-in.
- **Turn order.** Preflop the button (small blind) acts first, because the
  blinds count as the first two "actions". On the flop, turn and river the
  big blind acts first and the button acts last. Acting last is the
  button's advantage.
- **The big blind's option.** If the button just calls preflop ("limps"), the
  bets are equal but the big blind has not *chosen* anything yet (posting a
  blind is forced), so the big blind may still check or raise.
- **Amounts.** Chips are integers. ``Bet(x)`` and ``Raise(to=x)`` both mean
  "my total on this street becomes x" (see ``actions.py``).
- **Minimum raise.** ``min raise-to = current bet + last raise increment``.
  The *increment* is how much the last full bet or raise added on top of the
  bet before it. At the start of every street the increment is one big blind,
  so:

  * preflop the big blind is treated as an opening bet of 100 (increment
    100), so the smallest raise is to 200;
  * postflop the smallest first bet is 100 (0 + 100);
  * raise to 300 (increment 200)  -> next min raise is to 300 + 200 = 500;
  * 3-bet to 1000 (increment 700) -> next min raise is to 1000 + 700 = 1700.

  Any whole number between the minimum and the player's all-in is legal.
- **All-ins.** Going all-in is always allowed, even for less than a minimum
  bet or raise. An all-in raise that is smaller than a full raise ("short
  all-in") does **not** change the raise increment and does **not** reopen
  the betting: a player who has already acted may only call or fold against
  it. A full-size all-in raise does both, like any other raise.

  Heads-up note: once someone is all-in their opponent can never raise
  anyway (there is nobody left to raise against), so heads-up the reopening
  rule never changes which actions are legal. We still track it properly
  (``raise_allowed``) so the bookkeeping is right and would carry over to a
  multi-player table later.
- **Calling for less.** If a player cannot afford a full call, ``Call`` puts
  in their whole stack.
- **Uncalled bets.** If one player has put in more than the other could or
  did match (an all-in for less, or a bet that was folded to), the unmatched
  part is returned to its owner when the street closes. Heads-up there is
  only ever one pot, so no side pots are needed. Every return is recorded as
  a :class:`Refund`. See :func:`uncalled_bet` and :func:`contested_pot`.
- **Fold** ends the hand at once; the other player wins the pot. Folding is
  always legal for the player to act, even when checking is free. That fold
  is never better than checking, but it is allowed by the real rules, so the
  engine allows it and leaves it to the bot not to do it.

Streets always match the board
------------------------------
``state.street`` is decided by the board alone: PREFLOP while there are 0
board cards, FLOP with 3, TURN with 4, RIVER with 5. When a street's betting
is finished, the street is *closed* (``to_act`` becomes ``None``) but the
label stays until :func:`deal_board` adds the next street's cards; only then
does the next street *open* (bets reset, big blind to act). So "street ==
TURN" always means "the turn card is out", including during an all-in
run-out, where streets open and immediately close because nobody can bet.

How a hand flows
----------------
::

    state = new_hand(button=0)                 # blinds posted, button to act
    while not state.is_hand_over:
        if state.cards_needed:                 # a street just closed
            state = deal_board(state, deck.deal(state.cards_needed))
        else:
            state = apply_action(state, state.to_act, some_action)

When both players are all-in (or one is all-in and the other has called),
betting is over for the hand: ``to_act`` stays ``None`` and the loop above
simply deals the remaining streets one by one until there are five board
cards, then the hand is over and goes to showdown.

Rebuilding the chips from the record
------------------------------------
Every chip movement is recorded, so the stacks and pot can be recomputed
from the record alone (``tests/test_betting.py`` checks this)::

    stacks[s]     = starting_stacks[s] - blinds[s]
                    - sum(r.chips for r in history if r.player == s)
                    + sum(r.chips for r in refunds if r.player == s)
    total_bets[s] = starting_stacks[s] - stacks[s]

Why the state is immutable
--------------------------
:class:`HandState` is a frozen dataclass, and :func:`apply_action` returns a
*new* state instead of changing the old one. That costs a few microseconds
per action, which is nothing next to the evaluator, and buys a lot:

- a bot can be handed a state and cannot cheat by editing it;
- tests can keep the state from before an action and compare;
- later, game-tree search and CFR (Phase 6) need to try an action, look at
  the result and back up; with immutable states "backing up" is just using
  the old object again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.cards import Card

__all__ = [
    "SMALL_BLIND",
    "BIG_BLIND",
    "STARTING_STACK",
    "Street",
    "ActionRecord",
    "Refund",
    "LegalActions",
    "HandState",
    "PublicState",
    "new_hand",
    "legal_actions",
    "apply_action",
    "deal_board",
    "public_view",
    "uncalled_bet",
    "contested_pot",
    "check_seat",
]

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000
"""Every hand starts with both players at 10,000 chips (100 big blinds)."""


class Street(IntEnum):
    """The four betting rounds, in order."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    @property
    def board_size(self) -> int:
        """How many board cards are out during this street: 0, 3, 4 or 5."""
        return (0, 3, 4, 5)[self]

    @classmethod
    def for_board(cls, n_cards: int) -> Street:
        """The street that ``n_cards`` board cards belong to (0, 3, 4 or 5)."""
        try:
            return cls((0, 3, 4, 5).index(n_cards))
        except ValueError:
            raise ValueError(f"a board has 0, 3, 4 or 5 cards, not {n_cards}") from None


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One entry of the action history.

    Attributes:
        street: the street the action was taken on.
        player: the seat (0 or 1) that acted.
        action: what they did.
        chips: how many chips moved from their stack into the pot (0 for a
            check or fold; for a call it is the actual cost, which is less than
            the bet if the caller was all-in for less).
        all_in: the action put the player's last chip in (their stack is now 0).
    """

    street: Street
    player: int
    action: Action
    chips: int
    all_in: bool

    def __str__(self) -> str:
        tail = " (all-in)" if self.all_in else ""
        return f"{self.street.name.lower()}: seat {self.player} {self.action}{tail}"


@dataclass(frozen=True, slots=True)
class Refund:
    """Chips handed back because the opponent never matched them.

    Not a player action, so it is kept apart from the action history (bots
    reading the history for "what did my opponent do" never trip over it).
    Heads-up there is at most one refund per hand, but it is a tuple anyway.

    Attributes:
        street: the street whose close (or fold) triggered the refund.
        player: the seat that gets the chips back.
        chips: how many.
    """

    street: Street
    player: int
    chips: int

    def __str__(self) -> str:
        return f"{self.street.name.lower()}: uncalled {self.chips} returned to seat {self.player}"


@dataclass(frozen=True, slots=True)
class LegalActions:
    """Everything the player to act may do right now.

    Attributes:
        player: the seat these actions are for.
        can_fold: always true (folding is legal even when checking is free).
        can_check: not facing a bet.
        can_call: facing a bet.
        call_amount: chips a ``Call`` would cost: the gap to the current bet,
            or the whole stack if that is smaller. 0 when not facing a bet.
        can_bet: ``Bet`` is legal (nobody has bet this street).
        can_raise: ``Raise`` is legal (there is a bet, the player has chips
            beyond calling it, the betting is open to them, and the opponent
            is not all-in).
        min_to: smallest legal ``Bet`` amount / ``Raise`` total. If the player
            cannot afford a full minimum, this equals ``max_to`` (all-in only).
            0 when neither betting nor raising is legal.
        max_to: largest legal ``Bet`` amount / ``Raise`` total: the player's
            street total if they push their whole stack in. 0 when neither
            betting nor raising is legal.
    """

    player: int
    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int
    can_bet: bool
    can_raise: bool
    min_to: int
    max_to: int

    def is_legal(self, action: Action) -> bool:
        """Whether ``action`` is allowed for :attr:`player` right now."""
        match action:
            case Fold():
                return self.can_fold
            case Check():
                return self.can_check
            case Call():
                return self.can_call
            case Bet(amount=amount):
                return self.can_bet and self.min_to <= amount <= self.max_to
            case Raise(to=to):
                return self.can_raise and self.min_to <= to <= self.max_to
            case _:
                return False


@dataclass(frozen=True, slots=True)
class HandState:
    """The complete state of one heads-up hand, including private cards.

    Only the engine sees this. Bots get a :class:`PublicState` instead (see
    :func:`public_view`), which leaves out the hole cards.

    Per-seat values are 2-tuples indexed by seat (0 or 1).

    Attributes:
        button: seat on the button (posts the small blind).
        small_blind, big_blind: blind sizes in chips.
        starting_stacks: stacks before the blinds were posted. Chips are
            conserved: ``sum(stacks) + pot == sum(starting_stacks)`` always.
        blinds: chips each seat actually posted as a blind (less than the
            blind size if their stack was smaller).
        stacks: chips each player has left behind (not yet in the pot).
        street_bets: chips each player has put in on the current street.
        total_bets: chips each player has put in over the whole hand
            (including this street). ``pot == sum(total_bets)``.
        street: the current street; always ``Street.for_board(len(board))``.
        board: community cards dealt so far.
        hole_cards: each seat's two private cards, or ``()`` when the caller
            only cares about the betting (as most tests do).
        actor: the seat whose turn it is, or ``None`` if nobody may act now
            (street closed and waiting for cards, all-in run-out, hand over).
            :attr:`to_act` is the same thing under a friendlier name.
        last_raise: size of the last full bet/raise increment on this street
            (one big blind at the start of every street).
        needs_to_act: whether each seat still has to make a choice before the
            street can close. Posting a blind is not a choice, which is how
            the big blind gets its option after a limp.
        raise_allowed: whether the betting is open to each seat: true if they
            have not acted since the last *full* bet or raise. A short all-in
            does not set it back to true.
        folded: the seat that folded, or ``None``.
        history: every player action this hand, in order. Blinds are in
            :attr:`blinds` and returned chips in :attr:`refunds`, so together
            the three account for every chip.
        refunds: uncalled chips returned to their owner, in order.
    """

    button: int
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, int]
    blinds: tuple[int, int]
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    total_bets: tuple[int, int]
    street: Street
    board: tuple[Card, ...]
    hole_cards: tuple[tuple[Card, ...], ...]
    actor: int | None
    last_raise: int
    needs_to_act: tuple[bool, bool]
    raise_allowed: tuple[bool, bool]
    folded: int | None
    history: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]

    @property
    def big_blind_seat(self) -> int:
        """The seat that is not on the button."""
        return 1 - self.button

    @property
    def pot(self) -> int:
        """All chips put in so far by both players."""
        return self.total_bets[0] + self.total_bets[1]

    @property
    def current_bet(self) -> int:
        """The highest street total so far: what a player must match to call."""
        return max(self.street_bets)

    @property
    def cards_needed(self) -> int:
        """Board cards that must be dealt (via :func:`deal_board`) right now.

        Positive exactly when this street is closed (nobody to act), nobody
        folded and the board is not complete: then it is the size of the next
        street's deal (3 for the flop, 1 for the turn or river).
        """
        if self.folded is not None or self.actor is not None or len(self.board) == 5:
            return 0
        return Street(self.street + 1).board_size - len(self.board)

    @property
    def is_hand_over(self) -> bool:
        """True once someone has folded, or betting is done with a full board."""
        if self.folded is not None:
            return True
        return self.actor is None and len(self.board) == 5

    @property
    def is_showdown(self) -> bool:
        """True if the hand is over and nobody folded: compare the hands."""
        return self.is_hand_over and self.folded is None

    @property
    def to_act(self) -> int | None:
        """The seat that must act now, or ``None`` if nobody can act yet/again."""
        return self.actor


@dataclass(frozen=True, slots=True)
class PublicState:
    """What one seat is allowed to see: the table, minus all hidden cards.

    No hole cards (not even the viewer's own: those are passed to the bot
    separately, which keeps "public" and "private" visibly apart) and no
    deck. The fields mean the same as in :class:`HandState`.

    Attributes:
        seat: whose view this is. A bot needs to know its own seat even when
            it is not its turn (e.g. in the end-of-hand summary).
        legal: the legal actions for :attr:`to_act`, or ``None`` if nobody is
            to act. Shown to both seats: it follows from public information.
    """

    seat: int
    button: int
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, int]
    blinds: tuple[int, int]
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    total_bets: tuple[int, int]
    pot: int
    current_bet: int
    last_raise: int
    street: Street
    board: tuple[Card, ...]
    to_act: int | None
    folded: int | None
    history: tuple[ActionRecord, ...]
    refunds: tuple[Refund, ...]
    legal: LegalActions | None


# --- Starting a hand -------------------------------------------------------


def new_hand(
    button: int = 0,
    stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK),
    small_blind: int = SMALL_BLIND,
    big_blind: int = BIG_BLIND,
    hole_cards: Sequence[Sequence[Card]] = (),
) -> HandState:
    """Start a hand: post the blinds and hand the action to the button.

    Contract (checked by ``tests/test_betting.py``):

    - ``button`` is the int 0 or 1 (not ``True``, not ``1.0``); ``stacks``
      are two positive ints; ``1 <= small_blind <= big_blind``; ``hole_cards``
      is ``()`` or two lists of two cards, all distinct. Anything else raises
      ``ValueError``.
    - The button posts ``min(small_blind, stack)`` and the other seat posts
      ``min(big_blind, stack)``; a player who cannot cover their blind is
      all-in.
    - The button acts first. If the blinds alone already settle the betting
      (e.g. the small blind is all-in and the big blind has it covered), the
      preflop street closes immediately: any uncalled blind is returned (and
      recorded in ``refunds``) and the state waits for the flop.
    """
    _validate_new_hand(button, stacks, small_blind, big_blind, hole_cards)
    sb_seat, bb_seat = button, 1 - button
    posted = [0, 0]
    posted[sb_seat] = min(small_blind, stacks[sb_seat])
    posted[bb_seat] = min(big_blind, stacks[bb_seat])
    state = HandState(
        button=button,
        small_blind=small_blind,
        big_blind=big_blind,
        starting_stacks=(stacks[0], stacks[1]),
        blinds=(posted[0], posted[1]),
        stacks=(stacks[0] - posted[0], stacks[1] - posted[1]),
        street_bets=(posted[0], posted[1]),
        total_bets=(posted[0], posted[1]),
        street=Street.PREFLOP,
        board=(),
        hole_cards=tuple(tuple(h) for h in hole_cards),
        actor=None,
        # The big blind counts as an opening bet of one big blind, so the
        # increment starts at big_blind even if the BB was short and posted less.
        last_raise=big_blind,
        needs_to_act=(True, True),
        raise_allowed=(True, True),
        folded=None,
        history=(),
        refunds=(),
    )
    # Preflop the button speaks first; if neither seat has a real decision
    # (blind all-ins), the street closes on the spot.
    return _pass_turn(state, first=sb_seat)


def _validate_new_hand(
    button: int,
    stacks: tuple[int, int],
    small_blind: int,
    big_blind: int,
    hole_cards: Sequence[Sequence[Card]],
) -> None:
    """Raise ``ValueError`` unless the arguments describe a playable hand."""
    check_seat("button", button)
    if len(stacks) != 2 or not all(_is_int(s) and s > 0 for s in stacks):
        raise ValueError(f"stacks must be two positive ints, got {stacks!r}")
    if not (_is_int(small_blind) and _is_int(big_blind)):
        raise ValueError("blinds must be ints")
    if not 1 <= small_blind <= big_blind:
        raise ValueError(f"need 1 <= small blind <= big blind, got {small_blind}/{big_blind}")
    if len(hole_cards) == 0:
        return
    if len(hole_cards) != 2 or any(len(h) != 2 for h in hole_cards):
        raise ValueError("hole_cards must be two hands of two cards each")
    cards = [c for h in hole_cards for c in h]
    if not all(isinstance(c, Card) for c in cards) or len(set(cards)) != 4:
        raise ValueError(f"hole cards must be 4 distinct cards, got {cards!r}")


def _is_int(x: object) -> bool:
    """A real int (``True``/``False`` are ints in Python, but not chip counts)."""
    return isinstance(x, int) and not isinstance(x, bool)


def check_seat(name: str, seat: object) -> None:
    """Raise ``ValueError`` unless ``seat`` is the int 0 or 1.

    ``True in (0, 1)`` and ``1.0 in (0, 1)`` are both true in Python, so a
    plain membership test would let ``True`` and ``1.0`` through as seats.
    """
    if not (_is_int(seat) and seat in (0, 1)):
        raise ValueError(f"{name} must be seat 0 or 1, got {seat!r}")


# --- Legal actions ---------------------------------------------------------


def legal_actions(state: HandState) -> LegalActions:
    """List what the player to act may do, with the bet/raise size range.

    Contract:

    - Raises ``ValueError`` if nobody is to act (hand over, board waiting to
      be dealt, or players all-in).
    - Exactly one of ``can_check`` / ``can_call`` is true. ``can_fold`` is always
      true. At most one of ``can_bet`` / ``can_raise`` is true.
    - When betting or raising is legal, ``min_to <= max_to``, ``max_to`` is
      the all-in total, and ``min_to`` is ``current bet + last raise``
      (a full minimum) capped at ``max_to``.
    """
    p = state.to_act
    if p is None:
        raise ValueError(_why_no_one_acts(state))
    opponent = 1 - p
    current = state.current_bet
    to_call = current - state.street_bets[p]
    all_in_to = state.street_bets[p] + state.stacks[p]
    facing_bet = to_call > 0

    # You may put in more than a call only if: the betting is open to you,
    # you have chips beyond the call, and the opponent has chips left to
    # respond with (raising an all-in player achieves nothing).
    may_aggress = (
        state.raise_allowed[p] and all_in_to > current and state.stacks[opponent] > 0
    )
    can_bet = may_aggress and current == 0
    can_raise = may_aggress and current > 0
    if may_aggress:
        min_to = min(current + state.last_raise, all_in_to)
        max_to = all_in_to
    else:
        min_to = max_to = 0

    return LegalActions(
        player=p,
        can_fold=True,
        can_check=not facing_bet,
        can_call=facing_bet,
        call_amount=min(to_call, state.stacks[p]) if facing_bet else 0,
        can_bet=can_bet,
        can_raise=can_raise,
        min_to=min_to,
        max_to=max_to,
    )


def _why_no_one_acts(state: HandState) -> str:
    """A helpful message for acting when no action is possible."""
    if state.is_hand_over:
        return "the hand is over"
    if state.stacks[0] > 0 and state.stacks[1] > 0:
        return f"deal {state.cards_needed} board card(s) first"
    return "betting is over (players are all-in); deal the remaining board"


# --- Applying an action ----------------------------------------------------


def apply_action(state: HandState, player: int, action: Action) -> HandState:
    """Validate ``action`` by ``player`` and return the state after it.

    ``player`` is passed explicitly (rather than assumed to be ``to_act``) so
    that a runner bug that asks the wrong seat is caught instead of silently
    acting for the right one.

    Contract (checked by ``tests/test_betting.py``):

    - Raises ``ValueError`` (and changes nothing: the old state is untouched)
      if ``player`` is not the int 0 or 1, the hand is over, the board must be
      dealt first, ``player`` is not :attr:`HandState.to_act`, or the action
      is not in :func:`legal_actions`.
    - Chips are conserved: ``sum(stacks) + pot`` never changes.
    - ``Fold`` ends the hand; any uncalled bet goes back to its owner.
    - When the street's betting closes, any uncalled excess is returned and
      ``to_act`` becomes ``None``: either the hand is over (river) or the
      next street's cards must be dealt with :func:`deal_board`.
    """
    check_seat("player", player)
    legal = legal_actions(state)  # raises if nobody may act
    if player != legal.player:
        raise ValueError(f"seat {player} acted out of turn; seat {legal.player} is to act")
    if not legal.is_legal(action):
        raise ValueError(f"illegal action {action!r} for seat {player}: {legal}")

    if isinstance(action, Fold):
        return _fold(state, player)
    target = _street_total_after(state, player, action)
    return _put_in(state, player, action, target)


def _street_total_after(state: HandState, player: int, action: Action) -> int:
    """The player's street total after a (legal, non-fold) action."""
    match action:
        case Check():
            return state.street_bets[player]
        case Call():
            # A call matches the bet, or puts the whole stack in if that is less.
            all_in_to = state.street_bets[player] + state.stacks[player]
            return min(state.current_bet, all_in_to)
        case Bet(amount=amount):
            return amount
        case Raise(to=to):
            return to
    raise AssertionError(f"unhandled action {action!r}")  # pragma: no cover


def _put_in(state: HandState, player: int, action: Action, target: int) -> HandState:
    """Move chips so ``player``'s street total is ``target``, then pass the turn."""
    opponent = 1 - player
    chips = target - state.street_bets[player]
    increment = target - state.current_bet  # > 0 only for a bet or raise

    needs = list(state.needs_to_act)
    allowed = list(state.raise_allowed)
    last_raise = state.last_raise
    # Whatever the action, this player has now made their choice, and the
    # betting is closed to them until a full raise reopens it.
    needs[player] = False
    allowed[player] = False
    if increment > 0:
        # Any bet or raise, full or short, gives the opponent something to
        # respond to.
        needs[opponent] = True
        if increment >= state.last_raise:
            # A *full* bet/raise: it becomes the new minimum increment and
            # reopens the betting. A short all-in skips this block.
            last_raise = increment
            allowed[opponent] = True

    stacks = _add(state.stacks, player, -chips)
    record = ActionRecord(state.street, player, action, chips, all_in=stacks[player] == 0)
    state = replace(
        state,
        stacks=stacks,
        street_bets=_add(state.street_bets, player, chips),
        total_bets=_add(state.total_bets, player, chips),
        last_raise=last_raise,
        needs_to_act=(needs[0], needs[1]),
        raise_allowed=(allowed[0], allowed[1]),
        history=(*state.history, record),
    )
    return _pass_turn(state, first=opponent)


def _fold(state: HandState, player: int) -> HandState:
    """End the hand: ``player`` gives up, the uncalled bet goes back."""
    record = ActionRecord(state.street, player, Fold(), 0, all_in=False)
    state = replace(state, folded=player, actor=None, history=(*state.history, record))
    return _return_uncalled(state)


# --- Turn order and street transitions ------------------------------------


def _has_decision(state: HandState, seat: int) -> bool:
    """Does ``seat`` still have a real choice to make on this street?

    A player with chips has a decision if they are facing a bet (call or
    fold), or if they have not yet chosen this street and the opponent still
    has chips (so check/bet/raise all mean something). An all-in player never
    has a decision.
    """
    if state.stacks[seat] == 0:
        return False
    if state.street_bets[seat] < state.current_bet:
        return True
    return state.needs_to_act[seat] and state.stacks[1 - seat] > 0


def _pass_turn(state: HandState, first: int) -> HandState:
    """Give the turn to ``first`` (else the other seat), or close the street."""
    for seat in (first, 1 - first):
        if _has_decision(state, seat):
            return replace(state, actor=seat)
    return _close_street(state)


def _close_street(state: HandState) -> HandState:
    """Betting on this street is finished: return any uncalled chips, stop.

    Nobody is to act afterwards. On the river that means the hand is over
    (showdown); on earlier streets :attr:`HandState.cards_needed` becomes
    positive and the next street opens when :func:`deal_board` is called.
    """
    return replace(_return_uncalled(state), actor=None)


def _open_street(state: HandState) -> HandState:
    """Start the street that the just-dealt board cards belong to.

    Street bets go back to 0, the raise increment to one big blind, and the
    big blind (out of position) gets the first decision. If someone is
    all-in nobody has a decision, so :func:`_pass_turn` closes the street at
    once and the run-out continues.
    """
    state = replace(
        state,
        street=Street.for_board(len(state.board)),
        street_bets=(0, 0),
        last_raise=state.big_blind,
        needs_to_act=(True, True),
        raise_allowed=(True, True),
    )
    return _pass_turn(state, first=state.big_blind_seat)


def _return_uncalled(state: HandState) -> HandState:
    """Give any unmatched part of the biggest bet back to the player who made it."""
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


def _add(pair: tuple[int, int], seat: int, delta: int) -> tuple[int, int]:
    """Return ``pair`` with ``delta`` added to entry ``seat``."""
    return (pair[0] + delta, pair[1]) if seat == 0 else (pair[0], pair[1] + delta)


# --- Dealing the board -----------------------------------------------------


def deal_board(state: HandState, cards: Sequence[Card]) -> HandState:
    """Add the next street's board cards (3 for the flop, then 1 and 1).

    Contract:

    - ``len(cards)`` must equal :attr:`HandState.cards_needed` (which must be
      positive), and the cards must be distinct from each other, the board
      and the hole cards; otherwise ``ValueError``.
    - Afterwards ``street`` is the street of the new board, and the big blind
      is to act, unless someone is all-in: then the street closes at once
      and ``cards_needed`` already asks for the next street (or, after the
      river, the hand is over).
    """
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


def public_view(state: HandState, seat: int) -> PublicState:
    """The bot-safe view of ``state`` for ``seat``: no hole cards, no deck.

    Every field is copied explicitly (rather than "all fields except the
    hole cards") so that adding a private field to :class:`HandState` later
    can never leak it by accident.
    """
    check_seat("seat", seat)
    to_act = state.to_act
    return PublicState(
        seat=seat,
        button=state.button,
        small_blind=state.small_blind,
        big_blind=state.big_blind,
        starting_stacks=state.starting_stacks,
        blinds=state.blinds,
        stacks=state.stacks,
        street_bets=state.street_bets,
        total_bets=state.total_bets,
        pot=state.pot,
        current_bet=state.current_bet,
        last_raise=state.last_raise,
        street=state.street,
        board=state.board,
        to_act=to_act,
        folded=state.folded,
        history=state.history,
        refunds=state.refunds,
        legal=legal_actions(state) if to_act is not None else None,
    )


# --- Uncalled bets and the contested pot ------------------------------------


def uncalled_bet(total_bets: tuple[int, int]) -> tuple[int | None, int]:
    """Who put in chips the other player never matched, and how many.

    Heads-up, if one player has put in more than the other in total, the
    difference can never be won by anybody: the opponent did not (or could
    not, being all-in) put up matching chips. So it is simply returned.
    Returns ``(seat, amount)``, or ``(None, 0)`` if the totals are equal.

    Example: A (10,000 chips) shoves, B calls all-in for 3,000. Totals are
    ``(10000, 3000)``, so ``uncalled_bet`` is ``(0, 7000)``.
    """
    a, b = total_bets
    if a > b:
        return 0, a - b
    if b > a:
        return 1, b - a
    return None, 0


def contested_pot(total_bets: tuple[int, int]) -> int:
    """The pot that is actually played for: twice the smaller contribution.

    This is ``sum(total_bets)`` minus the :func:`uncalled_bet`. In the example
    above, ``contested_pot((10000, 3000)) == 6000``.
    """
    return 2 * min(total_bets)
