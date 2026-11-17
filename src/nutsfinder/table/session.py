"""Game logic of the poker table app (no HTTP here; see ``server.py``).

Two modes
---------
**Play** (:class:`PlaySession`): a human (seat 0, drawn at the bottom of the
table) against one bot (seat 1). The engine is authoritative and the server
drives the hand one step at a time, exactly like
:func:`nutsfinder.engine.runner.play_hand` does:

- the deck is ``Deck(hand_seed)``; seat 0 gets the first two cards, seat 1
  the next two, the board comes off the top (no burns);
- the bot is asked through ``runner._ask_bot`` (so it gets
  ``public_view(state, its seat)`` and its own hole cards, nothing else);
- at the end the bot is told through ``runner._tell_bots`` (so it gets the
  same :class:`HandSummary`, with the opponent's cards only after a
  showdown).

Reusing the runner's two helpers (rather than copying them) is deliberate:
the bot cannot tell a session hand from a match hand.
``tests/test_table_app.py`` checks it with a spy bot. Blinds are 50/100,
stacks reset to 10,000 every hand, the button alternates (seat 0 has it on
the first hand), and the same bot object is kept for the whole session, so
BayesBot keeps learning about you.

The bot's reasoning: bots are built with ``debug=True`` and, after every
``act``, their ``last_decision`` (a :class:`~nutsfinder.bots.common.Decision`
or :class:`~nutsfinder.bots.cfr_bot.CFRDecision`) is stored. It is hidden
from the human-facing payload during the hand unless the "live" (peeking)
toggle is on. The bot's hole cards are never in the payload before the
hand ends; afterwards only after a showdown or an explicit "show bot cards"
(and they are masked in the reasoning records until then).

**Replay** (:class:`ReplayJob`): two bots play ``n`` hands with
``play_hand`` itself (the same seeds and button rule as ``play_match``), on a
background thread. Each bot is wrapped in a :class:`_Recorder` that passes
``act`` / ``end_hand`` / ``reseed`` straight through and stores each
decision's debug record; the step-by-step frames shown in the browser are
rebuilt afterwards from the hand log with the engine.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from nutsfinder.bots import (
    AlwaysRaiseBot,
    BayesBot,
    CallStationBot,
    CFRBot,
    EquityBot,
    PotOddsBot,
    RandomBot,
)
from nutsfinder.bots.cfr_bot import CFRDecision, preload_turn_tables
from nutsfinder.bots.common import Decision, cards_text, raise_to_for_fraction
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    BIG_BLIND,
    SMALL_BLIND,
    STARTING_STACK,
    HandState,
    PublicState,
    Street,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.evaluator import category
from nutsfinder.engine.hand_log import HandLog, format_log, make_log
from nutsfinder.engine.runner import _ask_bot, _tell_bots, hand_seeds, play_hand

__all__ = [
    "BOT_SPECS",
    "BotCatalog",
    "BotSpec",
    "PlaySession",
    "ReplayJob",
    "SessionError",
    "StaleRequest",
    "MAX_REPLAY_HANDS",
    "derive_seed",
    "seed_secret",
    "parse_digits",
]

MAX_REPLAY_HANDS = 200

# Bots share module-level caches (CFRBot's card abstraction, its turn tables).
# The server runs a play session and a replay job on different threads, so
# every bot computation takes this lock: at most one bot thinks at a time.
_COMPUTE = threading.RLock()


class SessionError(ValueError):
    """A request the session cannot honour (bad input, wrong time). Safe to show the user."""


class StaleRequest(SessionError):
    """The request was made for an older state (e.g. a double click): the server answers 409."""


# --- Seeds -------------------------------------------------------------------------------
#
# Every seed the app uses is a hash: ``derive_seed(secret, label, i)`` is the first 8 bytes
# of sha256(secret | label | i). Revealing hand seeds therefore says nothing about future
# ones (unlike consecutive Mersenne Twister outputs, from which 624 words recover the
# state), and the bots' seeds come from a different label than the decks'.
#
# A session the user seeded uses ``seed_secret(seed)`` as its secret, so it is still fully
# reproducible from the seed; an unseeded session uses 32 random bytes from ``secrets``.

MAX_DIGITS = 20


def seed_secret(seed: int) -> bytes:
    """The secret of a user-seeded session: a fixed encoding of the seed."""
    return b"nutsfinder-seed:" + str(int(seed)).encode()


def derive_seed(secret: bytes, label: str, i: int) -> int:
    """A 64-bit seed: the first 8 bytes of sha256(secret | label | i)."""
    digest = hashlib.sha256(secret + b"|" + label.encode() + b"|" + str(i).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def parse_digits(value: Any, what: str, max_digits: int = MAX_DIGITS) -> int:
    """A non-negative int from an int or an ASCII digit string of at most ``max_digits`` digits."""
    if isinstance(value, str):
        text = value.strip()
        if not (text and text.isascii() and text.isdigit() and len(text) <= max_digits):
            raise SessionError(f"{what} must be a whole number (at most {max_digits} digits)")
        return int(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 10**max_digits:
        raise SessionError(f"{what} must be a whole number (at most {max_digits} digits)")
    return value


# --- The bots on offer ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BotSpec:
    key: str
    name: str
    description: str


BOT_SPECS: tuple[BotSpec, ...] = (
    BotSpec("random", "RandomBot", "Uniformly random legal action, random sizes. Folds even when it could check."),
    BotSpec("call", "CallStationBot", "Checks or calls, always. Never folds, never bets."),
    BotSpec("raise", "AlwaysRaiseBot", "Bets or raises pot every time it can, otherwise calls. Never folds."),
    BotSpec("potodds", "PotOddsBot", "Tier 1: Monte Carlo equity vs a random hand; calls when equity beats the pot "
            "odds, bets/raises above fixed thresholds."),
    BotSpec("equity", "EquityBot", "Tier 2: equity vs a range that narrows with your bets; picks the action with "
            "the best one-street EV."),
    BotSpec("bayes", "BayesBot", "Tier 3: EquityBot with a Beta-Binomial model of you (VPIP, aggression, fold "
            "rates, bluffs) that learns every hand of the session."),
    BotSpec("cfr", "CFRBot", "Tier 4: plays the MCCFR blueprint of an abstracted game (buckets, bet-size "
            "translation)."),
)
_SPEC = {s.key: s for s in BOT_SPECS}

_RULES = {
    "random": "RandomBot has no reasoning: it picked a legal action kind uniformly at random "
              "(fold / check-or-call / bet-or-raise), then a uniformly random size.",
    "call": "CallStationBot has no reasoning: it checks when it can and calls otherwise.",
    "raise": "AlwaysRaiseBot has no reasoning: it makes a pot-sized bet/raise whenever one is legal, "
             "otherwise it calls (or checks).",
}


class BotCatalog:
    """Builds bots by key, with debug decision logging on.

    CFRBot needs a trained blueprint: if ``model_path`` is missing, unreadable
    or untrained, CFRBot is reported unavailable (never silently replaced by
    the untrained uniform blueprint). ``overrides`` passes extra keyword
    arguments to a bot's constructor (tests use it for small Monte Carlo
    samples).
    """

    def __init__(self, model_path: str | Path | None = None, turn_tables: str | Path | None = None,
                 overrides: dict[str, dict[str, Any]] | None = None) -> None:
        self.overrides = {k: dict(v) for k, v in (overrides or {}).items()}
        self.blueprint = None
        self.cfr_error: str | None = None
        self.model_description: str | None = None
        self.turn_tables_loaded = False
        if model_path is None:
            self.cfr_error = "no CFR model configured"
        elif not Path(model_path).is_file():
            self.cfr_error = f"model file not found: {model_path}"
        else:
            try:
                from nutsfinder.abstraction.blueprint import Blueprint

                bp = Blueprint.load(model_path)
                if not bp.is_trained:
                    self.cfr_error = f"model {model_path} is not trained"
                else:
                    self.blueprint = bp
                    self.model_description = bp.describe()
            except Exception as err:  # a corrupt file must not stop the app
                self.cfr_error = f"could not load {model_path}: {err}"
        if self.blueprint is not None and turn_tables is not None and Path(turn_tables).is_file():
            try:
                preload_turn_tables(turn_tables)
                self.turn_tables_loaded = True
            except Exception:  # optional speed-up only
                self.turn_tables_loaded = False

    def available(self, key: str) -> tuple[bool, str | None]:
        if key not in _SPEC:
            return False, f"unknown bot {key!r}"
        if key == "cfr" and self.blueprint is None:
            return False, self.cfr_error
        return True, None

    def listing(self) -> list[dict[str, Any]]:
        out = []
        for spec in BOT_SPECS:
            ok, why = self.available(spec.key)
            desc = spec.description
            if spec.key == "cfr" and self.model_description:
                desc += f" Model: {self.model_description}."
            out.append({"key": spec.key, "name": spec.name, "description": desc, "available": ok,
                        "unavailable_reason": why})
        return out

    def name(self, key: str) -> str:
        return _SPEC[key].name

    def make(self, key: str, seed: int) -> Bot:
        ok, why = self.available(key)
        if not ok:
            raise SessionError(f"bot {key!r} is not available: {why}")
        kw = self.overrides.get(key, {})
        match key:
            case "random":
                return RandomBot(seed=seed, **kw)
            case "call":
                return CallStationBot(**kw)
            case "raise":
                return AlwaysRaiseBot(**kw)
            case "potodds":
                return PotOddsBot(seed=seed, debug=True, **kw)
            case "equity":
                return EquityBot(seed=seed, debug=True, **kw)
            case "bayes":
                return BayesBot(seed=seed, debug=True, **kw)
            case "cfr":
                return CFRBot(self.blueprint, seed=seed, debug=True, **kw)
        raise SessionError(f"unknown bot {key!r}")  # pragma: no cover


# --- JSON-ready views ----------------------------------------------------------------


def jsonable(x: Any) -> Any:
    """Tuples -> lists, non-finite floats -> None, cards/actions/enums -> text (recursively)."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Card):
        return str(x)
    if isinstance(x, (Fold, Check, Call, Bet, Raise)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "item"):  # numpy scalars
        return jsonable(x.item())
    return str(x)


def cards(cs: Sequence[Card]) -> list[str]:
    return [str(c) for c in cs]


def action_text(action: Action, chips: int = 0, all_in: bool = False) -> str:
    match action:
        case Fold():
            text = "folds"
        case Check():
            text = "checks"
        case Call():
            text = f"calls {chips:,}"
        case Bet(amount=amount):
            text = f"bets {amount:,}"
        case Raise(to=to):
            text = f"raises to {to:,}"
        case _:  # pragma: no cover
            text = str(action)
    return text + (" (all-in)" if all_in else "")


def say(name: str, text: str) -> str:
    """``"BayesBot raises to 300"`` but ``"You raise to 300"`` (drop the verb's s for "You")."""
    if name == "You":
        verb, _, rest = text.partition(" ")
        if verb.endswith("s"):
            verb = verb[:-1]
        text = f"{verb} {rest}".rstrip()
    return f"{name} {text}"


def hand_name(rank) -> str:
    return category(rank).name.replace("_", " ").lower()


def new_lines(prev: HandState | None, state: HandState, names: Sequence[str]) -> list[dict[str, Any]]:
    """Log lines for what changed from ``prev`` to ``state`` (one action or one deal)."""
    out: list[dict[str, Any]] = []
    if prev is None:
        for seat, blind in ((state.button, "small"), (1 - state.button, "big")):
            tail = " (all-in)" if state.blinds[seat] == state.starting_stacks[seat] else ""
            out.append({"seat": seat, "street": "preflop",
                        "text": say(names[seat], f"posts {blind} blind {state.blinds[seat]:,}{tail}")})
        return out
    for rec in state.history[len(prev.history):]:
        out.append({"seat": rec.player, "street": rec.street.name.lower(),
                    "text": say(names[rec.player], action_text(rec.action, rec.chips, rec.all_in))})
    for ref in state.refunds[len(prev.refunds):]:
        out.append({"seat": ref.player, "street": ref.street.name.lower(),
                    "text": f"Uncalled {ref.chips:,} returned to {'you' if names[ref.player] == 'You' else names[ref.player]}"})
    if len(state.board) > len(prev.board):
        street = Street.for_board(len(state.board))
        out.append({"seat": None, "street": street.name.lower(), "deal": True,
                    "text": f"*** {street.name} *** [{' '.join(cards(state.board))}]"})
    return out


def table_view(state: HandState, names: Sequence[str], visible: Collection[int],
               bot_flags: Sequence[bool] = (False, False)) -> dict[str, Any]:
    """The table as drawn: seats, board, pot. Hole cards only for seats in ``visible``."""
    seats = []
    n = len(state.stacks)
    for seat in range(n):
        folded = state.folded == seat
        show = seat in visible and seat < len(state.hole_cards)
        seats.append({
            "seat": seat,
            "name": names[seat],
            "is_bot": bool(bot_flags[seat]),
            "stack": state.stacks[seat],
            "bet": state.street_bets[seat],
            "total_bet": state.total_bets[seat],
            "button": seat == state.button,
            "blind": "SB" if seat == state.button else "BB",
            "cards": cards(state.hole_cards[seat]) if show else None,
            "hidden_cards": 0 if (show or folded) else 2,
            "folded": folded,
            "all_in": state.stacks[seat] == 0 and not folded,
            "to_act": state.to_act == seat,
        })
    return {
        "seats": seats,
        "board": cards(state.board),
        "street": state.street.name.lower(),
        "pot": state.pot,
        "pot_center": state.pot - sum(state.street_bets),
        "current_bet": state.current_bet,
        "to_act": state.to_act,
        "button": state.button,
        "small_blind": state.small_blind,
        "big_blind": state.big_blind,
        "hand_over": state.is_hand_over,
    }


def result_view(log: HandLog, names: Sequence[str]) -> dict[str, Any]:
    r = log.result
    showdown = log.folded is None
    names_at_sd = [hand_name(rank) if rank is not None else None for rank in r.ranks]
    if len(r.winners) == 2:
        headline = f"Split pot: {r.winnings[0]:,} each"
        if showdown:
            headline += f" ({names_at_sd[0]})"
    else:
        w = r.winners[0]
        headline = say(names[w], f"wins {r.winnings[w]:,}")
        if showdown:
            headline += f" with {names_at_sd[w]}"
        else:
            headline += f" ({names[log.folded]} folded)"
    return {
        "showdown": showdown,
        "folded": log.folded,
        "winners": list(r.winners),
        "winnings": list(r.winnings),
        "net": list(r.net),
        "hand_names": names_at_sd if showdown else [None, None],
        "headline": headline,
    }


def result_lines(log: HandLog, names: Sequence[str]) -> list[dict[str, Any]]:
    r = log.result
    out: list[dict[str, Any]] = []
    if log.folded is None:
        out.append({"seat": None, "street": "showdown", "deal": True, "text": "*** SHOWDOWN ***"})
        for seat in (0, 1):
            out.append({"seat": seat, "street": "showdown",
                        "text": say(names[seat], f"shows [{' '.join(cards(log.hole_cards[seat]))}] "
                                                     f"({hand_name(r.ranks[seat])})")})
    out.append({"seat": None, "street": "result", "text": result_view(log, names)["headline"]})
    return out


def decision_view(decision: Any) -> dict[str, Any] | None:
    """A bot's debug record as plain data (``kind`` = ``ev`` or ``cfr``), plus its text form."""
    if decision is None:
        return None
    data = {f.name: jsonable(getattr(decision, f.name)) for f in fields(decision)} if is_dataclass(decision) else {}
    if isinstance(decision, CFRDecision):
        data["kind"] = "cfr"
    elif isinstance(decision, Decision):
        data["kind"] = "ev"
    else:
        data["kind"] = "other"
    data["text"] = str(decision)
    return data


def reasoning_entry(index: int, seat: int, bot_key: str | None, view: PublicState, hole: Sequence[Card],
                    action: Action, decision: Any, seconds: float) -> dict[str, Any]:
    """One bot decision for the "how the bot thought" panel."""
    legal = view.legal
    detail = decision_view(decision)
    if detail is None:
        detail = {"kind": "rule", "text": _RULES.get(bot_key or "", "This bot does not log its reasoning.")}
    return {
        "index": index,
        "seat": seat,
        "street": view.street.name.lower(),
        "hole_cards": cards(hole),
        "board": cards(view.board),
        "pot": view.pot,
        "to_call": legal.call_amount if legal else 0,
        "action": str(action),
        "seconds": round(seconds, 4),
        "detail": detail,
    }


def mask_cards(entry: dict[str, Any]) -> dict[str, Any]:
    """A copy of a reasoning entry with the bot's hole cards replaced by ``None`` / ``????``.

    Only the cards themselves are removed. The numbers stay (equity, E[HS], bucket, the
    strategy), and they still say roughly how strong the hand was, i.e. hint at its class:
    that is why the live view is labelled as peeking.
    """
    hole = entry["hole_cards"]
    out = dict(entry)
    out["hole_cards"] = None
    out["cards_hidden"] = True
    detail = dict(entry["detail"])
    joined = "".join(hole)
    if "hole_cards" in detail:
        detail["hole_cards"] = None
    for key in ("text",):
        if isinstance(detail.get(key), str):
            detail[key] = detail[key].replace(joined, "????")
    out["detail"] = detail
    return out


def _last_decision(bot: Bot) -> Any:
    return getattr(bot, "last_decision", None)


def _clear_decision(bot: Bot) -> None:
    if hasattr(bot, "last_decision"):
        bot.last_decision = None


# --- Play mode -------------------------------------------------------------------------


class _HumanSeat(Bot):
    """Placeholder for the human's seat in the runner's helpers. Never asked to act."""

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:  # pragma: no cover
        raise AssertionError("the human's seat is driven by the session, not by a bot")


def parse_action(kind: Any, amount: Any, view: PublicState) -> Action:
    """Turn ``{"action": kind, "amount": n}`` from the browser into an engine action (validated)."""
    legal = view.legal
    if legal is None:
        raise SessionError("nobody can act right now")
    if not isinstance(kind, str):
        raise SessionError("action must be a string: fold, check, call, bet or raise")
    kind = kind.strip().lower()
    if kind == "fold":
        action: Action = Fold()
    elif kind == "check":
        action = Check()
    elif kind == "call":
        action = Call()
    elif kind in ("bet", "raise"):
        if isinstance(amount, bool) or not isinstance(amount, int):
            if isinstance(amount, float) and amount.is_integer():
                amount = int(amount)
            elif isinstance(amount, str):
                amount = parse_digits(amount, f"{kind} amount")
            else:
                raise SessionError(f"{kind} needs a whole-number amount (the street total), got {amount!r}")
        if not (legal.can_bet or legal.can_raise):
            raise SessionError("betting/raising is not allowed right now")
        if not legal.min_to <= amount <= legal.max_to:
            raise SessionError(f"amount must be between {legal.min_to:,} and {legal.max_to:,}, got {amount:,}")
        action = Bet(amount) if legal.can_bet else Raise(amount)
    else:
        raise SessionError(f"unknown action {kind!r}: use fold, check, call, bet or raise")
    if not legal.is_legal(action):
        raise SessionError(f"{action} is not legal right now")
    return action


def legal_view(view: PublicState) -> dict[str, Any] | None:
    """What the human may do, with the bet presets computed from the real pot."""
    legal = view.legal
    if legal is None or legal.player != view.seat:
        return None
    out: dict[str, Any] = {
        "can_fold": legal.can_fold, "can_check": legal.can_check, "can_call": legal.can_call,
        "call_amount": legal.call_amount, "can_bet": legal.can_bet, "can_raise": legal.can_raise,
        "min_to": legal.min_to, "max_to": legal.max_to, "street_bet": view.street_bets[view.seat],
        "presets": [],
    }
    if legal.can_bet or legal.can_raise:
        out["presets"] = [
            {"label": "Min", "to": legal.min_to},
            {"label": "½ pot", "to": raise_to_for_fraction(view, 0.5)},
            {"label": "Pot", "to": raise_to_for_fraction(view, 1.0)},
            {"label": "All-in", "to": legal.max_to},
        ]
    return out


class PlaySession:
    """You (seat 0) against one bot (seat 1), one hand at a time. See the module doc.

    Not thread-safe by itself: the server holds :attr:`lock` around every call.
    """

    HUMAN = 0
    BOT = 1

    def __init__(self, bot: Bot, seed: int | None = None, bot_key: str | None = None, bot_name: str | None = None,
                 stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK),
                 small_blind: int = SMALL_BLIND, big_blind: int = BIG_BLIND,
                 secret: bytes | None = None) -> None:
        """``seed``: the user's seed (reproducible deals, "known deck"), or ``None`` for secret random
        deals. ``secret`` overrides the derived secret (``create`` passes the one it used for the bot)."""
        self.bot = bot
        self.bot_key = bot_key
        self.bot_name = bot_name or type(bot).__name__
        self.seed = seed
        self.user_seeded = seed is not None
        if secret is None:
            secret = seed_secret(seed) if seed is not None else secrets.token_bytes(32)
        self._secret = secret
        self.ended = False
        self.step = 0  # bumped by every table change (deal, action, end); echoed back to reject stale requests
        self._snapshot: dict[str, Any] | None = None
        self.stacks = stacks
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.lock = threading.Lock()
        self.names = ("You", self.bot_name)
        self.live = False
        self.hands_started = 0
        self.hands_played = 0
        self.total_net = 0
        self.history: list[dict[str, Any]] = []
        self._clear_hand()

    @classmethod
    def create(cls, catalog: BotCatalog, key: str, seed: int | None) -> PlaySession:
        """A session against ``catalog``'s bot ``key``; the bot's seed is derived like the decks' (other label)."""
        secret = seed_secret(seed) if seed is not None else secrets.token_bytes(32)
        bot = catalog.make(key, derive_seed(secret, "bot", 1))
        return cls(bot, seed, bot_key=key, bot_name=catalog.name(key), secret=secret)

    def hand_seed_for(self, number: int) -> int:
        """The deck seed of hand ``number`` (1-based)."""
        return derive_seed(self._secret, "hand", number)

    def check_step(self, expected: Any) -> None:
        """Reject a request made for an older state (409): ``expected`` must equal :attr:`step`."""
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise SessionError("expected_step (the 'step' of the state you acted on) is required")
        if expected != self.step:
            raise StaleRequest(f"stale request: the table has moved on (step {self.step}, you sent {expected})")

    def _clear_hand(self) -> None:
        self.hand_seed: int | None = None
        self.deck: Deck | None = None
        self.state: HandState | None = None
        self.log: HandLog | None = None
        self.lines: list[dict[str, Any]] = []
        self.reasoning: list[dict[str, Any]] = []
        self.reveal = False
        self.error: str | None = None
        self.peeked = False

    # --- queries ----------------------------------------------------------------

    @property
    def in_hand(self) -> bool:
        return self.state is not None and self.log is None and self.error is None

    @property
    def human_to_act(self) -> bool:
        return self.in_hand and self.state.to_act == self.HUMAN

    @property
    def bot_cards_visible(self) -> bool:
        return self.log is not None and (self.log.folded is None or self.reveal)

    # --- actions ----------------------------------------------------------------

    def start_hand(self) -> None:
        """Shuffle and deal the next hand, then let the bot act until it is your turn (or the hand ends)."""
        if self.ended:
            raise SessionError("this session has ended: start a new game")
        if self.in_hand:
            raise SessionError("finish the current hand first")
        self._clear_hand()
        self.step += 1
        button = self.hands_started % 2
        self.hands_started += 1
        self.hand_seed = self.hand_seed_for(self.hands_started)
        self.peeked = self.live
        self.deck = Deck(self.hand_seed)
        hole = (tuple(self.deck.deal(2)), tuple(self.deck.deal(2)))
        self.state = new_hand(button, self.stacks, self.small_blind, self.big_blind, hole_cards=hole)
        self.lines = new_lines(None, self.state, self.names)
        self._advance()

    def act(self, kind: Any, amount: Any = None) -> None:
        """Apply the human's action from browser input (validated), then let the bot play on."""
        if not self.in_hand:
            raise SessionError("no hand in progress: deal the next hand")
        if self.state.to_act != self.HUMAN:
            raise SessionError("it is not your turn")
        action = parse_action(kind, amount, public_view(self.state, self.HUMAN))
        self.apply_human(action)

    def apply_human(self, action: Action) -> None:
        """Apply an engine action for the human seat (raises :class:`SessionError` if illegal)."""
        if not self.in_hand or self.state.to_act != self.HUMAN:
            raise SessionError("it is not your turn")
        prev = self.state
        try:
            self.state = apply_action(self.state, self.HUMAN, action)
        except ValueError as err:
            raise SessionError(str(err)) from None
        self.step += 1
        self.lines += new_lines(prev, self.state, self.names)
        self._advance()

    def _advance(self) -> None:
        """Deal and let the bot act until the human must act or the hand is over."""
        bots: list[Bot] = [_HumanSeat(), self.bot]
        try:
            while not self.state.is_hand_over:
                prev = self.state
                if self.state.cards_needed:
                    self.state = deal_board(self.state, self.deck.deal(self.state.cards_needed))
                elif self.state.to_act == self.BOT:
                    view = public_view(self.state, self.BOT)
                    hole = self.state.hole_cards[self.BOT]
                    with _COMPUTE:
                        _clear_decision(self.bot)
                        start = time.perf_counter()
                        self.state = _ask_bot(bots, self.state)  # what play_hand asks, verbatim
                        seconds = time.perf_counter() - start
                    action = self.state.history[-1].action
                    self.reasoning.append(reasoning_entry(len(self.reasoning), self.BOT, self.bot_key, view, hole,
                                                          action, _last_decision(self.bot), seconds))
                else:
                    break  # the human's turn
                self.lines += new_lines(prev, self.state, self.names)
            if self.state.is_hand_over:
                self._finish()
        except Exception as err:  # a bot bug must not kill the session
            self.error = f"hand abandoned: {type(err).__name__}: {err}"
            self.lines.append({"seat": None, "street": "error", "text": self.error})

    def _finish(self) -> None:
        log = make_log(self.state, self.hand_seed)
        with _COMPUTE:
            _tell_bots([_HumanSeat(), self.bot], self.state, log)  # the summary play_hand gives
        self.log = log
        self.hands_played += 1
        net = log.result.net[self.HUMAN]
        self.total_net += net
        self.lines += result_lines(log, self.names)
        self.history.append({
            "hand": self.hands_started, "net": net, "net_bb": net / self.big_blind, "showdown": log.folded is None,
            "button": log.button, "headline": result_view(log, self.names)["headline"], "peeked": self.peeked,
        })

    def set_live(self, on: Any) -> None:
        if not isinstance(on, bool):
            raise SessionError("'on' must be true or false")
        self.live = on
        if on and self.in_hand:
            self.peeked = True  # the score of this hand is no longer "clean"

    def reveal_bot_cards(self) -> None:
        if self.log is None:
            raise SessionError("the bot's cards can only be shown after the hand")
        self.reveal = True

    def end_session(self) -> None:
        """Stop dealing and reveal the session's seed (or its random secret) for verification."""
        if self.in_hand:
            raise SessionError("finish the current hand first")
        self.ended = True
        self.step += 1

    # --- cached snapshot (read without waiting for the bot) -------------------------------

    def refresh_snapshot(self) -> dict[str, Any]:
        """Recompute the payload; call with :attr:`lock` held after every change."""
        self._snapshot = self.payload()
        return self._snapshot

    def snapshot(self) -> dict[str, Any]:
        """The payload as of the last completed change (never blocks on a thinking bot)."""
        snap = self._snapshot
        if snap is None:
            with self.lock:
                snap = self.refresh_snapshot()
        return dict(snap, busy=self.lock.locked())

    # --- the human-facing payload --------------------------------------------------------

    def _score(self) -> dict[str, Any]:
        bb = self.big_blind
        clean = [h for h in self.history if not h["peeked"]]
        clean_chips = sum(h["net"] for h in clean)
        return {
            "hands": self.hands_played,
            "chips": self.total_net,
            "bb": self.total_net / bb,
            "bb_per_100": (100 * self.total_net / bb / self.hands_played) if self.hands_played else None,
            "clean_hands": len(clean),  # hands played without live peeking
            "clean_chips": clean_chips,
            "clean_bb": clean_chips / bb,
            "peeked_hands": self.hands_played - len(clean),
            "known_deck": self.user_seeded,  # you chose the seed: in principle you know the deck
            "big_blind": bb,
        }

    def payload(self) -> dict[str, Any]:
        """Everything the browser shows. Never contains the bot's cards before they may be seen."""
        over = self.log is not None
        bot_visible = self.bot_cards_visible
        visible = {self.HUMAN} | ({self.BOT} if bot_visible else set())
        status = ("no_hand" if self.state is None else "error" if self.error else "hand_over" if over
                  else "your_turn" if self.state.to_act == self.HUMAN else "bot_turn")
        out: dict[str, Any] = {
            "mode": "play",
            "status": status,
            "error": self.error,
            "bot": {"key": self.bot_key, "name": self.bot_name},
            "step": self.step,
            "ended": self.ended,
            "known_deck": self.user_seeded,
            "hand": {"number": self.hands_started, "button": self.state.button if self.state else None,
                     "seed": self.hand_seed if over else None},
            "hero_seat": self.HUMAN,
            "table": table_view(self.state, self.names, visible, (False, True)) if self.state else None,
            "legal": legal_view(public_view(self.state, self.HUMAN)) if self.human_to_act else None,
            "log": self.lines,
            "result": result_view(self.log, self.names) if over else None,
            "bot_cards_visible": bot_visible,
            "bot_cards_revealed_on_request": over and self.reveal and self.log.folded is not None,
            "live": self.live,
            "score": self._score(),
            "recent": self.history[-50:],
        }
        if self.ended:  # only now: the seed/secret would let anyone recompute every deck
            out["session_seed"] = self.seed
            out["session_secret"] = None if self.user_seeded else self._secret.hex()
        if over or self.error or self.live:
            entries = self.reasoning if bot_visible else [mask_cards(e) for e in self.reasoning]
            out["reasoning"] = entries
            out["reasoning_hidden"] = False
        else:
            out["reasoning"] = None
            out["reasoning_hidden"] = True
        if over and isinstance(self.bot, BayesBot):
            out["bot_model"] = self.bot.model.lines()
        else:
            out["bot_model"] = None
        return jsonable(out)


# --- Replay mode -----------------------------------------------------------------------


class _Recorder(Bot):
    """Passes everything through to ``inner`` and records each decision's debug info."""

    def __init__(self, inner: Bot, seat: int, key: str, sink: list) -> None:
        self.inner = inner
        self.seat = seat
        self.key = key
        self.sink = sink

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        _clear_decision(self.inner)
        start = time.perf_counter()
        action = self.inner.act(state, hole_cards)
        seconds = time.perf_counter() - start
        self.sink.append(reasoning_entry(len(self.sink), state.seat, self.key, state, hole_cards, action,
                                         _last_decision(self.inner), seconds))
        return action

    def end_hand(self, summary: HandSummary) -> None:
        self.inner.end_hand(summary)

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)


def replay_names(name_a: str, name_b: str) -> tuple[str, str]:
    if name_a == name_b:
        return f"{name_a} (A)", f"{name_b} (B)"
    return name_a, name_b


class ReplayJob:
    """Bot A (seat 0) vs bot B (seat 1) for ``n_hands`` hands, played on a background thread.

    Exactly ``play_match``'s schedule: per-hand seeds ``hand_seeds(seed, n)``,
    button on seat 0 for even hands, stacks reset every hand. The bots' own
    seeds come from a separate stream, ``derive_seed(seed_secret(seed), "bot", seat)``,
    so they never coincide with a deck seed.
    """

    def __init__(self, catalog: BotCatalog | None, key_a: str, key_b: str, seed: int, n_hands: int,
                 bots: Sequence[Bot] | None = None, names: Sequence[str] | None = None) -> None:
        if isinstance(n_hands, bool) or not isinstance(n_hands, int) or not 1 <= n_hands <= MAX_REPLAY_HANDS:
            raise SessionError(f"hands must be a whole number from 1 to {MAX_REPLAY_HANDS}")
        self.key_a, self.key_b = key_a, key_b
        self.seed = seed
        self.n_hands = n_hands
        if bots is None:
            assert catalog is not None
            secret = seed_secret(seed)
            bots = [catalog.make(key_a, derive_seed(secret, "bot", 0)),
                    catalog.make(key_b, derive_seed(secret, "bot", 1))]
            names = replay_names(catalog.name(key_a), catalog.name(key_b))
        self.bots = list(bots)
        self.names = tuple(names or (type(b).__name__ for b in self.bots))
        self.hands: list[dict[str, Any]] = []
        self.status = "pending"
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self.run, name="replay", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> None:
        """Play every hand (also callable directly, synchronously, e.g. in tests)."""
        self.status = "running"
        try:
            for i, hand_seed in enumerate(hand_seeds(self.seed, self.n_hands)):
                if self._cancel.is_set():
                    self.status = "cancelled"
                    break
                sink: list[dict[str, Any]] = []
                recorders = [_Recorder(b, s, k, sink) for s, (b, k) in enumerate(zip(self.bots, (self.key_a, self.key_b)))]
                with _COMPUTE:
                    log = play_hand(recorders, seed=hand_seed, button=i % 2)
                with self._lock:
                    self.hands.append({"index": i, "log": log, "reasoning": sink})
            else:
                self.status = "done"
        except Exception as err:
            self.error = f"{type(err).__name__}: {err}"
            self.status = "error"
        self.finished = time.time()

    @property
    def logs(self) -> list[HandLog]:
        with self._lock:
            return [h["log"] for h in self.hands]

    def _hand(self, index: Any) -> dict[str, Any]:
        index = parse_digits(index, "hand index", 6)
        with self._lock:
            if not 0 <= index < len(self.hands):
                raise SessionError(f"no hand {index} (have {len(self.hands)})")
            return self.hands[index]

    def summary(self, index: int) -> dict[str, Any]:
        h = self._hand(index)
        log: HandLog = h["log"]
        return {
            "index": h["index"], "seed": log.seed, "button": log.button,
            "net": list(log.result.net), "net_bb": [n / log.big_blind for n in log.result.net],
            "showdown": log.folded is None, "folded": log.folded,
            "street": Street.for_board(len(log.board)).name.lower(),
            "pot": sum(log.result.winnings), "actions": len(log.actions),
            "headline": result_view(log, self.names)["headline"],
        }

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            n = len(self.hands)
        sums = [self.summary(i) for i in range(n)]
        total = [sum(s["net"][0] for s in sums), sum(s["net"][1] for s in sums)]
        return jsonable({
            "mode": "replay",
            "status": self.status,
            "error": self.error,
            "bots": [{"key": self.key_a, "name": self.names[0]}, {"key": self.key_b, "name": self.names[1]}],
            "seed": self.seed,
            "n_hands": self.n_hands,
            "done": n,
            "hands": sums,
            "total": total,
            "total_bb": [t / BIG_BLIND for t in total],
            "elapsed": (self.finished or time.time()) - self.started,
        })

    def hand_payload(self, index: Any) -> dict[str, Any]:
        """Every step of one hand (a table snapshot per deal/action), all cards visible."""
        h = self._hand(index)
        log: HandLog = h["log"]
        reasoning: list[dict[str, Any]] = h["reasoning"]
        names = self.names
        visible = (0, 1)
        state = new_hand(log.button, log.starting_stacks, log.small_blind, log.big_blind, log.hole_cards)
        lines = new_lines(None, state, names)
        frames = [{"table": table_view(state, names, visible, (True, True)), "event": {"type": "start",
                   "text": f"Hand {index + 1}: blinds posted"}, "decision": None, "lines": len(lines)}]
        board = list(log.board)
        k = 0
        while not state.is_hand_over:
            prev = state
            if state.cards_needed:
                dealt, board = board[: state.cards_needed], board[state.cards_needed:]
                state = deal_board(state, dealt)
                event = {"type": "deal", "street": state.street.name.lower(), "cards": cards(dealt),
                         "text": f"{state.street.name.capitalize()}: {' '.join(cards(dealt))}"}
                decision = None
            else:
                rec = log.actions[k]
                state = apply_action(state, rec.player, rec.action)
                entry = reasoning[k]
                assert entry["seat"] == rec.player and entry["action"] == str(rec.action), (entry, rec)
                event = {"type": "action", "seat": rec.player, "street": rec.street.name.lower(),
                         "text": say(names[rec.player], action_text(rec.action, rec.chips, rec.all_in))}
                decision = k
                k += 1
            lines += new_lines(prev, state, names)
            frames.append({"table": table_view(state, names, visible, (True, True)), "event": event,
                           "decision": decision, "lines": len(lines)})
        lines += result_lines(log, names)
        result = result_view(log, names)
        frames.append({"table": table_view(state, names, visible, (True, True)),
                       "event": {"type": "result", "text": result["headline"]}, "decision": None,
                       "lines": len(lines), "result": result})
        return jsonable({
            "mode": "replay",
            "index": index,
            "summary": self.summary(index),
            "names": list(names),
            "frames": frames,
            "log": lines,
            "reasoning": reasoning,
            "text": format_log(log),
        })

    def export(self) -> dict[str, Any]:
        """The whole replay as JSON: settings, every hand's text log and every decision."""
        with self._lock:
            hands = list(self.hands)
        return jsonable({
            "format": "nutsfinder-replay-1",
            "bots": [{"key": self.key_a, "name": self.names[0], "repr": repr(self.bots[0])},
                     {"key": self.key_b, "name": self.names[1], "repr": repr(self.bots[1])}],
            "seed": self.seed,
            "n_hands": self.n_hands,
            "status": self.status,
            "hands": [{**self.summary(h["index"]),
                       "hole_cards": [cards(c) for c in h["log"].hole_cards],
                       "board": cards(h["log"].board),
                       "actions": [{"seat": r.player, "street": r.street.name.lower(), "action": str(r.action),
                                    "chips": r.chips, "all_in": r.all_in} for r in h["log"].actions],
                       "text": format_log(h["log"]),
                       "reasoning": h["reasoning"]} for h in hands],
        })

