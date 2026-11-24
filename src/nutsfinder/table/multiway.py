"""Multiway (3..8 seats) play and replay for the poker table, on the N-player engine.

The heads-up table (:mod:`nutsfinder.table.session`) is left exactly as it was;
this module adds the same two modes for :mod:`nutsfinder.engine.multiway`.

**Play** (:class:`MultiPlaySession`): the human sits in any seat, a bot in
every other seat. The server drives the hand step by step exactly like
:func:`nutsfinder.engine.multiway.runner.play_hand_multi`:

- the deck is ``Deck(hand_seed)``; hole cards are dealt by
  ``runner.deal_hole_cards`` (two to seat 0, then seat 1, ...), the board
  comes off the top (no burns);
- each bot is asked through the runner's own ``_ask_bot`` (it gets
  ``public_view(state, its seat)`` and its own two cards, nothing else) and
  told through ``_tell_bots`` (the same :class:`MultiHandSummary`, with
  only showdown hands revealed).

Reusing the runner's helpers (rather than copying them) means a bot cannot
tell a table hand from a match hand; ``tests/test_table_multiway.py`` checks
it with spy bots. Blinds 50/100, the button moves one seat per hand (seat 0
first). Stacks reset to 10,000 every hand, or, with ``keep_stacks``, carry
over between hands (side pots then appear) and a busted seat rebuys
10,000 before the next hand.

Hidden information: other seats' hole cards are in the payload only for the
seats that reached a showdown (or for every seat after the hand if the
human asks to "show all cards"); hand seeds only after the hand; the session
seed/secret only after "end session". Bot reasoning is hidden during the
hand unless the live (peeking) toggle is on, and then every bot's cards are
masked in it.

**Replay** (:class:`MultiReplayJob`): N bots play ``n`` hands with
``play_hand_multi`` itself, the schedule of ``play_match_multi`` (seeds
``runner.hand_seeds(seed, n)``, button ``i % n``, stacks reset).

**Bots** are pluggable: :func:`register_multi_bot` adds one (key, name,
description, factory). The baselines (Random, CallStation, AlwaysRaise) are
built in; ``MultiPotOddsBot`` / ``MultiEquityBot`` / ``MultiBayesBot`` /
``MultiCFRBot`` are registered automatically when ``nutsfinder.bots.multiway``
imports (see :func:`register_optional_multi_bots`).
"""

from __future__ import annotations

import importlib
import inspect
import re
import secrets
import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import BIG_BLIND, SMALL_BLIND, STARTING_STACK, Street
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.bots import MultiAlwaysRaiseBot, MultiCallStationBot, MultiRandomBot
from nutsfinder.engine.multiway.hand_log import MultiHandLog, format_log, make_log
from nutsfinder.engine.multiway.pots import side_pots
from nutsfinder.engine.multiway.positions import MAX_PLAYERS, MIN_PLAYERS, position_names
from nutsfinder.engine.multiway.runner import _ask_bot, _tell_bots, deal_hole_cards, play_hand_multi
from nutsfinder.engine.multiway.state import (
    MultiHandState,
    MultiPublicState,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)
from nutsfinder.engine.runner import hand_seeds
from nutsfinder.table.session import (
    _COMPUTE,
    MAX_REPLAY_HANDS,
    SessionError,
    StaleRequest,
    _clear_decision,
    _last_decision,
    action_text,
    cards,
    derive_seed,
    hand_name,
    jsonable,
    mask_cards,
    new_lines,
    parse_action,
    parse_digits,
    reasoning_entry,
    say,
    seed_secret,
)

__all__ = [
    "MultiBotSpec",
    "MULTI_BOTS",
    "register_multi_bot",
    "register_multi_bot_class",
    "unregister_multi_bot",
    "register_optional_multi_bots",
    "MultiCatalog",
    "MultiPlaySession",
    "MultiReplayJob",
    "MIN_MULTI_PLAYERS",
    "MAX_PLAYERS",
]

MIN_MULTI_PLAYERS = 3  # 2 players is the heads-up table


# --- The bot registry ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiBotSpec:
    """A multiway bot on offer.

    ``factory(seed, catalog, overrides)`` builds one (``overrides`` are the
    catalog's extra keyword arguments for this key); ``needs_blueprint``
    makes it unavailable when the catalog has no trained CFR model.
    """

    key: str
    name: str
    description: str
    factory: Callable[[int, Any, dict[str, Any]], MultiBot]
    needs_blueprint: bool = False


MULTI_BOTS: dict[str, MultiBotSpec] = {}


def register_multi_bot(spec: MultiBotSpec) -> MultiBotSpec:
    """Offer a multiway bot at the table (replaces a bot with the same key)."""
    MULTI_BOTS[spec.key] = spec
    return spec


def unregister_multi_bot(key: str) -> None:
    MULTI_BOTS.pop(key, None)


def _class_factory(cls: type) -> tuple[Callable[[int, Any, dict[str, Any]], MultiBot], bool]:
    """A factory passing only the constructor arguments ``cls`` accepts.

    ``seed`` and ``debug=True`` (decision records for the reasoning panel)
    are passed if accepted, the catalog's blueprint if there is a
    ``blueprint`` or ``model`` parameter, and the catalog overrides filtered the same way.
    """
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        params = {}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    bp_param = next((p for p in ("blueprint", "model") if p in params), None)  # the trained CFR model
    needs_bp = bp_param is not None

    def accepts(name: str) -> bool:
        return var_kw or name in params

    def factory(seed: int, catalog: Any, overrides: dict[str, Any]) -> MultiBot:
        kw: dict[str, Any] = {k: v for k, v in overrides.items() if accepts(k)}
        if accepts("seed"):
            kw.setdefault("seed", seed)
        if accepts("debug"):
            kw.setdefault("debug", True)
        if needs_bp:
            kw.setdefault(bp_param, getattr(catalog, "blueprint", None))
        return cls(**kw)

    return factory, needs_bp


def register_multi_bot_class(key: str, name: str, description: str, cls: type,
                             needs_blueprint: bool | None = None) -> MultiBotSpec:
    """One-line registration of a :class:`MultiBot` subclass (see :func:`_class_factory`)."""
    factory, needs_bp = _class_factory(cls)
    return register_multi_bot(MultiBotSpec(key, name, description, factory,
                                           needs_bp if needs_blueprint is None else needs_blueprint))


register_multi_bot_class("random", "RandomBot", "Uniformly random legal action kind, random sizes. Folds even "
                         "when it could check.", MultiRandomBot)
register_multi_bot_class("call", "CallStationBot", "Checks or calls, always. Never folds, never bets.",
                         MultiCallStationBot)
register_multi_bot_class("raise", "AlwaysRaiseBot", "Bets or raises pot every time it can, otherwise calls. "
                         "Never folds.", MultiAlwaysRaiseBot)

# The multiway versions of the tiered bots, registered if nutsfinder.bots.multiway provides them.
OPTIONAL_MULTI_BOTS: tuple[tuple[str, str, str, str], ...] = (
    ("potodds", "MultiPotOddsBot", "PotOddsBot", "Tier 1, multiway: Monte Carlo equity against the live "
     "opponents; calls when equity beats the pot odds."),
    ("equity", "MultiEquityBot", "EquityBot", "Tier 2, multiway: equity vs ranges that narrow with the "
     "betting; best one-street EV."),
    ("bayes", "MultiBayesBot", "BayesBot", "Tier 3, multiway: EquityBot with a per-opponent Beta-Binomial "
     "model that learns every hand of the session."),
    ("cfr", "MultiCFRBot", "CFRBot", "Tier 4, multiway: plays from the CFR blueprint."),
)
OPTIONAL_MODULE = "nutsfinder.bots.multiway"
# Where the classes may live if the package does not re-export them (each tried, missing ones skipped).
OPTIONAL_SUBMODULES: tuple[str, ...] = ("pot_odds", "equity_bot", "bayes_bot", "cfr_bot")
optional_import_error: str | None = None


def register_optional_multi_bots() -> list[str]:
    """Register the bots of ``nutsfinder.bots.multiway`` that exist; returns their keys.

    A missing module is normal (``ImportError``: nothing registered). A module
    that fails for another reason must not stop the table: the error is kept
    in :data:`optional_import_error` and shown in the bot listing.
    """
    global optional_import_error
    try:
        mod = importlib.import_module(OPTIONAL_MODULE)
    except ImportError:
        optional_import_error = None
        return []
    except Exception as err:  # a half-written module must not stop the app
        optional_import_error = f"nutsfinder.bots.multiway failed to import: {type(err).__name__}: {err}"
        return []
    optional_import_error = None
    places = [mod]
    for sub in OPTIONAL_SUBMODULES:
        try:
            places.append(importlib.import_module(f"{OPTIONAL_MODULE}.{sub}"))
        except Exception:  # not written yet, or broken: skip it
            continue
    added = []
    for key, cls_name, name, description in OPTIONAL_MULTI_BOTS:
        cls = next((getattr(m, cls_name) for m in places if isinstance(getattr(m, cls_name, None), type)), None)
        if isinstance(cls, type) and issubclass(cls, MultiBot):
            register_multi_bot_class(key, name, description, cls)
            added.append(key)
    return added


register_optional_multi_bots()


class MultiCatalog:
    """Builds multiway bots by key. Shares the heads-up catalog's blueprint and overrides."""

    def __init__(self, base: Any = None, overrides: dict[str, dict[str, Any]] | None = None) -> None:
        self.base = base
        self.blueprint = getattr(base, "blueprint", None)
        self.cfr_error = getattr(base, "cfr_error", None) or "no CFR model configured"
        if overrides is None:
            overrides = getattr(base, "overrides", None) or {}
        self.overrides = {k: dict(v) for k, v in overrides.items()}

    def available(self, key: Any) -> tuple[bool, str | None]:
        spec = MULTI_BOTS.get(key) if isinstance(key, str) else None
        if spec is None:
            return False, f"unknown multiway bot {key!r}"
        if spec.needs_blueprint and self.blueprint is None:
            return False, self.cfr_error
        return True, None

    def listing(self) -> list[dict[str, Any]]:
        out = []
        for spec in MULTI_BOTS.values():
            ok, why = self.available(spec.key)
            out.append({"key": spec.key, "name": spec.name, "description": spec.description,
                        "available": ok, "unavailable_reason": why})
        return out

    def name(self, key: str) -> str:
        return MULTI_BOTS[key].name

    def make(self, key: str, seed: int) -> MultiBot:
        ok, why = self.available(key)
        if not ok:
            raise SessionError(f"bot {key!r} is not available: {why}")
        try:
            bot = MULTI_BOTS[key].factory(seed, self, self.overrides.get(key, {}))
        except Exception as err:
            raise SessionError(f"could not build bot {key!r}: {type(err).__name__}: {err}") from None
        if not isinstance(bot, MultiBot):
            raise SessionError(f"bot {key!r} is not a MultiBot")
        return bot


def seat_names(names: Sequence[str], human: int | None = None) -> tuple[str, ...]:
    """Display names: the human is "You"; a name used twice gets its seat number ("CallStationBot #3")."""
    out = []
    for seat, name in enumerate(names):
        if seat == human:
            out.append("You")
        elif sum(1 for s, x in enumerate(names) if x == name and s != human) > 1:
            out.append(f"{name} #{seat}")
        else:
            out.append(name)
    return tuple(out)


# --- JSON-ready views ------------------------------------------------------------------------


def pot_label(i: int) -> str:
    return "Main pot" if i == 0 else f"Side pot {i}"


def blind_lines(state: MultiHandState, names: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for seat, blind in ((state.sb_seat, "small"), (state.bb_seat, "big")):
        tail = " (all-in)" if state.blinds[seat] == state.starting_stacks[seat] else ""
        out.append({"seat": seat, "street": "preflop",
                    "text": say(names[seat], f"posts {blind} blind {state.blinds[seat]:,}{tail}")})
    return out


def multi_lines(prev: MultiHandState | None, state: MultiHandState, names: Sequence[str]) -> list[dict[str, Any]]:
    """Log lines for what changed from ``prev`` to ``state`` (the heads-up helper after the blinds)."""
    if prev is None:
        return blind_lines(state, names)
    return new_lines(prev, state, names)  # type: ignore[arg-type] - same field names


def table_view(state: MultiHandState, names: Sequence[str], visible: Collection[int],
               bot_flags: Sequence[bool], hero: int | None = None,
               extra: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The table as drawn. Hole cards only for seats in ``visible``; folded hands are mucked."""
    n = state.n
    positions = position_names(n, state.button)
    seats = []
    for seat in range(n):
        folded = state.folded[seat]
        show = seat in visible and seat < len(state.hole_cards)
        seats.append({
            "seat": seat,
            "name": names[seat],
            "is_bot": bool(bot_flags[seat]),
            "is_hero": seat == hero,
            "position": positions[seat],
            "stack": state.stacks[seat],
            "starting_stack": state.starting_stacks[seat],
            "bet": state.street_bets[seat],
            "total_bet": state.total_bets[seat],
            "button": seat == state.button,
            "blind": "SB" if seat == state.sb_seat else "BB" if seat == state.bb_seat else "",
            "cards": cards(state.hole_cards[seat]) if show else None,
            "hidden_cards": 0 if (show or folded) else 2,
            "folded": folded,
            "mucked": folded and not show,
            "all_in": state.stacks[seat] == 0 and not folded,
            "to_act": state.to_act == seat,
            **(extra[seat] if extra else {}),
        })
    # Chips already pulled into the middle (previous streets), layered into main and side pots.
    collected = [t - b for t, b in zip(state.total_bets, state.street_bets, strict=True)]
    pots = [{"label": pot_label(i), "amount": p.amount, "eligible": list(p.eligible)}
            for i, p in enumerate(side_pots(collected, state.live))]
    return {
        "seats": seats,
        "n_players": n,
        "board": cards(state.board),
        "street": state.street.name.lower(),
        "pot": state.pot,
        "pot_center": state.pot - sum(state.street_bets),
        "pots": pots,
        "current_bet": state.current_bet,
        "to_act": state.to_act,
        "button": state.button,
        "small_blind": state.small_blind,
        "big_blind": state.big_blind,
        "hand_over": state.is_hand_over,
    }


def _join(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def result_view(log: MultiHandLog, names: Sequence[str]) -> dict[str, Any]:
    r = log.result
    showdown = bool(log.show_order)
    sd_names = [hand_name(rank) if rank is not None else None for rank in r.ranks]
    pots = []
    for i, award in enumerate(r.pots):
        pots.append({
            "label": pot_label(i), "amount": award.amount, "eligible": list(award.eligible),
            "winners": list(award.winners), "shares": list(award.shares),
            "hand_name": sd_names[award.winners[0]] if showdown else None,
        })
    if not showdown:
        w = r.winners[0]
        headline = say(names[w], f"wins {r.winnings[w]:,} (everyone else folded)")
    else:
        parts = []
        for p in pots:
            who = [names[s] for s in p["winners"]]
            verb = "wins" if len(who) == 1 else "split"
            label = "" if len(pots) == 1 else f"{p['label']} "
            text = f"{_join(who)} {verb} {label}{p['amount']:,} with {p['hand_name']}"
            if len(who) == 1 and who[0] == "You":
                text = text.replace("You wins", "You win", 1)
            parts.append(text)
        headline = "; ".join(parts)
    return {
        "showdown": showdown,
        "folded": list(log.folded),
        "show_order": list(log.show_order),
        "winners": list(r.winners),
        "winnings": list(r.winnings),
        "net": list(r.net),
        "final_stacks": list(r.final_stacks),
        "hand_names": sd_names,
        "pots": pots,
        "headline": headline,
    }


def result_lines(log: MultiHandLog, names: Sequence[str]) -> list[dict[str, Any]]:
    r = log.result
    view = result_view(log, names)
    out: list[dict[str, Any]] = []
    if log.show_order:
        out.append({"seat": None, "street": "showdown", "deal": True, "text": "*** SHOWDOWN ***"})
        for seat in log.show_order:
            out.append({"seat": seat, "street": "showdown",
                        "text": say(names[seat], f"shows [{' '.join(cards(log.hole_cards[seat]))}] "
                                                 f"({hand_name(r.ranks[seat])})")})
        for p in view["pots"]:
            shares = ", ".join(f"{names[s]} {c:,}" for s, c in zip(p["winners"], p["shares"], strict=True))
            out.append({"seat": None, "street": "showdown", "text": f"{p['label']} {p['amount']:,}: {shares}"})
    out.append({"seat": None, "street": "result", "text": view["headline"]})
    return out


def fraction_to(view: MultiPublicState, fraction: float) -> int:
    """A ``fraction``-of-the-pot bet/raise-to (call first, then ``fraction`` of the pot), clamped."""
    legal = view.legal
    assert legal is not None and (legal.can_bet or legal.can_raise)
    to = round(view.current_bet + fraction * (view.pot + legal.call_amount))
    return min(max(to, legal.min_to), legal.max_to)


def legal_view(view: MultiPublicState) -> dict[str, Any] | None:
    """What the human may do, with bet presets computed from the real pot."""
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
            {"label": "½ pot", "to": fraction_to(view, 0.5)},
            {"label": "Pot", "to": fraction_to(view, 1.0)},
            {"label": "All-in", "to": legal.max_to},
        ]
    return out


_CARD_RUN = re.compile(r"(?<![A-Za-z0-9])((?:[2-9TJQKA][cdhs])+)(?![A-Za-z0-9])")


def scrub_cards(obj: Any, hidden: Collection[str]) -> Any:
    """``obj`` (JSON-like) with every mention of a card in ``hidden`` replaced by ``??``.

    Catches "As", "AsKs" and "[As Ks]" in any string, key or nested value,
    so a bot's decision record cannot leak its cards through a field the
    table does not know about.
    """
    hidden = set(hidden)

    def fix(text: str) -> str:
        def run(m: re.Match) -> str:
            chunk = m.group(1)
            return "".join("??" if chunk[i:i + 2] in hidden else chunk[i:i + 2] for i in range(0, len(chunk), 2))
        return _CARD_RUN.sub(run, text)

    if isinstance(obj, str):
        return fix(obj)
    if isinstance(obj, dict):
        return {fix(str(k)): scrub_cards(v, hidden) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_cards(v, hidden) for v in obj]
    return obj


def multi_reasoning_entry(index: int, seat: int, bot_key: str | None, view: MultiPublicState,
                          hole: Sequence[Card], action: Action, decision: Any, seconds: float) -> dict[str, Any]:
    """:func:`reasoning_entry`, with a multiway EV record (equity, EVs, pot odds) shown like a heads-up one."""
    entry = reasoning_entry(index, seat, bot_key, view, hole, action, decision, seconds)  # type: ignore[arg-type]
    detail = entry["detail"]
    if detail.get("kind") == "other" and {"equity", "evs", "action"} <= set(detail):
        detail["kind"] = "ev"
        detail["multiway"] = True
    return entry


def mask_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A reasoning entry with the bot's own hole cards removed everywhere (numbers stay)."""
    hole = entry.get("hole_cards") or []
    return scrub_cards(mask_cards(jsonable(entry)), hole)


def _bot_model_lines(bot: Any, opponents: Sequence[int], names: Sequence[str]) -> list[str] | None:
    """What a learning bot believes, for the panel shown after the hand (``None`` if nothing).

    Two shapes are understood: a single model with ``model.lines()`` (the
    heads-up BayesBot style), and per-opponent models with
    ``models.lines(seat)`` (MultiBayesBot's ``PooledModels``), shown for
    every opponent seat under that player's name.
    """
    try:
        single = getattr(getattr(bot, "model", None), "lines", None)
        if callable(single):
            out = single()
            return [str(x) for x in out] if isinstance(out, (list, tuple)) else None
        per_seat = getattr(getattr(bot, "models", None), "lines", None)
        if callable(per_seat):
            lines: list[str] = []
            for seat in opponents:
                got = per_seat(seat)
                if isinstance(got, (list, tuple)) and got:
                    lines.append(f"about {names[seat]} (seat {seat}):")
                    lines += [f"  {x}" for x in got]
            return lines or None
    except Exception:  # an optional display only
        return None
    return None


# --- Play mode ------------------------------------------------------------------------------------


class _HumanSeat(MultiBot):
    """Placeholder for the human's seat in the runner's helpers. Never asked to act."""

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:  # pragma: no cover
        raise AssertionError("the human's seat is driven by the session, not by a bot")


def _bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise SessionError(f"{what} must be true or false")
    return value


class MultiPlaySession:
    """You in seat ``human_seat``, a bot in every other seat, one hand at a time (see the module doc).

    Same interface as :class:`~nutsfinder.table.session.PlaySession` (the
    server treats both alike). Not thread-safe by itself: the server holds
    :attr:`lock` around every call.
    """

    def __init__(self, bots: Sequence[MultiBot | None], human_seat: int, seed: int | None = None,
                 bot_keys: Sequence[str | None] | None = None, bot_names: Sequence[str | None] | None = None,
                 keep_stacks: bool = False, starting_stack: int = STARTING_STACK,
                 small_blind: int = SMALL_BLIND, big_blind: int = BIG_BLIND, secret: bytes | None = None) -> None:
        n = len(bots)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:
            raise SessionError(f"a table has {MIN_PLAYERS} to {MAX_PLAYERS} seats, got {n}")
        if isinstance(human_seat, bool) or not isinstance(human_seat, int) or not 0 <= human_seat < n:
            raise SessionError(f"your seat must be 0..{n - 1}")
        for seat, bot in enumerate(bots):
            if (seat == human_seat) != (bot is None):
                raise SessionError("every seat but yours needs a bot (and yours none)")
        self.n = n
        self.human = human_seat
        self.bots: list[MultiBot | None] = list(bots)
        self._seat_bots: list[MultiBot] = [b if b is not None else _HumanSeat() for b in self.bots]
        self.bot_keys = list(bot_keys) if bot_keys is not None else [None] * n
        raw = [("You" if s == human_seat else (bot_names[s] if bot_names and bot_names[s] else type(b).__name__))
               for s, b in enumerate(self.bots)]
        self.names = seat_names(raw, human_seat)
        self.keep_stacks = _bool(keep_stacks, "keep_stacks")
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.user_seeded = seed is not None
        if secret is None:
            secret = seed_secret(seed) if seed is not None else secrets.token_bytes(32)
        self._secret = secret
        self.ended = False
        self.step = 0
        self._snapshot: dict[str, Any] | None = None
        self.lock = threading.Lock()
        self.live = False
        self.hands_started = 0
        self.hands_played = 0
        self.total_net = 0
        self.totals = [0] * n  # every seat's session net (chips)
        self.stacks = [starting_stack] * n  # stacks carried between hands (keep_stacks)
        self.rebuys = [0] * n
        self.history: list[dict[str, Any]] = []
        self._clear_hand()

    @classmethod
    def create(cls, catalog: MultiCatalog, keys: Sequence[str | None], human_seat: int, seed: int | None,
               keep_stacks: bool = False) -> MultiPlaySession:
        """A table of ``catalog`` bots (``keys[human_seat]`` is ignored); bot seeds use their own label."""
        n = len(keys)
        secret = seed_secret(seed) if seed is not None else secrets.token_bytes(32)
        bots: list[MultiBot | None] = []
        names: list[str | None] = []
        clean: list[str | None] = []
        for seat, key in enumerate(keys):
            if seat == human_seat:
                bots.append(None)
                names.append(None)
                clean.append(None)
                continue
            bots.append(catalog.make(key, derive_seed(secret, "bot", seat)))  # type: ignore[arg-type]
            names.append(catalog.name(key))  # type: ignore[arg-type]
            clean.append(key)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:  # pragma: no cover - checked by the caller too
            raise SessionError(f"a table has {MIN_PLAYERS} to {MAX_PLAYERS} seats")
        return cls(bots, human_seat, seed, bot_keys=clean, bot_names=names, keep_stacks=keep_stacks, secret=secret)

    def hand_seed_for(self, number: int) -> int:
        return derive_seed(self._secret, "hand", number)

    def check_step(self, expected: Any) -> None:
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise SessionError("expected_step (the 'step' of the state you acted on) is required")
        if expected != self.step:
            raise StaleRequest(f"stale request: the table has moved on (step {self.step}, you sent {expected})")

    def _clear_hand(self) -> None:
        self.hand_seed: int | None = None
        self.deck: Deck | None = None
        self.state: MultiHandState | None = None
        self.log: MultiHandLog | None = None
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
        return self.in_hand and self.state.to_act == self.human

    def visible_seats(self) -> set[int]:
        """Seats whose hole cards the human may see now."""
        seen = {self.human}
        if self.log is not None:
            if self.reveal:
                seen |= set(range(self.n))
            else:
                seen |= set(self.log.show_order)
        return seen

    # --- actions ----------------------------------------------------------------

    def start_hand(self) -> None:
        if self.ended:
            raise SessionError("this session has ended: start a new game")
        if self.in_hand:
            raise SessionError("finish the current hand first")
        self._clear_hand()
        self.step += 1
        button = self.hands_started % self.n
        self.hands_started += 1
        self.hand_seed = self.hand_seed_for(self.hands_started)
        self.peeked = self.live
        rebuy_lines = []
        if self.keep_stacks:
            for seat in range(self.n):
                if self.stacks[seat] <= 0:
                    self.stacks[seat] = self.starting_stack
                    self.rebuys[seat] += 1
                    rebuy_lines.append({"seat": seat, "street": "preflop",
                                        "text": say(self.names[seat], f"rebuys {self.starting_stack:,}")})
            stacks = tuple(self.stacks)
        else:
            stacks = (self.starting_stack,) * self.n
        self.deck = Deck(self.hand_seed)
        hole = deal_hole_cards(self.deck, self.n)  # the runner's deal order
        self.state = new_hand(stacks, button, self.small_blind, self.big_blind, hole_cards=hole)
        self.lines = rebuy_lines + multi_lines(None, self.state, self.names)
        self._advance()

    def act(self, kind: Any, amount: Any = None) -> None:
        if not self.in_hand:
            raise SessionError("no hand in progress: deal the next hand")
        if self.state.to_act != self.human:
            raise SessionError("it is not your turn")
        action = parse_action(kind, amount, public_view(self.state, self.human))  # type: ignore[arg-type]
        self.apply_human(action)

    def apply_human(self, action: Action) -> None:
        if not self.in_hand or self.state.to_act != self.human:
            raise SessionError("it is not your turn")
        prev = self.state
        try:
            self.state = apply_action(self.state, self.human, action)
        except ValueError as err:
            raise SessionError(str(err)) from None
        self.step += 1
        self.lines += multi_lines(prev, self.state, self.names)
        self._advance()

    def _advance(self) -> None:
        """Deal and let the bots act until the human must act or the hand is over."""
        total = sum(self.state.starting_stacks)
        try:
            while not self.state.is_hand_over:
                prev = self.state
                if self.state.cards_needed:
                    self.state = deal_board(self.state, self.deck.deal(self.state.cards_needed))
                elif self.state.to_act != self.human:
                    seat = self.state.to_act
                    bot = self.bots[seat]
                    view = public_view(self.state, seat)
                    hole = self.state.hole_cards[seat]
                    with _COMPUTE:
                        _clear_decision(bot)
                        start = time.perf_counter()
                        self.state = _ask_bot(self._seat_bots, self.state)  # what play_hand_multi asks
                        seconds = time.perf_counter() - start
                    action = self.state.history[-1].action
                    self.reasoning.append(multi_reasoning_entry(len(self.reasoning), seat, self.bot_keys[seat], view,
                                                          hole, action, _last_decision(bot), seconds))
                else:
                    break  # the human's turn
                if sum(self.state.stacks) + self.state.pot != total:
                    raise RuntimeError(f"chips not conserved: {self.state.stacks} + pot {self.state.pot}")
                self.lines += multi_lines(prev, self.state, self.names)
            if self.state.is_hand_over:
                self._finish()
        except Exception as err:  # a bot bug must not kill the session
            self.error = f"hand abandoned: {type(err).__name__}: {err}"
            self.lines.append({"seat": None, "street": "error", "text": self.error})

    def _finish(self) -> None:
        log = make_log(self.state, self.hand_seed)
        with _COMPUTE:
            _tell_bots(self._seat_bots, self.state, log)  # the summary play_hand_multi gives
        self.log = log
        self.hands_played += 1
        if self.keep_stacks:
            self.stacks = list(log.result.final_stacks)
        for seat, v in enumerate(log.result.net):
            self.totals[seat] += v
        net = log.result.net[self.human]
        self.total_net += net
        self.lines += result_lines(log, self.names)
        self.history.append({
            "hand": self.hands_started, "net": net, "net_bb": net / self.big_blind,
            "showdown": bool(log.show_order), "button": log.button,
            "headline": result_view(log, self.names)["headline"], "peeked": self.peeked,
        })

    def set_live(self, on: Any) -> None:
        self.live = _bool(on, "'on'")
        if self.live and self.in_hand:
            self.peeked = True

    def reveal_bot_cards(self) -> None:
        if self.log is None:
            raise SessionError("the bots' cards can only be shown after the hand")
        self.reveal = True

    def end_session(self) -> None:
        if self.in_hand:
            raise SessionError("finish the current hand first")
        self.ended = True
        self.step += 1

    # --- cached snapshot ---------------------------------------------------------------

    def refresh_snapshot(self) -> dict[str, Any]:
        self._snapshot = self.payload()
        return self._snapshot

    def snapshot(self) -> dict[str, Any]:
        snap = self._snapshot
        if snap is None:
            with self.lock:
                snap = self.refresh_snapshot()
        return dict(snap, busy=self.lock.locked())

    # --- the human-facing payload -----------------------------------------------------------

    def _score(self) -> dict[str, Any]:
        bb = self.big_blind
        clean = [h for h in self.history if not h["peeked"]]
        clean_chips = sum(h["net"] for h in clean)
        return {
            "hands": self.hands_played,
            "chips": self.total_net,
            "bb": self.total_net / bb,
            "bb_per_100": (100 * self.total_net / bb / self.hands_played) if self.hands_played else None,
            "clean_hands": len(clean),
            "clean_chips": clean_chips,
            "clean_bb": clean_chips / bb,
            "peeked_hands": self.hands_played - len(clean),
            "known_deck": self.user_seeded,
            "big_blind": bb,
        }

    def payload(self) -> dict[str, Any]:
        """Everything the browser shows. Other seats' cards only once they may be seen."""
        over = self.log is not None
        visible = self.visible_seats()
        status = ("no_hand" if self.state is None else "error" if self.error else "hand_over" if over
                  else "your_turn" if self.state.to_act == self.human else "bot_turn")
        bot_seats = [s for s in range(self.n) if s != self.human]
        extra = [{"rebuys": self.rebuys[s], "session_net": self.totals[s]} for s in range(self.n)]
        out: dict[str, Any] = {
            "mode": "play",
            "multiway": True,
            "n_players": self.n,
            "status": status,
            "error": self.error,
            "bot": {"key": None, "name": "the bots"},
            "players": [{"seat": s, "key": self.bot_keys[s], "name": self.names[s], "is_bot": s != self.human,
                         "rebuys": self.rebuys[s], "session_net": self.totals[s],
                         "session_bb": self.totals[s] / self.big_blind} for s in range(self.n)],
            "step": self.step,
            "ended": self.ended,
            "known_deck": self.user_seeded,
            "keep_stacks": self.keep_stacks,
            "hand": {"number": self.hands_started, "button": self.state.button if self.state else None,
                     "seed": self.hand_seed if over else None},
            "hero_seat": self.human,
            "table": (table_view(self.state, self.names, visible, [s != self.human for s in range(self.n)],
                                 self.human, extra) if self.state else None),
            "legal": legal_view(public_view(self.state, self.human)) if self.human_to_act else None,
            "log": self.lines,
            "result": result_view(self.log, self.names) if over else None,
            "visible_seats": sorted(visible),
            "bot_cards_visible": over and all(s in visible for s in bot_seats),
            "bot_cards_revealed_on_request": over and self.reveal and set(self.log.show_order) != set(range(self.n)),
            "live": self.live,
            "score": self._score(),
            "recent": self.history[-50:],
        }
        if self.ended:
            out["session_seed"] = self.seed
            out["session_secret"] = None if self.user_seeded else self._secret.hex()
        if over or self.error or self.live:
            out["reasoning"] = [e if e["seat"] in visible else mask_entry(e) for e in self.reasoning]
            out["reasoning_hidden"] = False
        else:
            out["reasoning"] = None
            out["reasoning_hidden"] = True
        models = {}
        if over:
            for s in bot_seats:
                lines = _bot_model_lines(self.bots[s], [o for o in range(self.n) if o != s], self.names)
                if lines:
                    models[str(s)] = lines
        out["bot_models"] = models or None
        out["bot_model"] = None
        return jsonable(out)


# --- Replay mode ------------------------------------------------------------------------------


class _Recorder(MultiBot):
    """Passes everything through to ``inner`` and records each decision's debug info."""

    def __init__(self, inner: MultiBot, key: str | None, sink: list) -> None:
        self.inner = inner
        self.key = key
        self.sink = sink

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        _clear_decision(self.inner)
        start = time.perf_counter()
        action = self.inner.act(view, hole_cards)
        seconds = time.perf_counter() - start
        self.sink.append(multi_reasoning_entry(len(self.sink), view.seat, self.key, view, hole_cards, action,
                                               _last_decision(self.inner), seconds))
        return action

    def end_hand(self, summary: MultiHandSummary) -> None:
        self.inner.end_hand(summary)

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)

    def __repr__(self) -> str:
        return repr(self.inner)


class MultiReplayJob:
    """N bots for ``n_hands`` hands on a background thread, ``play_match_multi``'s schedule.

    Per-hand seeds ``hand_seeds(seed, n)``, button ``i % n``, stacks reset.
    Bot seeds: ``derive_seed(seed_secret(seed), "bot", seat)``. Same interface
    as :class:`~nutsfinder.table.session.ReplayJob`.
    """

    def __init__(self, catalog: MultiCatalog | None, keys: Sequence[str | None], seed: int, n_hands: int,
                 bots: Sequence[MultiBot] | None = None, names: Sequence[str] | None = None) -> None:
        if isinstance(n_hands, bool) or not isinstance(n_hands, int) or not 1 <= n_hands <= MAX_REPLAY_HANDS:
            raise SessionError(f"hands must be a whole number from 1 to {MAX_REPLAY_HANDS}")
        if not MIN_PLAYERS <= len(keys) <= MAX_PLAYERS:
            raise SessionError(f"a replay needs {MIN_PLAYERS} to {MAX_PLAYERS} bots")
        self.keys = list(keys)
        self.n = len(keys)
        self.seed = seed
        self.n_hands = n_hands
        if bots is None:
            assert catalog is not None
            secret = seed_secret(seed)
            bots = [catalog.make(k, derive_seed(secret, "bot", s)) for s, k in enumerate(keys)]  # type: ignore[arg-type]
            names = [catalog.name(k) for k in keys]  # type: ignore[arg-type]
        self.bots = list(bots)
        self.names = seat_names(list(names or (type(b).__name__ for b in self.bots)))
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
        self._thread = threading.Thread(target=self.run, name="replay-multi", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> None:
        self.status = "running"
        try:
            for i, hand_seed in enumerate(hand_seeds(self.seed, self.n_hands)):
                if self._cancel.is_set():
                    self.status = "cancelled"
                    break
                sink: list[dict[str, Any]] = []
                recorders = [_Recorder(b, k, sink) for b, k in zip(self.bots, self.keys, strict=True)]
                with _COMPUTE:
                    log = play_hand_multi(recorders, seed=hand_seed, button=i % self.n)
                with self._lock:
                    self.hands.append({"index": i, "log": log, "reasoning": sink})
            else:
                self.status = "done"
        except Exception as err:
            self.error = f"{type(err).__name__}: {err}"
            self.status = "error"
        self.finished = time.time()

    @property
    def logs(self) -> list[MultiHandLog]:
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
        log: MultiHandLog = h["log"]
        return {
            "index": h["index"], "seed": log.seed, "button": log.button,
            "net": list(log.result.net), "net_bb": [n / log.big_blind for n in log.result.net],
            "showdown": bool(log.show_order), "winners": list(log.result.winners),
            "street": Street.for_board(len(log.board)).name.lower(),
            "pot": sum(log.result.winnings), "pots": len(log.result.pots), "actions": len(log.actions),
            "headline": result_view(log, self.names)["headline"],
        }

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            n = len(self.hands)
        sums = [self.summary(i) for i in range(n)]
        total = [sum(s["net"][seat] for s in sums) for seat in range(self.n)]
        return jsonable({
            "mode": "replay",
            "multiway": True,
            "n_players": self.n,
            "status": self.status,
            "error": self.error,
            "bots": [{"key": k, "name": nm} for k, nm in zip(self.keys, self.names, strict=True)],
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
        log: MultiHandLog = h["log"]
        reasoning: list[dict[str, Any]] = h["reasoning"]
        names = self.names
        visible = range(self.n)
        flags = [True] * self.n
        state = new_hand(log.starting_stacks, log.button, log.small_blind, log.big_blind, log.hole_cards)
        lines = multi_lines(None, state, names)
        frames = [{"table": table_view(state, names, visible, flags), "event": {"type": "start",
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
            lines += multi_lines(prev, state, names)
            frames.append({"table": table_view(state, names, visible, flags), "event": event,
                           "decision": decision, "lines": len(lines)})
        lines += result_lines(log, names)
        result = result_view(log, names)
        frames.append({"table": table_view(state, names, visible, flags),
                       "event": {"type": "result", "text": result["headline"]}, "decision": None,
                       "lines": len(lines), "result": result})
        return jsonable({
            "mode": "replay",
            "multiway": True,
            "index": index,
            "summary": self.summary(index),
            "names": list(names),
            "frames": frames,
            "log": lines,
            "reasoning": reasoning,
            "text": format_log(log),
        })

    def export(self) -> dict[str, Any]:
        with self._lock:
            hands = list(self.hands)
        return jsonable({
            "format": "nutsfinder-replay-multi-1",
            "bots": [{"key": k, "name": nm, "repr": repr(b)} for k, nm, b in zip(self.keys, self.names, self.bots,
                                                                                  strict=True)],
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

