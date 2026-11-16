"""Exact best response, exploitability and expected value for any :class:`Game`.

Expected value
--------------
For a profile sigma = (sigma_0, sigma_1), the value to player 0 is

    u_0(sigma) = sum over terminal z of  pi^sigma(z) * u_0(z),

computed by one recursion: terminal -> u_0(z); chance -> sum p * v(child);
player node -> sum sigma(I, a) * v(child).

Best response, and why it is *not* a per-history max
----------------------------------------------------
Fix the opponent's strategy sigma_{-i}. A best response for player i picks,
at each of i's info sets I, one action. It cannot look at the opponent's
card, so the choice at I must be good *on average over the histories in I*,
weighted by how likely each one is given what i knows:

    BR(I) = argmax_a  sum_{h in I}  w(h) * v_i(h . a),
    w(h)  = pi_c(h) * pi_{-i}(h)    (chance x opponent reach; i's own reach
                                      is the same for every h in I under
                                      perfect recall, so it drops out)

where v_i(h . a) is i's value of continuing from h . a with i playing BR
below and the opponent playing sigma_{-i}. The best-response value is then
v_i(root).

Taking max_a v_i(h . a) separately at each *history* would let i choose a
different action for "I hold Q, opponent holds J" and "I hold Q, opponent
holds K": that is a player who sees the opponent's card, a "cheating" best
response, and it overstates exploitability.

Worked example (Kuhn, opponent always bets/calls). P0 with Q: checking then
calling wins +2 against J and loses -2 against K, average 0; folding to the
bet gives -1. The info-set BR calls (0 > -1). A per-history max would call
against J (+2) and fold against K (-1), average +0.5. Over the deal:
info-set BR = (-1 + 0 + 2)/3 = 1/3, per-history max = (-1 + 0.5 + 2)/3 = 1/2.

Algorithm (two passes, both linear in the tree size):

1. Walk the whole tree, carrying w(h); record ``I -> [(h, w(h)), ...]`` for
   every info set of player i.
2. Evaluate v_i(h) bottom-up with memoisation. At an info set of i, compute
   BR(I) once (from all its histories) and reuse it. Perfect recall
   guarantees that all histories of a deeper info set were recorded in pass
   1, so the recursion is well defined.

Ties are broken toward the first action in ``actions(h)`` (deterministic).

Exploitability convention
-------------------------
``exploitability(sigma) = (BR_0(sigma_1) + BR_1(sigma_0)) / 2``, where
BR_i(sigma_{-i}) is the value to player i of best-responding. In a zero-sum
game BR_0 >= v* and BR_1 >= -v* (v* = game value to player 0), so the sum is
>= 0 and equals 0 exactly at a Nash equilibrium. OpenSpiel's "NashConv" is
the *sum* BR_0 + BR_1 (see :func:`nash_conv`), i.e. twice ours.

Missing info sets in a profile are played uniformly (see
:func:`~nutsfinder.gametheory.game.action_probs`).
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.gametheory.game import (
    CHANCE,
    Action,
    Game,
    History,
    InfosetKey,
    Profile,
    action_probs,
)


@dataclass(frozen=True)
class BestResponse:
    """Result of :func:`best_response`: value to ``player`` and its pure policy."""

    player: int
    value: float
    policy: dict[InfosetKey, Action]


def expected_value(game: Game, profile: Profile) -> float:
    """Value to player 0 when both players follow ``profile``."""

    def v(h: History) -> float:
        if game.is_terminal(h):
            return game.utility(h)
        if game.player(h) == CHANCE:
            return sum(p * v(game.next(h, a)) for a, p in game.chance_outcomes(h))
        acts = game.actions(h)
        probs = action_probs(profile, game.infoset_key(h), acts)
        return sum(p * v(game.next(h, a)) for a, p in zip(acts, probs) if p > 0.0)

    return v(game.initial())


def _collect(game: Game, profile: Profile,
             br: int) -> dict[InfosetKey, list[tuple[History, float]]]:
    """Pass 1: every info set of ``br`` with its histories and weights w(h)."""
    groups: dict[InfosetKey, list[tuple[History, float]]] = {}
    stack: list[tuple[History, float]] = [(game.initial(), 1.0)]
    while stack:
        h, w = stack.pop()
        if game.is_terminal(h):
            continue
        p = game.player(h)
        if p == CHANCE:
            stack.extend((game.next(h, a), w * q) for a, q in game.chance_outcomes(h))
            continue
        key, acts = game.infoset_key(h), game.actions(h)
        if p == br:
            groups.setdefault(key, []).append((h, w))
            stack.extend((game.next(h, a), w) for a in acts)
        else:
            probs = action_probs(profile, key, acts)
            stack.extend((game.next(h, a), w * q) for a, q in zip(acts, probs))
    return groups


def best_response(game: Game, profile: Profile, br_player: int) -> BestResponse:
    """Exact info-set best response of ``br_player`` to the opponent's part of ``profile``.

    Only the opponent's entries of ``profile`` are read; ``br_player``'s own
    entries are ignored.
    """
    if br_player not in (0, 1):
        raise ValueError(f"br_player must be 0 or 1, got {br_player}")
    sign = 1.0 if br_player == 0 else -1.0
    groups = _collect(game, profile, br_player)
    policy: dict[InfosetKey, Action] = {}
    memo: dict[History, float] = {}

    def choose(key: InfosetKey, acts: tuple[Action, ...]) -> Action:
        if key not in policy:
            totals = [sum(w * v(game.next(h, a)) for h, w in groups[key]) for a in acts]
            policy[key] = acts[max(range(len(acts)), key=totals.__getitem__)]
        return policy[key]

    def v(h: History) -> float:
        if h in memo:
            return memo[h]
        if game.is_terminal(h):
            out = sign * game.utility(h)
        elif (p := game.player(h)) == CHANCE:
            out = sum(q * v(game.next(h, a)) for a, q in game.chance_outcomes(h))
        else:
            key, acts = game.infoset_key(h), game.actions(h)
            if p == br_player:
                out = v(game.next(h, choose(key, acts)))
            else:
                probs = action_probs(profile, key, acts)
                out = sum(q * v(game.next(h, a)) for a, q in zip(acts, probs) if q > 0.0)
        memo[h] = out
        return out

    value = v(game.initial())
    for key, hs in groups.items():  # info sets the recursion never reached
        if key not in policy:
            choose(key, game.actions(hs[0][0]))
    return BestResponse(br_player, value, policy)


def best_response_value(game: Game, profile: Profile, br_player: int) -> float:
    """BR_i: value to ``br_player`` when it best-responds to the opponent's strategy."""
    return best_response(game, profile, br_player).value


def nash_conv(game: Game, profile: Profile) -> float:
    """BR_0 + BR_1 (OpenSpiel's NashConv). Twice :func:`exploitability`."""
    return best_response_value(game, profile, 0) + best_response_value(game, profile, 1)


def exploitability(game: Game, profile: Profile) -> float:
    """(BR_0 + BR_1) / 2, in chips per game. Zero exactly at a Nash equilibrium."""
    return nash_conv(game, profile) / 2.0
