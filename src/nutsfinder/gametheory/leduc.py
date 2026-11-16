"""Leduc hold'em (Southey et al. 2005) on the :class:`~nutsfinder.gametheory.game.Game` protocol.

Rules
-----
- Deck: J J Q Q K K (two suits; suits never matter, so we deal *ranks*).
- Both players ante 1 and get one private card. Player 0 acts first in
  both rounds.
- Round 1: fixed bet/raise size 2. Then one public (board) card is dealt.
  Round 2: fixed bet/raise size 4.
- **Raise cap: at most 2 bets per round** (a bet plus one raise). This is
  Southey et al.'s "two-bet maximum" and OpenSpiel's ``kMaxRaises = 2``
  (where the opening bet counts), and it is the cap that produces the
  commonly quoted 936 info sets; a 3-bet cap gives a different count.
- Actions: ``"c"`` check/call, ``"r"`` bet/raise, ``"f"`` fold. Fold is only
  legal when facing a bet (folding to no bet is dominated and not offered);
  ``"r"`` is only legal while fewer than 2 bets have been made this round.
- A round ends on check-check (``cc``) or on a call of a bet. A fold ends
  the game: the folder loses what it has put in.
- Showdown: a private card that pairs the board wins; otherwise the higher
  private card wins; equal ranks split (payoff 0).

Each round therefore has 6 decision points, ``"", c, r, cr, rr, crr``, and 5
round-1 lines reach round 2: ``cc, rc, crc, rrc, crrc``.

Chance by rank (lossless)
-------------------------
We deal ranks with their exact probabilities instead of the 6 distinct
cards. P0's rank: 2/6 each. P1's rank: 1/5 for P0's rank (one copy left),
2/5 for each other rank. Board: (copies left)/4, e.g. after J, Q the board is
J 1/4, Q 1/4, K 2/4. This gives the same game as dealing suited cards and
hiding the suit, with fewer nodes.

Histories and keys
------------------
A history is a tuple: ``(p0_rank, p1_rank, *round1, board, *round2)``, e.g.
``("K", "Q", "r", "c", "K", "r", "f")``: P0 K, P1 Q, P0 bets 2, P1 calls,
board K, P0 bets 4, P1 folds -> P0 wins P1's 1 + 2 = +3.

Info set keys are strings: round 1 ``"<own>:<r1>"`` (e.g. ``"Q:cr"``), round
2 ``"<own><board>:<r1>/<r2>"`` (e.g. ``"QK:rc/c"``). The acting player is
implied by the parity of the current round's betting, so the two players'
keys never collide.

Info set count (rank-only): round 1 has 6 decision points x 3 own ranks =
18; round 2 has 5 lines x 6 decision points x 3 own x 3 board = 270; total
**288**. With suits (6 own cards, 5 boards) it is 6x6 + 5x6x6x5 = 936.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from nutsfinder.gametheory.game import CHANCE

RANKS: tuple[str, ...] = ("J", "Q", "K")
_RANK = {r: i for i, r in enumerate(RANKS)}
COPIES = 2
ANTE = 1
BET_SIZES: tuple[int, int] = (2, 4)
MAX_BETS = 2
_BETTING = frozenset("crf")

LeducHistory = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Round:
    """Summary of one betting round's action string."""

    to_act: int            # whose turn it is if the round is not over
    bets: int              # bets/raises made so far
    facing: bool           # True when the player to act faces a bet
    done: bool             # round closed by check-check or call
    folder: int | None     # who folded (game over), else None
    put_in: tuple[int, int]  # chips each player put in during this round


@dataclass(frozen=True, slots=True)
class _State:
    """Parsed history. ``phase`` is one of deal, r1, board, r2, end."""

    cards: tuple[str, ...]
    board: str | None
    r1: str
    r2: str
    phase: str
    folder: int | None
    put_in: tuple[int, int]  # total chips in pot per player, antes included


def _round(line: str, size: int) -> _Round:
    """Replay one round of betting. Raises ValueError on illegal actions.

    >>> _round("cr", 2).put_in, _round("cr", 2).facing
    ((0, 2), True)
    """
    put = [0, 0]
    bets, done, folder = 0, False, None
    for i, a in enumerate(line):
        if done or folder is not None:
            raise ValueError(f"action after round end: {line!r}")
        p = i % 2
        facing = put[1 - p] > put[p]
        if a == "r":
            if bets >= MAX_BETS:
                raise ValueError(f"raise over cap: {line!r}")
            put[p] = put[1 - p] + size
            bets += 1
        elif a == "c":
            put[p] = put[1 - p]
            done = facing or i == 1  # call of a bet, or check-check
        elif a == "f":
            if not facing:
                raise ValueError(f"fold without a bet: {line!r}")
            folder = p
        else:
            raise ValueError(f"bad action {a!r}")
    p = len(line) % 2
    return _Round(p, bets, put[1 - p] > put[p], done, folder, (put[0], put[1]))


@lru_cache(maxsize=None)
def _parse(h: LeducHistory) -> _State:
    """Split a history into cards and rounds and replay the betting."""
    cards = h[:2]
    if len(h) < 2:
        return _State(cards, None, "", "", "deal", None, (ANTE, ANTE))
    i = 2
    while i < len(h) and h[i] in _BETTING:
        i += 1
    r1 = "".join(h[2:i])
    one = _round(r1, BET_SIZES[0])
    put = (ANTE + one.put_in[0], ANTE + one.put_in[1])
    if one.folder is not None or not one.done:
        if i != len(h):
            raise ValueError(f"cards dealt mid-round: {h}")
        phase = "end" if one.folder is not None else "r1"
        return _State(cards, None, r1, "", phase, one.folder, put)
    if i == len(h):
        return _State(cards, None, r1, "", "board", None, put)
    board, r2 = h[i], "".join(h[i + 1:])
    if board not in _RANK or any(a not in _BETTING for a in r2):
        raise ValueError(f"bad history: {h}")
    two = _round(r2, BET_SIZES[1])
    put = (put[0] + two.put_in[0], put[1] + two.put_in[1])
    phase = "end" if (two.folder is not None or two.done) else "r2"
    return _State(cards, board, r1, r2, phase, two.folder, put)


def _strength(card: str, board: str) -> int:
    """Showdown strength: a pair beats any unpaired card, else rank decides."""
    return 10 + _RANK[card] if card == board else _RANK[card]


def _left(h: LeducHistory, rank: str) -> int:
    """Copies of ``rank`` not yet dealt (cards are the non-betting entries)."""
    return COPIES - sum(1 for x in h if x == rank)


class Leduc:
    """Leduc hold'em. Stateless: all state lives in the (tuple) history."""

    def initial(self) -> LeducHistory:
        return ()

    def is_terminal(self, h: LeducHistory) -> bool:
        return _parse(h).phase == "end"

    def utility(self, h: LeducHistory) -> float:
        s = _parse(h)
        if s.phase != "end":
            raise ValueError(f"not terminal: {h}")
        if s.folder is not None:
            return float(-s.put_in[0] if s.folder == 0 else s.put_in[1])
        a, b = _strength(s.cards[0], s.board), _strength(s.cards[1], s.board)
        if a == b:
            return 0.0
        return float(s.put_in[1] if a > b else -s.put_in[0])

    def player(self, h: LeducHistory) -> int:
        s = _parse(h)
        if s.phase in ("deal", "board"):
            return CHANCE
        line = s.r1 if s.phase == "r1" else s.r2
        return len(line) % 2

    def chance_outcomes(self, h: LeducHistory) -> list[tuple[str, float]]:
        counts = [(r, _left(h, r)) for r in RANKS]
        total = sum(n for _, n in counts)
        return [(r, n / total) for r, n in counts if n > 0]

    def infoset_key(self, h: LeducHistory) -> str:
        s = _parse(h)
        own = s.cards[self.player(h)]
        if s.phase == "r1":
            return f"{own}:{s.r1}"
        return f"{own}{s.board}:{s.r1}/{s.r2}"

    def actions(self, h: LeducHistory) -> tuple[str, ...]:
        s = _parse(h)
        rnd = _round(s.r1, BET_SIZES[0]) if s.phase == "r1" else _round(s.r2, BET_SIZES[1])
        can_raise = rnd.bets < MAX_BETS
        if rnd.facing:
            return ("f", "c", "r") if can_raise else ("f", "c")
        return ("c", "r")

    def next(self, h: LeducHistory, a: str) -> LeducHistory:
        return h + (a,)
