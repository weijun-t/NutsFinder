"""Seats, blinds and position names for a 2..8 seat table.

Seats are numbered ``0 .. n-1`` clockwise. "Left of" means the next seat
clockwise: ``(seat + 1) % n``.

- **n >= 3:** the small blind is the first seat left of the button, the big
  blind the second. Preflop the first player to act is the seat left of the
  big blind (UTG; 3-handed that is the button). Postflop the first player is
  the first seat left of the button that can still act.
- **n == 2 (heads-up exception, TDA Rule 33):** the button posts the small
  blind and acts first preflop; the other seat posts the big blind and acts
  first postflop. This is exactly what the heads-up engine does.

In both cases "first preflop" is ``(bb_seat + 1) % n`` and "first postflop"
is ``(button + 1) % n``, so the formulas below need no special case beyond
where the blinds sit.
"""

from __future__ import annotations

__all__ = [
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "blind_seats",
    "first_preflop",
    "first_postflop",
    "clockwise_from",
    "position_names",
]

MIN_PLAYERS = 2
MAX_PLAYERS = 8

# Names for the seats between the big blind and the button, after UTG,
# counted back from the button (the last k-1 of these are used).
_MIDDLE_NAMES = ("UTG+1", "LJ", "HJ", "CO")


def blind_seats(n: int, button: int) -> tuple[int, int]:
    """``(small blind seat, big blind seat)``; heads-up the button is the SB."""
    if n == 2:
        return button, (button + 1) % 2
    return (button + 1) % n, (button + 2) % n


def first_preflop(n: int, button: int) -> int:
    """The seat that acts first preflop: the one left of the big blind."""
    return (blind_seats(n, button)[1] + 1) % n


def first_postflop(n: int, button: int) -> int:
    """The seat that acts first on the flop, turn and river (if it can act)."""
    return (button + 1) % n


def clockwise_from(n: int, start: int) -> tuple[int, ...]:
    """All ``n`` seats in clockwise order, beginning with ``start``."""
    return tuple((start + k) % n for k in range(n))


def position_names(n: int, button: int) -> tuple[str, ...]:
    """Each seat's position name, computed from ``(seat - button) mod n``.

    8-handed: UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB. 6-max: UTG, HJ, CO, BTN,
    SB, BB. 3-handed: BTN, SB, BB. Heads-up: BTN (who is also the small
    blind) and BB.
    """
    if not MIN_PLAYERS <= n <= MAX_PLAYERS:
        raise ValueError(f"need {MIN_PLAYERS}..{MAX_PLAYERS} players, got {n}")
    if n == 2:
        by_offset = ["BTN", "BB"]
    else:
        middle = n - 3
        rest = list(_MIDDLE_NAMES[len(_MIDDLE_NAMES) - (middle - 1) :]) if middle > 1 else []
        by_offset = ["BTN", "SB", "BB"] + (["UTG"] if middle >= 1 else []) + rest
    return tuple(by_offset[(seat - button) % n] for seat in range(n))
