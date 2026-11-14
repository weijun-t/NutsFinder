"""Cards and a seeded deck.

Notation: a card is written as rank then suit, e.g. ``"As"`` (ace of spades),
``"Td"`` (ten of diamonds), ``"9h"``. Ranks are ``2-9, T, J, Q, K, A`` (``"10"``
is accepted for ten when parsing); suits are ``c, d, h, s``. Parsing is
case-insensitive.

Every card also has an integer ``id`` in ``0..51``::

    id = (rank - 2) * 4 + suit

so ids are sorted by rank first, then suit (2c=0, 2d=1, ..., As=51).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

__all__ = ["Rank", "Suit", "Card", "Deck", "parse_cards", "FULL_DECK"]


class Rank(IntEnum):
    """Card rank. The integer value is the pip value, with ace high (14)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def char(self) -> str:
        """One-character symbol: ``'2'..'9', 'T', 'J', 'Q', 'K', 'A'``."""
        return _RANK_CHARS[self]

    @classmethod
    def from_char(cls, s: str) -> Rank:
        """Parse a rank symbol (``'T'`` or ``'10'`` for ten, case-insensitive)."""
        try:
            return _CHAR_TO_RANK[s.upper()]
        except (KeyError, AttributeError):
            raise ValueError(f"invalid rank: {s!r}") from None


class Suit(IntEnum):
    """Card suit. Suits have no value in hold'em; the order only fixes ids."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def char(self) -> str:
        """One-character symbol: ``'c', 'd', 'h', 's'``."""
        return _SUIT_CHARS[self]

    @classmethod
    def from_char(cls, s: str) -> Suit:
        """Parse a suit symbol (case-insensitive)."""
        try:
            return _CHAR_TO_SUIT[s.lower()]
        except (KeyError, AttributeError):
            raise ValueError(f"invalid suit: {s!r}") from None


_RANK_CHARS = {r: (str(int(r)) if r <= 9 else "TJQKA"[r - 10]) for r in Rank}
_CHAR_TO_RANK = {c: r for r, c in _RANK_CHARS.items()} | {"10": Rank.TEN}
_SUIT_CHARS = {Suit.CLUBS: "c", Suit.DIAMONDS: "d", Suit.HEARTS: "h", Suit.SPADES: "s"}
_CHAR_TO_SUIT = {c: s for s, c in _SUIT_CHARS.items()}


@dataclass(frozen=True, order=True, slots=True)
class Card:
    """An immutable playing card. Hashable; orders by rank, then suit."""

    rank: Rank
    suit: Suit

    def __post_init__(self) -> None:
        # Coerce plain ints (e.g. Card(14, 3)) and reject out-of-range values.
        object.__setattr__(self, "rank", Rank(self.rank))
        object.__setattr__(self, "suit", Suit(self.suit))

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse one card, e.g. ``"As"``, ``"Td"``, ``"10h"``, ``"9H"``."""
        if not isinstance(s, str):
            raise ValueError(f"card must be a string, got {type(s).__name__}")
        text = s.strip()
        if len(text) not in (2, 3):
            raise ValueError(f"invalid card: {s!r}")
        try:
            return cls(Rank.from_char(text[:-1]), Suit.from_char(text[-1]))
        except ValueError:
            raise ValueError(f"invalid card: {s!r}") from None

    @property
    def id(self) -> int:
        """Integer id in ``0..51``: ``(rank - 2) * 4 + suit``."""
        return (self.rank - 2) * 4 + self.suit

    @classmethod
    def from_id(cls, card_id: int) -> Card:
        """Inverse of :attr:`id`."""
        if not isinstance(card_id, int) or not 0 <= card_id < 52:
            raise ValueError(f"card id must be an int in 0..51, got {card_id!r}")
        return cls(Rank(card_id // 4 + 2), Suit(card_id % 4))

    def __str__(self) -> str:
        return self.rank.char + self.suit.char

    def __repr__(self) -> str:
        return f"Card({str(self)!r})"


FULL_DECK: tuple[Card, ...] = tuple(Card.from_id(i) for i in range(52))
"""All 52 cards in id order (2c, 2d, 2h, 2s, 3c, ..., As)."""


def parse_cards(s: str) -> list[Card]:
    """Parse several cards: ``"As Ks"``, ``"AsKs"``, ``"As,Ks"`` or ``"10h 9h"``.

    Whitespace and commas between cards are optional. Duplicates are *not*
    rejected here; callers that need distinct cards must check themselves.
    """
    text = "".join(s.replace(",", " ").split())
    cards: list[Card] = []
    i = 0
    while i < len(text):
        width = 3 if text.startswith("10", i) else 2
        token = text[i : i + width]
        if len(token) < width:
            raise ValueError(f"invalid card list: {s!r} (trailing {token!r})")
        cards.append(Card.from_str(token))
        i += width
    return cards


class Deck:
    """A deck of cards with its own seeded random number generator.

    The deck never touches the global :mod:`random` state, so the same seed
    always gives the same sequence of shuffles and deals. A new deck holds all
    52 cards and is shuffled once on construction.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._cards: list[Card] = list(FULL_DECK)
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the cards currently in the deck (dealt/removed cards stay out)."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        """Take ``n`` cards off the top of the deck."""
        if n < 0:
            raise ValueError(f"cannot deal a negative number of cards: {n}")
        if n > len(self._cards):
            raise ValueError(f"cannot deal {n} cards, only {len(self._cards)} left")
        dealt = self._cards[:n]
        del self._cards[:n]
        return dealt

    def remove(self, cards: Iterable[Card]) -> None:
        """Take specific known cards out of the deck (e.g. hole cards, board).

        Raises ``ValueError`` (and leaves the deck unchanged) if any card is not
        in the deck or appears twice in ``cards``.
        """
        to_remove = list(cards)
        wanted = set(to_remove)
        if len(wanted) != len(to_remove):
            raise ValueError(f"duplicate cards in {[str(c) for c in to_remove]}")
        missing = wanted.difference(self._cards)
        if missing:
            raise ValueError(f"cards not in deck: {sorted(str(c) for c in missing)}")
        self._cards = [c for c in self._cards if c not in wanted]

    @property
    def cards(self) -> tuple[Card, ...]:
        """The cards still in the deck, top first (a read-only snapshot)."""
        return tuple(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card: object) -> bool:
        return card in self._cards

    def __repr__(self) -> str:
        return f"Deck({len(self._cards)} cards)"
