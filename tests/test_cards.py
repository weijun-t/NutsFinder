import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nutsfinder.engine.cards import FULL_DECK, Card, Deck, Rank, Suit, parse_cards

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
ALL_CARD_STRINGS = [r + s for r in RANK_CHARS for s in SUIT_CHARS]


# --- Rank / Suit -----------------------------------------------------------


def test_rank_values():
    assert [int(r) for r in Rank] == list(range(2, 15))
    assert "".join(r.char for r in Rank) == RANK_CHARS


def test_suit_chars():
    assert "".join(s.char for s in Suit) == SUIT_CHARS
    assert len(set(Suit)) == 4


# --- Card parsing ----------------------------------------------------------


@pytest.mark.parametrize("text", ALL_CARD_STRINGS)
def test_parse_round_trip_all_52(text):
    card = Card.from_str(text)
    assert str(card) == text
    assert Card.from_str(str(card)) == card
    assert Card.from_id(card.id) == card


def test_parse_examples():
    assert Card.from_str("As") == Card(Rank.ACE, Suit.SPADES)
    assert Card.from_str("Td") == Card(Rank.TEN, Suit.DIAMONDS)
    assert Card.from_str("9h") == Card(Rank.NINE, Suit.HEARTS)
    assert Card.from_str("2c") == Card(Rank.TWO, Suit.CLUBS)


@pytest.mark.parametrize("text", ["AS", "aS", "as", " As "])
def test_parse_case_and_whitespace_insensitive(text):
    assert Card.from_str(text) == Card(Rank.ACE, Suit.SPADES)


@pytest.mark.parametrize("text", ["10h", "10H", "th", "TH"])
def test_parse_ten(text):
    assert Card.from_str(text) == Card(Rank.TEN, Suit.HEARTS)


def test_str_of_ten_is_T():
    assert str(Card.from_str("10h")) == "Th"


@pytest.mark.parametrize(
    "text",
    ["", "A", "s", "Ax", "1s", "0s", "11s", "Zs", "AsK", "AAs", "sA", "As Ks", "1", "10"],
)
def test_parse_invalid_rejected(text):
    with pytest.raises(ValueError):
        Card.from_str(text)


@pytest.mark.parametrize("bad", [None, 51, b"As"])
def test_parse_non_string_rejected(bad):
    with pytest.raises(ValueError):
        Card.from_str(bad)


def test_constructor_rejects_bad_values():
    with pytest.raises(ValueError):
        Card(1, Suit.SPADES)
    with pytest.raises(ValueError):
        Card(Rank.ACE, 4)


def test_constructor_coerces_ints():
    card = Card(14, 3)
    assert card == Card(Rank.ACE, Suit.SPADES)
    assert isinstance(card.rank, Rank) and isinstance(card.suit, Suit)


# --- Card ids, hashing, ordering --------------------------------------------


def test_ids_unique_and_cover_0_to_51():
    ids = [Card.from_str(t).id for t in ALL_CARD_STRINGS]
    assert sorted(ids) == list(range(52))


def test_id_layout():
    assert Card.from_str("2c").id == 0
    assert Card.from_str("2s").id == 3
    assert Card.from_str("3c").id == 4
    assert Card.from_str("As").id == 51


@pytest.mark.parametrize("bad", [-1, 52, 3.0, "1"])
def test_from_id_rejects_bad(bad):
    with pytest.raises(ValueError):
        Card.from_id(bad)


def test_full_deck_is_id_order():
    assert [c.id for c in FULL_DECK] == list(range(52))


def test_cards_hashable_and_immutable():
    a, b = Card.from_str("As"), Card.from_str("as")
    assert a == b and hash(a) == hash(b)
    assert len({a, b, Card.from_str("Ks")}) == 2
    with pytest.raises(AttributeError):
        a.rank = Rank.KING  # type: ignore[misc]


def test_cards_order_by_rank_then_suit():
    assert Card.from_str("2s") < Card.from_str("3c")
    assert Card.from_str("Kh") < Card.from_str("Ac")
    assert Card.from_str("Ac") < Card.from_str("As")
    assert sorted(FULL_DECK, reverse=True)[0] == Card.from_str("As")
    # Sorting agrees with ids.
    shuffled = list(FULL_DECK)
    random.Random(0).shuffle(shuffled)
    assert sorted(shuffled) == list(FULL_DECK)


# --- parse_cards ----------------------------------------------------------


@pytest.mark.parametrize("text", ["As Ks", "AsKs", "As,Ks", "  As   ks ", "AS, KS"])
def test_parse_cards_formats(text):
    assert parse_cards(text) == [Card.from_str("As"), Card.from_str("Ks")]


def test_parse_cards_with_tens():
    assert [str(c) for c in parse_cards("10h9h Td 10c")] == ["Th", "9h", "Td", "Tc"]


def test_parse_cards_empty():
    assert parse_cards("") == []
    assert parse_cards("   ") == []


def test_parse_cards_board():
    assert [str(c) for c in parse_cards("Kh 7h 2s")] == ["Kh", "7h", "2s"]


@pytest.mark.parametrize("text", ["As K", "AsKsQ", "Xx", "As Kx", "1s"])
def test_parse_cards_invalid(text):
    with pytest.raises(ValueError):
        parse_cards(text)


def test_parse_cards_round_trip_full_deck():
    text = " ".join(ALL_CARD_STRINGS)
    assert parse_cards(text) == list(FULL_DECK)
    assert parse_cards(text.replace(" ", "")) == list(FULL_DECK)


# --- Deck -------------------------------------------------------------------


def test_deck_has_52_unique_cards():
    deck = Deck(seed=1)
    assert len(deck) == 52
    assert len(set(deck.cards)) == 52
    assert set(deck.cards) == set(FULL_DECK)


def test_deck_is_shuffled_on_construction():
    assert list(Deck(seed=1).cards) != list(FULL_DECK)


def test_same_seed_same_order():
    a, b = Deck(seed=42), Deck(seed=42)
    assert a.cards == b.cards
    a.shuffle()
    b.shuffle()
    assert a.cards == b.cards
    assert a.deal(5) == b.deal(5)


def test_different_seeds_differ():
    orders = {Deck(seed=s).cards for s in range(20)}
    assert len(orders) == 20


def test_deck_does_not_use_global_random():
    random.seed(123)
    before = random.getstate()
    deck = Deck(seed=7)
    deck.shuffle()
    deck.deal(3)
    assert random.getstate() == before

    # And the global state does not influence the deck.
    random.seed(1)
    a = Deck(seed=7).cards
    random.seed(2)
    b = Deck(seed=7).cards
    assert a == b


def test_unseeded_decks_are_valid():
    deck = Deck()
    assert set(deck.cards) == set(FULL_DECK)


def test_deal_takes_from_top():
    deck = Deck(seed=3)
    top = deck.cards[:2]
    hand = deck.deal(2)
    assert tuple(hand) == top
    assert len(deck) == 50
    assert not set(hand) & set(deck.cards)


def test_deal_all_then_nothing_left():
    deck = Deck(seed=3)
    dealt = deck.deal(52)
    assert len(deck) == 0
    assert set(dealt) == set(FULL_DECK)
    assert deck.deal(0) == []


def test_deal_too_many_raises():
    deck = Deck(seed=3)
    deck.deal(50)
    with pytest.raises(ValueError):
        deck.deal(3)
    assert len(deck) == 2  # a failed deal changes nothing


def test_deal_negative_raises():
    with pytest.raises(ValueError):
        Deck(seed=3).deal(-1)


def test_shuffle_keeps_dealt_cards_out():
    deck = Deck(seed=4)
    hand = deck.deal(2)
    deck.shuffle()
    assert len(deck) == 50
    assert not set(hand) & set(deck.cards)


def test_remove_known_cards():
    deck = Deck(seed=5)
    known = parse_cards("As Ks 9h 8h Kh 7h 2s")
    deck.remove(known)
    assert len(deck) == 45
    assert not set(known) & set(deck.cards)
    assert all(c not in deck for c in known)
    assert set(deck.cards) | set(known) == set(FULL_DECK)


def test_remove_keeps_relative_order():
    deck = Deck(seed=5)
    before = deck.cards
    known = parse_cards("As Ks")
    deck.remove(known)
    assert deck.cards == tuple(c for c in before if c not in known)


def test_remove_missing_card_raises_and_changes_nothing():
    deck = Deck(seed=5)
    deck.remove(parse_cards("As"))
    before = deck.cards
    with pytest.raises(ValueError):
        deck.remove(parse_cards("Ks As"))
    assert deck.cards == before


def test_remove_duplicate_raises():
    deck = Deck(seed=5)
    with pytest.raises(ValueError):
        deck.remove(parse_cards("As As"))
    assert len(deck) == 52


def test_seeded_remove_then_deal_is_deterministic():
    def run(seed):
        deck = Deck(seed)
        deck.remove(parse_cards("As Ks 9h 8h"))
        deck.shuffle()
        return deck.deal(5)

    assert run(9) == run(9)


@given(st.integers(min_value=-(2**63), max_value=2**63))
def test_any_seed_gives_permutation_of_52(seed):
    deck = Deck(seed)
    assert len(deck) == 52
    assert sorted(deck.cards) == list(FULL_DECK)
    dealt = deck.deal(7)
    assert len(set(dealt)) == 7
    assert sorted(dealt + list(deck.cards)) == list(FULL_DECK)
