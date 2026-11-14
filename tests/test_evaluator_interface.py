"""Evaluator interface checks that do not need ``evaluate`` (never skipped)."""

from nutsfinder.engine.evaluator import HandCategory, category


def test_hand_category_values():
    assert [c.value for c in HandCategory] == list(range(9))
    assert HandCategory.HIGH_CARD == 0
    assert HandCategory.STRAIGHT_FLUSH == 8


def test_category_reads_first_element():
    assert category((HandCategory.FULL_HOUSE, 3, 2)) is HandCategory.FULL_HOUSE
    assert category((0, 14, 13, 12, 11, 9)) is HandCategory.HIGH_CARD
