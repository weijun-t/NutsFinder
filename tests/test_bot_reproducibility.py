"""Every bot must be reproducible from the match's master seed.

A bot with its own RNG must override ``Bot.reseed``. If it forgets, a
no-argument factory seeds it from OS entropy, so two runs with the same
master seed give different hands: results stop being reproducible and the
two copies of a bot in a duplicate match stop making the same choices.

This test plays every exported bot twice, with the same master seed, against
RandomBot and against itself, and requires identical results. A new bot is
covered automatically as soon as it is exported from ``nutsfinder.bots``.
"""

from __future__ import annotations

import inspect

import pytest

import nutsfinder.bots as bots
from nutsfinder.engine.bot import Bot
from nutsfinder.eval.match import duplicate_match, plain_match

# CFRBot() with no model plays the untrained uniform blueprint and warns about it; that is expected
# here (this test is about seeding, not strength).
pytestmark = pytest.mark.filterwarnings("ignore::nutsfinder.bots.cfr_bot.UntrainedBlueprintWarning")

ALL_BOTS = [
    obj
    for name in bots.__all__
    if inspect.isclass(obj := getattr(bots, name)) and issubclass(obj, Bot)
]


def test_all_bots_are_found() -> None:
    assert {b.__name__ for b in ALL_BOTS} >= {
        "RandomBot",
        "CallStationBot",
        "AlwaysRaiseBot",
        "PotOddsBot",
        "EquityBot",
    }


@pytest.mark.parametrize("bot", ALL_BOTS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("opponent", ["RandomBot", "self"])
def test_same_master_seed_gives_identical_matches(bot: type[Bot], opponent: str) -> None:
    other = bot if opponent == "self" else bots.RandomBot
    for match, n in ((duplicate_match, 25), (plain_match, 50)):
        first = match(bot, other, n, master_seed=2024)
        second = match(bot, other, n, master_seed=2024)
        assert first.unit_results == second.unit_results
        assert [g.log for g in first.games] == [g.log for g in second.games]
