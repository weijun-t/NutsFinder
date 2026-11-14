"""Shared test helpers.

The evaluator and the equity calculator are stubs until the project owner
writes them (PLAN.md rule 4). Their test modules opt in to the fixtures below
with ``pytestmark = pytest.mark.usefixtures(...)``: each fixture calls the real
function once on a small sample input, and if that raises NotImplementedError
the whole module is SKIPPED (not failed). Any other exception propagates, so a
half-finished implementation fails loudly instead of being skipped.
"""

from collections.abc import Callable
from typing import Any

import pytest

from nutsfinder.engine.cards import parse_cards
from nutsfinder.engine.equity import monte_carlo_equity
from nutsfinder.engine.evaluator import evaluate

NOT_IMPLEMENTED_REASON = "not implemented yet (Phase 1, owner writes this)"


def skip_if_not_implemented(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Call ``func(*args, **kwargs)`` once; skip the calling test if it is still a stub."""
    try:
        func(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(NOT_IMPLEMENTED_REASON)


@pytest.fixture(scope="module")
def evaluator_implemented() -> None:
    skip_if_not_implemented(evaluate, parse_cards("As Ks Qs Js Ts"))


@pytest.fixture(scope="module")
def equity_implemented() -> None:
    skip_if_not_implemented(
        monte_carlo_equity, parse_cards("As Ks"), parse_cards("9h 8h"), samples=10, seed=0
    )
