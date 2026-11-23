"""Tests for RaiseOrFoldBot: folds to any bet/raise, otherwise pot-sized aggression; what BayesBot learns from it."""

from __future__ import annotations

from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, RaiseOrFoldBot
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.runner import play_hand
from nutsfinder.eval.match import duplicate_match


def test_small_blind_raises_pot_and_big_blind_folds_to_a_raise():
    # seat 0 on the button = small blind, acts first preflop
    log = play_hand([RaiseOrFoldBot(), RaiseOrFoldBot()], seed=1, button=0)
    first, second = log.actions[0], log.actions[1]
    assert first.player == 0 and isinstance(first.action, Raise) and first.action.to == 300
    assert second.player == 1 and isinstance(second.action, Fold)
    assert len(log.actions) == 2


def test_raises_a_limp_and_bets_when_checked_to_never_folds_for_free():
    # Against a calling station: RoF raises preflop, bets every street, station calls -> showdown.
    for button in (0, 1):
        log = play_hand([RaiseOrFoldBot(), CallStationBot()], seed=3, button=button)
        assert log.folded is None
        rof = [r for r in log.actions if r.player == 0]
        assert all(isinstance(r.action, (Bet, Raise, Check, Call)) for r in rof)
        assert any(isinstance(r.action, (Bet, Raise)) for r in rof)
        streets = {r.street for r in rof if isinstance(r.action, (Bet, Raise))}
        assert len(streets) >= 2  # aggressive on several streets (until all-in)


def test_folds_to_any_bet_or_raise():
    for seed in range(5):
        for button in (0, 1):
            log = play_hand([RaiseOrFoldBot(), AlwaysRaiseBot()], seed=seed, button=button)
            assert log.folded == 0  # AlwaysRaise always re-raises or bets into it
            last = log.actions[-1]
            assert last.player == 0 and isinstance(last.action, Fold)


def test_bayesbot_learns_high_fold_rates_from_it():
    seen = []

    class Keep(BayesBot):
        def __init__(self):
            super().__init__(samples=30, bluff_samples=30)
            seen.append(self)

    duplicate_match(Keep, RaiseOrFoldBot, 40, master_seed=12)
    assert len(seen) == 2
    for b in seen:
        folds = [b.model[k] for k in ("fold_pre", "fold_post", "fold_pre_big", "fold_post_big")]
        assert sum(st.n for st in folds) > 0 and all(st.yes == st.n for st in folds)  # every chance was a fold
        assert b.model["agg_pre"].mean > 0.6  # raises whenever unopposed (its big-blind folds count as "no")
