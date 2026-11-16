"""Tests for ``AbstractHoldem`` (the Game adapter) and the MCCFR chance-sampling hook.

- Spy tests: an info-set key never depends on the opponent's hole cards or
  on board cards not yet dealt on that street, and the card abstraction is
  only ever asked about one player's own cards.
- Utility uses the real evaluator on the real cards; the game is zero-sum.
- ``sample_chance`` hook: MCCFR on a Kuhn variant that only *samples* its
  deals converges like Kuhn; the Leduc/Kuhn regression numbers in
  ``tests/test_mccfr.py`` (unchanged) show games without the hook consume
  the same random stream as before.
- A short seeded MCCFR run on AbstractHoldem is deterministic. The slow test
  measures single-core iterations per second (wall time on this laptop is
  noisy: thermal throttling).
"""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from nutsfinder.abstraction.cards import CardAbstraction
from nutsfinder.abstraction.holdem_game import AbstractHoldem, Deal
from nutsfinder.abstraction.tree import TERMINAL
from nutsfinder.gametheory import CHANCE, Kuhn, exploitability
from nutsfinder.gametheory.mccfr import MCCFRSolver

GAME = AbstractHoldem()


def _decision_nodes(game, player=None, street=None):
    t = game.tree
    return [n for n in range(t.n_nodes) if t.player[n] in (0, 1)
            and (player is None or t.player[n] == player) and (street is None or t.street[n] == street)]


def _random_deal(rng):
    c = rng.sample(range(52), 9)
    return (c[0], c[1]), (c[2], c[3]), tuple(c[4:])


# --- spy tests ------------------------------------------------------------------


class SpyCards(CardAbstraction):
    """Records every (hole, board) the game asks the card abstraction about."""

    def __init__(self, base: CardAbstraction) -> None:
        super().__init__(base.config, base.edges, base.tables)
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def bucket(self, hole, board):
        self.calls.append((tuple(hole), tuple(board)))
        return super().bucket(hole, board)


def test_card_abstraction_only_sees_own_cards_and_dealt_board():
    spy = SpyCards(GAME.cards)
    game = AbstractHoldem(cards=spy, tree=GAME.tree)
    h0, h1, board = (51, 47), (0, 1), (8, 21, 46, 30, 13)
    game.make_deal(h0, h1, board)
    assert len(spy.calls) == 8
    for hole, seen in spy.calls:
        assert hole in (h0, h1)
        assert seen == board[:len(seen)] and len(seen) in (0, 3, 4, 5)
        other = h1 if hole == h0 else h0
        assert not set(other) & set(hole + seen)


def test_infoset_keys_ignore_opponent_cards_and_future_board():
    rng = random.Random(4)
    for _ in range(5):
        h0, h1, board = _random_deal(rng)
        base = GAME.make_deal(h0, h1, board)
        free = [c for c in range(52) if c not in h0 + h1 + board]
        new_h1 = tuple(rng.sample(free, 2))
        new_h0 = tuple(rng.sample([c for c in free if c not in new_h1], 2))
        swapped1 = GAME.make_deal(h0, new_h1, board)   # seat 1's cards changed
        swapped0 = GAME.make_deal(new_h0, h1, board)   # seat 0's cards changed
        for n in _decision_nodes(GAME):
            p = GAME.tree.player[n]
            same = swapped1 if p == 0 else swapped0
            assert GAME.infoset_key((n, base)) == GAME.infoset_key((n, same))
        # Future board cards: change the river (and turn) and compare earlier streets.
        later = [c for c in free if c not in new_h0 + new_h1]
        new_river = board[:4] + (later[0],)
        new_turn = board[:3] + (later[1], later[0])
        d_river, d_turn = GAME.make_deal(h0, h1, new_river), GAME.make_deal(h0, h1, new_turn)
        for n in _decision_nodes(GAME):
            s = GAME.tree.street[n]
            if s <= 2:
                assert GAME.infoset_key((n, base)) == GAME.infoset_key((n, d_river))
            if s <= 1:
                assert GAME.infoset_key((n, base)) == GAME.infoset_key((n, d_turn))


def test_infoset_keys_are_tree_indices():
    deal = GAME.make_deal((51, 50), (0, 5), (8, 21, 46, 30, 13))  # As Ah vs 2c 3d
    assert GAME.infoset_key((0, deal)) == GAME.tree.infoset(0, 0)  # AA is preflop bucket 0
    for n in _decision_nodes(GAME):
        k = GAME.infoset_key((n, deal))
        assert GAME.tree.decode_infoset(k)[0] == n and 0 <= k < GAME.tree.n_infosets


# --- utility --------------------------------------------------------------------


def _walk_to(game, deal, actions_by_str):
    h = game.next(game.initial(), deal)
    for s in actions_by_str:
        names = [str(a) for a in game.tree.actions(h[0])]
        h = game.next(h, names.index(s))
    return h


def test_showdown_uses_the_real_evaluator():
    # As Ah vs 7c 2d on Kd 9s 4c 3h 2s: aces beat a pair of deuces.
    deal = GAME.make_deal((51, 50), (20, 1), (45, 31, 8, 6, 3))
    assert deal.result == 1
    line = ["raise to 300", "call"] + ["check", "check"] * 3
    h = _walk_to(GAME, deal, line)
    assert GAME.is_terminal(h) and GAME.utility(h) == 300.0
    # Same line, the cards swapped: seat 0 loses 300.
    h = _walk_to(GAME, GAME.make_deal((20, 1), (51, 50), (45, 31, 8, 6, 3)), line)
    assert GAME.utility(h) == -300.0


def test_split_pot_and_fold_utilities():
    # Board royal flush in spades plays for both: a split.
    deal = GAME.make_deal((0, 1), (4, 5), (35, 39, 43, 47, 51))
    assert deal.result == 0
    assert GAME.utility(_walk_to(GAME, deal, ["call", "raise to 10000", "call"])) == 0.0
    assert GAME.utility(_walk_to(GAME, deal, ["fold"])) == -50.0
    assert GAME.utility(_walk_to(GAME, deal, ["raise to 300", "fold"])) == 100.0


def test_protocol_shape():
    h = GAME.initial()
    assert GAME.player(h) == CHANCE and not GAME.is_terminal(h)
    with pytest.raises(NotImplementedError):
        GAME.chance_outcomes(h)
    deal = GAME.sample_chance(h, np.random.default_rng(0))
    assert isinstance(deal, Deal) and len(set(deal.hole[0] + deal.hole[1] + deal.board)) == 9
    root = GAME.next(h, deal)
    assert GAME.player(root) == 0 and GAME.actions(root) == (0, 1, 2, 3)
    # Street transitions are skipped: every reachable node is a decision or a terminal.
    stack = [root]
    while stack:
        x = stack.pop()
        assert GAME.tree.player[x[0]] in (0, 1, TERMINAL)
        if not GAME.is_terminal(x):
            stack += [GAME.next(x, a) for a in GAME.actions(x)]
    with pytest.raises(ValueError):
        GAME.make_deal((0, 1), (1, 2), (3, 4, 5, 6, 7))


# --- the sample_chance hook ----------------------------------------------------------


class SampledKuhn(Kuhn):
    """Kuhn whose solver may only sample chance (``chance_outcomes`` kept for exact BR)."""

    calls = 0

    def sample_chance(self, h, rng):
        SampledKuhn.calls += 1
        outs = self.chance_outcomes(h)
        j = rng.choice(len(outs), p=[p for _, p in outs])
        return outs[j][0]


def test_mccfr_uses_sample_chance_hook_and_converges():
    game = SampledKuhn()
    s = MCCFRSolver(game, seed=3)
    s.run(10_000)
    assert SampledKuhn.calls >= 2 * 2 * 10_000  # two chance nodes per traversal
    assert exploitability(game, s.average_strategy()) < 2.6e-2  # Kuhn plain threshold of test_mccfr
    a, b = MCCFRSolver(SampledKuhn(), seed=4), MCCFRSolver(SampledKuhn(), seed=4)
    a.run(500)
    b.run(500)
    assert a.tables() == b.tables()  # the chance stream is seeded too


def test_games_without_the_hook_are_unchanged():
    s = MCCFRSolver(Kuhn(), seed=0)
    assert s._sample_chance is None and s._chance_rng is None


# --- MCCFR on abstract hold'em ------------------------------------------------------


def test_mccfr_smoke_run_is_deterministic():
    a = MCCFRSolver(GAME, seed=11, linear=True)
    a.run(30)
    b = MCCFRSolver(GAME, seed=11, linear=True)
    b.run(30)
    assert a.keys == b.keys and np.array_equal(a.regrets, b.regrets)
    assert np.array_equal(a.strategy_sum, b.strategy_sum)
    assert all(0 <= k < GAME.tree.n_infosets for k in a.keys)
    assert np.all(np.isfinite(a.regrets)) and np.abs(a.regrets).max() > 0
    c = MCCFRSolver(GAME, seed=12, linear=True)
    c.run(30)
    assert not np.array_equal(c.regrets[:10], a.regrets[:10]) or c.keys != a.keys


@pytest.mark.slow
def test_mccfr_iterations_per_second():
    s = MCCFRSolver(GAME, seed=0, linear=True)
    s.run(300)  # warm-up: fills turn tables and allocates info sets
    start = time.perf_counter()
    s.run(300)
    rate = 300 / (time.perf_counter() - start)
    print(f"\nMCCFR on AbstractHoldem: {rate:.1f} it/s, {s.nodes / s.iteration:.0f} nodes/it, "
          f"{len(s.keys)} info sets after {s.iteration} it")
    assert rate > 1


# --- config consistency (step 7 review fix) --------------------------------------------


def test_bucket_count_mismatch_between_tree_and_config_raises():
    from nutsfinder.abstraction.config import AbstractionConfig
    from nutsfinder.abstraction.tree import build_tree
    with pytest.raises(ValueError, match="bucket counts disagree on street 1"):
        AbstractHoldem(tree=build_tree(AbstractionConfig(postflop_buckets=(20, 20, 20))))


def test_bucket_count_mismatch_with_card_abstraction_raises_and_matching_configs_work():
    from nutsfinder.abstraction.config import AbstractionConfig
    cfg = AbstractionConfig(postflop_buckets=(20, 30, 40))
    edges = {s: tuple(np.linspace(0.05, 0.95, n - 1).tolist()) for s, n in zip((1, 2, 3), (20, 30, 40))}
    cards = CardAbstraction(cfg, edges=edges, tables=GAME.cards.tables)
    with pytest.raises(ValueError, match="cards.config 20"):
        AbstractHoldem(cards=cards, tree=GAME.tree)  # default config + default tree, 20-bucket cards
    game = AbstractHoldem(cfg, cards=cards)
    deal = game.make_deal((51, 47), (0, 1), (8, 21, 46, 30, 13))
    assert all(0 <= deal.buckets[p][s] < cfg.buckets(s) for p in (0, 1) for s in range(4))
