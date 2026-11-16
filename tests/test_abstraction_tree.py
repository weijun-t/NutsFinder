"""Tests for the abstract betting tree (``nutsfinder.abstraction.tree``).

The central check replays *every* root-to-leaf path of the tree through the
real engine (``new_hand`` / ``apply_action`` / ``deal_board``) and compares
the chips at every node. The info-set count is pinned (a regression number)
and compared with the docs/phase6_research.md estimate of about 1.3e5,
which is an upper bound: at 100bb, pot-size raises reach all-in quickly,
so many lines of the estimate collapse.

Recorded counts (default config): 2,899 nodes; decision nodes 16 / 76 / 268 /
748 (preflop / flop / turn / river); info sets 2,704 + 3,800 + 13,400 +
37,400 = 57,304.
"""

from __future__ import annotations

import pytest

from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.tree import (
    CHANCE, FOLD, SHOWDOWN, TERMINAL, BettingTree, build_tree,
)
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, apply_action, deal_board, legal_actions, new_hand
from nutsfinder.engine.cards import FULL_DECK

TREE = build_tree()
EXPECTED_INFOSETS = 57_304
NOTES_UPPER_BOUND = 130_000


def _root_state(config: AbstractionConfig) -> HandState:
    return new_hand(button=0, stacks=(config.stack, config.stack),
                    small_blind=config.small_blind, big_blind=config.big_blind)


def _replay(tree: BettingTree):
    """Yield (node, engine state, non-all-in raises per street) for every node, replaying the engine."""
    stack = [(0, _root_state(tree.config), (0, 0, 0, 0))]
    while stack:
        n, state, raises = stack.pop()
        yield n, state, raises
        kids, acts = tree.child_ids(n), tree.actions(n)
        if tree.player[n] == CHANCE:
            k = len(state.board)
            stack.append((kids[0], deal_board(state, FULL_DECK[k:k + state.cards_needed]), raises))
            continue
        for a, c in zip(acts, kids):
            legal = legal_actions(state)
            assert legal.is_legal(a), (n, a, legal)
            child = apply_action(state, tree.player[n], a)
            r = list(raises)
            if isinstance(a, (Bet, Raise)) and child.stacks[tree.player[n]] > 0:
                r[int(state.street)] += 1
            stack.append((c, child, tuple(r)))


@pytest.mark.parametrize("config", [
    DEFAULT_CONFIG,
    AbstractionConfig(postflop_fractions=(0.5, 1.0)),
    AbstractionConfig(stack=2_000, preflop_fractions=(0.75, 1.5), postflop_cap=3),
], ids=["default", "half-pot", "20bb"])
def test_every_path_replays_legally_with_matching_chips(config):
    tree = TREE if config == DEFAULT_CONFIG else build_tree(config)
    seen = 0
    for n, state, raises in _replay(tree):
        seen += 1
        assert state.total_bets == tree.contrib[n]
        assert tree.street[n] == int(state.street)
        assert sum(state.stacks) + state.pot == 2 * config.stack  # chips conserved
        assert raises[tree.street[n]] <= config.cap(tree.street[n])
        p = tree.player[n]
        if p == TERMINAL:
            assert state.is_hand_over or (state.to_act is None and min(state.stacks) == 0)
            assert tree.n_children[n] == 0
        elif p == CHANCE:
            assert state.cards_needed > 0 and min(state.stacks) > 0
        else:
            assert state.to_act == p
    assert seen == tree.n_nodes


def test_terminal_payoffs_are_zero_sum_and_bounded():
    n_terminals = 0
    for n in range(TREE.n_nodes):
        if TREE.player[n] != TERMINAL:
            continue
        n_terminals += 1
        a, b = TREE.contrib[n]
        # Uncalled chips are refunded, so both seats have the same amount at stake.
        assert 0 < a == b <= DEFAULT_CONFIG.stack
        # Seat 0 gets u and seat 1 gets -u (zero-sum); |u| is the loser's contribution.
        if TREE.terminal[n] == FOLD:
            assert TREE.fold_value[n] == (-a if TREE.player[TREE.parent[n]] == 0 else b)
        else:
            assert TREE.terminal[n] == SHOWDOWN and TREE.stake[n] == a
    assert n_terminals == TREE.counts()["terminals"]


def test_actions_unique_ordered_and_fold_only_facing_a_bet():
    for n in range(TREE.n_nodes):
        if TREE.player[n] not in (0, 1):
            continue
        acts = TREE.actions(n)
        assert len(set(acts)) == len(acts)
        facing = Call() in acts
        assert (Fold() in acts) == facing and (Check() in acts) != facing
        sizes = [a.amount if isinstance(a, Bet) else a.to for a in acts if isinstance(a, (Bet, Raise))]
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_all_in_offered_whenever_any_raise_is_legal():
    for n, state, _ in _replay(TREE):
        if TREE.player[n] not in (0, 1):
            continue
        legal = legal_actions(state)
        aggressive = [a for a in TREE.actions(n) if isinstance(a, (Bet, Raise))]
        if legal.can_bet or legal.can_raise:
            last = aggressive[-1]
            assert (last.amount if isinstance(last, Bet) else last.to) == legal.max_to
        else:
            assert not aggressive


def test_preflop_root_and_heads_up_order():
    acts = [str(a) for a in TREE.actions(0)]
    assert TREE.player[0] == 0  # the button (small blind) acts first preflop
    assert acts == ["fold", "call", "raise to 300", "raise to 10000"]
    limp = TREE.child_ids(0)[1]
    assert TREE.player[limp] == 1 and [str(a) for a in TREE.actions(limp)] == [
        "check", "raise to 300", "raise to 10000"]  # BB option; 3bb is the first raise
    flop = TREE.child_ids(TREE.child_ids(limp)[0])[0]  # limp, check, (street transition)
    assert TREE.street[flop] == 1 and TREE.player[flop] == 1  # BB acts first postflop


def test_pot_raise_sizes_match_the_worked_example():
    n = TREE.child_ids(0)[2]  # SB opens to 300
    assert [str(a) for a in TREE.actions(n)] == ["fold", "call", "raise to 900", "raise to 10000"]
    n = TREE.child_ids(n)[2]
    assert [str(a) for a in TREE.actions(n)] == ["fold", "call", "raise to 2700", "raise to 10000"]
    n = TREE.child_ids(n)[2]  # open, 3-bet, 4-bet: three non-all-in raises, cap reached
    assert [str(a) for a in TREE.actions(n)] == ["fold", "call", "raise to 10000"]


def test_infoset_count_against_the_notes():
    c = TREE.counts()
    print("\nabstract tree counts:", c)
    assert c["infosets"] == EXPECTED_INFOSETS == TREE.n_infosets
    assert c["infosets"] < NOTES_UPPER_BOUND
    assert c["player0_infosets"] + c["player1_infosets"] == c["infosets"]
    assert c["preflop_infosets"] + c["flop_infosets"] + c["turn_infosets"] + c["river_infosets"] == c["infosets"]


def test_infoset_indexing_is_a_bijection():
    seen = set()
    for n in range(TREE.n_nodes):
        if TREE.player[n] in (0, 1):
            for b in range(DEFAULT_CONFIG.buckets(TREE.street[n])):
                k = TREE.infoset(n, b)
                assert k not in seen
                seen.add(k)
                if b in (0, 7) and n % 50 == 0:
                    assert TREE.decode_infoset(k) == (n, b)
    assert seen == set(range(TREE.n_infosets))
    with pytest.raises(ValueError):
        TREE.infoset(0, 169)


def test_path_reconstructs_actions():
    leaf = max(range(TREE.n_nodes), key=lambda n: len(TREE.path(n)))
    state = _root_state(DEFAULT_CONFIG)
    for a in TREE.path(leaf):
        while state.cards_needed:
            k = len(state.board)
            state = deal_board(state, FULL_DECK[k:k + state.cards_needed])
        state = apply_action(state, state.to_act, a)
    assert state.total_bets == TREE.contrib[leaf]


def test_config_validation():
    with pytest.raises(ValueError):
        AbstractionConfig(preflop_open_bb=1.0)
    with pytest.raises(ValueError):
        AbstractionConfig(postflop_buckets=(50, 50))
    with pytest.raises(ValueError):
        AbstractionConfig(ehs_method="emd")


def test_as_arrays_round_trips_the_lists():
    arr = TREE.as_arrays()
    assert arr["player"].tolist() == TREE.player and arr["children"].tolist() == TREE.children
    assert arr["contrib"].shape == (TREE.n_nodes, 2)
    assert len(arr["edge_kind"]) == len(arr["edge_to"]) == len(TREE.children)
    for n in (0, 2, 13):
        s = TREE.first_child[n]
        for j, a in enumerate(TREE.actions(n)):
            to = a.amount if isinstance(a, Bet) else a.to if isinstance(a, Raise) else 0
            assert arr["edge_to"][s + j] == to
