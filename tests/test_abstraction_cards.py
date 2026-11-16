"""Tests for the card abstraction (``nutsfinder.abstraction.strength`` and ``.cards``).

- The vectorised 7-card strength agrees with the engine's evaluator (random
  hands here; every one of the 133,784,560 hands, by category count, in the
  slow test).
- Exact E[HS] agrees with the engine's equity code: exactly on the river
  (``equity_vs_range`` enumerates the 990 opponent hands there), by brute
  force over 46 rivers on the turn, within 4 standard errors of a long Monte
  Carlo run on the flop.
- Suit isomorphism: relabelling suits never changes E[HS] or the bucket.
- Buckets: preflop = the 169 classes; postflop monotone in E[HS] and equal
  frequency within +/-1 percentage point on fresh samples (flop here; turn
  and river in slow tests, since each fresh turn or river board costs a
  table). River E[HS] has atoms (multiples of 1/1980, single values holding
  up to ~1.1% of deals), so river shares spread over 1.5%-2.5%.
- Determinism: fresh tables, the flop file and the in-memory computation
  give identical buckets; Monte Carlo E[HS] is reproducible.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pytest

from nutsfinder.abstraction import strength
from nutsfinder.abstraction.bucket_data import EDGES
from nutsfinder.abstraction.cards import (
    FLOP_FILE, SUIT_PERMS, CardAbstraction, EHSTables, canonical_board, ehs_mc,
    equal_frequency_edges, preflop_bucket, sample_deals,
)
from nutsfinder.abstraction.config import AbstractionConfig
from nutsfinder.engine.cards import FULL_DECK
from nutsfinder.engine.equity import RANDOM_RANGE, equity_vs_range
from nutsfinder.engine.evaluator import HandCategory, evaluate
from nutsfinder.engine.preflop import ALL_CLASSES, hand_class

TABLES = EHSTables()        # loads data/flop_ehs.npz when present
CARDS = CardAbstraction(tables=TABLES)


def cards(ids):
    return [FULL_DECK[c] for c in ids]


def relabel(ids, k):
    perm = SUIT_PERMS[k]
    return [(c & ~3) | perm[c & 3] for c in ids]


# --- strength -------------------------------------------------------------------


def test_strength7_matches_evaluate_on_random_hands():
    _, _, _, values = strength.strength_tables()
    assert len(values) == 4824  # distinct 7-card hand values
    rng = random.Random(7)
    hands = np.array([rng.sample(range(52), 7) for _ in range(5000)])
    for h, s in zip(hands.tolist(), strength.strength7(hands).tolist()):
        assert values[s] == evaluate(cards(h))


def test_strength7_on_named_hands():
    _, _, _, values = strength.strength_tables()
    royal = [FULL_DECK.index(c) for c in FULL_DECK if c.suit == 3 and c.rank >= 10] + [0, 5]
    wheel_sf = [48, 0, 4, 8, 12, 51, 50]  # Ac 2c 3c 4c 5c + As Ah: the 5-high straight flush
    s = strength.strength7(np.array([royal, wheel_sf]))
    assert values[s[0]] == (HandCategory.STRAIGHT_FLUSH, 14)
    assert values[s[1]] == (HandCategory.STRAIGHT_FLUSH, 5)


@pytest.mark.slow
def test_strength7_category_counts_over_all_7_card_hands():
    """All 133,784,560 hands, each once: lowest two ids (a < b), then 5 of the cards above b."""
    _, _, _, values = strength.strength_tables()
    cat = np.array([v[0] for v in values])
    royal = values.index((HandCategory.STRAIGHT_FLUSH, 14))
    five = np.array(list(combinations(range(50), 5)), dtype=np.int64)
    counts = np.zeros(9, dtype=np.int64)
    n_royal = 0
    for b in range(1, 47):
        rest = five[five[:, 4] < 51 - b] + b + 1  # 5 distinct cards from b+1..51
        for a in range(b):
            for chunk in np.array_split(rest, max(1, len(rest) // 250_000)):
                hands = np.empty((len(chunk), 7), dtype=np.int64)
                hands[:, 0], hands[:, 1], hands[:, 2:] = a, b, chunk
                s = strength.strength7(hands)
                counts += np.bincount(cat[s], minlength=9)
                n_royal += int((s == royal).sum())
    assert counts.sum() == 133_784_560
    assert counts.tolist() == [23_294_460, 58_627_800, 31_433_400, 6_461_620, 6_180_020,
                               4_047_644, 3_473_184, 224_848, 41_584]
    assert n_royal == 4_324


# --- exact E[HS] ------------------------------------------------------------------


def test_river_hs_matches_equity_vs_range_exactly():
    rng = random.Random(11)
    for _ in range(6):
        ids = rng.sample(range(52), 7)
        exact = equity_vs_range(cards(ids[:2]), RANDOM_RANGE, cards(ids[2:])).equity
        assert strength.river_hs(ids[2:])[strength.combo_index(*ids[:2])] == pytest.approx(exact, abs=1e-12)
        assert TABLES.ehs(ids[:2], ids[2:]) == pytest.approx(exact, abs=1 / 65534)


def test_turn_ehs_matches_brute_force_over_rivers():
    hole, board = [51, 47], [0, 21, 46, 30]  # As Ks on 2c 7d Kh 9h
    rivers = [c for c in range(52) if c not in hole + board]
    brute = np.mean([equity_vs_range(cards(hole), RANDOM_RANGE, cards(board + [r])).equity for r in rivers])
    assert TABLES.ehs(hole, board) == pytest.approx(brute, abs=1 / 65534)


def test_flop_ehs_matches_long_monte_carlo():
    rng = random.Random(5)
    for _ in range(2):
        ids = rng.sample(range(52), 5)
        mc = equity_vs_range(cards(ids[:2]), RANDOM_RANGE, cards(ids[2:]), samples=40_000, seed=1)
        assert abs(TABLES.ehs(ids[:2], ids[2:]) - mc.equity) < 4 * mc.std_error


def test_suit_relabelling_never_changes_ehs():
    rng = random.Random(3)
    for n_board in (3, 4, 5):
        ids = rng.sample(range(52), 2 + n_board)
        base = TABLES.ehs(ids[:2], ids[2:])
        for k in rng.sample(range(24), 6):
            r = relabel(ids, k)
            assert TABLES.ehs(r[:2], r[2:]) == base
            assert canonical_board(r[2:])[0] == canonical_board(ids[2:])[0]


def test_canonical_board_counts():
    assert len(EHSTables.all_canonical_flops()) == 1755


def test_flop_file_matches_in_memory_computation():
    if not FLOP_FILE.exists():
        pytest.skip("data/flop_ehs.npz not generated (examples/make_bucket_edges.py)")
    fresh = EHSTables(flop_file=None)
    flop = TABLES.all_canonical_flops()[700]
    assert np.array_equal(fresh.flop_table(flop), TABLES.flop_table(flop))


# --- buckets ----------------------------------------------------------------------


def test_preflop_buckets_are_the_169_classes():
    seen = set()
    for a, b in combinations(range(52), 2):
        k = preflop_bucket((a, b))
        assert ALL_CLASSES[k] == hand_class(cards((a, b)))
        assert k == preflop_bucket((b, a)) == CARDS.bucket((a, b), ())
        seen.add(k)
    assert seen == set(range(169))


def test_edges_shape():
    for street in (1, 2, 3):
        e = EDGES[street]
        assert len(e) == 49 and list(e) == sorted(set(e)) and 0.0 < e[0] and e[-1] < 1.0


@pytest.mark.parametrize("n_board", [3, 4, 5])
def test_buckets_monotone_in_ehs(n_board):
    # Every hand on a few boards (one table per board), plus the edges themselves.
    rng = random.Random(n_board)
    pairs = []
    for _ in range(3):
        board = rng.sample(range(52), n_board)
        for a, b in combinations([c for c in range(52) if c not in board], 2):
            pairs.append((TABLES.ehs((a, b), board), CARDS.bucket((a, b), board)))
    pairs.sort()
    buckets = [b for _, b in pairs]
    assert buckets == sorted(buckets)
    assert len(set(buckets)) > 30


def _shares(street, n, seed):
    deals = sample_deals(street, n, seed)
    counts = np.bincount([CARDS.bucket(h, b) for h, b in deals], minlength=50)
    return counts / n


@pytest.mark.slow
def test_equal_frequency_on_fresh_river_samples():
    # ~0.15 ms per river lookup, but this laptop throttles to ~1.3 ms under sustained load.
    share = _shares(3, 20_000, seed=1001)  # not the generator's seed
    assert np.all(np.abs(share - 0.02) < 0.01), share


def test_equal_frequency_on_fresh_flop_samples():
    if not FLOP_FILE.exists():
        pytest.skip("needs data/flop_ehs.npz (else every fresh flop costs a table)")
    share = _shares(1, 20_000, seed=1001)
    assert np.all(np.abs(share - 0.02) < 0.01), share


@pytest.mark.slow
def test_equal_frequency_on_fresh_turn_samples():
    share = _shares(2, 20_000, seed=1001)
    assert np.all(np.abs(share - 0.02) < 0.01), share


def test_equal_frequency_edges_handles_atoms():
    assert equal_frequency_edges([0.1, 0.2, 0.3, 0.4], 2) == (0.25,)
    assert equal_frequency_edges([0.1, 0.5, 0.5, 0.5, 0.9, 0.9], 2) == (0.7,)
    with pytest.raises(ValueError):
        equal_frequency_edges([0.5] * 10, 3)


def test_bucket_lookup_is_deterministic_across_instances():
    other = CardAbstraction(tables=EHSTables())  # fresh caches (file vs memory: see above)
    rng = random.Random(9)
    for n_board in (0, 3, 4, 5):
        for _ in range(3):
            ids = rng.sample(range(52), 2 + n_board)
            assert CARDS.bucket(ids[:2], ids[2:]) == other.bucket(ids[:2], ids[2:]) \
                == CARDS.bucket(ids[:2], ids[2:])


def test_bucket_cards_accepts_card_objects():
    ids = [51, 47, 0, 21, 46]
    assert CARDS.bucket_cards(cards(ids[:2]), cards(ids[2:])) == CARDS.bucket(ids[:2], ids[2:])


# --- Monte Carlo E[HS] --------------------------------------------------------------


def test_ehs_mc_reproducible_isomorphism_invariant_and_unbiased():
    hole, board = [51, 47], [0, 21, 46, 30]
    a = ehs_mc(hole, board, 2000)
    assert a == ehs_mc(hole, board, 2000)
    r = relabel(hole + board, 5)
    assert ehs_mc(r[:2], r[2:], 2000) == a  # same canonical (hand, board) -> same stream
    assert ehs_mc(hole, board, 2000, salt=1) != a
    se = 0.5 / np.sqrt(2000)
    assert abs(a - TABLES.ehs(hole, board)) < 4 * se


def test_mc_method_config_is_usable():
    mc = CardAbstraction(AbstractionConfig(ehs_method="mc", mc_samples=500), tables=TABLES)
    b = mc.bucket([51, 47], [0, 21, 46])
    assert 0 <= b < 50 and b == mc.bucket([51, 47], [0, 21, 46])


def test_turn_tables_save_and_load_roundtrip(tmp_path):
    t = EHSTables(flop_file=None)
    boards = [canonical_board(b)[0] for b in ([0, 21, 46, 30], [4, 8, 12, 51])]
    for b in boards:
        t.turn_table(b)
    t.save_turns(tmp_path / "turn.npz")
    u = EHSTables(flop_file=None)
    u.load_turns(tmp_path / "turn.npz")
    assert set(u.turn) == set(boards)
    assert all(np.array_equal(u.turn[b], t.turn[b]) for b in boards)
