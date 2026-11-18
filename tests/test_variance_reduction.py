"""Phase 7 variance reduction: exact heads-up equity, the luck terms, and their unbiasedness."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pytest

from nutsfinder.abstraction.strength import strength7
from nutsfinder.bots import PotOddsBot, RandomBot
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import apply_action, deal_board, legal_actions, new_hand
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.engine.hand_log import make_log
from nutsfinder.eval import hu_equity as H
from nutsfinder.eval.match import duplicate_match
from nutsfinder.eval.variance_reduction import compare_estimators, hand_luck, session_adjustments

# AhAs vs KdKc: all 1,712,304 boards with strength7 one by one (an independent code path), computed once.
AA_VS_KK_OFFSUIT = 0.8125548967940273


def ids(s: str) -> list[int]:
    return [H.card_id(c) for c in parse_cards(s)]


# --- Exact equity ---------------------------------------------------------------------------


def test_aa_vs_kk_preflop():
    e = H.equity_hu(parse_cards("AhAs"), parse_cards("KdKc"))
    assert e == pytest.approx(AA_VS_KK_OFFSUIT, abs=1e-12)
    assert 0.81 < H.equity_hu(parse_cards("AhAs"), parse_cards("KsKh")) < 0.83  # the "about 0.82" of the books
    assert H.equity_hu(parse_cards("KdKc"), parse_cards("AhAs")) == pytest.approx(1 - AA_VS_KK_OFFSUIT, abs=1e-12)


def test_preflop_board_table_matches_strength7():
    rng = np.random.default_rng(31_415_926)
    t = H._board_tables()
    for _ in range(4):
        cards = rng.choice(52, size=4, replace=False).tolist()
        hole = cards[:2]
        dead = sum(1 << x for x in cards)
        sel = np.flatnonzero((t["cardmask"] & dead) == 0)
        sel = np.sort(rng.choice(sel, size=3000, replace=False))
        b = _all_boards()[sel]
        assert not np.isin(b, cards).any()
        want = strength7(np.concatenate([np.tile(hole, (len(b), 1)), b], axis=1))
        got = H._strength_preflop(hole, sel, t)
        np.testing.assert_array_equal(got, want)


_BOARDS = None


def _all_boards():
    global _BOARDS
    if _BOARDS is None:
        _BOARDS = np.array(list(combinations(range(52), 5)), dtype=np.int64)
    return _BOARDS


def test_preflop_equity_suit_and_order_invariant():
    a, b = ids("Ah7h"), ids("Kd7c")
    e = H.preflop_equity_hu(a, b)
    for p in list(permutations(range(4)))[::5]:
        pa = [4 * (x // 4) + p[x % 4] for x in a]
        pb = [4 * (x // 4) + p[x % 4] for x in b]
        H._PREFLOP_CACHE.clear()
        assert H.preflop_equity_hu(pa[::-1], pb) == pytest.approx(e, abs=1e-12)
    assert H.preflop_equity_hu(b, a) == pytest.approx(1 - e, abs=1e-12)


def _brute_equity(h0, h1, board):
    live = [Card.from_id(x) for x in range(52) if x not in set(h0 + h1 + board)]
    c0 = [Card.from_id(x) for x in h0]
    c1 = [Card.from_id(x) for x in h1]
    cb = [Card.from_id(x) for x in board]
    w = t = n = 0
    for run in combinations(live, 5 - len(board)):
        r0, r1 = evaluate(c0 + cb + list(run)), evaluate(c1 + cb + list(run))
        w += r0 > r1
        t += r0 == r1
        n += 1
    return (w + t / 2) / n


@pytest.mark.parametrize("h0,h1,board", [
    ("AhKh", "QsQd", "Th9h2c"), ("7c7d", "As8s", "7s8d9s"), ("2c3d", "4h5h", "6h7hKh8c"), ("AsKd", "AcKc", "QhJh2s"),
])
def test_postflop_equity_matches_engine_evaluator(h0, h1, board):
    a, b, bd = ids(h0), ids(h1), ids(board)
    assert H.equity_hu(a, b, bd) == pytest.approx(_brute_equity(a, b, bd), abs=1e-12)


def test_equity_is_a_martingale_over_the_next_card():
    a, b, flop = ids("AhKh"), ids("QsQd"), ids("Th9h2c")
    live = [x for x in range(52) if x not in set(a + b + flop)]
    e_flop = H.equity_hu(a, b, flop)
    assert np.mean([H.equity_hu(a, b, flop + [x]) for x in live]) == pytest.approx(e_flop, abs=1e-12)
    turn = flop + [live[0]]
    live2 = [x for x in live if x != live[0]]
    assert np.mean([H.equity_hu(a, b, turn + [x]) for x in live2]) == pytest.approx(H.equity_hu(a, b, turn), abs=1e-12)


def test_bad_inputs():
    with pytest.raises(ValueError):
        H.equity_hu(ids("AhAs"), ids("AhKd"))
    with pytest.raises(ValueError):
        H.equity_hu(ids("AhAs"), ids("KhKd"), ids("2c3c"))


# --- Luck terms on hand-built hands ------------------------------------------------------------


def _play(hole, board, policy, button=0, stacks=(10_000, 10_000)):
    """Play a hand with fixed cards; ``policy(state)`` returns the action of the seat to act."""
    st = new_hand(button, stacks, 50, 100, hole_cards=[parse_cards(hole[0]), parse_cards(hole[1])])
    bd = parse_cards(board) if isinstance(board, str) else list(board)
    while not st.is_hand_over:
        if st.cards_needed:
            n = st.cards_needed
            st = deal_board(st, bd[:n])
            bd = bd[n:]
        else:
            st = apply_action(st, st.to_act, policy(st))
    return make_log(st, None)


def _shove_call(st):
    """Go all-in if nobody is all-in yet, else call."""
    la = legal_actions(st)
    if any(r.all_in for r in st.history):
        return Call()
    return Bet(la.max_to) if la.can_bet else Raise(la.max_to)


def test_preflop_all_in_adjusted_to_equity():
    log = _play(("AhAs", "KdKc"), "2c3d4h5s9c", _shove_call)
    hl = hand_luck(log)
    assert hl.allin_street == 0
    want = (AA_VS_KK_OFFSUIT - 0.5) * 20_000
    assert hl.adjusted0("allin") == pytest.approx(want, abs=1e-6)
    assert hl.adjusted0("chance") == pytest.approx(want, abs=1e-6)  # all chance nodes are after the all-in
    assert hl.net0 == 10_000  # A-2-3-4-5: the aces made a wheel and won the 20,000 pot
    assert hl.pots == (20_000, 20_000, 20_000)


def test_flop_all_in_telescopes():
    def policy(st):
        if st.street == 0:
            return Call() if legal_actions(st).can_call else Check()
        return _shove_call(st)
    log = _play(("AhKh", "QsQd"), "Th9h2c3d4s", policy)
    hl = hand_luck(log)
    assert hl.allin_street == 1
    e_flop = H.equity_hu(ids("AhKh"), ids("QsQd"), ids("Th9h2c"))
    assert hl.adjusted0("allin") == pytest.approx((e_flop - 0.5) * 20_000, abs=1e-6)
    # The chance estimator also removes the flop's luck at the preflop pot of 200.
    e_pre = H.equity_hu(ids("AhKh"), ids("QsQd"))
    assert hl.adjusted0("chance") == pytest.approx((e_flop - 0.5) * 20_000 - 200 * (e_flop - e_pre), abs=1e-6)


def test_no_adjustment_without_all_in_or_board():
    log = _play(("AhKh", "QsQd"), "", lambda st: Fold())  # button folds preflop
    hl = hand_luck(log)
    assert hl.streets == () and hl.adjusted0("allin") == hl.adjusted0("chance") == hl.net0 == -50


def _scripted(st):
    """Card-dependent betting (both seats), deliberately weird: unbiasedness must not care."""
    seat = st.to_act
    hole = st.hole_cards[seat]
    la = legal_actions(st)
    hits = any(c.rank == b.rank for c in hole for b in st.board)          # paired the board
    hearts = sum(c.suit == 2 for c in list(hole) + list(st.board))
    last = st.board[-1] if st.board else None
    if st.street == 0:
        return Call() if la.can_call else Check()
    if la.can_call:
        return Call() if hits or hearts >= 4 or (last.rank >= 13 and seat == 1) else Fold()
    if la.can_bet and (last.rank >= 10 or hits or hearts >= 4):
        pot = st.total_bets[0] + st.total_bets[1]
        size = pot if last.suit == 2 else pot // 2   # bigger on a heart
        return Bet(min(la.max_to, max(la.min_to, size)))
    return Check()


def test_unbiased_by_exact_enumeration_of_turn_and_river():
    """Fix both hands and the flop; enumerate all 45 x 44 (turn, river) deals, each equally likely.

    The bots' betting depends on the cards, so which hands reach the river, and the pots,
    depend on the deal. Still E[L_turn + L_river | flop] = 0 exactly, so the mean adjusted
    result equals the mean raw result.
    """
    h0, h1, flop = "AhKd", "9h9c", "Qh7h2s"
    used = set(ids(h0 + h1 + flop))
    live = [x for x in range(52) if x not in used]
    raw, adj, peek = [], [], []
    for t in live:
        for r in live:
            if r == t:
                continue
            board = parse_cards(flop) + [Card.from_id(t), Card.from_id(r)]
            log = _play((h0, h1), board, _scripted)
            hl = hand_luck(log)
            raw.append(hl.net0)
            # chance estimator minus the flop term (the flop is fixed here, not random)
            flop_term = hl.luck[0] if hl.streets and hl.streets[0] == 1 else 0.0
            adj.append(hl.adjusted0("chance") + flop_term)
            # Negative control: adjust only the hands that reached showdown (peeks at the future).
            peek.append(adj[-1] if log.folded is None else raw[-1])
    raw, adj, peek = np.array(raw, float), np.array(adj, float), np.array(peek, float)
    assert len(raw) == 45 * 44
    assert adj.mean() == pytest.approx(raw.mean(), abs=1e-6)
    assert adj.std() < raw.std()  # and here it also reduces the spread
    assert len(set(raw.tolist())) > 3  # the betting really varied
    # Negative control: the same terms, applied only to hands that went to showdown, are biased
    # (whether a hand reaches showdown depends on the turn and river cards themselves).
    assert abs(peek.mean() - raw.mean()) > 1  # 4.5 chips per hand here; the unbiased one is off by < 1e-6


def _turn_shover(st):
    """Check/call to the turn; on the turn, shove iff the turn card is a heart or a queen-plus (card-dependent)."""
    la = legal_actions(st)
    if any(r.all_in for r in st.history):
        return Call()
    if st.street == 2 and (st.board[3].suit == 2 or st.board[3].rank >= 12):
        return Bet(la.max_to) if la.can_bet else Raise(la.max_to)
    return Call() if la.can_call else Check()


def test_allin_estimator_unbiased_by_enumeration():
    """Flop fixed; all 45 x 44 (turn, river) deals. Whether the hand is all-in on the turn depends on
    the turn card (known before the river is dealt), so subtracting only the river luck of the all-in
    hands keeps the mean exactly."""
    h0, h1, flop = "AhKd", "9h9c", "Qh7h2s"
    live = [x for x in range(52) if x not in set(ids(h0 + h1 + flop))]
    raw, adj, allin = [], [], 0
    for t in live:
        for r in live:
            if r == t:
                continue
            hl = hand_luck(_play((h0, h1), parse_cards(flop) + [Card.from_id(t), Card.from_id(r)], _turn_shover))
            raw.append(hl.net0)
            adj.append(hl.adjusted0("allin"))
            allin += hl.allin_street == 2
    raw, adj = np.array(raw, float), np.array(adj, float)
    assert 0 < allin < len(raw)  # both kinds of hands occur
    assert adj.mean() == pytest.approx(raw.mean(), abs=1e-6)
    assert adj.std() < raw.std()


def test_unequal_stacks_and_refund():
    """A covers B: B's all-in call for 3,000 leaves 7,000 uncalled; the contested pot is 6,000."""
    log = _play(("AhAs", "KdKc"), "2c3d4h5s9c", _shove_call, stacks=(10_000, 3_000))
    assert log.refunds and log.refunds[0].chips == 7_000
    hl = hand_luck(log)
    assert hl.pots == (6_000, 6_000, 6_000) and hl.allin_street == 0
    assert hl.net0 == 3_000
    assert hl.adjusted0("allin") == pytest.approx((AA_VS_KK_OFFSUIT - 0.5) * 6_000, abs=1e-6)
    # Seat 1 covers seat 0 this time (the shover is the short stack).
    log = _play(("KdKc", "AhAs"), "2c3d4h5s9c", _shove_call, stacks=(2_500, 10_000))
    hl = hand_luck(log)
    assert hl.pots[0] == 5_000
    assert hl.adjusted0("allin") == pytest.approx((1 - AA_VS_KK_OFFSUIT - 0.5) * 5_000, abs=1e-6)


@pytest.mark.parametrize("h0,h1", [("Ah7h", "Kh7c"), ("2c2d", "JsTs")])
def test_flop_chance_node_is_mean_zero(h0, h1):
    """The flop luck term has mean exactly 0: the average flop equity over all 17,296 flops equals the
    preflop equity from the board table (two independent code paths)."""
    a, b = ids(h0), ids(h1)
    live = [x for x in range(52) if x not in set(a + b)]
    flops = list(combinations(live, 3))
    assert len(flops) == 17_296
    mean_flop = np.mean([H._postflop_equity(a, b, list(f)) for f in flops])
    assert mean_flop == pytest.approx(H.preflop_equity_hu(a, b), abs=1e-12)


def test_session_adjustments_on_engine_hands():
    m = duplicate_match(PotOddsBot, RandomBot, 30, 14_142_135)
    a = session_adjustments(m.games)
    np.testing.assert_allclose(a["raw"], [g.a_net_bb for g in m.games])
    for g, raw, ai, st in zip(m.games, a["raw"], a["allin"], a["allin_street"]):
        if st < 0:
            assert ai == raw
        else:  # paid at the all-in equity: (e_j - 1/2) * pot, seat-flipped for A
            hl = hand_luck(g.log)
            e = hl.equities[st]  # equities[0] preflop, [1] flop, [2] turn
            pot = 2 * min(sum(r.chips for r in g.log.actions if r.player == s) + g.log.blinds[s]
                          - sum(f.chips for f in g.log.refunds if f.player == s) for s in (0, 1))
            sign = 1 if g.a_seat == 0 else -1
            assert ai == pytest.approx(sign * (e - 0.5) * pot / 100, abs=1e-9)
    b = session_adjustments(m.games)  # deterministic
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_compare_estimators_identity_and_reduction():
    rng = np.random.default_rng(0)
    raw = rng.normal(10, 50, 60)
    same = compare_estimators(raw, raw, 500, np.random.default_rng(1), 200)
    assert same["sd_ratio"] == 1 and same["difference_raw_minus_adjusted"]["mean_bb100"] == 0
    half = compare_estimators(raw, raw / 2 + 5, 500, np.random.default_rng(1), 200)
    assert half["sd_ratio"] == pytest.approx(0.5) and half["hands_saved"] == pytest.approx(0.75)
