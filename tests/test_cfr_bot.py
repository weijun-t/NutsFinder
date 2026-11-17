"""Tests for CFRBot (tier 4): the abstract walk, translation handling, legality, leaks, seeds.

The blueprint here is synthetic: random positive strategy sums from a fixed
seed (every info set "reached", every strategy mixed), or a one-hot
blueprint that always takes a chosen edge. That tests the *bot* (walk,
translation, sizing, fallbacks) without a training run; the trainer has
its own tests (``test_holdem_training.py``).

Hand-built spots use the engine directly (``new_hand`` / ``apply_action`` /
``deal_board``), then ask the bot for one decision.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
from test_effective_stack import Checked, play_unequal
from test_info_leak import SpyBot, check_spy

from nutsfinder.abstraction.blueprint import Blueprint, abstraction_fingerprint, slot_offsets, tree_for
from nutsfinder.abstraction.config import DEFAULT_CONFIG
from nutsfinder.abstraction.translation import pseudo_harmonic, to_for_fraction
from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, CFRBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.bots.cfr_bot import UntrainedBlueprintWarning, require_trained
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, PublicState, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.runner import hand_seeds, play_hand, play_match
from nutsfinder.eval.match import duplicate_match
from nutsfinder.eval.report import result_dict

TREE = tree_for(DEFAULT_CONFIG)
OFFSET, N_ACTIONS = slot_offsets(TREE)
FLOP, TURN, RIVER = parse_cards("2c7d9h"), parse_cards("Js"), parse_cards("3d")


def _random_blueprint(seed: int = 0) -> Blueprint:
    rng = np.random.default_rng(seed)
    return Blueprint.from_tables(DEFAULT_CONFIG, None, rng.random(int(N_ACTIONS.sum())) + 0.01)


def _one_hot_blueprint(prefer) -> Blueprint:
    """Every info set plays the first edge (in order) for which ``prefer(action, labels)`` is true, else call/check."""
    s = np.zeros(int(N_ACTIONS.sum()))
    for n in range(TREE.n_nodes):
        if TREE.infoset_base[n] < 0:
            continue
        acts, labels = TREE.actions(n), TREE.edge_labels(n)
        j = next((j for j, (a, lab) in enumerate(zip(acts, labels)) if prefer(a, lab)),
                 1 if isinstance(acts[0], Fold) else 0)
        for b in range(DEFAULT_CONFIG.buckets(TREE.street[n])):
            s[OFFSET[TREE.infoset_base[n] + b] + j] = 1.0
    return Blueprint.from_tables(DEFAULT_CONFIG, None, s)


BP = _random_blueprint()
POT_BP = _one_hot_blueprint(lambda a, labs: any(x.kind == "pot" for x in labs))
FOLD_BP = _one_hot_blueprint(lambda a, labs: isinstance(a, Fold))  # folds to every bet, else checks
CHECK_BP = _one_hot_blueprint(lambda a, labs: False)  # always check / call


def hole(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


HOLES = (hole("AsKd"), hole("8c8s"))


def play(state: HandState, *moves) -> HandState:
    for seat, action in moves:
        while state.cards_needed:
            state = deal_board(state, {3: FLOP, 1: TURN if len(state.board) == 3 else RIVER}[state.cards_needed])
        state = apply_action(state, seat, action)
    while state.cards_needed and state.to_act is None and not state.is_hand_over:
        state = deal_board(state, {3: FLOP, 1: TURN if len(state.board) == 3 else RIVER}[state.cards_needed])
    return state


def view(state: HandState) -> PublicState:
    assert state.to_act is not None
    return public_view(state, state.to_act)


def flop_spot(bet: int) -> HandState:
    """Button (seat 0) opens 3bb, BB calls; BB (seat 1) leads ``bet`` into 600 on the flop: button to act."""
    s = new_hand(button=0, hole_cards=HOLES)
    return play(s, (0, Raise(300)), (1, Call()), (1, Bet(bet)))


# --- the walk and translation ----------------------------------------------------------


def _frequency(state: HandState, mode: str, n: int = 400, bp: Blueprint = BP) -> float:
    v = view(state)
    hits = 0
    for seed in range(n):
        bot = CFRBot(bp, seed, debug=True)
        bot.act(v, HOLES[v.seat])
        hits += bot.last_decision.mode == mode
    return hits / n


def test_root_decision_is_the_blueprint_at_the_root():
    bot = CFRBot(BP, 1, debug=True)
    action = bot.act(view(new_hand(button=0, hole_cards=HOLES)), HOLES[0])
    d = bot.last_decision
    assert d.mode == "blueprint" and d.node == 0 and d.bucket == 2  # AKo is class 2
    assert [name for name, _ in d.strategy] == ["fold", "call", "raise[open_bb:3]", "raise[all_in]"]
    assert sum(p for _, p in d.strategy) == pytest.approx(1.0)
    np.testing.assert_allclose([p for _, p in d.strategy], BP.strategy(0, 2))
    assert action in (Fold(), Call(), Raise(300), Raise(10_000))
    assert "node 0" in str(d) and "pot odds 25.0%" in str(d)  # 50 / (150 + 50)


def _translation_frequency(state: HandState, suffix: str, n: int = 1000) -> float:
    v = view(state)
    hits = 0
    for seed in range(n):
        bot = CFRBot(BP, seed, debug=True)
        bot.act(v, HOLES[v.seat])
        hits += bot.last_decision.translations[-1].endswith(suffix)
    return hits / n


def test_small_bet_read_as_check_when_we_still_act_in_the_abstraction():
    # BB leads 100 into 600 (1/6 pot): read as a check with probability (1 - 1/6) / (1 + 1/6) = 0.714.
    # The button is then at its own abstract node after "check": the blueprint decides, but its
    # "check" must become a real call or fold (we face a real bet): the equity rule. AKo on 2c7d9h
    # (E[HS] 0.52) has the odds (100 / 800 = 0.125), so it calls.
    expected = pseudo_harmonic(100 / 600, 0.0, 1.0)
    assert abs(_translation_frequency(flop_spot(100), "-> check") - expected) < 0.05
    v = view(flop_spot(100))
    for seed in range(60):
        bot = CFRBot(BP, seed, debug=True)
        a = bot.act(v, HOLES[0])
        d = bot.last_decision
        assert d.mode == "blueprint"
        if d.translations[-1].endswith("-> check"):  # (read as pot, folding is a legitimate answer)
            assert d.path == "raise to 300 call check" and a != Fold()
            assert a == Call() if d.chosen == "check" else isinstance(a, Raise)
            if d.chosen == "check":
                assert any("equity rule" in n and "call" in n for n in d.notes)


def test_abstract_check_facing_a_real_bet_folds_a_dominated_hand():
    # Same spot, but with 3h4s (E[HS] 0.166 on 2c7d9h) facing a 1/3-pot lead (200 into 600):
    # pot odds 200 / 1000 = 0.2. When the lead is read as a check and the blueprint checks, the
    # old rule called; the equity rule folds (calling loses even against a random hand).
    s = new_hand(button=0, hole_cards=(hole("3h4s"), hole("8c8s")))
    s = play(s, (0, Raise(300)), (1, Call()), (1, Bet(200)))
    folds = 0
    for seed in range(200):
        bot = CFRBot(CHECK_BP, seed, debug=True)
        a = bot.act(view(s), hole("3h4s"))
        d = bot.last_decision
        if d.translations[-1].endswith("-> check"):
            assert d.chosen == "check" and a == Fold()
            folds += 1
    assert folds > 50  # P(read as check) = (1 - 1/3) / (1 + 1/3) = 0.5


def test_small_bet_that_closes_the_abstract_street_is_remapped_to_the_smallest_bet():
    # The BB (the bot, seat 1) checks the flop, the button bets 100: read as a check, the abstract
    # flop would be over (check-check) while the real game asks us to respond. Instead the bet is
    # remapped to the smallest bet edge (pot): the blueprint answers it (fold / call / raise) and
    # the node is "soft". So the walk never goes off-tree here, whichever way the bet is read.
    s = play(new_hand(button=0, hole_cards=HOLES), (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(100)))
    expected = pseudo_harmonic(100 / 600, 0.0, 1.0)
    remapped = 0
    for seed in range(1000):
        bot = CFRBot(BP, seed, debug=True)
        bot.act(view(s), HOLES[1])
        d = bot.last_decision
        assert d.mode == "blueprint" and d.path == "raise to 300 call check bet 600"
        assert [name for name, _ in d.strategy] == ["fold", "call", "raise[pot:1]", "raise[all_in]"]
        is_remap = "remapped to bet[pot:1]" in d.translations[-1]
        assert (d.node in bot._walk.soft) == is_remap
        remapped += is_remap
    assert abs(remapped / 1000 - expected) < 0.05


def test_soft_node_fold_follows_the_equity_rule():
    # FOLD_BP folds whenever it can. 8c8s (E[HS] 0.71) facing the 1/6-pot bet: at a soft node (the bet
    # was read as passive, then remapped) the fold is overridden by the equity rule -> call; when the
    # bet was read as pot (not soft) the blueprint's fold stands.
    s = play(new_hand(button=0, hole_cards=HOLES), (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(100)))
    seen = set()
    for seed in range(100):
        bot = CFRBot(FOLD_BP, seed, debug=True)
        a = bot.act(view(s), HOLES[1])
        soft = bot.last_decision.node in bot._walk.soft
        assert a == (Call() if soft else Fold())
        seen.add(soft)
    assert seen == {True, False}
    # The walk follows our real call (not the abstract fold edge): next street, on the tree.
    for seed in range(100):
        bot = CFRBot(FOLD_BP, seed, debug=True)
        if bot.act(view(s), HOLES[1]) == Call():
            break
    s2 = play(s, (1, Call()))
    bot.act(view(s2), HOLES[1])
    assert bot.last_decision.mode == "blueprint" and bot.last_decision.street == "turn"
    assert bot.last_decision.path == "raise to 300 call check bet 600 call"


def _river_spot(button_bet: int, holes) -> HandState:
    """Open, call; checked to the river; the BB (seat 1) checks the river and the button bets."""
    s = new_hand(button=0, hole_cards=holes)
    return play(s, (0, Raise(300)), (1, Call()), (1, Check()), (0, Check()), (1, Check()), (0, Check()),
                (1, Check()), (0, Bet(button_bet)))


def test_weak_hand_facing_river_half_pot_after_checking_does_not_always_call():
    # 4h5c on 2c7d9h Js 3d: E[HS] 0.005; half pot (300 into 600) needs 300 / 1200 = 0.25.
    # Before the fix a passive read (P = 1/3) closed the river and the bot called with 100%.
    weak, strong = hole("4h5c"), hole("9c9s")
    s = _river_spot(300, (hole("AsKd"), weak))
    actions = [CFRBot(BP, seed).act(view(s), weak) for seed in range(300)]
    assert actions.count(Fold()) > 0
    # With a blueprint that always folds to bets, the weak hand folds every time (both readings) ...
    assert all(CFRBot(FOLD_BP, seed).act(view(s), weak) == Fold() for seed in range(100))
    # ... the strong hand calls whenever the read was passive (soft node, equity 0.98 > 0.25) ...
    calls = sum(CFRBot(FOLD_BP, seed).act(view(s), strong) == Call() for seed in range(600))
    assert abs(calls / 600 - pseudo_harmonic(0.5, 0.0, 1.0)) < 0.06
    # ... and a raising blueprint raises it whichever way the bet was read.
    for seed in range(50):
        a = CFRBot(POT_BP, seed).act(view(s), strong)
        assert isinstance(a, Raise) and apply_action(s, 1, a)


def test_reviewer_third_pot_flop_case():
    # The reviewer's leak: a flop call with E[HS] ~0.20 facing 1/3 pot (pot odds exactly 0.2).
    # 4h5c (0.189 < 0.2) must fold whenever the blueprint folds or the read is passive and the
    # blueprint checks/folds; 4h6c (0.201 > 0.2) is a (marginal) call by the equity rule.
    s = play(new_hand(button=0, hole_cards=(hole("AsKd"), hole("4h5c"))),
             (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(200)))
    for seed in range(200):
        bot = CFRBot(FOLD_BP, seed, debug=True)
        assert bot.act(view(s), hole("4h5c")) == Fold()
        d = bot.last_decision
        assert d.mode == "blueprint"
        if d.node in bot._walk.soft:
            assert any("E[HS] 0.189 <= pot odds 0.200: fold" in n for n in d.notes)
    soft_calls = 0
    for seed in range(200):
        bot = CFRBot(FOLD_BP, seed, debug=True)
        a = bot.act(view(s), hole("4h6c"))
        if bot.last_decision.node in bot._walk.soft:
            assert a == Call()
            soft_calls += 1
    assert soft_calls > 60  # P(passive) = 1/2


def test_terminal_with_chips_behind_folds_junk_to_a_big_bet():
    # BB 3-bets to 4,000 (x = 3.08 between pot 1.0 and all-in 8.08): read as all-in about 65% of the
    # time; the button calls, so the abstract hand is an all-in showdown ("terminal") while 6,000 are
    # still behind. Checked down to the river, the BB shoves 6,000 into 8,000: pot odds 6,000 / 20,000
    # = 0.3. Junk (4h5c, 0.005) folds; the old rule called. The nuts-ish 9c9s (0.98) calls.
    s = new_hand(button=0, hole_cards=(hole("4h5c"), hole("AsKd")))
    s = play(s, (0, Raise(300)), (1, Raise(4000)), (0, Call()), (1, Check()), (0, Check()),
             (1, Check()), (0, Check()), (1, Bet(6000)))
    terminal = 0
    for seed in range(100):
        junk, nuts = CFRBot(BP, seed, debug=True), CFRBot(BP, seed, debug=True)
        a = junk.act(view(s), hole("4h5c"))
        b = nuts.act(view(s), hole("9c9s"))
        if junk.last_decision.mode == "terminal":
            terminal += 1
            assert a == Fold() and junk.last_decision.chosen == "fallback fold"
            assert b == Call() and nuts.last_decision.chosen == "fallback call"
    assert terminal > 40
    # With a check available the fallback checks (never folds for free, never bets).
    s2 = play(new_hand(button=0, hole_cards=(hole("4h5c"), hole("AsKd"))),
              (0, Raise(300)), (1, Raise(4000)), (0, Call()), (1, Check()))
    for seed in range(20):
        bot = CFRBot(BP, seed, debug=True)
        a = bot.act(view(s2), hole("9c9s"))
        if bot.last_decision.mode == "terminal":
            assert a == Check()


def test_preflop_min_raise_maps_between_passive_and_pot():
    # Button opens to 300, BB min-raises to 500: x = 200 / 600 = 1/3 between passive (0) and pot (1).
    # A passive read ("call") would end the abstract preflop, so it is remapped to the pot 3-bet:
    # the walk is always at "raise to 300 raise to 900"; it is soft with probability P(passive).
    s = play(new_hand(button=0, hole_cards=HOLES), (0, Raise(300)), (1, Raise(500)))
    expected = pseudo_harmonic(1 / 3, 0.0, 1.0)
    soft = 0
    for seed in range(1000):
        bot = CFRBot(FOLD_BP, seed, debug=True)
        a = bot.act(view(s), HOLES[0])
        d = bot.last_decision
        assert d.mode == "blueprint" and d.path == "raise to 300 raise to 900"
        is_soft = d.node in bot._walk.soft
        soft += is_soft
        # Soft: AKo (0.65 vs a random hand) has the odds (200 / 1,000) -> call; else the blueprint folds.
        assert a == (Call() if is_soft else Fold())
    assert abs(soft / 1000 - expected) < 0.05


def test_overbet_maps_between_pot_and_all_in():
    # 1800 into 600 = 3x pot; the node's sizes are pot (1.0) and all-in (9700 / 600).
    s = flop_spot(1800)
    v = view(s)
    p_pot = pseudo_harmonic(3.0, 1.0, 9700 / 600)
    seen = {"pot": 0, "all_in": 0}
    for seed in range(1000):
        bot = CFRBot(BP, seed, debug=True)
        bot.act(v, HOLES[0])
        tr = bot.last_decision.translations[-1]
        seen["pot" if tr.endswith("bet[pot:1]") else "all_in"] += 1
        assert tr.endswith("bet[pot:1]") or tr.endswith("bet[all_in]")
        assert bot.last_decision.mode == "blueprint"
    assert abs(seen["pot"] / 1000 - p_pot) < 0.05  # SE 0.016


def test_bet_beyond_the_effective_stack_is_all_in_deterministically():
    s = flop_spot(9700)  # all-in
    bot = CFRBot(BP, 0, debug=True)
    a = bot.act(view(s), HOLES[0])
    d = bot.last_decision
    assert "deterministic" in d.translations[-1] and d.translations[-1].endswith("bet[all_in]")
    assert [name for name, _ in d.strategy] == ["fold", "call"] and a in (Fold(), Call())


def test_own_bets_are_sized_from_the_real_pot_and_followed_by_the_chosen_edge():
    # The opponent's 1,800 overbet is read as pot (abstract pot after it: 1,200) or as all-in.
    # When it is read as pot, POT_BP raises "pot" -- sized from the REAL pot (2,400 + 1,800).
    s = flop_spot(1800)
    v = view(s)
    real_pot_raise = to_for_fraction(v, 1.0)
    assert real_pot_raise == 1800 + 1 * (2400 + 1800)
    for seed in range(30):
        bot = CFRBot(POT_BP, seed, debug=True)
        a = bot.act(v, HOLES[0])
        if bot.last_decision.translations[-1].endswith("bet[pot:1]"):
            break
    else:
        pytest.fail("no seed read the overbet as pot")
    assert a == Raise(real_pot_raise)
    # The opponent calls; on the turn the bot's walk must follow *its own* edge, not re-translate 6,000.
    s2 = play(s, (0, a), (1, Call()), (1, Check()))
    bot.act(view(s2), HOLES[0])
    d = bot.last_decision
    # Translated: the open (made before this bot was asked, so not a cached choice) and the overbet.
    assert len(d.translations) == 2 and all("to 6000" not in t for t in d.translations)
    assert d.path == "raise to 300 call bet 600 raise to 2400 call check"  # abstract chips, not real
    assert d.street == "turn" and d.mode == "blueprint"


def test_translations_are_cached_within_a_hand():
    s = flop_spot(1800)
    v = view(s)
    bot = CFRBot(BP, 11, debug=True)
    bot.act(v, HOLES[0])
    first = (bot.last_decision.node, bot.last_decision.translations)
    for k in range(10):
        bot.reseed(1000 + k)  # a fresh stream would redraw a translation if the walk were redone
        bot.act(v, HOLES[0])
        assert (bot.last_decision.node, bot.last_decision.translations) == first


def test_new_hand_resets_the_walk_even_without_end_hand():
    bot = CFRBot(BP, 2, debug=True)
    bot.act(view(flop_spot(1800)), HOLES[0])
    s = new_hand(button=1, hole_cards=HOLES)
    s = play(s, (1, Call()))
    bot.act(view(s), HOLES[0])  # BB after a limp, a different hand
    assert bot.last_decision.translations == () and bot.last_decision.path == "call"


def test_cap_reached_raise_has_only_all_in_to_map_to():
    # Flop: BB bets pot, button raises pot, BB re-raises small: the postflop cap (2) is reached,
    # so the only sizes are passive and all-in.
    s = new_hand(button=0, hole_cards=HOLES)
    s = play(s, (0, Raise(300)), (1, Call()), (1, Bet(600)), (0, Raise(2400)), (1, Raise(4200)))
    bot = CFRBot(BP, 5, debug=True)
    a = bot.act(view(s), HOLES[0])
    tr = bot.last_decision.translations[-1]
    assert "between 0.000 and" in tr and (tr.endswith("raise[all_in]") or tr.endswith("call"))
    assert s.to_act == 0 and apply_action(s, 0, a)  # legal


def test_real_action_mapping_never_folds_for_free_and_degrades_bets():
    bot = CFRBot(BP, 0)
    # A walk at a node facing an abstract bet, while the real spot allows a check:
    s = play(new_hand(button=0, hole_cards=HOLES), (0, Raise(300)), (1, Call()), (1, Check()))
    v = view(s)
    bot._sync(v)
    bot._walk.node = TREE.child_ids(0)[2]  # root -> open: a node facing a raise (fold, call, pot, all-in)
    assert bot._real_action(v, 0) == Check()  # fold edge, check is free -> check
    assert bot._real_action(v, 1) == Check()
    assert isinstance(bot._real_action(v, 2), Bet)  # pot edge -> a real bet from the real pot
    assert bot._real_action(v, 3) == Bet(9700)  # all-in edge -> the effective all-in


# --- whole hands: legality, leaks, reproducibility -------------------------------------


class SpyCFR(CFRBot):
    """A CFRBot that records everything it is handed (see ``test_info_leak.py``)."""

    def __init__(self, seed: int) -> None:
        super().__init__(BP, seed)
        self.seen_acts: list = []
        self.seen_ends: list = []

    def act(self, state, hole_cards):
        self.seen_acts.append((state, hole_cards))
        return super().act(state, hole_cards)

    def end_hand(self, summary):
        self.seen_ends.append(summary)
        super().end_hand(summary)


def test_cfr_bot_never_sees_hidden_cards():
    spies = [SpyCFR(1), SpyBot(2)]
    for i, seed in enumerate(hand_seeds(13, 150)):
        order = spies if i % 2 == 0 else spies[::-1]
        log = play_hand(order, seed, button=i % 2)
        seat = 0 if i % 2 == 0 else 1
        spy = spies[0]
        assert len(spy.seen_ends) == 1
        check_spy(spy, seat, log, spy.seen_acts, spy.seen_ends[0])
        for s in spies:
            s.seen_acts.clear()
            s.seen_ends.clear()


def test_bucket_uses_only_own_cards_and_the_board():
    # Same public view, same own cards, different (hidden) opponent cards -> same decision.
    a = new_hand(button=0, hole_cards=(hole("AsKd"), hole("8c8s")))
    b = new_hand(button=0, hole_cards=(hole("AsKd"), hole("QhQd")))
    moves = ((0, Raise(300)), (1, Call()), (1, Bet(1800)))
    for seed in range(20):
        x, y = CFRBot(BP, seed, debug=True), CFRBot(BP, seed, debug=True)
        assert x.act(view(play(a, *moves)), hole("AsKd")) == y.act(view(play(b, *moves)), hole("AsKd"))
        assert x.last_decision.bucket == y.last_decision.bucket


def test_same_seed_same_model_identical_hands():
    make = functools.partial(CFRBot, BP, 7)
    for match, n in ((duplicate_match, 40), ):
        first = match(make, functools.partial(EquityBot, 5, samples=30), n, master_seed=99)
        second = match(make, functools.partial(EquityBot, 5, samples=30), n, master_seed=99)
        assert [g.log for g in first.games] == [g.log for g in second.games]
    a = play_match([CFRBot(BP, 1), CFRBot(BP, 2)], 60, master_seed=4)
    b = play_match([CFRBot(BP, 1), CFRBot(BP, 2)], 60, master_seed=4)
    assert a == b
    c = play_match([CFRBot(BP, 1), CFRBot(BP, 3)], 60, master_seed=4)
    assert a != c  # the seed matters (mixed strategies)


class NoFreeFold(Checked):
    """Checked (bets within the effective stack) + never folds when a check is free."""

    def act(self, state, hole_cards):
        action = super().act(state, hole_cards)
        assert not (isinstance(action, Fold) and state.legal.can_check), state
        return action


OPPONENTS = {
    "RandomBot": lambda: RandomBot(1),
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": lambda: PotOddsBot(2, samples=30),
    "EquityBot": lambda: EquityBot(5, samples=30),
    "BayesBot": lambda: BayesBot(6, samples=30),
    "CFRBot": lambda: CFRBot(_random_blueprint(1), 9),
}


def _fuzz(opponent, n_equal: int, n_unequal: int, seed: int) -> None:
    bot = NoFreeFold(CFRBot(BP, seed))
    for log in play_match([bot, OPPONENTS[opponent]()], n_equal, master_seed=seed):
        assert sum(log.result.net) == 0 and sum(log.result.final_stacks) == sum(log.starting_stacks)
    if n_unequal:
        assert play_unequal(lambda: NoFreeFold(CFRBot(BP, seed + 1)), OPPONENTS[opponent], n_unequal, seed) >= 0


@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_legal_and_conserved_against_every_bot(opponent):
    _fuzz(opponent, 80, 80, seed=21)


@pytest.mark.slow
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_fuzz_10k_hands_against_every_bot(opponent):
    """5,000 hands at 100bb + 5,000 with random unequal stacks (some below the blinds)."""
    _fuzz(opponent, 5_000, 5_000, seed=22)


def test_untrained_default_and_repr():
    with pytest.warns(UntrainedBlueprintWarning):
        bot = CFRBot()
    assert repr(bot) == "CFRBot(model=untrained (uniform))"
    a = bot.act(view(new_hand(button=0, hole_cards=HOLES)), HOLES[0])
    assert a in (Fold(), Call(), Raise(300), Raise(10_000))
    assert CFRBot(BP, 0, time_budget=5.0).time_budget == 5.0


# --- strict on-tree self-play ---------------------------------------------------------


class StrictCFR(CFRBot):
    """Asserts at every decision that the walk is exactly on the tree (the reviewer's self-play check)."""

    def __init__(self, bp: Blueprint, seed: int) -> None:
        super().__init__(bp, seed, debug=True)
        self.decisions = 0

    def act(self, state, hole_cards):
        a = super().act(state, hole_cards)
        d = self.last_decision
        n, b = d.node, state.button
        assert d.mode == "blueprint", str(d)
        assert TREE.player[n] == (0 if state.seat == b else 1), str(d)  # button = tree seat 0
        assert TREE.street[n] == int(state.street), str(d)
        assert tuple(TREE.contrib[n]) == (state.total_bets[b], state.total_bets[1 - b]), str(d)
        assert not d.notes, str(d)  # no passive filler, no remap / equity-rule fallback
        assert all("remapped" not in t and "no bet edge" not in t for t in d.translations), str(d)
        self.decisions += 1
        return a


def _strict_self_play(bp: Blueprint, n_hands: int, master_seed: int) -> None:
    bots = [StrictCFR(bp, 1), StrictCFR(bp, 2)]
    logs = play_match(bots, n_hands, master_seed=master_seed)  # button alternates
    assert {log.button for log in logs} == {0, 1}
    assert min(b.decisions for b in bots) > n_hands  # both seats, several decisions per hand


def test_strict_self_play_stays_on_the_tree():
    _strict_self_play(BP, 300, master_seed=31)
    _strict_self_play(_one_hot_blueprint(lambda a, labs: bool(labs)), 60, master_seed=32)  # always bets


@pytest.mark.slow
def test_strict_self_play_official_blueprint():
    path = Path(__file__).resolve().parents[1] / "models" / "holdem_official" / "latest.npz"
    if not path.exists():
        pytest.skip(f"{path} not present (models/ is not in git)")
    _strict_self_play(Blueprint.load(path), 400, master_seed=33)


def test_one_instance_in_both_seats_keeps_one_walk_per_seat():
    # The button opens to 250 (x = 0.75: a randomised translation). The same instance then plays
    # both seats; each seat's walk (and its translations) must survive the other seat's decisions.
    bot = CFRBot(BP, 4, debug=True)
    s = play(new_hand(button=0, hole_cards=HOLES), (0, Raise(250)))
    s = play(s, (1, bot.act(view(s), HOLES[1])))
    w1, first = bot._walks[1], bot.last_decision.translations
    while not s.is_hand_over:
        seat = s.to_act
        s = play(s, (seat, bot.act(view(s), HOLES[seat])))
        if seat == 1:
            assert bot._walks[1] is w1 and bot.last_decision.translations[:len(first)] == first
        if s.to_act is None and s.cards_needed:
            s = play(s)
    assert 0 in bot._walks and bot._walks[1] is w1
    # Whole matches with one instance in both seats: legal, and self-play stays on the tree.
    one = StrictCFR(BP, 5)
    play_match([one, one], 100, master_seed=8)


# --- seeds, trained/untrained, metadata ------------------------------------------------


def test_default_seed_is_fixed_so_play_match_is_reproducible():
    a = play_match([CFRBot(BP), RandomBot(1)], 40, master_seed=5)
    b = play_match([CFRBot(BP), RandomBot(1)], 40, master_seed=5)
    assert a == b


def test_untrained_warns_and_is_detectable():
    with pytest.warns(UntrainedBlueprintWarning):
        bot = CFRBot()
    assert not bot.is_trained
    with pytest.raises(ValueError, match="not trained"):
        require_trained(bot)
    assert not CFRBot(BP).is_trained  # synthetic tables: no training iterations recorded
    meta = {"iteration": 1000, "commit": "abc1234", "abstraction": abstraction_fingerprint(DEFAULT_CONFIG)}
    trained = CFRBot(Blueprint.from_tables(DEFAULT_CONFIG, None, BP.sigma.copy(), meta))
    assert trained.is_trained
    require_trained(trained)
    with pytest.raises(ValueError):
        require_trained(trained, min_iterations=10_000)
    with pytest.raises(ValueError, match="not a CFRBot"):
        require_trained(RandomBot(0))
    assert repr(trained) == f"CFRBot(model=tables @ 1,000 it, commit abc1234, abstraction {meta['abstraction'][:12]})"


def test_match_results_record_which_blueprint_played():
    meta = {"iteration": 1000, "commit": "abc1234", "abstraction": abstraction_fingerprint(DEFAULT_CONFIG)}
    trained = Blueprint.from_tables(DEFAULT_CONFIG, None, BP.sigma.copy(), meta)
    with pytest.warns(UntrainedBlueprintWarning):
        res = duplicate_match(functools.partial(CFRBot, trained, 1), CFRBot, 3, master_seed=1)
    assert res.bot_a == res.bot_b == "CFRBot"  # unchanged: the class names
    assert res.bot_a_desc == repr(CFRBot(trained)) and "1,000 it" in res.bot_a_desc
    assert res.bot_b_desc == "CFRBot(model=untrained (uniform))"
    d = result_dict(res)
    assert d["bot_a"] == "CFRBot" and d["bot_a_desc"] == res.bot_a_desc and d["bot_b_desc"] == res.bot_b_desc


def test_equity_rule_prices_an_all_in_for_less_against_the_effective_pot():
    # Stacks (10,000, 600): limp/check, BB checks the flop, button bets 3,000. The BB has 500 behind:
    # P = 3,200 but 2,500 of the bet is refunded, so P_eff = 700 and the call needs 500 / 1,200 = 0.417
    # (the raw pot would say 500 / 3,700 = 0.135 and call far too loosely).
    s = new_hand(button=0, stacks=(10_000, 600), hole_cards=(hole("AsKd"), hole("8s6s")))
    s = play(s, (0, Call()), (1, Check()), (1, Check()), (0, Bet(3000)))
    v = view(s)
    assert v.seat == 1 and v.legal.call_amount == 500 and v.pot == 3200
    bot = CFRBot(BP, 0)
    for cards in (hole("8s6s"), hole("AhAc")):
        action, note = bot._equity_rule(v, cards)
        e = bot._equity(v, cards)
        assert "pot odds 0.417" in note
        assert action == (Call() if e > 500 / 1200 else Fold())
