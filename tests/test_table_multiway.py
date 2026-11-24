"""The multiway (3..8 seat) poker table: ``nutsfinder.table.multiway`` and its HTTP endpoints.

- The bots are treated exactly as in ``play_hand_multi``: spy bots in a
  MultiPlaySession see the same views / hole cards and the same end-of-hand
  summaries as in the runner, for the same seeds and the same human actions
  (reset stacks and kept stacks, several table sizes and human seats).
- Legality (illegal input rejected, nothing changed), chip conservation at
  every step, the moving button, kept stacks with rebuys, side pots.
- Leaks: at every step of many hands, the payload mentions no card but the
  human's and the board's; after the hand only the showdown hands (or all,
  on request); live reasoning never shows a bot's cards, in any field;
  no hand seed / session seed before allowed. The same over HTTP, for every
  play endpoint.
- Replays are deterministic and equal to ``play_match_multi``.
- The pluggable bot registry, including auto-registration of
  ``nutsfinder.bots.multiway``.
- HTTP smoke test on a random port; heads-up still works next to it.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from nutsfinder.engine.actions import Action, Call, Raise
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway import (
    MultiAlwaysRaiseBot,
    MultiBot,
    MultiCallStationBot,
    MultiHandSummary,
    MultiPublicState,
    MultiRandomBot,
    play_hand_multi,
    play_match_multi,
    position_names,
    public_view,
)
from nutsfinder.engine.multiway.hand_log import make_log
from nutsfinder.engine.multiway.state import apply_action, deal_board, new_hand
from nutsfinder.table import multiway as mw
from nutsfinder.table.multiway import (
    MULTI_BOTS,
    MultiCatalog,
    MultiPlaySession,
    MultiReplayJob,
    register_multi_bot_class,
    register_optional_multi_bots,
    result_view,
    table_view,
)
from nutsfinder.table.server import make_server
from nutsfinder.table.session import BotCatalog, SessionError, StaleRequest, derive_seed, seed_secret


class SpyBot(MultiRandomBot):
    """A MultiRandomBot that keeps everything it is handed."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.acts: list[tuple[MultiPublicState, tuple[Card, ...]]] = []
        self.ends: list[MultiHandSummary] = []

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        self.acts.append((view, hole_cards))
        return super().act(view, hole_cards)

    def end_hand(self, summary: MultiHandSummary) -> None:
        self.ends.append(summary)


def play_out(s: MultiPlaySession, human: MultiBot, check=None) -> None:
    """Let a bot play the human's seat until the hand is over (calling ``check`` at each step)."""
    while s.human_to_act:
        if check:
            check(s)
        s.apply_human(human.act(public_view(s.state, s.human), s.state.hole_cards[s.human]))
    if check:
        check(s)


def conserved(s: MultiPlaySession) -> None:
    st = s.state
    assert sum(st.stacks) + st.pot == sum(st.starting_stacks)


def session(n: int, hero: int, seed: int = 1, bots=None, **kw) -> MultiPlaySession:
    if bots is None:
        bots = [MultiRandomBot(100 + s) for s in range(n)]
    bots = [None if s == hero else b for s, b in enumerate(bots)]
    return MultiPlaySession(bots, hero, seed=seed, **kw)


# --- the bots see exactly what play_hand_multi shows them ---------------------------------------


@pytest.mark.parametrize("n,hero,keep", [(3, 0, False), (6, 2, False), (8, 7, False), (6, 4, True)])
def test_bots_see_exactly_what_play_hand_multi_shows_them(n, hero, keep):
    spies = [SpyBot(10 + s) for s in range(n)]
    s = session(n, hero, seed=2024 + n, bots=spies, keep_stacks=keep)
    human = MultiRandomBot(99)
    played = []
    for _ in range(25):
        s.start_hand()
        play_out(s, human)
        assert s.error is None, s.error
        played.append(s.log)
    twins = [SpyBot(10 + seat) for seat in range(n)]
    run_human = MultiRandomBot(99)
    for log in played:
        seats = [run_human if seat == hero else twins[seat] for seat in range(n)]
        assert play_hand_multi(seats, seed=log.seed, button=log.button, stacks=log.starting_stacks) == log
    for seat in range(n):
        if seat == hero:
            continue
        assert spies[seat].acts == twins[seat].acts
        assert spies[seat].ends == twins[seat].ends
    ends = [e for seat in range(n) if seat != hero for e in spies[seat].ends]
    assert any(any(c is not None for c in e.shown_cards) for e in ends)  # showdowns happened ...
    assert any(all(c is None for c in e.shown_cards) for e in ends)  # ... and fold-won hands hid every card


# --- legality, chips, button, stacks ---------------------------------------------------------


def test_button_positions_and_chips_conserved():
    n = 6
    s = session(n, 3, seed=5)
    human = MultiRandomBot(4)
    total = 0
    for k in range(14):
        s.start_hand()
        assert s.state.button == k % n
        p = s.payload()
        assert [x["position"] for x in p["table"]["seats"]] == list(position_names(n, k % n))
        play_out(s, human, conserved)
        assert s.log is not None and s.error is None
        assert sum(s.log.result.net) == 0
        total += s.log.result.net[3]
        p = s.payload()
        assert p["score"]["hands"] == k + 1 and p["score"]["chips"] == total
        assert sum(x["session_net"] for x in p["players"]) == 0
        assert all(st == 10_000 for st in s.state.starting_stacks)  # reset every hand


def test_keep_stacks_carries_over_and_rebuys_busted_seats():
    n = 5
    bots = [MultiAlwaysRaiseBot() for _ in range(n)]  # all-ins every hand: seats bust
    s = session(n, 0, seed=8, bots=bots, keep_stacks=True)
    human = MultiCallStationBot()
    prev_final = None
    saw_side_pot = False
    for _ in range(20):
        s.start_hand()
        start = s.state.starting_stacks
        if prev_final is not None:
            assert start == tuple(f if f > 0 else 10_000 for f in prev_final)
        assert sum(start) == 10_000 * (n + sum(s.rebuys))  # chips only enter through rebuys
        play_out(s, human, conserved)
        assert s.error is None
        prev_final = s.log.result.final_stacks
        saw_side_pot |= len(s.log.result.pots) > 1
    assert sum(s.rebuys) > 0 and saw_side_pot
    assert sum(s.totals) == 0


def test_illegal_actions_are_rejected_and_change_nothing():
    s = session(4, 0, seed=3, bots=[MultiCallStationBot() for _ in range(4)])
    with pytest.raises(SessionError):
        s.act("fold")  # no hand yet
    s.start_hand()  # button 0, 4-handed: UTG is seat 3, then the human on the button
    assert s.human_to_act
    before = s.state
    bad = [("check", None), ("bet", 150), ("raise", 150), ("raise", 10_001), ("raise", True), ("raise", "lots"),
           ("raise", None), ("raise", 300.5), ("shove", None), (None, None), (7, None), ({"x": 1}, None)]
    for kind, amount in bad:
        with pytest.raises(SessionError):
            s.act(kind, amount)
        assert s.state is before
    with pytest.raises(SessionError):
        s.start_hand()
    with pytest.raises(SessionError):
        s.reveal_bot_cards()
    with pytest.raises(SessionError):
        s.set_live("yes")
    with pytest.raises(SessionError):
        s.end_session()
    s.act("raise", "300")
    assert s.state.history[1].action == Raise(300)


def test_constructor_validation():
    with pytest.raises(SessionError):
        MultiPlaySession([None, MultiCallStationBot(), MultiCallStationBot()], 3)
    with pytest.raises(SessionError):
        MultiPlaySession([MultiCallStationBot()] * 3, 0)  # a bot in the human's seat
    with pytest.raises(SessionError):
        MultiPlaySession([None] + [MultiCallStationBot()] * 8, 0)  # 9 seats


def test_stale_step_rejected():
    s = session(3, 1, seed=1, bots=[MultiCallStationBot()] * 3)
    s.start_hand()
    while not s.human_to_act:
        s.start_hand()
    step = s.step
    s.check_step(step)
    s.act("call")
    with pytest.raises(StaleRequest):
        s.check_step(step)
    with pytest.raises(SessionError):
        s.check_step(None)


def test_presets_are_legal():
    s = session(6, 5, seed=2, bots=[MultiCallStationBot()] * 6)
    for _ in range(6):
        s.start_hand()
        while s.human_to_act:
            legal = s.payload()["legal"]
            for pr in legal["presets"]:
                assert legal["min_to"] <= pr["to"] <= legal["max_to"]
            s.act("call" if legal["can_call"] else "check")


# --- what the human-facing payload may contain ---------------------------------------------------

CARD_TOKEN = re.compile(r"(?<![A-Za-z0-9])((?:[2-9TJQKA][cdhs])+)(?![A-Za-z0-9])")


def card_tokens(obj) -> set[str]:
    found: set[str] = set()
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str):
            for run in CARD_TOKEN.findall(x):
                found.update(run[i:i + 2] for i in range(0, len(run), 2))
    return found


@dataclass
class _ChattyDecision:
    """A decision record that mentions the bot's cards in every way it could."""

    hole_cards: tuple
    note: str
    seen: list
    by_card: dict
    equity: float


class ChattyBot(MultiRandomBot):
    """Keeps a ``last_decision`` full of its own hole cards (a worst case for the masking)."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.last_decision = None

    def act(self, view, hole_cards):
        a, b = (str(c) for c in hole_cards)
        self.last_decision = _ChattyDecision(hole_cards, f"I hold {a}{b} ([{a} {b}])", [a, b], {a: 1, b: 2}, 0.5)
        return super().act(view, hole_cards)


def _seat_cards(state, seats) -> set[str]:
    return {str(c) for s in seats for c in state.hole_cards[s]}


@pytest.mark.parametrize("n,hero", [(6, 0), (8, 5), (3, 2)])
def test_other_seats_cards_never_in_payload_before_allowed(n, hero):
    bots = [ChattyBot(50 + s) for s in range(n)]
    s = session(n, hero, seed=n * 7, bots=bots, bot_keys=[None] * n)
    human = MultiRandomBot(3)
    folds = showdowns = 0
    for k in range(30):
        s.start_hand()
        others = [x for x in range(n) if x != hero]

        def check(sess):
            for live in (False, True):
                sess.live = live
                p = sess.payload()
                json.dumps(p, allow_nan=False)
                assert "session_seed" not in p and "session_secret" not in p
                allowed = _seat_cards(sess.state, [hero]) | {str(c) for c in sess.state.board}
                if sess.log is None:
                    assert card_tokens(p) <= allowed, card_tokens(p) - allowed
                    assert p["hand"]["seed"] is None
                    assert all(x["cards"] is None for x in p["table"]["seats"] if x["seat"] != hero)
                    if live and sess.reasoning:
                        assert p["reasoning"] and all(e["hole_cards"] is None for e in p["reasoning"])
                        assert p["reasoning"][0]["detail"]["equity"] == 0.5  # the numbers stay
            sess.live = False

        play_out(s, human, check)
        p = s.payload()
        assert p["hand"]["seed"] == s.log.seed
        shown = set(s.log.show_order)
        hidden = [x for x in others if x not in shown]
        if shown:
            showdowns += 1
            assert _seat_cards(s.state, shown) <= card_tokens(p)
        else:
            folds += 1
        assert not card_tokens(p) & _seat_cards(s.state, hidden)  # mucked / fold-won hands, even in the reasoning
        for x in p["table"]["seats"]:
            assert (x["cards"] is not None) == (x["seat"] in shown | {hero})
        if hidden:
            s.reveal_bot_cards()
            p = s.payload()
            assert p["bot_cards_revealed_on_request"] and p["bot_cards_visible"]
            assert _seat_cards(s.state, range(n)) <= card_tokens(p)
    assert folds and showdowns


def test_session_end_reveals_seed_only_then():
    s = session(4, 1, seed=None, bots=[MultiCallStationBot()] * 4)
    s.start_hand()
    play_out(s, MultiRandomBot(2))
    assert "session_secret" not in s.payload()
    s.end_session()
    p = s.payload()
    assert p["ended"] and p["session_seed"] is None and len(p["session_secret"]) == 64
    assert derive_seed(bytes.fromhex(p["session_secret"]), "hand", 1) == s.log.seed
    with pytest.raises(SessionError):
        s.start_hand()


def test_reasoning_hidden_during_hand_unless_live_and_scored_as_peeked():
    s = session(5, 0, seed=4, bots=[ChattyBot(s) for s in range(5)])
    human = MultiRandomBot(1)
    found = False
    for k in range(8):
        s.start_hand()
        if s.in_hand and s.reasoning:
            found = True
            p = s.payload()
            assert p["reasoning"] is None and p["reasoning_hidden"]
            if k % 2:
                s.set_live(True)
        play_out(s, human)
        s.set_live(False)
        p = s.payload()
        assert len(p["reasoning"]) == sum(1 for r in s.log.actions if r.player != 0)
        assert {e["seat"] for e in p["reasoning"]} <= {1, 2, 3, 4}
    assert found
    sc = s.payload()["score"]
    assert sc["clean_hands"] + sc["peeked_hands"] == 8 and sc["peeked_hands"] > 0


# --- determinism ---------------------------------------------------------------------------------


def test_same_seed_same_actions_same_hands():
    cat = MultiCatalog()
    keys = ["random", None, "raise", "call", "random", "random"]
    runs = []
    for _ in range(2):
        s = MultiPlaySession.create(cat, keys, 1, seed=77)
        assert s.hand_seed_for(3) == derive_seed(seed_secret(77), "hand", 3)
        human = MultiRandomBot(8)
        logs = []
        for _ in range(10):
            s.start_hand()
            play_out(s, human)
            logs.append(s.log)
        runs.append(logs)
    assert runs[0] == runs[1]
    twin = MultiRandomBot(derive_seed(seed_secret(77), "bot", 0))
    s = MultiPlaySession.create(cat, keys, 1, seed=77)
    assert s.bots[0]._rng.getstate() == twin._rng.getstate()
    other = MultiPlaySession.create(cat, keys, 1, seed=None)
    assert other.hand_seed_for(1) != s.hand_seed_for(1)


def test_replay_is_deterministic_and_equals_play_match_multi():
    cat = MultiCatalog()
    keys = ["random", "raise", "call", "random", "random", "call", "raise"]
    jobs = [MultiReplayJob(cat, keys, seed=31, n_hands=14) for _ in range(2)]
    for job in jobs:
        job.run()
        assert job.status == "done", job.error
    assert jobs[0].logs == jobs[1].logs
    secret = seed_secret(31)
    bots = [cat.make(k, derive_seed(secret, "bot", s)) for s, k in enumerate(keys)]
    assert jobs[0].logs == play_match_multi(bots, 14, 31)
    other = MultiReplayJob(cat, keys, seed=32, n_hands=14)
    other.run()
    assert other.logs != jobs[0].logs


def test_replay_frames():
    job = MultiReplayJob(MultiCatalog(), ["random"] * 8, seed=5, n_hands=10)
    job.run()
    n_side = 0
    for i in range(10):
        h = job.hand_payload(i)
        json.dumps(h, allow_nan=False)
        log = job.hands[i]["log"]
        frames = h["frames"]
        assert sum(1 for f in frames if f["event"]["type"] == "action") == len(log.actions) == len(h["reasoning"])
        assert sum(1 for f in frames if f["event"]["type"] == "deal") == {0: 0, 3: 1, 4: 2, 5: 3}[len(log.board)]
        assert frames[-1]["result"]["net"] == list(log.result.net)
        for f in frames:
            assert all(seat["cards"] is not None for seat in f["table"]["seats"])
            assert sum(st["stack"] for st in f["table"]["seats"]) + f["table"]["pot"] == 80_000
        n_side += len(frames[-1]["result"]["pots"]) > 1
        assert {e["detail"]["kind"] for e in h["reasoning"]} <= {"rule"}
    st = job.status_payload()
    assert sum(st["total"]) == 0 and len(st["bots"]) == 8 and st["bots"][0]["name"] == "RandomBot #0"
    assert len(job.export()["hands"]) == 10
    with pytest.raises(SessionError):
        job.hand_payload(10)
    with pytest.raises(SessionError):
        MultiReplayJob(MultiCatalog(), ["call"] * 3, seed=1, n_hands=201)
    with pytest.raises(SessionError):
        MultiReplayJob(MultiCatalog(), ["call"] * 9, seed=1, n_hands=2)


# --- side pots in the views ---------------------------------------------------------------------


def test_side_pots_in_table_and_result_views():
    names = ["Short", "Mid", "Big"]
    hole = [(Card.from_str("As"), Card.from_str("Ad")), (Card.from_str("Ks"), Card.from_str("Kd")),
            (Card.from_str("Qs"), Card.from_str("Qd"))]
    st = new_hand((500, 3000, 10_000), 0, 50, 100, hole)  # SB 1, BB 2, UTG 0
    st = apply_action(st, 0, Raise(500))  # all-in
    st = apply_action(st, 1, Raise(3000))  # all-in
    st = apply_action(st, 2, Call())
    view = table_view(st, names, {0, 1, 2}, [True] * 3)
    assert view["pots"] == []  # preflop: everything is still in front of the players
    board = [Card.from_str(c) for c in ("2c", "7d", "9h", "Jc", "3s")]
    st = deal_board(st, board[:3])
    view = table_view(st, names, {0, 1, 2}, [True] * 3)
    assert [(p["label"], p["amount"], p["eligible"]) for p in view["pots"]] == [
        ("Main pot", 1500, [0, 1, 2]), ("Side pot 1", 5000, [1, 2])]
    assert [x["all_in"] for x in view["seats"]] == [True, True, False]
    st = deal_board(deal_board(st, board[3:4]), board[4:])
    assert st.is_hand_over
    r = result_view(make_log(st, None), names)
    assert [(p["label"], p["amount"], p["winners"]) for p in r["pots"]] == [
        ("Main pot", 1500, [0]), ("Side pot 1", 5000, [1])]
    assert r["headline"] == "Short wins Main pot 1,500 with pair; Mid wins Side pot 1 5,000 with pair"
    assert r["net"] == [1000, 2000, -3000]


# --- the bot registry -----------------------------------------------------------------------------


@pytest.fixture()
def registry():
    saved = dict(MULTI_BOTS)
    saved_err = mw.optional_import_error
    yield MULTI_BOTS
    MULTI_BOTS.clear()
    MULTI_BOTS.update(saved)
    mw.optional_import_error = saved_err


class _FakeEquity(MultiCallStationBot):
    def __init__(self, seed=None, debug=False, samples=500):
        self.seed, self.debug, self.samples = seed, debug, samples


class _FakeCFR(MultiCallStationBot):
    def __init__(self, blueprint, seed=None):
        self.blueprint, self.seed = blueprint, seed


def test_register_one_line_and_build(registry):
    register_multi_bot_class("fake", "FakeBot", "a test bot", _FakeEquity)
    cat = MultiCatalog(overrides={"fake": {"samples": 40, "unknown_kw": 1}})
    assert {"key": "fake", "name": "FakeBot", "description": "a test bot", "available": True,
            "unavailable_reason": None} in cat.listing()
    bot = cat.make("fake", 7)
    assert (bot.seed, bot.debug, bot.samples) == (7, True, 40)  # unknown kwargs filtered out
    register_multi_bot_class("fakecfr", "FakeCFR", "needs a model", _FakeCFR)
    ok, why = cat.available("fakecfr")
    assert not ok and why
    with pytest.raises(SessionError):
        cat.make("fakecfr", 1)
    cat.blueprint = object()
    assert cat.make("fakecfr", 3).blueprint is cat.blueprint
    assert not cat.available("nope")[0] and not cat.available(None)[0]


def test_optional_bots_auto_register(registry, monkeypatch):
    fake = types.ModuleType("nutsfinder.bots.multiway")
    fake.MultiEquityBot = _FakeEquity
    fake.MultiCFRBot = _FakeCFR
    monkeypatch.setitem(sys.modules, "nutsfinder.bots.multiway", fake)
    monkeypatch.setattr(mw, "OPTIONAL_SUBMODULES", ())
    assert register_optional_multi_bots() == ["equity", "cfr"]
    assert MULTI_BOTS["equity"].name == "EquityBot" and MULTI_BOTS["cfr"].needs_blueprint
    monkeypatch.setitem(sys.modules, "nutsfinder.bots.multiway", None)  # "not installed": ImportError
    assert register_optional_multi_bots() == []
    assert mw.optional_import_error is None


def test_a_crashing_bot_abandons_the_hand_not_the_session():
    class Crash(MultiCallStationBot):
        def act(self, view, hole_cards):
            raise RuntimeError("boom")

    s = session(3, 0, seed=1, bots=[None, Crash(), MultiCallStationBot()])
    for _ in range(3):
        s.start_hand()
        if s.error:
            break
        play_out(s, MultiCallStationBot())
    assert s.error and "boom" in s.error
    p = s.payload()
    assert p["status"] == "error"
    s.start_hand()  # the next hand can still be dealt


# --- HTTP --------------------------------------------------------------------------------------


@pytest.fixture()
def server():
    srv = make_server(BotCatalog(None), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def jcall(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_http_multiway_smoke_and_leaks(server):
    srv, base = server
    status, bots = jcall(base, "GET", "/api/bots")
    assert status == 200 and bots["max_players"] == 8 and bots["min_multi_players"] == 3
    assert {b["key"] for b in bots["multi_bots"]} >= {"random", "call", "raise"}
    sid = jcall(base, "POST", "/api/session", {})[1]["session"]

    bad = [{"players": 9, "fill": "call"}, {"players": 1, "fill": "call"}, {"players": "x", "fill": "call"},
           {"players": 6, "hero_seat": 6, "fill": "call"}, {"players": 6, "seats": ["call"] * 5},
           {"players": 6, "fill": "skynet"}, {"players": 6}, {"players": 6, "fill": "call", "keep_stacks": "yes"},
           {"players": 6, "fill": "call", "seed": -1}, {"players": 3, "seats": ["call", None, 5], "hero_seat": 1},
           {"players": 6, "fill": "cfr"}]
    for body in bad:
        st, err = jcall(base, "POST", "/api/play/new", {"session": sid, **body})
        assert st == 400, (body, st, err)

    status, p = jcall(base, "POST", "/api/play/new", {"session": sid, "players": 6, "hero_seat": 2,
                                                      "seats": ["random", "raise", None, None, "random", "call"],
                                                      "fill": "call", "seed": "4242", "keep_stacks": True})
    assert status == 200 and p["multiway"] and p["n_players"] == 6 and p["hero_seat"] == 2
    assert [x["key"] for x in p["players"]] == ["random", "raise", None, "call", "random", "call"]
    app = srv.app

    def play():
        return app._sessions[sid].play

    def check(payload):
        s = play()
        if s.log is None and s.state is not None:
            allowed = _seat_cards(s.state, [2]) | {str(c) for c in s.state.board}
            assert card_tokens(payload) <= allowed
            assert payload["hand"]["seed"] is None
        assert "session_seed" not in payload and "session_secret" not in payload

    check(p)
    assert jcall(base, "POST", "/api/play/live", {"session": sid, "on": True})[0] == 200
    hands = 0
    while hands < 6:
        check(jcall(base, "GET", "/api/play/state?session=" + sid)[1])
        if p["status"] == "your_turn":
            kind = "call" if p["legal"]["can_call"] else "check"
            status, p = jcall(base, "POST", "/api/play/act", {"session": sid, "action": kind, "expected_step": p["step"]})
            assert status == 200, p
            assert jcall(base, "POST", "/api/play/act", {"session": sid, "action": kind,
                                                        "expected_step": p["step"] - 1})[0] == 409  # stale
        else:
            assert p["status"] == "hand_over", p.get("error")
            hands += 1
            s = play()
            hidden = [x for x in range(6) if x != 2 and x not in s.log.show_order]
            assert not card_tokens(p) & _seat_cards(s.state, hidden)
            if hidden:
                status, q = jcall(base, "POST", "/api/play/reveal", {"session": sid})
                assert status == 200 and _seat_cards(s.state, hidden) <= card_tokens(q)
            status, p = jcall(base, "POST", "/api/play/next", {"session": sid, "expected_step": p["step"]})
            assert status == 200, p
        check(p)
    while p["status"] == "your_turn":
        p = jcall(base, "POST", "/api/play/act", {"session": sid, "action": "fold", "expected_step": p["step"]})[1]
    status, p = jcall(base, "POST", "/api/play/end", {"session": sid})
    assert status == 200 and p["ended"] and p["session_seed"] == 4242

    # replay: N bots
    for body in ({"bots": ["call"] * 9, "hands": 2}, {"bots": ["call", "nope", "call"], "hands": 2},
                 {"bots": "call", "hands": 2}, {"bots": ["call"] * 3, "hands": 999}):
        assert jcall(base, "POST", "/api/replay/start", {"session": sid, **body})[0] == 400
    status, r = jcall(base, "POST", "/api/replay/start", {"session": sid, "bots": ["random", "raise", "call", "random"],
                                                          "seed": 4, "hands": 5})
    assert status == 200 and r["multiway"]
    import time

    for _ in range(200):
        r = jcall(base, "GET", "/api/replay/status?session=" + sid)[1]
        if r["status"] != "running":
            break
        time.sleep(0.05)
    assert r["status"] == "done" and r["done"] == 5 and len(r["total"]) == 4
    status, h = jcall(base, "GET", f"/api/replay/hand?session={sid}&index=4")
    assert status == 200 and h["multiway"] and h["frames"][-1]["result"]
    assert jcall(base, "GET", f"/api/replay/hand?session={sid}&index=5")[0] == 400
    status, ex = jcall(base, "GET", "/api/replay/export?session=" + sid)
    assert status == 200 and ex["format"] == "nutsfinder-replay-multi-1"

    # heads-up still goes the old way, also with players=2
    status, p = jcall(base, "POST", "/api/play/new", {"session": sid, "bot": "call", "players": 2})
    assert status == 200 and "multiway" not in p and p["bot"]["key"] == "call"
    status, r = jcall(base, "POST", "/api/replay/start", {"session": sid, "bot_a": "call", "bot_b": "random", "hands": 2})
    assert status == 200 and "multiway" not in r


def test_http_page_has_multiway_controls(server):
    _, base = server
    with urllib.request.urlopen(base + "/", timeout=10) as res:
        page = res.read().decode()
    for needle in ('id="nPlayers"', 'id="heroSeat"', 'id="fillSel"', 'id="keepStacks"', 'id="rPlayers"',
                   'id="seatCfg"', 'id="reasonSeat"', 'id="potres"'):
        assert needle in page


# --- the tiered multiway bots, when nutsfinder.bots.multiway provides them ----------------------------

MODEL = Path(__file__).resolve().parents[1] / "models" / "holdem_official" / "latest.npz"
SMALL = {"potodds": {"samples": 40}, "equity": {"samples": 40}, "bayes": {"samples": 40},
         "cfr": {"fallback_samples": 40}}


def _tier_params():
    """Every registered tier bot; ``cfr`` only runs when the trained model is present."""
    out = []
    for key, *_ in mw.OPTIONAL_MULTI_BOTS:
        if key not in MULTI_BOTS:
            continue
        marks = ()
        if MULTI_BOTS[key].needs_blueprint:
            marks = pytest.mark.skipif(not MODEL.is_file(), reason="trained CFR model not present")
        out.append(pytest.param(key, marks=marks))
    return out or [pytest.param("none", marks=pytest.mark.skip(reason="no tiered multiway bots registered"))]


@pytest.mark.parametrize("key", _tier_params())
def test_registered_tier_bots_play_without_leaking(key):
    needs_model = MULTI_BOTS[key].needs_blueprint
    cat = MultiCatalog(BotCatalog(MODEL if needs_model else None, overrides=SMALL))
    assert cat.available(key) == (True, None)
    s = MultiPlaySession.create(cat, [key, None, key, "call"], 1, seed=3)
    human = MultiRandomBot(5)
    for _ in range(4):
        s.start_hand()
        while s.human_to_act:
            s.live = True
            p = s.payload()
            allowed = _seat_cards(s.state, [1]) | {str(c) for c in s.state.board}
            assert card_tokens(p) <= allowed  # their decision records are masked too
            s.live = False
            s.apply_human(human.act(public_view(s.state, 1), s.state.hole_cards[1]))
        assert s.error is None, s.error
    p = s.payload()
    json.dumps(p, allow_nan=False)
    assert p["reasoning"] and {e["detail"]["kind"] for e in p["reasoning"]} <= {"ev", "cfr", "rule", "other"}
    if key == "bayes":  # per-opponent models (PooledModels.lines(seat)) shown for both Bayes seats
        assert set(p["bot_models"]) == {"0", "2"}
        assert any(line.startswith("about You (seat 1)") for line in p["bot_models"]["0"])


def test_bot_model_lines_single_and_per_seat():
    class Single:
        class model:  # noqa: N801 - mimics BayesBot.model
            @staticmethod
            def lines():
                return ["model (3 hands)"]

    class PerSeat:
        class models:  # noqa: N801 - mimics MultiBayesBot.models
            @staticmethod
            def lines(seat):
                return [f"seat {seat} model"] if seat != 2 else []

    names = ["A", "B", "You"]
    assert mw._bot_model_lines(Single(), [1, 2], names) == ["model (3 hands)"]
    assert mw._bot_model_lines(PerSeat(), [1, 2], names) == ["about B (seat 1):", "  seat 1 model"]
    assert mw._bot_model_lines(MultiCallStationBot(), [1], names) is None


def test_http_refused_post_answers_403_not_a_reset(server):
    """A wrong Content-Type is refused only after the body is read (Windows resets otherwise)."""
    import http.client
    from urllib.parse import urlparse

    _, base = server
    u = urlparse(base)
    for size in (10, 60_000, 64 * 1024, 200_000):
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        conn.request("POST", "/api/session", body=b"x" * size, headers={"Content-Type": "text/plain"})
        res = conn.getresponse()
        assert res.status == 403 and "Content-Type" in json.loads(res.read())["error"]
        conn.close()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request("POST", "/api/session", body=b"{" + b" " * 70_000 + b"}", headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    assert res.status == 413
    res.read()
    conn.close()
    # keep-alive still works after a refused (fully read) request
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request("POST", "/api/session", body=b"{}", headers={"Content-Type": "text/plain"})
    res = conn.getresponse()
    assert res.status == 403
    res.read()
    conn.request("POST", "/api/session", body=b"{}", headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    assert res.status == 200 and "session" in json.loads(res.read())
    conn.close()
