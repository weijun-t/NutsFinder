"""The browser poker table (``nutsfinder.table``): session logic, fairness to the bot, leaks, HTTP.

- Session logic without HTTP: dealing, acting, illegal input rejected and
  nothing changed, chips conserved at every step, button alternation, score.
- The bot is treated exactly as in a match: a spy bot in a PlaySession sees
  the same ``act`` views / hole cards and the same end-of-hand summaries as
  in ``runner.play_hand`` for the same seeds and the same human actions.
- Replays are deterministic and equal to ``play_hand`` with the same bots.
- The human-facing payload hides the bot's reasoning during a hand (unless
  the live toggle is on) and never contains the bot's hole cards before
  they may be seen (showdown, or "show bot cards" after the hand).
- An HTTP smoke test on a random free port, including the HTML page.
"""

from __future__ import annotations

import json
import random
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nutsfinder.bots import BayesBot, CallStationBot, RandomBot
from nutsfinder.engine.actions import Action, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState, public_view
from nutsfinder.engine.bot import HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.runner import hand_seeds, play_hand
from nutsfinder.table.server import make_server
from nutsfinder.table.session import (
    BotCatalog,
    PlaySession,
    ReplayJob,
    SessionError,
    StaleRequest,
    derive_seed,
    seed_secret,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "holdem_official" / "latest.npz"
TURNS = ROOT / "models" / "turn_ehs.npz"
FAST = {"potodds": {"samples": 40}, "equity": {"samples": 40}, "bayes": {"samples": 40, "bluff_samples": 20}}
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="trained CFR model not present")


def fast_catalog(model: Path | None = None) -> BotCatalog:
    return BotCatalog(model, TURNS if model else None, overrides=FAST)


class SpyBot(RandomBot):
    """A RandomBot that keeps everything it is handed."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.acts: list[tuple[PublicState, tuple[Card, ...]]] = []
        self.ends: list[HandSummary] = []

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        self.acts.append((state, hole_cards))
        return super().act(state, hole_cards)

    def end_hand(self, summary: HandSummary) -> None:
        self.ends.append(summary)


def play_out(session: PlaySession, human: RandomBot, check=None) -> None:
    """Let a RandomBot play the human seat until the hand is over (calling ``check`` at each step)."""
    while session.human_to_act:
        if check:
            check(session)
        view = public_view(session.state, 0)
        session.apply_human(human.act(view, session.state.hole_cards[0]))
    if check:
        check(session)


def chips_conserved(session: PlaySession) -> None:
    st = session.state
    assert sum(st.stacks) + st.pot == sum(st.starting_stacks)


# --- session logic ---------------------------------------------------------------------


def test_deal_act_score_and_button():
    s = PlaySession(CallStationBot(), seed=11, bot_key="call", bot_name="CallStationBot")
    assert s.payload()["status"] == "no_hand"
    human = RandomBot(5)
    total = 0
    for k in range(12):
        s.start_hand()
        assert s.state.button == k % 2
        assert len(s.state.hole_cards) == 2
        play_out(s, human, chips_conserved)
        assert s.log is not None and s.error is None
        total += s.log.result.net[0]
        p = s.payload()
        assert p["status"] == "hand_over"
        assert p["score"]["hands"] == k + 1
        assert p["score"]["chips"] == total
        assert p["score"]["bb"] == total / 100
        assert sum(p["result"]["net"]) == 0
    assert s.total_net == total


def test_illegal_actions_are_rejected_and_change_nothing():
    s = PlaySession(CallStationBot(), seed=3)
    with pytest.raises(SessionError):
        s.act("fold")  # no hand yet
    s.start_hand()  # human on the button preflop: facing the big blind
    assert s.human_to_act
    before = s.state
    bad = [("check", None), ("bet", 150), ("raise", 150), ("raise", 10_001), ("raise", True), ("raise", "lots"),
           ("raise", None), ("raise", 300.5), ("shove", None), (None, None), (7, None), ({"x": 1}, None)]
    for kind, amount in bad:
        with pytest.raises(SessionError):
            s.act(kind, amount)
        assert s.state is before
    with pytest.raises(SessionError):
        s.start_hand()  # mid-hand
    with pytest.raises(SessionError):
        s.reveal_bot_cards()
    with pytest.raises(SessionError):
        s.set_live("yes")
    s.act("raise", "300")  # digit strings are accepted (form input)
    assert s.state.history[0].action == Raise(300)  # then the call station called
    assert s.state.history[1].action == Call()


def test_presets_match_bot_sizing():
    from nutsfinder.bots.common import raise_to_for_fraction

    s = PlaySession(CallStationBot(), seed=1)
    s.start_hand()
    legal = s.payload()["legal"]
    view = public_view(s.state, 0)
    presets = {p["label"]: p["to"] for p in legal["presets"]}
    assert presets["½ pot"] == raise_to_for_fraction(view, 0.5)
    assert presets["Pot"] == raise_to_for_fraction(view, 1.0) == 300
    assert presets["All-in"] == legal["max_to"] == 10_000
    assert presets["Min"] == legal["min_to"] == 200


def test_bot_sees_exactly_what_play_hand_shows_it():
    seed = 2024
    app_spy, app_human = SpyBot(77), RandomBot(99)
    s = PlaySession(app_spy, seed=seed)
    played = []
    for _ in range(40):
        s.start_hand()
        play_out(s, app_human)
        played.append((s.log.seed, s.log.button, s.log))
    run_spy, run_human = SpyBot(77), RandomBot(99)
    for hand_seed, button, log in played:
        assert play_hand([run_human, run_spy], seed=hand_seed, button=button) == log
    assert len(app_spy.acts) == len(run_spy.acts) > 0
    assert app_spy.acts == run_spy.acts
    assert app_spy.ends == run_spy.ends
    assert any(e.shown_cards[0] is not None for e in app_spy.ends)  # showdowns happened ...
    assert any(e.shown_cards[0] is None for e in app_spy.ends)  # ... and folds hid the cards


def test_bayes_keeps_learning_across_hands():
    cat = fast_catalog()
    s = PlaySession(cat.make("bayes", 4), seed=4, bot_key="bayes", bot_name="BayesBot")
    human = RandomBot(1)
    for _ in range(4):
        s.start_hand()
        play_out(s, human)
    assert isinstance(s.bot, BayesBot) and s.bot.model.hands == 4
    p = s.payload()
    assert p["bot_model"][0].startswith("model (4 hands)")


# --- what the human-facing payload may contain ---------------------------------------------

CARD_TOKEN = re.compile(r"(?<![A-Za-z0-9])((?:[2-9TJQKA][cdhs])+)(?![A-Za-z0-9])")


def card_tokens(obj) -> set[str]:
    """Every card mentioned in any string of a JSON-like object ("As", "AsKs", "[Kh 7h 2s]")."""
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


def test_card_token_finder():
    assert card_tokens({"a": ["As", "x"], "b": "*** FLOP *** [Kh 7h 2s]", "c": "flop AsKs board"}) == {
        "As", "Kh", "7h", "2s", "Ks"}
    assert card_tokens("Thinking about The pot") == set()


def test_reasoning_hidden_during_hand_unless_live():
    s = PlaySession(fast_catalog().make("equity", 8), seed=8, bot_key="equity", bot_name="EquityBot")
    human = RandomBot(2)
    found = False
    for _ in range(30):
        s.start_hand()
        while s.human_to_act:
            if s.reasoning:  # the bot has decided something this hand
                found = True
                p = s.payload()
                assert p["reasoning"] is None and p["reasoning_hidden"]
                assert "std_error" not in json.dumps(p)
                s.set_live(True)
                p = s.payload()
                assert p["reasoning"] and p["reasoning"][0]["detail"]["kind"] == "ev"
                assert all(e["hole_cards"] is None for e in p["reasoning"])
                bot_cards = {str(c) for c in s.state.hole_cards[1]}
                assert not card_tokens(p) & bot_cards  # live peeking shows numbers, never cards
                s.set_live(False)
            s.apply_human(human.act(public_view(s.state, 0), s.state.hole_cards[0]))
        p = s.payload()
        assert p["reasoning"] is not None  # after the hand: shown
        assert len(p["reasoning"]) == sum(1 for r in s.log.actions if r.player == 1)
    assert found


@pytest.mark.parametrize("key", ["random", "raise", "potodds", "bayes", pytest.param("cfr", marks=needs_model)])
def test_bot_hole_cards_never_in_payload_before_they_may_be_seen(key):
    cat = fast_catalog(MODEL if key == "cfr" else None)
    s = PlaySession(cat.make(key, 6), seed=6, bot_key=key, bot_name=cat.name(key))
    human = RandomBot(3)
    folds = showdowns = 0
    for _ in range(25):
        s.start_hand()
        bot_cards = {str(c) for c in s.state.hole_cards[1]}

        def check(sess):
            p = sess.payload()
            json.dumps(p, allow_nan=False)  # valid JSON for the browser
            allowed = {str(c) for c in sess.state.hole_cards[0]} | {str(c) for c in sess.state.board}
            assert "session_seed" not in p and "session_secret" not in p  # would give every deck away
            if sess.log is None:
                assert card_tokens(p) <= allowed, (card_tokens(p) - allowed)
                assert p["hand"]["seed"] is None  # the deck seed would give the cards away
                assert p["table"]["seats"][1]["cards"] is None

        play_out(s, human, check)
        p = s.payload()
        if s.log.folded is None:
            showdowns += 1
            assert p["table"]["seats"][1]["cards"] == [str(c) for c in s.state.hole_cards[1]]
            assert bot_cards <= card_tokens(p)
        else:
            folds += 1
            assert not card_tokens(p) & bot_cards  # not even in the reasoning
            s.reveal_bot_cards()
            p = s.payload()
            assert p["bot_cards_revealed_on_request"]
            assert p["table"]["seats"][1]["cards"] is not None
            assert bot_cards <= card_tokens(p)
    assert folds > 0


# --- replay ----------------------------------------------------------------------------------


def test_replay_is_deterministic_and_equals_play_hand():
    cat = fast_catalog()
    jobs = [ReplayJob(cat, "potodds", "bayes", seed=31, n_hands=12) for _ in range(2)]
    for job in jobs:
        job.run()
        assert job.status == "done", job.error
    assert jobs[0].logs == jobs[1].logs
    def decisions(job):  # everything but the timing
        return [[{k: v for k, v in e.items() if k != "seconds"} | {"detail": {k: v for k, v in e["detail"].items()
                 if k not in ("seconds", "text")}} for e in h["reasoning"]] for h in job.hands]

    assert decisions(jobs[0]) == decisions(jobs[1])
    # Same as play_hand with identically built bots and play_match's schedule.
    from nutsfinder.engine.runner import hand_seeds

    secret = seed_secret(31)
    bots = [cat.make("potodds", derive_seed(secret, "bot", 0)), cat.make("bayes", derive_seed(secret, "bot", 1))]
    expected = [play_hand(bots, s, button=i % 2) for i, s in enumerate(hand_seeds(31, 12))]
    assert jobs[0].logs == expected
    other = ReplayJob(cat, "potodds", "bayes", seed=32, n_hands=12)
    other.run()
    assert other.logs != expected


def test_replay_frames_and_reasoning():
    job = ReplayJob(fast_catalog(), "equity", "call", seed=5, n_hands=6)
    job.run()
    for i in range(6):
        h = job.hand_payload(i)
        json.dumps(h, allow_nan=False)
        log = job.hands[i]["log"]
        frames = h["frames"]
        n_deals = sum(1 for f in frames if f["event"]["type"] == "deal")
        n_acts = sum(1 for f in frames if f["event"]["type"] == "action")
        assert n_acts == len(log.actions) == len(h["reasoning"])
        assert n_deals == {0: 0, 3: 1, 4: 2, 5: 3}[len(log.board)]
        assert frames[-1]["result"]["net"] == list(log.result.net)
        for f in frames:  # all hole cards visible in a replay
            assert all(seat["cards"] is not None for seat in f["table"]["seats"])
            assert sum(st["stack"] for st in f["table"]["seats"]) + f["table"]["pot"] == 20_000
        kinds = {e["detail"]["kind"] for e in h["reasoning"] if e["seat"] == 0}
        assert kinds <= {"ev"}
        assert {e["detail"]["kind"] for e in h["reasoning"] if e["seat"] == 1} <= {"rule"}
    exported = job.export()
    assert len(exported["hands"]) == 6 and exported["hands"][0]["text"].startswith("Hand (seed")
    with pytest.raises(SessionError):
        job.hand_payload(6)
    with pytest.raises(SessionError):
        ReplayJob(fast_catalog(), "call", "call", seed=1, n_hands=201)


# --- catalog / CFR ---------------------------------------------------------------------------


def test_missing_model_makes_cfr_unavailable():
    cat = BotCatalog(ROOT / "models" / "does_not_exist.npz")
    ok, why = cat.available("cfr")
    assert not ok and "not found" in why
    assert {b["key"]: b["available"] for b in cat.listing()}["cfr"] is False
    with pytest.raises(SessionError):
        cat.make("cfr", 1)
    assert not cat.available("nonsense")[0]


@needs_model
def test_cfr_play_and_replay():
    cat = fast_catalog(MODEL)
    assert cat.available("cfr") == (True, None)
    s = PlaySession(cat.make("cfr", 3), seed=3, bot_key="cfr", bot_name="CFRBot")
    human = RandomBot(4)
    for _ in range(4):
        s.start_hand()
        play_out(s, human)
        assert s.error is None
    p = s.payload()
    if p["reasoning"]:
        d = p["reasoning"][0]["detail"]
        assert d["kind"] == "cfr" and {"mode", "node", "bucket", "ehs", "strategy", "chosen"} <= set(d)
    job = ReplayJob(cat, "cfr", "bayes", seed=9, n_hands=4)
    job.run()
    assert job.status == "done", job.error
    h = job.hand_payload(0)
    assert {e["detail"]["kind"] for e in h["reasoning"]} <= {"cfr", "ev"}


# --- HTTP -------------------------------------------------------------------------------------


@pytest.fixture()
def server():
    srv = make_server(fast_catalog(), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(base: str, method: str, path: str, body=None, raw: bytes | None = None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, res.headers.get("Content-Type", ""), res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type", ""), err.read()


def jcall(base, method, path, body=None, raw=None):
    status, ctype, data = call(base, method, path, body, raw)
    assert ctype.startswith("application/json"), (status, ctype, data[:200])
    return status, json.loads(data)


def test_http_smoke(server):
    status, ctype, page = call(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"<title>NutsFinder Table</title>" in page and b"<script>" in page
    assert b"http://" not in page.replace(b"http://www.w3.org", b"")  # self-contained: no CDN

    status, bots = jcall(server, "GET", "/api/bots")
    assert status == 200 and {b["key"] for b in bots["bots"]} >= {"random", "call", "bayes", "cfr"}
    status, sess = jcall(server, "POST", "/api/session", {})
    sid = sess["session"]

    # bad input -> JSON errors, never a crash
    assert jcall(server, "POST", "/api/play/act", {"session": sid, "action": "fold"})[0] == 409
    assert jcall(server, "POST", "/api/play/new", raw=b"{not json")[0] == 400
    assert jcall(server, "POST", "/api/play/new", raw=b"[1, 2]")[0] == 400
    assert jcall(server, "POST", "/api/play/new", {"session": "nope", "bot": "call"})[0] == 404
    assert jcall(server, "POST", "/api/play/new", {"session": sid, "bot": "skynet"})[0] == 400
    assert jcall(server, "POST", "/api/play/new", {"session": sid, "bot": "call", "seed": -5})[0] == 400
    assert jcall(server, "GET", "/api/nothing")[0] == 404
    assert jcall(server, "GET", "/api/replay/hand?session=" + sid + "&index=x")[0] == 400

    status, p = jcall(server, "POST", "/api/play/new", {"session": sid, "bot": "call", "seed": "12345"})
    assert status == 200 and p["status"] in ("your_turn", "hand_over")
    assert "session_seed" not in p and p["known_deck"] is True
    assert jcall(server, "POST", "/api/play/act", {"session": sid, "action": "raise", "amount": 1})[0] == 400
    assert jcall(server, "POST", "/api/play/live", {"session": sid, "on": True})[1]["live"] is True
    hands = 0
    while hands < 3:
        if p["status"] == "your_turn":
            kind = "call" if p["legal"]["can_call"] else "check"
            status, p = jcall(server, "POST", "/api/play/act",
                              {"session": sid, "action": kind, "expected_step": p["step"]})
            assert status == 200, p
        else:
            assert p["status"] == "hand_over"
            hands += 1
            if p["bot_cards_visible"] is False:
                assert jcall(server, "POST", "/api/play/reveal", {"session": sid})[1]["bot_cards_visible"]
            status, p = jcall(server, "POST", "/api/play/next", {"session": sid, "expected_step": p["step"]})
            assert status == 200, p
    status, state = jcall(server, "GET", "/api/play/state?session=" + sid)
    assert status == 200 and state["score"]["hands"] >= 3 and state["busy"] is False

    status, r = jcall(server, "POST", "/api/replay/start",
                      {"session": sid, "bot_a": "random", "bot_b": "call", "seed": 4, "hands": 5})
    assert status == 200 and r["status"] in ("running", "done")
    assert jcall(server, "POST", "/api/replay/start",
                 {"session": sid, "bot_a": "random", "bot_b": "call", "seed": 4, "hands": 999})[0] == 400
    import time

    for _ in range(200):
        status, r = jcall(server, "GET", "/api/replay/status?session=" + sid)
        if r["status"] != "running":
            break
        time.sleep(0.05)
    assert r["status"] == "done" and r["done"] == 5
    status, h = jcall(server, "GET", f"/api/replay/hand?session={sid}&index=0")
    assert status == 200 and h["frames"] and h["frames"][-1]["result"]
    assert jcall(server, "GET", f"/api/replay/hand?session={sid}&index=5")[0] == 400
    status, ctype, data = call(server, "GET", "/api/replay/export?session=" + sid)
    assert status == 200 and len(json.loads(data)["hands"]) == 5


# --- seeds, stale requests, request hygiene (review fixes) ---------------------------------------


def test_seeds_are_hash_derived_and_separate():
    s = PlaySession(CallStationBot(), seed=77)
    human = RandomBot(1)
    seeds = []
    for _ in range(5):
        s.start_hand()
        play_out(s, human)
        seeds.append(s.log.seed)
    # sha256-derived (a revealed hand seed says nothing about the next: no Mersenne Twister state to recover)
    assert seeds == [derive_seed(seed_secret(77), "hand", n) for n in range(1, 6)]
    assert seeds != [random.Random(77).getrandbits(64) for _ in range(5)]
    # an unseeded session has a fresh 32-byte secret: different deals each time
    a, b = PlaySession(CallStationBot()), PlaySession(CallStationBot())
    assert a.hand_seed_for(1) != b.hand_seed_for(1)
    assert not a.user_seeded and s.user_seeded
    # replay bots never get a deck seed
    for seed in (0, 1, 31, 2**40):
        decks = set(hand_seeds(seed, 200))
        assert derive_seed(seed_secret(seed), "bot", 0) not in decks
        assert derive_seed(seed_secret(seed), "bot", 1) not in decks


def test_create_derives_bot_seed_from_other_label():
    cat = fast_catalog()
    p, q = PlaySession.create(cat, "random", 5), PlaySession.create(cat, "random", 5)
    assert p.user_seeded and p.hand_seed_for(1) == derive_seed(seed_secret(5), "hand", 1)
    twin = RandomBot(derive_seed(seed_secret(5), "bot", 1))
    assert p.bot._rng.getstate() == twin._rng.getstate()
    for sess in (p, q):  # same seed and same human actions -> the same hands (reproducible)
        human = RandomBot(8)
        for _ in range(5):
            sess.start_hand()
            play_out(sess, human)
    assert [h["net"] for h in p.history] == [h["net"] for h in q.history]


def test_session_end_reveals_seed_only_then():
    s = PlaySession(CallStationBot(), seed=None)
    s.start_hand()
    with pytest.raises(SessionError):
        s.end_session()  # mid-hand
    play_out(s, RandomBot(2))
    p = s.payload()
    assert "session_secret" not in p and "session_seed" not in p
    s.end_session()
    p = s.payload()
    assert p["ended"] and p["session_seed"] is None and len(p["session_secret"]) == 64
    assert derive_seed(bytes.fromhex(p["session_secret"]), "hand", 1) == s.log.seed
    with pytest.raises(SessionError):
        s.start_hand()


def test_peeking_is_scored_separately():
    s = PlaySession(CallStationBot(), seed=3)
    human = RandomBot(4)
    for k in range(6):
        s.start_hand()
        if k % 2 and s.in_hand:
            s.set_live(True)
        play_out(s, human)
        s.set_live(False)
    sc = s.payload()["score"]
    peeked = [h for h in s.history if h["peeked"]]
    assert sc["peeked_hands"] == len(peeked) > 0
    assert sc["clean_hands"] + sc["peeked_hands"] == sc["hands"] == 6
    assert sc["clean_chips"] == sum(h["net"] for h in s.history if not h["peeked"])
    assert sc["known_deck"] is True


def test_stale_step_rejected():
    s = PlaySession(CallStationBot(), seed=1)
    s.start_hand()
    step = s.step
    s.check_step(step)
    with pytest.raises(StaleRequest):
        s.check_step(step - 1)
    with pytest.raises(SessionError):
        s.check_step(None)
    s.act("call")
    with pytest.raises(StaleRequest):
        s.check_step(step)  # the same click again


def test_http_duplicate_act_and_request_hygiene(server):
    import http.client
    from urllib.parse import urlparse

    sid = jcall(server, "POST", "/api/session", {})[1]["session"]
    status, p = jcall(server, "POST", "/api/play/new", {"session": sid, "bot": "call", "seed": 9})
    while p["status"] != "your_turn":
        p = jcall(server, "POST", "/api/play/next", {"session": sid, "expected_step": p["step"]})[1]
    body = {"session": sid, "action": "call" if p["legal"]["can_call"] else "check", "expected_step": p["step"]}
    assert jcall(server, "POST", "/api/play/act", body)[0] == 200
    assert jcall(server, "POST", "/api/play/act", body)[0] == 409  # the double click
    no_step = {k: v for k, v in body.items() if k != "expected_step"}
    assert jcall(server, "POST", "/api/play/act", no_step)[0] in (400, 409)

    # malformed inputs: 400, never 500
    bad_bodies = [
        b'{"session": "%s", "bot": "call", "seed": "' % sid.encode() + b"9" * 5000 + b'"}',
        b'{"session": "%s", "bot": "call", "seed": ' % sid.encode() + b"9" * 5000 + b"}",
        b'{"session": "%s", "bot": "call", "seed": "\\u00b2"}' % sid.encode(),  # JSON escape for a superscript 2
        b"[" * 100_00 + b"]" * 100_00,
        b'{"a": ' * 3000 + b"1" + b"}" * 3000,
        b"\xff\xfe",
    ]
    for raw in bad_bodies:
        status, _ = jcall(server, "POST", "/api/play/new", raw=raw)
        assert status in (400, 413), raw[:40]
    state = jcall(server, "GET", "/api/play/state?session=" + sid)[1]
    if state["status"] == "your_turn" and (state["legal"]["can_raise"] or state["legal"]["can_bet"]):
        status, err = jcall(server, "POST", "/api/play/act", {"session": sid, "action": "raise", "amount": "\u00b2",
                                                              "expected_step": state["step"]})
        assert status == 400 and "whole number" in err["error"]
    for idx in ("%C2%B2", "9" * 50, "-1", "x"):  # %C2%B2 is a superscript 2
        assert jcall(server, "GET", f"/api/replay/hand?session={sid}&index={idx}")[0] in (400, 409)
    assert jcall(server, "POST", "/api/replay/start", {"session": sid, "bot_a": "call", "bot_b": "call",
                                                       "hands": "\u00b2"})[0] == 400

    # Host and Content-Type checks
    u = urlparse(server)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request("GET", "/api/bots", headers={"Host": "evil.example:80"})
    res = conn.getresponse()
    assert res.status == 403 and json.loads(res.read())["error"].startswith("forbidden Host")
    conn.request("GET", "/", headers={"Host": f"localhost:{u.port}"})
    res = conn.getresponse()
    assert res.status == 200
    res.read()
    conn.request("POST", "/api/session", body=b"{}", headers={"Content-Type": "text/plain"})
    res = conn.getresponse()
    assert res.status == 403
    res.read()
    conn.close()


def test_second_server_cannot_bind_same_port(server):
    from urllib.parse import urlparse

    port = urlparse(server).port
    with pytest.raises(OSError):
        make_server(fast_catalog(), "127.0.0.1", port)
