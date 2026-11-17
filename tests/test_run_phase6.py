"""Tests for the pure helpers of examples/run_phase6.py (the Phase 6 tier test) and its seed handling."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
from test_cfr_bot import BP

from nutsfinder.bots import CallStationBot, CFRBot, RandomBot
from nutsfinder.eval.sessions import Session, combine_sessions, play_session, session_seed
from nutsfinder.eval.stats import required_units, summarize

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_phase6_cli", ROOT / "examples" / "run_phase6.py")
rp6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp6)


def _sessions(totals, seed=7, opponent="X", pairs=2):
    out = []
    for i, t in enumerate(totals):
        pr = (t / 2, t / 2) + (0.0,) * (pairs - 2)
        out.append(Session(i, session_seed(seed, i), "A", opponent, tuple(x for p in pr for x in (p, 0.0)), pr, ()))
    return combine_sessions(out, seed, pairs)


def test_sessions_for_rounds_up_inflates_and_floors():
    assert rp6.sessions_for(78, 20, margin=1.0, floor=0, step=1) == required_units(78, 20) == 120
    assert rp6.sessions_for(78, 20, margin=1.2, floor=0, step=1) == required_units(1.2 * 78, 20)
    n = rp6.sessions_for(78, 20, margin=1.2, floor=0, step=10)
    assert n % 10 == 0 and 0 <= n - required_units(1.2 * 78, 20) < 10
    assert rp6.sessions_for(10, 50) == rp6.MIN_SESSIONS  # tiny requirement: the floor


def test_mde_and_power_are_inverse():
    sd, n = 93.6, 180
    e = rp6.mde(sd, n)
    assert rp6.power_at(e, sd, n) == pytest.approx(0.8, abs=1e-9)
    assert rp6.power_at(0.0, sd, n) == pytest.approx(0.025, abs=1e-9)  # upper tail only at a zero edge
    assert rp6.mde(sd, 4 * n) == pytest.approx(e / 2)


def test_paired_difference_is_per_session_and_checks_pairing():
    x, y = _sessions([10, 20, 30, 50]), _sessions([4, 4, 10, 30])
    d = rp6.paired_difference(x, y)
    ref = summarize([6, 16, 20, 20], hands_per_unit=4)
    assert d.mean_bb100 == pytest.approx(ref.mean_bb100) and d.se_bb100 == pytest.approx(ref.se_bb100)
    with pytest.raises(ValueError, match="seeds"):
        rp6.paired_difference(x, _sessions([1, 2, 3, 4], seed=8))
    with pytest.raises(ValueError, match="same sessions"):
        rp6.paired_difference(x, _sessions([1, 2, 3]))
    with pytest.raises(ValueError, match="opponent"):
        rp6.paired_difference(x, _sessions([1, 2, 3, 4], opponent="Y"))


def test_tier_verdict_is_the_preregistered_rule():
    assert rp6.tier_verdict(summarize([5.0, 6.0, 7.0, 5.5, 6.5])) == "beats"
    assert rp6.tier_verdict(summarize([-5.0, -6.0, -7.0, -5.5, -6.5])) == "loses"
    assert rp6.tier_verdict(summarize([5.0, -6.0, 7.0, -5.5, 1.0])) == "no detectable edge"
    assert rp6.tier_verdict(summarize([3.0, 3.0, 3.0])) == "no detectable edge"  # no variation is never significant
    s = summarize([1.0, 2.0, 3.0, 2.5, 1.5] * 4)
    assert s.ci_bb100[0] > 0 and rp6.tier_verdict(s) == "beats"


def test_splice_results_once_only():
    md = "# prereg\n\ntext\n\n" + rp6.RESULTS_MARKER + "\n\n"
    out = rp6.splice_results(md, "## Results\n\nnumbers\n")
    assert out.startswith("# prereg\n\ntext\n\n" + rp6.RESULTS_MARKER) and out.endswith("numbers\n")
    with pytest.raises(ValueError, match="already present"):
        rp6.splice_results(out, "again")
    with pytest.raises(ValueError, match="marker not found"):
        rp6.splice_results("# prereg without marker", "x")


def test_street_contributions_add_up_to_each_session():
    res = combine_sessions([play_session(RandomBot, CallStationBot, 6, 11, s) for s in range(4)], 11, 6)
    st = rp6.street_contributions(res)
    assert sum(v["share"] for v in st.values()) == pytest.approx(1.0)
    assert sum(v["bb100"] for v in st.values()) == pytest.approx(res.mean_bb100)
    assert all(v["ci_low"] <= v["bb100"] <= v["ci_high"] for v in st.values())
    lean = combine_sessions([play_session(RandomBot, CallStationBot, 6, 11, s, keep_games=False) for s in range(2)], 11, 6)
    with pytest.raises(ValueError, match="hand logs"):
        rp6.street_contributions(lean)


def test_merge_peaks_and_memory_probe():
    peaks = rp6.merge_peaks([(1, {"a": 1.0, "b": 5.0}), (1, {"a": 3.0, "b": 2.0}), (2, None), (2, {"a": 0.5, "b": 0.5})])
    assert peaks == {1: {"a": 3.0, "b": 5.0}, 2: {"a": 0.5, "b": 0.5}}
    m = rp6.process_memory_mb()
    assert m is None or m["peak_working_set"] > 0


def test_file_sha256(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    assert rp6.file_sha256(p) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_cfrbot_constructor_seed_is_overridden_in_session_play():
    # duplicate_match reseeds every bot before every hand, so CFRBot's default seed (0) is irrelevant.
    a = play_session(functools.partial(CFRBot, BP, 0), RandomBot, 8, 5, 0, keep_games=False)
    b = play_session(functools.partial(CFRBot, BP, 987_654), RandomBot, 8, 5, 0, keep_games=False)
    c = play_session(functools.partial(CFRBot, BP, 0), RandomBot, 8, 5, 1, keep_games=False)
    assert a.hand_results == b.hand_results
    assert a.hand_results != c.hand_results  # the session seed does matter


def test_design_is_paired_and_the_seed_is_fresh():
    assert rp6.PAIRINGS[0] == rp6.PRIMARY == ("CFRBot", "BayesBot")
    assert {b for a, b in rp6.PAIRINGS if a == "CFRBot"} - {"BayesBot"} == set(rp6.OPPONENTS)
    assert {b for a, b in rp6.PAIRINGS if a == "BayesBot"} == set(rp6.OPPONENTS)
    if rp6.N_SESSIONS:
        assert set(rp6.N_SESSIONS) == set(rp6.PAIRINGS)
        assert all(rp6.N_SESSIONS[("CFRBot", x)] == rp6.N_SESSIONS[("BayesBot", x)] for x in rp6.OPPONENTS)
        assert all(n >= rp6.MIN_SESSIONS for n in rp6.N_SESSIONS.values())
    # Seeds used before Phase 6 (results/*.json master seeds, tuning/holdout seeds, training and table seeds).
    assert rp6.MASTER_SEED not in {999, 555, 556, 777, 4242, 20261115, 20261116, 20261117, 7, 61}
    assert not 20261116 <= rp6.MASTER_SEED <= 20261116 + 100 * 169 + 100  # preflop table seeds
    assert rp6.PILOT_SEED != rp6.MASTER_SEED
    assert math.isfinite(rp6.SD_MARGIN) and rp6.SD_MARGIN >= 1


def test_results_markdown_renders_a_full_summary():
    # The official run writes this at the very end: it must not crash on real rows.
    res = combine_sessions([play_session(RandomBot, CallStationBot, 250, 13, s) for s in range(2)], 13, 250)
    rows = {f"{a} vs {b}": rp6.pairing_row((a, b), res, [1.0, 1.2], 22, "stem") for a, b in rp6.PAIRINGS}
    diff = {"mean_bb100": -1.0, "ci95_bb100": [-2.0, 0.0], "se_bb100": 0.5, "sd_bb100_per_session": 3.0,
            "p_value": 0.05, "n_sessions": 2}
    mem = {"parent_peak": {"peak_working_set": 150.0}, "workers_peak": {},
           "max_worker_peak_working_set": 140.0, "max_worker_peak_private": 130.0}
    summary = {"rows": rows, "paired_differences": {x: diff for x in rp6.OPPONENTS}, "elapsed_s": 12.0,
               "memory_mb": mem, "primary": rows["CFRBot vs BayesBot"]}
    meta = {"date": "2026-01-01", "commit": "abc1234", "master_seed": 1, "processes": 22,
            "model": {"description": "CFRBot(model=x)", "sha256": "0" * 64}}
    md = rp6.results_markdown(summary, meta)
    assert "Primary test: CFRBot vs BayesBot" in md and "Pre-registered verdict:" in md
    assert md.count("| CFRBot vs ") == 6 and md.count("| BayesBot vs ") == 5
    out = rp6.splice_results("prereg\n" + rp6.RESULTS_MARKER + "\n", md)
    assert out.endswith(md.rstrip() + "\n")
    assert rows["CFRBot vs BayesBot"]["tier_verdict"] in {"beats", "loses", "no detectable edge"}
    # The memory probe can return None (or NaN off Windows): the report must still render.
    summary["memory_mb"] = {"parent_peak": None, "workers_peak": {}, "max_worker_peak_working_set": None,
                            "max_worker_peak_private": float("nan")}
    assert "peak memory: worker working set max n/a" in rp6.results_markdown(summary, meta)


def test_official_mode_refuses_allow_dirty(monkeypatch):
    called = []
    monkeypatch.setattr(rp6, "commit_hash", lambda allow: called.append(allow) or "x")
    with pytest.raises(SystemExit, match="allow-dirty is refused"):
        rp6.main(["--allow-dirty"])
    assert not called  # refused before anything else happens


def test_official_per_pairing_files_carry_the_phase6_tag(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(rp6, "check_model", lambda: {"description": "d", "sha256": "0" * 64})
    monkeypatch.setattr(rp6, "commit_hash", lambda allow: "abc")
    monkeypatch.setattr(rp6, "N_SESSIONS", {p: 1 for p in rp6.PAIRINGS})
    monkeypatch.setattr(rp6, "sessions_for", lambda *a, **k: 0)
    monkeypatch.setattr(rp6, "splice_results", lambda md, results: md)
    monkeypatch.setattr(rp6.PHASE6_MD.__class__, "read_text", lambda self, encoding=None: "")

    def fake_run(n_sessions, seed, stem_prefix, meta, processes):
        seen.append((seed, stem_prefix))
        raise KeyboardInterrupt  # stop before playing anything

    monkeypatch.setattr(rp6, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        rp6.main([])
    assert seen == [(rp6.MASTER_SEED, "phase6_")]
