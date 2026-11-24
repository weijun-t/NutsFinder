"""Tests for examples/run_6max.py: rotation, pairing, determinism/resume, verdict logic, plan numbers, guards."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from nutsfinder.engine.actions import Call, Check, Fold
from nutsfinder.engine.betting import Street
from nutsfinder.eval.stats import required_units

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_6max_cli", ROOT / "examples" / "run_6max.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

SMOKE = ["--smoke", "--processes", "1"]
CHEAP_FIELD = ("Random", "CallStation", "AlwaysRaise", "Random", "CallStation")


@pytest.fixture
def cheap(monkeypatch, tmp_path):
    """Baseline bots only (milliseconds per hand), 4-block sessions, no git/model, outputs in tmp."""
    monkeypatch.setattr(sm, "RESULTS", tmp_path)
    monkeypatch.setattr(sm, "SMOKE_BLOCKS", 4)
    monkeypatch.setattr(sm, "WINDOW_BLOCKS", 2)
    monkeypatch.setattr(sm, "FIELDS", {"mixed": CHEAP_FIELD})
    monkeypatch.setattr(sm, "ARMS", {"mixed": ("Random", "CallStation", "AlwaysRaise")})
    monkeypatch.setattr(sm, "PRIMARY", ("mixed", "Random", "CallStation"))
    monkeypatch.setattr(sm, "SECONDARY_PAIRS", (("mixed", "AlwaysRaise", "CallStation"),))
    monkeypatch.setattr(sm, "commit_hash", lambda allow_dirty: "test")
    monkeypatch.setattr(sm, "check_model", lambda: {"sha256": "x" * 64, "description": "stub"})
    return tmp_path


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(sm, "save_outputs", lambda *a, **k: [])


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "ms_per_hand_wall"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


# --- Rotation and pairing --------------------------------------------------------------------------


def test_every_bot_sits_in_every_seat_exactly_once_per_deal():
    for s in range(10):
        order = sm.field_order("mixed", s)
        assert sorted(order) == sorted(sm.FIELDS["mixed"])
        layouts = [sm.seat_layout("mixed", "Bayes", s, r) for r in range(sm.N_SEATS)]
        for r, lay in enumerate(layouts):
            assert lay[r] == "Bayes" and [lay[(r + 1 + i) % 6] for i in range(5)] == list(order)
        for seat in range(sm.N_SEATS):
            assert Counter(lay[seat] for lay in layouts) == Counter(("Bayes",) + sm.FIELDS["mixed"])
    assert len({sm.field_order("mixed", s) for s in range(5)}) == 5  # the 5 cyclic orders, balanced


def test_field_bots_identical_across_arms():
    for s in range(5):
        for r in range(sm.N_SEATS):
            a, b = sm.seat_layout("mixed", "Bayes", s, r), sm.seat_layout("mixed", "Equity", s, r)
            assert [k for j, k in enumerate(a) if j != r] == [k for j, k in enumerate(b) if j != r]


def test_seeds_shared_across_arms_and_rooms_and_salted_by_field():
    f1, f2 = sm.field_seed(sm.MASTER_SEED, "mixed"), sm.field_seed(sm.MASTER_SEED, "tough")
    assert f1 != f2 and sm.field_seed(sm.PILOT_SEED, "mixed") != f1
    seeds = sm.block_seeds(123, 5)
    assert seeds == sm.block_seeds(123, 5) and sm.block_seeds(123, 3) == seeds[:3]  # prefix-stable
    assert all(len(r) == sm.N_SEATS for _, r in seeds)


def test_smoke_run_pairs_arms_and_passes_integrity(cheap):
    out = sm.main(SMOKE)
    an = out["analysis"]
    assert an["integrity"]["ok"], an["integrity"]
    pk = "mixed|Random-CallStation"
    assert set(an["pairs"]) == {pk, "mixed|AlwaysRaise-CallStation"}
    assert an["pairs"][pk]["difference"]["n"] == 2
    assert an["pairs"][pk]["difference"]["hands"] == 2 * 6 * 4
    pos = an["arms"]["mixed|Random"]["position"]
    assert set(pos) == set(sm.POSITIONS) and all(v["hands"] == 2 * 4 for v in pos.values())
    files = {p.name for p in cheap.iterdir()}
    assert any(f.endswith("_minus_CallStation.png") for f in files)
    summary = next(p for p in cheap.iterdir() if p.name.endswith("summary.json"))
    assert "_smoke_" in summary.name and json.loads(summary.read_text())["meta"]["official"] is False


def test_pairing_breaks_are_caught(cheap, no_plots):
    sm.main(SMOKE)
    partial = next(p for p in cheap.iterdir() if p.name.endswith("_partial.jsonl"))
    header = json.loads(partial.read_text(encoding="utf-8").splitlines()[0])["header"]
    done = sm.reb.read_partial(partial, {k: v for k, v in header.items() if k != "started"})
    assert sm.integrity(done, header["n_sessions"], sm.PILOT_SEED, header["blocks"])["ok"]
    key = ("mixed", "CallStation", 1, 3)
    bad = dict(done)
    bad[key] = ({**done[key][0], "seeds_digest": "0" * 16}, 0.0)
    assert not sm.integrity(bad, header["n_sessions"], sm.PILOT_SEED, header["blocks"])["ok"]
    bad = dict(done)
    lay = list(done[key][0]["layout"])
    lay[(3 + 1) % 6], lay[(3 + 2) % 6] = lay[(3 + 2) % 6], lay[(3 + 1) % 6]
    bad[key] = ({**done[key][0], "layout": lay}, 0.0)
    assert not sm.integrity(bad, header["n_sessions"], sm.PILOT_SEED, header["blocks"])["ok"]
    with pytest.raises(ValueError, match="rooms"):
        sm.group_sessions({k: v for k, v in done.items() if k != key}, "mixed", "CallStation", 2)
    chk = lambda d: sm.integrity(d, header["n_sessions"], sm.PILOT_SEED, header["blocks"])  # noqa: E731
    other = ("mixed", "Random", 1, 3)
    bad = dict(done)
    bad[key], bad[other] = done[other], done[key]  # two records swapped: each misfiled
    assert any("filed under" in p for p in chk(bad)["problems"])
    bad = dict(done)
    bad[key] = ({**done[key][0], "cards_digest": "f" * 16}, 0.0)  # e.g. a no-op reseed / different deal
    assert any("dealt cards" in p for p in chk(bad)["problems"])
    assert not sm.integrity(done, header["n_sessions"], sm.PILOT_SEED + 1, header["blocks"])["ok"]  # wrong master seed


def test_rebuild_outputs_from_a_complete_log_touches_nothing_else(cheap, monkeypatch, tmp_path):
    md = tmp_path / "pre.md"
    md.write_text("unchanged", encoding="utf-8")
    monkeypatch.setattr(sm, "MD", md)
    full = sm.main(SMOKE)
    partial = next(p for p in cheap.iterdir() if p.name.endswith("_partial.jsonl"))
    for f in cheap.iterdir():
        if f != partial and f != md:
            f.unlink()
    out = sm.main(["--rebuild-outputs", str(partial)])
    assert _strip(out["analysis"]) == _strip(full["analysis"])
    assert any(p.name.endswith("summary.json") for p in cheap.iterdir())
    assert md.read_text(encoding="utf-8") == "unchanged"
    lines = partial.read_text(encoding="utf-8").splitlines()
    partial.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="incomplete"):
        sm.main(["--rebuild-outputs", str(partial)])


def test_test_bot_equal_to_the_field_scores_exactly_zero_per_block(monkeypatch):
    # 6 identical, per-seat reseeded bots: the 6 rooms are the same table, so the test seat's
    # block total over the rooms is the sum of all nets = 0 (the research note's sanity check).
    monkeypatch.setattr(sm, "FIELDS", {"copies": ("Random",) * 5})
    rooms = [sm.play_room("copies", "Random", 7, 0, r, 12) for r in range(sm.N_SEATS)]
    assert sm.block_totals_bb(rooms) == [0.0] * 12
    assert any(x != 0 for r in rooms for x in r["nets"])


def test_monte_carlo_bots_are_deterministic_given_the_seeds():
    a = sm.play_room("mixed", "PotOdds", 11, 0, 2, 2)
    b = sm.play_room("mixed", "PotOdds", 11, 0, 2, 2)
    assert a == b


# --- Determinism / resume ----------------------------------------------------------------------------


def test_interrupted_then_resumed_run_equals_an_uninterrupted_run(cheap, no_plots):
    full = sm.main(SMOKE)
    for f in cheap.iterdir():
        f.unlink()
    with pytest.raises(sm._Interrupted):
        sm.main(SMOKE, stop_after=7)
    partial = next(p for p in cheap.iterdir() if p.name.endswith("_partial.jsonl"))
    with open(partial, "a", encoding="utf-8") as f:
        f.write('{"key": ["torn')  # a crash mid-line
    with pytest.raises(sm._Interrupted):
        sm.main(SMOKE + ["--resume-partial"], stop_after=5)
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 7 + 5 and all(json.loads(x) for x in lines)
    resumed = sm.main(SMOKE + ["--resume-partial"])
    assert resumed["meta"]["resumed_rooms"] == 12
    assert _strip(resumed["analysis"]) == _strip(full["analysis"])


def test_resume_refuses_a_different_design(cheap, monkeypatch):
    with pytest.raises(sm._Interrupted):
        sm.main(SMOKE, stop_after=2)
    monkeypatch.setattr(sm, "SMOKE_BLOCKS", 5)
    with pytest.raises(SystemExit, match="different run"):
        sm.main(SMOKE + ["--resume-partial"])


# --- Verdict, plan, design ----------------------------------------------------------------------------


def _d(mean, lo, hi, nov=False, p=0.01):
    return {"mean_bb100": mean, "ci95_bb100": [lo, hi], "no_variation": nov, "p_value": p}


def test_verdict_logic():
    assert sm.tier_verdict(_d(60, 10, 110)) == "beats"
    assert sm.tier_verdict(_d(-60, -110, -10)) == "loses"
    assert sm.tier_verdict(_d(20, -5, 45)) == "no detectable edge"
    assert sm.tier_verdict(_d(0, 0, 0, nov=True, p=float("nan"))) == "no detectable edge"
    d = {**_d(20, -5, 45), "verdict": "no detectable edge", "sessions_needed_observed": 400}
    assert "no detectable edge" in sm.overall_verdict(d, True) and "400" in sm.overall_verdict(d, True)
    assert sm.overall_verdict({**d, "verdict": "beats"}, True).endswith("(tier 3 counts in this field)")
    assert sm.overall_verdict({**d, "verdict": "beats"}, False).startswith("INVALID")


def test_required_sessions_and_estimate():
    sd = 150.0
    n = required_units(sm.SD_MARGIN * sd, sm.EDGE_BB100, alpha=0.05, power=0.8)
    assert sm.required_sessions(sd) == max(sm.MIN_SESSIONS, 10 * -(-n // 10))
    assert sm.required_sessions(1.0) == sm.MIN_SESSIONS
    assert sm.sessions_needed(100.0, 0.0) is None
    ms = {f"mixed|{a}": 10.0 for a in sm.ARMS["mixed"]}
    est = sm.estimate_minutes({"mixed": 60}, ms, blocks=250)
    assert est == pytest.approx(len(sm.ARMS["mixed"]) * 60 * 6 * 250 * 10 / 1000 / 60)
    assert sm.estimate_minutes({"mixed": 60}, {}) is None


def test_plan_prints_power(monkeypatch, capsys):
    monkeypatch.setattr(sm, "PILOT_SD", {"mixed|Bayes-Equity": 150.0})
    monkeypatch.setattr(sm, "N_SESSIONS", {"mixed": 140})
    monkeypatch.setattr(sm, "MS_PER_HAND_WALL", {f"mixed|{a}": 10.0 for a in sm.ARMS["mixed"]})
    sm.plan()
    out = capsys.readouterr().out
    assert f"required {sm.required_sessions(150.0)}" in out and "N = 140" in out and "power at 60" in out
    assert "at the pre-pilot 50" in out


def test_design_check(monkeypatch):
    monkeypatch.setattr(sm, "PILOT_SD", {"mixed|Bayes-Equity": 150.0})
    monkeypatch.setattr(sm, "N_SESSIONS", {"mixed": sm.required_sessions(150.0) - 10})
    with pytest.raises(SystemExit, match="design error"):
        sm.design_check()
    monkeypatch.setattr(sm, "N_SESSIONS", {"mixed": sm.required_sessions(150.0) + 3})
    with pytest.raises(SystemExit, match="multiple of 5"):
        sm.design_check()
    monkeypatch.setattr(sm, "N_SESSIONS", {})
    with pytest.raises(SystemExit, match="design not fixed"):
        sm.design_check()


def test_pinned_design_meets_its_power_requirement():
    if not sm.N_SESSIONS:
        pytest.skip("design not fixed yet")
    sm.design_check()


def test_seeds_are_fresh():
    used = {999, 555, 556, 777, 4242, 20261115, 20261116, 20261117, 202611186, 202611188, 202611231, 7, 61,
            27182818, 16180339, 31415926, 17320508, 14142135, 46368, 20261124, 20251123, 987654321}
    assert sm.MASTER_SEED not in used and sm.PILOT_SEED not in used and sm.MASTER_SEED != sm.PILOT_SEED
    assert not 20261116 <= sm.MASTER_SEED <= 20278000


def test_no_cfr_arm_because_no_blueprint_spot_exists_in_this_field():
    # The CFR arm was dropped because CallStation and AlwaysRaise (never fold) sit in every hand.
    assert "CFR" not in sm.ARMS["mixed"] and {"CallStation", "AlwaysRaise"} <= set(sm.FIELDS["mixed"])


def test_exit_street_is_the_test_seats_fold_street():
    rec = lambda seat, street, action: SimpleNamespace(player=seat, street=street, action=action, chips=0)  # noqa: E731
    log = SimpleNamespace(actions=[rec(0, Street.PREFLOP, Call()), rec(1, Street.PREFLOP, Call()),
                                   rec(0, Street.FLOP, Check()), rec(1, Street.TURN, Fold())])
    assert sm.exit_street(log, 1) == "turn" and sm.exit_street(log, 0) == "showdown"


# --- Guards -------------------------------------------------------------------------------------------


def test_official_refusals(cheap, monkeypatch):
    with pytest.raises(SystemExit, match="allow-dirty"):
        sm.main(["--allow-dirty"])
    with pytest.raises(SystemExit, match="22 processes"):
        sm.main(["--smoke", "--processes", "23"])
    monkeypatch.setattr(sm, "N_SESSIONS", {})
    with pytest.raises(SystemExit, match="design not fixed"):
        sm.main([])
    partial = cheap / sm.OFFICIAL_PARTIAL
    (cheap / "2026-01-01_6max_summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exist"):
        sm.official_guard(cheap, partial, resume=False)
    (cheap / "2026-01-01_6max_summary.json").unlink()
    (cheap / "2026-01-01_6max_pilot_seed1_summary.json").write_text("{}")  # pilots do not block
    sm.official_guard(cheap, partial, resume=False)
    partial.write_text("{}\n")
    with pytest.raises(SystemExit, match="resume-partial"):
        sm.official_guard(cheap, partial, resume=False)
    sm.official_guard(cheap, partial, resume=True)


def test_official_refuses_a_dirty_tree(cheap, monkeypatch, tmp_path):
    md = tmp_path / "pre.md"
    md.write_text(f"prereg\n{sm.RESULTS_MARKER}\n", encoding="utf-8")
    monkeypatch.setattr(sm, "MD", md)
    monkeypatch.setattr(sm, "design_check", lambda: None)
    monkeypatch.setattr(sm, "commit_hash", lambda allow_dirty: sm.reb.commit_hash(allow_dirty))
    monkeypatch.setattr(sm.reb, "git_state", lambda: ("abc1234", "?? untracked.txt"))
    with pytest.raises(SystemExit, match="untracked"):
        sm.main(["--processes", "1"])


def test_official_refuses_results_already_below_the_marker(cheap, monkeypatch, tmp_path):
    md = tmp_path / "pre.md"
    md.write_text(f"prereg\n{sm.RESULTS_MARKER}\n\nresults\n", encoding="utf-8")
    monkeypatch.setattr(sm, "MD", md)
    monkeypatch.setattr(sm, "design_check", lambda: None)
    with pytest.raises(ValueError, match="already present"):
        sm.main([])


def test_official_splices_results_below_the_marker_once(cheap, monkeypatch, tmp_path, no_plots):
    md = tmp_path / "pre.md"
    md.write_text(f"prereg\n\n{sm.RESULTS_MARKER}\n", encoding="utf-8")
    monkeypatch.setattr(sm, "MD", md)
    monkeypatch.setattr(sm, "design_check", lambda: None)
    monkeypatch.setattr(sm, "N_SESSIONS", {"mixed": 2})
    monkeypatch.setattr(sm, "BLOCKS_PER_SESSION", 3)
    monkeypatch.setattr(sm, "MASTER_SEED", 424242424242)  # tests never play the official seed
    out = sm.main(["--processes", "1"])
    assert out["meta"]["official"] and out["meta"]["master_seed"] == 424242424242
    head, _, tail = md.read_text(encoding="utf-8").partition(sm.RESULTS_MARKER)
    assert head.startswith("prereg") and "**Verdict:" in tail
    with pytest.raises(SystemExit, match="already exist|exists"):
        sm.main(["--processes", "1"])


def test_preregistration_has_the_marker_and_no_results_yet():
    text = sm.MD.read_text(encoding="utf-8")
    head, sep, tail = text.partition(sm.RESULTS_MARKER)
    assert sep and not tail.strip()
    assert str(sm.MASTER_SEED) in head and f"N = {sm.N_SESSIONS['mixed']}" in head


def test_pinned_numbers_after_the_owner_s_post_pilot_edge_change():
    assert sm.EDGE_BB100 == 60.0 and sm.EDGE_BB100_PRE_PILOT == 50.0
    assert sm.required_sessions(sm.PILOT_SD["mixed|Bayes-Equity"]) == 110 == sm.N_SESSIONS["mixed"]
    sd = sm.SD_MARGIN * sm.PILOT_SD["mixed|Bayes-Equity"]
    assert sm.rp6.power_at(60, sd, 110) == pytest.approx(0.827, abs=0.005)
    assert sm.rp6.power_at(50, sd, 110) == pytest.approx(0.677, abs=0.005)
    assert sm.rp6.mde(sd, 110) == pytest.approx(57.9, abs=0.1)
    assert sm.estimate_minutes(sm.N_SESSIONS) == pytest.approx(173.1, abs=0.2)
