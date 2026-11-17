"""Tests for examples/run_lbr.py: once-only guard, crash-safe resume, official-mode refusals, verdict."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, RandomBot
from nutsfinder.eval.exact_equity import load_preflop_table

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_lbr_cli", ROOT / "examples" / "run_lbr.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)

TIMING_KEYS = {"worker_seconds_per_hand", "query_seconds"}


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in TIMING_KEYS}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


@pytest.fixture
def cheap(monkeypatch, tmp_path):
    """run_lbr with cheap opponents, results in tmp_path, no model check (in-process play)."""
    monkeypatch.setattr(rl, "RESULTS", tmp_path)
    monkeypatch.setattr(rl, "FACTORIES", {"CFRBot": CallStationBot, "BayesBot": AlwaysRaiseBot,
                                          "EquityBot": RandomBot})
    monkeypatch.setattr(rl, "ORACLE_OPTIONS", {"CFRBot": {}, "BayesBot": {}, "EquityBot": {}})
    monkeypatch.setattr(rl, "check_model", lambda: {"sha256": "x" * 64, "description": "stub"})
    monkeypatch.setattr(rl, "check_preflop_table", lambda official, processes: load_preflop_table())
    monkeypatch.setattr(rl, "commit_hash", lambda allow_dirty: "test")
    return tmp_path


PILOT = ["--pilot", "--seed", "5", "--sessions", "3", "--pairs", "2", "--processes", "1"]


def test_interrupted_then_resumed_run_equals_an_uninterrupted_run(cheap, tmp_path, monkeypatch):
    full = rl.main(PILOT)
    for f in tmp_path.iterdir():
        f.unlink()
    with pytest.raises(rl._Interrupted):
        rl.main(PILOT, stop_after=7)
    partial = next(tmp_path.glob("*_partial.jsonl"))
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 7 and "header" in json.loads(lines[0])
    with open(partial, "a", encoding="utf-8") as f:  # a crash in the middle of a write
        f.write('{"rule": "ev", "opponent": "Bay')
    resumed = rl.main(PILOT + ["--resume-partial"])
    assert resumed["meta"]["resumed_sessions"] == 7
    assert _strip(resumed["results"]) == _strip(full["results"])
    assert resumed["results"]["verdict"] == full["results"]["verdict"]


def test_resume_refuses_a_different_design(cheap, tmp_path):
    with pytest.raises(rl._Interrupted):
        rl.main(PILOT, stop_after=2)
    with pytest.raises(SystemExit, match="different run"):
        rl.main(["--pilot", "--seed", "5", "--sessions", "3", "--pairs", "3", "--processes", "1", "--resume-partial"])


def test_official_guard_is_once_only_but_allows_resuming(tmp_path):
    partial = tmp_path / rl.OFFICIAL_PARTIAL
    rl.official_guard(tmp_path, partial, resume=False)  # nothing there: fine
    (tmp_path / "phase6_lbr_pilot_seed999_ev-call_2026-11-17.json").write_text("{}")
    rl.official_guard(tmp_path, partial, resume=False)  # pilots do not count
    with pytest.raises(SystemExit, match="not found"):
        rl.official_guard(tmp_path, partial, resume=True)
    partial.write_text('{"header": {}}\n')
    with pytest.raises(SystemExit, match="resume-partial"):
        rl.official_guard(tmp_path, partial, resume=False)
    rl.official_guard(tmp_path, partial, resume=True)  # the same partial run may be finished
    for name in ("phase6_lbr_2026-11-18.json", "phase6_lbr_2026-11-18.md"):
        (tmp_path / name).write_text("{}")
        with pytest.raises(SystemExit, match="happens once"):
            rl.official_guard(tmp_path, partial, resume=True)  # never a second complete run
        (tmp_path / name).unlink()


def test_official_mode_refuses_allow_dirty():
    with pytest.raises(SystemExit, match="allow-dirty"):
        rl.main(["--allow-dirty"])


def test_official_preflop_table_must_exist_with_the_pinned_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(rl, "PREFLOP_TABLE_FILE", tmp_path / "missing.npz")
    with pytest.raises(SystemExit, match="never builds"):
        rl.check_preflop_table(official=True, processes=1)
    other = tmp_path / "t.npz"
    other.write_bytes(b"not the table")
    monkeypatch.setattr(rl, "PREFLOP_TABLE_FILE", other)
    with pytest.raises(SystemExit, match="expected"):
        rl.check_preflop_table(official=True, processes=1)


def test_pinned_preflop_table_hash_matches_the_local_file():
    if not rl.PREFLOP_TABLE_FILE.exists():
        pytest.skip("no local preflop table")
    assert rl.file_sha256(rl.PREFLOP_TABLE_FILE) == rl.PREFLOP_TABLE_SHA256


def _res(bayes, equity):
    def d(mean, lo, hi):
        return {"sessions": {"mean_mbb_g": mean, "ci_mbb_g": [lo, hi], "p_value": 0.01}}
    return {"differences": {"BayesBot - CFRBot": d(*bayes), "EquityBot - CFRBot": d(*equity)}}


def test_verdict_is_the_preregistered_rule_and_flags_rule_dependence():
    v = rl.verdict({"ev": _res((900, 100, 1700), (800, 50, 1550)), "call": _res((300, -500, 1100), (800, 50, 1550))},
                   "ev")
    assert v["hard_to_exploit"] is True and v["by_rule"]["call"]["hard_to_exploit"] is False
    assert v["differs_between_rules"] and "only under the chosen rule" in v["sentence"]
    assert v["labels"] == {"ev": "primary", "call": "sensitivity"}
    v = rl.verdict({"ev": _res((900, 100, 1700), (800, -50, 1650)), "call": _res((900, 100, 1700), (800, 50, 1550))},
                   "ev")
    assert v["hard_to_exploit"] is False and v["differs_between_rules"]
    assert "pre-registered rule decides" in v["sentence"]
    v = rl.verdict({"ev": _res((-900, -1700, -100), (800, 50, 1550))}, "ev")  # negative and significant: no
    assert v["hard_to_exploit"] is False and not v["differs_between_rules"]


def test_memory_formatting_never_crashes():
    assert rl._fmt_mb(None) == "n/a" and rl._fmt_mb(float("nan")) == "n/a" and rl._fmt_mb(12.4) == "12 MB"
