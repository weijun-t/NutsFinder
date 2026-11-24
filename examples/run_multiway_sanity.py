r"""Sanity check of the multiway bots: speed, and a short 6-max tournament against baseline fields.

Run:
    .venv\Scripts\python examples\run_multiway_sanity.py [hands] [processes] [bots, e.g. Equity,Bayes]

DESCRIPTIVE ONLY. This is not a tier test: no pre-registration, a single
master seed, short sessions, and fields of baseline bots only.

Tournament design (rotation duplicate, docs/phase7_multiway_research.md section 3)
---------------------------------------------------------------------------------
- 6 seats, 10,000 chips (100bb) reset every hand, blinds 50/100.
- A *block* is one deal seed played in 6 rooms. In room r the test bot sits
  in seat r and the field fills seats r+1, ..., r+5 in its fixed order, so
  over a block the test bot holds every seat's cards once for the same
  deal. The button is ``block % 6`` in every room.
- Each room is one persistent session (the same bot objects for all
  blocks, so MultiBayesBot keeps learning; the others do not learn).
- ``hands`` (default 2,000) is rounded up to whole blocks.
- bb/100 = 100 × mean net per hand / 100 chips. 95% CI = ±1.96 SE, with
  the SE taken over block totals (each block sums the test bot's 6 hands).
  Blocks of one room are not independent for a learning bot, and the
  rooms share cards, so treat the CI as rough.

Paired comparison: every bot plays the same blocks with the same field
seeds in the same rooms, so ``Bayes - Equity`` is reported per block
(the difference of the two block totals), with its own CI: shared cards
and field make it much tighter than either bot's own CI.

Fields: ``mixed`` = [Random, CallStation, AlwaysRaise, CallStation, Random]
and ``stations`` = 5 × CallStation. MultiCFRBot needs
``models/holdem_official/latest.npz`` and is skipped without it.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nutsfinder.bots.multiway import MultiBayesBot, MultiCFRBot, MultiEquityBot, MultiPotOddsBot
from nutsfinder.bots.multiway.equity import warm_up
from nutsfinder.engine.multiway import (
    MultiAlwaysRaiseBot,
    MultiBot,
    MultiCallStationBot,
    MultiRandomBot,
    play_hand_multi,
)

MODEL = Path(__file__).resolve().parent.parent / "models" / "holdem_official" / "latest.npz"
MASTER_SEED = 20261124
N = 6
BB = 100
FIELDS = {
    "mixed": ("random", "station", "raise", "station", "random"),
    "stations": ("station",) * 5,
}
BOTS = ("PotOdds", "Equity", "Bayes", "CFR")


class Timed(MultiBot):
    """Times every decision of the wrapped bot."""

    def __init__(self, inner: MultiBot) -> None:
        self.inner = inner
        self.times: list[float] = []

    def act(self, view, hole_cards):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        a = self.inner.act(view, hole_cards)
        self.times.append(time.perf_counter() - t)
        return a

    def end_hand(self, summary):  # type: ignore[no-untyped-def]
        self.inner.end_hand(summary)

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)


def make_test_bot(name: str, seed: int) -> MultiBot:
    if name == "PotOdds":
        return MultiPotOddsBot(seed)
    if name == "Equity":
        return MultiEquityBot(seed)
    if name == "Bayes":
        return MultiBayesBot(seed)
    if name == "CFR":
        return MultiCFRBot(MODEL, seed=seed)
    raise ValueError(name)


def make_field_bot(kind: str, seed: int) -> MultiBot:
    return {"random": lambda: MultiRandomBot(seed), "station": MultiCallStationBot,
            "raise": MultiAlwaysRaiseBot}[kind]()


def block_seeds(n_blocks: int) -> list[int]:
    rng = random.Random(MASTER_SEED)
    return [rng.getrandbits(64) for _ in range(n_blocks)]


def run_room(args: tuple[str, str, int, int]) -> tuple[str, str, int, list[int], list[float], dict[str, int]]:
    """One room: the test bot in seat ``room``; returns its net per block, decision times, CFR modes."""
    name, field, room, n_blocks = args
    warm_up()
    test = Timed(make_test_bot(name, seed=MASTER_SEED + 17 * room))
    bots: list[MultiBot] = [None] * N  # type: ignore[list-item]
    bots[room] = test
    for i, kind in enumerate(FIELDS[field]):
        bots[(room + 1 + i) % N] = make_field_bot(kind, seed=MASTER_SEED + 1000 * room + i)
    nets = []
    for b, seed in enumerate(block_seeds(n_blocks)):
        log = play_hand_multi(bots, seed, button=b % N)
        nets.append(log.result.net[room])
    modes = dict(test.inner.mode_counts) if isinstance(test.inner, MultiCFRBot) else {}
    return name, field, room, nets, test.times, modes


def main() -> None:
    hands = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    n_blocks = math.ceil(hands / N)
    wanted = sys.argv[3].split(",") if len(sys.argv) > 3 else list(BOTS)
    names = [b for b in BOTS if b in wanted and (b != "CFR" or MODEL.is_file())]
    jobs = [(name, field, room, n_blocks) for name in names for field in FIELDS for room in range(N)]
    start = time.perf_counter()
    with ProcessPoolExecutor(processes) as pool:
        results = list(pool.map(run_room, jobs))
    print(f"{n_blocks} blocks x {N} rooms = {n_blocks * N} hands per (bot, field); "
          f"master seed {MASTER_SEED}; {time.perf_counter() - start:.0f} s wall")
    print(f"{'bot':8s} {'field':9s} {'bb/100':>9s} {'95% CI':>20s} {'ms/decision':>12s} {'p95 ms':>7s}  modes")
    for name in names:
        for field in FIELDS:
            rows = [r for r in results if r[0] == name and r[1] == field]
            block_totals = [sum(r[3][b] for r in rows) for b in range(n_blocks)]
            times = sorted(t for r in rows for t in r[4])
            modes: Counter[str] = Counter()
            for r in rows:
                modes.update(r[5])
            mean_hand = sum(block_totals) / (n_blocks * N)
            sd = math.sqrt(sum((x - sum(block_totals) / n_blocks) ** 2 for x in block_totals) / (n_blocks - 1))
            se_hand = sd / math.sqrt(n_blocks) / N
            bb100, half = 100 * mean_hand / BB, 1.96 * 100 * se_hand / BB
            ms = 1000 * sum(times) / len(times)
            p95 = 1000 * times[int(0.95 * (len(times) - 1))]
            print(f"{name:8s} {field:9s} {bb100:+9.1f} [{bb100 - half:+8.1f}, {bb100 + half:+8.1f}] "
                  f"{ms:12.1f} {p95:7.1f}  {dict(modes) if modes else ''}")
    if {"Bayes", "Equity"} <= set(names):
        for field in FIELDS:
            tot = {}
            for name in ("Bayes", "Equity"):
                rows = [r for r in results if r[0] == name and r[1] == field]
                tot[name] = [sum(r[3][b] for r in rows) for b in range(n_blocks)]
            d = [x - y for x, y in zip(tot["Bayes"], tot["Equity"])]
            mean = sum(d) / n_blocks
            sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n_blocks - 1))
            scale = 100 / (N * BB)
            half = 1.96 * sd / math.sqrt(n_blocks) * scale
            print(f"paired Bayes - Equity, {field}: {mean * scale:+.1f} +/- {half:.1f} bb/100 (95% CI, over blocks)")


if __name__ == "__main__":
    main()
