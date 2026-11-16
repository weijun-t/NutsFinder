r"""Quick, NOT OFFICIAL sanity match: does a (briefly trained) CFRBot play sensibly?

Run:
    .venv\Scripts\python examples\cfr_sanity_match.py --model models\bench\latest.npz --pairs 300

Plays a duplicate match (``nutsfinder.eval.match.duplicate_match``) of CFRBot
against each opponent and prints bb/100 with a 95% CI. This is a smoke test
of the pipeline (model file -> bot -> legal, non-absurd play), not a tier
evaluation: the model is a few minutes of training, the sample is small,
the seed is not pre-registered, and the opponents are not BayesBot. The
tier test is pre-registered separately (docs/phase6_research.md section 6).
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nutsfinder.bots import CallStationBot, CFRBot, EquityBot  # noqa: E402
from nutsfinder.bots.cfr_bot import preload_turn_tables  # noqa: E402
from nutsfinder.eval.match import duplicate_match  # noqa: E402

OPPONENTS = {"CallStationBot": CallStationBot, "EquityBot": EquityBot}


def one(args: tuple[str, str, int, int, str | None]) -> str:
    model, opponent, pairs, seed, turns = args
    if turns:
        preload_turn_tables(turns)
    start = time.perf_counter()
    res = duplicate_match(functools.partial(CFRBot, model), OPPONENTS[opponent], pairs, master_seed=seed)
    lo, hi = res.ci_bb100
    return (f"CFRBot vs {opponent}: {res.mean_bb100:+.1f} bb/100, 95% CI [{lo:+.1f}, {hi:+.1f}], "
            f"{pairs} pairs ({2 * pairs} hands), master seed {seed}, {time.perf_counter() - start:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=314)
    ap.add_argument("--turn-tables", default=str(ROOT / "models" / "turn_ehs.npz"))
    args = ap.parse_args()
    turns = args.turn_tables if Path(args.turn_tables).exists() else None
    print(f"NOT OFFICIAL sanity match; model {args.model}")
    tasks = [(str(args.model), name, args.pairs, args.seed, turns) for name in OPPONENTS]
    with get_context("spawn").Pool(len(tasks)) as pool:
        for line in pool.map(one, tasks):
            print(line)


if __name__ == "__main__":
    main()
