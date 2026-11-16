r"""Phase 6, step 7: train the hold'em blueprint with parallel ES-MCCFR.

Run (the step-7 recommendation: 1,200 rounds of 22 x 500 = 13.2M iterations,
a checkpoint every 50 rounds = 550k iterations, every checkpoint kept):
    .venv\Scripts\python examples\train_holdem.py --out models\holdem_official --seed 7 --workers 22 ^
        --batch 500 --iterations 13200000 --checkpoint-every 50 --keep 1000

Resume or extend (options are read from the checkpoint; only the stopping rule may change,
and a resumed run is bit-identical to an uninterrupted one):
    .venv\Scripts\python examples\train_holdem.py --resume models\holdem_official\latest.npz --iterations 20000000

Short benchmark (stop after N minutes, checkpoint at the end):
    .venv\Scripts\python examples\train_holdem.py --out models\bench --seed 2026 --max-minutes 5 --checkpoint-every 10

Measured (step 7, 5-minute run, 22 workers x 500): ~1,530 iterations/s sustained
(~72 it/s per worker, against ~700 for one process on an idle machine:
hyper-threading plus thermal throttling), 60 -> 120 nodes per iteration as the
strategy got more aggressive, ~7 s per round of which ~3% is master overhead.

How it works: :mod:`nutsfinder.abstraction.training` (rounds of W workers x
B iterations against a frozen regret snapshot, deltas summed in worker
order, seeds hashed from (seed, round, worker), linear weights by global
iteration; the module doc explains the stale-snapshot approximation and the
convergence proxies). This script adds the process pool, the turn tables,
checkpoints and the log.

Outputs (``--out``, default ``models/holdem``; ``models/`` is gitignored):

- ``ckpt_<iteration>.npz``: checkpoints (see :mod:`nutsfinder.abstraction.blueprint`).
  The newest ``--keep`` are kept, plus every ``--milestone-every``-th.
- ``latest.npz``: a copy of the newest checkpoint (what CFRBot loads).
- ``train_log.jsonl``: one JSON line per checkpoint (speed, info sets
  touched, convergence proxies, commit).

Turn tables: every worker preloads all 16,432 exact turn E[HS] tables from
``--turn-tables`` (default ``models/turn_ehs.npz``, ~14 MB); without them
MCCFR runs ~25x slower while it fills them lazily. If the file is missing it
is generated first (``examples/make_bucket_edges.py``'s turn step, ~1 min on
22 cores).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from nutsfinder.abstraction.training import (  # noqa: E402
    TrainOptions, Trainer, init_worker, rounds_for,
)

DEFAULT_OUT = ROOT / "models" / "holdem"
DEFAULT_TURNS = ROOT / "models" / "turn_ehs.npz"


def commit_hash() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + (" + uncommitted changes" if dirty else "")


def ensure_turn_tables(path: Path, workers: int) -> None:
    if path.exists():
        return
    from make_bucket_edges import compute_turn_tables
    print(f"{path} missing: generating every turn table with {workers} workers ...")
    compute_turn_tables(workers).save_turns(path)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def prune_checkpoints(out: Path, keep: int, milestone_every: int, checkpoint_every: int, per_round: int) -> None:
    """Delete old ckpt files except the newest ``keep`` and the milestones."""
    files = sorted(out.glob("ckpt_*.npz"))
    milestone = milestone_every * checkpoint_every * per_round
    for f in files[:-keep] if keep > 0 else files:
        it = int(f.stem.split("_")[1])
        if milestone and it % milestone == 0:
            continue
        f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default models/holdem)")
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--batch", type=int, default=500, help="iterations per worker per round")
    ap.add_argument("--no-linear", action="store_true", help="plain (unweighted) ES-MCCFR")
    ap.add_argument("--linear-until", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None, help="stop at this total (rounded up to whole rounds)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop after the round that passes this")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="rounds between checkpoints")
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--milestone-every", type=int, default=10, help="also keep every n-th checkpoint")
    ap.add_argument("--turn-tables", type=Path, default=DEFAULT_TURNS)
    args = ap.parse_args()
    if args.iterations is None and args.max_minutes is None:
        ap.error("give --iterations and/or --max-minutes")

    turns = args.turn_tables.resolve()
    if args.resume is not None:
        out = args.out or args.resume.resolve().parent
        opts = None
    else:
        out = args.out or DEFAULT_OUT
        opts = TrainOptions(seed=args.seed, workers=args.workers, batch=args.batch, linear=not args.no_linear,
                            linear_until=args.linear_until)
    out.mkdir(parents=True, exist_ok=True)
    ensure_turn_tables(turns, args.workers if opts is None else opts.workers)
    commit = commit_hash()

    if args.resume is not None:
        trainer = Trainer.resume(args.resume, turns)
        opts = trainer.opts
        print(f"resumed {args.resume} at iteration {trainer.iteration:,} (round {trainer.round})")
    else:
        if any(out.glob("ckpt_*.npz")):
            sys.exit(f"{out} already has checkpoints: use --resume or another --out")
        trainer = Trainer(opts, turns)
    assert opts is not None
    run_file = "run.json" if args.resume is None else f"run_resume_{trainer.iteration:012d}.json"
    (out / run_file).write_text(json.dumps({**opts.as_meta(), "commit": commit, "turn_tables": str(turns),
                                              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                              "resumed_from": str(args.resume) if args.resume else None},
                                             indent=2))
    target_rounds = None if args.iterations is None else rounds_for(args.iterations, opts)
    print(f"seed {opts.seed}, {opts.workers} workers x {opts.batch} iterations/round "
          f"({opts.per_round:,}/round), linear={opts.linear}, commit {commit}")

    ctx = mp.get_context("spawn")
    start = time.perf_counter()
    with ctx.Pool(opts.workers, initializer=init_worker, initargs=(opts, str(turns))) as pool:
        trainer.pool = pool
        print(f"pool started in {time.perf_counter() - start:.1f} s (workers load the tables during round 1)")
        start = time.perf_counter()
        since = 0
        while True:
            if target_rounds is not None and trainer.round >= target_rounds:
                break
            if args.max_minutes is not None and time.perf_counter() - start >= args.max_minutes * 60:
                break
            trainer.run_round()
            since += 1
            done = target_rounds is not None and trainer.round >= target_rounds
            timed_out = args.max_minutes is not None and time.perf_counter() - start >= args.max_minutes * 60
            if since >= args.checkpoint_every or done or timed_out:
                since = 0
                row = {**trainer.checkpoint_stats(), "commit": commit}
                path = out / f"ckpt_{trainer.iteration:012d}.npz"
                trainer.save(path, commit)
                shutil.copyfile(path, out / "latest.npz.tmp")
                os.replace(out / "latest.npz.tmp", out / "latest.npz")
                with open(out / "train_log.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                prune_checkpoints(out, args.keep, args.milestone_every, args.checkpoint_every, opts.per_round)
                print(json.dumps(row))
    print(f"done: {trainer.iteration:,} iterations, {trainer.wall / 60:.1f} min in rounds; latest -> {out / 'latest.npz'}")


if __name__ == "__main__":
    main()
