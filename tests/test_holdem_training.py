"""Tests for the parallel hold'em trainer (``nutsfinder.abstraction.training``) and blueprints.

- Determinism: a run is a pure function of (seed, workers, batch, config,
  iterations); the in-process runner equals a real process pool.
- Resume: checkpoint after k rounds + resume == the uninterrupted run, bit for bit.
- Delta summing: with 1 worker the parallel scheme is serial MCCFR reseeded
  every batch (reference computed here with a plain MCCFRSolver).
- Linear weighting by global iteration; checkpoint metadata; the proxies.

Speed: the fast tests use Monte Carlo E[HS] with a handful of samples
(``FAST``), so no turn tables are needed and a few hundred iterations take
seconds. The bucket *quality* is irrelevant here; the tree (57,304 info
sets) is the real one.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from nutsfinder.abstraction.blueprint import (
    Blueprint, config_from_dict, config_to_dict, known_infosets, load_tables, slot_offsets, tree_for,
)
from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.training import (
    TrainOptions, Trainer, convergence_proxies, init_worker, make_game, round_seed, weight_sum,
)
from nutsfinder.gametheory.mccfr import MCCFRSolver

FAST = AbstractionConfig(ehs_method="mc", mc_samples=4)


def _opts(**kw) -> TrainOptions:
    return TrainOptions(**{"seed": 3, "workers": 2, "batch": 20, "config": FAST, **kw})


def _run(opts: TrainOptions, rounds: int, pool=None) -> Trainer:
    t = Trainer(opts, pool=pool)
    for _ in range(rounds):
        t.run_round()
    return t


def test_round_seeds_are_hashed_and_distinct():
    seeds = {round_seed(s, r, w) for s in (0, 1) for r in range(20) for w in range(22)}
    assert len(seeds) == 2 * 20 * 22
    assert round_seed(7, 3, 2) == round_seed(7, 3, 2) and 0 <= round_seed(7, 3, 2) < 2**64


def test_weight_sum_matches_solver_weights():
    s = MCCFRSolver(make_game(FAST, None), linear=True, linear_until=5)
    assert weight_sum(9, True, 5) == sum(s.weight(t) for t in range(1, 10))
    assert weight_sum(9, True, None) == 45 and weight_sum(9, False, None) == 9


def test_layout_is_the_preregistered_solver_layout():
    tree = tree_for(DEFAULT_CONFIG)
    offset, n_actions = slot_offsets(tree)
    s = MCCFRSolver(make_game(FAST, None), known_infosets=known_infosets(tree))
    assert len(s.keys) == tree.n_infosets == 57_304
    assert s.keys == list(range(tree.n_infosets))
    assert np.array_equal(s.offsets, offset) and int(n_actions.sum()) == len(s.regrets)


def test_run_is_deterministic_and_seed_dependent():
    a, b = _run(_opts(), 3), _run(_opts(), 3)
    assert a.iteration == 3 * 2 * 20 and a.round == 3
    assert np.array_equal(a.regrets, b.regrets) and np.array_equal(a.strategy_sum, b.strategy_sum)
    c = _run(_opts(seed=4), 3)
    assert not np.array_equal(a.regrets, c.regrets)


def test_resume_equals_uninterrupted_run(tmp_path):
    full = _run(_opts(), 4)
    first = _run(_opts(), 2)
    first.checkpoint_stats()
    path = tmp_path / "ckpt.npz"
    first.save(path, commit="test")
    resumed = Trainer.resume(path)
    assert resumed.opts == _opts() and resumed.iteration == 80 and resumed.round == 2
    for _ in range(2):
        resumed.run_round()
    assert np.array_equal(full.regrets, resumed.regrets)
    assert np.array_equal(full.strategy_sum, resumed.strategy_sum)
    assert resumed.iteration == full.iteration and resumed.nodes == full.nodes


def test_resume_refuses_different_options(tmp_path):
    t = _run(_opts(), 1)
    t.save(tmp_path / "c.npz")
    with pytest.raises(ValueError, match="options differ"):
        Trainer.resume(tmp_path / "c.npz", opts=_opts(batch=21))


def test_one_worker_equals_serial_mccfr():
    """With W = 1 the master's delta sums reproduce a serial solver reseeded each round."""
    opts = _opts(workers=1, batch=30)
    par = _run(opts, 3)
    game = make_game(FAST, None)
    ref = MCCFRSolver(game, linear=True, known_infosets=known_infosets(game.tree))
    for r in range(3):
        ref.reseed(round_seed(opts.seed, r, 0))
        ref.run(opts.batch)
    assert ref.iteration == par.iteration
    # Deltas are taken as (snapshot + increments) - snapshot, so allow float rounding only.
    np.testing.assert_allclose(par.regrets, ref.regrets, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(par.strategy_sum, ref.strategy_sum, rtol=1e-12, atol=1e-9)
    assert par.nodes == ref.nodes  # the very same samples were drawn


def test_iterations_get_their_global_linear_weight():
    # Every iteration t contributes weight t, so with W workers the strategy mass added in a
    # round is sum(t) over that round's global iterations times (sum of sigma = 1) per visit.
    opts = _opts(workers=3, batch=4)
    t = _run(opts, 2)
    assert t.iteration == 24 and t.weight_sum() == 24 * 25 / 2


def test_process_pool_equals_in_process_run():
    opts = _opts(batch=15)
    inline = _run(opts, 2)
    ctx = mp.get_context("spawn")
    with ctx.Pool(2, initializer=init_worker, initargs=(opts, None)) as pool:
        pooled = _run(opts, 2, pool=pool)
    assert np.array_equal(inline.regrets, pooled.regrets)
    assert np.array_equal(inline.strategy_sum, pooled.strategy_sum)


def test_checkpoint_metadata_and_blueprint(tmp_path):
    t = _run(_opts(), 2)
    row = t.checkpoint_stats()
    assert row["iteration"] == 80 and row["infosets_touched"] > 0 and row["regret_proxy_mbb"] > 0
    t.save(tmp_path / "c.npz", commit="abc123")
    regrets, ssum, meta = load_tables(tmp_path / "c.npz")
    assert meta["seed"] == 3 and meta["commit"] == "abc123" and meta["iteration"] == 80
    assert meta["weight_sum"] == 80 * 81 / 2 and meta["n_infosets"] == 57_304
    assert config_from_dict(meta["config"]) == FAST
    bp = Blueprint.load(tmp_path / "c.npz")
    assert bp.iteration == 80 and bp.config == FAST
    sums = np.add.reduceat(bp.sigma, bp.offset)
    np.testing.assert_allclose(sums, 1.0)
    assert bp.reached.any() and not bp.reached.all()
    k = int(np.flatnonzero(~bp.reached)[0])
    node, bucket = bp.tree.decode_infoset(k)
    assert np.allclose(bp.strategy(node, bucket), 1.0 / bp.tree.n_children[node])


def test_config_round_trip_and_uniform_blueprint():
    assert config_from_dict(config_to_dict(DEFAULT_CONFIG)) == DEFAULT_CONFIG
    bp = Blueprint.uniform()
    assert bp.describe() == "untrained (uniform)"
    assert np.allclose(bp.strategy(0, 0), 0.25)
    with pytest.raises(ValueError):
        Blueprint.from_tables(DEFAULT_CONFIG, None, np.zeros(5))


def test_convergence_proxies():
    tree = tree_for(DEFAULT_CONFIG)
    offset, n_actions = slot_offsets(tree)
    owner = np.array([p for p, _ in known_infosets(tree).values()])
    zeros = np.zeros(int(n_actions.sum()))
    out, sigma = convergence_proxies(zeros, zeros, None, offset, n_actions, owner, 1.0, 100)
    assert out["regret_proxy_mbb"] == 0 and out["infosets_reached"] == 0
    s = zeros.copy()
    s[offset[0]] = 1.0  # info set 0 plays its first action
    out2, sigma2 = convergence_proxies(zeros, s, sigma, offset, n_actions, owner, 1.0, 100)
    assert out2["strategy_tv_weighted"] == pytest.approx(0.75)  # uniform over 4 -> pure: TV 3/4
    r = zeros.copy()
    r[offset[0] + 1] = 200.0  # one positive regret of 200 chips for player 0 = 2bb; averaged over 2 players
    out3, _ = convergence_proxies(r, s, sigma2, offset, n_actions, owner, 1.0, 100)
    assert out3["regret_proxy_mbb"] == pytest.approx(1000.0) and out3["strategy_tv_weighted"] == 0.0


def test_checkpoint_fingerprint_guards_tree_and_bucket_changes(tmp_path):
    """A checkpoint records which tree and bucket edges its slots mean; a mismatch is refused."""
    from nutsfinder.abstraction.blueprint import abstraction_fingerprint, save_tables

    t = _run(_opts(), 1)
    t.save(tmp_path / "c.npz", commit="test")
    regrets, ssum, meta = load_tables(tmp_path / "c.npz")
    assert meta["abstraction"] == abstraction_fingerprint(FAST)
    # A different tree gives a different fingerprint; bucket edges are part of it too.
    assert abstraction_fingerprint(AbstractionConfig(postflop_fractions=(0.5, 1.0))) != meta["abstraction"]
    Blueprint.load(tmp_path / "c.npz")  # matching: loads fine
    Trainer.resume(tmp_path / "c.npz")

    # Same config and slot count, but written by "different code": refused.
    save_tables(tmp_path / "bad.npz", regrets, ssum, {**meta, "abstraction": "0" * 64})
    with pytest.raises(ValueError, match="different abstraction"):
        Blueprint.load(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="different abstraction"):
        Trainer.resume(tmp_path / "bad.npz")

    # Written before fingerprints existed: loads, with a warning.
    old = {k: v for k, v in meta.items() if k != "abstraction"}
    save_tables(tmp_path / "old.npz", regrets, ssum, old)
    with pytest.warns(UserWarning, match="no abstraction fingerprint"):
        Blueprint.load(tmp_path / "old.npz")


# --- examples/train_holdem.py: refuses a dirty work tree ------------------------------------


def _train_cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "train_holdem.py"
    spec = importlib.util.spec_from_file_location("train_holdem_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_cli_exits_on_a_dirty_tree(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    cli = _train_cli()
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (repo / ".gitignore").write_text("models/\n")
    (repo / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    (repo / "models").mkdir()
    (repo / "models" / "latest.npz").write_bytes(b"ignored")  # gitignored files do not count
    assert cli.commit_hash(root=repo) == head
    (repo / "a.py").write_text("x = 2\n")  # a modified tracked file
    with pytest.raises(SystemExit, match="uncommitted changes"):
        cli.commit_hash(root=repo)
    assert cli.commit_hash(allow_dirty=True, root=repo) == head + " + uncommitted changes"
    git("checkout", "--", "a.py")
    (repo / "new.py").write_text("y = 1\n")  # an untracked (not ignored) file is dirty too
    with pytest.raises(SystemExit):
        cli.commit_hash(root=repo)
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(SystemExit, match="cannot read the git state"):
        cli.commit_hash(root=not_a_repo)
    assert cli.commit_hash(allow_dirty=True, root=not_a_repo) == "unknown"
