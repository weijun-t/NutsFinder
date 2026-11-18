r"""Phase 7, part B: variance reduction (all-in luck adjustment, chance control variate) on fresh matches.

Run:
    .venv\Scripts\python examples\run_phase7_variance.py --smoke   # 2 sessions per pairing, smoke seed, nothing saved
    .venv\Scripts\python examples\run_phase7_variance.py           # the fixed design below, saves results
    .venv\Scripts\python examples\run_phase7_variance.py --power-check   # the extra unbiasedness check (below)
    .venv\Scripts\python examples\run_phase7_variance.py --posthoc       # post-hoc ratios from the saved JSONs

Why new matches: the official Phase 5/6 JSONs keep only per-hand nets, not the
cards, and the adjustments need both hole cards and the board of every hand.
So the two pairings are replayed on a FRESH master seed with the hand logs
kept (in the workers; only per-hand numbers come back to the parent).

Fixed design (decided before the first hand; no early stop, no re-runs)
-----------------------------------------------------------------------
- Pairings: CFRBot vs BayesBot (the Phase 6 primary) and BayesBot vs EquityBot.
- 40 sessions per pairing, 250 duplicate pairs (500 hands) each, fresh bots
  per session: exactly the Phase 6 session design (``play_session``), same bot
  factories and pinned blueprint as ``examples/run_phase6.py`` (imported).
- Master seed 31,415,926 (never used before in this repository; Phase 5/6 seeds
  were 999, 555, 556, 777, 4242, 20261115-20261117, 202611186, 202611188, 7, 61
  and 20261116 + k). Smoke seed 14,142,135 (smoke output is not saved).
- Estimators on the same hands: raw, all-in adjusted, chance control variate
  (see ``nutsfinder/eval/variance_reduction.py``). Unit of analysis: the
  session (duplicate total of 500 hands); also reported for a single room
  (250 hands, no duplicate), i.e. what the adjustment does without duplicate.
- Checks: paired difference raw - adjusted per session (unbiasedness predicts
  0; the power of that check is reported as the bias detectable at 80% power),
  SD ratio with a bootstrap CI over sessions (bootstrap seed 27,182,818).

Added after the main run, BEFORE playing it (2026-11-17): a higher-power unbiasedness check
--------------------------------------------------------------------------------------------
The main run's paired-difference check can only detect a bias of ~20 bb/100 (80% power). As
an extra code check on many all-in hands: CFRBot vs AlwaysRaiseBot (cheap to play, all-in
heavy), 200 sessions x 500 hands, master seed 17,320,508 (fresh), same estimators. Reported
separately; it is not part of the variance-reduction claims for the two main pairings.

Post-hoc (decided after seeing the main run, labelled as such in the write-up): per-hand and
per-pair SD ratios with a session-bootstrap CI, computed from the saved per-hand arrays.

Shared seed (disclosed after review): the two main pairings use the SAME master seed, so session s
of CFRBot vs BayesBot and session s of BayesBot vs EquityBot deal the same cards (session_seed does
not depend on the pairing). Their luck terms are therefore correlated, and the two unbiasedness
checks are not independent. ``--posthoc`` estimates that correlation from the sessions (paired by
index) and combines the checks with a covariance-aware test. For future runs: salt the master seed
per pairing (e.g. hash(master_seed, pairing)), outside ``nutsfinder/eval/sessions.py`` whose seed
derivation the official results depend on.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_phase6_for_phase7", ROOT / "examples" / "run_phase6.py")
rp6 = importlib.util.module_from_spec(_spec)
sys.modules["run_phase6_for_phase7"] = rp6  # spawn workers re-import it by this name
_spec.loader.exec_module(rp6)

from nutsfinder.eval.breakdown import END_STREETS, _fold_dict, end_street, fold_stats  # noqa: E402
from nutsfinder.eval.sessions import play_session  # noqa: E402
from nutsfinder.eval.stats import summarize  # noqa: E402
from nutsfinder.eval.variance_reduction import ESTIMATORS, compare_estimators, session_adjustments  # noqa: E402

RESULTS = ROOT / "results"
MASTER_SEED = 31_415_926
SMOKE_SEED = 14_142_135
BOOT_SEED = 27_182_818
N_BOOT = 4000
N_SESSIONS = 40
PAIRS = 250
PAIRINGS = [("BayesBot", "EquityBot"), ("CFRBot", "BayesBot")]  # slowest first (better packing)
PROCESSES = 22
POWER_SEED = 17_320_508
POWER_SESSIONS = 200
POWER_PAIRINGS = [("CFRBot", "AlwaysRaiseBot")]


def _task(args: tuple) -> dict:
    """Play one session in a worker, compute every estimator per hand, return numbers only."""
    pairing, pairs, seed, s = args
    t0 = time.perf_counter()
    sess = play_session(rp6.BOTS[pairing[0]], rp6.BOTS[pairing[1]], pairs, seed, s, True)
    t1 = time.perf_counter()
    adj = session_adjustments(sess.games)
    t2 = time.perf_counter()
    return {
        "pairing": pairing, "index": s, "seed": sess.seed, "a_desc": sess.bot_a_desc, "b_desc": sess.bot_b_desc,
        "adj": adj, "end": [end_street(g.log) for g in sess.games],
        "folds_a": fold_stats(sess.games, "a"), "folds_b": fold_stats(sess.games, "b"),
        "play_s": t1 - t0, "adjust_s": t2 - t1,
    }


def _merge_folds(fs_list) -> dict:
    tot = fs_list[0]
    for fs in fs_list[1:]:
        for d_tot, d in ((tot.faced_bet, fs.faced_bet), (tot.folded_to_bet, fs.folded_to_bet),
                         (tot.free_folds, fs.free_folds)):
            for k in d_tot:
                d_tot[k] += d[k]
    return _fold_dict(tot)


def analyse(pairing: tuple[str, str], sessions: list[dict], rng: np.random.Generator) -> dict:
    """Everything reported for one pairing (pure: no I/O)."""
    hps = 2 * PAIRS
    per = {e: np.stack([s["adj"][e] for s in sessions]) for e in ESTIMATORS}  # (n, 500)
    out: dict = {"pairing": f"{pairing[0]} vs {pairing[1]}", "n_sessions": len(sessions), "hands_per_session": hps,
                 "hands_played": len(sessions) * hps}
    # Duplicate sessions (the evaluation unit).
    out["duplicate"] = {e: compare_estimators(per["raw"].sum(1), per[e].sum(1), hps, rng, N_BOOT)
                        for e in ESTIMATORS if e != "raw"}
    out["duplicate_raw"] = _summ(per["raw"].sum(1), hps)
    # Single room (no duplicate): room 1 = game 1 of every pair, 250 hands.
    out["room1_plain"] = {e: compare_estimators(per["raw"][:, 0::2].sum(1), per[e][:, 0::2].sum(1), PAIRS, rng, N_BOOT)
                          for e in ESTIMATORS if e != "raw"}
    # Duplicate vs plain for reference: SD per session of each design, raw.
    out["per_hand_sd"] = {e: float(per[e].std(ddof=1)) for e in ESTIMATORS}
    allin = np.concatenate([s["adj"]["allin_street"] for s in sessions])
    out["allin_hands"] = {"preflop": int((allin == 0).sum()), "flop": int((allin == 1).sum()),
                          "turn": int((allin == 2).sum()), "share": float((allin >= 0).mean())}
    # Breakdown by position (per session, CI across sessions), raw and adjusted.
    btn = np.stack([s["adj"]["a_on_button"] for s in sessions])
    out["by_position"] = {}
    for label, mask in (("button", btn), ("big_blind", ~btn)):
        out["by_position"][label] = {e: _summ((per[e] * mask).sum(1), PAIRS) for e in ESTIMATORS}
    # By end street: contribution to bb/100 (descriptive), raw and adjusted.
    ends = np.array([s["end"] for s in sessions])
    out["by_end_street"] = {st: {"share": float((ends == st).mean()),
                                 **{f"bb100_{e}": float(100 * (per[e] * (ends == st)).sum() / per[e].size)
                                    for e in ESTIMATORS}} for st in END_STREETS}
    out["folds_a"] = _merge_folds([s["folds_a"] for s in sessions])
    out["folds_b"] = _merge_folds([s["folds_b"] for s in sessions])
    out["leaks"] = [f"{pairing[i]} folds {v['fold_to_bet_rate']:.0%} to {st} bets ({v['faced_bet']:,} faced)"
                    for i, key in ((0, "folds_a"), (1, "folds_b")) for st, v in out[key].items()
                    if v["faced_bet"] >= 100 and v["fold_to_bet_rate"] > 0.5]
    out["seconds"] = {"play_sum": float(sum(s["play_s"] for s in sessions)),
                      "adjust_sum": float(sum(s["adjust_s"] for s in sessions))}
    out["session_seeds"] = [s["seed"] for s in sessions]
    out["bot_descs"] = [sessions[0]["a_desc"], sessions[0]["b_desc"]]
    out["hand_results_bb"] = {e: per[e].round(6).tolist() for e in ESTIMATORS}
    return out


def _summ(totals: np.ndarray, hands: int) -> dict:
    s = summarize(np.asarray(totals, dtype=float).tolist(), hands)
    return {"mean_bb100": s.mean_bb100, "se_bb100": s.se_bb100, "ci95_bb100": list(s.ci_bb100),
            "sd_bb100_per_unit": s.sd * 100 / hands, "p_value": s.p_value}


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = {e: np.array(res["hand_results_bb"][e]) for e in ESTIMATORS}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"raw": "tab:gray", "allin": "tab:blue", "chance": "tab:green"}
    hps = res["hands_per_session"]
    for e in ESTIMATORS:
        tot = per[e].sum(1)
        cum = np.cumsum(tot)
        k = np.arange(1, len(tot) + 1)
        m, sd = tot.mean(), tot.std(ddof=1)
        ax[0].plot(k * hps, cum, color=colors[e], label=f"{e}: {100 * m / hps:+.1f} bb/100, SD/session {100 * sd / hps:.1f}")
        ax[0].fill_between(k * hps, m * k - 1.96 * sd * np.sqrt(k), m * k + 1.96 * sd * np.sqrt(k), color=colors[e], alpha=0.12)
    ax[0].set_xlabel("hands"); ax[0].set_ylabel("cumulative bb (A)")
    ax[0].set_title(f"{res['pairing']}: cumulative result, 95% band around the mean line")
    ax[0].legend(fontsize=8)
    bins = np.linspace(min(per[e].sum(1).min() for e in ESTIMATORS), max(per[e].sum(1).max() for e in ESTIMATORS), 25)
    for e in ESTIMATORS:
        ax[1].hist(per[e].sum(1), bins=bins, histtype="step", lw=2, color=colors[e], label=e)
    ax[1].set_xlabel(f"session total (bb over {hps} hands)"); ax[1].set_ylabel("sessions"); ax[1].legend()
    ax[1].set_title("per-session results, raw vs adjusted")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def posthoc() -> None:
    """Per-hand and per-pair SD ratios (session-bootstrap CIs) from the saved JSONs, and the combined
    unbiasedness test. POST-HOC."""
    diffs: dict[str, dict] = {}
    for j, p in enumerate(PAIRINGS + POWER_PAIRINGS):
        rng = np.random.default_rng([BOOT_SEED, 1, j])  # per pairing: independent of which files exist
        stem = f"phase7_variance_{rp6.SHORT[p[0]]}_vs_{rp6.SHORT[p[1]]}"
        if p in POWER_PAIRINGS:
            stem = stem.replace("variance_", "variance_powercheck_")
        files = sorted(RESULTS.glob(f"*_{stem}.json"))
        if not files:
            continue
        d = json.loads(files[-1].read_text())
        per = {e: np.array(d["hand_results_bb"][e]) for e in ESTIMATORS}
        n = per["raw"].shape[0]
        idx = rng.integers(0, n, size=(N_BOOT, n))
        out = {}
        for e in ("allin", "chance"):
            row = {}
            for unit, f in (("hand", lambda a: a), ("pair", lambda a: a[:, 0::2] + a[:, 1::2]),
                            ("room_session", lambda a: np.stack([a[:, 0::2].sum(1), a[:, 1::2].sum(1)], 1)),
                            ("duplicate_session", lambda a: a.sum(1, keepdims=True))):
                r, a = f(per["raw"]), f(per[e])
                point = float(a.std(ddof=1) / r.std(ddof=1))
                boot = np.array([a[i].std(ddof=1) / r[i].std(ddof=1) for i in idx])
                lo, hi = np.quantile(boot, [0.025, 0.975])
                row[unit] = {"sd_ratio": point, "ci95": [float(lo), float(hi)], "var_ratio": point**2}
            # Mean luck removed, with SEs at the pair and session level only. A per-hand SE would be
            # wrong: the two rooms of a duplicate pair share the board with the hands swapped, so their
            # luck terms are correlated (reported below).
            diff = per["raw"] - per[e]
            pair = diff[:, 0::2] + diff[:, 1::2]
            sess = diff.sum(1)
            row["luck_mean_bb100"] = float(100 * diff.mean())
            row["luck_se_bb100_pair_level"] = float(100 * pair.std(ddof=1) / np.sqrt(pair.size) / 2)
            row["luck_se_bb100_session_level"] = float(100 * sess.std(ddof=1) / np.sqrt(n) / diff.shape[1])
            row["within_pair_corr_of_luck"] = float(np.corrcoef(diff[:, 0::2].ravel(), diff[:, 1::2].ravel())[0, 1])
            out[e] = row
            diffs.setdefault(e, {})[f"{p[0]} vs {p[1]}"] = {
                "session_diff_bb100": (100 * sess / diff.shape[1]).tolist(), "master_seed": d["master_seed"]}
        d["posthoc"] = out
        files[-1].write_text(json.dumps(d, indent=1))
        print(p, json.dumps(out, indent=1))
    combined = {e: combined_unbiasedness(v) for e, v in diffs.items() if len(v) == 3}
    if combined:
        today = dt.date.today().isoformat()
        path = RESULTS / f"{today}_phase7_variance_combined_unbiasedness.json"
        path.write_text(json.dumps({"posthoc": True, "date": today, "tests": combined}, indent=1))
        print(json.dumps(combined, indent=1))
        print(f"saved {path.name}")


def combined_unbiasedness(runs: dict[str, dict]) -> dict:
    """Combine the per-run checks "mean of (raw - adjusted) = 0" allowing for shared seeds.

    Runs with the same master seed have the same decks session by session, so their session
    differences are paired by index and correlated: estimate their covariance from the sessions.
    Runs with different seeds are independent. With z_i = mean_i / SE_i and R the correlation matrix
    of the means: chi2 = z' R^-1 z (df = number of runs) and Z = 1'z / sqrt(1'R1).
    """
    from statistics import NormalDist

    names = list(runs)
    x = [np.array(runs[k]["session_diff_bb100"]) for k in names]
    seeds = [runs[k]["master_seed"] for k in names]
    means = np.array([v.mean() for v in x])
    ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in x])
    z = means / ses
    k = len(names)
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            if seeds[i] == seeds[j] and len(x[i]) == len(x[j]):
                corr[i, j] = corr[j, i] = float(np.corrcoef(x[i], x[j])[0, 1])  # corr of means = corr of sessions
    chi2 = float(z @ np.linalg.solve(corr, z))
    big_z = float(z.sum() / np.sqrt(corr.sum()))
    # chi2 survival for odd df via the incomplete gamma series (df = 3 here): no scipy needed.
    p_chi2 = _chi2_sf(chi2, k)
    shared = [(names[i], names[j], corr[i, j]) for i in range(k) for j in range(i + 1, k) if corr[i, j] != 0]
    naive_chi2 = float((z**2).sum())
    return {"runs": names, "master_seeds": seeds, "mean_bb100": means.tolist(), "se_bb100": ses.tolist(),
            "z": z.tolist(), "correlation_shared_seed": [list(t) for t in shared],
            "chi2": chi2, "df": k, "p_chi2": p_chi2,
            "stouffer_z": big_z, "p_stouffer": float(2 * (1 - NormalDist().cdf(abs(big_z)))),
            "naive_independent_chi2": naive_chi2, "naive_p_chi2": _chi2_sf(naive_chi2, k)}


def _chi2_sf(x: float, df: int) -> float:
    """P(chi2_df >= x) via the regularised upper incomplete gamma Q(df/2, x/2) (series + continued fraction)."""
    import math

    a, y = df / 2, x / 2
    if y <= 0:
        return 1.0
    if y < a + 1:  # series for P
        term = total = 1 / a
        n = a
        for _ in range(500):
            n += 1
            term *= y / n
            total += term
        return 1 - total * math.exp(-y + a * math.log(y) - math.lgamma(a))
    b, c, d = y + 1 - a, 1e300, 1 / (y + 1 - a)  # Lentz continued fraction for Q
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / d if d != 0 else 1e300
        c = b + an / c
        h *= d * c
    return math.exp(-y + a * math.log(y) - math.lgamma(a)) * h


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--power-check", action="store_true")
    ap.add_argument("--posthoc", action="store_true")
    args = ap.parse_args(argv)
    if args.posthoc:
        posthoc()
        return
    pairings = POWER_PAIRINGS if args.power_check else PAIRINGS
    if args.smoke:
        n, seed = 2, SMOKE_SEED
    elif args.power_check:
        n, seed = POWER_SESSIONS, POWER_SEED
    else:
        n, seed = N_SESSIONS, MASTER_SEED
    tasks = [(p, PAIRS, seed, s) for p in pairings for s in range(n)]
    turn = str(rp6.TURN_TABLES) if rp6.TURN_TABLES.exists() else None
    t0 = time.perf_counter()
    got: dict[tuple, list] = {p: [] for p in pairings}
    with mp.get_context("spawn").Pool(min(PROCESSES, len(tasks)), initializer=rp6._init_worker, initargs=(turn,)) as pool:
        for r in pool.imap(_task, tasks, chunksize=1):
            got[r["pairing"]].append(r)
            print(f"[{(time.perf_counter() - t0) / 60:.1f} min] {r['pairing']} session {r['index']} "
                  f"play {r['play_s']:.0f}s adjust {r['adjust_s']:.0f}s", flush=True)
    elapsed = time.perf_counter() - t0
    rng = np.random.default_rng(BOOT_SEED)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = dt.date.today().isoformat()
    for p in pairings:
        res = analyse(p, got[p], rng)
        res.update({"master_seed": seed, "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT, "date": today,
                    "commit": commit + " + uncommitted phase 7 code", "elapsed_s_whole_run": elapsed,
                    "processes": PROCESSES, "smoke": args.smoke, "design": __doc__.split("Fixed design")[1][:1800]})
        d = res["duplicate"]
        print(f"\n{res['pairing']}: raw {res['duplicate_raw']['mean_bb100']:+.1f} "
              f"[{res['duplicate_raw']['ci95_bb100'][0]:+.1f}, {res['duplicate_raw']['ci95_bb100'][1]:+.1f}]")
        for e, c in d.items():
            print(f"  {e}: adj {c['adjusted']['mean_bb100']:+.1f}, diff {c['difference_raw_minus_adjusted']['mean_bb100']:+.2f} "
                  f"CI {c['difference_raw_minus_adjusted']['ci95_bb100']}, SD ratio {c['sd_ratio']:.3f} {c['sd_ratio_ci95']}, "
                  f"hands saved {c['hands_saved']:.1%}")
        for e, c in res["room1_plain"].items():
            print(f"  room1 plain {e}: SD ratio {c['sd_ratio']:.3f} {c['sd_ratio_ci95']}, hands saved {c['hands_saved']:.1%}")
        if not args.smoke:
            kind = "variance_powercheck" if args.power_check else "variance"
            stem = f"{today}_phase7_{kind}_{rp6.SHORT[p[0]]}_vs_{rp6.SHORT[p[1]]}"
            (RESULTS / f"{stem}.json").write_text(json.dumps(res, indent=1))
            plot(res, RESULTS / f"{stem}.png")
            print(f"  saved results/{stem}.json/.png")
    print(f"total {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
