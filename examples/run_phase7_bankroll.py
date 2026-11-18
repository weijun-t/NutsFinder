r"""Phase 7, part A: Kelly sizing, risk of ruin and bankroll simulation on the official Phase 5/6 data.

Run:
    .venv\Scripts\python examples\run_phase7_bankroll.py

Reads (never modifies) the official session JSONs, plays no poker. Writes
``results/<date>_phase7_bankroll.json`` and the ``results/phase7_*.png`` plots.

Fixed choices (made before looking at any simulated number)
------------------------------------------------------------
- Data: the 11 official Phase 6 pairings (master seed 202611186) and the
  Phase 5 BayesBot vs EquityBot replication (seed 20261117, 150 sessions).
- Fixed-stake ruin (and stakes changed only between sessions): sigma = room-level effective SD.
  Kelly with per-hand resizing: f* = mu / E[X^2] per hand; its drawdown exponent is scaled by
  E[X^2] / var_room (see nutsfinder/bankroll/estimate.py and the notes, section 4). [Revised after review.]
- Bootstrap: 4,000 resamples of sessions, seed 27,182,818.
- Monte Carlo: seed 16,180,339 (each experiment gets its own stream
  ``default_rng([16180339, k])`` so the numbers do not depend on run order);
  fixed-stake ruin 100,000 paths x 200 room blocks (50,000 hands); i.i.d.-hand
  comparison 20,000 paths x 50,000 hands; Kelly paths 20,000 x 50,000 hands. The data uncertainty of
  the simulated 1%-ruin bankroll: nested bootstrap, 200 session resamples x 20,000 paths (added after
  the first run of this script, before looking at its result).
- Ruin = bankroll <= 0 at any hand. Rake scenario: 5 bb/100 taken from A on every hand.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime as dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nutsfinder.bankroll import kelly as K  # noqa: E402
from nutsfinder.bankroll.estimate import (  # noqa: E402
    SessionData, bootstrap_draws, bootstrap_estimates, load_session_data, point_estimates,
)
from nutsfinder.bankroll.simulate import (  # noqa: E402
    ror_curve, running_min_fixed_stake, simulate_continuous_kelly, simulate_stake_levels, wilson_ci,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BOOT_SEED = 27_182_818
MC_SEED = 16_180_339
N_BOOT = 4000
RAKE_BB100 = 5.0
RUIN_PATHS, RUIN_BLOCKS = 100_000, 200
IID_PATHS, IID_HANDS = 20_000, 50_000
KELLY_PATHS, KELLY_BLOCKS = 20_000, 200
NESTED_REPS, NESTED_PATHS = 200, 20_000
PHASE6 = ["CFR_vs_Bayes", "CFR_vs_Equity", "CFR_vs_PotOdds", "CFR_vs_CallStation", "CFR_vs_AlwaysRaise",
          "CFR_vs_Random", "Bayes_vs_Equity", "Bayes_vs_PotOdds", "Bayes_vs_CallStation", "Bayes_vs_AlwaysRaise",
          "Bayes_vs_Random"]
RUIN_MATCHUPS = ["CFR_vs_Bayes", "CFR_vs_Equity", "Bayes_vs_Equity"]


def rng_for(k: int) -> np.random.Generator:
    return np.random.default_rng([MC_SEED, k])


def load_all() -> dict[str, SessionData]:
    out = {}
    for m in PHASE6:
        out[m] = load_session_data(RESULTS / f"2026-11-17_phase6_{m}_sessions.json")
    out["Bayes_vs_Equity_phase5"] = load_session_data(RESULTS / "2026-11-16_Bayes_vs_Equity_sessions.json",
                                                      "BayesBot vs EquityBot (Phase 5)")
    return out


KEYS = ("bb100", "sd_hand", "sd_room", "sd_dup", "ratio_room", "ratio_dup", "kelly_bankroll", "bankroll_1pct",
        "bankroll_1pct_dup", "kelly_growth", "kelly_bankroll_session_gauss", "drawdown_exponent_scale")


def estimates(data: dict[str, SessionData]) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    rows = {}
    for name, d in data.items():
        p = point_estimates(d)
        ci = bootstrap_estimates(d, rng, N_BOOT, KEYS)
        rake = d.shifted(RAKE_BB100 / 100)
        pr = point_estimates(rake)
        cir = bootstrap_estimates(rake, rng, N_BOOT, ("bb100", "kelly_bankroll", "bankroll_1pct"))
        rows[name] = {"name": d.name, "n_sessions": d.n_sessions, "hands": int(d.hands.size),
                      "master_seed": d.master_seed, "point": p, "ci95": {k: list(v) for k, v in ci.items()},
                      "rake": {"point": {k: pr[k] for k in ("bb100", "kelly_bankroll", "bankroll_1pct", "kelly_growth")},
                               "ci95": {k: list(v) for k, v in cir.items()}}}
    return rows


# --- Risk of ruin: formula vs simulation -------------------------------------------------


def ruin_study(d: SessionData, k: int, bankrolls: np.ndarray) -> dict:
    p = point_estimates(d)
    t0 = time.perf_counter()
    # Horizon check on the SAME paths: their running minimum after half the horizon (a checkpoint
    # inside the full run). Ruin can only grow with the horizon; a big gap would mean the horizon is
    # too short to stand in for "ever".
    cp: dict = {RUIN_BLOCKS // 2: None}
    m, _ = running_min_fixed_stake(d.room_blocks, RUIN_PATHS, RUIN_BLOCKS, rng_for(k), checkpoints=cp)
    block = ror_curve(m, bankrolls)
    half = ror_curve(cp[RUIN_BLOCKS // 2], bankrolls)
    hands = d.hands.reshape(-1, 1)
    m_iid, _ = running_min_fixed_stake(hands, IID_PATHS, IID_HANDS, rng_for(k + 1000), chunk=500)
    iid = ror_curve(m_iid, bankrolls)
    draws = bootstrap_draws(d, np.random.default_rng([BOOT_SEED, k]), N_BOOT)
    mu, vr = draws["mu"], draws["var_room"]
    f_band = np.exp(-2 * np.maximum(mu, 1e-12)[:, None] * bankrolls[None, :] / vr[:, None])
    f_band = np.where(mu[:, None] > 0, f_band, 1.0)
    out = {
        "bankroll": bankrolls.tolist(),
        "formula_room": K.risk_of_ruin(p["mu"], p["sd_room"], bankrolls).tolist(),
        "formula_room_ci": [np.quantile(f_band, 0.025, axis=0).tolist(), np.quantile(f_band, 0.975, axis=0).tolist()],
        "formula_dup": K.risk_of_ruin(p["mu"], p["sd_dup"], bankrolls).tolist(),
        "lundberg_hand_bound": np.exp(-p["lundberg_hand"] * bankrolls).tolist(),
        "mc_block": {kk: np.asarray(v).tolist() for kk, v in block.items() if kk != "n"} | {"n": block["n"]},
        "mc_block_half_horizon_p": half["p"].tolist(),
        "mc_iid": {kk: np.asarray(v).tolist() for kk, v in iid.items() if kk != "n"} | {"n": iid["n"]},
        "seconds": time.perf_counter() - t0,
        "paths": RUIN_PATHS, "horizon_hands": RUIN_BLOCKS * d.room_blocks.shape[1],
        "iid_paths": IID_PATHS, "iid_horizon_hands": IID_HANDS,
    }
    # Bankroll for 1% ruin from the simulation (smallest grid B with MC p <= 1%).
    ok = np.flatnonzero(block["p"] <= 0.01)
    out["mc_bankroll_1pct"] = float(bankrolls[ok[0]]) if len(ok) else None
    ok = np.flatnonzero(block["hi"] <= 0.01)
    out["mc_bankroll_1pct_conservative"] = float(bankrolls[ok[0]]) if len(ok) else None
    return out


def mc_bankroll_ci(d: SessionData, k: int, bankrolls: np.ndarray, reps: int = NESTED_REPS,
                   paths: int = NESTED_PATHS) -> dict:
    """Data uncertainty of the simulated 1%-ruin bankroll: nested bootstrap.

    Outer: resample the sessions (the data); inner: block-bootstrap Monte Carlo on that resample.
    The spread of the outer replicates is the data uncertainty (the inner MC error is small by comparison).
    """
    rng = np.random.default_rng([BOOT_SEED, k, 1])  # own stream: not the formula band's resamples
    out = []
    for r in range(reps):
        idx = rng.integers(0, d.n_sessions, size=d.n_sessions)
        dd = SessionData(d.name, d.hands[idx])
        m, _ = running_min_fixed_stake(dd.room_blocks, paths, RUIN_BLOCKS, rng_for(100_000 + 1000 * k + r))
        p = ror_curve(m, bankrolls)["p"]
        ok = np.flatnonzero(p <= 0.01)
        out.append(float(bankrolls[ok[0]]) if len(ok) else math.inf)
    v = np.array(out)
    return {"reps": reps, "paths": paths, "median": float(np.median(v)),
            "ci95": [float(np.quantile(v, 0.025, method="inverted_cdf")), float(np.quantile(v, 0.975, method="inverted_cdf"))]}


# --- Kelly: continuous resizing and discrete stake levels -----------------------------------


def kelly_study(d: SessionData, k: int) -> dict:
    p = point_estimates(d)
    # Per-hand resizing: f* = mu / E[X^2] (per-hand second moment); drawdowns: exponent scaled by
    # E[X^2] / var_room (log W fluctuates with the long-run variance). See notes section 4.
    f_star = p["mu"] / p["ex2_hand"]
    g_star = p["kelly_growth"]
    scale = p["drawdown_exponent_scale"]
    rows = []
    for j, c in enumerate((0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0)):
        r = simulate_continuous_kelly(d.room_blocks, c * f_star, KELLY_PATHS, KELLY_BLOCKS, rng_for(k))  # CRN over c
        hands = r["hands"]
        g = r["logw"] / hands
        halved = r["minlog"] <= math.log(0.5)
        tenth = r["minlog"] <= math.log(0.1)
        lo, hi = wilson_ci(int(halved.sum()), KELLY_PATHS)
        lo10, hi10 = wilson_ci(int(tenth.sum()), KELLY_PATHS)
        rows.append({"c": c, "growth_mc": float(g.mean()), "growth_mc_se": float(g.std(ddof=1) / math.sqrt(KELLY_PATHS)),
                     "growth_formula": float(K.fractional_growth_ratio(c) * g_star),
                     "p_halve_mc": float(halved.mean()), "p_halve_ci": [float(lo), float(hi)],
                     "p_halve_formula": K.prob_ever_halve(c, scale), "p_halve_formula_iid": K.prob_ever_halve(c),
                     "p_tenth_mc": float(tenth.mean()), "p_tenth_ci": [float(lo10), float(hi10)],
                     "p_tenth_formula": K.prob_ever_fall_to(c, 0.1, scale),
                     "wiped_out": int(r["dead"].sum()), "median_multiple": float(np.exp(np.median(r["logw"])))})
    return {"f_star": f_star, "kelly_bankroll": 1 / f_star, "g_star": g_star, "drawdown_exponent_scale": scale,
            "horizon_hands": KELLY_BLOCKS * 250,
            "paths": KELLY_PATHS, "rows": rows}


def levels_study(d: SessionData, k: int, w0: float) -> dict:
    """Same start bankroll w0: fixed stake vs discrete stake levels vs continuous Kelly."""
    p = point_estimates(d)
    kb = p["kelly_bankroll_session_gauss"]  # stakes change between sessions: the bet is a session (var_room)
    kb_hand = p["kelly_bankroll"]           # continuous resizing: per-hand Kelly E[X^2] / mu
    out = {"w0": w0, "kelly_bankroll": kb, "kelly_bankroll_per_hand": kb_hand, "horizon_hands": KELLY_BLOCKS * 250,
           "paths": KELLY_PATHS, "rows": []}
    blocks = d.room_blocks
    low = np.cumsum(blocks, axis=1).min(axis=1)
    out["room_sessions"] = len(blocks)
    out["share_room_drawdown_beyond"] = {f"{x:.0f}": float((low <= -x).mean()) for x in (kb / 2, kb, 1.5 * kb, 2 * kb)}
    out["worst_room_drawdown"] = float(low.min())
    # Fixed stake 1.
    m, fin = running_min_fixed_stake(blocks, KELLY_PATHS, KELLY_BLOCKS, rng_for(k))
    ruin = m <= -w0
    half = m <= -w0 / 2
    out["rows"].append(_lvl_row("fixed stake 1", ruin, half, np.where(ruin, 0.0, w0 + fin) / w0, None,
                                K.risk_of_ruin(p["mu"], p["sd_room"], w0)))
    for c in (1.0, 0.5):
        r = simulate_stake_levels(blocks, w0, kb / c, KELLY_PATHS, KELLY_BLOCKS, rng_for(k))
        row = _lvl_row(f"stake levels x2, {c:g} Kelly (N = {kb / c:.0f} bb)", r["ruined"], r["halved"],
                       r["W"] / w0, float(r["changes"].mean()), None)
        row["ruin_by_hands"] = {str(250 * b): float(r["ruin_by_block"][b - 1]) for b in (20, 50, 100, 150, 200)}
        if kb / c > -out["worst_room_drawdown"]:
            # Every session starts with W >= N s and no resampled session loses more than N bb: ruin is
            # impossible by construction, so the bootstrap says nothing about it (no CI reported).
            row["p_ruin"], row["p_ruin_ci"] = None, None
            row["p_ruin_note"] = (f"not estimable by this bootstrap: W/s >= {kb / c:.0f} bb > worst observed "
                                  f"session drawdown {-out['worst_room_drawdown']:.0f} bb")
        out["rows"].append(row)
    for c in (1.0, 0.5):
        r = simulate_continuous_kelly(blocks, c / kb_hand, KELLY_PATHS, KELLY_BLOCKS, rng_for(k))
        out["rows"].append(_lvl_row(f"continuous {c:g} Kelly (every hand)", r["dead"], r["minlog"] <= math.log(0.5),
                                    np.exp(r["logw"]), None, 0.0))
    return out


def _lvl_row(label, ruin, half, multiple, changes, formula) -> dict:
    n = len(ruin)
    lo, hi = wilson_ci(int(ruin.sum()), n)
    hlo, hhi = wilson_ci(int(half.sum()), n)
    return {"strategy": label, "p_ruin": float(ruin.mean()), "p_ruin_ci": [float(lo), float(hi)],
            "p_ruin_formula": formula, "p_half": float(half.mean()), "p_half_ci": [float(hlo), float(hhi)],
            "median_multiple": float(np.median(multiple)), "p_below_start": float((multiple < 1).mean()),
            "mean_stake_changes": changes}


# --- Parameter uncertainty --------------------------------------------------------------------


def uncertainty_study(d: SessionData, k: int, first_sessions: int | None = None) -> dict:
    if first_sessions:
        d = SessionData(d.name + f" (first {first_sessions} sessions)", d.hands[:first_sessions], d.master_seed, d.source)
    p = point_estimates(d)
    draws = bootstrap_draws(d, np.random.default_rng([BOOT_SEED, k]), N_BOOT)
    t = p["mu"] / p["se_mu"]
    mu_lo, mu_hi = p["mu"] - 1.96 * p["se_mu"], p["mu"] + 1.96 * p["se_mu"]
    f_hat = p["mu"] / p["ex2_hand"]          # per-hand resizing Kelly
    ex2 = draws["var_hand"] + draws["mu"] ** 2
    # Bootstrap as an APPROXIMATE posterior: E[f mu - f^2 E[X^2] / 2] is maximised at
    # f = E[mu] / E[E[X^2]], which is ~ the plug-in by construction (growth is linear in mu).
    f_post = draws["mu"].mean() / ex2.mean()
    # Fraction maximising the 5% quantile of the growth across the bootstrap (a robust choice).
    cs = np.linspace(0.01, 2.0, 400)
    g = cs[:, None] * f_hat * draws["mu"][None, :] - (cs[:, None] * f_hat) ** 2 * ex2[None, :] / 2
    q05 = np.quantile(g, 0.05, axis=1)
    c_robust = float(cs[np.argmax(q05)])
    return {
        "name": d.name, "hands": int(d.hands.size), "bb100": p["bb100"], "se_bb100": 100 * p["se_mu"],
        "t": t, "kelly_bankroll": p["kelly_bankroll"],
        "kelly_bankroll_at_mu_ci": [p["ex2_hand"] / mu_hi if mu_hi > 0 else math.inf,
                                    p["ex2_hand"] / mu_lo if mu_lo > 0 else math.inf],
        "c_shrinkage_opt": K.optimal_shrinkage(t),
        "growth_ratio_full_plugin": K.noisy_plugin_growth_ratio(1.0, t),
        "growth_ratio_half_plugin": K.noisy_plugin_growth_ratio(0.5, t),
        "c_at_lower_bound": max(mu_lo, 0) / p["mu"],
        "c_bootstrap_posterior_mean": f_post / f_hat,
        "c_robust_q05": c_robust,
    }


def misspec_table() -> list[dict]:
    rows = []
    for lam in (0.5, 0.75, 1.0, 1.5, 2.0):
        rows.append({"lambda": lam, **{f"c={c:g}": K.growth_ratio_misspecified(c, lam) for c in (0.25, 0.5, 1.0, 1.5, 2.0)}})
    return rows


# --- Plots ---------------------------------------------------------------------------------


def plot_ror(ruin: dict, rows: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ruin), figsize=(6 * len(ruin), 4.8), squeeze=False)
    for ax, (name, r) in zip(axes[0], ruin.items()):
        b = np.array(r["bankroll"])
        mcb = r["mc_block"]
        ax.plot(b, r["formula_room"], color="k", label=r"formula $e^{-2\mu B/\sigma^2}$, $\sigma$ = room-level")
        ax.fill_between(b, *r["formula_room_ci"], color="k", alpha=0.12, label="formula, 95% band (bootstrap of $\\mu,\\sigma$)")
        ax.plot(b, r["formula_dup"], color="tab:red", ls="--", label=r"formula with duplicate $\sigma$ (wrong for a bankroll)")
        ax.plot(b, r["lundberg_hand_bound"], color="tab:purple", ls=":", label="Lundberg bound $e^{-RB}$ (i.i.d. hands)")
        ax.plot(b, mcb["p"], color="tab:blue", label="MC: session-block bootstrap")
        ax.fill_between(b, mcb["lo"], mcb["hi"], color="tab:blue", alpha=0.25)
        ax.plot(b, r["mc_iid"]["p"], color="tab:orange", lw=1, label="MC: i.i.d. hand bootstrap")
        ax.set_yscale("log"); ax.set_ylim(5e-5, 1.05)
        ax.axhline(0.01, color="gray", lw=0.6)
        ax.set_xlim(0, 2500 if rows[name]["point"]["bankroll_1pct"] < 2000 else 6000)
        ax.set_xlabel("bankroll B (bb), fixed stake"); ax.set_ylabel("P(ruin)")
        ax.set_title(f"{rows[name]['name']}: {rows[name]['point']['bb100']:+.0f} bb/100")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_growth(d: SessionData, k: int, unc: list[dict], path: Path) -> None:
    p = point_estimates(d)
    draws = bootstrap_draws(d, np.random.default_rng([BOOT_SEED, k]), N_BOOT)
    f_hat = p["mu"] / p["ex2_hand"]
    ex2 = draws["var_hand"] + draws["mu"] ** 2
    cs = np.linspace(0, 2.4, 241)
    g = cs[:, None] * f_hat * draws["mu"][None, :] - (cs[:, None] * f_hat) ** 2 * ex2[None, :] / 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    scale = 1e4
    ax[0].plot(cs, scale * (cs * f_hat * p["mu"] - (cs * f_hat) ** 2 * p["ex2_hand"] / 2), color="k", label="at the point estimate")
    ax[0].fill_between(cs, scale * np.quantile(g, 0.025, axis=1), scale * np.quantile(g, 0.975, axis=1), alpha=0.2,
                       label=r"95% band: true $(\mu,\sigma)$ across the bootstrap")
    ax[0].axhline(0, color="gray", lw=0.6); ax[0].axvline(1, color="gray", ls=":"); ax[0].axvline(0.5, color="gray", ls=":")
    ax[0].set_xlabel(r"stake as a multiple c of the estimated per-hand Kelly stake $\hat\mu/\hat E[X^2]$")
    ax[0].set_ylabel(r"log-growth per hand ($\times 10^{-4}$)")
    ax[0].set_title(f"{d.name}: growth vs Kelly fraction"); ax[0].legend(fontsize=8)
    ks = np.linspace(0, 2, 201)
    for t, ls in ((1.5, ":"), (3.0, "--"), (unc[0]["t"], "-")):
        ax[1].plot(ks, K.noisy_plugin_growth_ratio(ks, t), ls=ls, label=f"t = mu/SE = {t:.1f}" + (" (CFR vs Bayes)" if ls == "-" else ""))
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_ylim(-1, 1.05)
    ax[1].set_xlabel(r"k: stake = k $\hat\mu/\sigma^2$ with $\hat\mu$ ~ N($\mu$, SE$^2$)")
    ax[1].set_ylabel("expected growth / max growth")
    ax[1].set_title("estimation noise: expected growth of plug-in Kelly"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_paths(d: SessionData, k: int, path: Path) -> None:
    p = point_estimates(d)
    f = 1 / p["kelly_bankroll"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, color in ((1.0, "tab:red"), (0.5, "tab:blue")):
        r = simulate_continuous_kelly(d.room_blocks, c * f, 8, 40, rng_for(k), record_paths=8, record_every=10)
        for i, row in enumerate(r["paths"]):
            ax.plot(r["path_hands"], row / math.log(10), color=color, lw=0.8, alpha=0.7,
                    label=f"{c:g} Kelly (same resampled hands)" if i == 0 else None)
        ax.plot(r["path_hands"], K.fractional_growth_ratio(c) * p["kelly_growth"] * r["path_hands"] / math.log(10),
                color=color, ls="--", lw=2, label=f"{c:g} Kelly: expected log-growth")
    ax.axhline(math.log10(0.5), color="gray", ls=":", label="half the starting bankroll")
    ax.set_xlabel("hands"); ax.set_ylabel("log10(bankroll / start)")
    ax.set_title(f"{d.name}: 8 resampled bankroll paths, full vs half Kelly, resized every hand")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_required(d: SessionData, k: int, path: Path) -> None:
    p = point_estimates(d)
    draws = bootstrap_draws(d, np.random.default_rng([BOOT_SEED, k]), N_BOOT)
    lo, hi = p["mu"] - 1.96 * p["se_mu"], p["mu"] + 1.96 * p["se_mu"]
    mus = np.linspace(max(lo * 0.6, 1e-3), hi * 1.15, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Fixed-stake ruin uses the long-run (room) variance; per-hand Kelly uses the per-hand second moment.
    for mult, key, label, color in (
            (math.log(100) / 2, "var_room", "fixed stake, 1% ruin: $\\sigma_{room}^2\\ln(100)/(2\\mu)$", "tab:blue"),
            (1.0, "var_hand", "full Kelly bankroll (per-hand resizing) $E[X^2]/\\mu$", "tab:red"),
            (2.0, "var_hand", "half Kelly bankroll $2E[X^2]/\\mu$", "tab:green")):
        s2_lo, s2_hi = np.quantile(draws[key], [0.025, 0.975])
        extra = mus**2 if key == "var_hand" else 0.0
        ax.plot(100 * mus, mult * (p[key] + extra) / mus, color=color, label=label)
        ax.fill_between(100 * mus, mult * (s2_lo + extra) / mus, mult * (s2_hi + extra) / mus, color=color, alpha=0.15)
    ax.axvspan(100 * lo, 100 * hi, color="gray", alpha=0.15, label="95% CI of the win rate")
    ax.axvline(100 * p["mu"], color="gray", ls="--")
    ax.axvline(100 * p["mu"] - RAKE_BB100, color="k", ls=":", label=f"win rate after {RAKE_BB100:g} bb/100 rake")
    ax.set_xlabel("true win rate (bb/100)"); ax.set_ylabel("bankroll (bb)")
    ax.set_title(f"{d.name}: required bankroll vs the true win rate (bands: 95% CI of the variance)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    data = load_all()
    rows = estimates(data)
    print(f"estimates {time.perf_counter() - t0:.1f}s", flush=True)
    for name, r in rows.items():
        p, c = r["point"], r["ci95"]
        print(f"{r['name']:36s} {p['bb100']:+8.1f} [{c['bb100'][0]:+.1f},{c['bb100'][1]:+.1f}] sd_room {p['sd_room']:.1f} "
              f"r_room {p['ratio_room']:.2f} r_dup {p['ratio_dup']:.2f} Kelly {p['kelly_bankroll']:.0f} "
              f"[{c['kelly_bankroll'][0]:.0f},{c['kelly_bankroll'][1]:.0f}] B1% {p['bankroll_1pct']:.0f} "
              f"[{c['bankroll_1pct'][0]:.0f},{c['bankroll_1pct'][1]:.0f}]")
    bankrolls = np.arange(0, 8001, 10.0)
    ruin = {}
    for j, m in enumerate(RUIN_MATCHUPS):
        ruin[m] = ruin_study(data[m], 10 + j, bankrolls)
        ruin[m]["mc_bankroll_1pct_nested"] = mc_bankroll_ci(data[m], 10 + j, bankrolls)
        print(f"ruin {m}: {ruin[m]['seconds']:.1f}s, MC B1% {ruin[m]['mc_bankroll_1pct']} "
              f"nested {ruin[m]['mc_bankroll_1pct_nested']}", flush=True)
    ruin["CFR_vs_Bayes_rake"] = ruin_study(data["CFR_vs_Bayes"].shifted(RAKE_BB100 / 100), 20, bankrolls)
    rows["CFR_vs_Bayes_rake"] = {"name": f"CFRBot vs BayesBot, {RAKE_BB100:g} bb/100 rake",
                                 "point": point_estimates(data["CFR_vs_Bayes"].shifted(RAKE_BB100 / 100))}
    kel = kelly_study(data["CFR_vs_Bayes"], 30)
    print(f"kelly study {time.perf_counter() - t0:.1f}s", flush=True)
    w0 = round(point_estimates(data["CFR_vs_Bayes"])["bankroll_1pct"], -1)
    lv = levels_study(data["CFR_vs_Bayes"], 40, w0)
    print(f"levels study {time.perf_counter() - t0:.1f}s", flush=True)
    unc = [uncertainty_study(data["CFR_vs_Bayes"], 50), uncertainty_study(data["CFR_vs_Bayes"], 51, first_sessions=20),
           uncertainty_study(data["Bayes_vs_Equity"], 52), uncertainty_study(data["Bayes_vs_Equity"], 53, first_sessions=10)]
    plot_ror({m: ruin[m] for m in RUIN_MATCHUPS}, rows, RESULTS / "phase7_ror_vs_bankroll.png")
    plot_growth(data["CFR_vs_Bayes"], 60, unc, RESULTS / "phase7_growth_vs_kelly_fraction.png")
    plot_paths(data["CFR_vs_Bayes"], 70, RESULTS / "phase7_bankroll_paths.png")
    plot_required(data["CFR_vs_Bayes"], 80, RESULTS / "phase7_required_bankroll_vs_mu.png")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = dt.date.today().isoformat()
    out = {"date": today, "commit": commit + " + uncommitted phase 7 code", "bootstrap_seed": BOOT_SEED, "mc_seed": MC_SEED,
           "n_boot": N_BOOT, "rake_bb100": RAKE_BB100, "design": __doc__.split("Fixed choices")[1],
           "estimates": rows, "ruin": ruin, "kelly_continuous": kel, "levels": lv, "uncertainty": unc,
           "misspecification": misspec_table(), "elapsed_s": time.perf_counter() - t0}
    path = RESULTS / f"{today}_phase7_bankroll.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"kelly": kel["rows"], "levels": lv["rows"], "uncertainty": unc}, indent=1, default=float))
    print(f"saved {path.name}; total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
