"""Saving match results: a JSON file and a plot per match.

The plot has two panels:

- Cumulative bb won by A against hands played, with the 95% band that pure
  luck would allow a *zero-edge* bot: +/- 1.96 * sigma_unit * sqrt(k) after k
  units (the SD of a sum of k independent units is sigma * sqrt(k)). If the
  line ends outside the band, the result is significant at the 5% level. The
  band grows like sqrt(k) while a real edge grows like k, which is the whole
  reason long matches eventually separate skill from luck.
- A histogram of per-unit results (log scale on the counts, because a few
  stack-sized results would otherwise be invisible next to the many small
  ones). The fat tails are why poker needs so many hands.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402

from nutsfinder.eval.breakdown import breakdown  # noqa: E402
from nutsfinder.eval.match import MatchResult  # noqa: E402
from nutsfinder.eval.sessions import SessionMatchResult, learning_curve, session_breakdown  # noqa: E402
from nutsfinder.eval.stats import z_critical  # noqa: E402

__all__ = [
    "result_dict",
    "save_json",
    "save_plot",
    "session_result_dict",
    "save_session_json",
    "save_session_plot",
]


def _clean(x: Any) -> Any:
    """Replace NaN with None so the JSON is standard."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def result_dict(result: MatchResult, extra: dict | None = None) -> dict:
    """The match as a JSON-friendly dict (per-unit results included)."""
    lo, hi = result.ci_bb100
    d = {
        "bot_a": result.bot_a,
        "bot_b": result.bot_b,
        "bot_a_desc": getattr(result, "bot_a_desc", ""),
        "bot_b_desc": getattr(result, "bot_b_desc", ""),
        "kind": result.kind,
        "master_seed": result.master_seed,
        "seats_swapped": result.kind == "duplicate",
        "unit": result.unit,
        "n_units": result.n_units,
        "hands_played": result.hands_played,
        "mean_bb100": result.mean_bb100,
        "sd_bb_per_unit": result.sd_per_unit,
        "se_bb100": result.se_bb100,
        "ci95_bb100": [lo, hi],
        "p_value": result.p_value,
        "significant": result.stats.significant,
        "verdict": result.verdict(),
        "hands_needed_80pct_power": result.hands_needed(),
        "breakdown": breakdown(result),
        "unit_results_bb": list(result.unit_results),
    }
    if extra:
        d.update(extra)
    return _clean(d)


def save_json(result: MatchResult, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_dict(result, extra), indent=1), encoding="utf-8")


def save_plot(result: MatchResult, path: Path) -> None:
    hpu = result.stats.hands_per_unit
    k = list(range(1, result.n_units + 1))
    hands = [i * hpu for i in k]
    cum, total = [], 0.0
    for x in result.unit_results:
        total += x
        cum.append(total)
    z = z_critical(0.95)
    band = [z * result.sd_per_unit * math.sqrt(i) for i in k]
    slope = [result.stats.mean * i for i in k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(hands, cum, lw=0.8, label=f"{result.bot_a} cumulative bb")
    ax1.plot(hands, slope, "k--", lw=0.8, label=f"observed rate {result.mean_bb100:+.1f} bb/100")
    ax1.fill_between(hands, [-b for b in band], band, color="grey", alpha=0.25,
                     label="95% band for a zero-edge bot")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("hands played")
    ax1.set_ylabel("cumulative bb")
    ax1.set_title(f"{result.bot_a} vs {result.bot_b} ({result.kind})")
    ax1.legend(fontsize=8)

    ax2.hist(result.unit_results, bins=80)
    ax2.set_yscale("log")
    ax2.set_xlabel(f"bb per {result.unit}")
    ax2.set_ylabel("count (log scale)")
    ax2.set_title(f"per-{result.unit} results, SD {result.sd_per_unit:.1f} bb")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- Session matches (Phase 5: learning bots) ----------------------------------


def session_result_dict(result: SessionMatchResult, bucket_pairs: int = 25, extra: dict | None = None) -> dict:
    """A session match as a JSON-friendly dict.

    Includes per-session totals, the learning curve, and every per-hand net
    (``hand_results_bb[s]`` = session s's hands in play order), so the curve
    can be recomputed with any bucket size. Hand logs are not stored.
    """
    lo, hi = result.ci_bb100
    naive = result.naive_pair_stats()
    curve = learning_curve(result, bucket_pairs)
    has_games = all(s.games for s in result.sessions)
    d = {
        "bot_a": result.bot_a,
        "bot_b": result.bot_b,
        "bot_a_desc": getattr(result, "bot_a_desc", ""),
        "bot_b_desc": getattr(result, "bot_b_desc", ""),
        "kind": result.kind,
        "master_seed": result.master_seed,
        "session_seeds": [s.seed for s in result.sessions],
        "seats_swapped": True,
        "fresh_bots_per_session": True,
        "unit": "session",
        "n_sessions": result.n_sessions,
        "pairs_per_session": result.pairs_per_session,
        "hands_per_session": result.hands_per_session,
        "hands_played": result.hands_played,
        "mean_bb100": result.mean_bb100,
        "sd_bb100_per_session": result.sd_session_bb100,
        "se_bb100": result.se_bb100,
        "ci95_bb100": [lo, hi],
        "p_value": result.p_value,
        "significant": result.stats.significant,
        "verdict": result.verdict(),
        "sessions_needed_80pct_power": result.sessions_needed(),
        "naive_per_pair_se_bb100": naive.se_bb100,
        "design_effect": result.design_effect(),
        "learning_curve": {
            "bucket_pairs": bucket_pairs,
            "buckets": [
                {
                    "pairs": [b.start, b.stop],
                    "mean_bb100": b.stats.mean_bb100,
                    "se_bb100": b.stats.se_bb100,
                    "ci95_bb100": list(b.stats.ci_bb100),
                }
                for b in curve
            ],
        },
        "breakdown": session_breakdown(result) if has_games else None,
        "session_bb100": list(result.session_bb100),
        "hand_results_bb": [list(s.hand_results) for s in result.sessions],
    }
    if extra:
        d.update(extra)
    return _clean(d)


def save_session_json(
    result: SessionMatchResult, path: Path, bucket_pairs: int = 25, extra: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session_result_dict(result, bucket_pairs, extra), indent=1), encoding="utf-8")


def save_session_plot(result: SessionMatchResult, path: Path, bucket_pairs: int = 25) -> None:
    """Four panels, every band computed ACROSS independent sessions.

    - learning curve: bb/100 per bucket of pairs with its 95% CI;
    - mean cumulative bb within a session vs pair index, +/- 1.96 SE across
      sessions (valid pointwise because each session gives one value);
    - histogram of per-session bb/100 (the unit of analysis);
    - histogram of per-hand results (descriptive: the fat tails).
    """
    z = z_critical(0.95)
    n = result.n_sessions
    curve = learning_curve(result, bucket_pairs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (ax1, ax2), (ax3, ax4) = axes

    x = [b.mid for b in curve]
    m = [b.stats.mean_bb100 for b in curve]
    lo = [b.stats.ci_bb100[0] for b in curve]
    hi = [b.stats.ci_bb100[1] for b in curve]
    ax1.plot(x, m, "o-", ms=3, lw=1, label="bucket mean")
    ax1.fill_between(x, lo, hi, color="C0", alpha=0.2, label="95% CI per bucket (across sessions)")
    ax1.axhline(result.mean_bb100, color="k", ls="--", lw=0.8, label=f"session mean {result.mean_bb100:+.1f}")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel(f"pair index within session (hands of experience; buckets of {bucket_pairs})")
    ax1.set_ylabel("A bb/100")
    ax1.set_title(f"{result.bot_a} vs {result.bot_b}: learning curve")
    ax1.legend(fontsize=8)

    cums = []
    for s in result.sessions:
        total, row = 0.0, []
        for v in s.pair_results:
            total += v
            row.append(total)
        cums.append(row)
    k = list(range(1, result.pairs_per_session + 1))
    mean_c, band_lo, band_hi = [], [], []
    for i in range(result.pairs_per_session):
        col = [c[i] for c in cums]
        mu = math.fsum(col) / n
        sd = math.sqrt(math.fsum((v - mu) ** 2 for v in col) / (n - 1))
        mean_c.append(mu)
        band_lo.append(mu - z * sd / math.sqrt(n))
        band_hi.append(mu + z * sd / math.sqrt(n))
    hands = [2 * i for i in k]
    ax2.plot(hands, mean_c, lw=1, label="mean over sessions")
    ax2.fill_between(hands, band_lo, band_hi, color="C0", alpha=0.2, label="95% CI (across sessions)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("hands played in session")
    ax2.set_ylabel("cumulative bb (A)")
    ax2.set_title(f"{n} sessions x {result.hands_per_session} hands")
    ax2.legend(fontsize=8)

    ax3.hist(result.session_bb100, bins=min(60, max(10, n // 3)))
    ax3.axvline(0, color="k", lw=0.5)
    ax3.set_xlabel("bb/100 per session")
    ax3.set_ylabel("sessions")
    ax3.set_title(f"per-session results, SD {result.sd_session_bb100:.1f} bb/100, "
                  f"95% CI [{result.ci_bb100[0]:+.1f}, {result.ci_bb100[1]:+.1f}]")

    ax4.hist([h for s in result.sessions for h in s.hand_results], bins=80)
    ax4.set_yscale("log")
    ax4.set_xlabel("bb per hand")
    ax4.set_ylabel("count (log scale)")
    ax4.set_title("per-hand results (descriptive)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
