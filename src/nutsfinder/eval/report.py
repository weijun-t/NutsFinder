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
from nutsfinder.eval.stats import z_critical  # noqa: E402

__all__ = ["result_dict", "save_json", "save_plot"]


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
