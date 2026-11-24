r"""A local poker table in your browser: play against a bot, or replay bot-vs-bot hands.

Run:  .venv\Scripts\python examples\poker_table.py [--port 8765] [--open]

then open the printed URL (http://127.0.0.1:8765/). Only 127.0.0.1 is served.

- Play: pick an opponent (Random, CallStation, AlwaysRaise, PotOdds, Equity,
  Bayes, CFR) and an optional seed. You sit at the bottom; blinds 50/100,
  10,000 stacks reset every hand, the button alternates. After each hand the
  "how the bot thought" panel lists every bot decision (equity / pot odds /
  EV per action, BayesBot's model of you, CFRBot's bucket and strategy).
- Replay: two bots play up to 200 hands; step through any hand action by
  action with all cards shown and the acting bot's reasoning.
- Multiway (Players 3..8): pick your seat and a bot per seat ("Fill with"
  sets them all); positions, main/side pots, every bot's reasoning, and
  optional "Keep stacks" (stacks carry over, busted seats rebuy 10,000).
  Replay also takes 3..8 bots. Built-in multiway bots: Random,
  CallStation, AlwaysRaise; the tiered ones from nutsfinder.bots.multiway
  (PotOdds, Equity, Bayes, CFR) appear automatically once that module
  provides them. Autostart links: #play=call&players=6&seat=2&keep=1 and
  #replay=random,raise,call,call&hands=20&seed=7.

CFRBot needs the trained model (default models\holdem_official\latest.npz);
if it is missing CFRBot is shown as unavailable. The turn E[HS] tables
(models\turn_ehs.npz) are preloaded if present (faster first turn decisions).
The logic lives in ``nutsfinder.table`` (session.py: game state, server.py: HTTP).
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nutsfinder.table.server import serve  # noqa: E402
from nutsfinder.table.session import BotCatalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765; 0 = any free port)")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "holdem_official" / "latest.npz",
                    help="CFRBot checkpoint (CFRBot is unavailable if it is missing)")
    ap.add_argument("--turn-tables", type=Path, default=ROOT / "models" / "turn_ehs.npz",
                    help="exact turn E[HS] tables to preload for CFRBot (optional)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = ap.parse_args()
    if not 0 <= args.port <= 65535:
        ap.error("--port must be 0..65535")

    catalog = BotCatalog(args.model, args.turn_tables)
    if catalog.blueprint is None:
        print(f"CFRBot unavailable: {catalog.cfr_error}")
    else:
        tables = "turn tables preloaded" if catalog.turn_tables_loaded else "no turn tables (slower turns)"
        print(f"CFRBot model: {catalog.model_description}; {tables}")
    if args.open and args.port:
        threading.Timer(0.8, webbrowser.open, (f"http://127.0.0.1:{args.port}/",)).start()
    try:
        serve(catalog, "127.0.0.1", args.port, verbose=args.verbose)
    except OSError as err:
        sys.exit(f"could not listen on 127.0.0.1:{args.port}: {err} (try --port)")


if __name__ == "__main__":
    main()
