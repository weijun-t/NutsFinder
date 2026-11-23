"""N-player (2..8 seats) no-limit hold'em engine, separate from the heads-up one.

The heads-up engine (``nutsfinder.engine.betting`` / ``runner`` / ...) is left
untouched; this package reuses its cards, deck, action types, evaluator and
record types, and at two players reproduces it exactly. See ``state.py`` for
the rules and ``runner.py`` for the deal order.
"""

from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.bots import (
    HUBotAdapter,
    MultiAlwaysRaiseBot,
    MultiCallStationBot,
    MultiRandomBot,
)
from nutsfinder.engine.multiway.hand_log import MultiHandLog, format_log, make_log, replay
from nutsfinder.engine.multiway.pots import Pot, side_pots, split_amount, uncalled_bet
from nutsfinder.engine.multiway.positions import blind_seats, position_names
from nutsfinder.engine.multiway.runner import play_hand_multi, play_match_multi
from nutsfinder.engine.multiway.showdown import MultiSettlement, PotAward, settle, showdown_order
from nutsfinder.engine.multiway.state import (
    MultiHandState,
    MultiPublicState,
    apply_action,
    deal_board,
    legal_actions,
    new_hand,
    public_view,
)

__all__ = [
    "MultiBot",
    "MultiHandSummary",
    "HUBotAdapter",
    "MultiAlwaysRaiseBot",
    "MultiCallStationBot",
    "MultiRandomBot",
    "MultiHandLog",
    "format_log",
    "make_log",
    "replay",
    "Pot",
    "side_pots",
    "split_amount",
    "uncalled_bet",
    "blind_seats",
    "position_names",
    "play_hand_multi",
    "play_match_multi",
    "MultiSettlement",
    "PotAward",
    "settle",
    "showdown_order",
    "MultiHandState",
    "MultiPublicState",
    "apply_action",
    "deal_board",
    "legal_actions",
    "new_hand",
    "public_view",
]
