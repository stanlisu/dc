"""Agamotto research library for Binance Futures datasets - Compatibility shim.

This module is kept for backward compatibility.
All functionality has been split into:
- utils.py: Utility functions
- research.py: AgamottoResearch class
- trading.py: AgamottoTrading class
- telegram.py: Telegram functions
"""

from __future__ import annotations

# Import everything from split modules for backward compatibility
from .utils import _symbol_to_native, _timeframe_to_seconds, _round_down_to_timeframe_boundary
from .research import AgamottoResearch
from .trading import AgamottoTrading
from .telegram import send_telegram_message, format_decisions_for_telegram

__all__ = [
    "_symbol_to_native",
    "_timeframe_to_seconds",
    "_round_down_to_timeframe_boundary",
    "AgamottoResearch",
    "AgamottoTrading",
    "send_telegram_message",
    "format_decisions_for_telegram",
]
