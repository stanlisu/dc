"""Agamotto research library for Binance Futures datasets."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import joblib
import glob
import numpy as np
import pandas as pd
import re
import os
import sys
import time
import requests
import logging

# Set up logger for Agamotto - ensure it uses root logger's handlers
logger = logging.getLogger(__name__)
# Don't add handlers here - let it propagate to root logger
# This ensures it uses the same handler configuration as the main process

# Use relative import for internal module
try:
    from .lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    # Fallback to assume it's in the path or installed
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass

from typing import Dict

# Telegram Bot Configuration
# To set up:
# 1. Create a bot via @BotFather on Telegram to get bot_token
# 2. Start a chat with your bot
# 3. Get your chat_id by visiting: https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
#
# Set credentials via environment variables or setting.json


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """
    Send message to Telegram using Bot API.
    
    To set up:
    1. Create a bot via @BotFather on Telegram to get bot_token
    2. Start a chat with your bot
    3. Get your chat_id by visiting: https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
    
    Args:
        bot_token: Telegram bot token from @BotFather
        chat_id: Your chat ID (can be found via getUpdates API)
        message: Message text to send
        
    Returns:
        True if successful, False otherwise
    """
    if not bot_token or not chat_id:
        logger.warning("⚠️  Telegram bot_token or chat_id not configured")
        logger.warning("   Set environment variables: AGAMOTTO_BOT_TOKEN and AGAMOTTO_CHAT_ID")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"  # Allows basic formatting
        }
        
        logger.info(f"Sending Telegram message ({len(message)} chars)...")
        response = requests.post(url, data=payload, timeout=10)
        result = response.json()
        
        if result.get("ok"):
            msg_id = result.get("result", {}).get("message_id", "unknown")
            logger.info(f"✓ Telegram message sent successfully (message_id: {msg_id})")
            return True
        else:
            error_desc = result.get('description', 'Unknown error')
            error_code = result.get('error_code', 'Unknown')
            logger.error(f"✗ Telegram API error ({error_code}): {error_desc}")
            logger.error(f"   Full response: {result}")
            return False
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        try:
            result = e.response.json()
            error_desc = result.get('description', 'Unknown error')
            error_code = result.get('error_code', 'Unknown')
            logger.error(f"✗ Telegram HTTP error ({error_code}): {error_desc}")
            logger.error(f"   Full response: {result}")
        except:
            logger.error(f"✗ Telegram HTTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Telegram message failed: {e}")
        logger.error(f"   Message length: {len(message)} chars")
        logger.error(f"   First 200 chars: {message[:200]}")
        return False


def format_decisions_for_telegram(decisions: Dict[str, list[float, float]], version: str, algo_label: str = "Agamotto") -> str:
    """
    Format trading decisions for Telegram message with HTML formatting.
    
    Args:
        decisions: Dict mapping symbol -> [price, qty]
                   - qty > 0 means buy (long position)
                   - qty < 0 means sell (short position)
                   - qty == 0 means close position
        version: Config version name
        
    Returns:
        Formatted string for Telegram (HTML)
    """
    if not decisions:
        return "No trading decisions"
    
    # Group decisions by category (BINANCE_PERP vs others)
    binance_perp_lines = []
    other_lines = []
    
    buy_count = 0
    sell_count = 0
    close_count = 0
    
    for symbol, decision in decisions.items():
        price, qty = decision
        
        # Determine strict symbol display
        if "BINANCE_PERP_" in symbol or "BINANCEFUTURES_PERP_" in symbol:
            # Strip prefix to get e.g. BTC_USDT
            display_symbol = symbol.replace("BINANCEFUTURES_PERP_", "").replace("BINANCE_PERP_", "")
            is_binance_perp = True
        else:
            display_symbol = symbol
            is_binance_perp = False
            
        # Calculate dollar value (integer) — guard against NaN
        import math
        if math.isnan(price) or math.isnan(qty):
            price = 0.0 if math.isnan(price) else price
            qty = 0.0 if math.isnan(qty) else qty
        dollar_value = int(round(abs(qty * price)))
        
        line = ""
        if abs(qty) < 0.0001:
            line = f"⚪ CLOSE {display_symbol}"
            close_count += 1
        elif qty > 0:
            line = f"🟢 <b>BUY</b> {display_symbol}: {dollar_value}$ @ {price}"
            buy_count += 1
        else:  # qty < 0
            line = f"🔴 <b>SELL</b> {display_symbol}: {dollar_value}$ @ {price}"
            sell_count += 1
            
        if is_binance_perp:
            binance_perp_lines.append(line)
        else:
            other_lines.append(line)
    
    # Construct message
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    msg_parts = []
    
    # Header
    msg_parts.append(f"🤖 <b>{algo_label} Decisions</b>")
    msg_parts.append(f"🕐 {timestamp}")
    msg_parts.append(f"Summary: {buy_count} BUY | {sell_count} SELL | {close_count} CLOSE\n")
    
    # BINANCE_PERP Section
    if binance_perp_lines:
        msg_parts.append("<b># BINANCE_PERP</b>")
        msg_parts.extend(binance_perp_lines)
        if other_lines:
            msg_parts.append("") # Spacer
            
    # Others Section
    if other_lines:
        msg_parts.append("<b># OTHERS</b>")
        msg_parts.extend(other_lines)
        
    return "\n".join(msg_parts)

