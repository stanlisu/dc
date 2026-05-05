"""
vibranium/carnage.py — Pair position manager for Vibranium.

Carnage tracks 2-leg pair positions: queries exchange state, computes
combined PnL, detects drift (one leg missing), and provides close-pair.

Usage:
    carnage = Carnage(api_key, api_secret)
    carnage.add_pair("DOGE_AVAX", "DOGEUSDT", "AVAXUSDT", hedge_ratio=1.5)
    status = carnage.query_all()
    carnage.close_pair("DOGE_AVAX")
"""
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Dict

import requests

logger = logging.getLogger(__name__)

_PAPI_BASE = "https://papi.binance.com"


def _sign(params: dict, secret: str) -> str:
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


@dataclass
class LegStatus:
    """Exchange state for one leg."""
    symbol: str
    qty: float = 0.0           # positive = long, negative = short
    entry_price: float = 0.0
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PairConfig:
    """Configuration for one pair."""
    pair_id: str
    sym_a: str
    sym_b: str
    hedge_ratio: float = 1.0


@dataclass
class PairStatus:
    """Combined status for a pair."""
    pair_id: str
    leg_a: LegStatus
    leg_b: LegStatus
    combined_upnl: float = 0.0
    is_open: bool = False       # both legs have non-zero qty
    is_drifted: bool = False    # one leg open, other flat (leg risk!)
    drift_leg: str = ""         # which leg is exposed ("a", "b", or "")

    def summary(self) -> str:
        """One-line summary for monitoring."""
        flag = " DRIFT!" if self.is_drifted else ""
        a = self.leg_a
        b = self.leg_b
        return (
            f"{self.pair_id}: "
            f"A={a.symbol} {a.qty:+.4f}@{a.mark_price:.2f} "
            f"B={b.symbol} {b.qty:+.4f}@{b.mark_price:.2f} "
            f"uPnL={self.combined_upnl:+.4f}{flag}"
        )


@dataclass
class ReconcileResult:
    """Result from reconcile(): what doesn't match between exchange and Carnage."""
    orphans: Dict[str, float]       # {symbol: qty} — on exchange, not in any pair
    drifted: Dict[str, str]         # {pair_id: drift_leg} — one leg open, other flat
    stale: Dict[str, dict]          # {pair_id: {"fsm": state, "exchange": state}}


class Carnage:
    """
    Pair position manager. Queries Binance PAPI for position state,
    computes combined PnL, detects drift, reconciles with FSM state.

    Enforces no shared symbols across pairs to enable unambiguous reconciliation.
    """

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = _PAPI_BASE, timeout: int = 10):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.timeout = timeout
        self._pairs: Dict[str, PairConfig] = {}
        self._claimed_symbols: Dict[str, str] = {}  # symbol → pair_id

    def add_pair(self, pair_id: str, sym_a: str, sym_b: str,
                 hedge_ratio: float = 1.0):
        """Register a pair for tracking. Raises if symbols are already claimed."""
        for sym in (sym_a, sym_b):
            if sym in self._claimed_symbols:
                owner = self._claimed_symbols[sym]
                raise ValueError(
                    f"Symbol {sym} already claimed by pair {owner}. "
                    f"Cannot use in {pair_id}. "
                    f"No shared symbols allowed across pairs.")
        self._pairs[pair_id] = PairConfig(
            pair_id=pair_id, sym_a=sym_a, sym_b=sym_b,
            hedge_ratio=hedge_ratio)
        self._claimed_symbols[sym_a] = pair_id
        self._claimed_symbols[sym_b] = pair_id

    def _get(self, path: str, params: dict = None) -> dict:
        """Authenticated PAPI GET request."""
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        signed = _sign(params, self.api_secret)
        resp = requests.get(
            f"{self.base_url}{path}?{signed}",
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"PAPI {path} failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    def _delete(self, path: str, params: dict = None) -> dict:
        """Authenticated PAPI DELETE request."""
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        signed = _sign(params, self.api_secret)
        resp = requests.delete(
            f"{self.base_url}{path}?{signed}",
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=self.timeout)
        return resp.json() if resp.text else {}

    def _post(self, path: str, params: dict) -> dict:
        """Authenticated PAPI POST request."""
        params["timestamp"] = int(time.time() * 1000)
        signed = _sign(params, self.api_secret)
        resp = requests.post(
            f"{self.base_url}{path}?{signed}",
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"PAPI {path} failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    def _query_positions(self) -> Dict[str, dict]:
        """Query all UM positions from PAPI. Returns {symbol: position_dict}."""
        data = self._get("/papi/v1/um/account")
        result = {}
        for p in data.get("positions", []):
            sym = p["symbol"]
            qty = float(p.get("positionAmt", 0))
            result[sym] = {
                "qty": qty,
                "entryPrice": float(p.get("entryPrice", 0)),
                "markPrice": float(p.get("markPrice", 0)),
                "unrealizedProfit": float(p.get("unrealizedProfit", 0)),
            }
        return result

    def query_pair(self, pair_id: str,
                   positions: Dict[str, dict] = None) -> PairStatus:
        """
        Query exchange state for one pair.

        Parameters
        ----------
        pair_id : str
            Registered pair ID.
        positions : dict or None
            Pre-fetched positions (from _query_positions). If None, queries live.
        """
        cfg = self._pairs[pair_id]
        if positions is None:
            positions = self._query_positions()

        def _leg(sym):
            p = positions.get(sym, {})
            return LegStatus(
                symbol=sym,
                qty=p.get("qty", 0.0),
                entry_price=p.get("entryPrice", 0.0),
                mark_price=p.get("markPrice", 0.0),
                unrealized_pnl=p.get("unrealizedProfit", 0.0),
            )

        leg_a = _leg(cfg.sym_a)
        leg_b = _leg(cfg.sym_b)

        a_open = abs(leg_a.qty) > 1e-8
        b_open = abs(leg_b.qty) > 1e-8

        is_open = a_open and b_open
        is_drifted = a_open != b_open
        drift_leg = ""
        if is_drifted:
            drift_leg = "a" if a_open else "b"

        return PairStatus(
            pair_id=pair_id,
            leg_a=leg_a,
            leg_b=leg_b,
            combined_upnl=leg_a.unrealized_pnl + leg_b.unrealized_pnl,
            is_open=is_open,
            is_drifted=is_drifted,
            drift_leg=drift_leg,
        )

    def query_all(self) -> list:
        """Query all registered pairs. Single API call for all positions."""
        positions = self._query_positions()
        return [self.query_pair(pid, positions) for pid in self._pairs]

    def close_pair(self, pair_id: str) -> dict:
        """
        Close both legs of a pair. Cancel open orders first, then market close.

        Returns dict with results per leg.
        """
        cfg = self._pairs[pair_id]
        positions = self._query_positions()
        results = {}

        for sym in [cfg.sym_a, cfg.sym_b]:
            p = positions.get(sym, {})
            qty = p.get("qty", 0.0)
            if abs(qty) < 1e-8:
                results[sym] = {"status": "already_flat"}
                continue

            # Cancel open orders
            try:
                self._delete("/papi/v1/um/allOpenOrders",
                             {"symbol": sym})
            except Exception as exc:
                logger.warning("Cancel orders %s: %s", sym, exc)

            # Close with limit at mark price
            side = "SELL" if qty > 0 else "BUY"
            close_qty = abs(qty)
            mark = p.get("markPrice", 0)
            if mark <= 0:
                results[sym] = {"status": "no_mark_price", "qty": qty}
                continue

            try:
                order = self._post("/papi/v1/um/order", {
                    "symbol": sym,
                    "side": side,
                    "type": "LIMIT",
                    "quantity": f"{close_qty:.8f}".rstrip("0").rstrip("."),
                    "price": f"{mark:.8f}".rstrip("0").rstrip("."),
                    "timeInForce": "GTC",
                    "reduceOnly": "true",
                })
                results[sym] = {
                    "status": "closing",
                    "orderId": order.get("orderId"),
                    "qty": close_qty,
                    "side": side,
                }
                logger.info("Closing %s: %s %.4f @ %.2f",
                            sym, side, close_qty, mark)
            except Exception as exc:
                results[sym] = {"status": "error", "error": str(exc)}
                logger.error("Close %s failed: %s", sym, exc)

        return results

    def close_all(self) -> dict:
        """Close all registered pairs."""
        results = {}
        for pair_id in self._pairs:
            results[pair_id] = self.close_pair(pair_id)
        return results

    def print_status(self, statuses: list = None):
        """Print status table for all pairs."""
        if statuses is None:
            statuses = self.query_all()

        print("-" * 80)
        print("CARNAGE — Vibranium Pair Position Manager")
        print("-" * 80)

        for s in statuses:
            print(s.summary())

        open_pairs = sum(1 for s in statuses if s.is_open)
        drifted = sum(1 for s in statuses if s.is_drifted)
        total_upnl = sum(s.combined_upnl for s in statuses)

        print("-" * 80)
        print(f"Open: {open_pairs}/{len(statuses)}  "
              f"Drifted: {drifted}  "
              f"Total uPnL: {total_upnl:+.4f}")
        print("-" * 80)

    def reconcile(self, positions: Dict[str, dict] = None,
                  fsm_states: dict = None) -> ReconcileResult:
        """
        Reconcile exchange positions against registered pairs and FSM state.

        Parameters
        ----------
        positions : dict or None
            Pre-fetched positions {symbol: {qty, ...}}. If None, queries live.
        fsm_states : dict or None
            {pair_id: PairPosition} from PairExecutor. If None, skips FSM check.

        Returns
        -------
        ReconcileResult with:
            orphans — exchange positions not claimed by any pair
            drifted — pairs with one leg open and other flat
            stale — pairs where FSM state doesn't match exchange reality
        """
        if positions is None:
            positions = self._query_positions()

        # All symbols claimed by registered pairs
        all_claimed = set(self._claimed_symbols.keys())

        # 1. Orphans: exchange positions not in any pair
        orphans = {}
        for sym, p in positions.items():
            qty = p.get("qty", 0.0)
            if abs(qty) > 1e-8 and sym not in all_claimed:
                orphans[sym] = qty

        # 2. Drifted: one leg open, other flat
        drifted = {}
        for pair_id, cfg in self._pairs.items():
            qty_a = positions.get(cfg.sym_a, {}).get("qty", 0.0)
            qty_b = positions.get(cfg.sym_b, {}).get("qty", 0.0)
            a_open = abs(qty_a) > 1e-8
            b_open = abs(qty_b) > 1e-8
            if a_open != b_open:
                drifted[pair_id] = "a" if a_open else "b"

        # 3. Stale: FSM state vs exchange reality
        stale = {}
        if fsm_states:
            for pair_id, fsm in fsm_states.items():
                if pair_id not in self._pairs:
                    continue
                cfg = self._pairs[pair_id]
                qty_a = positions.get(cfg.sym_a, {}).get("qty", 0.0)
                qty_b = positions.get(cfg.sym_b, {}).get("qty", 0.0)
                a_open = abs(qty_a) > 1e-8
                b_open = abs(qty_b) > 1e-8

                # Determine exchange state
                if a_open and b_open:
                    exchange_state = "open"
                elif a_open or b_open:
                    exchange_state = "drifted"
                else:
                    exchange_state = "flat"

                # Determine FSM state category
                fsm_state = fsm.state.value
                if fsm_state == "BOTH_OPEN" and exchange_state != "open":
                    stale[pair_id] = {"fsm": fsm_state,
                                      "exchange": exchange_state}
                elif fsm_state == "FLAT" and exchange_state != "flat":
                    stale[pair_id] = {"fsm": fsm_state,
                                      "exchange": exchange_state}

        return ReconcileResult(
            orphans=orphans, drifted=drifted, stale=stale)
