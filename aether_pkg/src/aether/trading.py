"""Aether trading: pooled cross-TF live inference."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .research import AetherResearch
from agamotto.research import _obf
# ONE implementation, in `agamotto/trading.py` — see the rationale block in
# `orb/trading.py` for why this is imported and not copied.
#
# WHY AETHER NEEDS IT (verified 2026-08-17, not inherited by assumption).
# `AetherTrading` extends `AetherResearch` -> `OrbResearch`, NOT `AgamottoTrading`
# and NOT `OrbTrading`, so neither fix reached it: it carried its own
# `self.raw[col].iloc[-2]` until this commit. `dc f47d588` left it explicitly
# unverified ("aether has no upstream drop, so its iloc[-2] may be correct").
# It does have an upstream drop — aether's `load_data` delegates straight to
# `OrbTrading.load_data(self, ...)` (:144-147), so `self.raw` is orb's, assigned
# only downstream of orb's in-flight-bar drop. And `AetherResearch` overrides only
# `__init__`, `create` and `train_pooled_models`, so `engineer_features`,
# `verticalize` and `_align_timeframes` are OrbResearch's verbatim, putting
# `vertical_features["timestamp"].max() == self.raw.index.max()`. Hence
# `iloc[-2]` was a full BASE_TF stale, exactly as in agamotto and orb.
from agamotto.trading import _closes_at_timestamp

logger = logging.getLogger(__name__)


class AetherTrading(AetherResearch):
    """Live pooled cross-TF inference.

    Key difference vs OrbTrading:
    - Loads ONE model per position (long/short) instead of one per regime
    - make_decision() does 2 forward passes then applies regime masks
    """

    def __init__(
        self,
        config: Dict[str, object],
        home_root: str,
        period: Optional[str] = None,
        skip_load: bool = False,
    ) -> None:
        super().__init__(config, home_root)
        self.period = self.config.get("WEIGHTS_PERIOD") or period
        self.trading_mode = self.config.get("TRADING_MODE", "both")
        self.models: Dict[str, dict] = {}
        self.regime_stack: List[dict] = []
        self.decisions: Dict[str, list] = {}
        self._data_fresh = False

        # Load regime stack from CSV
        stack_path = self.config.get("REGIME_STACK_PATH")
        if not stack_path:
            raise ValueError("REGIME_STACK_PATH not provided in configuration.")
        if not os.path.isabs(stack_path):
            stack_path = os.path.join(home_root, stack_path)
        if not os.path.exists(stack_path):
            raise FileNotFoundError(f"REGIME_STACK_PATH not found: {stack_path}")
        self._load_regime_stack(stack_path)

        # Load pooled model artifacts
        self._load_models()

        # Calculate position sizes (reuse OrbTrading logic via parent)
        from orb.trading import OrbTrading
        OrbTrading._calculate_sizes(self)

        if not skip_load:
            try:
                self.load_data(limit=700)
            except Exception as e:
                logger.warning(
                    f"Failed to load initial data in __init__: {e}")

        logger.info(
            f"AetherTrading initiated — period: {self.period}, "
            f"base_tf: {self.base_tf}, "
            f"regimes: {len(self.regime_stack)}, "
            f"models: {list(self.models.keys())}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_regime_stack(self, stack_path: str) -> None:
        """Load regime stack CSV into self.regime_stack.

        CSV columns: regime, position, optimal_threshold
        No model weights per regime — threshold is the only regime artifact.
        """
        import csv as _csv
        with open(stack_path, newline="") as f:
            rows = list(_csv.DictReader(f))

        self.regime_stack = []
        for row in rows:
            regime = row.get("regime", "")
            if not regime or regime.startswith("__"):
                continue
            position = row.get("position", "long")
            threshold = float(row.get("optimal_threshold", 0.0))
            self.regime_stack.append({
                "regime": regime,
                "position": position,
                "threshold": threshold,
            })
        logger.info(
            f"Loaded {len(self.regime_stack)} regime entries from {stack_path}")

    def _load_models(self) -> None:
        """Load one model artifact per position (long + short)."""
        weights_path = self.config.get("WEIGHTS_PATH")
        if not weights_path:
            raise ValueError("WEIGHTS_PATH not provided in configuration.")
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.home_root, weights_path)

        for position in ["long", "short"]:
            key = f"{self.base_tf}_{position}"
            pos_dir = os.path.join(weights_path, key)
            if not os.path.isdir(pos_dir):
                raise FileNotFoundError(
                    f"Model directory not found for {key}: {pos_dir}")

            model_files = [
                f for f in os.listdir(pos_dir) if f.endswith("_model.pkl")]
            if not model_files:
                raise FileNotFoundError(
                    f"No *_model.pkl files found in {pos_dir}")

            low = model_files[0].replace("_model.pkl", "")
            model  = joblib.load(os.path.join(pos_dir, f"{low}_model.pkl"))
            scaler = joblib.load(os.path.join(pos_dir, f"{low}_scaler.pkl"))
            meta   = joblib.load(os.path.join(pos_dir, f"{low}_meta.pkl"))

            self.models[position] = {
                "model": model,
                "scaler": scaler,
                "feature_columns": meta["feature_columns"],
            }
            logger.info(
                f"Loaded {low} model for {key} "
                f"({len(meta['feature_columns'])} features)")

    # ------------------------------------------------------------------
    # Data loading — delegate entirely to OrbTrading
    # ------------------------------------------------------------------

    def load_data(self, limit: int = 700) -> None:
        """Delegate to OrbTrading.load_data() — identical multi-TF fetch."""
        from orb.trading import OrbTrading
        OrbTrading.load_data(self, limit=limit)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def make_decision(self, label: str = "both") -> Dict[str, list]:
        self.decisions = {
            sym: [0.0, 0.0] for sym in self.config.get("SYMBOLS", [])}

        if not getattr(self, "_data_fresh", True):
            logger.warning(
                "Data not fresh — returning CLOSE (all zeros) for all symbols.")
            return self.decisions

        if self.features is None:
            self.engineer_features()
        self.verticalize()

        if self.vertical_features is None or self.vertical_features.empty:
            logger.error("vertical_features is empty — cannot make decision")
            return self.decisions

        # ---- Step 1: one forward pass per position ----
        target_ts = self.vertical_features["timestamp"].max()
        latest = self.vertical_features[
            self.vertical_features["timestamp"] == target_ts].copy()
        # Obfuscation: add coded aliases so the feature_columns selection + scaler
        # name-check match either coded (new) or real (old) weights.
        latest = _obf().add_feature_aliases(latest)

        y_pred: Dict[str, Dict[str, float]] = {"long": {}, "short": {}}
        for position in ["long", "short"]:
            if position not in self.models:
                continue
            artifact = self.models[position]
            f_cols = artifact["feature_columns"]
            missing = [c for c in f_cols if c not in latest.columns]
            if missing:
                logger.warning(
                    f"Missing {len(missing)} features for {position} model")
                fill = pd.DataFrame(
                    0.0, index=latest.index,
                    columns=[c for c in f_cols if c not in latest.columns])
                latest = pd.concat([latest, fill], axis=1)
            X = latest[f_cols].replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
            X_scaled = artifact["scaler"].transform(X)
            preds = artifact["model"].predict(X_scaled)
            for i, row in enumerate(latest.itertuples(index=False)):
                sym = getattr(row, "symbol", None)
                if sym:
                    y_pred[position][sym] = float(preds[i])

        # ---- Step 2: regime voting ----
        long_votes: Dict[str, int] = {
            s: 0 for s in self.config.get("SYMBOLS", [])}
        short_votes: Dict[str, int] = {
            s: 0 for s in self.config.get("SYMBOLS", [])}

        for regime in self.regime_stack:
            try:
                sig = self.filter_signals(regime, save=False)
            except Exception as e:
                logger.error(
                    f"filter_signals error for {regime['regime']}: {e}")
                continue
            if sig is None or sig.empty:
                continue

            active_syms = set(
                sig[sig["timestamp"] == target_ts]["symbol"].tolist())
            threshold = regime["threshold"]
            position = regime["position"]

            for sym in self.config.get("SYMBOLS", []):
                if sym not in active_syms:
                    continue
                pred = y_pred.get(position, {}).get(sym)
                if pred is None:
                    continue
                if position == "long" and pred > threshold:
                    long_votes[sym] += 1
                elif position == "short" and pred < threshold:
                    short_votes[sym] += 1

        # ---- Step 3: net position per symbol ----
        from decimal import Decimal
        capital = self.config.get("CAPITAL", 100)
        lot_sizes = self.config.get("LOT_SIZES", {})
        sizes = self.config.get("SIZES", [])
        symbols = self.config.get("SYMBOLS", [])

        # Close prices for sizing AND for the price the executor anchors on.
        # Taken from the SAME row the forward passes above ran on — `target_ts`,
        # `vertical_features["timestamp"].max()` — selected by LABEL. See
        # `agamotto.trading._closes_at_timestamp` for why this must never be
        # positional again.
        latest_closes: Dict[str, float] = _closes_at_timestamp(
            getattr(self, "raw", None), symbols, target_ts=target_ts)

        for sym in symbols:
            net = long_votes.get(sym, 0) - short_votes.get(sym, 0)
            close = latest_closes.get(sym, 0.0)
            lot_info = lot_sizes.get(sym, {})
            step = lot_info.get("step_size", 0)
            if close > 0 and step > 0:
                qty = capital / close
                prec = max(
                    0, int(round(-Decimal(str(step)).log10()))) if step < 1 else 0
                qty = round(round(qty / step) * step, prec)
                base_size = qty
            else:
                idx = symbols.index(sym)
                base_size = sizes[idx] if idx < len(sizes) else 0.0
            price = close
            self.decisions[sym] = [price, float(base_size) * net]

            if net != 0:
                logger.info(
                    f"Decision for {sym}: long_votes={long_votes[sym]} "
                    f"short_votes={short_votes[sym]} → net={net}")

        return self.decisions

    def clean(self) -> Dict[str, list]:
        symbols = self.config.get("SYMBOLS", [])
        self.decisions = {sym: [0.0, 0.0] for sym in symbols}
        return self.decisions
