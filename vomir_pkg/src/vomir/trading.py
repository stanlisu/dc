"""
vomir/trading.py — VomirTrading: live inference using cluster-then-classify.

Inherits AgamottoTrading for kline fetching, feature engineering, and position
sizing. Replaces regime-stack prediction with:

    impute → StandardScaler → PCA → KMeans → [scaled_features | cluster_id]
    → LGBMClassifier(3 class: SHORT=0 / NEUTRAL=1 / LONG=2)

P(LONG) > VOMIR_LONG_THRESHOLD  → decisions[sym] = [price,  +size]
P(SHORT) > VOMIR_SHORT_THRESHOLD → decisions[sym] = [price, -size]
else                             → decisions[sym] = [0.0, 0.0]
"""

from __future__ import annotations

import logging
import math
import os
from decimal import Decimal
from typing import Dict, Optional

import numpy as np
import pandas as pd

from agamotto.research import AgamottoResearch, _obf
from agamotto.trading import AgamottoTrading
from vomir.classifier import VomirClassifier, CLASS_LONG, CLASS_SHORT

try:
    from agamotto.utils import _symbol_to_native
except ImportError:
    from agamotto import _symbol_to_native  # type: ignore

logger = logging.getLogger(__name__)


class VomirTrading(AgamottoTrading):
    """Live inference using the Vomir global cluster-classify model.

    Replaces AgamottoTrading's regime-stack logic with a single
    LGBMClassifier that receives [scaled_features | cluster_id] as input.
    All kline fetching, freshness checks, and feature engineering are
    inherited unchanged from AgamottoTrading / AgamottoResearch.
    """

    def __init__(
        self,
        config: Dict,
        home_root: str,
        period: Optional[str] = None,
        skip_load: bool = False,
    ) -> None:
        # Call AgamottoResearch.__init__ directly — bypasses AgamottoTrading.__init__
        # which requires REGIME_STACK_PATH.
        AgamottoResearch.__init__(self, config, home_root)

        self.period = config.get("WEIGHTS_PERIOD") or period
        self.models: dict = {}
        self.latest_predictions = None
        self.latest_predictions_path = None
        self.decisions: dict = {}
        self.regime_stack: list = []  # unused — keeps AgamottoTrading compat

        self.long_pred_threshold = float(config.get("VOMIR_LONG_THRESHOLD", 0.4))
        self.short_pred_threshold = float(config.get("VOMIR_SHORT_THRESHOLD", 0.4))

        self._load_vomir_model()
        self._calculate_sizes()

        if not skip_load:
            try:
                self.load_data(limit=500)
            except Exception as e:
                logger.warning("Failed to load initial data in __init__: %s", e)

        logger.info(
            "VomirTrading initialized — period: %s  tf: %s  "
            "long_thresh: %.3f  short_thresh: %.3f",
            self.period, self.config.get("TIME_UNIT"),
            self.long_pred_threshold, self.short_pred_threshold,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_vomir_model(self) -> None:
        weights_path = self.config.get("VOMIR_WEIGHTS_PATH")
        if not weights_path:
            raise ValueError("VOMIR_WEIGHTS_PATH not set in config")
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.home_root, weights_path)

        logger.info("Loading Vomir model from %s", weights_path)
        artifacts = VomirClassifier.load_model_artifacts(weights_path)

        self._vomir_imputer = artifacts["imputer"]
        self._vomir_scaler = artifacts["scaler"]
        self._vomir_pca = artifacts["pca"]
        self._vomir_kmeans = artifacts["kmeans"]
        self._vomir_clf = artifacts["clf"]
        self._vomir_meta = artifacts["meta"]
        self._vomir_feature_cols = artifacts["meta"]["feature_cols"]

        logger.info(
            "Vomir model loaded — n_features: %d  n_clusters: %d  pca_components: %d",
            len(self._vomir_feature_cols),
            artifacts["meta"]["n_clusters"],
            artifacts["meta"]["pca_components"],
        )

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def make_decision(self, label: str = "both") -> dict:
        """Generate trading decisions using cluster-then-classify.

        Returns
        -------
        dict
            {symbol: [price, qty]}
            qty > 0 = LONG, qty < 0 = SHORT, qty = 0 = NEUTRAL / stale
        """
        self.decisions = {sym: [0.0, 0.0] for sym in self.config.get("SYMBOLS", [])}

        if not getattr(self, "_data_fresh", True):
            logger.warning("Data not fresh — returning CLOSE (all zeros) for all symbols.")
            return self.decisions

        vf = getattr(self, "vertical_features", None)
        if vf is None or vf.empty:
            logger.warning("No vertical_features available.")
            return self.decisions

        target_ts = vf["timestamp"].max()
        latest = vf[vf["timestamp"] == target_ts].copy().reset_index(drop=True)
        # Obfuscation: add coded aliases so the feature_cols selection + scaler
        # name-check match either coded (new) or real (old) weights.
        latest = _obf().add_feature_aliases(latest)
        if latest.empty:
            logger.warning("No rows at target_ts %s", target_ts)
            return self.decisions

        # Check feature alignment
        missing = [c for c in self._vomir_feature_cols if c not in latest.columns]
        if missing:
            logger.error(
                "Missing %d feature cols at inference: %s ...",
                len(missing), missing[:5],
            )
            return self.decisions

        X_raw = latest[self._vomir_feature_cols].values.astype(np.float64)
        np.nan_to_num(X_raw, nan=0.0, copy=False)

        # impute → scale → PCA → KMeans → augment with cluster_id
        X_imp = self._vomir_imputer.transform(X_raw)
        X_sc = self._vomir_scaler.transform(X_imp)
        X_pca = self._vomir_pca.transform(X_sc)
        cluster_ids = self._vomir_kmeans.predict(X_pca).reshape(-1, 1).astype(np.float64)
        X_aug = np.hstack([X_sc, cluster_ids])

        # Classify — map clf.classes_ → [SHORT=0, NEUTRAL=1, LONG=2]
        raw_proba = self._vomir_clf.predict_proba(X_aug)
        class_order = list(self._vomir_clf.classes_)
        prob_long = (
            raw_proba[:, class_order.index(CLASS_LONG)]
            if CLASS_LONG in class_order
            else np.zeros(len(latest))
        )
        prob_short = (
            raw_proba[:, class_order.index(CLASS_SHORT)]
            if CLASS_SHORT in class_order
            else np.zeros(len(latest))
        )

        latest_closes = self._get_latest_closes()
        capital = self.config.get("CAPITAL", 100)
        lot_sizes = self.config.get("LOT_SIZES", {})

        for i, sym in enumerate(latest["symbol"].tolist()):
            if sym not in self.decisions:
                continue

            p_long = float(prob_long[i])
            p_short = float(prob_short[i])
            price = latest_closes.get(sym, 0.0)
            size = self._compute_size(sym, price, capital, lot_sizes)

            # DIRECTION IS THE CLASSIFIER'S ANSWER. There is no config gate on
            # it -- the same contract every other algo has, where direction
            # comes from the regime stack's `position` column. The gate that
            # used to stand here read the COLLIDING TRADING_MODE key, which
            # knull's BaseExecutor reads as an execution style; neither side
            # rejected the other's vocabulary, so "MarketMaker" in that key
            # matched no branch and vomir traded nothing, silently, on every
            # one of 1323 measured cells of the decision surface. See
            # vomir_pkg/tests/test_trading_decisions.py for the measurement.
            if p_long > self.long_pred_threshold:
                # If both exceed threshold, take the stronger signal
                if p_short <= self.short_pred_threshold or p_long >= p_short:
                    self.decisions[sym] = [price, size]
                    logger.info(
                        "LONG  %s — P(L)=%.3f  P(S)=%.3f  size=%s",
                        sym, p_long, p_short, size,
                    )
            elif p_short > self.short_pred_threshold:
                self.decisions[sym] = [price, -size]
                logger.info(
                    "SHORT %s — P(L)=%.3f  P(S)=%.3f  size=%s",
                    sym, p_long, p_short, size,
                )

        return self.decisions

    def clean(self) -> dict:
        self.decisions = {sym: [0.0, 0.0] for sym in self.config.get("SYMBOLS", [])}
        return self.decisions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_latest_closes(self) -> Dict[str, float]:
        closes: Dict[str, float] = {}
        raw = getattr(self, "raw", None)
        if raw is None or raw.empty:
            return closes
        for sym in self.config.get("SYMBOLS", []):
            native = _symbol_to_native(sym)
            col = f"{native}_close"
            if col in raw.columns:
                val = raw[col].iloc[-1] if len(raw) >= 1 else 0.0
                if pd.isna(val):
                    valid = raw[col].dropna()
                    val = float(valid.iloc[-1]) if len(valid) > 0 else 0.0
                closes[sym] = float(val)
            else:
                closes[sym] = 0.0
        return closes

    def _compute_size(
        self,
        sym: str,
        price: float,
        capital: float,
        lot_sizes: dict,
    ) -> float:
        lot_info = lot_sizes.get(sym, {})
        step = lot_info.get("step_size", 0)
        if price > 0 and step > 0:
            qty = capital / price
            prec = max(0, int(round(-Decimal(str(step)).log10()))) if step < 1 else 0
            qty = round(qty / step) * step
            qty = round(qty, prec)
            min_notional = lot_info.get("min_notional", 0.0)
            if min_notional > 0 and qty * price < min_notional:
                qty = math.ceil(min_notional / price / step) * step
                qty = round(qty, prec)
            return float(qty)
        # Fallback to init-time size
        symbols = self.config.get("SYMBOLS", [])
        if sym in symbols:
            idx = symbols.index(sym)
            init_sizes = self.config.get("SIZES", [])
            if idx < len(init_sizes):
                return float(init_sizes[idx])
        return 0.0
