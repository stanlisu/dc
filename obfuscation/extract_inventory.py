#!/usr/bin/env python3
"""Authoritative inventory of regime ATOMS and FEATURE columns across dc/*_pkg.

Two distinct namespaces are obfuscated separately:
  * regime atoms  — strings compared against a filter/name/regime variable in the
                    *_apply_filter_mask* / regime-filter if-chains, plus members of
                    the *_ONLY / *_FILTERS / *_BOTH frozensets.
  * feature cols  — subscript-assignment targets (df["x"] = ...) inside the feature
                    engineering code, i.e. the columns that land in parquet/meta.pkl.

AST-based (no runtime / TA-Lib needed). Emits JSON to stdout:
    {"regime_atoms": [...], "feature_cols": [...], "provenance": {...}}

Run:
    cd ~/Documents/sandbox/dc
    PYTHONPATH=$(ls -d *_pkg/src | paste -sd: -) python obfuscation/extract_inventory.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

DC_ROOT = Path(__file__).resolve().parent.parent

# Variables whose `== "x"` / `in {...}` comparisons name a regime atom.
_FILTER_VARS = {"filter_name", "name", "regime", "regime_name", "f", "base"}
# Frozenset/set names whose string members are regime atoms.
_ATOM_SET_SUFFIXES = ("_FILTERS", "_ONLY", "_BOTH", "_REGIMES")
# List/set names whose string members are ML FEATURE columns (obfuscate these).
_FEATURE_LIST_NAMES = ("_DERIVED_FEATURES",)
_FEATURE_LIST_SUFFIXES = ("_FEATURES",)
# List/set names that are PASSTHROUGH (exchange-standard / metadata) — never
# obfuscated, and subtracted from the feature set even if also assigned via df[..].
_PASSTHROUGH_LIST_SUFFIXES = ("_RAW_COLUMNS", "_RETURN_COLUMNS", "_META_COLS",
                              "_COLUMNS", "_RAW_COLS", "_META")
# DataFrame-like receivers for feature-column assignments.
_DF_VARS = {"df", "features_df", "out", "feats", "features", "X", "result", "data"}

# Never atoms: positions, banned regimes, boolean/structural tokens, composites.
_NOT_ATOMS = {
    "long", "short", "both", "baseline", "trend_aligned",
    "combined_union", "combined_intersection", "combined_union_plus",
    "combined_union_alt", "&", "|", "_and_", "_or_",
}


def _str_const(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class Visitor(ast.NodeVisitor):
    def __init__(self, relpath: str):
        self.relpath = relpath
        self.atoms: dict[str, str] = {}        # atom -> "file:line"
        self.feats: dict[str, str] = {}        # feature -> "file:line"
        self.passthrough: set[str] = set()     # explicit non-IP / standard columns

    def _add_atom(self, s: str, line: int):
        if s and s not in _NOT_ATOMS and "_and_" not in s and not s.startswith("combined"):
            self.atoms.setdefault(s, f"{self.relpath}:{line}")

    # filter_name == "x"  /  filter_name in ("a", "b")
    def visit_Compare(self, node: ast.Compare):
        left = node.left
        if isinstance(left, ast.Name) and left.id in _FILTER_VARS:
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq,)):
                    s = _str_const(comp)
                    if s is not None:
                        self._add_atom(s, node.lineno)
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                    for e in comp.elts:
                        s = _str_const(e)
                        if s is not None:
                            self._add_atom(s, node.lineno)
        self.generic_visit(node)

    # _LONG_ONLY = frozenset({"a", "b"}) / _BASE_REGIMES = ["a", ...]
    def visit_Assign(self, node: ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        upper = [n.upper() for n in names]
        if any(n.endswith(_ATOM_SET_SUFFIXES) for n in upper):
            for s in self._collect_str_members(node.value):
                self._add_atom(s, node.lineno)
        # named ML feature lists (e.g. _DERIVED_FEATURES)
        if any(n in _FEATURE_LIST_NAMES or n.endswith(_FEATURE_LIST_SUFFIXES) for n in names):
            for s in self._collect_str_members(node.value):
                self.feats.setdefault(s, f"{self.relpath}:{node.lineno}")
        # explicit passthrough lists (raw OHLCV / returns / metadata) — never obfuscate
        if any(n.endswith(_PASSTHROUGH_LIST_SUFFIXES) for n in upper):
            for s in self._collect_str_members(node.value):
                self.passthrough.add(s)
        # feature-column assignment: df["x"] = ...  (single target subscript)
        for t in node.targets:
            self._maybe_feature_target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._maybe_feature_target(node.target)
        self.generic_visit(node)

    def _maybe_feature_target(self, t: ast.AST):
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in _DF_VARS:
            s = _str_const(t.slice)
            if s is not None:
                self.feats.setdefault(s, f"{self.relpath}:{getattr(t, 'lineno', 0)}")

    @staticmethod
    def _collect_str_members(value: ast.AST):
        out = []
        targets = []
        if isinstance(value, ast.Call):
            for a in value.args:
                targets.append(a)
        else:
            targets.append(value)
        for tgt in targets:
            if isinstance(tgt, (ast.Set, ast.List, ast.Tuple)):
                for e in tgt.elts:
                    s = _str_const(e)
                    if s is not None:
                        out.append(s)
        return out


# Regime atoms defined MARVEL-side (not in dc) — scepter BTC anchor states.
# These are composed into "{own}_and_{btc}" regimes in marvel's generator, so
# they must be in the map even though dc never compares against them directly.
_MARVEL_ATOMS = ("btc_trending_up", "btc_trending_down", "btc_high_vol", "btc_low_vol")

# Hard passthrough: exchange-standard / time / metadata columns that are never IP.
# Subtracted in addition to any code-declared *_RAW_COLUMNS / *_META_COLS lists.
_HARD_PASSTHROUGH = {
    "open", "high", "low", "close", "volume", "timestamp", "symbol",
    "quote_volume", "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "regime", "position", "position_type",
    "year", "month", "bar", "type", "asset", "amount", "price", "expiration_dt",
    "mvg1", "mvg2", "mvg3", "bid_price", "ask_price", "close_timestamp",
}


def main():
    atoms: dict[str, str] = {}
    feats: dict[str, str] = {}
    passthrough: set[str] = set()
    for py in sorted(DC_ROOT.glob("*_pkg/src/**/*.py")):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(DC_ROOT))
        try:
            tree = ast.parse(py.read_text(), filename=rel)
        except SyntaxError as e:
            print(f"# parse error {rel}: {e}", file=sys.stderr)
            continue
        v = Visitor(rel)
        v.visit(tree)
        for k, prov in v.atoms.items():
            atoms.setdefault(k, prov)
        for k, prov in v.feats.items():
            feats.setdefault(k, prov)
        passthrough |= v.passthrough

    for a in _MARVEL_ATOMS:
        atoms.setdefault(a, "marvel/gauntlet/generate_scepter_regimes.py")

    # Options (valkyrie) feature columns are IP per user decision (2026-06-30):
    # promote greeks / implied-vol / ATM option+straddle price columns out of
    # passthrough into the obfuscated feature set. Targets (ret*/return*) and
    # generic temporal/market refs (strike/spot/open_interest/date) stay passthrough.
    _OPTIONS_PROMOTE = {
        "delta", "gamma", "theta", "vega",
        "mark_iv", "bid_iv", "ask_iv",
        "atm_straddle_price_open", "atm_straddle_price_high", "atm_straddle_price_close",
        "atm_call_price_open", "atm_call_price_high", "atm_call_price_close",
        "atm_put_price_open", "atm_put_price_high", "atm_put_price_close",
    }
    for c in _OPTIONS_PROMOTE:
        feats.setdefault(c, "valkyrie (options IP, promoted)")

    # Tick (mjolnir) IP feature bases that the AST misses because they are
    # produced via f-string targets (depth_imbalance_L{n}) or only appear as
    # regime-filter atoms (trade_imbalance). The rolling-stat transforms
    # (<base>_roll{w}_{mean,std}) are handled structurally by the codec and are
    # NOT mapped. See codec._ROLL_SUFFIX.
    _TICK_PROMOTE = {
        "depth_imbalance_L1", "depth_imbalance_L3", "depth_imbalance_L5",
        "trade_imbalance", "liq_long_notional", "liq_short_notional",
    }
    for c in _TICK_PROMOTE:
        feats.setdefault(c, "mjolnir (tick IP, promoted)")
    # Raw / exchange-standard tick columns (never IP) -> passthrough.
    _HARD_PASSTHROUGH.update({"ask_amount", "bid_amount", "index_price", "n_trades"})

    drop = (passthrough | _HARD_PASSTHROUGH) - _OPTIONS_PROMOTE
    feat_names = sorted(k for k in feats if k not in drop)
    atom_names = sorted(atoms)

    out = {
        "regime_atoms": atom_names,
        "feature_cols": feat_names,
        "passthrough_excluded": sorted(drop),
        "counts": {"regime_atoms": len(atom_names), "feature_cols": len(feat_names),
                   "passthrough_excluded": len(drop)},
        "provenance": {
            "regime_atoms": {k: atoms[k] for k in atom_names},
            "feature_cols": {k: feats[k] for k in feat_names},
        },
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
