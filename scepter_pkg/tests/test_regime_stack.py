"""Directional correctness for Scepter coded regime stack (private — real names OK)."""
from scepter.research import ScepterResearch
from scepter._obf.codec import default


def test_bearish_own_state_is_short_only():
    c = default()
    rows = ScepterResearch.generate_regime_stack()
    real = [(c.decode_regime(r["regime"]), r["position"]) for r in rows]
    bearish = [(name, pos) for name, pos in real if "macd_bearish" in name]
    assert bearish, "expected some macd_bearish regimes"
    assert all(pos == "short" for _, pos in bearish)


def test_all_crossed_and_anchored():
    c = default()
    anchors = {"btc_trending_up", "btc_trending_down", "btc_high_vol", "btc_low_vol"}
    for r in ScepterResearch.generate_regime_stack():
        parts = set(c.decode_regime(r["regime"]).split("_and_"))
        assert parts & anchors, f"no anchor component in {r['regime']}"
