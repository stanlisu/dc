"""Directional correctness for Scepter coded regime stack (private — real names OK)."""
from scepter.research import ScepterResearch
from scepter._obf.codec import default

_ORIGINAL_OWN_STATE = [
    "above_all_mas", "high_volume", "low_volume", "adx_trend", "vol_breakout",
    "low_vol", "high_vol", "strong_trend", "ma_momentum",
    "rsi_oversold", "rsi_overbought", "macd_bullish", "macd_bearish",
    "stoch_bullish", "bb_rebound", "mom_positive",
]
_ORIGINAL_BTC_STATE = [
    "btc_trending_up", "btc_trending_down", "btc_high_vol", "btc_low_vol",
]


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


def test_zero_arg_default_is_anchor_prefix_btc():
    assert ScepterResearch.generate_regime_stack() == ScepterResearch.generate_regime_stack(anchor_prefix="btc")


def test_zero_arg_preserves_pre_existing_own_state_regimes_byte_identical():
    """The anchor_prefix generalisation + the new convergence_tight own-state
    must not silently change the regimes that already existed. Every (own,
    btc) pair from the ORIGINAL 16-own x 4-btc cross must appear in the
    zero-arg output unchanged (same encoded regime, same position, same
    order) — computed here by an independent re-implementation of the cross,
    not by re-running generate_regime_stack for the expectation — and nothing
    besides convergence_tight's own cross may be added on top of it."""
    c = default()
    seen, expected = set(), []
    for own in _ORIGINAL_OWN_STATE:
        positions = ScepterResearch.allowed_positions(own)
        for btc in _ORIGINAL_BTC_STATE:
            name = f"{own}_and_{btc}"
            for pos in positions:
                key = (name, pos)
                if key in seen:
                    continue
                seen.add(key)
                expected.append({"regime": c.encode_regime(name), "position": pos})

    got_all = ScepterResearch.generate_regime_stack()
    got_pre_existing = [
        r for r in got_all if "convergence_tight" not in c.decode_regime(r["regime"])
    ]
    assert got_pre_existing == expected

    got_new = [r for r in got_all if r not in got_pre_existing]
    assert got_new, "expected convergence_tight to add new rows"
    assert all("convergence_tight" in c.decode_regime(r["regime"]) for r in got_new)


def test_convergence_tight_crossed_with_all_four_btc_states():
    c = default()
    rows = ScepterResearch.generate_regime_stack()
    conv_names = {
        c.decode_regime(r["regime"]) for r in rows
        if c.decode_regime(r["regime"]).startswith("convergence_tight_and_")
    }
    assert conv_names == {
        "convergence_tight_and_btc_trending_up",
        "convergence_tight_and_btc_trending_down",
        "convergence_tight_and_btc_high_vol",
        "convergence_tight_and_btc_low_vol",
    }
    # Breakout can go either direction — both positions for every cross.
    conv_rows = [r for r in rows if c.decode_regime(r["regime"]).startswith("convergence_tight_and_")]
    positions = {(c.decode_regime(r["regime"]), r["position"]) for r in conv_rows}
    for name in conv_names:
        assert (name, "long") in positions
        assert (name, "short") in positions


def test_anchor_prefix_qqq_produces_qqq_named_regimes():
    c = default()
    rows = ScepterResearch.generate_regime_stack(anchor_prefix="qqq")
    real = [c.decode_regime(r["regime"]) for r in rows]

    for suffix in ("qqq_trending_up", "qqq_trending_down", "qqq_high_vol", "qqq_low_vol"):
        assert any(name.endswith(f"_and_{suffix}") for name in real), suffix
    assert not any("btc_" in name for name in real), "no crypto anchor state should leak in"

    conv_cross = {
        name.split("_and_", 1)[1] for name in real
        if name.startswith("convergence_tight_and_")
    }
    assert conv_cross == {
        "qqq_trending_up", "qqq_trending_down", "qqq_high_vol", "qqq_low_vol",
    }
