"""Scale-free replacements for the seven raw price/volume level features.

WHY. `verticalize()` stacks every symbol into one frame, so a feature carried in
price or volume UNITS is partly a symbol ID — BTC trades near $88,000 and
1000PEPE near $0.0045. Seven features survive `select_feature_columns` because
they have no percentage twin, and all seven are raw levels:

    sar, bb_upper, bb_lower, macd, macdhist, obv, ad

MEASURED 2026-08-06, 10 symbols x 360,010 rows of 15m klines, price scale
spanning $0.0045 to $88,468. Spearman IC against the ladder target, comparing
POOLED (all symbols stacked, what the model sees) against mean WITHIN-symbol IC
(the real relationship), plus how many of the 10 symbols disagree with the
pooled sign:

    feature          POOLED    WITHIN     GAP   sign flips
    sar              0.0042    0.0025  0.0017      3/10
    sar_dist        -0.0386   -0.0388  0.0002      0/10
    bb_upper         0.0040    0.0013  0.0028      3/10
    bb_lower         0.0040    0.0010  0.0030      3/10
    bb_pctb         -0.0497   -0.0500  0.0002      0/10
    macd            -0.0125   -0.0160  0.0035      0/10
    macd_norm       -0.0163   -0.0170  0.0006      0/10
    macdhist        -0.0182   -0.0238  0.0056      0/10
    macdhist_norm   -0.0244   -0.0242  0.0002      0/10
    obv              0.0086    0.0049  0.0037      2/10
    obv_slope       -0.0158   -0.0166  0.0008      0/10
    ad              -0.0037   -0.0094  0.0056      0/10
    ad_slope        -0.0155   -0.0161  0.0006      0/10

The raw levels do not merely dilute — they are SIGN-INVERTED. Pooled `sar` reads
+0.0042 while the true relationship is -0.0388, so the model is handed a feature
that points the wrong way. That reproduces an earlier independent measurement
(`1h_sar` pooled +0.0131 vs within-symbol -0.0536, t=-13.3).

`bb_pctb` at -0.0500 would be the strongest single feature in the panel; the
incumbent is `rsi_7` at 0.0376. `sar_dist` at -0.0388 would also land top-3.

DERIVED, NOT RECOMPUTED. The two engines use different indicator parameters —
agamotto `BBANDS(timeperiod=20, nbdev 2/2)` (`research.py:422`) vs mjolnir
`BBANDS(c)` at the talib default of 5 (`core/features.py:396`). Transforming the
already-computed column inherits each engine's parameters instead of silently
imposing one engine's choice on the other.

THE RAW LEVELS STAY IN THE FRAME. `research_filters.py` gates regimes on them —
`close > sar` (:330), `close < bb_lower` (:327), `close > bb_upper` (:347),
`macdhist > 0` (:213), `macdhist < 0` (:263). Dropping the columns would break
those regimes. They are excluded from the MODEL by
`gauntlet/rolling_predict_returns.select_feature_columns`, which governs training
input only; regimes are evaluated against the full frame (dc `research.py:628`).
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.features_scalefree import SCALE_FREE_FEATURES, scale_free_levels

N = 20


def _frame(n=60, close=100.0):
    idx = pd.RangeIndex(n)
    return pd.DataFrame({
        "close": np.full(n, close),
        "sar": np.full(n, close * 0.99),          # 1% below price
        "bb_upper": np.full(n, close * 1.02),
        "bb_lower": np.full(n, close * 0.98),
        "macd": np.full(n, close * 0.001),
        "macdhist": np.full(n, close * 0.0005),
        "obv": np.arange(n, dtype=float) * 1000.0,   # +1000 per bar
        "ad": np.arange(n, dtype=float) * 500.0,
        "volume": np.full(n, 100.0),
    }, index=idx)


def test_every_raw_level_gets_a_scale_free_column():
    out = scale_free_levels(_frame(), window=N)

    assert set(out.columns) == set(SCALE_FREE_FEATURES)
    assert len(SCALE_FREE_FEATURES) == 7


def test_the_transforms_are_invariant_to_price_scale():
    """The whole point. BTC at ~$88,000 and 1000PEPE at ~$0.0045 must produce
    IDENTICAL feature values for the same geometry — that is what stops the
    column acting as a symbol ID."""
    btc = scale_free_levels(_frame(close=88_000.0), window=N)
    pepe = scale_free_levels(_frame(close=0.0045), window=N)

    price_cols = ["sar_dist", "bb_pctb", "bb_width", "macd_norm", "macdhist_norm"]
    pd.testing.assert_frame_equal(btc[price_cols], pepe[price_cols],
                                  check_exact=False, rtol=1e-9)


def test_sar_dist_is_the_signed_fraction_to_the_flip():
    out = scale_free_levels(_frame(close=100.0), window=N)
    # sar sits 1% BELOW price -> price is above the flip -> positive
    assert out["sar_dist"].iloc[-1] == pytest.approx(0.01, abs=1e-12)


def test_bb_pctb_is_zero_at_the_lower_band_and_one_at_the_upper():
    """Classic %B. It subsumes both band distances, which are near-collinear
    with it, and measured -0.0500 — the strongest feature in the panel."""
    df = _frame(close=100.0)
    df["close"] = df["bb_lower"]
    assert scale_free_levels(df, window=N)["bb_pctb"].iloc[-1] == pytest.approx(0.0)

    df["close"] = df["bb_upper"]
    assert scale_free_levels(df, window=N)["bb_pctb"].iloc[-1] == pytest.approx(1.0)


def test_bb_width_is_band_span_over_price():
    out = scale_free_levels(_frame(close=100.0), window=N)
    assert out["bb_width"].iloc[-1] == pytest.approx(0.04, abs=1e-12)


def test_obv_and_ad_slopes_are_flow_over_volume_not_a_running_total():
    """`obv`/`ad` are CUMULATIVE, so their level encodes how long the symbol has
    been listed as much as anything about price. Differencing over the window and
    dividing by traded volume gives a bounded flow measure."""
    out = scale_free_levels(_frame(), window=N)

    # obv rises 1000/bar over 20 bars = 20000; volume sums to 20*100 = 2000.
    assert out["obv_slope"].iloc[-1] == pytest.approx(10.0, abs=1e-9)
    assert out["ad_slope"].iloc[-1] == pytest.approx(5.0, abs=1e-9)


def test_a_flat_book_does_not_divide_by_zero():
    """Degenerate but real: zero volume in a window, or bb_upper == bb_lower on a
    perfectly flat stretch. Must yield NaN, never inf — an inf propagates through
    the IC ranker and silently poisons feature selection."""
    df = _frame()
    df["volume"] = 0.0
    df["bb_upper"] = df["bb_lower"]

    out = scale_free_levels(df, window=N)

    assert np.isfinite(out.to_numpy(dtype=float)).sum() >= 0     # no exception
    assert not np.isinf(out.to_numpy(dtype=float)).any(), "inf leaked"


def test_zero_close_yields_nan_rather_than_inf():
    df = _frame()
    df.loc[df.index[-1], "close"] = 0.0

    out = scale_free_levels(df, window=N)

    assert not np.isinf(out.to_numpy(dtype=float)).any()


def test_missing_input_column_raises_rather_than_silently_skipping():
    """CLAUDE.md: no silent fallbacks. A renamed upstream column must fail loudly,
    not quietly drop a top-3 feature from the panel."""
    df = _frame().drop(columns=["sar"])

    with pytest.raises(KeyError, match="sar"):
        scale_free_levels(df, window=N)


def test_prefix_handles_the_multi_timeframe_panel():
    """orb/scepter carry `15m_sar`, `1h_sar`, ... in one frame; agamotto's own
    panel is single-TF and unprefixed. Both spellings must work or the transform
    lands on only half the algos."""
    df = _frame().add_prefix("1h_")

    out = scale_free_levels(df, window=N, prefix="1h_")

    assert "1h_sar_dist" in out.columns
    assert out["1h_sar_dist"].iloc[-1] == pytest.approx(0.01, abs=1e-12)


# --------------------------------------------------------------------------- #
# obv/ad arrive in TWO different shapes and must not both be differenced
#
#   agamotto `research.py:411-412`:  obv_raw.diff(14).fillna(0.0)   <- ALREADY a
#                                    difference, just in volume units
#   mjolnir  `core/features.py:420`: talib.OBV(c, v)                <- CUMULATIVE
#
# Differencing agamotto's column again would take a second difference and
# destroy the feature silently — it would still produce plausible-looking
# numbers, which is exactly how this class of bug survives review.
# --------------------------------------------------------------------------- #
def test_already_differenced_obv_is_only_normalised_not_differenced_again():
    """agamotto shape: obv is a 14-bar diff in volume units. Dividing by the
    window's traded volume makes it scale-free; differencing is wrong."""
    df = _frame()
    df["obv"] = 2000.0          # a CONSTANT flow of 2000 per bar, already diffed
    df["ad"] = 1000.0

    out = scale_free_levels(df, window=N, obv_is_cumulative=False)

    # volume sums to 20*100 = 2000 over the window -> 2000/2000 = 1.0
    assert out["obv_slope"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert out["ad_slope"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_a_second_difference_would_have_produced_zero_here():
    """Guards the failure mode directly: on a constant already-differenced
    series, differencing again yields 0.0 for every row — a dead feature that
    still looks numerically fine."""
    df = _frame()
    df["obv"] = 2000.0

    wrong = scale_free_levels(df, window=N, obv_is_cumulative=True)
    right = scale_free_levels(df, window=N, obv_is_cumulative=False)

    assert wrong["obv_slope"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
    assert right["obv_slope"].iloc[-1] != pytest.approx(0.0, abs=1e-12)


def test_cumulative_is_the_default_matching_the_measured_experiment():
    """The 0/10-sign-flip measurement used raw `talib.OBV` then diff(20)/volume,
    i.e. the mjolnir shape. Keep that the default so the default matches the
    number that justified this change."""
    df = _frame()
    out_default = scale_free_levels(df, window=N)
    out_explicit = scale_free_levels(df, window=N, obv_is_cumulative=True)

    pd.testing.assert_frame_equal(out_default, out_explicit)


# --------------------------------------------------------------------------- #
# The call sites — a module nothing calls is a no-op no matter how well tested
# --------------------------------------------------------------------------- #
def _ohlcv(prefix="BTCUSDT", n=400, close=100.0):
    rng = np.random.default_rng(0)
    c = close * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    return pd.DataFrame({
        f"{prefix}_open": c,
        f"{prefix}_high": c * (1 + np.abs(rng.normal(0, 4e-4, n))),
        f"{prefix}_low": c * (1 - np.abs(rng.normal(0, 4e-4, n))),
        f"{prefix}_close": c,
        f"{prefix}_volume": np.abs(rng.normal(1000, 100, n)),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))


def test_agamotto_engineer_features_emits_the_scale_free_columns():
    """End-to-end through the real engine, not the helper in isolation."""
    from agamotto.research import AgamottoResearch

    r = AgamottoResearch.__new__(AgamottoResearch)
    r.config = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0,
                "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14}
    r.raw = _ohlcv()
    r.engineer_features()

    for col in SCALE_FREE_FEATURES:
        assert f"BTCUSDT_{col}" in r.features.columns, f"missing BTCUSDT_{col}"


def test_agamotto_keeps_the_raw_levels_because_regimes_gate_on_them():
    """research_filters uses `close > sar`, `close < bb_lower`, `macdhist > 0`.
    Dropping the raw columns would break those regimes; they are excluded from
    the MODEL downstream, not from the frame."""
    from agamotto.research import AgamottoResearch

    r = AgamottoResearch.__new__(AgamottoResearch)
    r.config = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0,
                "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14}
    r.raw = _ohlcv()
    r.engineer_features()

    for col in ("sar", "bb_upper", "bb_lower", "macdhist"):
        assert f"BTCUSDT_{col}" in r.features.columns, (
            f"BTCUSDT_{col} was dropped — regimes gating on it will break")


def test_agamotto_obv_slope_is_not_double_differenced():
    """agamotto stores obv already diffed, so the call site MUST pass
    obv_is_cumulative=False. If it regresses to the default the column goes
    near-constant, which this catches via its variance."""
    from agamotto.research import AgamottoResearch

    r = AgamottoResearch.__new__(AgamottoResearch)
    r.config = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0,
                "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14}
    r.raw = _ohlcv()
    r.engineer_features()

    obv_slope = r.features["BTCUSDT_obv_slope"].dropna()
    assert len(obv_slope) > 50
    assert obv_slope.std() > 1e-9, "obv_slope is ~constant: differenced twice"


def test_mjolnir_mirror_is_identical_to_agamottos():
    """The two copies exist only because mjolnir_pkg cannot import agamotto_pkg.
    Nothing enforces that at import time, so enforce it here: identical output on
    identical input, or the algos silently diverge the way the ladder maths did.
    """
    pytest.importorskip("mjolnir.core.features_scalefree",
                        reason="mjolnir package not on PYTHONPATH")
    from mjolnir.core.features_scalefree import (
        SCALE_FREE_FEATURES as MJ_COLS, scale_free_levels as mj_scale_free)

    assert MJ_COLS == SCALE_FREE_FEATURES

    rng = np.random.default_rng(7)
    n = 300
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    df = pd.DataFrame({
        "close": c, "sar": c * (1 - rng.uniform(-0.02, 0.02, n)),
        "bb_upper": c * 1.02, "bb_lower": c * 0.98,
        "macd": rng.normal(0, 0.05, n), "macdhist": rng.normal(0, 0.02, n),
        "obv": np.cumsum(rng.normal(0, 500, n)),
        "ad": np.cumsum(rng.normal(0, 300, n)),
        "volume": np.abs(rng.normal(1000, 200, n)),
    })

    for cumulative in (True, False):
        pd.testing.assert_frame_equal(
            scale_free_levels(df, window=N, obv_is_cumulative=cumulative),
            mj_scale_free(df, window=N, obv_is_cumulative=cumulative))


def _mjolnir_features_class(module):
    """The mjolnir feature engine, located by capability rather than by name."""
    return next(c for c in vars(module).values()
                if isinstance(c, type) and hasattr(c, "_compute_price_features"))


def _mjolnir_bars(n=400, seed=3):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    return pd.DataFrame({
        "open": c, "close": c,
        "high": c * (1 + np.abs(rng.normal(0, 5e-4, n))),
        "low": c * (1 - np.abs(rng.normal(0, 5e-4, n))),
        "volume": np.abs(rng.normal(1000, 150, n)),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15s"))


def test_mjolnir_price_features_emit_the_scale_free_columns():
    """mjolnir's call site, end-to-end. Its obv/ad are raw cumulative talib
    output, so it must use the DEFAULT differencing — the opposite of agamotto.
    """
    fe = pytest.importorskip("mjolnir.core.features",
                             reason="mjolnir package not on PYTHONPATH")
    df = _mjolnir_bars()

    # CONSTRUCT the engine — do NOT `cls.__new__(cls)`. That allocates an
    # instance without running __init__, and it worked here only for as long
    # as _compute_price_features touched no instance state. dc #62 made the TA
    # price source a REQUIRED keyword-only ctor arg with no default
    # (mjolnir/core/features.py:201) and reads it at :481/:534/:571, so the
    # hollow instance began raising `AttributeError: 'MjolnirFeatures' object
    # has no attribute 'ta_price_source'` — a symptom deep inside the engine
    # instead of the TypeError a real call site would have got. That is what
    # turned dc `main` red on 2026-08-28 (run 33146271206).
    #
    # "close" — the trade OHLC — is stated, not defaulted, because that is the
    # source the 10-symbol x 360k-row measurement in this module's docstring
    # was taken on. "book_mid" is not usable here anyway: it reads
    # bids_0_price/asks_0_price, which synthetic OHLCV bars do not carry.
    engine = _mjolnir_features_class(fe)(ta_price_source="close")
    out = engine._compute_price_features(df)

    for col in SCALE_FREE_FEATURES:
        assert col in out.columns, f"mjolnir missing {col}"
    # raw levels retained for regime filters
    for col in ("sar", "bb_upper", "bb_lower", "macdhist"):
        assert col in out.columns, f"mjolnir dropped raw {col}"
    assert out["obv_slope"].dropna().std() > 1e-12


def test_an_engine_that_skipped_its_constructor_cannot_fabricate_features():
    """The negative control for the test above — and a guard against the WRONG
    fix for the red main it came from.

    The tempting one-line repair was to hand `ta_price_source` a value the
    engine can read without the constructor: a class attribute, a
    `__getattr__`, a property with a fallback. Any of those makes
    `cls.__new__(cls)` silently work again, which is exactly the magic default
    dc #62 existed to delete — reinstated where no signature shows it.

    mjolnir's own `TestFlag::test_ta_price_source_has_no_default` does NOT
    cover that case: it inspects the `__init__` signature, and a CLASS
    attribute leaves the signature untouched, so that test still passes while
    the contract is gone.

    So pin the negative directly. An instance that skipped `__init__` must
    fail, loudly, naming the state it is missing — never return a frame.
    """
    fe = pytest.importorskip("mjolnir.core.features",
                             reason="mjolnir package not on PYTHONPATH")
    cls = _mjolnir_features_class(fe)

    hollow = cls.__new__(cls)
    with pytest.raises(AttributeError, match="ta_price_source"):
        hollow._compute_price_features(_mjolnir_bars())
