"""Opt-in rolling-quantile atom thresholds (2026-08-01).

deep_book fired 0 times in 2026Q2 because its cutoffs were measured on a
stale era; the rolling_quantile atom mode replaces the full-frame quantile
with a CAUSAL trailing per-day quantile. These tests pin down:

1. causality      — perturbing future values never changes past masks
2. day-shift      — today's cutoff is built from data through YESTERDAY only
3. adaptivity     — a level-shifted second month re-fires the regime where a
                    stale fixed cutoff goes silent
4. back-compat    — fixed mode (default AND explicit) is byte-identical to
                    the legacy masks
plus config-parser fail-fast semantics and per-symbol independence.

Bars here are 5-minute (288/day): the mechanism infers bar spacing from the
timestamps, so the test grid is interval-agnostic and fast.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mjolnir.core.regime_filters import (
    ATOM_QUANTILE_LEVELS,
    DEFAULT_QUANTILE_WINDOW_BARS,
    RollingQuantileAtoms,
    _window_days,
    apply_filter_mask,
    rolling_quantile_atoms_from_config,
    rolling_quantile_cutoff,
)

BARS_PER_DAY = 288  # 5-minute bars

# Fixed intraday pattern, identical every day (deterministic seed): a shuffled
# grid over [-1, 1] so each day's quantile-q level is the same known value.
_PATTERN = np.random.default_rng(7).permutation(
    np.linspace(-1.0, 1.0, BARS_PER_DAY))


def _frame(day_levels, start="2026-01-01", scale=0.1) -> pd.DataFrame:
    """depth_imbalance_L5 = day_level + scale * fixed intraday pattern."""
    vals = np.concatenate([lvl + scale * _PATTERN for lvl in day_levels])
    idx = pd.date_range(start, periods=len(vals), freq="5min")
    return pd.DataFrame({"depth_imbalance_L5": vals}, index=idx)


def _cfg(window_days=3, levels=None) -> RollingQuantileAtoms:
    return RollingQuantileAtoms(
        window_bars=window_days * BARS_PER_DAY,
        levels={**ATOM_QUANTILE_LEVELS, **(levels or {})},
    )


class TestCausality:
    def test_future_perturbation_leaves_past_masks_unchanged(self):
        df = _frame([1.0] * 10)
        cfg = _cfg(window_days=3)
        base = apply_filter_mask(df, "deep_book", "short", cfg)

        perturbed = df.copy()
        # Blow up the LAST 3 days (days 7..9) by 5x.
        perturbed.iloc[7 * BARS_PER_DAY:, 0] *= 5.0
        after = apply_filter_mask(perturbed, "deep_book", "short", cfg)

        # Days 0..7 use only data through day 6 (values AND cutoffs
        # unperturbed through day 7's cutoff) -> masks identical.
        n_past = 7 * BARS_PER_DAY
        assert (base.to_numpy()[:n_past] == after.to_numpy()[:n_past]).all()
        # Sanity: the perturbation does change SOMETHING later on, so the
        # equality above is not vacuous.
        assert (base.to_numpy() != after.to_numpy()).any()


class TestDayShift:
    def test_todays_cutoff_uses_yesterday_only(self):
        # Day 0: values 0..287 -> quantile(0.4) ~= 114.8.
        # Day 1: half the bars at 100, half at 200. If day 1's cutoff came
        # from day 0 (correct), the 100s fire the short leg and the 200s
        # don't. If day 1's OWN values leaked in, its q0.4 would be 100.0
        # and the 100-group could not be strictly below it.
        day0 = np.arange(BARS_PER_DAY, dtype=float)
        day1 = np.concatenate([
            np.full(BARS_PER_DAY // 2, 100.0),
            np.full(BARS_PER_DAY // 2, 200.0),
        ])
        idx = pd.date_range("2026-01-01", periods=2 * BARS_PER_DAY,
                            freq="5min")
        df = pd.DataFrame(
            {"depth_imbalance_L5": np.concatenate([day0, day1])}, index=idx)

        # window_days=1: min_periods equals the window, so day 1's cutoff is
        # exactly day 0's daily quantile (one full prior day, nothing else).
        mask = apply_filter_mask(df, "deep_book", "short", _cfg(1))
        m = mask.to_numpy()

        # Warmup: day 0 has no prior day -> NaN cutoff -> fails CLOSED.
        assert not m[:BARS_PER_DAY].any()
        # Day 1: exactly the 100-group fires.
        assert m[BARS_PER_DAY:BARS_PER_DAY + BARS_PER_DAY // 2].all()
        assert not m[BARS_PER_DAY + BARS_PER_DAY // 2:].any()


class TestWindowDaysUnitProof:
    """The bar-spacing conversion must not read raw .asi8 ints.

    Root cause of the 2026-08-04 warmup failure: `_window_days` divided
    `np.diff(ts.asi8)` by 1e9 assuming nanoseconds. `.asi8` is raw ints in the
    index's OWN unit, so on a [us] index (parquet round-trip — the production
    path — and pd.date_range's default on pandas 3) spacing was under-counted
    1000x and every window collapsed to the old `max(1, ...)` floor of ONE day.

    Parametrizing over the unit is the point: with the default unit only, this
    reads as "a pandas 3 bug" and passes on pandas 2. It is not version-
    specific — `[us]` reproduces it on 2.3.3 and 3.0.3 alike.
    """

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    @pytest.mark.parametrize(
        "freq, bar_sec", [("5s", 5), ("30s", 30), ("5min", 300)])
    @pytest.mark.parametrize("want_days", [1, 14, 30])
    def test_window_days_independent_of_index_unit(
            self, unit, freq, bar_sec, want_days):
        idx = pd.date_range(
            "2026-03-01", periods=1000, freq=freq).as_unit(unit)
        window_bars = want_days * 86_400 // bar_sec
        assert _window_days(idx, window_bars) == want_days

    @pytest.mark.parametrize("unit", ["ns", "us"])
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_production_hlp_window_is_fourteen_days(self, unit, tz):
        # The exact shield2 hlp rerun config: 241,920 bars of 5s = 14 days.
        # tz="UTC" + unit="us" is the REAL production dtype: bar parquets read
        # back as datetime64[us, UTC] (verified on shield2, 2026-08-04), which
        # is what made _window_days return 1 on every machine and every pandas.
        idx = pd.date_range(
            "2026-03-01", periods=1000, freq="5s", tz=tz).as_unit(unit)
        assert _window_days(idx, 241_920) == 14

    @pytest.mark.parametrize("unit", ["ns", "us"])
    def test_default_window_is_thirty_days(self, unit):
        idx = pd.date_range(
            "2026-03-01", periods=1000, freq="5s").as_unit(unit)
        assert _window_days(idx, DEFAULT_QUANTILE_WINDOW_BARS) == 30

    def test_sub_day_window_raises_instead_of_flooring_to_one(self):
        # The old `max(1, ...)` floor is what let the unit bug masquerade as a
        # working 1-day window. A sub-day request must fail loud.
        idx = pd.date_range("2026-03-01", periods=1000, freq="5s")
        with pytest.raises(ValueError, match="rounds to 0"):
            _window_days(idx, 100)  # 100 * 5s = 500s << 1 day


class TestWarmupFailsClosedEveryUnit:
    """End-to-end warmup guarantee, across index units.

    TestWindowDaysUnitProof pins the helper; this pins the documented contract
    the helper feeds: the first `window_days` per symbol get NaN cutoffs, so
    the regime fires on NO bar. Runs the whole apply_filter_mask path on a [us]
    index — the production (parquet) unit — not just the default.
    """

    @pytest.mark.parametrize("unit", ["ns", "us"])
    @pytest.mark.parametrize("window_days", [3, 5])
    def test_first_window_days_fire_on_no_bar(self, unit, window_days):
        df = _frame([1.0] * 20)
        df.index = df.index.as_unit(unit)
        mask = apply_filter_mask(df, "deep_book", "short", _cfg(window_days))
        warm = mask.iloc[:window_days * BARS_PER_DAY]
        assert not warm.any(), (
            f"{int(warm.sum())} of {len(warm)} warmup bars fired "
            f"(unit={unit}, window_days={window_days}); first offender "
            f"{warm[warm].index.min() if warm.any() else None}")
        # Non-vacuous: the very next day DOES fire once the window is full.
        nxt = mask.iloc[window_days * BARS_PER_DAY:
                        (window_days + 1) * BARS_PER_DAY]
        assert nxt.any()

    @pytest.mark.parametrize("unit", ["ns", "us"])
    @pytest.mark.parametrize("probe_day", [5, 8])
    def test_no_same_day_or_past_cutoff_leakage(self, unit, probe_day):
        """dc #25's invariant, re-proved per unit, ON THE CUTOFF ITSELF.

        Asserting on the mask is weaker and can go vacuous: a mask only flips
        where a cutoff crosses a bar's value, so a real leak can hide behind
        "no bar happened to cross". The cutoff series is the causal object —
        perturbing all of day d must leave every cutoff through the END of
        day d bit-identical, and must move a later one.
        """
        df = _frame([1.0] * 12)
        df.index = df.index.as_unit(unit)
        window_days = 3
        base = rolling_quantile_cutoff(
            df, "depth_imbalance_L5", ATOM_QUANTILE_LEVELS["deep_book_short"],
            window_days * BARS_PER_DAY).to_numpy()

        pert = df.copy()
        lo, hi = probe_day * BARS_PER_DAY, (probe_day + 1) * BARS_PER_DAY
        pert.iloc[lo:hi, 0] += 10.0          # move the WHOLE day's distribution
        after = rolling_quantile_cutoff(
            pert, "depth_imbalance_L5", ATOM_QUANTILE_LEVELS["deep_book_short"],
            window_days * BARS_PER_DAY).to_numpy()

        # Through the end of the perturbed day: identical (NaN-aware).
        np.testing.assert_array_equal(base[:hi], after[:hi])
        # Non-vacuous: a LATER day's cutoff must have moved.
        assert not np.array_equal(base[hi:], after[hi:], equal_nan=True), (
            f"unit={unit}: perturbing day {probe_day} moved no future cutoff — "
            "the test is vacuous, not the code correct")


class TestAdaptivity:
    def test_level_shift_refires_where_fixed_goes_silent(self):
        # Month 1 at level 1.0, month 2 at DOUBLE the level (2.0).
        df = _frame([1.0] * 30 + [2.0] * 30)
        col = df["depth_imbalance_L5"]
        m1_end = 30 * BARS_PER_DAY

        # Stale FIXED rule: short cutoff frozen from month 1's distribution.
        stale_cut = col.iloc[:m1_end].quantile(
            ATOM_QUANTILE_LEVELS["deep_book_short"])
        fired_fixed_m2 = (col.iloc[m1_end:] < stale_cut).sum()
        assert fired_fixed_m2 == 0  # the 2026Q2 failure mode, reproduced

        # ROLLING rule re-adapts and fires at ~ the target rate again.
        mask = apply_filter_mask(df, "deep_book", "short", _cfg(5))
        # Warmup: the first window_days (5) fail CLOSED (min_periods=window).
        assert not mask.iloc[:5 * BARS_PER_DAY].any()
        # Month 1, after the window_days warmup: ~40% firing rate.
        m1_rate = mask.iloc[5 * BARS_PER_DAY:m1_end].mean()
        assert abs(m1_rate - 0.4) < 0.05
        # Month 2, after the 5-day trailing window fully re-adapts (skip 6
        # local days): ~40% again — same rate as before the level shift.
        settled = mask.iloc[m1_end + 6 * BARS_PER_DAY:]
        assert abs(settled.mean() - 0.4) < 0.05

    def test_compound_name_threads_atom_cfg(self):
        df = _frame([1.0] * 5)
        df["relative_spread"] = df["depth_imbalance_L5"].to_numpy() + 10.0
        cfg = _cfg(2)
        combined = apply_filter_mask(
            df, "deep_book_and_wide_spread", "short", cfg)
        expect = (
            apply_filter_mask(df, "deep_book", "short", cfg)
            & apply_filter_mask(df, "wide_spread", "short", cfg)
        )
        pd.testing.assert_series_equal(combined, expect)


class TestFixedModeByteIdentical:
    def _fixture(self) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        n = 500
        return pd.DataFrame({
            "depth_imbalance_L5": rng.normal(size=n),
            "relative_spread": rng.random(n),
            "liq_burst_ratio": rng.random(n),
            "price_range_pct": rng.random(n),
            # mvg1/mvg2/close present so high_vol/low_vol reach their
            # quantile branch (legacy guard returns all-True without them).
            "mvg1": rng.random(n),
            "mvg2": rng.random(n),
            "close": rng.random(n),
        })

    def _legacy_masks(self, df):
        # The pre-2026-08-01 expressions, verbatim.
        db = df["depth_imbalance_L5"]
        sp = df["relative_spread"]
        lq = df["liq_burst_ratio"]
        pr = df["price_range_pct"]
        return {
            ("deep_book", "long"): db > db.quantile(0.6),
            ("deep_book", "short"): db < db.quantile(0.4),
            ("wide_spread", "long"): sp > sp.quantile(0.5),
            ("tight_spread", "long"): sp < sp.quantile(0.5),
            ("high_liquidation_pressure", "long"): lq > lq.quantile(0.75),
            ("low_liquidation_pressure", "long"): lq < lq.quantile(0.25),
            ("high_vol", "long"): pr > pr.quantile(0.5),
            ("low_vol", "long"): pr < pr.quantile(0.5),
        }

    def test_default_call_matches_legacy(self):
        df = self._fixture()
        for (name, pos), expected in self._legacy_masks(df).items():
            got = apply_filter_mask(df, name, pos)
            pd.testing.assert_series_equal(
                got, expected, check_names=False, obj=f"{name}/{pos}")

    def test_explicit_fixed_mode_matches_legacy(self):
        df = self._fixture()
        cfg = rolling_quantile_atoms_from_config({"REGIME_ATOM_MODE": "fixed"})
        assert cfg is None  # fixed parses to the legacy code path
        for (name, pos), expected in self._legacy_masks(df).items():
            got = apply_filter_mask(df, name, pos, cfg)
            pd.testing.assert_series_equal(
                got, expected, check_names=False, obj=f"{name}/{pos}")

    def test_fixed_mode_needs_no_timestamps(self):
        # Integer index, no timestamp column — legacy frames must keep
        # working untouched in fixed mode.
        df = self._fixture()
        mask = apply_filter_mask(df, "deep_book", "short")
        assert mask.dtype == bool and len(mask) == len(df)


class TestPerSymbol:
    def test_each_symbol_gets_its_own_quantile_stream(self):
        # Production shape: integer index + 'timestamp' + 'symbol' columns
        # (streaming.py resets the index before masks are applied).
        a = _frame([1.0] * 4)
        b = _frame([50.0] * 4)  # wildly different level
        fa = a.reset_index().rename(columns={"index": "timestamp"})
        fa["symbol"] = "AAA"
        fb = b.reset_index().rename(columns={"index": "timestamp"})
        fb["symbol"] = "BBB"
        combined = pd.concat([fa, fb], ignore_index=True)

        cfg = _cfg(2)
        mask = apply_filter_mask(combined, "deep_book", "short", cfg)
        solo_a = apply_filter_mask(a, "deep_book", "short", cfg)
        solo_b = apply_filter_mask(b, "deep_book", "short", cfg)

        n = len(a)
        assert (mask.to_numpy()[:n] == solo_a.to_numpy()).all()
        assert (mask.to_numpy()[n:] == solo_b.to_numpy()).all()
        # Both symbols fire after the 2-day warmup — pooling would silence
        # one side.
        assert solo_a.iloc[2 * BARS_PER_DAY:].any()
        assert solo_b.iloc[2 * BARS_PER_DAY:].any()


class TestConfigParser:
    def test_absent_key_is_backcompat_fixed(self):
        assert rolling_quantile_atoms_from_config({}) is None

    def test_explicit_fixed(self):
        assert rolling_quantile_atoms_from_config(
            {"REGIME_ATOM_MODE": "fixed"}) is None

    def test_rolling_defaults(self):
        cfg = rolling_quantile_atoms_from_config(
            {"REGIME_ATOM_MODE": "rolling_quantile"})
        assert cfg.window_bars == DEFAULT_QUANTILE_WINDOW_BARS == 518_400
        assert cfg.levels["deep_book_long"] == 0.6
        assert cfg.levels["deep_book_short"] == 0.4

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="REGIME_ATOM_MODE"):
            rolling_quantile_atoms_from_config(
                {"REGIME_ATOM_MODE": "adaptive"})

    @pytest.mark.parametrize("bad", [0, -1, "30d", True, 1.5, None])
    def test_invalid_window_raises(self, bad):
        with pytest.raises(ValueError, match="WINDOW_BARS"):
            rolling_quantile_atoms_from_config({
                "REGIME_ATOM_MODE": "rolling_quantile",
                "REGIME_ATOM_QUANTILE_WINDOW_BARS": bad,
            })

    def test_window_override(self):
        cfg = rolling_quantile_atoms_from_config({
            "REGIME_ATOM_MODE": "rolling_quantile",
            "REGIME_ATOM_QUANTILE_WINDOW_BARS": 100,
        })
        assert cfg.window_bars == 100

    def test_level_override_and_validation(self):
        cfg = rolling_quantile_atoms_from_config({
            "REGIME_ATOM_MODE": "rolling_quantile",
            "REGIME_ATOM_QUANTILE_LEVELS": {"deep_book_short": 0.2},
        })
        assert cfg.levels["deep_book_short"] == 0.2
        assert cfg.levels["deep_book_long"] == 0.6  # untouched default
        with pytest.raises(ValueError, match="unknown atom"):
            rolling_quantile_atoms_from_config({
                "REGIME_ATOM_MODE": "rolling_quantile",
                "REGIME_ATOM_QUANTILE_LEVELS": {"no_such_atom": 0.5},
            })
        for bad in (0.0, 1.0, -0.5, True, "0.4"):
            with pytest.raises(ValueError, match="QUANTILE_LEVELS"):
                rolling_quantile_atoms_from_config({
                    "REGIME_ATOM_MODE": "rolling_quantile",
                    "REGIME_ATOM_QUANTILE_LEVELS": {"deep_book_short": bad},
                })

    @pytest.mark.parametrize("mode_kv", [
        {},                                # mode absent
        {"REGIME_ATOM_MODE": "fixed"},     # mode explicitly fixed
    ])
    @pytest.mark.parametrize("stray_kv", [
        {"REGIME_ATOM_QUANTILE_WINDOW_BARS": 518_400},
        {"REGIME_ATOM_QUANTILE_LEVELS": {"deep_book_short": 0.4}},
    ])
    def test_rolling_only_keys_without_rolling_mode_raise(
            self, mode_kv, stray_kv):
        # The keys are inert without REGIME_ATOM_MODE="rolling_quantile" —
        # a config carrying them under fixed mode must fail loud, not look
        # active while doing nothing.
        with pytest.raises(ValueError, match="inert"):
            rolling_quantile_atoms_from_config({**mode_kv, **stray_kv})

    def test_rolling_mode_without_timestamps_fails_loud(self):
        df = pd.DataFrame({"depth_imbalance_L5": np.random.default_rng(1)
                          .normal(size=64)})  # integer index, no timestamps
        with pytest.raises(ValueError, match="DatetimeIndex"):
            apply_filter_mask(df, "deep_book", "short", _cfg(2))


class TestLiveBootFailFast:
    def test_trading_constructor_rejects_rolling_mode(self):
        # C1: rolling mode is research-only — the live buffer (1000 bars,
        # ~83 min of 5s) cannot span the trailing window (default 30 days).
        # MjolnirTrading must refuse to BOOT, before any mask call.
        from mjolnir.trading import MjolnirTrading
        cfg = {
            "TIME_UNIT": "5s",
            "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
            "TARGET_HORIZON_BARS": 1,
            "FEE": 2.0,
            "OUTPUT_DIR": "/tmp/mjolnir_test",
            "REGIME_STACK_PATH": "/tmp/regime_stack.csv",
            "REGIME_ATOM_MODE": "rolling_quantile",
        }
        with patch.object(MjolnirTrading, "_load_regime_stack",
                          return_value=[]), \
                patch.object(MjolnirTrading, "_load_models"):
            with pytest.raises(ValueError, match="research-only"):
                MjolnirTrading(config=cfg, home_root="/tmp")
