#pragma once
// Agamotto feature computation. PRIVATE — this is the alpha.
//
// STAGE 2.0/2.1/2.2/2.3/2.4/2.5: the column table, the numeric primitives, the
// OHLC / returns / MA / volume feature blocks, the TA-Lib indicator block, the
// rolling return-moment stats and the scale-free level transforms. That is the
// COMPLETE feature panel — 65 columns. The regime gate and the model are later
// stages and deliberately absent — a half-built engine that emits some
// columns would be indistinguishable from one that silently drops them, which
// is why `engineerFeatures` emits a FIXED, declared column set and
// tests/feature_parity.py asserts set equality against it rather than
// "everything that happened to line up".
//
// NO TARGET COLUMN IS COMPUTED HERE, EVER. research.py's return_dip,
// return_rip, return / return_long* / return_short* and the ret_2bar* family
// all read close/high/low at shift(-1) or shift(-2): they are LOOKAHEAD target
// construction for the trainer, not features. A live engine that computed one
// would be reading a bar that has not closed. tests/feature_parity.py declares
// them in EXPECTED_ABSENT_TARGETS and fails if any of them ever appears in the
// engine's output.
//
// PANEL-BASED BY DESIGN, for the same reason mjolnir's is (see
// ../../sentinel_core/src/feature_engine.hpp): the gate is numerical parity
// against a pandas panel, and optimising to an incremental ring buffer before
// the reference is matched leaves no trustworthy baseline to optimise against.
//
// ---------------------------------------------------------------------------
// EPS: agamotto's is 1e-8, written INLINE per expression. There is deliberately
// no shared EPS constant here.
//
//   research.py:362  price_range_pct  = (high - low)      / (open + 1e-8)
//   research.py:378  open_close_pct   = open_close_diff   / (open + 1e-8)
//   research.py:379  high_open_pct    = (high - open)     / (open + 1e-8)
//   research.py:380  low_open_pct     = (low  - open)     / (open + 1e-8)
//   research.py:475  vol_ratio        = vol               / (vol_ma + 1e-8)
//   research.py:485  quote_vol_ratio  = quote_vol         / (quote_vol_ma + 1e-8)
//   research.py:491  buy_pressure     = taker_buy_quote   / (quote_vol + 1e-8)
//   research.py:498  trade_intensity  = num_trades        / (trades_ma + 1e-8)
//
// mjolnir uses a shared 1e-10 (sentinel_core/src/feature_engine.cpp `EPS`).
// The two must NOT be unified: a shared constant is one edit away from
// silently re-scaling every ratio feature in one of the two algos, and the
// change would look like a tidy-up in review.
//
// The epsilon is ABSOLUTE, so its size is RELATIVE TO THE SYMBOL'S PRICE. On
// BTC (~64000) 1e-8 is a 1.6e-13 perturbation — below double's ability to care.
// On 1000PEPE (~0.0045) it is 2.2e-6, i.e. it moves the SIXTH significant
// figure of price_range_pct, which is a top-5 IC-selected feature on several
// deployed regimes. tests/feature_parity.py therefore runs BOTH price scales;
// a harness that only ever sees BTC cannot distinguish `+1e-8` from `+0`.
// ---------------------------------------------------------------------------
// NO inf/NaN SANITISATION HAPPENS IN THIS FILE.
//
// mjolnir's engine replaces non-finite cells with 0.0 panel-wide
// (sentinel_core/src/feature_engine.cpp). agamotto does NOT, and the difference
// is load-bearing rather than stylistic:
//
//   * the ONLY fill in the whole agamotto live path is `X.fillna(0.0)`
//     (trading.py:700) applied to the SELECTED MODEL COLUMNS of the SINGLE
//     scored row, after the regime gate has already run;
//   * inf is never touched anywhere;
//   * the regime predicates are evaluated on the RAW panel, and they are
//     built out of `>` / `<` comparisons, for which NaN is False. A NaN that
//     gets helpfully replaced by 0.0 stops being "this gate cannot fire" and
//     becomes "this gate compares against zero", which is how an inert regime
//     turns into an always-firing one.
//
// So NaN and inf propagate out of `engineerFeatures` exactly as pandas
// produces them, and the parity harness compares the NaN/+inf/-inf/finite
// CLASSIFICATION of every cell before it looks at a single value.
// ---------------------------------------------------------------------------
//
// pandas semantics reproduced here, each verified cell-by-cell against pandas
// 2.3.3 by tests/pdops_golden.py + tests/pdops_parity_driver.cpp:
//
//   * min_periods counts NON-NaN OBSERVATIONS in the window, never rows
//   * rolling std/var are ddof=1 (SAMPLE), NaN at nobs <= 1
//   * skew is the SAMPLE (bias-corrected) G1, NaN unless nobs >= 3
//   * kurt is the SAMPLE EXCESS G2, NaN unless nobs >= 4
//   * quantile interpolates LINEARLY between the two bracketing order
//     statistics of the NON-NaN values, on the NON-NaN COUNT
//   * corr PAIRWISE-MASKS BOTH SERIES FIRST (pandas rolling.py prep_binary:
//     `X = x + 0*y; Y = y + 0*x`) and only then takes means and variances
//   * the constant-window guards: pandas forces std -> exactly 0, skew -> 0,
//     kurt -> exactly -3 when every observation in the window is equal,
//     instead of returning the floating-point residue
//
// The rolling accumulators are STREAMING (add/remove with Kahan compensation),
// transcribed from pandas 2.3.3 `pandas/_libs/window/aggregations.pyx` (the
// sdist source, read — not inferred from behaviour), because that is what
// pandas actually computes. A per-window re-summation is MORE accurate but
// differs from pandas by up to 1.1e-10 on a price-scale series (measured),
// which is larger than the parity gate.
//
// Two pandas details that are invisible from the docs and that cost parity
// outright if guessed:
//
//   * roll_skew and roll_kurt PRE-CENTRE THE WHOLE ARRAY by round(nanmean),
//     guarded on `nanmin - mean > -1e5` for skew and `> -1e4` for kurt. The
//     guards really are different constants. Without this the raw 4th moment
//     of a price-scale series cancels ~1e12 to 1 and the answer is noise.
//   * calc_mean's constant-window guard replaces the mean with prev_value
//     ITSELF and SHORT-CIRCUITS the sign clamps (an `elif` chain), rather
//     than dividing prev*nobs by nobs — which would re-introduce exactly the
//     rounding the guard exists to remove.
//
// CONSEQUENCE, and it is a real one: a streaming accumulator plus a WHOLE-ARRAY
// centring constant makes the answer depend on where the panel STARTS, not only
// on the window. pandas has the same property and disagrees with ITSELF by up
// to 5.6e-8 on a price-scale 14-bar kurt when the frame start moves
// (tests/pdops_golden.py prints the measurement). Live will run a ~700-bar
// panel while research runs years of bars, so the last digits of
// std/skew/kurt CANNOT match research exactly however this is written. The
// measured agreement is ~1e-14 ABSOLUTE on the return-scale columns the
// reference actually computes (research.py:569-571 feed skew/kurt/std
// `hist_return` only); it is not 1e-16 anywhere, and a pure relative gate on a
// skew that legitimately sits near zero is measuring the denominator, not the
// port.
//
// STAGE 2.4 OUTCOME, since that paragraph predicted trouble: none of it
// materialised at the FEATURE level, and the reason is worth stating so nobody
// later "fixes" a gate that is not loose. The panel-start instability is a
// statement about comparing pandas on a 699-row frame against pandas on a
// multi-year frame. The feature gate does not do that: BOTH sides see the SAME
// 699 rows, so `std`, `skew`, `kurt` and `acf_lag1` all pass the ordinary
// 1e-9 RELATIVE gate on all five scenarios and both toolchains, with no probe
// tier and no widened tolerance. The 1e-12 / PROBE treatment stage 2.1 applies
// to the pdops primitives stays where it is — it grades a DIFFERENT comparison
// (pandas against itself across frame starts) and is still needed there. What
// remains true is the research-vs-live caveat: the last digits of these four
// columns cannot match a years-long research panel however this is written.

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace agamotto {

// Column-major panel: insertion-ordered names (feature CODES once stage 2.2
// lands — see src/codes_generated.hpp) each mapping to one column of doubles.
// Ordered rather than a bare map so the emitted panel is diff-stable, and
// lookup is by name so a consumer cannot silently read the wrong column after
// an insertion.
struct Table {
    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;
    std::map<std::string, size_t> idx;

    void put(const std::string& n, std::vector<double> v);
    bool has(const std::string& n) const { return idx.count(n) != 0; }
    // Both throw std::out_of_range on an unknown name. Deliberate: returning
    // an empty column would let a missing feature read as a column of zeros.
    const std::vector<double>& get(const std::string& n) const;
    std::vector<double>& mut(const std::string& n);
    size_t size() const { return names.size(); }
};

// --- pandas-equivalent primitives ------------------------------------------
// Every `mp` argument is REQUIRED, never defaulted: the reference calls the
// same window at min_periods=1 and at min_periods=w (research.py:363 vs :373)
// and the two differ on hundreds of rows.
namespace pdops {

// x.diff()  -> out[0] = NaN, out[i] = x[i] - x[i-1]
std::vector<double> diff(const std::vector<double>& x);
// x.diff(n) -> first n are NaN. research.py:538 uses .diff(14) on obv/ad.
std::vector<double> diffN(const std::vector<double>& x, int n);
// x.shift(n) -> first n are NaN. n <= 0 is rejected (see .cpp).
std::vector<double> shift(const std::vector<double>& x, int n);
// x.pct_change(periods, fill_method=None) -> x[i]/x[i-periods] - 1.
// Yields +/-inf on a zero denominator exactly as pandas does; the caller
// sanitises, this does not hide it.
std::vector<double> pctChange(const std::vector<double>& x, int periods);

// x.rolling(w, min_periods=mp).sum()
std::vector<double> rollSum(const std::vector<double>& x, int w, int mp);
// x.rolling(w, min_periods=mp).mean()
std::vector<double> rollMean(const std::vector<double>& x, int w, int mp);
// x.rolling(w, min_periods=mp).var(ddof)
std::vector<double> rollVar(const std::vector<double>& x, int w, int mp, int ddof);
// x.rolling(w, min_periods=mp).std()   [ddof=1]
std::vector<double> rollStd(const std::vector<double>& x, int w, int mp);
// x.rolling(w, min_periods=mp).skew()  SAMPLE G1, NaN unless nobs >= 3
std::vector<double> rollSkew(const std::vector<double>& x, int w, int mp);
// x.rolling(w, min_periods=mp).kurt()  SAMPLE EXCESS G2, NaN unless nobs >= 4
std::vector<double> rollKurt(const std::vector<double>& x, int w, int mp);
// x.rolling(w, min_periods=mp).corr(y)
std::vector<double> rollCorr(const std::vector<double>& x, const std::vector<double>& y,
                             int w, int mp);
// x.rolling(w, min_periods=mp).quantile(q), interpolation='linear'
std::vector<double> rollQuantile(const std::vector<double>& x, int w, int mp, double q);
// k levels from ONE sort per window. Result[j] == rollQuantile(x, w, mp, qs[j])
// cell for cell — asserted by the parity driver, not assumed.
std::vector<std::vector<double>> rollQuantiles(const std::vector<double>& x, int w, int mp,
                                               const std::vector<double>& qs);

} // namespace pdops

// --- stage 2.3: the TA-Lib indicator block ---------------------------------
//
// Implemented in src/talib_block.cpp, which calls libta-lib (pinned 0.6.4)
// DIRECTLY — the same C library the reference's `talib.*` wrapper calls — so
// the indicator values are identical by construction rather than
// reimplemented. 25 calls, 29 columns, plus `parkinson_vol`, which is NOT
// TA-Lib but sits INSIDE the reference's `try:` (research.py:545-548) and so
// shares the block's fate.
//
// A SEPARATE TRANSLATION UNIT for exactly one reason: it is the only file in
// the core that needs a ta-lib include and a ta-lib link, and keeping it out of
// feature_engine.cpp keeps that dependency visible in the build recipe rather
// than buried in a 900-line file.
namespace talib_block {

// TA_Initialize, once. Idempotent; `compute` calls it, so callers need not.
void init();

// Appends 30 (column code, column) pairs to `out`, in research.py:501-553
// order. All five inputs must be the same length.
//
// `volume` is handed to TA-Lib RAW. research.py:535 does not fill it, and
// mjolnir's `volume.fillna(0)` must NOT be copied here: a NaN volume bar is
// meant to poison OBV/AD/MFI from that bar onward, because that is what
// production computes.
//
// LEADING NaNs ARE SKIPPED, per input set, exactly as the reference's Cython
// wrapper's `check_begidx1..4` do; interior NaNs are not. See the banner in
// src/talib_block.cpp for the measurements.
//
// Throws std::invalid_argument if any input column is ENTIRELY NaN — a
// DECLARED DIVERGENCE. The reference's wrapper raises "inputs are all NaN"
// there and research.py:554 swallows it into a `logger.warning`, silently
// dropping all 30 of these columns from the panel. A live core must not emit a
// panel whose missing columns are indistinguishable from a dead feed.
void compute(const std::vector<double>& open,
             const std::vector<double>& high,
             const std::vector<double>& low,
             const std::vector<double>& close,
             const std::vector<double>& volume,
             std::vector<std::pair<std::string, std::vector<double>>>& out);

} // namespace talib_block

// --- stage 2.2: the feature blocks ------------------------------------------
//
// PANEL WIDTH IS A CORRECTNESS PARAMETER, NOT A BUFFER SIZE.
//
//   trading.py:443  load_data(limit: int = 700)
//   trading.py:480  combined = combined.tail(limit)   -> 700 rows
//   trading.py:485  combined = combined.iloc[:-1]     -> the incomplete bar is
//                                                        dropped -> 699 CLOSED
//
// So live engineers EXACTLY 699 bars, and two columns read the row count
// directly rather than only a trailing window of it:
//
//   * price_range_pct_q50 is rolling(700, min_periods=1) (research.py:363).
//     On any frame SHORTER than 700 rows that is an EXPANDING median — the
//     value on row i is the median of rows 0..i, of ALL of them. Hand it 700
//     rows instead of 699 and every cell from row 0 on can move; hand it 1000
//     and they all do. "More history is safer" is a silent parity break here,
//     not a performance tweak.
//   * price_range_pct_q80/q90/q95 are rolling(700, min_periods=700), so on a
//     699-row frame they are ALL-NaN by construction (see VOL_Q_WINDOW).
//
// `engineerFeatures` therefore REFUSES a panel of any other width instead of
// quietly producing numbers live would never see.
constexpr size_t PANEL_BARS = 699;

// research.py:61. Doubles as the min_periods of the q80/q90/q95 cutoffs
// (research.py:373), which is why those three columns are ENTIRELY NaN on a
// PANEL_BARS-wide panel: 699 observations < 700 min_periods, on every row.
//
// THIS IS REPRODUCED DELIBERATELY AND MUST NOT BE "FIXED" HERE. It is the
// live behaviour, and it is the subject of a filed production finding —
// marvel PR #532, docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md
// — which measures that 53 of 62 deployed regimes cannot fire live because of
// it (`x > NaN` is False on every bar). A port that "helpfully" lowered
// min_periods, or widened the panel to 700, would make those regimes start
// firing against models that were never trained on a firing regime, and the
// port would look like the cause. tests/feature_parity.py asserts the
// all-NaN property explicitly so it is pinned rather than incidental; when the
// finding is resolved, it is resolved in research.py first and mirrored here.
constexpr int VOL_Q_WINDOW = 700;

// --- stage 2.4: the rolling return-moment stats ------------------------------
//
// research.py:557 `stats_window = int(self.config.get("STATS_WINDOW", 14))`.
// The `.get` default is the shape CLAUDE.md bans, but it does NOT bind on the
// deployed arm: marvel/gauntlet/pred_agamotto.base.15m_1/setting.json carries
// `"STATS_WINDOW": 14` EXPLICITLY (verified 2026-08-19). research.py:564 then
// raises outright below 4, because `_kurt` needs 4 observations and `_acf_lag1`
// is a (w-1)-wide correlation that needs 3 pairs. Compile-time here for the
// same reason MA_PERIODS is: a live core that silently accepted a different
// window would score frozen weights against differently-defined columns.
//
// ALL FOUR ARE COMPUTED ON `hist_return`, NEVER ON `close` (research.py:569-571
// and :587 all take `hist_return`, which is `close.pct_change(fill_method=None)`
// from research.py:382). That distinction is the whole reason the `std` column
// was withheld from stage 2.3:
//
//   *** `f085` IS A HOMONYM ACROSS THE TWO ALGOS. ***
//   sentinel_core/src/talib_block.cpp:133 emits `codes::F_STD` (= "f085") from
//   `TA_STDDEV(close, 14)` — the rolling standard deviation of PRICE, in price
//   units, ~1e4 on BTC. agamotto's `f085` is the rolling standard deviation of
//   RETURNS, dimensionless, ~4e-3. Same code, same 14, different quantity by
//   seven orders of magnitude. Stage 2.3 therefore deliberately did NOT emit
//   f085 from the TA-Lib block (see the ta-lib file's own banner); it is
//   emitted HERE and NOWHERE ELSE in this core. The two definitions must never
//   coexist in one binary — a panel carrying a price-scale f085 would be scored
//   by weights trained on a return-scale one, and nothing would raise.
//
// min_periods is the pandas DEFAULT, i.e. equal to the window
// (`hist_return.rolling(window=stats_window)` with no min_periods argument), so
// std/skew/kurt need 14 non-NaN observations and acf_lag1 needs 13 non-NaN
// PAIRS. Both are passed explicitly to the pdops primitives, which have no
// defaults.
//
// acf_lag1 carries a `.fillna(0.0)` (research.py:589) that runs AFTER the
// correlation, so rows 0..12 — and every window the min_periods rejects, and
// every window in which pandas' constant-window guard yields NaN — read 0.0,
// not NaN. That fill is the ONLY sanitisation in the whole agamotto panel and
// it is one column wide; it is not a licence to fill anything else.
constexpr int STATS_WINDOW = 14;

// --- stage 2.5: the scale-free level transforms ------------------------------
//
// agamotto_pkg/src/agamotto/features_scalefree.py:113-129, called from
// research.py:639-646 with `window` left at its `DEFAULT_WINDOW = 20` and
// `obv_is_cumulative=False`.
//
// `obv_is_cumulative=False` IS LOAD-BEARING, not a tuning knob. research.py:538
// already stores `obv_raw.diff(14).fillna(0.0)`, so the `obv`/`ad` columns this
// block consumes are ALREADY differenced. Passing True would apply
// `_flow`'s `.diff(window)` a SECOND time. That failure is silent in the worst
// way: on a steady flow a second difference returns ~0 for every row while
// still looking like a valid feature, so the column survives every "is it
// present / is it finite" check and simply carries no information.
// tests/run_feature_parity.sh --negative-secondiff builds exactly that mutant
// and requires the gate to go red.
//
// Every one of the seven divisions goes through `_safe`
// (features_scalefree.py:61-63), which is TWO DISTINCT STEPS:
//
//     out = num / den.replace(0.0, np.nan)          # step 1
//     return out.replace([np.inf, -np.inf], np.nan) # step 2
//
// Step 1 maps an EXACTLY-zero denominator to NaN — and pandas' `replace`
// compares with `==`, under which `-0.0 == 0.0`, so negative zero is replaced
// too (verified against pandas 2.3.3). Step 2 maps any REMAINING +/-inf to NaN,
// which is a different case: it catches an infinite NUMERATOR, and an overflow
// from a finite numerator over a tiny-but-nonzero denominator. A single
// `if (den == 0.0) return NaN;` implements step 1 only and is NOT equivalent.
// See `safeDiv` in the .cpp and the `--negative-safeinf` control.
//
// These seven columns CONSUME stage-2.3 outputs (sar, bb_upper, bb_lower, macd,
// macdhist, obv, ad) and the raw close/volume, so they must be computed AFTER
// the TA-Lib block, and the raw levels STAY in the panel — research_filters
// gates regimes on them directly (`close > sar`, `close < bb_lower`,
// `macdhist > 0`).
//
// DECLARED DIVERGENCE, inherited from stage 2.3. research.py:637-657 wraps this
// block in `if all(k in _src for k in _need): ... else: logger.warning(...)`,
// so a TA-Lib failure upstream drops all seven columns with only a warning.
// This core throws from the TA-Lib block instead of producing a short panel, so
// the `else` branch is unreachable here and is not reproduced.
constexpr int SCALE_FREE_WINDOW = 20;

// The raw wide-frame columns for ONE symbol, in the shape research.py's
// `self.raw` carries them (research.py:210 `df[existing_cols]`).
//
// `open/high/low/close/volume` are REQUIRED — research.py:203 raises on a
// missing one, and so does this.
//
//   DECLARED DIVERGENCE. research.py:356-358 does
//       open_series = df.get(open_col, close)
//   i.e. it SILENTLY SUBSTITUTES CLOSE for a missing open/high/low, which would
//   turn price_range_pct into 0/(close+1e-8) and high_open_pct/low_open_pct into
//   exact zeros with nothing logged. It is unreachable through research.py's own
//   loader (:203 raises on a missing required column) and through the kline
//   builder (which always emits OHLC), so no live or research panel can take it
//   — but it is the `cfg.get(K, X)` shape CLAUDE.md bans, and a live core is the
//   wrong place to reproduce a fallback whose only effect is to hide a broken
//   feed. This throws instead. Reported, not fixed, in research.py.
//
// The remaining three are genuinely OPTIONAL in
// the reference (`if f"{base}_quote_volume" in df.columns`, research.py:470,
// :483, :495): leave the vector EMPTY to mean "the feed does not carry it",
// which drops exactly the columns the reference drops. An empty vector is NOT
// the same as a vector of zeros and must never be silently substituted for one
// — zeros would emit a real-looking quote_vol_ratio of 0/1e-8.
struct RawBars {
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;
    std::vector<double> quote_volume;            // optional; empty = absent
    std::vector<double> taker_buy_quote_volume;  // optional; empty = absent
    std::vector<double> number_of_trades;        // optional; empty = absent
};

// The OHLC / returns / MA / volume blocks of research.py `engineer_features`
// (:361-380, :382-387, :461-468, :470-499), the TA-Lib block (:501-553), the
// rolling return-moment stats (:557-593) and the scale-free level transforms
// (:637-651 -> features_scalefree.py:113-129), and NOTHING else.
//
// 65 columns. The reference frame for one symbol is 82 wide; the 17 that are
// NOT here are, exhaustively:
//   * 7 raw passthroughs (open, high, low, volume, quote_volume,
//     taker_buy_quote_volume, number_of_trades) — research.py:336 seeds
//     `engineered_frames = [df]`, so the input frame rides through the concat.
//     They are INPUTS, carried in RawBars; `close` is the one the engineered
//     panel re-emits because the regime predicates compare against it.
//   * 7 LOOKAHEAD target columns (return, return_long, return_short,
//     return_long_raw, return_short_raw, return_dip, return_rip) — every one
//     reads close/high/low at shift(-1). Never computed here; declared in
//     tests/feature_parity.py EXPECTED_ABSENT_TARGETS and proven absent.
//   * 3 index metadata columns (year, month, close_timestamp,
//     research.py:667-672) — calendar/bookkeeping, not features.
//
// Column keys are the obfuscation CODES (src/codes_generated.hpp), matching
// what the vertical panel and the model artifacts use, with three exceptions
// that carry their REAL names because dc/obfuscation/map.json has no code for
// them: `mvg1`, `mvg2`, `mvg3` and the `close` passthrough. See README.
//
// MA periods are the research.py:461 default [7, 25, 99]. That default is what
// BINDS in production: the deployed 15m arm's setting.json carries no
// MA_PERIODS key at all (verified 2026-08-19,
// marvel/gauntlet/pred_agamotto.base.15m_1/setting.json). They are compile-time
// here rather than configurable because a live core that silently accepted a
// different set would score frozen weights against differently-defined columns.
//
// Throws std::invalid_argument on a panel that is not PANEL_BARS wide, or on
// ragged / missing required columns.
Table engineerFeatures(const RawBars& bars);

} // namespace agamotto
