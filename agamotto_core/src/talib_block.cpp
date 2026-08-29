// STAGE 2.3 — the TA-Lib indicator block.
//
// research.py:501-553, transcribed call for call. 25 TA-Lib calls producing 29
// columns, plus `parkinson_vol`, which is NOT a TA-Lib call but lives INSIDE
// the reference's `try:` (research.py:545-548) and therefore shares the block's
// fate — it is computed here so this file diffs line-for-line against that
// block rather than being split across two files for tidiness.
//
// Calls libta-lib DIRECTLY — the same C library `talib.RSI(...)` calls through
// its Cython wrapper — so the indicator values are identical BY CONSTRUCTION
// rather than reimplemented. The version is pinned to 0.6.4 in CMakeLists.txt
// (and asserted against `talib.__ta_version__` by tests/feature_parity.py):
// TA-Lib has changed indicator internals across releases, so a different
// version is a silent parity break, not a rounding difference.
//
// ---------------------------------------------------------------------------
// THIS IS NOT sentinel_core/src/talib_block.cpp WITH THE NAMESPACE CHANGED.
//
// `place()`, `init()` and the TA_RetCode -> all-NaN-on-failure shape are reused
// verbatim, and the call lines whose parameters genuinely match are reused as
// they stand. FIVE things differ, and copying mjolnir is wrong for each:
//
//   1. BBANDS timeperiod is 20 (research.py:549). mjolnir uses 5. The
//      divergence is documented in agamotto_pkg/.../features_scalefree.py:20-23
//      and is real: at 5 the band is a 5-bar SMA +/- 2 sigma, at 20 a 20-bar
//      one, and `bb_upper`/`bb_lower` feed the `bb_rebound` regime predicate.
//      A mutant driver built with 5 is the negative control for this stage.
//   2. NO F_STD IS EMITTED HERE. mjolnir maps f085 to TA_STDDEV(close, 14);
//      agamotto's `std` is `hist_return.rolling(14).std()` (research.py:569) —
//      a COMPLETELY DIFFERENT QUANTITY under the SAME CODE (price-scale vs
//      return-scale, close vs hist_return). It belongs to stage 2.4 and is
//      deliberately absent from this file.
//   3. VOLUME IS PASSED RAW. mjolnir does `volume.fillna(0)` before TA-Lib;
//      research.py:535 does `df[f"{base}_volume"].values.astype(float)` with no
//      fill at all, so a NaN volume bar poisons OBV/AD/MFI from that bar on,
//      exactly as production computes it. Adding the fill would make three
//      columns quietly wrong on any feed with a volume hole.
//   4. TRIX(30), ULTOSC(7,14,28), STOCHRSI(14,5,3,SMA) and BOP(o,h,l,c) do not
//      exist in mjolnir's block at all — new code, transcribed from
//      research.py:528, :529, :531 and :541.
//   5. BOP needs `open`, which mjolnir's `compute()` signature does not carry.
//      Threaded through here.
//
// ---------------------------------------------------------------------------
// THE PYTHON WRAPPER SKIPS LEADING NaNs. This file must too.
//
// The reference is `talib.RSI(arr, ...)`, i.e. the WRAPPER, not the bare C
// function. Every generated wrapper does `begidx = check_begidxN(inputs...)`
// — the first index at which EVERY input is non-NaN, i.e. the MAX over the
// inputs' individual first-valid indices — calls the C function on
// `data + begidx` over `endIdx = n - begidx - 1`, and writes the result at
// `begidx + lookback`. Measured on the pinned wrapper (0.6.8 / lib 0.6.4):
//
//     RSI(x,14)                       first valid index 14
//     RSI(x with 5 leading NaN, 14)   first valid index 19   (skipped, not poisoned)
//     STOCH with leads (h=3, l=5, c=2) first valid index 13 = max(3,5,2) + 8
//     BOP  with leads (o=4,h=2,l=1,c=3) first valid index 4  = max(4,2,1,3) + 0
//
// INTERIOR NaNs are NOT skipped and DO poison (measured: one NaN at index 40
// leaves zero non-NaN RSI cells after it). Both behaviours fall out of calling
// the same C library on the same sub-array, which is what this file does.
//
// A wrapper whose inputs are ENTIRELY NaN raises `Exception("inputs are all
// NaN")`, which research.py's `except Exception` (line 554) swallows into a
// log line — dropping ALL 30 columns of this block from the panel with nothing
// but a warning. This file THROWS instead; see the banner in feature_engine.hpp
// under "declared divergences".
//
// ---------------------------------------------------------------------------
// TA-Lib writes a COMPACTED output starting at outBegIdx: out[0] corresponds to
// input index `begidx + outBegIdx`. The unstable head is left NaN and the
// payload placed at its true index. Getting this wrong shifts an entire column
// by its lookback and still looks completely plausible on a value diff of the
// overlapping region, which is why tests/feature_parity.py asserts the
// FIRST-VALID-INDEX of every column against the reference, per column, before
// it compares any value.
//
// ---------------------------------------------------------------------------
// NO CONVERGENCE CORRECTION, AND NONE IS NEEDED FOR THE GATE. Several of these
// indicators (RSI's Wilder smoothing, ADX, TRIX's triple EMA, SAR) are
// recursive and never forget their start, so their values depend on where the
// input STARTS. The port does not try to correct for that; it MATCHES it,
// because `engineerFeatures` refuses any panel that is not PANEL_BARS = 699
// rows wide and the parity harness drives the reference over the SAME 699 rows.
// The unstable period is therefore identical by construction.
//
// The figure that was circulating — "0.000e+00 on SAR/ADX/TRIX and <= 1.5e-11
// on the rest" — is true only of the LAST ROW. Measured 2026-08-19 (ta-lib
// 0.6.4, 5000 synthetic BTC-scale bars, full history vs its trailing 699):
//
//   at the LAST row  rsi/adx/dx/trix/sar/macd/cmo/atr/natr agree EXACTLY;
//                    rsi_28 does NOT — 5.2e-11 abs, 9.8e-13 rel (slowest
//                    Wilder decay in the block; three orders under the gate,
//                    but not zero)
//   over the WINDOW  max abs sar 7.8e+02, macd 4.0e+01, atr 1.5e+01, adx 9.3,
//                    cmo 4.8, rsi 2.4; exact agreement only from row 414
//                    (macd) to 567 (trix)
//
// i.e. feeding the reference more history than this file gets would compare
// numbers differing in the FIRST DIGIT over the first half of the panel. That
// is why PANEL_BARS is enforced on both sides, not treated as a buffer size.
#include "feature_engine.hpp"

#include "codes_generated.hpp"

#include <ta-lib/ta_libc.h>

#include <cmath>     // std::isnan/log/sqrt — gcc 8 does not pull this in transitively
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace agamotto {
namespace talib_block {

namespace {
constexpr double NA = std::numeric_limits<double>::quiet_NaN();

// The wrapper's `check_begidx1..4`: the first index at which every input is
// non-NaN. Returns -1 when no such index exists (the wrapper raises there).
int firstValid(const std::vector<double>& x)
{
    for (size_t i = 0; i < x.size(); ++i)
        if (!std::isnan(x[i])) return static_cast<int>(i);
    return -1;
}

// Scatter a TA-Lib compacted result back onto the full index.
//
// `begidx` is the wrapper's leading-NaN skip, `outBeg` the C function's own
// lookback within the sub-array; the true index of raw[i] is their SUM. The
// bounds check is not decoration: an outBeg the caller did not expect would
// otherwise write past the column.
std::vector<double> place(int n, int begidx, int outBeg, int outN,
                          const std::vector<double>& raw)
{
    std::vector<double> o(static_cast<size_t>(n), NA);
    for (int i = 0; i < outN; ++i) {
        const int idx = begidx + outBeg + i;
        if (idx >= 0 && idx < n) o[static_cast<size_t>(idx)] = raw[static_cast<size_t>(i)];
    }
    return o;
}

// `.fillna(0.0)` — EVERY NaN, not just the diff head.
//
// research.py:538-539 is `obv_raw.diff(14).fillna(0.0)`. Rows 0..13 are 0.0
// rather than NaN, which is the part that is easy to get right; the part that
// is easy to get WRONG is that a NaN volume bar poisons TA_OBV from that bar
// onward, so the fill turns the whole poisoned TAIL into 0.0 as well. A
// head-only fill leaves NaN there and changes the model input.
// +/-inf is NOT touched: `fillna` fills NaN only.
std::vector<double> fillNaZero(std::vector<double> x)
{
    for (double& v : x) if (std::isnan(v)) v = 0.0;
    return x;
}

} // namespace

// TA_Initialize is required once before any TA_* call.
void init()
{
    static bool done = false;
    if (!done) { TA_Initialize(); done = true; }
}

void compute(const std::vector<double>& open,
             const std::vector<double>& high,
             const std::vector<double>& low,
             const std::vector<double>& close,
             const std::vector<double>& volume,
             std::vector<std::pair<std::string, std::vector<double>>>& out)
{
    init();
    const int n = static_cast<int>(close.size());
    if (n == 0) return;

    const double* o = open.data();
    const double* h = high.data();
    const double* l = low.data();
    const double* c = close.data();
    // RAW. research.py:535 does NOT fill volume before handing it to TA-Lib —
    // see divergence 3 in the banner. Do not add a fillna here.
    const double* v = volume.data();

    // The wrapper's leading-NaN skip, per input, combined per call site with
    // max() exactly as check_begidx2/3/4 do.
    const int fo = firstValid(open), fh = firstValid(high), fl = firstValid(low);
    const int fc = firstValid(close), fv = firstValid(volume);
    if (fo < 0 || fh < 0 || fl < 0 || fc < 0 || fv < 0)
        throw std::invalid_argument(
            "talib_block: an OHLCV column is entirely NaN. The reference's "
            "wrapper raises 'inputs are all NaN' here and research.py:554 "
            "swallows it into a log line, silently dropping all 30 columns of "
            "this block from the panel. A live core must not produce a panel "
            "whose missing columns are indistinguishable from a dead feed.");

    const auto mx = [](int a, int b) { return a > b ? a : b; };
    const int bC    = fc;
    const int bHL   = mx(fh, fl);
    const int bHLC  = mx(mx(fh, fl), fc);
    const int bOHLC = mx(mx(fo, fh), mx(fl, fc));
    const int bCV   = mx(fc, fv);
    const int bHLCV = mx(mx(fh, fl), mx(fc, fv));

    int outBeg = 0, outCnt = 0;
    std::vector<double> r(static_cast<size_t>(n));

    // `rc` is evaluated before the call, so outBeg/outCnt are already written
    // by the time the body reads them. Same shape as sentinel_core's `one`.
    auto one = [&](const char* name, int begidx, TA_RetCode rc) {
        if (rc != TA_SUCCESS) { out.emplace_back(name, std::vector<double>(n, NA)); return; }
        out.emplace_back(name, place(n, begidx, outBeg, outCnt, r));
    };

    // research.py:508-510
    one(codes::F_RSI,    bC, TA_RSI(0, n - bC - 1, c + bC, 14, &outBeg, &outCnt, r.data()));
    one(codes::F_RSI_7,  bC, TA_RSI(0, n - bC - 1, c + bC, 7,  &outBeg, &outCnt, r.data()));
    one(codes::F_RSI_28, bC, TA_RSI(0, n - bC - 1, c + bC, 28, &outBeg, &outCnt, r.data()));

    {   // research.py:511-513. MACD(12, 26, 9); macd and macdhist are kept and
        // THE SIGNAL LINE IS DISCARDED — `macdsignal` is bound and never
        // appended. All three share one outBegIdx.
        std::vector<double> macd(n), sig(n), hist(n);
        const TA_RetCode rc = TA_MACD(0, n - bC - 1, c + bC, 12, 26, 9, &outBeg, &outCnt,
                                      macd.data(), sig.data(), hist.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_MACD, std::vector<double>(n, NA));
            out.emplace_back(codes::F_MACDHIST, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_MACD, place(n, bC, outBeg, outCnt, macd));
            out.emplace_back(codes::F_MACDHIST, place(n, bC, outBeg, outCnt, hist));
        }
    }

    {   // research.py:515-517. STOCH(h, l, c, 5, 3, SMA, 3, SMA) — the
        // reference writes slowk_matype=0 / slowd_matype=0 explicitly, and
        // TA_MAType_SMA == 0.
        std::vector<double> k(n), d(n);
        const TA_RetCode rc = TA_STOCH(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC,
                                       5, 3, TA_MAType_SMA, 3, TA_MAType_SMA,
                                       &outBeg, &outCnt, k.data(), d.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_STOCH_K, std::vector<double>(n, NA));
            out.emplace_back(codes::F_STOCH_D, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_STOCH_K, place(n, bHLC, outBeg, outCnt, k));
            out.emplace_back(codes::F_STOCH_D, place(n, bHLC, outBeg, outCnt, d));
        }
    }

    // research.py:519-529
    one(codes::F_CCI,      bHLC, TA_CCI(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                        &outBeg, &outCnt, r.data()));
    one(codes::F_ADX,      bHLC, TA_ADX(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                        &outBeg, &outCnt, r.data()));
    one(codes::F_DX,       bHLC, TA_DX(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                       &outBeg, &outCnt, r.data()));
    one(codes::F_PLUS_DI,  bHLC, TA_PLUS_DI(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                            &outBeg, &outCnt, r.data()));
    one(codes::F_MINUS_DI, bHLC, TA_MINUS_DI(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                             &outBeg, &outCnt, r.data()));
    one(codes::F_MOM,      bC,   TA_MOM(0, n - bC - 1, c + bC, 10, &outBeg, &outCnt, r.data()));
    one(codes::F_ROC,      bC,   TA_ROC(0, n - bC - 1, c + bC, 10, &outBeg, &outCnt, r.data()));
    one(codes::F_WILLR,    bHLC, TA_WILLR(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                          &outBeg, &outCnt, r.data()));
    one(codes::F_CMO,      bC,   TA_CMO(0, n - bC - 1, c + bC, 14, &outBeg, &outCnt, r.data()));
    // TRIX period 30 (research.py:528) — a TRIPLE 30-bar EMA, lookback 3*29+9.
    one(codes::F_TRIX,     bC,   TA_TRIX(0, n - bC - 1, c + bC, 30, &outBeg, &outCnt, r.data()));
    one(codes::F_ULTOSC,   bHLC, TA_ULTOSC(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC,
                                           7, 14, 28, &outBeg, &outCnt, r.data()));

    {   // research.py:531-533. STOCHRSI(c, 14, 5, 3, SMA) -> fastk, fastd.
        std::vector<double> fk(n), fd(n);
        const TA_RetCode rc = TA_STOCHRSI(0, n - bC - 1, c + bC, 14, 5, 3, TA_MAType_SMA,
                                          &outBeg, &outCnt, fk.data(), fd.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_STOCHRSI_K, std::vector<double>(n, NA));
            out.emplace_back(codes::F_STOCHRSI_D, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_STOCHRSI_K, place(n, bC, outBeg, outCnt, fk));
            out.emplace_back(codes::F_STOCHRSI_D, place(n, bC, outBeg, outCnt, fd));
        }
    }

    {   // research.py:536-539. obv and ad are the 14-bar DIFFERENCE of the raw
        // cumulative series, NaN-filled with 0.0 — NOT the raw indicator.
        const TA_RetCode rcO = TA_OBV(0, n - bCV - 1, c + bCV, v + bCV,
                                      &outBeg, &outCnt, r.data());
        const std::vector<double> obv_raw =
            (rcO != TA_SUCCESS) ? std::vector<double>(n, NA)
                                : place(n, bCV, outBeg, outCnt, r);
        out.emplace_back(codes::F_OBV, fillNaZero(pdops::diffN(obv_raw, 14)));

        const TA_RetCode rcA = TA_AD(0, n - bHLCV - 1, h + bHLCV, l + bHLCV, c + bHLCV,
                                     v + bHLCV, &outBeg, &outCnt, r.data());
        const std::vector<double> ad_raw =
            (rcA != TA_SUCCESS) ? std::vector<double>(n, NA)
                                : place(n, bHLCV, outBeg, outCnt, r);
        out.emplace_back(codes::F_AD, fillNaZero(pdops::diffN(ad_raw, 14)));
    }

    // research.py:540-541
    one(codes::F_MFI, bHLCV, TA_MFI(0, n - bHLCV - 1, h + bHLCV, l + bHLCV, c + bHLCV,
                                    v + bHLCV, 14, &outBeg, &outCnt, r.data()));
    one(codes::F_BOP, bOHLC, TA_BOP(0, n - bOHLC - 1, o + bOHLC, h + bOHLC, l + bOHLC,
                                    c + bOHLC, &outBeg, &outCnt, r.data()));

    // research.py:543-544
    one(codes::F_ATR,  bHLC, TA_ATR(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                    &outBeg, &outCnt, r.data()));
    one(codes::F_NATR, bHLC, TA_NATR(0, n - bHLC - 1, h + bHLC, l + bHLC, c + bHLC, 14,
                                     &outBeg, &outCnt, r.data()));

    {   // research.py:545-548 — NOT a TA-Lib call, but inside the reference's
        // try block, so it is here for the line-for-line diff.
        //
        //   np.sqrt(1.0 / (4.0 * np.log(2)) * (np.log(high / low) ** 2))
        //       .rolling(14).mean()
        //
        // Three transcription details, each of which changes the last bits:
        //
        //  * `np.log(high / low)`: DIVIDE FIRST, then log. `log(h) - log(l)` is
        //    a mathematically equal but numerically DIFFERENT double.
        //  * `** 2` on a float64 numpy array takes the fast path and computes
        //    `x * x`, not `std::pow(x, 2.0)`.
        //  * the scalar `1/(4 ln2)` is evaluated ONCE and multiplied onto the
        //    square (Python `*` is left-associative, so it is `k * (r*r)`),
        //    not folded into the sqrt or distributed.
        //
        // `.rolling(14)` takes its DEFAULT min_periods, which for an integer
        // window is the window itself -> min_periods = 14, and pandas'
        // `_prep_values` maps +/-inf to NaN before the mean. pdops::rollMean
        // reproduces both.
        const double k = 1.0 / (4.0 * std::log(2.0));
        std::vector<double> pv(static_cast<size_t>(n));
        for (size_t i = 0; i < pv.size(); ++i) {
            const double rr = std::log(high[i] / low[i]);
            pv[i] = std::sqrt(k * (rr * rr));
        }
        out.emplace_back(codes::F_PARKINSON_VOL, pdops::rollMean(pv, 14, 14));
    }

    {   // research.py:549-551. BBANDS timeperiod = 20, NOT mjolnir's 5. The
        // MIDDLE BAND IS DISCARDED — `middle` is bound and never appended.
        std::vector<double> up(n), mid(n), lo(n);
        const TA_RetCode rc = TA_BBANDS(0, n - bC - 1, c + bC, 20, 2.0, 2.0, TA_MAType_SMA,
                                        &outBeg, &outCnt, up.data(), mid.data(), lo.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_BB_UPPER, std::vector<double>(n, NA));
            out.emplace_back(codes::F_BB_LOWER, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_BB_UPPER, place(n, bC, outBeg, outCnt, up));
            out.emplace_back(codes::F_BB_LOWER, place(n, bC, outBeg, outCnt, lo));
        }
    }

    // research.py:553. SAR takes high/low only.
    one(codes::F_SAR, bHL, TA_SAR(0, n - bHL - 1, h + bHL, l + bHL, 0.02, 0.2,
                                  &outBeg, &outCnt, r.data()));
}

} // namespace talib_block
} // namespace agamotto
