// TA-Lib indicator block.
//
// Calls libta-lib DIRECTLY — the same C library the reference's Python wrapper
// calls — so values are identical by construction rather than reimplemented.
// The version must match production (currently 0.6.4); TA-Lib has changed
// indicator internals across releases, so a different version is a silent
// parity break, not a rounding difference.
//
// TA-Lib writes a COMPACTED output starting at outBegIdx, i.e. out[0]
// corresponds to input index outBegIdx. The unstable head must be left NaN and
// the payload placed at its true index — getting this wrong shifts every value
// of an indicator by its lookback, which still "looks plausible".
#include "feature_engine.hpp"
#include "codes_generated.hpp"

#include <ta-lib/ta_libc.h>

#include <cmath>     // std::isnan — gcc 8 does not pull this in transitively
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mjolnir {
namespace talib_block {

namespace {
constexpr double NA = std::numeric_limits<double>::quiet_NaN();

// Scatter a TA-Lib compacted result back onto the full index.
std::vector<double> place(int n, int outBeg, int outN, const std::vector<double>& raw)
{
    std::vector<double> o(static_cast<size_t>(n), NA);
    for (int i = 0; i < outN; ++i) {
        const int idx = outBeg + i;
        if (idx >= 0 && idx < n) o[static_cast<size_t>(idx)] = raw[static_cast<size_t>(i)];
    }
    return o;
}
} // namespace

// TA_Initialize is required once before any TA_* call.
void init()
{
    static bool done = false;
    if (!done) { TA_Initialize(); done = true; }
}

void compute(const std::vector<double>& close,
             const std::vector<double>& high,
             const std::vector<double>& low,
             const std::vector<double>& volume,
             std::vector<std::pair<std::string, std::vector<double>>>& out)
{
    init();
    const int n = static_cast<int>(close.size());
    if (n == 0) return;

    const double* c = close.data();
    const double* h = high.data();
    const double* l = low.data();
    // The reference passes volume.fillna(0) into TA-Lib.
    std::vector<double> vfill(volume);
    for (double& x : vfill) if (std::isnan(x)) x = 0.0;
    const double* v = vfill.data();

    int beg = 0, cnt = 0;
    std::vector<double> r(static_cast<size_t>(n));

    auto one = [&](const char* name, TA_RetCode rc) {
        if (rc != TA_SUCCESS) { out.emplace_back(name, std::vector<double>(n, NA)); return; }
        out.emplace_back(name, place(n, beg, cnt, r));
    };

    one(codes::F_RSI,    TA_RSI(0, n - 1, c, 14, &beg, &cnt, r.data()));
    one(codes::F_RSI_7,  TA_RSI(0, n - 1, c, 7,  &beg, &cnt, r.data()));
    one(codes::F_RSI_28, TA_RSI(0, n - 1, c, 28, &beg, &cnt, r.data()));

    {   // MACD: the reference keeps macd and macdhist (signal is discarded).
        std::vector<double> macd(n), sig(n), hist(n);
        TA_RetCode rc = TA_MACD(0, n - 1, c, 12, 26, 9, &beg, &cnt,
                                macd.data(), sig.data(), hist.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_MACD, std::vector<double>(n, NA));
            out.emplace_back(codes::F_MACDHIST, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_MACD, place(n, beg, cnt, macd));
            out.emplace_back(codes::F_MACDHIST, place(n, beg, cnt, hist));
        }
    }
    {   // STOCH defaults in the Python wrapper: 5, 3, SMA, 3, SMA
        std::vector<double> k(n), d(n);
        TA_RetCode rc = TA_STOCH(0, n - 1, h, l, c, 5, 3, TA_MAType_SMA, 3, TA_MAType_SMA,
                                 &beg, &cnt, k.data(), d.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_STOCH_K, std::vector<double>(n, NA));
            out.emplace_back(codes::F_STOCH_D, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_STOCH_K, place(n, beg, cnt, k));
            out.emplace_back(codes::F_STOCH_D, place(n, beg, cnt, d));
        }
    }

    one(codes::F_CCI,      TA_CCI(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_ADX,      TA_ADX(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_DX,       TA_DX(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_PLUS_DI,  TA_PLUS_DI(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_MINUS_DI, TA_MINUS_DI(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_MOM,      TA_MOM(0, n - 1, c, 10, &beg, &cnt, r.data()));
    one(codes::F_ROC,      TA_ROC(0, n - 1, c, 10, &beg, &cnt, r.data()));
    one(codes::F_WILLR,    TA_WILLR(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_CMO,      TA_CMO(0, n - 1, c, 14, &beg, &cnt, r.data()));
    one(codes::F_ATR,      TA_ATR(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));
    one(codes::F_NATR,     TA_NATR(0, n - 1, h, l, c, 14, &beg, &cnt, r.data()));

    {   // BBANDS defaults: 5, 2.0, 2.0, SMA. Middle band is discarded.
        std::vector<double> up(n), mid(n), lo(n);
        TA_RetCode rc = TA_BBANDS(0, n - 1, c, 5, 2.0, 2.0, TA_MAType_SMA,
                                  &beg, &cnt, up.data(), mid.data(), lo.data());
        if (rc != TA_SUCCESS) {
            out.emplace_back(codes::F_BB_UPPER, std::vector<double>(n, NA));
            out.emplace_back(codes::F_BB_LOWER, std::vector<double>(n, NA));
        } else {
            out.emplace_back(codes::F_BB_UPPER, place(n, beg, cnt, up));
            out.emplace_back(codes::F_BB_LOWER, place(n, beg, cnt, lo));
        }
    }

    one(codes::F_SAR, TA_SAR(0, n - 1, h, l, 0.02, 0.2, &beg, &cnt, r.data()));
    one(codes::F_OBV, TA_OBV(0, n - 1, c, v, &beg, &cnt, r.data()));
    one(codes::F_AD,  TA_AD(0, n - 1, h, l, c, v, &beg, &cnt, r.data()));
    one(codes::F_MFI, TA_MFI(0, n - 1, h, l, c, v, 14, &beg, &cnt, r.data()));
    one(codes::F_STD, TA_STDDEV(0, n - 1, c, 14, 1.0, &beg, &cnt, r.data()));
}

} // namespace talib_block
} // namespace mjolnir
