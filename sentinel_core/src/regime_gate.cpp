#include "regime_gate.hpp"
#include "codes_generated.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace mjolnir {

namespace {

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string trim(const std::string& s)
{
    const size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> splitOn(const std::string& s, const std::string& sep)
{
    std::vector<std::string> parts;
    size_t pos = 0, prev = 0;
    while ((pos = s.find(sep, prev)) != std::string::npos) {
        parts.push_back(s.substr(prev, pos - prev));
        prev = pos + sep.size();
    }
    parts.push_back(s.substr(prev));
    return parts;
}

void replaceAll(std::string& s, const std::string& from)
{
    size_t p;
    while ((p = s.find(from)) != std::string::npos) s.erase(p, from.size());
}

std::vector<char> allTrue(size_t n) { return std::vector<char>(n, 1); }

// cmp against a scalar, NaN-safe: a NaN row compares false, matching pandas
// where any comparison with NaN yields False.
template <typename F>
std::vector<char> cmpCol(const std::vector<double>& c, F pred)
{
    std::vector<char> m(c.size(), 0);
    for (size_t i = 0; i < c.size(); ++i) m[i] = (!std::isnan(c[i]) && pred(c[i])) ? 1 : 0;
    return m;
}

} // namespace

FeaturePanel::FeaturePanel(const std::vector<std::string>& names,
                           const std::vector<std::vector<double>>& cols)
    : mNames(names), mCols(cols)
{
    mRows = cols.empty() ? 0 : cols[0].size();
}

bool FeaturePanel::has(const std::string& name) const
{
    return std::find(mNames.begin(), mNames.end(), name) != mNames.end();
}

const std::vector<double>& FeaturePanel::get(const std::string& name) const
{
    auto it = std::find(mNames.begin(), mNames.end(), name);
    if (it == mNames.end()) throw std::runtime_error("FeaturePanel: no column " + name);
    return mCols[static_cast<size_t>(it - mNames.begin())];
}

double quantileLinear(const std::vector<double>& x, double q)
{
    // pandas default interpolation="linear", NaN excluded.
    std::vector<double> v;
    v.reserve(x.size());
    for (double d : x) if (!std::isnan(d)) v.push_back(d);
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double idx = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

uint16_t regimeCodeFromString(const std::string& s)
{
    // Accepts "rNNN" (coded) only. Real names are deliberately NOT accepted
    // here: matching them would require embedding the name table, which is
    // exactly what made the built .so recoverable with `strings`. The launcher
    // encodes the stack before it reaches the core.
    if (s.size() >= 4 && (s[0] == 'r' || s[0] == 'R')) {
        char* end = nullptr;
        const long v = std::strtol(s.c_str() + 1, &end, 10);
        if (end && *end == '\0' && v > 0 && v < 65536) return static_cast<uint16_t>(v);
    }
    return 0;   // 0 == not a regime code
}

std::vector<char> applyFilterMask(const FeaturePanel& panel,
                                  const std::string& raw_name,
                                  const std::string& position)
{
    const size_t n = panel.rows();
    std::string name = trim(lower(raw_name));

    // --- composition (checked BEFORE the _long/_short strip, as in the
    //     reference, so a composed name's parts each decode on their own) ---
    if (name.find("_and_") != std::string::npos) {
        std::vector<char> mask;
        for (const auto& p : splitOn(name, "_and_")) {
            auto sub = applyFilterMask(panel, trim(p), position);
            if (mask.empty()) mask = sub;
            else for (size_t i = 0; i < n; ++i) mask[i] = mask[i] && sub[i];
        }
        return mask.empty() ? allTrue(n) : mask;
    }
    if (name.find("_or_") != std::string::npos) {
        std::vector<char> mask;
        for (const auto& p : splitOn(name, "_or_")) {
            auto sub = applyFilterMask(panel, trim(p), position);
            if (mask.empty()) mask = sub;
            else for (size_t i = 0; i < n; ++i) mask[i] = mask[i] || sub[i];
        }
        return mask.empty() ? allTrue(n) : mask;
    }

    if (n == 0) return {};

    replaceAll(name, "_long");
    replaceAll(name, "_short");
    const uint16_t code = regimeCodeFromString(name);
    if (code == 0)
        throw std::runtime_error("regime is not a code (expected rNNN): " + name);

    const bool is_long = (position == "long");
    // Missing column -> all-true, matching the reference's per-branch guard.
    auto guard = [&](const char* col) { return !panel.has(col); };

    if (code == codes::R_BASELINE) {
        throw std::runtime_error(
            "baseline regime removed — unconditional fires-every-bar no-brainer; "
            "it must never reach a stack.");
    }

    if (code == codes::R_HIGH_LIQUIDATION_PRESSURE) {
        if (guard(codes::F_LIQ_BURST_RATIO)) return allTrue(n);
        const auto& c = panel.get(codes::F_LIQ_BURST_RATIO);
        const double q = quantileLinear(c, 0.75);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (code == codes::R_LOW_LIQUIDATION_PRESSURE) {
        if (guard(codes::F_LIQ_BURST_RATIO)) return allTrue(n);
        const auto& c = panel.get(codes::F_LIQ_BURST_RATIO);
        const double q = quantileLinear(c, 0.25);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (code == codes::R_FUNDING_POSITIVE) {
        if (guard("funding_rate")) return allTrue(n);
        return cmpCol(panel.get("funding_rate"), [](double v) { return v > 0; });
    }
    if (code == codes::R_FUNDING_NEGATIVE) {
        if (guard("funding_rate")) return allTrue(n);
        return cmpCol(panel.get("funding_rate"), [](double v) { return v < 0; });
    }
    if (code == codes::R_DEEP_BOOK) {
        if (guard(codes::F_DEPTH_IMBALANCE_L5)) return allTrue(n);
        const auto& c = panel.get(codes::F_DEPTH_IMBALANCE_L5);
        const double q = quantileLinear(c, is_long ? 0.6 : 0.4);
        return is_long ? cmpCol(c, [q](double v) { return v > q; })
                       : cmpCol(c, [q](double v) { return v < q; });
    }
    if (code == codes::R_TRADE_IMBALANCE) {
        if (guard(codes::F_TRADE_IMBALANCE)) return allTrue(n);
        const auto& c = panel.get(codes::F_TRADE_IMBALANCE);
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (code == codes::R_BASIS_PREMIUM) {
        if (guard(codes::F_BASIS_PCT)) return allTrue(n);
        return cmpCol(panel.get(codes::F_BASIS_PCT), [](double v) { return v > 0; });
    }
    if (code == codes::R_BASIS_DISCOUNT) {
        if (guard(codes::F_BASIS_PCT)) return allTrue(n);
        return cmpCol(panel.get(codes::F_BASIS_PCT), [](double v) { return v < 0; });
    }
    if (code == codes::R_PRE_FUNDING_SETTLEMENT) {
        if (guard(codes::F_PRE_FUNDING)) return allTrue(n);
        return cmpCol(panel.get(codes::F_PRE_FUNDING), [](double v) { return v > 0; });
    }
    if (code == codes::R_OFI_POSITIVE) {
        if (guard(codes::F_OFI_AGG)) return allTrue(n);
        const auto& c = panel.get(codes::F_OFI_AGG);
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (code == codes::R_TIGHT_SPREAD) {
        if (guard(codes::F_RELATIVE_SPREAD)) return allTrue(n);
        const auto& c = panel.get(codes::F_RELATIVE_SPREAD);
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (code == codes::R_WIDE_SPREAD) {
        if (guard(codes::F_RELATIVE_SPREAD)) return allTrue(n);
        const auto& c = panel.get(codes::F_RELATIVE_SPREAD);
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (code == codes::R_RSI_OVERSOLD) {
        if (guard(codes::F_RSI)) return allTrue(n);
        return cmpCol(panel.get(codes::F_RSI), [](double v) { return v < 30; });
    }
    if (code == codes::R_RSI_OVERBOUGHT) {
        if (guard(codes::F_RSI)) return allTrue(n);
        return cmpCol(panel.get(codes::F_RSI), [](double v) { return v > 70; });
    }
    if (code == codes::R_MACD_BULLISH) {
        if (guard(codes::F_MACDHIST)) return allTrue(n);
        const auto& c = panel.get(codes::F_MACDHIST);
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (code == codes::R_MACD_BEARISH) {
        if (guard(codes::F_MACDHIST)) return allTrue(n);
        const auto& c = panel.get(codes::F_MACDHIST);
        return is_long ? cmpCol(c, [](double v) { return v < 0; })
                       : cmpCol(c, [](double v) { return v > 0; });
    }
    if (code == codes::R_ADX_TREND) {
        if (guard(codes::F_ADX)) return allTrue(n);
        return cmpCol(panel.get(codes::F_ADX), [](double v) { return v > 25; });
    }
    if (code == codes::R_VOL_BREAKOUT || code == codes::R_HIGH_VOLUME) {
        if (guard(codes::F_VOL_RATIO)) return allTrue(n);
        const double thr = (code == codes::R_VOL_BREAKOUT) ? 2.0 : 1.0;
        return cmpCol(panel.get(codes::F_VOL_RATIO), [thr](double v) { return v > thr; });
    }
    if (code == codes::R_LOW_VOLUME) {
        if (guard(codes::F_VOL_RATIO)) return allTrue(n);
        return cmpCol(panel.get(codes::F_VOL_RATIO), [](double v) { return v < 1.0; });
    }
    if (code == codes::R_HIGH_VOL) {
        if (guard(codes::F_PRICE_RANGE_PCT)) return allTrue(n);
        const auto& c = panel.get(codes::F_PRICE_RANGE_PCT);
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (code == codes::R_LOW_VOL) {
        if (guard(codes::F_PRICE_RANGE_PCT)) return allTrue(n);
        const auto& c = panel.get(codes::F_PRICE_RANGE_PCT);
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (code == codes::R_MOM_POSITIVE) {
        if (guard(codes::F_MOM)) return allTrue(n);
        const auto& c = panel.get(codes::F_MOM);
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }

    // Unknown name must RAISE. Returning all-true would silently promote a typo
    // into an unconditional always-fire regime.
    throw std::runtime_error("Unknown regime code: r" + std::to_string(code));
}

} // namespace mjolnir
