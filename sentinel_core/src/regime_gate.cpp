#include "regime_gate.hpp"
#include "regime_map_generated.hpp"

#include <algorithm>
#include <cmath>
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

std::string decodeRegimeTolerant(const std::string& name)
{
    const auto& M = regimeCodeToName();
    auto it = M.find(name);
    return (it == M.end()) ? name : it->second;   // not a code -> pass through
}

std::vector<char> applyFilterMask(const FeaturePanel& panel,
                                  const std::string& raw_name,
                                  const std::string& position)
{
    const size_t n = panel.rows();
    std::string name = trim(lower(raw_name));
    name = decodeRegimeTolerant(name);

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

    const bool is_long = (position == "long");
    // Missing column -> all-true, matching the reference's per-branch guard.
    auto guard = [&](const char* col) { return !panel.has(col); };

    if (name == "baseline") {
        throw std::runtime_error(
            "baseline regime removed — unconditional fires-every-bar no-brainer; "
            "it must never reach a stack.");
    }

    if (name == "high_liquidation_pressure") {
        if (guard("liq_burst_ratio")) return allTrue(n);
        const auto& c = panel.get("liq_burst_ratio");
        const double q = quantileLinear(c, 0.75);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (name == "low_liquidation_pressure") {
        if (guard("liq_burst_ratio")) return allTrue(n);
        const auto& c = panel.get("liq_burst_ratio");
        const double q = quantileLinear(c, 0.25);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (name == "funding_positive") {
        if (guard("funding_rate")) return allTrue(n);
        return cmpCol(panel.get("funding_rate"), [](double v) { return v > 0; });
    }
    if (name == "funding_negative") {
        if (guard("funding_rate")) return allTrue(n);
        return cmpCol(panel.get("funding_rate"), [](double v) { return v < 0; });
    }
    if (name == "deep_book") {
        if (guard("depth_imbalance_L5")) return allTrue(n);
        const auto& c = panel.get("depth_imbalance_L5");
        const double q = quantileLinear(c, is_long ? 0.6 : 0.4);
        return is_long ? cmpCol(c, [q](double v) { return v > q; })
                       : cmpCol(c, [q](double v) { return v < q; });
    }
    if (name == "trade_imbalance") {
        if (guard("trade_imbalance")) return allTrue(n);
        const auto& c = panel.get("trade_imbalance");
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (name == "basis_premium") {
        if (guard("basis_pct")) return allTrue(n);
        return cmpCol(panel.get("basis_pct"), [](double v) { return v > 0; });
    }
    if (name == "basis_discount") {
        if (guard("basis_pct")) return allTrue(n);
        return cmpCol(panel.get("basis_pct"), [](double v) { return v < 0; });
    }
    if (name == "pre_funding_settlement") {
        if (guard("pre_funding")) return allTrue(n);
        return cmpCol(panel.get("pre_funding"), [](double v) { return v > 0; });
    }
    if (name == "ofi_positive") {
        if (guard("ofi_agg")) return allTrue(n);
        const auto& c = panel.get("ofi_agg");
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (name == "tight_spread") {
        if (guard("relative_spread")) return allTrue(n);
        const auto& c = panel.get("relative_spread");
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (name == "wide_spread") {
        if (guard("relative_spread")) return allTrue(n);
        const auto& c = panel.get("relative_spread");
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (name == "rsi_oversold") {
        if (guard("rsi")) return allTrue(n);
        return cmpCol(panel.get("rsi"), [](double v) { return v < 30; });
    }
    if (name == "rsi_overbought") {
        if (guard("rsi")) return allTrue(n);
        return cmpCol(panel.get("rsi"), [](double v) { return v > 70; });
    }
    if (name == "macd_bullish") {
        if (guard("macdhist")) return allTrue(n);
        const auto& c = panel.get("macdhist");
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }
    if (name == "macd_bearish") {
        if (guard("macdhist")) return allTrue(n);
        const auto& c = panel.get("macdhist");
        return is_long ? cmpCol(c, [](double v) { return v < 0; })
                       : cmpCol(c, [](double v) { return v > 0; });
    }
    if (name == "adx_trend") {
        if (guard("adx")) return allTrue(n);
        return cmpCol(panel.get("adx"), [](double v) { return v > 25; });
    }
    if (name == "vol_breakout" || name == "high_volume") {
        if (guard("vol_ratio")) return allTrue(n);
        const double thr = (name == "vol_breakout") ? 2.0 : 1.0;
        return cmpCol(panel.get("vol_ratio"), [thr](double v) { return v > thr; });
    }
    if (name == "low_volume") {
        if (guard("vol_ratio")) return allTrue(n);
        return cmpCol(panel.get("vol_ratio"), [](double v) { return v < 1.0; });
    }
    if (name == "high_vol") {
        if (guard("price_range_pct")) return allTrue(n);
        const auto& c = panel.get("price_range_pct");
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v > q; });
    }
    if (name == "low_vol") {
        if (guard("price_range_pct")) return allTrue(n);
        const auto& c = panel.get("price_range_pct");
        const double q = quantileLinear(c, 0.5);
        return cmpCol(c, [q](double v) { return v < q; });
    }
    if (name == "mom_positive") {
        if (guard("mom")) return allTrue(n);
        const auto& c = panel.get("mom");
        return is_long ? cmpCol(c, [](double v) { return v > 0; })
                       : cmpCol(c, [](double v) { return v < 0; });
    }

    // Unknown name must RAISE. Returning all-true would silently promote a typo
    // into an unconditional always-fire regime.
    throw std::runtime_error("Unknown regime filter: " + name);
}

} // namespace mjolnir
