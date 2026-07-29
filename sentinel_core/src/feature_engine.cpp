#include "feature_engine.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <map>

namespace mjolnir {

namespace {
constexpr double NA = std::numeric_limits<double>::quiet_NaN();
constexpr double EPS = 1e-10;
const int MA_PERIODS[3] = {7, 25, 99};

inline bool isna(double v) { return std::isnan(v); }
} // namespace

namespace pdops {

std::vector<double> diff(const std::vector<double>& x)
{
    std::vector<double> o(x.size(), NA);
    for (size_t i = 1; i < x.size(); ++i) o[i] = x[i] - x[i - 1];
    return o;
}

std::vector<double> shift(const std::vector<double>& x, int n)
{
    std::vector<double> o(x.size(), NA);
    if (n <= 0) return x;
    for (size_t i = static_cast<size_t>(n); i < x.size(); ++i) o[i] = x[i - n];
    return o;
}

std::vector<double> pctChange(const std::vector<double>& x, int periods)
{
    std::vector<double> o(x.size(), NA);
    for (size_t i = static_cast<size_t>(periods); i < x.size(); ++i) {
        const double prev = x[i - periods];
        // pandas yields inf on a zero denominator rather than raising; the
        // final sanitisation pass turns inf into 0.0, so preserve it here.
        o[i] = (prev == 0.0) ? (x[i] == 0.0 ? NA : (x[i] > 0 ? INFINITY : -INFINITY))
                             : (x[i] / prev - 1.0);
        if (isna(x[i]) || isna(prev)) o[i] = NA;
    }
    return o;
}

std::vector<double> rollMean(const std::vector<double>& x, int w, int mp)
{
    // Summed per window rather than by a running sum: the key signals span a
    // huge dynamic range, and running-sum cancellation would break 1e-9 parity.
    std::vector<double> o(x.size(), NA);
    for (size_t i = 0; i < x.size(); ++i) {
        const size_t start = (i + 1 >= static_cast<size_t>(w)) ? i + 1 - w : 0;
        double sum = 0.0;
        int cnt = 0;
        for (size_t j = start; j <= i; ++j) {
            if (!isna(x[j])) { sum += x[j]; ++cnt; }
        }
        o[i] = (cnt >= mp && cnt > 0) ? sum / cnt : NA;
    }
    return o;
}

std::vector<double> rollStd(const std::vector<double>& x, int w, int mp)
{
    // ddof=1 (pandas default). Computed directly per window rather than by a
    // running sum-of-squares: catastrophic cancellation on large-magnitude
    // inputs (prices ~1e4 with tiny variance) would break 1e-9 parity.
    std::vector<double> o(x.size(), NA);
    for (size_t i = 0; i < x.size(); ++i) {
        const size_t start = (i + 1 >= static_cast<size_t>(w)) ? i + 1 - w : 0;
        double sum = 0.0;
        int cnt = 0;
        for (size_t j = start; j <= i; ++j) {
            if (!isna(x[j])) { sum += x[j]; ++cnt; }
        }
        if (cnt < mp || cnt < 2) { o[i] = NA; continue; }  // min_periods, then ddof=1
        const double mean = sum / cnt;
        double ss = 0.0;
        for (size_t j = start; j <= i; ++j) {
            if (!isna(x[j])) { const double d = x[j] - mean; ss += d * d; }
        }
        o[i] = std::sqrt(ss / (cnt - 1));
    }
    return o;
}

} // namespace pdops

FeatureEngine::FeatureEngine(std::vector<int> windows, int bar_sec, int target_sec)
    : mWindows(std::move(windows)), mBarSec(bar_sec), mTargetSec(target_sec) {}

namespace {

// Ordered column table: preserves insertion order (parity output is compared by
// name, but stable order keeps diffs readable) with O(1) lookup by name.
struct Table {
    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;
    std::map<std::string, size_t> idx;

    void put(const std::string& n, std::vector<double> v)
    {
        auto it = idx.find(n);
        if (it == idx.end()) {
            idx[n] = names.size();
            names.push_back(n);
            cols.push_back(std::move(v));
        } else {
            cols[it->second] = std::move(v);
        }
    }
    bool has(const std::string& n) const { return idx.count(n) != 0; }
    const std::vector<double>& get(const std::string& n) const { return cols.at(idx.at(n)); }
    std::vector<double>& mut(const std::string& n) { return cols.at(idx.at(n)); }
};

// The point-in-time set: book/snapshot/derivative state stamped at bar-OPEN but
// reflecting bar-CLOSE. Shifted +1 so row T holds the prior bar's closed state.
bool isPointInTime(const std::string& c)
{
    static const char* PREFIXES[] = {"bids_", "asks_", "depth_bid_L", "depth_ask_L",
                                     "depth_imbalance_L", "ofi_L", "ofi_agg"};
    for (const char* p : PREFIXES) {
        if (c.rfind(p, 0) == 0) return true;
    }
    static const std::vector<std::string> EXACT = {
        "bid_price", "bid_amount", "ask_price", "ask_amount",
        "spread", "mid_price", "relative_spread", "microprice", "microprice_vs_mid",
        "mark_price", "index_price", "open_interest",
        "funding_rate", "next_funding_time", "predicted_funding_rate", "basis_pct"};
    return std::find(EXACT.begin(), EXACT.end(), c) != EXACT.end();
}

} // namespace

void FeatureEngine::compute(const std::vector<Bar>& bars,
                            std::vector<std::string>& out_names,
                            std::vector<std::vector<double>>& out_cols) const
{
    const size_t n = bars.size();
    Table t;

    auto col = [&](auto getter) {
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = getter(bars[i]);
        return v;
    };

    // ---- raw passthrough bar columns ------------------------------------
    t.put("open",   col([](const Bar& b) { return b.open; }));
    t.put("high",   col([](const Bar& b) { return b.high; }));
    t.put("low",    col([](const Bar& b) { return b.low; }));
    t.put("close",  col([](const Bar& b) { return b.close; }));
    t.put("volume", col([](const Bar& b) { return b.volume; }));
    t.put("buy_vol",  col([](const Bar& b) { return b.buy_vol; }));
    t.put("sell_vol", col([](const Bar& b) { return b.sell_vol; }));
    t.put("n_trades", col([](const Bar& b) { return double(b.n_trades); }));
    t.put("vwap",     col([](const Bar& b) { return b.vwap; }));
    t.put("trade_imbalance", col([](const Bar& b) { return b.trade_imbalance; }));
    t.put("bid_price",  col([](const Bar& b) { return b.bid_price; }));
    t.put("bid_amount", col([](const Bar& b) { return b.bid_amount; }));
    t.put("ask_price",  col([](const Bar& b) { return b.ask_price; }));
    t.put("ask_amount", col([](const Bar& b) { return b.ask_amount; }));
    for (int i = 0; i < BOOK_LEVELS; ++i) {
        t.put("bids_" + std::to_string(i) + "_price", col([i](const Bar& b) { return b.bids_price[i]; }));
        t.put("bids_" + std::to_string(i) + "_qty",   col([i](const Bar& b) { return b.bids_qty[i]; }));
        t.put("asks_" + std::to_string(i) + "_price", col([i](const Bar& b) { return b.asks_price[i]; }));
        t.put("asks_" + std::to_string(i) + "_qty",   col([i](const Bar& b) { return b.asks_qty[i]; }));
    }
    t.put("depth_bid_L1", col([](const Bar& b) { return b.depth_bid_L1; }));
    t.put("depth_bid_L3", col([](const Bar& b) { return b.depth_bid_L3; }));
    t.put("depth_bid_L5", col([](const Bar& b) { return b.depth_bid_L5; }));
    t.put("depth_ask_L1", col([](const Bar& b) { return b.depth_ask_L1; }));
    t.put("depth_ask_L3", col([](const Bar& b) { return b.depth_ask_L3; }));
    t.put("depth_ask_L5", col([](const Bar& b) { return b.depth_ask_L5; }));
    t.put("mark_price",   col([](const Bar& b) { return b.mark_price; }));
    t.put("index_price",  col([](const Bar& b) { return b.index_price; }));
    t.put("funding_rate", col([](const Bar& b) { return b.funding_rate; }));
    t.put("predicted_funding_rate", col([](const Bar& b) { return b.predicted_funding_rate; }));
    t.put("open_interest", col([](const Bar& b) { return b.open_interest; }));
    t.put("liq_long_notional",  col([](const Bar& b) { return b.liq_long_notional; }));
    t.put("liq_short_notional", col([](const Bar& b) { return b.liq_short_notional; }));
    t.put("liq_long_count",  col([](const Bar& b) { return double(b.liq_long_count); }));
    t.put("liq_short_count", col([](const Bar& b) { return double(b.liq_short_count); }));
    t.put("liq_total_count", col([](const Bar& b) { return double(b.liq_total_count); }));
    t.put("cycle_progress",   col([](const Bar& b) { return b.cycle_progress; }));
    t.put("secs_to_boundary", col([](const Bar& b) { return double(b.secs_to_boundary); }));

    // ---- book features ---------------------------------------------------
    {
        const auto& bp = t.get("bid_price"); const auto& ap = t.get("ask_price");
        std::vector<double> spread(n), mid(n), rel(n), micro(n), mvm(n);
        const auto& bq0 = t.get("bid_amount"); const auto& aq0 = t.get("ask_amount");
        for (size_t i = 0; i < n; ++i) {
            spread[i] = ap[i] - bp[i];
            mid[i] = (bp[i] + ap[i]) / 2.0;
            rel[i] = spread[i] / (mid[i] + EPS);
            const double bq = std::max(0.0, bq0[i]);   // .clip(lower=0)
            const double aq = std::max(0.0, aq0[i]);
            const double tot = bq + aq + EPS;
            micro[i] = (bp[i] * aq + ap[i] * bq) / tot;
            mvm[i] = micro[i] - mid[i];
        }
        t.put("spread", spread); t.put("mid_price", mid); t.put("relative_spread", rel);
        t.put("microprice", micro); t.put("microprice_vs_mid", mvm);
        for (int L : {1, 3, 5}) {
            const auto& b = t.get("depth_bid_L" + std::to_string(L));
            const auto& a = t.get("depth_ask_L" + std::to_string(L));
            std::vector<double> di(n);
            for (size_t i = 0; i < n; ++i) di[i] = (b[i] - a[i]) / (b[i] + a[i] + EPS);
            t.put("depth_imbalance_L" + std::to_string(L), di);
        }
    }

    // ---- OFI --------------------------------------------------------------
    {
        std::vector<double> agg(n, 0.0);
        std::vector<bool> any(n, false);
        for (int L : {1, 3, 5}) {
            const auto db = pdops::diff(t.get("depth_bid_L" + std::to_string(L)));
            const auto da = pdops::diff(t.get("depth_ask_L" + std::to_string(L)));
            std::vector<double> ofi(n);
            for (size_t i = 0; i < n; ++i) ofi[i] = db[i] - da[i];
            t.put("ofi_L" + std::to_string(L), ofi);
            // DataFrame.sum(axis=1) SKIPS NaN and yields 0.0 when all are NaN.
            for (size_t i = 0; i < n; ++i) {
                if (!isna(ofi[i])) { agg[i] += ofi[i]; any[i] = true; }
            }
        }
        for (size_t i = 0; i < n; ++i) if (!any[i]) agg[i] = 0.0;
        t.put("ofi_agg", agg);
    }

    // ---- derivative -------------------------------------------------------
    {
        const auto& mk = t.get("mark_price"); const auto& ix = t.get("index_price");
        std::vector<double> basis(n);
        for (size_t i = 0; i < n; ++i) basis[i] = (mk[i] - ix[i]) / (ix[i] + EPS) * 100.0;
        t.put("basis_pct", basis);

        // pre_funding: within 2h before 00:00 / 08:00 / 16:00 UTC.
        std::vector<double> pf(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const int64_t sec = bars[i].bucket_ms / 1000;
            const int64_t sod = ((sec % 86400) + 86400) % 86400;
            const int64_t hhmm = (sod / 3600) * 3600 + ((sod % 3600) / 60) * 60;
            bool hit = false;
            for (int64_t epoch : {int64_t(0), int64_t(8 * 3600), int64_t(16 * 3600)}) {
                const int64_t ws = ((epoch - 2 * 3600) % 86400 + 86400) % 86400;
                if (ws < epoch) hit |= (hhmm >= ws && hhmm < epoch);
                else            hit |= (hhmm >= ws || hhmm < epoch);   // wraps midnight
            }
            pf[i] = hit ? 1.0 : 0.0;
        }
        t.put("pre_funding", pf);
    }

    // ---- POINT-IN-TIME SHIFT (before rolling stats and targets) ----------
    // Load-bearing: without it row T sees the end of its own target window.
    for (const auto& name : t.names) {
        if (isPointInTime(name)) t.mut(name) = pdops::shift(t.get(name), 1);
    }

    // ---- liquidation ------------------------------------------------------
    {
        const auto& ll = t.get("liq_long_notional");
        const auto& ls = t.get("liq_short_notional");
        std::vector<double> tot(n);
        for (size_t i = 0; i < n; ++i) {
            tot[i] = (isna(ll[i]) ? 0.0 : ll[i]) + (isna(ls[i]) ? 0.0 : ls[i]);
        }
        t.put("liq_total_notional", tot);
        const auto r60 = pdops::rollMean(tot, 60);
        std::vector<double> burst(n), dimb(n);
        for (size_t i = 0; i < n; ++i) {
            burst[i] = tot[i] / (r60[i] + EPS);
            dimb[i] = ((isna(ll[i]) ? 0.0 : ll[i]) - (isna(ls[i]) ? 0.0 : ls[i])) / (tot[i] + EPS);
        }
        t.put("liq_burst_ratio", burst);
        t.put("liq_directional_imbalance", dimb);
    }

    // ---- trade flow -------------------------------------------------------
    {
        const auto& close = t.get("close");
        const auto& ti = t.get("trade_imbalance");
        const auto& vol = t.get("volume");
        const auto& nt = t.get("n_trades");
        const auto& open_ = t.get("open");
        std::vector<double> dtsi(n), tint(n), kyle(n);
        const auto r60 = pdops::rollMean(nt, 60);
        for (size_t i = 0; i < n; ++i) {
            dtsi[i] = ti[i] * vol[i] * close[i];
            tint[i] = nt[i] / (r60[i] + EPS);
            kyle[i] = std::fabs(close[i] - open_[i]) / (vol[i] + EPS);
        }
        t.put("dollar_tsi", dtsi);
        t.put("trade_intensity", tint);
        t.put("kyle_lambda", kyle);
    }

    // ---- price features (non-TA-Lib) --------------------------------------
    {
        const auto& close = t.get("close");
        const auto& high = t.get("high");
        const auto& low = t.get("low");
        const auto& open_ = t.get("open");
        const auto hist_ret = pdops::pctChange(close, 1);
        for (int i = 0; i < 3; ++i) {
            t.put("mvg" + std::to_string(i + 1), pdops::rollMean(close, MA_PERIODS[i]));
        }
        t.put("ret_lag1", pdops::shift(hist_ret, 1));
        t.put("ret_lag2", pdops::shift(hist_ret, 2));
        t.put("ret_lag3", pdops::shift(hist_ret, 3));
        std::vector<double> pr(n), prp(n), ocd(n), ocp(n), hop(n), lop(n);
        for (size_t i = 0; i < n; ++i) {
            pr[i] = high[i] - low[i];
            prp[i] = (high[i] - low[i]) / (open_[i] + EPS);
            ocd[i] = close[i] - open_[i];
            ocp[i] = ocd[i] / (open_[i] + EPS);
            hop[i] = (high[i] - open_[i]) / (open_[i] + EPS);
            lop[i] = (low[i] - open_[i]) / (open_[i] + EPS);
        }
        t.put("price_range", pr); t.put("price_range_pct", prp);
        t.put("open_close_diff", ocd); t.put("open_close_pct", ocp);
        t.put("high_open_pct", hop); t.put("low_open_pct", lop);
    }

    // ---- TA-Lib indicators ------------------------------------------------
    {
        std::vector<std::pair<std::string, std::vector<double>>> ta;
        talib_block::compute(t.get("close"), t.get("high"), t.get("low"),
                             t.get("volume"), ta);
        for (auto& kv : ta) t.put(kv.first, std::move(kv.second));
    }

    // ---- rolling stats on key signals -------------------------------------
    {
        static const char* KEY[] = {"trade_imbalance", "ofi_agg", "depth_imbalance_L5",
                                    "relative_spread", "dollar_tsi", "kyle_lambda"};
        for (const char* k : KEY) {
            if (!t.has(k)) continue;
            const auto src = t.get(k);   // copy: put() may reallocate storage
            for (int w : mWindows) {
                const int mp = std::max(1, w / 4);   // reference: max(1, w // 4)
                t.put(std::string(k) + "_roll" + std::to_string(w) + "_mean",
                      pdops::rollMean(src, w, mp));
                t.put(std::string(k) + "_roll" + std::to_string(w) + "_std",
                      pdops::rollStd(src, w, mp));
            }
        }
        if (t.has("volume")) {
            const auto ma7 = pdops::rollMean(t.get("volume"), 7);
            const auto& v = t.get("volume");
            std::vector<double> vr(n);
            for (size_t i = 0; i < n; ++i) vr[i] = v[i] / (ma7[i] + EPS);
            t.put("vol_ratio", vr);
        }
    }

    // ---- temporal ---------------------------------------------------------
    {
        std::vector<double> yr(n), mo(n);
        for (size_t i = 0; i < n; ++i) {
            const time_t sec = static_cast<time_t>(bars[i].bucket_ms / 1000);
            struct tm g;
            gmtime_r(&sec, &g);
            yr[i] = g.tm_year + 1900;
            mo[i] = g.tm_mon + 1;
        }
        t.put("year", yr); t.put("month", mo);
    }

    // ---- final sanitisation ----------------------------------------------
    // Feature columns only: +/-inf -> 0.0, NaN -> 0.0. Target columns are
    // excluded upstream (they are not produced here at all in live inference).
    static const std::vector<std::string> META = {"year", "month"};
    for (size_t j = 0; j < t.cols.size(); ++j) {
        if (std::find(META.begin(), META.end(), t.names[j]) != META.end()) continue;
        for (double& v : t.cols[j]) {
            if (std::isinf(v) || std::isnan(v)) v = 0.0;
        }
    }

    out_names = std::move(t.names);
    out_cols = std::move(t.cols);
}

} // namespace mjolnir
