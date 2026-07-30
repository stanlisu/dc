// M1 EXIT GATE (variant A): score the bot's OWN dumped bars with the frozen
// deployed weights and reproduce the live y_pred.
//
// This is the assembly test. The four modules are individually at parity; this
// exercises the CHAIN on real production data, which is what catches a wiring
// error between them.
//
// Faithful reproduction of the live scoring position:
//   * the buffer at scoring time ends at the bar whose timestamp == bar_ts
//     (decisions.bar_ts joins 1:1 onto bars.timestamp_ns)
//   * the buffer is the last BUFFER_MAXLEN(=1000) bars
//   * the scored row is iloc[-2] of that window, NOT the last bar
// Any one of those off by one produces plausible-but-wrong numbers.
//
// Usage: m1_gate_driver <bars.csv> <tasks.csv> <weights_root>
//   tasks.csv: symbol,bar_ts_ns,regime_dir
#include "../src/bar_builder.hpp"
#include "../src/feature_engine.hpp"
#include "../src/regime_gate.hpp"
#include "../src/model_runner.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>

using namespace mjolnir;

namespace {

constexpr int BUFFER_MAXLEN = 1000;

std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> o; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, d)) o.push_back(it);
    return o;
}

double toD(const std::string& s)
{
    if (s.empty() || s == "nan" || s == "NaN" || s == "") return 0.0;   // dumped 'nan' -> 0.0
    try { return std::stod(s); } catch (...) { return 0.0; }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: %s <bars.csv> <tasks.csv> <weights_root> [zero_cols]\n", argv[0]); return 2; }
    // Optional 4th arg: comma-separated CODED feature names to force to 0.0.
    // Used to measure what a feature the LTP feed cannot supply actually costs.
    std::vector<std::string> zero_cols;
    if (argc >= 5) {
        std::stringstream zs(argv[4]); std::string tok;
        while (std::getline(zs, tok, ',')) if (!tok.empty()) zero_cols.push_back(tok);
        std::fprintf(stderr, "[m1] zeroing %zu feature column(s)\n", zero_cols.size());
    }

    // ---- load bars, grouped by symbol in file order -----------------------
    std::ifstream bf(argv[1]);
    if (!bf) { std::fprintf(stderr, "cannot open bars\n"); return 2; }
    std::string line;
    std::getline(bf, line);
    const auto hdr = split(line, ',');
    std::unordered_map<std::string, int> ci;
    for (size_t i = 0; i < hdr.size(); ++i) ci[hdr[i]] = static_cast<int>(i);

    std::unordered_map<std::string, std::vector<Bar>> by_sym;
    std::unordered_map<std::string, std::unordered_map<int64_t, size_t>> idx_of;

    while (std::getline(bf, line)) {
        if (line.empty()) continue;
        const auto f = split(line, ',');
        if (f.size() < hdr.size()) continue;
        const std::string sym = f[ci["symbol"]];
        Bar b;
        b.bucket_ms = std::stoll(f[ci["timestamp_ns"]]) / 1000000LL;
        b.open = toD(f[ci["open"]]); b.high = toD(f[ci["high"]]);
        b.low = toD(f[ci["low"]]);   b.close = toD(f[ci["close"]]);
        b.volume = toD(f[ci["volume"]]); b.buy_vol = toD(f[ci["buy_vol"]]);
        b.sell_vol = toD(f[ci["sell_vol"]]);
        b.n_trades = static_cast<int64_t>(toD(f[ci["n_trades"]]));
        b.vwap = toD(f[ci["vwap"]]); b.trade_imbalance = toD(f[ci["trade_imbalance"]]);
        b.bid_price = toD(f[ci["bid_price"]]); b.bid_amount = toD(f[ci["bid_amount"]]);
        b.ask_price = toD(f[ci["ask_price"]]); b.ask_amount = toD(f[ci["ask_amount"]]);
        for (int i = 0; i < BOOK_LEVELS; ++i) {
            b.bids_price[i] = toD(f[ci["bids_" + std::to_string(i) + "_price"]]);
            b.bids_qty[i]   = toD(f[ci["bids_" + std::to_string(i) + "_qty"]]);
            b.asks_price[i] = toD(f[ci["asks_" + std::to_string(i) + "_price"]]);
            b.asks_qty[i]   = toD(f[ci["asks_" + std::to_string(i) + "_qty"]]);
        }
        b.depth_bid_L1 = toD(f[ci["depth_bid_L1"]]); b.depth_bid_L3 = toD(f[ci["depth_bid_L3"]]);
        b.depth_bid_L5 = toD(f[ci["depth_bid_L5"]]); b.depth_ask_L1 = toD(f[ci["depth_ask_L1"]]);
        b.depth_ask_L3 = toD(f[ci["depth_ask_L3"]]); b.depth_ask_L5 = toD(f[ci["depth_ask_L5"]]);
        b.mark_price = toD(f[ci["mark_price"]]); b.index_price = toD(f[ci["index_price"]]);
        b.funding_rate = toD(f[ci["funding_rate"]]);
        b.predicted_funding_rate = toD(f[ci["predicted_funding_rate"]]);
        b.open_interest = toD(f[ci["open_interest"]]);
        b.liq_long_notional = toD(f[ci["liq_long_notional"]]);
        b.liq_short_notional = toD(f[ci["liq_short_notional"]]);
        b.liq_long_count = static_cast<int64_t>(toD(f[ci["liq_long_count"]]));
        b.liq_short_count = static_cast<int64_t>(toD(f[ci["liq_short_count"]]));
        b.liq_total_count = static_cast<int64_t>(toD(f[ci["liq_total_count"]]));
        b.cycle_progress = toD(f[ci["cycle_progress"]]);
        b.secs_to_boundary = static_cast<int64_t>(toD(f[ci["secs_to_boundary"]]));

        auto& v = by_sym[sym];
        idx_of[sym][std::stoll(f[ci["timestamp_ns"]])] = v.size();
        v.push_back(b);
    }
    std::fprintf(stderr, "[m1] loaded %zu symbols\n", by_sym.size());

    // ---- models (loaded once per regime dir) -------------------------------
    const std::string wroot = argv[3];
    std::map<std::string, std::unique_ptr<ModelRunner>> models;

    // ---- tasks -------------------------------------------------------------
    std::ifstream tf(argv[2]);
    if (!tf) { std::fprintf(stderr, "cannot open tasks\n"); return 2; }
    std::getline(tf, line);   // header
    std::printf("symbol,bar_ts_ns,y_pred_cpp\n");

    size_t done = 0, skipped = 0;
    FeatureEngine fe({30, 60, 300, 900}, 30, 30);

    while (std::getline(tf, line)) {
        if (line.empty()) continue;
        const auto f = split(line, ',');
        const std::string sym = f[0];
        const int64_t ts_ns = std::stoll(f[1]);
        const std::string regime_dir = f[2];

        auto sit = by_sym.find(sym);
        if (sit == by_sym.end()) { ++skipped; continue; }
        auto& bars = sit->second;
        auto iit = idx_of[sym].find(ts_ns);
        if (iit == idx_of[sym].end()) { ++skipped; continue; }
        const size_t idx = iit->second;

        // Window = last BUFFER_MAXLEN bars ENDING at this bar (inclusive).
        const size_t start = (idx + 1 > static_cast<size_t>(BUFFER_MAXLEN))
                                 ? idx + 1 - BUFFER_MAXLEN : 0;
        std::vector<Bar> win(bars.begin() + static_cast<long>(start),
                             bars.begin() + static_cast<long>(idx) + 1);
        if (win.size() < 2) { ++skipped; continue; }

        std::vector<std::string> names;
        std::vector<std::vector<double>> cols;
        fe.compute(win, names, cols);
        for (const auto& z : zero_cols)
            for (size_t j = 0; j < names.size(); ++j)
                if (names[j] == z) std::fill(cols[j].begin(), cols[j].end(), 0.0);

        FeaturePanel panel(names, cols);

        auto mit = models.find(regime_dir);
        if (mit == models.end()) {
            auto mr = std::unique_ptr<ModelRunner>(new ModelRunner());
            mr->load(wroot + "/" + regime_dir);
            mit = models.emplace(regime_dir, std::move(mr)).first;
        }
        // iloc[-2]: the row BEFORE the closing bar.
        const size_t score_row = win.size() - 2;
        const double y = mit->second->predictRow(panel, score_row);
        std::printf("%s,%lld,%.17g\n", sym.c_str(), (long long)ts_ns, y);
        if (++done % 200 == 0) std::fprintf(stderr, "[m1] %zu scored\n", done);
    }
    std::fprintf(stderr, "[m1] done=%zu skipped=%zu\n", done, skipped);
    return 0;
}
